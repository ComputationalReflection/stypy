
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Authors: Nils Wagner, Ed Schofield, Pauli Virtanen, John Travers
2: '''
3: Tests for numerical integration.
4: '''
5: from __future__ import division, print_function, absolute_import
6: 
7: import numpy as np
8: from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
9:                    allclose)
10: 
11: from scipy._lib._numpy_compat import _assert_warns
12: from scipy._lib.six import xrange
13: 
14: from numpy.testing import (
15:     assert_, assert_array_almost_equal,
16:     assert_allclose, assert_array_equal, assert_equal)
17: import pytest
18: from pytest import raises as assert_raises
19: from scipy.integrate import odeint, ode, complex_ode
20: 
21: #------------------------------------------------------------------------------
22: # Test ODE integrators
23: #------------------------------------------------------------------------------
24: 
25: 
26: class TestOdeint(object):
27:     # Check integrate.odeint
28:     def _do_problem(self, problem):
29:         t = arange(0.0, problem.stop_t, 0.05)
30:         z, infodict = odeint(problem.f, problem.z0, t, full_output=True)
31:         assert_(problem.verify(z, t))
32: 
33:     def test_odeint(self):
34:         for problem_cls in PROBLEMS:
35:             problem = problem_cls()
36:             if problem.cmplx:
37:                 continue
38:             self._do_problem(problem)
39: 
40: 
41: class TestODEClass(object):
42: 
43:     ode_class = None   # Set in subclass.
44: 
45:     def _do_problem(self, problem, integrator, method='adams'):
46: 
47:         # ode has callback arguments in different order than odeint
48:         f = lambda t, z: problem.f(z, t)
49:         jac = None
50:         if hasattr(problem, 'jac'):
51:             jac = lambda t, z: problem.jac(z, t)
52: 
53:         integrator_params = {}
54:         if problem.lband is not None or problem.uband is not None:
55:             integrator_params['uband'] = problem.uband
56:             integrator_params['lband'] = problem.lband
57: 
58:         ig = self.ode_class(f, jac)
59:         ig.set_integrator(integrator,
60:                           atol=problem.atol/10,
61:                           rtol=problem.rtol/10,
62:                           method=method,
63:                           **integrator_params)
64: 
65:         ig.set_initial_value(problem.z0, t=0.0)
66:         z = ig.integrate(problem.stop_t)
67: 
68:         assert_array_equal(z, ig.y)
69:         assert_(ig.successful(), (problem, method))
70:         assert_(ig.get_return_code() > 0, (problem, method))
71:         assert_(problem.verify(array([z]), problem.stop_t), (problem, method))
72: 
73: 
74: class TestOde(TestODEClass):
75: 
76:     ode_class = ode
77: 
78:     def test_vode(self):
79:         # Check the vode solver
80:         for problem_cls in PROBLEMS:
81:             problem = problem_cls()
82:             if problem.cmplx:
83:                 continue
84:             if not problem.stiff:
85:                 self._do_problem(problem, 'vode', 'adams')
86:             self._do_problem(problem, 'vode', 'bdf')
87: 
88:     def test_zvode(self):
89:         # Check the zvode solver
90:         for problem_cls in PROBLEMS:
91:             problem = problem_cls()
92:             if not problem.stiff:
93:                 self._do_problem(problem, 'zvode', 'adams')
94:             self._do_problem(problem, 'zvode', 'bdf')
95: 
96:     def test_lsoda(self):
97:         # Check the lsoda solver
98:         for problem_cls in PROBLEMS:
99:             problem = problem_cls()
100:             if problem.cmplx:
101:                 continue
102:             self._do_problem(problem, 'lsoda')
103: 
104:     def test_dopri5(self):
105:         # Check the dopri5 solver
106:         for problem_cls in PROBLEMS:
107:             problem = problem_cls()
108:             if problem.cmplx:
109:                 continue
110:             if problem.stiff:
111:                 continue
112:             if hasattr(problem, 'jac'):
113:                 continue
114:             self._do_problem(problem, 'dopri5')
115: 
116:     def test_dop853(self):
117:         # Check the dop853 solver
118:         for problem_cls in PROBLEMS:
119:             problem = problem_cls()
120:             if problem.cmplx:
121:                 continue
122:             if problem.stiff:
123:                 continue
124:             if hasattr(problem, 'jac'):
125:                 continue
126:             self._do_problem(problem, 'dop853')
127: 
128:     def test_concurrent_fail(self):
129:         for sol in ('vode', 'zvode', 'lsoda'):
130:             f = lambda t, y: 1.0
131: 
132:             r = ode(f).set_integrator(sol)
133:             r.set_initial_value(0, 0)
134: 
135:             r2 = ode(f).set_integrator(sol)
136:             r2.set_initial_value(0, 0)
137: 
138:             r.integrate(r.t + 0.1)
139:             r2.integrate(r2.t + 0.1)
140: 
141:             assert_raises(RuntimeError, r.integrate, r.t + 0.1)
142: 
143:     def test_concurrent_ok(self):
144:         f = lambda t, y: 1.0
145: 
146:         for k in xrange(3):
147:             for sol in ('vode', 'zvode', 'lsoda', 'dopri5', 'dop853'):
148:                 r = ode(f).set_integrator(sol)
149:                 r.set_initial_value(0, 0)
150: 
151:                 r2 = ode(f).set_integrator(sol)
152:                 r2.set_initial_value(0, 0)
153: 
154:                 r.integrate(r.t + 0.1)
155:                 r2.integrate(r2.t + 0.1)
156:                 r2.integrate(r2.t + 0.1)
157: 
158:                 assert_allclose(r.y, 0.1)
159:                 assert_allclose(r2.y, 0.2)
160: 
161:             for sol in ('dopri5', 'dop853'):
162:                 r = ode(f).set_integrator(sol)
163:                 r.set_initial_value(0, 0)
164: 
165:                 r2 = ode(f).set_integrator(sol)
166:                 r2.set_initial_value(0, 0)
167: 
168:                 r.integrate(r.t + 0.1)
169:                 r.integrate(r.t + 0.1)
170:                 r2.integrate(r2.t + 0.1)
171:                 r.integrate(r.t + 0.1)
172:                 r2.integrate(r2.t + 0.1)
173: 
174:                 assert_allclose(r.y, 0.3)
175:                 assert_allclose(r2.y, 0.2)
176: 
177: 
178: class TestComplexOde(TestODEClass):
179: 
180:     ode_class = complex_ode
181: 
182:     def test_vode(self):
183:         # Check the vode solver
184:         for problem_cls in PROBLEMS:
185:             problem = problem_cls()
186:             if not problem.stiff:
187:                 self._do_problem(problem, 'vode', 'adams')
188:             else:
189:                 self._do_problem(problem, 'vode', 'bdf')
190: 
191:     def test_lsoda(self):
192:         # Check the lsoda solver
193:         for problem_cls in PROBLEMS:
194:             problem = problem_cls()
195:             self._do_problem(problem, 'lsoda')
196: 
197:     def test_dopri5(self):
198:         # Check the dopri5 solver
199:         for problem_cls in PROBLEMS:
200:             problem = problem_cls()
201:             if problem.stiff:
202:                 continue
203:             if hasattr(problem, 'jac'):
204:                 continue
205:             self._do_problem(problem, 'dopri5')
206: 
207:     def test_dop853(self):
208:         # Check the dop853 solver
209:         for problem_cls in PROBLEMS:
210:             problem = problem_cls()
211:             if problem.stiff:
212:                 continue
213:             if hasattr(problem, 'jac'):
214:                 continue
215:             self._do_problem(problem, 'dop853')
216: 
217: 
218: class TestSolout(object):
219:     # Check integrate.ode correctly handles solout for dopri5 and dop853
220:     def _run_solout_test(self, integrator):
221:         # Check correct usage of solout
222:         ts = []
223:         ys = []
224:         t0 = 0.0
225:         tend = 10.0
226:         y0 = [1.0, 2.0]
227: 
228:         def solout(t, y):
229:             ts.append(t)
230:             ys.append(y.copy())
231: 
232:         def rhs(t, y):
233:             return [y[0] + y[1], -y[1]**2]
234: 
235:         ig = ode(rhs).set_integrator(integrator)
236:         ig.set_solout(solout)
237:         ig.set_initial_value(y0, t0)
238:         ret = ig.integrate(tend)
239:         assert_array_equal(ys[0], y0)
240:         assert_array_equal(ys[-1], ret)
241:         assert_equal(ts[0], t0)
242:         assert_equal(ts[-1], tend)
243: 
244:     def test_solout(self):
245:         for integrator in ('dopri5', 'dop853'):
246:             self._run_solout_test(integrator)
247: 
248:     def _run_solout_after_initial_test(self, integrator):
249:         # Check if solout works even if it is set after the initial value.
250:         ts = []
251:         ys = []
252:         t0 = 0.0
253:         tend = 10.0
254:         y0 = [1.0, 2.0]
255: 
256:         def solout(t, y):
257:             ts.append(t)
258:             ys.append(y.copy())
259: 
260:         def rhs(t, y):
261:             return [y[0] + y[1], -y[1]**2]
262: 
263:         ig = ode(rhs).set_integrator(integrator)
264:         ig.set_initial_value(y0, t0)
265:         ig.set_solout(solout)
266:         ret = ig.integrate(tend)
267:         assert_array_equal(ys[0], y0)
268:         assert_array_equal(ys[-1], ret)
269:         assert_equal(ts[0], t0)
270:         assert_equal(ts[-1], tend)
271: 
272:     def test_solout_after_initial(self):
273:         for integrator in ('dopri5', 'dop853'):
274:             self._run_solout_after_initial_test(integrator)
275: 
276:     def _run_solout_break_test(self, integrator):
277:         # Check correct usage of stopping via solout
278:         ts = []
279:         ys = []
280:         t0 = 0.0
281:         tend = 10.0
282:         y0 = [1.0, 2.0]
283: 
284:         def solout(t, y):
285:             ts.append(t)
286:             ys.append(y.copy())
287:             if t > tend/2.0:
288:                 return -1
289: 
290:         def rhs(t, y):
291:             return [y[0] + y[1], -y[1]**2]
292: 
293:         ig = ode(rhs).set_integrator(integrator)
294:         ig.set_solout(solout)
295:         ig.set_initial_value(y0, t0)
296:         ret = ig.integrate(tend)
297:         assert_array_equal(ys[0], y0)
298:         assert_array_equal(ys[-1], ret)
299:         assert_equal(ts[0], t0)
300:         assert_(ts[-1] > tend/2.0)
301:         assert_(ts[-1] < tend)
302: 
303:     def test_solout_break(self):
304:         for integrator in ('dopri5', 'dop853'):
305:             self._run_solout_break_test(integrator)
306: 
307: 
308: class TestComplexSolout(object):
309:     # Check integrate.ode correctly handles solout for dopri5 and dop853
310:     def _run_solout_test(self, integrator):
311:         # Check correct usage of solout
312:         ts = []
313:         ys = []
314:         t0 = 0.0
315:         tend = 20.0
316:         y0 = [0.0]
317: 
318:         def solout(t, y):
319:             ts.append(t)
320:             ys.append(y.copy())
321: 
322:         def rhs(t, y):
323:             return [1.0/(t - 10.0 - 1j)]
324: 
325:         ig = complex_ode(rhs).set_integrator(integrator)
326:         ig.set_solout(solout)
327:         ig.set_initial_value(y0, t0)
328:         ret = ig.integrate(tend)
329:         assert_array_equal(ys[0], y0)
330:         assert_array_equal(ys[-1], ret)
331:         assert_equal(ts[0], t0)
332:         assert_equal(ts[-1], tend)
333: 
334:     def test_solout(self):
335:         for integrator in ('dopri5', 'dop853'):
336:             self._run_solout_test(integrator)
337: 
338:     def _run_solout_break_test(self, integrator):
339:         # Check correct usage of stopping via solout
340:         ts = []
341:         ys = []
342:         t0 = 0.0
343:         tend = 20.0
344:         y0 = [0.0]
345: 
346:         def solout(t, y):
347:             ts.append(t)
348:             ys.append(y.copy())
349:             if t > tend/2.0:
350:                 return -1
351: 
352:         def rhs(t, y):
353:             return [1.0/(t - 10.0 - 1j)]
354: 
355:         ig = complex_ode(rhs).set_integrator(integrator)
356:         ig.set_solout(solout)
357:         ig.set_initial_value(y0, t0)
358:         ret = ig.integrate(tend)
359:         assert_array_equal(ys[0], y0)
360:         assert_array_equal(ys[-1], ret)
361:         assert_equal(ts[0], t0)
362:         assert_(ts[-1] > tend/2.0)
363:         assert_(ts[-1] < tend)
364: 
365:     def test_solout_break(self):
366:         for integrator in ('dopri5', 'dop853'):
367:             self._run_solout_break_test(integrator)
368: 
369: 
370: #------------------------------------------------------------------------------
371: # Test problems
372: #------------------------------------------------------------------------------
373: 
374: 
375: class ODE:
376:     '''
377:     ODE problem
378:     '''
379:     stiff = False
380:     cmplx = False
381:     stop_t = 1
382:     z0 = []
383: 
384:     lband = None
385:     uband = None
386: 
387:     atol = 1e-6
388:     rtol = 1e-5
389: 
390: 
391: class SimpleOscillator(ODE):
392:     r'''
393:     Free vibration of a simple oscillator::
394:         m \ddot{u} + k u = 0, u(0) = u_0 \dot{u}(0) \dot{u}_0
395:     Solution::
396:         u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)
397:     '''
398:     stop_t = 1 + 0.09
399:     z0 = array([1.0, 0.1], float)
400: 
401:     k = 4.0
402:     m = 1.0
403: 
404:     def f(self, z, t):
405:         tmp = zeros((2, 2), float)
406:         tmp[0, 1] = 1.0
407:         tmp[1, 0] = -self.k / self.m
408:         return dot(tmp, z)
409: 
410:     def verify(self, zs, t):
411:         omega = sqrt(self.k / self.m)
412:         u = self.z0[0]*cos(omega*t) + self.z0[1]*sin(omega*t)/omega
413:         return allclose(u, zs[:, 0], atol=self.atol, rtol=self.rtol)
414: 
415: 
416: class ComplexExp(ODE):
417:     r'''The equation :lm:`\dot u = i u`'''
418:     stop_t = 1.23*pi
419:     z0 = exp([1j, 2j, 3j, 4j, 5j])
420:     cmplx = True
421: 
422:     def f(self, z, t):
423:         return 1j*z
424: 
425:     def jac(self, z, t):
426:         return 1j*eye(5)
427: 
428:     def verify(self, zs, t):
429:         u = self.z0 * exp(1j*t)
430:         return allclose(u, zs, atol=self.atol, rtol=self.rtol)
431: 
432: 
433: class Pi(ODE):
434:     r'''Integrate 1/(t + 1j) from t=-10 to t=10'''
435:     stop_t = 20
436:     z0 = [0]
437:     cmplx = True
438: 
439:     def f(self, z, t):
440:         return array([1./(t - 10 + 1j)])
441: 
442:     def verify(self, zs, t):
443:         u = -2j * np.arctan(10)
444:         return allclose(u, zs[-1, :], atol=self.atol, rtol=self.rtol)
445: 
446: 
447: class CoupledDecay(ODE):
448:     r'''
449:     3 coupled decays suited for banded treatment
450:     (banded mode makes it necessary when N>>3)
451:     '''
452: 
453:     stiff = True
454:     stop_t = 0.5
455:     z0 = [5.0, 7.0, 13.0]
456:     lband = 1
457:     uband = 0
458: 
459:     lmbd = [0.17, 0.23, 0.29]  # fictious decay constants
460: 
461:     def f(self, z, t):
462:         lmbd = self.lmbd
463:         return np.array([-lmbd[0]*z[0],
464:                          -lmbd[1]*z[1] + lmbd[0]*z[0],
465:                          -lmbd[2]*z[2] + lmbd[1]*z[1]])
466: 
467:     def jac(self, z, t):
468:         # The full Jacobian is
469:         #
470:         #    [-lmbd[0]      0         0   ]
471:         #    [ lmbd[0]  -lmbd[1]      0   ]
472:         #    [    0      lmbd[1]  -lmbd[2]]
473:         #
474:         # The lower and upper bandwidths are lband=1 and uband=0, resp.
475:         # The representation of this array in packed format is
476:         #
477:         #    [-lmbd[0]  -lmbd[1]  -lmbd[2]]
478:         #    [ lmbd[0]   lmbd[1]      0   ]
479: 
480:         lmbd = self.lmbd
481:         j = np.zeros((self.lband + self.uband + 1, 3), order='F')
482: 
483:         def set_j(ri, ci, val):
484:             j[self.uband + ri - ci, ci] = val
485:         set_j(0, 0, -lmbd[0])
486:         set_j(1, 0, lmbd[0])
487:         set_j(1, 1, -lmbd[1])
488:         set_j(2, 1, lmbd[1])
489:         set_j(2, 2, -lmbd[2])
490:         return j
491: 
492:     def verify(self, zs, t):
493:         # Formulae derived by hand
494:         lmbd = np.array(self.lmbd)
495:         d10 = lmbd[1] - lmbd[0]
496:         d21 = lmbd[2] - lmbd[1]
497:         d20 = lmbd[2] - lmbd[0]
498:         e0 = np.exp(-lmbd[0] * t)
499:         e1 = np.exp(-lmbd[1] * t)
500:         e2 = np.exp(-lmbd[2] * t)
501:         u = np.vstack((
502:             self.z0[0] * e0,
503:             self.z0[1] * e1 + self.z0[0] * lmbd[0] / d10 * (e0 - e1),
504:             self.z0[2] * e2 + self.z0[1] * lmbd[1] / d21 * (e1 - e2) +
505:             lmbd[1] * lmbd[0] * self.z0[0] / d10 *
506:             (1 / d20 * (e0 - e2) - 1 / d21 * (e1 - e2)))).transpose()
507:         return allclose(u, zs, atol=self.atol, rtol=self.rtol)
508: 
509: 
510: PROBLEMS = [SimpleOscillator, ComplexExp, Pi, CoupledDecay]
511: 
512: #------------------------------------------------------------------------------
513: 
514: 
515: def f(t, x):
516:     dxdt = [x[1], -x[0]]
517:     return dxdt
518: 
519: 
520: def jac(t, x):
521:     j = array([[0.0, 1.0],
522:                [-1.0, 0.0]])
523:     return j
524: 
525: 
526: def f1(t, x, omega):
527:     dxdt = [omega*x[1], -omega*x[0]]
528:     return dxdt
529: 
530: 
531: def jac1(t, x, omega):
532:     j = array([[0.0, omega],
533:                [-omega, 0.0]])
534:     return j
535: 
536: 
537: def f2(t, x, omega1, omega2):
538:     dxdt = [omega1*x[1], -omega2*x[0]]
539:     return dxdt
540: 
541: 
542: def jac2(t, x, omega1, omega2):
543:     j = array([[0.0, omega1],
544:                [-omega2, 0.0]])
545:     return j
546: 
547: 
548: def fv(t, x, omega):
549:     dxdt = [omega[0]*x[1], -omega[1]*x[0]]
550:     return dxdt
551: 
552: 
553: def jacv(t, x, omega):
554:     j = array([[0.0, omega[0]],
555:                [-omega[1], 0.0]])
556:     return j
557: 
558: 
559: class ODECheckParameterUse(object):
560:     '''Call an ode-class solver with several cases of parameter use.'''
561: 
562:     # solver_name must be set before tests can be run with this class.
563: 
564:     # Set these in subclasses.
565:     solver_name = ''
566:     solver_uses_jac = False
567: 
568:     def _get_solver(self, f, jac):
569:         solver = ode(f, jac)
570:         if self.solver_uses_jac:
571:             solver.set_integrator(self.solver_name, atol=1e-9, rtol=1e-7,
572:                                   with_jacobian=self.solver_uses_jac)
573:         else:
574:             # XXX Shouldn't set_integrator *always* accept the keyword arg
575:             # 'with_jacobian', and perhaps raise an exception if it is set
576:             # to True if the solver can't actually use it?
577:             solver.set_integrator(self.solver_name, atol=1e-9, rtol=1e-7)
578:         return solver
579: 
580:     def _check_solver(self, solver):
581:         ic = [1.0, 0.0]
582:         solver.set_initial_value(ic, 0.0)
583:         solver.integrate(pi)
584:         assert_array_almost_equal(solver.y, [-1.0, 0.0])
585: 
586:     def test_no_params(self):
587:         solver = self._get_solver(f, jac)
588:         self._check_solver(solver)
589: 
590:     def test_one_scalar_param(self):
591:         solver = self._get_solver(f1, jac1)
592:         omega = 1.0
593:         solver.set_f_params(omega)
594:         if self.solver_uses_jac:
595:             solver.set_jac_params(omega)
596:         self._check_solver(solver)
597: 
598:     def test_two_scalar_params(self):
599:         solver = self._get_solver(f2, jac2)
600:         omega1 = 1.0
601:         omega2 = 1.0
602:         solver.set_f_params(omega1, omega2)
603:         if self.solver_uses_jac:
604:             solver.set_jac_params(omega1, omega2)
605:         self._check_solver(solver)
606: 
607:     def test_vector_param(self):
608:         solver = self._get_solver(fv, jacv)
609:         omega = [1.0, 1.0]
610:         solver.set_f_params(omega)
611:         if self.solver_uses_jac:
612:             solver.set_jac_params(omega)
613:         self._check_solver(solver)
614: 
615:     @pytest.mark.skip("Gives spurious warning messages, see gh-7888")
616:     def test_warns_on_failure(self):
617:         # Set nsteps small to ensure failure
618:         solver = self._get_solver(f, jac)
619:         solver.set_integrator(self.solver_name, nsteps=1)
620:         ic = [1.0, 0.0]
621:         solver.set_initial_value(ic, 0.0)
622:         _assert_warns(UserWarning, solver.integrate, pi)
623: 
624: 
625: class TestDOPRI5CheckParameterUse(ODECheckParameterUse):
626:     solver_name = 'dopri5'
627:     solver_uses_jac = False
628: 
629: 
630: class TestDOP853CheckParameterUse(ODECheckParameterUse):
631:     solver_name = 'dop853'
632:     solver_uses_jac = False
633: 
634: 
635: class TestVODECheckParameterUse(ODECheckParameterUse):
636:     solver_name = 'vode'
637:     solver_uses_jac = True
638: 
639: 
640: class TestZVODECheckParameterUse(ODECheckParameterUse):
641:     solver_name = 'zvode'
642:     solver_uses_jac = True
643: 
644: 
645: class TestLSODACheckParameterUse(ODECheckParameterUse):
646:     solver_name = 'lsoda'
647:     solver_uses_jac = True
648: 
649: 
650: def test_odeint_trivial_time():
651:     # Test that odeint succeeds when given a single time point
652:     # and full_output=True.  This is a regression test for gh-4282.
653:     y0 = 1
654:     t = [0]
655:     y, info = odeint(lambda y, t: -y, y0, t, full_output=True)
656:     assert_array_equal(y, np.array([[y0]]))
657: 
658: 
659: def test_odeint_banded_jacobian():
660:     # Test the use of the `Dfun`, `ml` and `mu` options of odeint.
661: 
662:     def func(y, t, c):
663:         return c.dot(y)
664: 
665:     def jac(y, t, c):
666:         return c
667: 
668:     def jac_transpose(y, t, c):
669:         return c.T.copy(order='C')
670: 
671:     def bjac_rows(y, t, c):
672:         jac = np.row_stack((np.r_[0, np.diag(c, 1)],
673:                             np.diag(c),
674:                             np.r_[np.diag(c, -1), 0],
675:                             np.r_[np.diag(c, -2), 0, 0]))
676:         return jac
677: 
678:     def bjac_cols(y, t, c):
679:         return bjac_rows(y, t, c).T.copy(order='C')
680: 
681:     c = array([[-205, 0.01, 0.00, 0.0],
682:                [0.1, -2.50, 0.02, 0.0],
683:                [1e-3, 0.01, -2.0, 0.01],
684:                [0.00, 0.00, 0.1, -1.0]])
685: 
686:     y0 = np.ones(4)
687:     t = np.array([0, 5, 10, 100])
688: 
689:     # Use the full Jacobian.
690:     sol1, info1 = odeint(func, y0, t, args=(c,), full_output=True,
691:                          atol=1e-13, rtol=1e-11, mxstep=10000,
692:                          Dfun=jac)
693: 
694:     # Use the transposed full Jacobian, with col_deriv=True.
695:     sol2, info2 = odeint(func, y0, t, args=(c,), full_output=True,
696:                          atol=1e-13, rtol=1e-11, mxstep=10000,
697:                          Dfun=jac_transpose, col_deriv=True)
698: 
699:     # Use the banded Jacobian.
700:     sol3, info3 = odeint(func, y0, t, args=(c,), full_output=True,
701:                          atol=1e-13, rtol=1e-11, mxstep=10000,
702:                          Dfun=bjac_rows, ml=2, mu=1)
703: 
704:     # Use the transposed banded Jacobian, with col_deriv=True.
705:     sol4, info4 = odeint(func, y0, t, args=(c,), full_output=True,
706:                          atol=1e-13, rtol=1e-11, mxstep=10000,
707:                          Dfun=bjac_cols, ml=2, mu=1, col_deriv=True)
708: 
709:     assert_allclose(sol1, sol2, err_msg="sol1 != sol2")
710:     assert_allclose(sol1, sol3, atol=1e-12, err_msg="sol1 != sol3")
711:     assert_allclose(sol3, sol4, err_msg="sol3 != sol4")
712: 
713:     # Verify that the number of jacobian evaluations was the same for the
714:     # calls of odeint with a full jacobian and with a banded jacobian. This is
715:     # a regression test--there was a bug in the handling of banded jacobians
716:     # that resulted in an incorrect jacobian matrix being passed to the LSODA
717:     # code.  That would cause errors or excessive jacobian evaluations.
718:     assert_array_equal(info1['nje'], info2['nje'])
719:     assert_array_equal(info3['nje'], info4['nje'])
720: 
721: 
722: def test_odeint_errors():
723:     def sys1d(x, t):
724:         return -100*x
725: 
726:     def bad1(x, t):
727:         return 1.0/0
728: 
729:     def bad2(x, t):
730:         return "foo"
731: 
732:     def bad_jac1(x, t):
733:         return 1.0/0
734: 
735:     def bad_jac2(x, t):
736:         return [["foo"]]
737: 
738:     def sys2d(x, t):
739:         return [-100*x[0], -0.1*x[1]]
740: 
741:     def sys2d_bad_jac(x, t):
742:         return [[1.0/0, 0], [0, -0.1]]
743: 
744:     assert_raises(ZeroDivisionError, odeint, bad1, 1.0, [0, 1])
745:     assert_raises(ValueError, odeint, bad2, 1.0, [0, 1])
746: 
747:     assert_raises(ZeroDivisionError, odeint, sys1d, 1.0, [0, 1], Dfun=bad_jac1)
748:     assert_raises(ValueError, odeint, sys1d, 1.0, [0, 1], Dfun=bad_jac2)
749: 
750:     assert_raises(ZeroDivisionError, odeint, sys2d, [1.0, 1.0], [0, 1],
751:                   Dfun=sys2d_bad_jac)
752: 
753: 
754: def test_odeint_bad_shapes():
755:     # Tests of some errors that can occur with odeint.
756: 
757:     def badrhs(x, t):
758:         return [1, -1]
759: 
760:     def sys1(x, t):
761:         return -100*x
762: 
763:     def badjac(x, t):
764:         return [[0, 0, 0]]
765: 
766:     # y0 must be at most 1-d.
767:     bad_y0 = [[0, 0], [0, 0]]
768:     assert_raises(ValueError, odeint, sys1, bad_y0, [0, 1])
769: 
770:     # t must be at most 1-d.
771:     bad_t = [[0, 1], [2, 3]]
772:     assert_raises(ValueError, odeint, sys1, [10.0], bad_t)
773: 
774:     # y0 is 10, but badrhs(x, t) returns [1, -1].
775:     assert_raises(RuntimeError, odeint, badrhs, 10, [0, 1])
776: 
777:     # shape of array returned by badjac(x, t) is not correct.
778:     assert_raises(RuntimeError, odeint, sys1, [10, 10], [0, 1], Dfun=badjac)
779: 
780: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_42135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nTests for numerical integration.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_42136 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_42136) is not StypyTypeError):

    if (import_42136 != 'pyd_module'):
        __import__(import_42136)
        sys_modules_42137 = sys.modules[import_42136]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_42137.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_42136)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy import arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp, allclose' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_42138 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_42138) is not StypyTypeError):

    if (import_42138 != 'pyd_module'):
        __import__(import_42138)
        sys_modules_42139 = sys.modules[import_42138]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', sys_modules_42139.module_type_store, module_type_store, ['arange', 'zeros', 'array', 'dot', 'sqrt', 'cos', 'sin', 'eye', 'pi', 'exp', 'allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_42139, sys_modules_42139.module_type_store, module_type_store)
    else:
        from numpy import arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp, allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', None, module_type_store, ['arange', 'zeros', 'array', 'dot', 'sqrt', 'cos', 'sin', 'eye', 'pi', 'exp', 'allclose'], [arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp, allclose])

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_42138)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib._numpy_compat import _assert_warns' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_42140 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat')

if (type(import_42140) is not StypyTypeError):

    if (import_42140 != 'pyd_module'):
        __import__(import_42140)
        sys_modules_42141 = sys.modules[import_42140]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', sys_modules_42141.module_type_store, module_type_store, ['_assert_warns'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_42141, sys_modules_42141.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import _assert_warns

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['_assert_warns'], [_assert_warns])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', import_42140)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy._lib.six import xrange' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_42142 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six')

if (type(import_42142) is not StypyTypeError):

    if (import_42142 != 'pyd_module'):
        __import__(import_42142)
        sys_modules_42143 = sys.modules[import_42142]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', sys_modules_42143.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_42143, sys_modules_42143.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', import_42142)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.testing import assert_, assert_array_almost_equal, assert_allclose, assert_array_equal, assert_equal' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_42144 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing')

if (type(import_42144) is not StypyTypeError):

    if (import_42144 != 'pyd_module'):
        __import__(import_42144)
        sys_modules_42145 = sys.modules[import_42144]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', sys_modules_42145.module_type_store, module_type_store, ['assert_', 'assert_array_almost_equal', 'assert_allclose', 'assert_array_equal', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_42145, sys_modules_42145.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_array_almost_equal, assert_allclose, assert_array_equal, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_array_almost_equal', 'assert_allclose', 'assert_array_equal', 'assert_equal'], [assert_, assert_array_almost_equal, assert_allclose, assert_array_equal, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', import_42144)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import pytest' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_42146 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'pytest')

if (type(import_42146) is not StypyTypeError):

    if (import_42146 != 'pyd_module'):
        __import__(import_42146)
        sys_modules_42147 = sys.modules[import_42146]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'pytest', sys_modules_42147.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'pytest', import_42146)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from pytest import assert_raises' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_42148 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'pytest')

if (type(import_42148) is not StypyTypeError):

    if (import_42148 != 'pyd_module'):
        __import__(import_42148)
        sys_modules_42149 = sys.modules[import_42148]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'pytest', sys_modules_42149.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_42149, sys_modules_42149.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'pytest', import_42148)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy.integrate import odeint, ode, complex_ode' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_42150 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.integrate')

if (type(import_42150) is not StypyTypeError):

    if (import_42150 != 'pyd_module'):
        __import__(import_42150)
        sys_modules_42151 = sys.modules[import_42150]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.integrate', sys_modules_42151.module_type_store, module_type_store, ['odeint', 'ode', 'complex_ode'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_42151, sys_modules_42151.module_type_store, module_type_store)
    else:
        from scipy.integrate import odeint, ode, complex_ode

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.integrate', None, module_type_store, ['odeint', 'ode', 'complex_ode'], [odeint, ode, complex_ode])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.integrate', import_42150)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

# Declaration of the 'TestOdeint' class

class TestOdeint(object, ):

    @norecursion
    def _do_problem(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_do_problem'
        module_type_store = module_type_store.open_function_context('_do_problem', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOdeint._do_problem.__dict__.__setitem__('stypy_localization', localization)
        TestOdeint._do_problem.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOdeint._do_problem.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOdeint._do_problem.__dict__.__setitem__('stypy_function_name', 'TestOdeint._do_problem')
        TestOdeint._do_problem.__dict__.__setitem__('stypy_param_names_list', ['problem'])
        TestOdeint._do_problem.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOdeint._do_problem.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOdeint._do_problem.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOdeint._do_problem.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOdeint._do_problem.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOdeint._do_problem.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOdeint._do_problem', ['problem'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_do_problem', localization, ['problem'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_do_problem(...)' code ##################

        
        # Assigning a Call to a Name (line 29):
        
        # Assigning a Call to a Name (line 29):
        
        # Call to arange(...): (line 29)
        # Processing the call arguments (line 29)
        float_42153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'float')
        # Getting the type of 'problem' (line 29)
        problem_42154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'problem', False)
        # Obtaining the member 'stop_t' of a type (line 29)
        stop_t_42155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 24), problem_42154, 'stop_t')
        float_42156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 40), 'float')
        # Processing the call keyword arguments (line 29)
        kwargs_42157 = {}
        # Getting the type of 'arange' (line 29)
        arange_42152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 29)
        arange_call_result_42158 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), arange_42152, *[float_42153, stop_t_42155, float_42156], **kwargs_42157)
        
        # Assigning a type to the variable 't' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 't', arange_call_result_42158)
        
        # Assigning a Call to a Tuple (line 30):
        
        # Assigning a Subscript to a Name (line 30):
        
        # Obtaining the type of the subscript
        int_42159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'int')
        
        # Call to odeint(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'problem' (line 30)
        problem_42161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'problem', False)
        # Obtaining the member 'f' of a type (line 30)
        f_42162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 29), problem_42161, 'f')
        # Getting the type of 'problem' (line 30)
        problem_42163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 40), 'problem', False)
        # Obtaining the member 'z0' of a type (line 30)
        z0_42164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 40), problem_42163, 'z0')
        # Getting the type of 't' (line 30)
        t_42165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 52), 't', False)
        # Processing the call keyword arguments (line 30)
        # Getting the type of 'True' (line 30)
        True_42166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 67), 'True', False)
        keyword_42167 = True_42166
        kwargs_42168 = {'full_output': keyword_42167}
        # Getting the type of 'odeint' (line 30)
        odeint_42160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'odeint', False)
        # Calling odeint(args, kwargs) (line 30)
        odeint_call_result_42169 = invoke(stypy.reporting.localization.Localization(__file__, 30, 22), odeint_42160, *[f_42162, z0_42164, t_42165], **kwargs_42168)
        
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___42170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), odeint_call_result_42169, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_42171 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), getitem___42170, int_42159)
        
        # Assigning a type to the variable 'tuple_var_assignment_42123' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'tuple_var_assignment_42123', subscript_call_result_42171)
        
        # Assigning a Subscript to a Name (line 30):
        
        # Obtaining the type of the subscript
        int_42172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'int')
        
        # Call to odeint(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'problem' (line 30)
        problem_42174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'problem', False)
        # Obtaining the member 'f' of a type (line 30)
        f_42175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 29), problem_42174, 'f')
        # Getting the type of 'problem' (line 30)
        problem_42176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 40), 'problem', False)
        # Obtaining the member 'z0' of a type (line 30)
        z0_42177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 40), problem_42176, 'z0')
        # Getting the type of 't' (line 30)
        t_42178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 52), 't', False)
        # Processing the call keyword arguments (line 30)
        # Getting the type of 'True' (line 30)
        True_42179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 67), 'True', False)
        keyword_42180 = True_42179
        kwargs_42181 = {'full_output': keyword_42180}
        # Getting the type of 'odeint' (line 30)
        odeint_42173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'odeint', False)
        # Calling odeint(args, kwargs) (line 30)
        odeint_call_result_42182 = invoke(stypy.reporting.localization.Localization(__file__, 30, 22), odeint_42173, *[f_42175, z0_42177, t_42178], **kwargs_42181)
        
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___42183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), odeint_call_result_42182, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_42184 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), getitem___42183, int_42172)
        
        # Assigning a type to the variable 'tuple_var_assignment_42124' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'tuple_var_assignment_42124', subscript_call_result_42184)
        
        # Assigning a Name to a Name (line 30):
        # Getting the type of 'tuple_var_assignment_42123' (line 30)
        tuple_var_assignment_42123_42185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'tuple_var_assignment_42123')
        # Assigning a type to the variable 'z' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'z', tuple_var_assignment_42123_42185)
        
        # Assigning a Name to a Name (line 30):
        # Getting the type of 'tuple_var_assignment_42124' (line 30)
        tuple_var_assignment_42124_42186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'tuple_var_assignment_42124')
        # Assigning a type to the variable 'infodict' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'infodict', tuple_var_assignment_42124_42186)
        
        # Call to assert_(...): (line 31)
        # Processing the call arguments (line 31)
        
        # Call to verify(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'z' (line 31)
        z_42190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'z', False)
        # Getting the type of 't' (line 31)
        t_42191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 34), 't', False)
        # Processing the call keyword arguments (line 31)
        kwargs_42192 = {}
        # Getting the type of 'problem' (line 31)
        problem_42188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'problem', False)
        # Obtaining the member 'verify' of a type (line 31)
        verify_42189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), problem_42188, 'verify')
        # Calling verify(args, kwargs) (line 31)
        verify_call_result_42193 = invoke(stypy.reporting.localization.Localization(__file__, 31, 16), verify_42189, *[z_42190, t_42191], **kwargs_42192)
        
        # Processing the call keyword arguments (line 31)
        kwargs_42194 = {}
        # Getting the type of 'assert_' (line 31)
        assert__42187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 31)
        assert__call_result_42195 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), assert__42187, *[verify_call_result_42193], **kwargs_42194)
        
        
        # ################# End of '_do_problem(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_do_problem' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_42196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_do_problem'
        return stypy_return_type_42196


    @norecursion
    def test_odeint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_odeint'
        module_type_store = module_type_store.open_function_context('test_odeint', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOdeint.test_odeint.__dict__.__setitem__('stypy_localization', localization)
        TestOdeint.test_odeint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOdeint.test_odeint.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOdeint.test_odeint.__dict__.__setitem__('stypy_function_name', 'TestOdeint.test_odeint')
        TestOdeint.test_odeint.__dict__.__setitem__('stypy_param_names_list', [])
        TestOdeint.test_odeint.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOdeint.test_odeint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOdeint.test_odeint.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOdeint.test_odeint.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOdeint.test_odeint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOdeint.test_odeint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOdeint.test_odeint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_odeint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_odeint(...)' code ##################

        
        # Getting the type of 'PROBLEMS' (line 34)
        PROBLEMS_42197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'PROBLEMS')
        # Testing the type of a for loop iterable (line 34)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 8), PROBLEMS_42197)
        # Getting the type of the for loop variable (line 34)
        for_loop_var_42198 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 8), PROBLEMS_42197)
        # Assigning a type to the variable 'problem_cls' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'problem_cls', for_loop_var_42198)
        # SSA begins for a for statement (line 34)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to problem_cls(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_42200 = {}
        # Getting the type of 'problem_cls' (line 35)
        problem_cls_42199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'problem_cls', False)
        # Calling problem_cls(args, kwargs) (line 35)
        problem_cls_call_result_42201 = invoke(stypy.reporting.localization.Localization(__file__, 35, 22), problem_cls_42199, *[], **kwargs_42200)
        
        # Assigning a type to the variable 'problem' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'problem', problem_cls_call_result_42201)
        
        # Getting the type of 'problem' (line 36)
        problem_42202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'problem')
        # Obtaining the member 'cmplx' of a type (line 36)
        cmplx_42203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 15), problem_42202, 'cmplx')
        # Testing the type of an if condition (line 36)
        if_condition_42204 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 12), cmplx_42203)
        # Assigning a type to the variable 'if_condition_42204' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'if_condition_42204', if_condition_42204)
        # SSA begins for if statement (line 36)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 36)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _do_problem(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'problem' (line 38)
        problem_42207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'problem', False)
        # Processing the call keyword arguments (line 38)
        kwargs_42208 = {}
        # Getting the type of 'self' (line 38)
        self_42205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'self', False)
        # Obtaining the member '_do_problem' of a type (line 38)
        _do_problem_42206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), self_42205, '_do_problem')
        # Calling _do_problem(args, kwargs) (line 38)
        _do_problem_call_result_42209 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), _do_problem_42206, *[problem_42207], **kwargs_42208)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_odeint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_odeint' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_42210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42210)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_odeint'
        return stypy_return_type_42210


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 26, 0, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOdeint.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestOdeint' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'TestOdeint', TestOdeint)
# Declaration of the 'TestODEClass' class

class TestODEClass(object, ):
    
    # Assigning a Name to a Name (line 43):

    @norecursion
    def _do_problem(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_42211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 54), 'str', 'adams')
        defaults = [str_42211]
        # Create a new context for function '_do_problem'
        module_type_store = module_type_store.open_function_context('_do_problem', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODEClass._do_problem.__dict__.__setitem__('stypy_localization', localization)
        TestODEClass._do_problem.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODEClass._do_problem.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODEClass._do_problem.__dict__.__setitem__('stypy_function_name', 'TestODEClass._do_problem')
        TestODEClass._do_problem.__dict__.__setitem__('stypy_param_names_list', ['problem', 'integrator', 'method'])
        TestODEClass._do_problem.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODEClass._do_problem.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODEClass._do_problem.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODEClass._do_problem.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODEClass._do_problem.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODEClass._do_problem.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODEClass._do_problem', ['problem', 'integrator', 'method'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_do_problem', localization, ['problem', 'integrator', 'method'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_do_problem(...)' code ##################

        
        # Assigning a Lambda to a Name (line 48):
        
        # Assigning a Lambda to a Name (line 48):

        @norecursion
        def _stypy_temp_lambda_17(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_17'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_17', 48, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_17.stypy_localization = localization
            _stypy_temp_lambda_17.stypy_type_of_self = None
            _stypy_temp_lambda_17.stypy_type_store = module_type_store
            _stypy_temp_lambda_17.stypy_function_name = '_stypy_temp_lambda_17'
            _stypy_temp_lambda_17.stypy_param_names_list = ['t', 'z']
            _stypy_temp_lambda_17.stypy_varargs_param_name = None
            _stypy_temp_lambda_17.stypy_kwargs_param_name = None
            _stypy_temp_lambda_17.stypy_call_defaults = defaults
            _stypy_temp_lambda_17.stypy_call_varargs = varargs
            _stypy_temp_lambda_17.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_17', ['t', 'z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_17', ['t', 'z'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to f(...): (line 48)
            # Processing the call arguments (line 48)
            # Getting the type of 'z' (line 48)
            z_42214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 35), 'z', False)
            # Getting the type of 't' (line 48)
            t_42215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 38), 't', False)
            # Processing the call keyword arguments (line 48)
            kwargs_42216 = {}
            # Getting the type of 'problem' (line 48)
            problem_42212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'problem', False)
            # Obtaining the member 'f' of a type (line 48)
            f_42213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 25), problem_42212, 'f')
            # Calling f(args, kwargs) (line 48)
            f_call_result_42217 = invoke(stypy.reporting.localization.Localization(__file__, 48, 25), f_42213, *[z_42214, t_42215], **kwargs_42216)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'stypy_return_type', f_call_result_42217)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_17' in the type store
            # Getting the type of 'stypy_return_type' (line 48)
            stypy_return_type_42218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_42218)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_17'
            return stypy_return_type_42218

        # Assigning a type to the variable '_stypy_temp_lambda_17' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), '_stypy_temp_lambda_17', _stypy_temp_lambda_17)
        # Getting the type of '_stypy_temp_lambda_17' (line 48)
        _stypy_temp_lambda_17_42219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), '_stypy_temp_lambda_17')
        # Assigning a type to the variable 'f' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'f', _stypy_temp_lambda_17_42219)
        
        # Assigning a Name to a Name (line 49):
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'None' (line 49)
        None_42220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 14), 'None')
        # Assigning a type to the variable 'jac' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'jac', None_42220)
        
        # Type idiom detected: calculating its left and rigth part (line 50)
        str_42221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 28), 'str', 'jac')
        # Getting the type of 'problem' (line 50)
        problem_42222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'problem')
        
        (may_be_42223, more_types_in_union_42224) = may_provide_member(str_42221, problem_42222)

        if may_be_42223:

            if more_types_in_union_42224:
                # Runtime conditional SSA (line 50)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'problem' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'problem', remove_not_member_provider_from_union(problem_42222, 'jac'))
            
            # Assigning a Lambda to a Name (line 51):
            
            # Assigning a Lambda to a Name (line 51):

            @norecursion
            def _stypy_temp_lambda_18(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_18'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_18', 51, 18, True)
                # Passed parameters checking function
                _stypy_temp_lambda_18.stypy_localization = localization
                _stypy_temp_lambda_18.stypy_type_of_self = None
                _stypy_temp_lambda_18.stypy_type_store = module_type_store
                _stypy_temp_lambda_18.stypy_function_name = '_stypy_temp_lambda_18'
                _stypy_temp_lambda_18.stypy_param_names_list = ['t', 'z']
                _stypy_temp_lambda_18.stypy_varargs_param_name = None
                _stypy_temp_lambda_18.stypy_kwargs_param_name = None
                _stypy_temp_lambda_18.stypy_call_defaults = defaults
                _stypy_temp_lambda_18.stypy_call_varargs = varargs
                _stypy_temp_lambda_18.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_18', ['t', 'z'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_18', ['t', 'z'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to jac(...): (line 51)
                # Processing the call arguments (line 51)
                # Getting the type of 'z' (line 51)
                z_42227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 43), 'z', False)
                # Getting the type of 't' (line 51)
                t_42228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 46), 't', False)
                # Processing the call keyword arguments (line 51)
                kwargs_42229 = {}
                # Getting the type of 'problem' (line 51)
                problem_42225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'problem', False)
                # Obtaining the member 'jac' of a type (line 51)
                jac_42226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 31), problem_42225, 'jac')
                # Calling jac(args, kwargs) (line 51)
                jac_call_result_42230 = invoke(stypy.reporting.localization.Localization(__file__, 51, 31), jac_42226, *[z_42227, t_42228], **kwargs_42229)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 51)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'stypy_return_type', jac_call_result_42230)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_18' in the type store
                # Getting the type of 'stypy_return_type' (line 51)
                stypy_return_type_42231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_42231)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_18'
                return stypy_return_type_42231

            # Assigning a type to the variable '_stypy_temp_lambda_18' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), '_stypy_temp_lambda_18', _stypy_temp_lambda_18)
            # Getting the type of '_stypy_temp_lambda_18' (line 51)
            _stypy_temp_lambda_18_42232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), '_stypy_temp_lambda_18')
            # Assigning a type to the variable 'jac' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'jac', _stypy_temp_lambda_18_42232)

            if more_types_in_union_42224:
                # SSA join for if statement (line 50)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Dict to a Name (line 53):
        
        # Assigning a Dict to a Name (line 53):
        
        # Obtaining an instance of the builtin type 'dict' (line 53)
        dict_42233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 53)
        
        # Assigning a type to the variable 'integrator_params' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'integrator_params', dict_42233)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'problem' (line 54)
        problem_42234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'problem')
        # Obtaining the member 'lband' of a type (line 54)
        lband_42235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), problem_42234, 'lband')
        # Getting the type of 'None' (line 54)
        None_42236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 32), 'None')
        # Applying the binary operator 'isnot' (line 54)
        result_is_not_42237 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 11), 'isnot', lband_42235, None_42236)
        
        
        # Getting the type of 'problem' (line 54)
        problem_42238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 40), 'problem')
        # Obtaining the member 'uband' of a type (line 54)
        uband_42239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 40), problem_42238, 'uband')
        # Getting the type of 'None' (line 54)
        None_42240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 61), 'None')
        # Applying the binary operator 'isnot' (line 54)
        result_is_not_42241 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 40), 'isnot', uband_42239, None_42240)
        
        # Applying the binary operator 'or' (line 54)
        result_or_keyword_42242 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 11), 'or', result_is_not_42237, result_is_not_42241)
        
        # Testing the type of an if condition (line 54)
        if_condition_42243 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 8), result_or_keyword_42242)
        # Assigning a type to the variable 'if_condition_42243' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'if_condition_42243', if_condition_42243)
        # SSA begins for if statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 55):
        
        # Assigning a Attribute to a Subscript (line 55):
        # Getting the type of 'problem' (line 55)
        problem_42244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 41), 'problem')
        # Obtaining the member 'uband' of a type (line 55)
        uband_42245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 41), problem_42244, 'uband')
        # Getting the type of 'integrator_params' (line 55)
        integrator_params_42246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'integrator_params')
        str_42247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 30), 'str', 'uband')
        # Storing an element on a container (line 55)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 12), integrator_params_42246, (str_42247, uband_42245))
        
        # Assigning a Attribute to a Subscript (line 56):
        
        # Assigning a Attribute to a Subscript (line 56):
        # Getting the type of 'problem' (line 56)
        problem_42248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 41), 'problem')
        # Obtaining the member 'lband' of a type (line 56)
        lband_42249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 41), problem_42248, 'lband')
        # Getting the type of 'integrator_params' (line 56)
        integrator_params_42250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'integrator_params')
        str_42251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 30), 'str', 'lband')
        # Storing an element on a container (line 56)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 12), integrator_params_42250, (str_42251, lband_42249))
        # SSA join for if statement (line 54)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Call to ode_class(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'f' (line 58)
        f_42254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'f', False)
        # Getting the type of 'jac' (line 58)
        jac_42255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 31), 'jac', False)
        # Processing the call keyword arguments (line 58)
        kwargs_42256 = {}
        # Getting the type of 'self' (line 58)
        self_42252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 13), 'self', False)
        # Obtaining the member 'ode_class' of a type (line 58)
        ode_class_42253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 13), self_42252, 'ode_class')
        # Calling ode_class(args, kwargs) (line 58)
        ode_class_call_result_42257 = invoke(stypy.reporting.localization.Localization(__file__, 58, 13), ode_class_42253, *[f_42254, jac_42255], **kwargs_42256)
        
        # Assigning a type to the variable 'ig' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'ig', ode_class_call_result_42257)
        
        # Call to set_integrator(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'integrator' (line 59)
        integrator_42260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'integrator', False)
        # Processing the call keyword arguments (line 59)
        # Getting the type of 'problem' (line 60)
        problem_42261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 'problem', False)
        # Obtaining the member 'atol' of a type (line 60)
        atol_42262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 31), problem_42261, 'atol')
        int_42263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 44), 'int')
        # Applying the binary operator 'div' (line 60)
        result_div_42264 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 31), 'div', atol_42262, int_42263)
        
        keyword_42265 = result_div_42264
        # Getting the type of 'problem' (line 61)
        problem_42266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 31), 'problem', False)
        # Obtaining the member 'rtol' of a type (line 61)
        rtol_42267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 31), problem_42266, 'rtol')
        int_42268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 44), 'int')
        # Applying the binary operator 'div' (line 61)
        result_div_42269 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 31), 'div', rtol_42267, int_42268)
        
        keyword_42270 = result_div_42269
        # Getting the type of 'method' (line 62)
        method_42271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 33), 'method', False)
        keyword_42272 = method_42271
        # Getting the type of 'integrator_params' (line 63)
        integrator_params_42273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'integrator_params', False)
        kwargs_42274 = {'integrator_params_42273': integrator_params_42273, 'rtol': keyword_42270, 'method': keyword_42272, 'atol': keyword_42265}
        # Getting the type of 'ig' (line 59)
        ig_42258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'ig', False)
        # Obtaining the member 'set_integrator' of a type (line 59)
        set_integrator_42259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), ig_42258, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 59)
        set_integrator_call_result_42275 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), set_integrator_42259, *[integrator_42260], **kwargs_42274)
        
        
        # Call to set_initial_value(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'problem' (line 65)
        problem_42278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 29), 'problem', False)
        # Obtaining the member 'z0' of a type (line 65)
        z0_42279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 29), problem_42278, 'z0')
        # Processing the call keyword arguments (line 65)
        float_42280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 43), 'float')
        keyword_42281 = float_42280
        kwargs_42282 = {'t': keyword_42281}
        # Getting the type of 'ig' (line 65)
        ig_42276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'ig', False)
        # Obtaining the member 'set_initial_value' of a type (line 65)
        set_initial_value_42277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), ig_42276, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 65)
        set_initial_value_call_result_42283 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), set_initial_value_42277, *[z0_42279], **kwargs_42282)
        
        
        # Assigning a Call to a Name (line 66):
        
        # Assigning a Call to a Name (line 66):
        
        # Call to integrate(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'problem' (line 66)
        problem_42286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'problem', False)
        # Obtaining the member 'stop_t' of a type (line 66)
        stop_t_42287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 25), problem_42286, 'stop_t')
        # Processing the call keyword arguments (line 66)
        kwargs_42288 = {}
        # Getting the type of 'ig' (line 66)
        ig_42284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'ig', False)
        # Obtaining the member 'integrate' of a type (line 66)
        integrate_42285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), ig_42284, 'integrate')
        # Calling integrate(args, kwargs) (line 66)
        integrate_call_result_42289 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), integrate_42285, *[stop_t_42287], **kwargs_42288)
        
        # Assigning a type to the variable 'z' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'z', integrate_call_result_42289)
        
        # Call to assert_array_equal(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'z' (line 68)
        z_42291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'z', False)
        # Getting the type of 'ig' (line 68)
        ig_42292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'ig', False)
        # Obtaining the member 'y' of a type (line 68)
        y_42293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 30), ig_42292, 'y')
        # Processing the call keyword arguments (line 68)
        kwargs_42294 = {}
        # Getting the type of 'assert_array_equal' (line 68)
        assert_array_equal_42290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 68)
        assert_array_equal_call_result_42295 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), assert_array_equal_42290, *[z_42291, y_42293], **kwargs_42294)
        
        
        # Call to assert_(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Call to successful(...): (line 69)
        # Processing the call keyword arguments (line 69)
        kwargs_42299 = {}
        # Getting the type of 'ig' (line 69)
        ig_42297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'ig', False)
        # Obtaining the member 'successful' of a type (line 69)
        successful_42298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), ig_42297, 'successful')
        # Calling successful(args, kwargs) (line 69)
        successful_call_result_42300 = invoke(stypy.reporting.localization.Localization(__file__, 69, 16), successful_42298, *[], **kwargs_42299)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 69)
        tuple_42301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 69)
        # Adding element type (line 69)
        # Getting the type of 'problem' (line 69)
        problem_42302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 34), 'problem', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 34), tuple_42301, problem_42302)
        # Adding element type (line 69)
        # Getting the type of 'method' (line 69)
        method_42303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), 'method', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 34), tuple_42301, method_42303)
        
        # Processing the call keyword arguments (line 69)
        kwargs_42304 = {}
        # Getting the type of 'assert_' (line 69)
        assert__42296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 69)
        assert__call_result_42305 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), assert__42296, *[successful_call_result_42300, tuple_42301], **kwargs_42304)
        
        
        # Call to assert_(...): (line 70)
        # Processing the call arguments (line 70)
        
        
        # Call to get_return_code(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_42309 = {}
        # Getting the type of 'ig' (line 70)
        ig_42307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'ig', False)
        # Obtaining the member 'get_return_code' of a type (line 70)
        get_return_code_42308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), ig_42307, 'get_return_code')
        # Calling get_return_code(args, kwargs) (line 70)
        get_return_code_call_result_42310 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), get_return_code_42308, *[], **kwargs_42309)
        
        int_42311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 39), 'int')
        # Applying the binary operator '>' (line 70)
        result_gt_42312 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 16), '>', get_return_code_call_result_42310, int_42311)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 70)
        tuple_42313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 70)
        # Adding element type (line 70)
        # Getting the type of 'problem' (line 70)
        problem_42314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 43), 'problem', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 43), tuple_42313, problem_42314)
        # Adding element type (line 70)
        # Getting the type of 'method' (line 70)
        method_42315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 52), 'method', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 43), tuple_42313, method_42315)
        
        # Processing the call keyword arguments (line 70)
        kwargs_42316 = {}
        # Getting the type of 'assert_' (line 70)
        assert__42306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 70)
        assert__call_result_42317 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), assert__42306, *[result_gt_42312, tuple_42313], **kwargs_42316)
        
        
        # Call to assert_(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Call to verify(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Call to array(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_42322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        # Getting the type of 'z' (line 71)
        z_42323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 38), 'z', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 37), list_42322, z_42323)
        
        # Processing the call keyword arguments (line 71)
        kwargs_42324 = {}
        # Getting the type of 'array' (line 71)
        array_42321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 31), 'array', False)
        # Calling array(args, kwargs) (line 71)
        array_call_result_42325 = invoke(stypy.reporting.localization.Localization(__file__, 71, 31), array_42321, *[list_42322], **kwargs_42324)
        
        # Getting the type of 'problem' (line 71)
        problem_42326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 43), 'problem', False)
        # Obtaining the member 'stop_t' of a type (line 71)
        stop_t_42327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 43), problem_42326, 'stop_t')
        # Processing the call keyword arguments (line 71)
        kwargs_42328 = {}
        # Getting the type of 'problem' (line 71)
        problem_42319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'problem', False)
        # Obtaining the member 'verify' of a type (line 71)
        verify_42320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 16), problem_42319, 'verify')
        # Calling verify(args, kwargs) (line 71)
        verify_call_result_42329 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), verify_42320, *[array_call_result_42325, stop_t_42327], **kwargs_42328)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 71)
        tuple_42330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 61), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 71)
        # Adding element type (line 71)
        # Getting the type of 'problem' (line 71)
        problem_42331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 61), 'problem', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 61), tuple_42330, problem_42331)
        # Adding element type (line 71)
        # Getting the type of 'method' (line 71)
        method_42332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 70), 'method', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 61), tuple_42330, method_42332)
        
        # Processing the call keyword arguments (line 71)
        kwargs_42333 = {}
        # Getting the type of 'assert_' (line 71)
        assert__42318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 71)
        assert__call_result_42334 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), assert__42318, *[verify_call_result_42329, tuple_42330], **kwargs_42333)
        
        
        # ################# End of '_do_problem(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_do_problem' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_42335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42335)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_do_problem'
        return stypy_return_type_42335


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 41, 0, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODEClass.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestODEClass' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'TestODEClass', TestODEClass)

# Assigning a Name to a Name (line 43):
# Getting the type of 'None' (line 43)
None_42336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'None')
# Getting the type of 'TestODEClass'
TestODEClass_42337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestODEClass')
# Setting the type of the member 'ode_class' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestODEClass_42337, 'ode_class', None_42336)
# Declaration of the 'TestOde' class
# Getting the type of 'TestODEClass' (line 74)
TestODEClass_42338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 14), 'TestODEClass')

class TestOde(TestODEClass_42338, ):
    
    # Assigning a Name to a Name (line 76):

    @norecursion
    def test_vode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vode'
        module_type_store = module_type_store.open_function_context('test_vode', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOde.test_vode.__dict__.__setitem__('stypy_localization', localization)
        TestOde.test_vode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOde.test_vode.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOde.test_vode.__dict__.__setitem__('stypy_function_name', 'TestOde.test_vode')
        TestOde.test_vode.__dict__.__setitem__('stypy_param_names_list', [])
        TestOde.test_vode.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOde.test_vode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOde.test_vode.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOde.test_vode.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOde.test_vode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOde.test_vode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOde.test_vode', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vode', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vode(...)' code ##################

        
        # Getting the type of 'PROBLEMS' (line 80)
        PROBLEMS_42339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'PROBLEMS')
        # Testing the type of a for loop iterable (line 80)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 80, 8), PROBLEMS_42339)
        # Getting the type of the for loop variable (line 80)
        for_loop_var_42340 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 80, 8), PROBLEMS_42339)
        # Assigning a type to the variable 'problem_cls' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'problem_cls', for_loop_var_42340)
        # SSA begins for a for statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 81):
        
        # Assigning a Call to a Name (line 81):
        
        # Call to problem_cls(...): (line 81)
        # Processing the call keyword arguments (line 81)
        kwargs_42342 = {}
        # Getting the type of 'problem_cls' (line 81)
        problem_cls_42341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'problem_cls', False)
        # Calling problem_cls(args, kwargs) (line 81)
        problem_cls_call_result_42343 = invoke(stypy.reporting.localization.Localization(__file__, 81, 22), problem_cls_42341, *[], **kwargs_42342)
        
        # Assigning a type to the variable 'problem' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'problem', problem_cls_call_result_42343)
        
        # Getting the type of 'problem' (line 82)
        problem_42344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'problem')
        # Obtaining the member 'cmplx' of a type (line 82)
        cmplx_42345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), problem_42344, 'cmplx')
        # Testing the type of an if condition (line 82)
        if_condition_42346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 12), cmplx_42345)
        # Assigning a type to the variable 'if_condition_42346' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'if_condition_42346', if_condition_42346)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'problem' (line 84)
        problem_42347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'problem')
        # Obtaining the member 'stiff' of a type (line 84)
        stiff_42348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 19), problem_42347, 'stiff')
        # Applying the 'not' unary operator (line 84)
        result_not__42349 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 15), 'not', stiff_42348)
        
        # Testing the type of an if condition (line 84)
        if_condition_42350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 12), result_not__42349)
        # Assigning a type to the variable 'if_condition_42350' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'if_condition_42350', if_condition_42350)
        # SSA begins for if statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _do_problem(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'problem' (line 85)
        problem_42353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 33), 'problem', False)
        str_42354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 42), 'str', 'vode')
        str_42355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 50), 'str', 'adams')
        # Processing the call keyword arguments (line 85)
        kwargs_42356 = {}
        # Getting the type of 'self' (line 85)
        self_42351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'self', False)
        # Obtaining the member '_do_problem' of a type (line 85)
        _do_problem_42352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 16), self_42351, '_do_problem')
        # Calling _do_problem(args, kwargs) (line 85)
        _do_problem_call_result_42357 = invoke(stypy.reporting.localization.Localization(__file__, 85, 16), _do_problem_42352, *[problem_42353, str_42354, str_42355], **kwargs_42356)
        
        # SSA join for if statement (line 84)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _do_problem(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'problem' (line 86)
        problem_42360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 29), 'problem', False)
        str_42361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 38), 'str', 'vode')
        str_42362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 46), 'str', 'bdf')
        # Processing the call keyword arguments (line 86)
        kwargs_42363 = {}
        # Getting the type of 'self' (line 86)
        self_42358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'self', False)
        # Obtaining the member '_do_problem' of a type (line 86)
        _do_problem_42359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), self_42358, '_do_problem')
        # Calling _do_problem(args, kwargs) (line 86)
        _do_problem_call_result_42364 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), _do_problem_42359, *[problem_42360, str_42361, str_42362], **kwargs_42363)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_vode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vode' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_42365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42365)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vode'
        return stypy_return_type_42365


    @norecursion
    def test_zvode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_zvode'
        module_type_store = module_type_store.open_function_context('test_zvode', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOde.test_zvode.__dict__.__setitem__('stypy_localization', localization)
        TestOde.test_zvode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOde.test_zvode.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOde.test_zvode.__dict__.__setitem__('stypy_function_name', 'TestOde.test_zvode')
        TestOde.test_zvode.__dict__.__setitem__('stypy_param_names_list', [])
        TestOde.test_zvode.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOde.test_zvode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOde.test_zvode.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOde.test_zvode.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOde.test_zvode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOde.test_zvode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOde.test_zvode', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_zvode', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_zvode(...)' code ##################

        
        # Getting the type of 'PROBLEMS' (line 90)
        PROBLEMS_42366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 27), 'PROBLEMS')
        # Testing the type of a for loop iterable (line 90)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 90, 8), PROBLEMS_42366)
        # Getting the type of the for loop variable (line 90)
        for_loop_var_42367 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 90, 8), PROBLEMS_42366)
        # Assigning a type to the variable 'problem_cls' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'problem_cls', for_loop_var_42367)
        # SSA begins for a for statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to problem_cls(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_42369 = {}
        # Getting the type of 'problem_cls' (line 91)
        problem_cls_42368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'problem_cls', False)
        # Calling problem_cls(args, kwargs) (line 91)
        problem_cls_call_result_42370 = invoke(stypy.reporting.localization.Localization(__file__, 91, 22), problem_cls_42368, *[], **kwargs_42369)
        
        # Assigning a type to the variable 'problem' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'problem', problem_cls_call_result_42370)
        
        
        # Getting the type of 'problem' (line 92)
        problem_42371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'problem')
        # Obtaining the member 'stiff' of a type (line 92)
        stiff_42372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 19), problem_42371, 'stiff')
        # Applying the 'not' unary operator (line 92)
        result_not__42373 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 15), 'not', stiff_42372)
        
        # Testing the type of an if condition (line 92)
        if_condition_42374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 12), result_not__42373)
        # Assigning a type to the variable 'if_condition_42374' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'if_condition_42374', if_condition_42374)
        # SSA begins for if statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _do_problem(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'problem' (line 93)
        problem_42377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 33), 'problem', False)
        str_42378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 42), 'str', 'zvode')
        str_42379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 51), 'str', 'adams')
        # Processing the call keyword arguments (line 93)
        kwargs_42380 = {}
        # Getting the type of 'self' (line 93)
        self_42375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'self', False)
        # Obtaining the member '_do_problem' of a type (line 93)
        _do_problem_42376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 16), self_42375, '_do_problem')
        # Calling _do_problem(args, kwargs) (line 93)
        _do_problem_call_result_42381 = invoke(stypy.reporting.localization.Localization(__file__, 93, 16), _do_problem_42376, *[problem_42377, str_42378, str_42379], **kwargs_42380)
        
        # SSA join for if statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _do_problem(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'problem' (line 94)
        problem_42384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 29), 'problem', False)
        str_42385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 38), 'str', 'zvode')
        str_42386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 47), 'str', 'bdf')
        # Processing the call keyword arguments (line 94)
        kwargs_42387 = {}
        # Getting the type of 'self' (line 94)
        self_42382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'self', False)
        # Obtaining the member '_do_problem' of a type (line 94)
        _do_problem_42383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), self_42382, '_do_problem')
        # Calling _do_problem(args, kwargs) (line 94)
        _do_problem_call_result_42388 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), _do_problem_42383, *[problem_42384, str_42385, str_42386], **kwargs_42387)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_zvode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_zvode' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_42389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42389)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_zvode'
        return stypy_return_type_42389


    @norecursion
    def test_lsoda(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_lsoda'
        module_type_store = module_type_store.open_function_context('test_lsoda', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOde.test_lsoda.__dict__.__setitem__('stypy_localization', localization)
        TestOde.test_lsoda.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOde.test_lsoda.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOde.test_lsoda.__dict__.__setitem__('stypy_function_name', 'TestOde.test_lsoda')
        TestOde.test_lsoda.__dict__.__setitem__('stypy_param_names_list', [])
        TestOde.test_lsoda.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOde.test_lsoda.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOde.test_lsoda.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOde.test_lsoda.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOde.test_lsoda.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOde.test_lsoda.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOde.test_lsoda', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_lsoda', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_lsoda(...)' code ##################

        
        # Getting the type of 'PROBLEMS' (line 98)
        PROBLEMS_42390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'PROBLEMS')
        # Testing the type of a for loop iterable (line 98)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 98, 8), PROBLEMS_42390)
        # Getting the type of the for loop variable (line 98)
        for_loop_var_42391 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 98, 8), PROBLEMS_42390)
        # Assigning a type to the variable 'problem_cls' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'problem_cls', for_loop_var_42391)
        # SSA begins for a for statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to problem_cls(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_42393 = {}
        # Getting the type of 'problem_cls' (line 99)
        problem_cls_42392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'problem_cls', False)
        # Calling problem_cls(args, kwargs) (line 99)
        problem_cls_call_result_42394 = invoke(stypy.reporting.localization.Localization(__file__, 99, 22), problem_cls_42392, *[], **kwargs_42393)
        
        # Assigning a type to the variable 'problem' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'problem', problem_cls_call_result_42394)
        
        # Getting the type of 'problem' (line 100)
        problem_42395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'problem')
        # Obtaining the member 'cmplx' of a type (line 100)
        cmplx_42396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), problem_42395, 'cmplx')
        # Testing the type of an if condition (line 100)
        if_condition_42397 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 12), cmplx_42396)
        # Assigning a type to the variable 'if_condition_42397' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'if_condition_42397', if_condition_42397)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _do_problem(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'problem' (line 102)
        problem_42400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 'problem', False)
        str_42401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 38), 'str', 'lsoda')
        # Processing the call keyword arguments (line 102)
        kwargs_42402 = {}
        # Getting the type of 'self' (line 102)
        self_42398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self', False)
        # Obtaining the member '_do_problem' of a type (line 102)
        _do_problem_42399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), self_42398, '_do_problem')
        # Calling _do_problem(args, kwargs) (line 102)
        _do_problem_call_result_42403 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), _do_problem_42399, *[problem_42400, str_42401], **kwargs_42402)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_lsoda(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_lsoda' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_42404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42404)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_lsoda'
        return stypy_return_type_42404


    @norecursion
    def test_dopri5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dopri5'
        module_type_store = module_type_store.open_function_context('test_dopri5', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOde.test_dopri5.__dict__.__setitem__('stypy_localization', localization)
        TestOde.test_dopri5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOde.test_dopri5.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOde.test_dopri5.__dict__.__setitem__('stypy_function_name', 'TestOde.test_dopri5')
        TestOde.test_dopri5.__dict__.__setitem__('stypy_param_names_list', [])
        TestOde.test_dopri5.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOde.test_dopri5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOde.test_dopri5.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOde.test_dopri5.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOde.test_dopri5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOde.test_dopri5.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOde.test_dopri5', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dopri5', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dopri5(...)' code ##################

        
        # Getting the type of 'PROBLEMS' (line 106)
        PROBLEMS_42405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'PROBLEMS')
        # Testing the type of a for loop iterable (line 106)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 106, 8), PROBLEMS_42405)
        # Getting the type of the for loop variable (line 106)
        for_loop_var_42406 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 106, 8), PROBLEMS_42405)
        # Assigning a type to the variable 'problem_cls' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'problem_cls', for_loop_var_42406)
        # SSA begins for a for statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to problem_cls(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_42408 = {}
        # Getting the type of 'problem_cls' (line 107)
        problem_cls_42407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'problem_cls', False)
        # Calling problem_cls(args, kwargs) (line 107)
        problem_cls_call_result_42409 = invoke(stypy.reporting.localization.Localization(__file__, 107, 22), problem_cls_42407, *[], **kwargs_42408)
        
        # Assigning a type to the variable 'problem' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'problem', problem_cls_call_result_42409)
        
        # Getting the type of 'problem' (line 108)
        problem_42410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'problem')
        # Obtaining the member 'cmplx' of a type (line 108)
        cmplx_42411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), problem_42410, 'cmplx')
        # Testing the type of an if condition (line 108)
        if_condition_42412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 12), cmplx_42411)
        # Assigning a type to the variable 'if_condition_42412' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'if_condition_42412', if_condition_42412)
        # SSA begins for if statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 108)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'problem' (line 110)
        problem_42413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'problem')
        # Obtaining the member 'stiff' of a type (line 110)
        stiff_42414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 15), problem_42413, 'stiff')
        # Testing the type of an if condition (line 110)
        if_condition_42415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 12), stiff_42414)
        # Assigning a type to the variable 'if_condition_42415' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'if_condition_42415', if_condition_42415)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 112)
        str_42416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'str', 'jac')
        # Getting the type of 'problem' (line 112)
        problem_42417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'problem')
        
        (may_be_42418, more_types_in_union_42419) = may_provide_member(str_42416, problem_42417)

        if may_be_42418:

            if more_types_in_union_42419:
                # Runtime conditional SSA (line 112)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'problem' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'problem', remove_not_member_provider_from_union(problem_42417, 'jac'))

            if more_types_in_union_42419:
                # SSA join for if statement (line 112)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _do_problem(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'problem' (line 114)
        problem_42422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 29), 'problem', False)
        str_42423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 38), 'str', 'dopri5')
        # Processing the call keyword arguments (line 114)
        kwargs_42424 = {}
        # Getting the type of 'self' (line 114)
        self_42420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'self', False)
        # Obtaining the member '_do_problem' of a type (line 114)
        _do_problem_42421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), self_42420, '_do_problem')
        # Calling _do_problem(args, kwargs) (line 114)
        _do_problem_call_result_42425 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), _do_problem_42421, *[problem_42422, str_42423], **kwargs_42424)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dopri5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dopri5' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_42426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42426)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dopri5'
        return stypy_return_type_42426


    @norecursion
    def test_dop853(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dop853'
        module_type_store = module_type_store.open_function_context('test_dop853', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOde.test_dop853.__dict__.__setitem__('stypy_localization', localization)
        TestOde.test_dop853.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOde.test_dop853.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOde.test_dop853.__dict__.__setitem__('stypy_function_name', 'TestOde.test_dop853')
        TestOde.test_dop853.__dict__.__setitem__('stypy_param_names_list', [])
        TestOde.test_dop853.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOde.test_dop853.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOde.test_dop853.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOde.test_dop853.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOde.test_dop853.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOde.test_dop853.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOde.test_dop853', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dop853', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dop853(...)' code ##################

        
        # Getting the type of 'PROBLEMS' (line 118)
        PROBLEMS_42427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'PROBLEMS')
        # Testing the type of a for loop iterable (line 118)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 8), PROBLEMS_42427)
        # Getting the type of the for loop variable (line 118)
        for_loop_var_42428 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 8), PROBLEMS_42427)
        # Assigning a type to the variable 'problem_cls' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'problem_cls', for_loop_var_42428)
        # SSA begins for a for statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 119):
        
        # Assigning a Call to a Name (line 119):
        
        # Call to problem_cls(...): (line 119)
        # Processing the call keyword arguments (line 119)
        kwargs_42430 = {}
        # Getting the type of 'problem_cls' (line 119)
        problem_cls_42429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 22), 'problem_cls', False)
        # Calling problem_cls(args, kwargs) (line 119)
        problem_cls_call_result_42431 = invoke(stypy.reporting.localization.Localization(__file__, 119, 22), problem_cls_42429, *[], **kwargs_42430)
        
        # Assigning a type to the variable 'problem' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'problem', problem_cls_call_result_42431)
        
        # Getting the type of 'problem' (line 120)
        problem_42432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'problem')
        # Obtaining the member 'cmplx' of a type (line 120)
        cmplx_42433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 15), problem_42432, 'cmplx')
        # Testing the type of an if condition (line 120)
        if_condition_42434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 12), cmplx_42433)
        # Assigning a type to the variable 'if_condition_42434' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'if_condition_42434', if_condition_42434)
        # SSA begins for if statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 120)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'problem' (line 122)
        problem_42435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'problem')
        # Obtaining the member 'stiff' of a type (line 122)
        stiff_42436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 15), problem_42435, 'stiff')
        # Testing the type of an if condition (line 122)
        if_condition_42437 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 12), stiff_42436)
        # Assigning a type to the variable 'if_condition_42437' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'if_condition_42437', if_condition_42437)
        # SSA begins for if statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 124)
        str_42438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 32), 'str', 'jac')
        # Getting the type of 'problem' (line 124)
        problem_42439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'problem')
        
        (may_be_42440, more_types_in_union_42441) = may_provide_member(str_42438, problem_42439)

        if may_be_42440:

            if more_types_in_union_42441:
                # Runtime conditional SSA (line 124)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'problem' (line 124)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'problem', remove_not_member_provider_from_union(problem_42439, 'jac'))

            if more_types_in_union_42441:
                # SSA join for if statement (line 124)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _do_problem(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'problem' (line 126)
        problem_42444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 29), 'problem', False)
        str_42445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 38), 'str', 'dop853')
        # Processing the call keyword arguments (line 126)
        kwargs_42446 = {}
        # Getting the type of 'self' (line 126)
        self_42442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'self', False)
        # Obtaining the member '_do_problem' of a type (line 126)
        _do_problem_42443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), self_42442, '_do_problem')
        # Calling _do_problem(args, kwargs) (line 126)
        _do_problem_call_result_42447 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), _do_problem_42443, *[problem_42444, str_42445], **kwargs_42446)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dop853(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dop853' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_42448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42448)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dop853'
        return stypy_return_type_42448


    @norecursion
    def test_concurrent_fail(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_concurrent_fail'
        module_type_store = module_type_store.open_function_context('test_concurrent_fail', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOde.test_concurrent_fail.__dict__.__setitem__('stypy_localization', localization)
        TestOde.test_concurrent_fail.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOde.test_concurrent_fail.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOde.test_concurrent_fail.__dict__.__setitem__('stypy_function_name', 'TestOde.test_concurrent_fail')
        TestOde.test_concurrent_fail.__dict__.__setitem__('stypy_param_names_list', [])
        TestOde.test_concurrent_fail.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOde.test_concurrent_fail.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOde.test_concurrent_fail.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOde.test_concurrent_fail.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOde.test_concurrent_fail.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOde.test_concurrent_fail.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOde.test_concurrent_fail', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_concurrent_fail', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_concurrent_fail(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 129)
        tuple_42449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 129)
        # Adding element type (line 129)
        str_42450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 20), 'str', 'vode')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 20), tuple_42449, str_42450)
        # Adding element type (line 129)
        str_42451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 28), 'str', 'zvode')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 20), tuple_42449, str_42451)
        # Adding element type (line 129)
        str_42452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 37), 'str', 'lsoda')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 20), tuple_42449, str_42452)
        
        # Testing the type of a for loop iterable (line 129)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 129, 8), tuple_42449)
        # Getting the type of the for loop variable (line 129)
        for_loop_var_42453 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 129, 8), tuple_42449)
        # Assigning a type to the variable 'sol' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'sol', for_loop_var_42453)
        # SSA begins for a for statement (line 129)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Lambda to a Name (line 130):
        
        # Assigning a Lambda to a Name (line 130):

        @norecursion
        def _stypy_temp_lambda_19(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_19'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_19', 130, 16, True)
            # Passed parameters checking function
            _stypy_temp_lambda_19.stypy_localization = localization
            _stypy_temp_lambda_19.stypy_type_of_self = None
            _stypy_temp_lambda_19.stypy_type_store = module_type_store
            _stypy_temp_lambda_19.stypy_function_name = '_stypy_temp_lambda_19'
            _stypy_temp_lambda_19.stypy_param_names_list = ['t', 'y']
            _stypy_temp_lambda_19.stypy_varargs_param_name = None
            _stypy_temp_lambda_19.stypy_kwargs_param_name = None
            _stypy_temp_lambda_19.stypy_call_defaults = defaults
            _stypy_temp_lambda_19.stypy_call_varargs = varargs
            _stypy_temp_lambda_19.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_19', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_19', ['t', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            float_42454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 29), 'float')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'stypy_return_type', float_42454)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_19' in the type store
            # Getting the type of 'stypy_return_type' (line 130)
            stypy_return_type_42455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_42455)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_19'
            return stypy_return_type_42455

        # Assigning a type to the variable '_stypy_temp_lambda_19' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), '_stypy_temp_lambda_19', _stypy_temp_lambda_19)
        # Getting the type of '_stypy_temp_lambda_19' (line 130)
        _stypy_temp_lambda_19_42456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), '_stypy_temp_lambda_19')
        # Assigning a type to the variable 'f' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'f', _stypy_temp_lambda_19_42456)
        
        # Assigning a Call to a Name (line 132):
        
        # Assigning a Call to a Name (line 132):
        
        # Call to set_integrator(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'sol' (line 132)
        sol_42462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 38), 'sol', False)
        # Processing the call keyword arguments (line 132)
        kwargs_42463 = {}
        
        # Call to ode(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'f' (line 132)
        f_42458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 20), 'f', False)
        # Processing the call keyword arguments (line 132)
        kwargs_42459 = {}
        # Getting the type of 'ode' (line 132)
        ode_42457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'ode', False)
        # Calling ode(args, kwargs) (line 132)
        ode_call_result_42460 = invoke(stypy.reporting.localization.Localization(__file__, 132, 16), ode_42457, *[f_42458], **kwargs_42459)
        
        # Obtaining the member 'set_integrator' of a type (line 132)
        set_integrator_42461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), ode_call_result_42460, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 132)
        set_integrator_call_result_42464 = invoke(stypy.reporting.localization.Localization(__file__, 132, 16), set_integrator_42461, *[sol_42462], **kwargs_42463)
        
        # Assigning a type to the variable 'r' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'r', set_integrator_call_result_42464)
        
        # Call to set_initial_value(...): (line 133)
        # Processing the call arguments (line 133)
        int_42467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 32), 'int')
        int_42468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 35), 'int')
        # Processing the call keyword arguments (line 133)
        kwargs_42469 = {}
        # Getting the type of 'r' (line 133)
        r_42465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'r', False)
        # Obtaining the member 'set_initial_value' of a type (line 133)
        set_initial_value_42466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), r_42465, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 133)
        set_initial_value_call_result_42470 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), set_initial_value_42466, *[int_42467, int_42468], **kwargs_42469)
        
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to set_integrator(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'sol' (line 135)
        sol_42476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 39), 'sol', False)
        # Processing the call keyword arguments (line 135)
        kwargs_42477 = {}
        
        # Call to ode(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'f' (line 135)
        f_42472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 21), 'f', False)
        # Processing the call keyword arguments (line 135)
        kwargs_42473 = {}
        # Getting the type of 'ode' (line 135)
        ode_42471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'ode', False)
        # Calling ode(args, kwargs) (line 135)
        ode_call_result_42474 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), ode_42471, *[f_42472], **kwargs_42473)
        
        # Obtaining the member 'set_integrator' of a type (line 135)
        set_integrator_42475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 17), ode_call_result_42474, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 135)
        set_integrator_call_result_42478 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), set_integrator_42475, *[sol_42476], **kwargs_42477)
        
        # Assigning a type to the variable 'r2' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'r2', set_integrator_call_result_42478)
        
        # Call to set_initial_value(...): (line 136)
        # Processing the call arguments (line 136)
        int_42481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 33), 'int')
        int_42482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 36), 'int')
        # Processing the call keyword arguments (line 136)
        kwargs_42483 = {}
        # Getting the type of 'r2' (line 136)
        r2_42479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'r2', False)
        # Obtaining the member 'set_initial_value' of a type (line 136)
        set_initial_value_42480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 12), r2_42479, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 136)
        set_initial_value_call_result_42484 = invoke(stypy.reporting.localization.Localization(__file__, 136, 12), set_initial_value_42480, *[int_42481, int_42482], **kwargs_42483)
        
        
        # Call to integrate(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'r' (line 138)
        r_42487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 24), 'r', False)
        # Obtaining the member 't' of a type (line 138)
        t_42488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 24), r_42487, 't')
        float_42489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 30), 'float')
        # Applying the binary operator '+' (line 138)
        result_add_42490 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 24), '+', t_42488, float_42489)
        
        # Processing the call keyword arguments (line 138)
        kwargs_42491 = {}
        # Getting the type of 'r' (line 138)
        r_42485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'r', False)
        # Obtaining the member 'integrate' of a type (line 138)
        integrate_42486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), r_42485, 'integrate')
        # Calling integrate(args, kwargs) (line 138)
        integrate_call_result_42492 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), integrate_42486, *[result_add_42490], **kwargs_42491)
        
        
        # Call to integrate(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'r2' (line 139)
        r2_42495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 25), 'r2', False)
        # Obtaining the member 't' of a type (line 139)
        t_42496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 25), r2_42495, 't')
        float_42497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 32), 'float')
        # Applying the binary operator '+' (line 139)
        result_add_42498 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 25), '+', t_42496, float_42497)
        
        # Processing the call keyword arguments (line 139)
        kwargs_42499 = {}
        # Getting the type of 'r2' (line 139)
        r2_42493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'r2', False)
        # Obtaining the member 'integrate' of a type (line 139)
        integrate_42494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), r2_42493, 'integrate')
        # Calling integrate(args, kwargs) (line 139)
        integrate_call_result_42500 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), integrate_42494, *[result_add_42498], **kwargs_42499)
        
        
        # Call to assert_raises(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'RuntimeError' (line 141)
        RuntimeError_42502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 26), 'RuntimeError', False)
        # Getting the type of 'r' (line 141)
        r_42503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 40), 'r', False)
        # Obtaining the member 'integrate' of a type (line 141)
        integrate_42504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 40), r_42503, 'integrate')
        # Getting the type of 'r' (line 141)
        r_42505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 53), 'r', False)
        # Obtaining the member 't' of a type (line 141)
        t_42506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 53), r_42505, 't')
        float_42507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 59), 'float')
        # Applying the binary operator '+' (line 141)
        result_add_42508 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 53), '+', t_42506, float_42507)
        
        # Processing the call keyword arguments (line 141)
        kwargs_42509 = {}
        # Getting the type of 'assert_raises' (line 141)
        assert_raises_42501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 141)
        assert_raises_call_result_42510 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), assert_raises_42501, *[RuntimeError_42502, integrate_42504, result_add_42508], **kwargs_42509)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_concurrent_fail(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_concurrent_fail' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_42511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42511)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_concurrent_fail'
        return stypy_return_type_42511


    @norecursion
    def test_concurrent_ok(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_concurrent_ok'
        module_type_store = module_type_store.open_function_context('test_concurrent_ok', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOde.test_concurrent_ok.__dict__.__setitem__('stypy_localization', localization)
        TestOde.test_concurrent_ok.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOde.test_concurrent_ok.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOde.test_concurrent_ok.__dict__.__setitem__('stypy_function_name', 'TestOde.test_concurrent_ok')
        TestOde.test_concurrent_ok.__dict__.__setitem__('stypy_param_names_list', [])
        TestOde.test_concurrent_ok.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOde.test_concurrent_ok.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOde.test_concurrent_ok.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOde.test_concurrent_ok.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOde.test_concurrent_ok.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOde.test_concurrent_ok.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOde.test_concurrent_ok', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_concurrent_ok', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_concurrent_ok(...)' code ##################

        
        # Assigning a Lambda to a Name (line 144):
        
        # Assigning a Lambda to a Name (line 144):

        @norecursion
        def _stypy_temp_lambda_20(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_20'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_20', 144, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_20.stypy_localization = localization
            _stypy_temp_lambda_20.stypy_type_of_self = None
            _stypy_temp_lambda_20.stypy_type_store = module_type_store
            _stypy_temp_lambda_20.stypy_function_name = '_stypy_temp_lambda_20'
            _stypy_temp_lambda_20.stypy_param_names_list = ['t', 'y']
            _stypy_temp_lambda_20.stypy_varargs_param_name = None
            _stypy_temp_lambda_20.stypy_kwargs_param_name = None
            _stypy_temp_lambda_20.stypy_call_defaults = defaults
            _stypy_temp_lambda_20.stypy_call_varargs = varargs
            _stypy_temp_lambda_20.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_20', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_20', ['t', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            float_42512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 25), 'float')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'stypy_return_type', float_42512)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_20' in the type store
            # Getting the type of 'stypy_return_type' (line 144)
            stypy_return_type_42513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_42513)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_20'
            return stypy_return_type_42513

        # Assigning a type to the variable '_stypy_temp_lambda_20' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), '_stypy_temp_lambda_20', _stypy_temp_lambda_20)
        # Getting the type of '_stypy_temp_lambda_20' (line 144)
        _stypy_temp_lambda_20_42514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), '_stypy_temp_lambda_20')
        # Assigning a type to the variable 'f' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'f', _stypy_temp_lambda_20_42514)
        
        
        # Call to xrange(...): (line 146)
        # Processing the call arguments (line 146)
        int_42516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 24), 'int')
        # Processing the call keyword arguments (line 146)
        kwargs_42517 = {}
        # Getting the type of 'xrange' (line 146)
        xrange_42515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 146)
        xrange_call_result_42518 = invoke(stypy.reporting.localization.Localization(__file__, 146, 17), xrange_42515, *[int_42516], **kwargs_42517)
        
        # Testing the type of a for loop iterable (line 146)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 8), xrange_call_result_42518)
        # Getting the type of the for loop variable (line 146)
        for_loop_var_42519 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 8), xrange_call_result_42518)
        # Assigning a type to the variable 'k' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'k', for_loop_var_42519)
        # SSA begins for a for statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 147)
        tuple_42520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 147)
        # Adding element type (line 147)
        str_42521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 24), 'str', 'vode')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 24), tuple_42520, str_42521)
        # Adding element type (line 147)
        str_42522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 32), 'str', 'zvode')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 24), tuple_42520, str_42522)
        # Adding element type (line 147)
        str_42523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 41), 'str', 'lsoda')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 24), tuple_42520, str_42523)
        # Adding element type (line 147)
        str_42524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 50), 'str', 'dopri5')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 24), tuple_42520, str_42524)
        # Adding element type (line 147)
        str_42525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 60), 'str', 'dop853')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 24), tuple_42520, str_42525)
        
        # Testing the type of a for loop iterable (line 147)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 147, 12), tuple_42520)
        # Getting the type of the for loop variable (line 147)
        for_loop_var_42526 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 147, 12), tuple_42520)
        # Assigning a type to the variable 'sol' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'sol', for_loop_var_42526)
        # SSA begins for a for statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 148):
        
        # Assigning a Call to a Name (line 148):
        
        # Call to set_integrator(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'sol' (line 148)
        sol_42532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 42), 'sol', False)
        # Processing the call keyword arguments (line 148)
        kwargs_42533 = {}
        
        # Call to ode(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'f' (line 148)
        f_42528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'f', False)
        # Processing the call keyword arguments (line 148)
        kwargs_42529 = {}
        # Getting the type of 'ode' (line 148)
        ode_42527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'ode', False)
        # Calling ode(args, kwargs) (line 148)
        ode_call_result_42530 = invoke(stypy.reporting.localization.Localization(__file__, 148, 20), ode_42527, *[f_42528], **kwargs_42529)
        
        # Obtaining the member 'set_integrator' of a type (line 148)
        set_integrator_42531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 20), ode_call_result_42530, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 148)
        set_integrator_call_result_42534 = invoke(stypy.reporting.localization.Localization(__file__, 148, 20), set_integrator_42531, *[sol_42532], **kwargs_42533)
        
        # Assigning a type to the variable 'r' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'r', set_integrator_call_result_42534)
        
        # Call to set_initial_value(...): (line 149)
        # Processing the call arguments (line 149)
        int_42537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 36), 'int')
        int_42538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 39), 'int')
        # Processing the call keyword arguments (line 149)
        kwargs_42539 = {}
        # Getting the type of 'r' (line 149)
        r_42535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'r', False)
        # Obtaining the member 'set_initial_value' of a type (line 149)
        set_initial_value_42536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), r_42535, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 149)
        set_initial_value_call_result_42540 = invoke(stypy.reporting.localization.Localization(__file__, 149, 16), set_initial_value_42536, *[int_42537, int_42538], **kwargs_42539)
        
        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to set_integrator(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'sol' (line 151)
        sol_42546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 43), 'sol', False)
        # Processing the call keyword arguments (line 151)
        kwargs_42547 = {}
        
        # Call to ode(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'f' (line 151)
        f_42542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'f', False)
        # Processing the call keyword arguments (line 151)
        kwargs_42543 = {}
        # Getting the type of 'ode' (line 151)
        ode_42541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'ode', False)
        # Calling ode(args, kwargs) (line 151)
        ode_call_result_42544 = invoke(stypy.reporting.localization.Localization(__file__, 151, 21), ode_42541, *[f_42542], **kwargs_42543)
        
        # Obtaining the member 'set_integrator' of a type (line 151)
        set_integrator_42545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 21), ode_call_result_42544, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 151)
        set_integrator_call_result_42548 = invoke(stypy.reporting.localization.Localization(__file__, 151, 21), set_integrator_42545, *[sol_42546], **kwargs_42547)
        
        # Assigning a type to the variable 'r2' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'r2', set_integrator_call_result_42548)
        
        # Call to set_initial_value(...): (line 152)
        # Processing the call arguments (line 152)
        int_42551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 37), 'int')
        int_42552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 40), 'int')
        # Processing the call keyword arguments (line 152)
        kwargs_42553 = {}
        # Getting the type of 'r2' (line 152)
        r2_42549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'r2', False)
        # Obtaining the member 'set_initial_value' of a type (line 152)
        set_initial_value_42550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), r2_42549, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 152)
        set_initial_value_call_result_42554 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), set_initial_value_42550, *[int_42551, int_42552], **kwargs_42553)
        
        
        # Call to integrate(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'r' (line 154)
        r_42557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 28), 'r', False)
        # Obtaining the member 't' of a type (line 154)
        t_42558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 28), r_42557, 't')
        float_42559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 34), 'float')
        # Applying the binary operator '+' (line 154)
        result_add_42560 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 28), '+', t_42558, float_42559)
        
        # Processing the call keyword arguments (line 154)
        kwargs_42561 = {}
        # Getting the type of 'r' (line 154)
        r_42555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'r', False)
        # Obtaining the member 'integrate' of a type (line 154)
        integrate_42556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 16), r_42555, 'integrate')
        # Calling integrate(args, kwargs) (line 154)
        integrate_call_result_42562 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), integrate_42556, *[result_add_42560], **kwargs_42561)
        
        
        # Call to integrate(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'r2' (line 155)
        r2_42565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 29), 'r2', False)
        # Obtaining the member 't' of a type (line 155)
        t_42566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 29), r2_42565, 't')
        float_42567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 36), 'float')
        # Applying the binary operator '+' (line 155)
        result_add_42568 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 29), '+', t_42566, float_42567)
        
        # Processing the call keyword arguments (line 155)
        kwargs_42569 = {}
        # Getting the type of 'r2' (line 155)
        r2_42563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'r2', False)
        # Obtaining the member 'integrate' of a type (line 155)
        integrate_42564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 16), r2_42563, 'integrate')
        # Calling integrate(args, kwargs) (line 155)
        integrate_call_result_42570 = invoke(stypy.reporting.localization.Localization(__file__, 155, 16), integrate_42564, *[result_add_42568], **kwargs_42569)
        
        
        # Call to integrate(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'r2' (line 156)
        r2_42573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 29), 'r2', False)
        # Obtaining the member 't' of a type (line 156)
        t_42574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 29), r2_42573, 't')
        float_42575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 36), 'float')
        # Applying the binary operator '+' (line 156)
        result_add_42576 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 29), '+', t_42574, float_42575)
        
        # Processing the call keyword arguments (line 156)
        kwargs_42577 = {}
        # Getting the type of 'r2' (line 156)
        r2_42571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'r2', False)
        # Obtaining the member 'integrate' of a type (line 156)
        integrate_42572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 16), r2_42571, 'integrate')
        # Calling integrate(args, kwargs) (line 156)
        integrate_call_result_42578 = invoke(stypy.reporting.localization.Localization(__file__, 156, 16), integrate_42572, *[result_add_42576], **kwargs_42577)
        
        
        # Call to assert_allclose(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'r' (line 158)
        r_42580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 32), 'r', False)
        # Obtaining the member 'y' of a type (line 158)
        y_42581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 32), r_42580, 'y')
        float_42582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 37), 'float')
        # Processing the call keyword arguments (line 158)
        kwargs_42583 = {}
        # Getting the type of 'assert_allclose' (line 158)
        assert_allclose_42579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 158)
        assert_allclose_call_result_42584 = invoke(stypy.reporting.localization.Localization(__file__, 158, 16), assert_allclose_42579, *[y_42581, float_42582], **kwargs_42583)
        
        
        # Call to assert_allclose(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'r2' (line 159)
        r2_42586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 'r2', False)
        # Obtaining the member 'y' of a type (line 159)
        y_42587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 32), r2_42586, 'y')
        float_42588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 38), 'float')
        # Processing the call keyword arguments (line 159)
        kwargs_42589 = {}
        # Getting the type of 'assert_allclose' (line 159)
        assert_allclose_42585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 159)
        assert_allclose_call_result_42590 = invoke(stypy.reporting.localization.Localization(__file__, 159, 16), assert_allclose_42585, *[y_42587, float_42588], **kwargs_42589)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 161)
        tuple_42591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 161)
        # Adding element type (line 161)
        str_42592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 24), 'str', 'dopri5')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 24), tuple_42591, str_42592)
        # Adding element type (line 161)
        str_42593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 34), 'str', 'dop853')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 24), tuple_42591, str_42593)
        
        # Testing the type of a for loop iterable (line 161)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 161, 12), tuple_42591)
        # Getting the type of the for loop variable (line 161)
        for_loop_var_42594 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 161, 12), tuple_42591)
        # Assigning a type to the variable 'sol' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'sol', for_loop_var_42594)
        # SSA begins for a for statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 162):
        
        # Assigning a Call to a Name (line 162):
        
        # Call to set_integrator(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'sol' (line 162)
        sol_42600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 42), 'sol', False)
        # Processing the call keyword arguments (line 162)
        kwargs_42601 = {}
        
        # Call to ode(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'f' (line 162)
        f_42596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 24), 'f', False)
        # Processing the call keyword arguments (line 162)
        kwargs_42597 = {}
        # Getting the type of 'ode' (line 162)
        ode_42595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'ode', False)
        # Calling ode(args, kwargs) (line 162)
        ode_call_result_42598 = invoke(stypy.reporting.localization.Localization(__file__, 162, 20), ode_42595, *[f_42596], **kwargs_42597)
        
        # Obtaining the member 'set_integrator' of a type (line 162)
        set_integrator_42599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 20), ode_call_result_42598, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 162)
        set_integrator_call_result_42602 = invoke(stypy.reporting.localization.Localization(__file__, 162, 20), set_integrator_42599, *[sol_42600], **kwargs_42601)
        
        # Assigning a type to the variable 'r' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'r', set_integrator_call_result_42602)
        
        # Call to set_initial_value(...): (line 163)
        # Processing the call arguments (line 163)
        int_42605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 36), 'int')
        int_42606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 39), 'int')
        # Processing the call keyword arguments (line 163)
        kwargs_42607 = {}
        # Getting the type of 'r' (line 163)
        r_42603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'r', False)
        # Obtaining the member 'set_initial_value' of a type (line 163)
        set_initial_value_42604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 16), r_42603, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 163)
        set_initial_value_call_result_42608 = invoke(stypy.reporting.localization.Localization(__file__, 163, 16), set_initial_value_42604, *[int_42605, int_42606], **kwargs_42607)
        
        
        # Assigning a Call to a Name (line 165):
        
        # Assigning a Call to a Name (line 165):
        
        # Call to set_integrator(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'sol' (line 165)
        sol_42614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 43), 'sol', False)
        # Processing the call keyword arguments (line 165)
        kwargs_42615 = {}
        
        # Call to ode(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'f' (line 165)
        f_42610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'f', False)
        # Processing the call keyword arguments (line 165)
        kwargs_42611 = {}
        # Getting the type of 'ode' (line 165)
        ode_42609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 21), 'ode', False)
        # Calling ode(args, kwargs) (line 165)
        ode_call_result_42612 = invoke(stypy.reporting.localization.Localization(__file__, 165, 21), ode_42609, *[f_42610], **kwargs_42611)
        
        # Obtaining the member 'set_integrator' of a type (line 165)
        set_integrator_42613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 21), ode_call_result_42612, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 165)
        set_integrator_call_result_42616 = invoke(stypy.reporting.localization.Localization(__file__, 165, 21), set_integrator_42613, *[sol_42614], **kwargs_42615)
        
        # Assigning a type to the variable 'r2' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'r2', set_integrator_call_result_42616)
        
        # Call to set_initial_value(...): (line 166)
        # Processing the call arguments (line 166)
        int_42619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 37), 'int')
        int_42620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 40), 'int')
        # Processing the call keyword arguments (line 166)
        kwargs_42621 = {}
        # Getting the type of 'r2' (line 166)
        r2_42617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'r2', False)
        # Obtaining the member 'set_initial_value' of a type (line 166)
        set_initial_value_42618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 16), r2_42617, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 166)
        set_initial_value_call_result_42622 = invoke(stypy.reporting.localization.Localization(__file__, 166, 16), set_initial_value_42618, *[int_42619, int_42620], **kwargs_42621)
        
        
        # Call to integrate(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'r' (line 168)
        r_42625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 28), 'r', False)
        # Obtaining the member 't' of a type (line 168)
        t_42626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 28), r_42625, 't')
        float_42627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 34), 'float')
        # Applying the binary operator '+' (line 168)
        result_add_42628 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 28), '+', t_42626, float_42627)
        
        # Processing the call keyword arguments (line 168)
        kwargs_42629 = {}
        # Getting the type of 'r' (line 168)
        r_42623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'r', False)
        # Obtaining the member 'integrate' of a type (line 168)
        integrate_42624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 16), r_42623, 'integrate')
        # Calling integrate(args, kwargs) (line 168)
        integrate_call_result_42630 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), integrate_42624, *[result_add_42628], **kwargs_42629)
        
        
        # Call to integrate(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'r' (line 169)
        r_42633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'r', False)
        # Obtaining the member 't' of a type (line 169)
        t_42634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 28), r_42633, 't')
        float_42635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 34), 'float')
        # Applying the binary operator '+' (line 169)
        result_add_42636 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 28), '+', t_42634, float_42635)
        
        # Processing the call keyword arguments (line 169)
        kwargs_42637 = {}
        # Getting the type of 'r' (line 169)
        r_42631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'r', False)
        # Obtaining the member 'integrate' of a type (line 169)
        integrate_42632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 16), r_42631, 'integrate')
        # Calling integrate(args, kwargs) (line 169)
        integrate_call_result_42638 = invoke(stypy.reporting.localization.Localization(__file__, 169, 16), integrate_42632, *[result_add_42636], **kwargs_42637)
        
        
        # Call to integrate(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'r2' (line 170)
        r2_42641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 29), 'r2', False)
        # Obtaining the member 't' of a type (line 170)
        t_42642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 29), r2_42641, 't')
        float_42643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 36), 'float')
        # Applying the binary operator '+' (line 170)
        result_add_42644 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 29), '+', t_42642, float_42643)
        
        # Processing the call keyword arguments (line 170)
        kwargs_42645 = {}
        # Getting the type of 'r2' (line 170)
        r2_42639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'r2', False)
        # Obtaining the member 'integrate' of a type (line 170)
        integrate_42640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), r2_42639, 'integrate')
        # Calling integrate(args, kwargs) (line 170)
        integrate_call_result_42646 = invoke(stypy.reporting.localization.Localization(__file__, 170, 16), integrate_42640, *[result_add_42644], **kwargs_42645)
        
        
        # Call to integrate(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'r' (line 171)
        r_42649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 28), 'r', False)
        # Obtaining the member 't' of a type (line 171)
        t_42650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 28), r_42649, 't')
        float_42651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 34), 'float')
        # Applying the binary operator '+' (line 171)
        result_add_42652 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 28), '+', t_42650, float_42651)
        
        # Processing the call keyword arguments (line 171)
        kwargs_42653 = {}
        # Getting the type of 'r' (line 171)
        r_42647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'r', False)
        # Obtaining the member 'integrate' of a type (line 171)
        integrate_42648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 16), r_42647, 'integrate')
        # Calling integrate(args, kwargs) (line 171)
        integrate_call_result_42654 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), integrate_42648, *[result_add_42652], **kwargs_42653)
        
        
        # Call to integrate(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'r2' (line 172)
        r2_42657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 29), 'r2', False)
        # Obtaining the member 't' of a type (line 172)
        t_42658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 29), r2_42657, 't')
        float_42659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 36), 'float')
        # Applying the binary operator '+' (line 172)
        result_add_42660 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 29), '+', t_42658, float_42659)
        
        # Processing the call keyword arguments (line 172)
        kwargs_42661 = {}
        # Getting the type of 'r2' (line 172)
        r2_42655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'r2', False)
        # Obtaining the member 'integrate' of a type (line 172)
        integrate_42656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), r2_42655, 'integrate')
        # Calling integrate(args, kwargs) (line 172)
        integrate_call_result_42662 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), integrate_42656, *[result_add_42660], **kwargs_42661)
        
        
        # Call to assert_allclose(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'r' (line 174)
        r_42664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'r', False)
        # Obtaining the member 'y' of a type (line 174)
        y_42665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 32), r_42664, 'y')
        float_42666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 37), 'float')
        # Processing the call keyword arguments (line 174)
        kwargs_42667 = {}
        # Getting the type of 'assert_allclose' (line 174)
        assert_allclose_42663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 174)
        assert_allclose_call_result_42668 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), assert_allclose_42663, *[y_42665, float_42666], **kwargs_42667)
        
        
        # Call to assert_allclose(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'r2' (line 175)
        r2_42670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 32), 'r2', False)
        # Obtaining the member 'y' of a type (line 175)
        y_42671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 32), r2_42670, 'y')
        float_42672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 38), 'float')
        # Processing the call keyword arguments (line 175)
        kwargs_42673 = {}
        # Getting the type of 'assert_allclose' (line 175)
        assert_allclose_42669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 175)
        assert_allclose_call_result_42674 = invoke(stypy.reporting.localization.Localization(__file__, 175, 16), assert_allclose_42669, *[y_42671, float_42672], **kwargs_42673)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_concurrent_ok(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_concurrent_ok' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_42675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42675)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_concurrent_ok'
        return stypy_return_type_42675


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 74, 0, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOde.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestOde' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'TestOde', TestOde)

# Assigning a Name to a Name (line 76):
# Getting the type of 'ode' (line 76)
ode_42676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'ode')
# Getting the type of 'TestOde'
TestOde_42677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestOde')
# Setting the type of the member 'ode_class' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestOde_42677, 'ode_class', ode_42676)
# Declaration of the 'TestComplexOde' class
# Getting the type of 'TestODEClass' (line 178)
TestODEClass_42678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'TestODEClass')

class TestComplexOde(TestODEClass_42678, ):
    
    # Assigning a Name to a Name (line 180):

    @norecursion
    def test_vode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vode'
        module_type_store = module_type_store.open_function_context('test_vode', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplexOde.test_vode.__dict__.__setitem__('stypy_localization', localization)
        TestComplexOde.test_vode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplexOde.test_vode.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplexOde.test_vode.__dict__.__setitem__('stypy_function_name', 'TestComplexOde.test_vode')
        TestComplexOde.test_vode.__dict__.__setitem__('stypy_param_names_list', [])
        TestComplexOde.test_vode.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplexOde.test_vode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplexOde.test_vode.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplexOde.test_vode.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplexOde.test_vode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplexOde.test_vode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplexOde.test_vode', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vode', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vode(...)' code ##################

        
        # Getting the type of 'PROBLEMS' (line 184)
        PROBLEMS_42679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'PROBLEMS')
        # Testing the type of a for loop iterable (line 184)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 184, 8), PROBLEMS_42679)
        # Getting the type of the for loop variable (line 184)
        for_loop_var_42680 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 184, 8), PROBLEMS_42679)
        # Assigning a type to the variable 'problem_cls' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'problem_cls', for_loop_var_42680)
        # SSA begins for a for statement (line 184)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 185):
        
        # Assigning a Call to a Name (line 185):
        
        # Call to problem_cls(...): (line 185)
        # Processing the call keyword arguments (line 185)
        kwargs_42682 = {}
        # Getting the type of 'problem_cls' (line 185)
        problem_cls_42681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 22), 'problem_cls', False)
        # Calling problem_cls(args, kwargs) (line 185)
        problem_cls_call_result_42683 = invoke(stypy.reporting.localization.Localization(__file__, 185, 22), problem_cls_42681, *[], **kwargs_42682)
        
        # Assigning a type to the variable 'problem' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'problem', problem_cls_call_result_42683)
        
        
        # Getting the type of 'problem' (line 186)
        problem_42684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 19), 'problem')
        # Obtaining the member 'stiff' of a type (line 186)
        stiff_42685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 19), problem_42684, 'stiff')
        # Applying the 'not' unary operator (line 186)
        result_not__42686 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 15), 'not', stiff_42685)
        
        # Testing the type of an if condition (line 186)
        if_condition_42687 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 12), result_not__42686)
        # Assigning a type to the variable 'if_condition_42687' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'if_condition_42687', if_condition_42687)
        # SSA begins for if statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _do_problem(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'problem' (line 187)
        problem_42690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 33), 'problem', False)
        str_42691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 42), 'str', 'vode')
        str_42692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 50), 'str', 'adams')
        # Processing the call keyword arguments (line 187)
        kwargs_42693 = {}
        # Getting the type of 'self' (line 187)
        self_42688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'self', False)
        # Obtaining the member '_do_problem' of a type (line 187)
        _do_problem_42689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 16), self_42688, '_do_problem')
        # Calling _do_problem(args, kwargs) (line 187)
        _do_problem_call_result_42694 = invoke(stypy.reporting.localization.Localization(__file__, 187, 16), _do_problem_42689, *[problem_42690, str_42691, str_42692], **kwargs_42693)
        
        # SSA branch for the else part of an if statement (line 186)
        module_type_store.open_ssa_branch('else')
        
        # Call to _do_problem(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'problem' (line 189)
        problem_42697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 33), 'problem', False)
        str_42698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 42), 'str', 'vode')
        str_42699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 50), 'str', 'bdf')
        # Processing the call keyword arguments (line 189)
        kwargs_42700 = {}
        # Getting the type of 'self' (line 189)
        self_42695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'self', False)
        # Obtaining the member '_do_problem' of a type (line 189)
        _do_problem_42696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 16), self_42695, '_do_problem')
        # Calling _do_problem(args, kwargs) (line 189)
        _do_problem_call_result_42701 = invoke(stypy.reporting.localization.Localization(__file__, 189, 16), _do_problem_42696, *[problem_42697, str_42698, str_42699], **kwargs_42700)
        
        # SSA join for if statement (line 186)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_vode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vode' in the type store
        # Getting the type of 'stypy_return_type' (line 182)
        stypy_return_type_42702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42702)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vode'
        return stypy_return_type_42702


    @norecursion
    def test_lsoda(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_lsoda'
        module_type_store = module_type_store.open_function_context('test_lsoda', 191, 4, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplexOde.test_lsoda.__dict__.__setitem__('stypy_localization', localization)
        TestComplexOde.test_lsoda.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplexOde.test_lsoda.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplexOde.test_lsoda.__dict__.__setitem__('stypy_function_name', 'TestComplexOde.test_lsoda')
        TestComplexOde.test_lsoda.__dict__.__setitem__('stypy_param_names_list', [])
        TestComplexOde.test_lsoda.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplexOde.test_lsoda.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplexOde.test_lsoda.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplexOde.test_lsoda.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplexOde.test_lsoda.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplexOde.test_lsoda.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplexOde.test_lsoda', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_lsoda', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_lsoda(...)' code ##################

        
        # Getting the type of 'PROBLEMS' (line 193)
        PROBLEMS_42703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 27), 'PROBLEMS')
        # Testing the type of a for loop iterable (line 193)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 193, 8), PROBLEMS_42703)
        # Getting the type of the for loop variable (line 193)
        for_loop_var_42704 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 193, 8), PROBLEMS_42703)
        # Assigning a type to the variable 'problem_cls' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'problem_cls', for_loop_var_42704)
        # SSA begins for a for statement (line 193)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 194):
        
        # Assigning a Call to a Name (line 194):
        
        # Call to problem_cls(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_42706 = {}
        # Getting the type of 'problem_cls' (line 194)
        problem_cls_42705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 22), 'problem_cls', False)
        # Calling problem_cls(args, kwargs) (line 194)
        problem_cls_call_result_42707 = invoke(stypy.reporting.localization.Localization(__file__, 194, 22), problem_cls_42705, *[], **kwargs_42706)
        
        # Assigning a type to the variable 'problem' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'problem', problem_cls_call_result_42707)
        
        # Call to _do_problem(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'problem' (line 195)
        problem_42710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 29), 'problem', False)
        str_42711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 38), 'str', 'lsoda')
        # Processing the call keyword arguments (line 195)
        kwargs_42712 = {}
        # Getting the type of 'self' (line 195)
        self_42708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'self', False)
        # Obtaining the member '_do_problem' of a type (line 195)
        _do_problem_42709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), self_42708, '_do_problem')
        # Calling _do_problem(args, kwargs) (line 195)
        _do_problem_call_result_42713 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), _do_problem_42709, *[problem_42710, str_42711], **kwargs_42712)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_lsoda(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_lsoda' in the type store
        # Getting the type of 'stypy_return_type' (line 191)
        stypy_return_type_42714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42714)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_lsoda'
        return stypy_return_type_42714


    @norecursion
    def test_dopri5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dopri5'
        module_type_store = module_type_store.open_function_context('test_dopri5', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplexOde.test_dopri5.__dict__.__setitem__('stypy_localization', localization)
        TestComplexOde.test_dopri5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplexOde.test_dopri5.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplexOde.test_dopri5.__dict__.__setitem__('stypy_function_name', 'TestComplexOde.test_dopri5')
        TestComplexOde.test_dopri5.__dict__.__setitem__('stypy_param_names_list', [])
        TestComplexOde.test_dopri5.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplexOde.test_dopri5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplexOde.test_dopri5.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplexOde.test_dopri5.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplexOde.test_dopri5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplexOde.test_dopri5.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplexOde.test_dopri5', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dopri5', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dopri5(...)' code ##################

        
        # Getting the type of 'PROBLEMS' (line 199)
        PROBLEMS_42715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'PROBLEMS')
        # Testing the type of a for loop iterable (line 199)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 199, 8), PROBLEMS_42715)
        # Getting the type of the for loop variable (line 199)
        for_loop_var_42716 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 199, 8), PROBLEMS_42715)
        # Assigning a type to the variable 'problem_cls' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'problem_cls', for_loop_var_42716)
        # SSA begins for a for statement (line 199)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 200):
        
        # Assigning a Call to a Name (line 200):
        
        # Call to problem_cls(...): (line 200)
        # Processing the call keyword arguments (line 200)
        kwargs_42718 = {}
        # Getting the type of 'problem_cls' (line 200)
        problem_cls_42717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'problem_cls', False)
        # Calling problem_cls(args, kwargs) (line 200)
        problem_cls_call_result_42719 = invoke(stypy.reporting.localization.Localization(__file__, 200, 22), problem_cls_42717, *[], **kwargs_42718)
        
        # Assigning a type to the variable 'problem' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'problem', problem_cls_call_result_42719)
        
        # Getting the type of 'problem' (line 201)
        problem_42720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'problem')
        # Obtaining the member 'stiff' of a type (line 201)
        stiff_42721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 15), problem_42720, 'stiff')
        # Testing the type of an if condition (line 201)
        if_condition_42722 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 12), stiff_42721)
        # Assigning a type to the variable 'if_condition_42722' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'if_condition_42722', if_condition_42722)
        # SSA begins for if statement (line 201)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 201)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 203)
        str_42723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 32), 'str', 'jac')
        # Getting the type of 'problem' (line 203)
        problem_42724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'problem')
        
        (may_be_42725, more_types_in_union_42726) = may_provide_member(str_42723, problem_42724)

        if may_be_42725:

            if more_types_in_union_42726:
                # Runtime conditional SSA (line 203)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'problem' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'problem', remove_not_member_provider_from_union(problem_42724, 'jac'))

            if more_types_in_union_42726:
                # SSA join for if statement (line 203)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _do_problem(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'problem' (line 205)
        problem_42729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 29), 'problem', False)
        str_42730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 38), 'str', 'dopri5')
        # Processing the call keyword arguments (line 205)
        kwargs_42731 = {}
        # Getting the type of 'self' (line 205)
        self_42727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'self', False)
        # Obtaining the member '_do_problem' of a type (line 205)
        _do_problem_42728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), self_42727, '_do_problem')
        # Calling _do_problem(args, kwargs) (line 205)
        _do_problem_call_result_42732 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), _do_problem_42728, *[problem_42729, str_42730], **kwargs_42731)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dopri5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dopri5' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_42733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42733)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dopri5'
        return stypy_return_type_42733


    @norecursion
    def test_dop853(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dop853'
        module_type_store = module_type_store.open_function_context('test_dop853', 207, 4, False)
        # Assigning a type to the variable 'self' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplexOde.test_dop853.__dict__.__setitem__('stypy_localization', localization)
        TestComplexOde.test_dop853.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplexOde.test_dop853.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplexOde.test_dop853.__dict__.__setitem__('stypy_function_name', 'TestComplexOde.test_dop853')
        TestComplexOde.test_dop853.__dict__.__setitem__('stypy_param_names_list', [])
        TestComplexOde.test_dop853.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplexOde.test_dop853.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplexOde.test_dop853.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplexOde.test_dop853.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplexOde.test_dop853.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplexOde.test_dop853.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplexOde.test_dop853', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dop853', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dop853(...)' code ##################

        
        # Getting the type of 'PROBLEMS' (line 209)
        PROBLEMS_42734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 27), 'PROBLEMS')
        # Testing the type of a for loop iterable (line 209)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 209, 8), PROBLEMS_42734)
        # Getting the type of the for loop variable (line 209)
        for_loop_var_42735 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 209, 8), PROBLEMS_42734)
        # Assigning a type to the variable 'problem_cls' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'problem_cls', for_loop_var_42735)
        # SSA begins for a for statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 210):
        
        # Assigning a Call to a Name (line 210):
        
        # Call to problem_cls(...): (line 210)
        # Processing the call keyword arguments (line 210)
        kwargs_42737 = {}
        # Getting the type of 'problem_cls' (line 210)
        problem_cls_42736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'problem_cls', False)
        # Calling problem_cls(args, kwargs) (line 210)
        problem_cls_call_result_42738 = invoke(stypy.reporting.localization.Localization(__file__, 210, 22), problem_cls_42736, *[], **kwargs_42737)
        
        # Assigning a type to the variable 'problem' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'problem', problem_cls_call_result_42738)
        
        # Getting the type of 'problem' (line 211)
        problem_42739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'problem')
        # Obtaining the member 'stiff' of a type (line 211)
        stiff_42740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 15), problem_42739, 'stiff')
        # Testing the type of an if condition (line 211)
        if_condition_42741 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 12), stiff_42740)
        # Assigning a type to the variable 'if_condition_42741' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'if_condition_42741', if_condition_42741)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 213)
        str_42742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 32), 'str', 'jac')
        # Getting the type of 'problem' (line 213)
        problem_42743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 23), 'problem')
        
        (may_be_42744, more_types_in_union_42745) = may_provide_member(str_42742, problem_42743)

        if may_be_42744:

            if more_types_in_union_42745:
                # Runtime conditional SSA (line 213)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'problem' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'problem', remove_not_member_provider_from_union(problem_42743, 'jac'))

            if more_types_in_union_42745:
                # SSA join for if statement (line 213)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _do_problem(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'problem' (line 215)
        problem_42748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 29), 'problem', False)
        str_42749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 38), 'str', 'dop853')
        # Processing the call keyword arguments (line 215)
        kwargs_42750 = {}
        # Getting the type of 'self' (line 215)
        self_42746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'self', False)
        # Obtaining the member '_do_problem' of a type (line 215)
        _do_problem_42747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), self_42746, '_do_problem')
        # Calling _do_problem(args, kwargs) (line 215)
        _do_problem_call_result_42751 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), _do_problem_42747, *[problem_42748, str_42749], **kwargs_42750)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dop853(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dop853' in the type store
        # Getting the type of 'stypy_return_type' (line 207)
        stypy_return_type_42752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42752)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dop853'
        return stypy_return_type_42752


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 178, 0, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplexOde.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestComplexOde' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'TestComplexOde', TestComplexOde)

# Assigning a Name to a Name (line 180):
# Getting the type of 'complex_ode' (line 180)
complex_ode_42753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'complex_ode')
# Getting the type of 'TestComplexOde'
TestComplexOde_42754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestComplexOde')
# Setting the type of the member 'ode_class' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestComplexOde_42754, 'ode_class', complex_ode_42753)
# Declaration of the 'TestSolout' class

class TestSolout(object, ):

    @norecursion
    def _run_solout_test(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_run_solout_test'
        module_type_store = module_type_store.open_function_context('_run_solout_test', 220, 4, False)
        # Assigning a type to the variable 'self' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSolout._run_solout_test.__dict__.__setitem__('stypy_localization', localization)
        TestSolout._run_solout_test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSolout._run_solout_test.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSolout._run_solout_test.__dict__.__setitem__('stypy_function_name', 'TestSolout._run_solout_test')
        TestSolout._run_solout_test.__dict__.__setitem__('stypy_param_names_list', ['integrator'])
        TestSolout._run_solout_test.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSolout._run_solout_test.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSolout._run_solout_test.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSolout._run_solout_test.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSolout._run_solout_test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSolout._run_solout_test.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSolout._run_solout_test', ['integrator'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_run_solout_test', localization, ['integrator'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_run_solout_test(...)' code ##################

        
        # Assigning a List to a Name (line 222):
        
        # Assigning a List to a Name (line 222):
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_42755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        
        # Assigning a type to the variable 'ts' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'ts', list_42755)
        
        # Assigning a List to a Name (line 223):
        
        # Assigning a List to a Name (line 223):
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_42756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        
        # Assigning a type to the variable 'ys' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'ys', list_42756)
        
        # Assigning a Num to a Name (line 224):
        
        # Assigning a Num to a Name (line 224):
        float_42757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 13), 'float')
        # Assigning a type to the variable 't0' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 't0', float_42757)
        
        # Assigning a Num to a Name (line 225):
        
        # Assigning a Num to a Name (line 225):
        float_42758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 15), 'float')
        # Assigning a type to the variable 'tend' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tend', float_42758)
        
        # Assigning a List to a Name (line 226):
        
        # Assigning a List to a Name (line 226):
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_42759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        float_42760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 13), list_42759, float_42760)
        # Adding element type (line 226)
        float_42761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 13), list_42759, float_42761)
        
        # Assigning a type to the variable 'y0' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'y0', list_42759)

        @norecursion
        def solout(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'solout'
            module_type_store = module_type_store.open_function_context('solout', 228, 8, False)
            
            # Passed parameters checking function
            solout.stypy_localization = localization
            solout.stypy_type_of_self = None
            solout.stypy_type_store = module_type_store
            solout.stypy_function_name = 'solout'
            solout.stypy_param_names_list = ['t', 'y']
            solout.stypy_varargs_param_name = None
            solout.stypy_kwargs_param_name = None
            solout.stypy_call_defaults = defaults
            solout.stypy_call_varargs = varargs
            solout.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'solout', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'solout', localization, ['t', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'solout(...)' code ##################

            
            # Call to append(...): (line 229)
            # Processing the call arguments (line 229)
            # Getting the type of 't' (line 229)
            t_42764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 22), 't', False)
            # Processing the call keyword arguments (line 229)
            kwargs_42765 = {}
            # Getting the type of 'ts' (line 229)
            ts_42762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'ts', False)
            # Obtaining the member 'append' of a type (line 229)
            append_42763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), ts_42762, 'append')
            # Calling append(args, kwargs) (line 229)
            append_call_result_42766 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), append_42763, *[t_42764], **kwargs_42765)
            
            
            # Call to append(...): (line 230)
            # Processing the call arguments (line 230)
            
            # Call to copy(...): (line 230)
            # Processing the call keyword arguments (line 230)
            kwargs_42771 = {}
            # Getting the type of 'y' (line 230)
            y_42769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 22), 'y', False)
            # Obtaining the member 'copy' of a type (line 230)
            copy_42770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 22), y_42769, 'copy')
            # Calling copy(args, kwargs) (line 230)
            copy_call_result_42772 = invoke(stypy.reporting.localization.Localization(__file__, 230, 22), copy_42770, *[], **kwargs_42771)
            
            # Processing the call keyword arguments (line 230)
            kwargs_42773 = {}
            # Getting the type of 'ys' (line 230)
            ys_42767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'ys', False)
            # Obtaining the member 'append' of a type (line 230)
            append_42768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), ys_42767, 'append')
            # Calling append(args, kwargs) (line 230)
            append_call_result_42774 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), append_42768, *[copy_call_result_42772], **kwargs_42773)
            
            
            # ################# End of 'solout(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'solout' in the type store
            # Getting the type of 'stypy_return_type' (line 228)
            stypy_return_type_42775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_42775)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'solout'
            return stypy_return_type_42775

        # Assigning a type to the variable 'solout' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'solout', solout)

        @norecursion
        def rhs(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'rhs'
            module_type_store = module_type_store.open_function_context('rhs', 232, 8, False)
            
            # Passed parameters checking function
            rhs.stypy_localization = localization
            rhs.stypy_type_of_self = None
            rhs.stypy_type_store = module_type_store
            rhs.stypy_function_name = 'rhs'
            rhs.stypy_param_names_list = ['t', 'y']
            rhs.stypy_varargs_param_name = None
            rhs.stypy_kwargs_param_name = None
            rhs.stypy_call_defaults = defaults
            rhs.stypy_call_varargs = varargs
            rhs.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'rhs', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'rhs', localization, ['t', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'rhs(...)' code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 233)
            list_42776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 233)
            # Adding element type (line 233)
            
            # Obtaining the type of the subscript
            int_42777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 22), 'int')
            # Getting the type of 'y' (line 233)
            y_42778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'y')
            # Obtaining the member '__getitem__' of a type (line 233)
            getitem___42779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 20), y_42778, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 233)
            subscript_call_result_42780 = invoke(stypy.reporting.localization.Localization(__file__, 233, 20), getitem___42779, int_42777)
            
            
            # Obtaining the type of the subscript
            int_42781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 29), 'int')
            # Getting the type of 'y' (line 233)
            y_42782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'y')
            # Obtaining the member '__getitem__' of a type (line 233)
            getitem___42783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 27), y_42782, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 233)
            subscript_call_result_42784 = invoke(stypy.reporting.localization.Localization(__file__, 233, 27), getitem___42783, int_42781)
            
            # Applying the binary operator '+' (line 233)
            result_add_42785 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 20), '+', subscript_call_result_42780, subscript_call_result_42784)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 19), list_42776, result_add_42785)
            # Adding element type (line 233)
            
            
            # Obtaining the type of the subscript
            int_42786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 36), 'int')
            # Getting the type of 'y' (line 233)
            y_42787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 34), 'y')
            # Obtaining the member '__getitem__' of a type (line 233)
            getitem___42788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 34), y_42787, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 233)
            subscript_call_result_42789 = invoke(stypy.reporting.localization.Localization(__file__, 233, 34), getitem___42788, int_42786)
            
            int_42790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 40), 'int')
            # Applying the binary operator '**' (line 233)
            result_pow_42791 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 34), '**', subscript_call_result_42789, int_42790)
            
            # Applying the 'usub' unary operator (line 233)
            result___neg___42792 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 33), 'usub', result_pow_42791)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 19), list_42776, result___neg___42792)
            
            # Assigning a type to the variable 'stypy_return_type' (line 233)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'stypy_return_type', list_42776)
            
            # ################# End of 'rhs(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'rhs' in the type store
            # Getting the type of 'stypy_return_type' (line 232)
            stypy_return_type_42793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_42793)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'rhs'
            return stypy_return_type_42793

        # Assigning a type to the variable 'rhs' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'rhs', rhs)
        
        # Assigning a Call to a Name (line 235):
        
        # Assigning a Call to a Name (line 235):
        
        # Call to set_integrator(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'integrator' (line 235)
        integrator_42799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 37), 'integrator', False)
        # Processing the call keyword arguments (line 235)
        kwargs_42800 = {}
        
        # Call to ode(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'rhs' (line 235)
        rhs_42795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 17), 'rhs', False)
        # Processing the call keyword arguments (line 235)
        kwargs_42796 = {}
        # Getting the type of 'ode' (line 235)
        ode_42794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'ode', False)
        # Calling ode(args, kwargs) (line 235)
        ode_call_result_42797 = invoke(stypy.reporting.localization.Localization(__file__, 235, 13), ode_42794, *[rhs_42795], **kwargs_42796)
        
        # Obtaining the member 'set_integrator' of a type (line 235)
        set_integrator_42798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 13), ode_call_result_42797, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 235)
        set_integrator_call_result_42801 = invoke(stypy.reporting.localization.Localization(__file__, 235, 13), set_integrator_42798, *[integrator_42799], **kwargs_42800)
        
        # Assigning a type to the variable 'ig' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'ig', set_integrator_call_result_42801)
        
        # Call to set_solout(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'solout' (line 236)
        solout_42804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 22), 'solout', False)
        # Processing the call keyword arguments (line 236)
        kwargs_42805 = {}
        # Getting the type of 'ig' (line 236)
        ig_42802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'ig', False)
        # Obtaining the member 'set_solout' of a type (line 236)
        set_solout_42803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), ig_42802, 'set_solout')
        # Calling set_solout(args, kwargs) (line 236)
        set_solout_call_result_42806 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), set_solout_42803, *[solout_42804], **kwargs_42805)
        
        
        # Call to set_initial_value(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'y0' (line 237)
        y0_42809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 29), 'y0', False)
        # Getting the type of 't0' (line 237)
        t0_42810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 't0', False)
        # Processing the call keyword arguments (line 237)
        kwargs_42811 = {}
        # Getting the type of 'ig' (line 237)
        ig_42807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'ig', False)
        # Obtaining the member 'set_initial_value' of a type (line 237)
        set_initial_value_42808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), ig_42807, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 237)
        set_initial_value_call_result_42812 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), set_initial_value_42808, *[y0_42809, t0_42810], **kwargs_42811)
        
        
        # Assigning a Call to a Name (line 238):
        
        # Assigning a Call to a Name (line 238):
        
        # Call to integrate(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'tend' (line 238)
        tend_42815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 27), 'tend', False)
        # Processing the call keyword arguments (line 238)
        kwargs_42816 = {}
        # Getting the type of 'ig' (line 238)
        ig_42813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 14), 'ig', False)
        # Obtaining the member 'integrate' of a type (line 238)
        integrate_42814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 14), ig_42813, 'integrate')
        # Calling integrate(args, kwargs) (line 238)
        integrate_call_result_42817 = invoke(stypy.reporting.localization.Localization(__file__, 238, 14), integrate_42814, *[tend_42815], **kwargs_42816)
        
        # Assigning a type to the variable 'ret' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'ret', integrate_call_result_42817)
        
        # Call to assert_array_equal(...): (line 239)
        # Processing the call arguments (line 239)
        
        # Obtaining the type of the subscript
        int_42819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 30), 'int')
        # Getting the type of 'ys' (line 239)
        ys_42820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'ys', False)
        # Obtaining the member '__getitem__' of a type (line 239)
        getitem___42821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 27), ys_42820, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 239)
        subscript_call_result_42822 = invoke(stypy.reporting.localization.Localization(__file__, 239, 27), getitem___42821, int_42819)
        
        # Getting the type of 'y0' (line 239)
        y0_42823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 34), 'y0', False)
        # Processing the call keyword arguments (line 239)
        kwargs_42824 = {}
        # Getting the type of 'assert_array_equal' (line 239)
        assert_array_equal_42818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 239)
        assert_array_equal_call_result_42825 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), assert_array_equal_42818, *[subscript_call_result_42822, y0_42823], **kwargs_42824)
        
        
        # Call to assert_array_equal(...): (line 240)
        # Processing the call arguments (line 240)
        
        # Obtaining the type of the subscript
        int_42827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 30), 'int')
        # Getting the type of 'ys' (line 240)
        ys_42828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 27), 'ys', False)
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___42829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 27), ys_42828, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 240)
        subscript_call_result_42830 = invoke(stypy.reporting.localization.Localization(__file__, 240, 27), getitem___42829, int_42827)
        
        # Getting the type of 'ret' (line 240)
        ret_42831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 35), 'ret', False)
        # Processing the call keyword arguments (line 240)
        kwargs_42832 = {}
        # Getting the type of 'assert_array_equal' (line 240)
        assert_array_equal_42826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 240)
        assert_array_equal_call_result_42833 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), assert_array_equal_42826, *[subscript_call_result_42830, ret_42831], **kwargs_42832)
        
        
        # Call to assert_equal(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Obtaining the type of the subscript
        int_42835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 24), 'int')
        # Getting the type of 'ts' (line 241)
        ts_42836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'ts', False)
        # Obtaining the member '__getitem__' of a type (line 241)
        getitem___42837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 21), ts_42836, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 241)
        subscript_call_result_42838 = invoke(stypy.reporting.localization.Localization(__file__, 241, 21), getitem___42837, int_42835)
        
        # Getting the type of 't0' (line 241)
        t0_42839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 28), 't0', False)
        # Processing the call keyword arguments (line 241)
        kwargs_42840 = {}
        # Getting the type of 'assert_equal' (line 241)
        assert_equal_42834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 241)
        assert_equal_call_result_42841 = invoke(stypy.reporting.localization.Localization(__file__, 241, 8), assert_equal_42834, *[subscript_call_result_42838, t0_42839], **kwargs_42840)
        
        
        # Call to assert_equal(...): (line 242)
        # Processing the call arguments (line 242)
        
        # Obtaining the type of the subscript
        int_42843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 24), 'int')
        # Getting the type of 'ts' (line 242)
        ts_42844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 21), 'ts', False)
        # Obtaining the member '__getitem__' of a type (line 242)
        getitem___42845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 21), ts_42844, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 242)
        subscript_call_result_42846 = invoke(stypy.reporting.localization.Localization(__file__, 242, 21), getitem___42845, int_42843)
        
        # Getting the type of 'tend' (line 242)
        tend_42847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 29), 'tend', False)
        # Processing the call keyword arguments (line 242)
        kwargs_42848 = {}
        # Getting the type of 'assert_equal' (line 242)
        assert_equal_42842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 242)
        assert_equal_call_result_42849 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), assert_equal_42842, *[subscript_call_result_42846, tend_42847], **kwargs_42848)
        
        
        # ################# End of '_run_solout_test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_run_solout_test' in the type store
        # Getting the type of 'stypy_return_type' (line 220)
        stypy_return_type_42850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42850)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_run_solout_test'
        return stypy_return_type_42850


    @norecursion
    def test_solout(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_solout'
        module_type_store = module_type_store.open_function_context('test_solout', 244, 4, False)
        # Assigning a type to the variable 'self' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSolout.test_solout.__dict__.__setitem__('stypy_localization', localization)
        TestSolout.test_solout.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSolout.test_solout.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSolout.test_solout.__dict__.__setitem__('stypy_function_name', 'TestSolout.test_solout')
        TestSolout.test_solout.__dict__.__setitem__('stypy_param_names_list', [])
        TestSolout.test_solout.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSolout.test_solout.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSolout.test_solout.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSolout.test_solout.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSolout.test_solout.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSolout.test_solout.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSolout.test_solout', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_solout', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_solout(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 245)
        tuple_42851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 245)
        # Adding element type (line 245)
        str_42852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 27), 'str', 'dopri5')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 27), tuple_42851, str_42852)
        # Adding element type (line 245)
        str_42853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 37), 'str', 'dop853')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 27), tuple_42851, str_42853)
        
        # Testing the type of a for loop iterable (line 245)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 245, 8), tuple_42851)
        # Getting the type of the for loop variable (line 245)
        for_loop_var_42854 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 245, 8), tuple_42851)
        # Assigning a type to the variable 'integrator' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'integrator', for_loop_var_42854)
        # SSA begins for a for statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _run_solout_test(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'integrator' (line 246)
        integrator_42857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 34), 'integrator', False)
        # Processing the call keyword arguments (line 246)
        kwargs_42858 = {}
        # Getting the type of 'self' (line 246)
        self_42855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'self', False)
        # Obtaining the member '_run_solout_test' of a type (line 246)
        _run_solout_test_42856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 12), self_42855, '_run_solout_test')
        # Calling _run_solout_test(args, kwargs) (line 246)
        _run_solout_test_call_result_42859 = invoke(stypy.reporting.localization.Localization(__file__, 246, 12), _run_solout_test_42856, *[integrator_42857], **kwargs_42858)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_solout(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_solout' in the type store
        # Getting the type of 'stypy_return_type' (line 244)
        stypy_return_type_42860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42860)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_solout'
        return stypy_return_type_42860


    @norecursion
    def _run_solout_after_initial_test(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_run_solout_after_initial_test'
        module_type_store = module_type_store.open_function_context('_run_solout_after_initial_test', 248, 4, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSolout._run_solout_after_initial_test.__dict__.__setitem__('stypy_localization', localization)
        TestSolout._run_solout_after_initial_test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSolout._run_solout_after_initial_test.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSolout._run_solout_after_initial_test.__dict__.__setitem__('stypy_function_name', 'TestSolout._run_solout_after_initial_test')
        TestSolout._run_solout_after_initial_test.__dict__.__setitem__('stypy_param_names_list', ['integrator'])
        TestSolout._run_solout_after_initial_test.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSolout._run_solout_after_initial_test.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSolout._run_solout_after_initial_test.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSolout._run_solout_after_initial_test.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSolout._run_solout_after_initial_test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSolout._run_solout_after_initial_test.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSolout._run_solout_after_initial_test', ['integrator'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_run_solout_after_initial_test', localization, ['integrator'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_run_solout_after_initial_test(...)' code ##################

        
        # Assigning a List to a Name (line 250):
        
        # Assigning a List to a Name (line 250):
        
        # Obtaining an instance of the builtin type 'list' (line 250)
        list_42861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 250)
        
        # Assigning a type to the variable 'ts' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'ts', list_42861)
        
        # Assigning a List to a Name (line 251):
        
        # Assigning a List to a Name (line 251):
        
        # Obtaining an instance of the builtin type 'list' (line 251)
        list_42862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 251)
        
        # Assigning a type to the variable 'ys' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'ys', list_42862)
        
        # Assigning a Num to a Name (line 252):
        
        # Assigning a Num to a Name (line 252):
        float_42863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 13), 'float')
        # Assigning a type to the variable 't0' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 't0', float_42863)
        
        # Assigning a Num to a Name (line 253):
        
        # Assigning a Num to a Name (line 253):
        float_42864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 15), 'float')
        # Assigning a type to the variable 'tend' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'tend', float_42864)
        
        # Assigning a List to a Name (line 254):
        
        # Assigning a List to a Name (line 254):
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_42865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        float_42866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 13), list_42865, float_42866)
        # Adding element type (line 254)
        float_42867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 13), list_42865, float_42867)
        
        # Assigning a type to the variable 'y0' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'y0', list_42865)

        @norecursion
        def solout(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'solout'
            module_type_store = module_type_store.open_function_context('solout', 256, 8, False)
            
            # Passed parameters checking function
            solout.stypy_localization = localization
            solout.stypy_type_of_self = None
            solout.stypy_type_store = module_type_store
            solout.stypy_function_name = 'solout'
            solout.stypy_param_names_list = ['t', 'y']
            solout.stypy_varargs_param_name = None
            solout.stypy_kwargs_param_name = None
            solout.stypy_call_defaults = defaults
            solout.stypy_call_varargs = varargs
            solout.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'solout', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'solout', localization, ['t', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'solout(...)' code ##################

            
            # Call to append(...): (line 257)
            # Processing the call arguments (line 257)
            # Getting the type of 't' (line 257)
            t_42870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 22), 't', False)
            # Processing the call keyword arguments (line 257)
            kwargs_42871 = {}
            # Getting the type of 'ts' (line 257)
            ts_42868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'ts', False)
            # Obtaining the member 'append' of a type (line 257)
            append_42869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), ts_42868, 'append')
            # Calling append(args, kwargs) (line 257)
            append_call_result_42872 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), append_42869, *[t_42870], **kwargs_42871)
            
            
            # Call to append(...): (line 258)
            # Processing the call arguments (line 258)
            
            # Call to copy(...): (line 258)
            # Processing the call keyword arguments (line 258)
            kwargs_42877 = {}
            # Getting the type of 'y' (line 258)
            y_42875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 22), 'y', False)
            # Obtaining the member 'copy' of a type (line 258)
            copy_42876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 22), y_42875, 'copy')
            # Calling copy(args, kwargs) (line 258)
            copy_call_result_42878 = invoke(stypy.reporting.localization.Localization(__file__, 258, 22), copy_42876, *[], **kwargs_42877)
            
            # Processing the call keyword arguments (line 258)
            kwargs_42879 = {}
            # Getting the type of 'ys' (line 258)
            ys_42873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'ys', False)
            # Obtaining the member 'append' of a type (line 258)
            append_42874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 12), ys_42873, 'append')
            # Calling append(args, kwargs) (line 258)
            append_call_result_42880 = invoke(stypy.reporting.localization.Localization(__file__, 258, 12), append_42874, *[copy_call_result_42878], **kwargs_42879)
            
            
            # ################# End of 'solout(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'solout' in the type store
            # Getting the type of 'stypy_return_type' (line 256)
            stypy_return_type_42881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_42881)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'solout'
            return stypy_return_type_42881

        # Assigning a type to the variable 'solout' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'solout', solout)

        @norecursion
        def rhs(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'rhs'
            module_type_store = module_type_store.open_function_context('rhs', 260, 8, False)
            
            # Passed parameters checking function
            rhs.stypy_localization = localization
            rhs.stypy_type_of_self = None
            rhs.stypy_type_store = module_type_store
            rhs.stypy_function_name = 'rhs'
            rhs.stypy_param_names_list = ['t', 'y']
            rhs.stypy_varargs_param_name = None
            rhs.stypy_kwargs_param_name = None
            rhs.stypy_call_defaults = defaults
            rhs.stypy_call_varargs = varargs
            rhs.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'rhs', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'rhs', localization, ['t', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'rhs(...)' code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 261)
            list_42882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 261)
            # Adding element type (line 261)
            
            # Obtaining the type of the subscript
            int_42883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 22), 'int')
            # Getting the type of 'y' (line 261)
            y_42884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'y')
            # Obtaining the member '__getitem__' of a type (line 261)
            getitem___42885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 20), y_42884, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 261)
            subscript_call_result_42886 = invoke(stypy.reporting.localization.Localization(__file__, 261, 20), getitem___42885, int_42883)
            
            
            # Obtaining the type of the subscript
            int_42887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 29), 'int')
            # Getting the type of 'y' (line 261)
            y_42888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 27), 'y')
            # Obtaining the member '__getitem__' of a type (line 261)
            getitem___42889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 27), y_42888, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 261)
            subscript_call_result_42890 = invoke(stypy.reporting.localization.Localization(__file__, 261, 27), getitem___42889, int_42887)
            
            # Applying the binary operator '+' (line 261)
            result_add_42891 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 20), '+', subscript_call_result_42886, subscript_call_result_42890)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 19), list_42882, result_add_42891)
            # Adding element type (line 261)
            
            
            # Obtaining the type of the subscript
            int_42892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 36), 'int')
            # Getting the type of 'y' (line 261)
            y_42893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 34), 'y')
            # Obtaining the member '__getitem__' of a type (line 261)
            getitem___42894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 34), y_42893, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 261)
            subscript_call_result_42895 = invoke(stypy.reporting.localization.Localization(__file__, 261, 34), getitem___42894, int_42892)
            
            int_42896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 40), 'int')
            # Applying the binary operator '**' (line 261)
            result_pow_42897 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 34), '**', subscript_call_result_42895, int_42896)
            
            # Applying the 'usub' unary operator (line 261)
            result___neg___42898 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 33), 'usub', result_pow_42897)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 19), list_42882, result___neg___42898)
            
            # Assigning a type to the variable 'stypy_return_type' (line 261)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'stypy_return_type', list_42882)
            
            # ################# End of 'rhs(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'rhs' in the type store
            # Getting the type of 'stypy_return_type' (line 260)
            stypy_return_type_42899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_42899)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'rhs'
            return stypy_return_type_42899

        # Assigning a type to the variable 'rhs' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'rhs', rhs)
        
        # Assigning a Call to a Name (line 263):
        
        # Assigning a Call to a Name (line 263):
        
        # Call to set_integrator(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'integrator' (line 263)
        integrator_42905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 37), 'integrator', False)
        # Processing the call keyword arguments (line 263)
        kwargs_42906 = {}
        
        # Call to ode(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'rhs' (line 263)
        rhs_42901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 17), 'rhs', False)
        # Processing the call keyword arguments (line 263)
        kwargs_42902 = {}
        # Getting the type of 'ode' (line 263)
        ode_42900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 13), 'ode', False)
        # Calling ode(args, kwargs) (line 263)
        ode_call_result_42903 = invoke(stypy.reporting.localization.Localization(__file__, 263, 13), ode_42900, *[rhs_42901], **kwargs_42902)
        
        # Obtaining the member 'set_integrator' of a type (line 263)
        set_integrator_42904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 13), ode_call_result_42903, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 263)
        set_integrator_call_result_42907 = invoke(stypy.reporting.localization.Localization(__file__, 263, 13), set_integrator_42904, *[integrator_42905], **kwargs_42906)
        
        # Assigning a type to the variable 'ig' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'ig', set_integrator_call_result_42907)
        
        # Call to set_initial_value(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'y0' (line 264)
        y0_42910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 29), 'y0', False)
        # Getting the type of 't0' (line 264)
        t0_42911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 33), 't0', False)
        # Processing the call keyword arguments (line 264)
        kwargs_42912 = {}
        # Getting the type of 'ig' (line 264)
        ig_42908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'ig', False)
        # Obtaining the member 'set_initial_value' of a type (line 264)
        set_initial_value_42909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), ig_42908, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 264)
        set_initial_value_call_result_42913 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), set_initial_value_42909, *[y0_42910, t0_42911], **kwargs_42912)
        
        
        # Call to set_solout(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'solout' (line 265)
        solout_42916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 22), 'solout', False)
        # Processing the call keyword arguments (line 265)
        kwargs_42917 = {}
        # Getting the type of 'ig' (line 265)
        ig_42914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'ig', False)
        # Obtaining the member 'set_solout' of a type (line 265)
        set_solout_42915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), ig_42914, 'set_solout')
        # Calling set_solout(args, kwargs) (line 265)
        set_solout_call_result_42918 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), set_solout_42915, *[solout_42916], **kwargs_42917)
        
        
        # Assigning a Call to a Name (line 266):
        
        # Assigning a Call to a Name (line 266):
        
        # Call to integrate(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'tend' (line 266)
        tend_42921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 27), 'tend', False)
        # Processing the call keyword arguments (line 266)
        kwargs_42922 = {}
        # Getting the type of 'ig' (line 266)
        ig_42919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 14), 'ig', False)
        # Obtaining the member 'integrate' of a type (line 266)
        integrate_42920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 14), ig_42919, 'integrate')
        # Calling integrate(args, kwargs) (line 266)
        integrate_call_result_42923 = invoke(stypy.reporting.localization.Localization(__file__, 266, 14), integrate_42920, *[tend_42921], **kwargs_42922)
        
        # Assigning a type to the variable 'ret' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'ret', integrate_call_result_42923)
        
        # Call to assert_array_equal(...): (line 267)
        # Processing the call arguments (line 267)
        
        # Obtaining the type of the subscript
        int_42925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 30), 'int')
        # Getting the type of 'ys' (line 267)
        ys_42926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 27), 'ys', False)
        # Obtaining the member '__getitem__' of a type (line 267)
        getitem___42927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 27), ys_42926, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 267)
        subscript_call_result_42928 = invoke(stypy.reporting.localization.Localization(__file__, 267, 27), getitem___42927, int_42925)
        
        # Getting the type of 'y0' (line 267)
        y0_42929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 34), 'y0', False)
        # Processing the call keyword arguments (line 267)
        kwargs_42930 = {}
        # Getting the type of 'assert_array_equal' (line 267)
        assert_array_equal_42924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 267)
        assert_array_equal_call_result_42931 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), assert_array_equal_42924, *[subscript_call_result_42928, y0_42929], **kwargs_42930)
        
        
        # Call to assert_array_equal(...): (line 268)
        # Processing the call arguments (line 268)
        
        # Obtaining the type of the subscript
        int_42933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 30), 'int')
        # Getting the type of 'ys' (line 268)
        ys_42934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 27), 'ys', False)
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___42935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 27), ys_42934, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_42936 = invoke(stypy.reporting.localization.Localization(__file__, 268, 27), getitem___42935, int_42933)
        
        # Getting the type of 'ret' (line 268)
        ret_42937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 35), 'ret', False)
        # Processing the call keyword arguments (line 268)
        kwargs_42938 = {}
        # Getting the type of 'assert_array_equal' (line 268)
        assert_array_equal_42932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 268)
        assert_array_equal_call_result_42939 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), assert_array_equal_42932, *[subscript_call_result_42936, ret_42937], **kwargs_42938)
        
        
        # Call to assert_equal(...): (line 269)
        # Processing the call arguments (line 269)
        
        # Obtaining the type of the subscript
        int_42941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 24), 'int')
        # Getting the type of 'ts' (line 269)
        ts_42942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 21), 'ts', False)
        # Obtaining the member '__getitem__' of a type (line 269)
        getitem___42943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 21), ts_42942, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 269)
        subscript_call_result_42944 = invoke(stypy.reporting.localization.Localization(__file__, 269, 21), getitem___42943, int_42941)
        
        # Getting the type of 't0' (line 269)
        t0_42945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 28), 't0', False)
        # Processing the call keyword arguments (line 269)
        kwargs_42946 = {}
        # Getting the type of 'assert_equal' (line 269)
        assert_equal_42940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 269)
        assert_equal_call_result_42947 = invoke(stypy.reporting.localization.Localization(__file__, 269, 8), assert_equal_42940, *[subscript_call_result_42944, t0_42945], **kwargs_42946)
        
        
        # Call to assert_equal(...): (line 270)
        # Processing the call arguments (line 270)
        
        # Obtaining the type of the subscript
        int_42949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 24), 'int')
        # Getting the type of 'ts' (line 270)
        ts_42950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 21), 'ts', False)
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___42951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 21), ts_42950, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_42952 = invoke(stypy.reporting.localization.Localization(__file__, 270, 21), getitem___42951, int_42949)
        
        # Getting the type of 'tend' (line 270)
        tend_42953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 29), 'tend', False)
        # Processing the call keyword arguments (line 270)
        kwargs_42954 = {}
        # Getting the type of 'assert_equal' (line 270)
        assert_equal_42948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 270)
        assert_equal_call_result_42955 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), assert_equal_42948, *[subscript_call_result_42952, tend_42953], **kwargs_42954)
        
        
        # ################# End of '_run_solout_after_initial_test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_run_solout_after_initial_test' in the type store
        # Getting the type of 'stypy_return_type' (line 248)
        stypy_return_type_42956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42956)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_run_solout_after_initial_test'
        return stypy_return_type_42956


    @norecursion
    def test_solout_after_initial(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_solout_after_initial'
        module_type_store = module_type_store.open_function_context('test_solout_after_initial', 272, 4, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSolout.test_solout_after_initial.__dict__.__setitem__('stypy_localization', localization)
        TestSolout.test_solout_after_initial.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSolout.test_solout_after_initial.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSolout.test_solout_after_initial.__dict__.__setitem__('stypy_function_name', 'TestSolout.test_solout_after_initial')
        TestSolout.test_solout_after_initial.__dict__.__setitem__('stypy_param_names_list', [])
        TestSolout.test_solout_after_initial.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSolout.test_solout_after_initial.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSolout.test_solout_after_initial.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSolout.test_solout_after_initial.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSolout.test_solout_after_initial.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSolout.test_solout_after_initial.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSolout.test_solout_after_initial', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_solout_after_initial', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_solout_after_initial(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 273)
        tuple_42957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 273)
        # Adding element type (line 273)
        str_42958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 27), 'str', 'dopri5')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 27), tuple_42957, str_42958)
        # Adding element type (line 273)
        str_42959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 37), 'str', 'dop853')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 27), tuple_42957, str_42959)
        
        # Testing the type of a for loop iterable (line 273)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 273, 8), tuple_42957)
        # Getting the type of the for loop variable (line 273)
        for_loop_var_42960 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 273, 8), tuple_42957)
        # Assigning a type to the variable 'integrator' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'integrator', for_loop_var_42960)
        # SSA begins for a for statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _run_solout_after_initial_test(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'integrator' (line 274)
        integrator_42963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 48), 'integrator', False)
        # Processing the call keyword arguments (line 274)
        kwargs_42964 = {}
        # Getting the type of 'self' (line 274)
        self_42961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'self', False)
        # Obtaining the member '_run_solout_after_initial_test' of a type (line 274)
        _run_solout_after_initial_test_42962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 12), self_42961, '_run_solout_after_initial_test')
        # Calling _run_solout_after_initial_test(args, kwargs) (line 274)
        _run_solout_after_initial_test_call_result_42965 = invoke(stypy.reporting.localization.Localization(__file__, 274, 12), _run_solout_after_initial_test_42962, *[integrator_42963], **kwargs_42964)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_solout_after_initial(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_solout_after_initial' in the type store
        # Getting the type of 'stypy_return_type' (line 272)
        stypy_return_type_42966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42966)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_solout_after_initial'
        return stypy_return_type_42966


    @norecursion
    def _run_solout_break_test(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_run_solout_break_test'
        module_type_store = module_type_store.open_function_context('_run_solout_break_test', 276, 4, False)
        # Assigning a type to the variable 'self' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSolout._run_solout_break_test.__dict__.__setitem__('stypy_localization', localization)
        TestSolout._run_solout_break_test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSolout._run_solout_break_test.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSolout._run_solout_break_test.__dict__.__setitem__('stypy_function_name', 'TestSolout._run_solout_break_test')
        TestSolout._run_solout_break_test.__dict__.__setitem__('stypy_param_names_list', ['integrator'])
        TestSolout._run_solout_break_test.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSolout._run_solout_break_test.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSolout._run_solout_break_test.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSolout._run_solout_break_test.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSolout._run_solout_break_test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSolout._run_solout_break_test.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSolout._run_solout_break_test', ['integrator'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_run_solout_break_test', localization, ['integrator'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_run_solout_break_test(...)' code ##################

        
        # Assigning a List to a Name (line 278):
        
        # Assigning a List to a Name (line 278):
        
        # Obtaining an instance of the builtin type 'list' (line 278)
        list_42967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 278)
        
        # Assigning a type to the variable 'ts' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'ts', list_42967)
        
        # Assigning a List to a Name (line 279):
        
        # Assigning a List to a Name (line 279):
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_42968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        
        # Assigning a type to the variable 'ys' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'ys', list_42968)
        
        # Assigning a Num to a Name (line 280):
        
        # Assigning a Num to a Name (line 280):
        float_42969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 13), 'float')
        # Assigning a type to the variable 't0' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 't0', float_42969)
        
        # Assigning a Num to a Name (line 281):
        
        # Assigning a Num to a Name (line 281):
        float_42970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 15), 'float')
        # Assigning a type to the variable 'tend' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'tend', float_42970)
        
        # Assigning a List to a Name (line 282):
        
        # Assigning a List to a Name (line 282):
        
        # Obtaining an instance of the builtin type 'list' (line 282)
        list_42971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 282)
        # Adding element type (line 282)
        float_42972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 13), list_42971, float_42972)
        # Adding element type (line 282)
        float_42973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 13), list_42971, float_42973)
        
        # Assigning a type to the variable 'y0' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'y0', list_42971)

        @norecursion
        def solout(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'solout'
            module_type_store = module_type_store.open_function_context('solout', 284, 8, False)
            
            # Passed parameters checking function
            solout.stypy_localization = localization
            solout.stypy_type_of_self = None
            solout.stypy_type_store = module_type_store
            solout.stypy_function_name = 'solout'
            solout.stypy_param_names_list = ['t', 'y']
            solout.stypy_varargs_param_name = None
            solout.stypy_kwargs_param_name = None
            solout.stypy_call_defaults = defaults
            solout.stypy_call_varargs = varargs
            solout.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'solout', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'solout', localization, ['t', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'solout(...)' code ##################

            
            # Call to append(...): (line 285)
            # Processing the call arguments (line 285)
            # Getting the type of 't' (line 285)
            t_42976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 22), 't', False)
            # Processing the call keyword arguments (line 285)
            kwargs_42977 = {}
            # Getting the type of 'ts' (line 285)
            ts_42974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'ts', False)
            # Obtaining the member 'append' of a type (line 285)
            append_42975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 12), ts_42974, 'append')
            # Calling append(args, kwargs) (line 285)
            append_call_result_42978 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), append_42975, *[t_42976], **kwargs_42977)
            
            
            # Call to append(...): (line 286)
            # Processing the call arguments (line 286)
            
            # Call to copy(...): (line 286)
            # Processing the call keyword arguments (line 286)
            kwargs_42983 = {}
            # Getting the type of 'y' (line 286)
            y_42981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'y', False)
            # Obtaining the member 'copy' of a type (line 286)
            copy_42982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 22), y_42981, 'copy')
            # Calling copy(args, kwargs) (line 286)
            copy_call_result_42984 = invoke(stypy.reporting.localization.Localization(__file__, 286, 22), copy_42982, *[], **kwargs_42983)
            
            # Processing the call keyword arguments (line 286)
            kwargs_42985 = {}
            # Getting the type of 'ys' (line 286)
            ys_42979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'ys', False)
            # Obtaining the member 'append' of a type (line 286)
            append_42980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 12), ys_42979, 'append')
            # Calling append(args, kwargs) (line 286)
            append_call_result_42986 = invoke(stypy.reporting.localization.Localization(__file__, 286, 12), append_42980, *[copy_call_result_42984], **kwargs_42985)
            
            
            
            # Getting the type of 't' (line 287)
            t_42987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 15), 't')
            # Getting the type of 'tend' (line 287)
            tend_42988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 19), 'tend')
            float_42989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 24), 'float')
            # Applying the binary operator 'div' (line 287)
            result_div_42990 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 19), 'div', tend_42988, float_42989)
            
            # Applying the binary operator '>' (line 287)
            result_gt_42991 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 15), '>', t_42987, result_div_42990)
            
            # Testing the type of an if condition (line 287)
            if_condition_42992 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 12), result_gt_42991)
            # Assigning a type to the variable 'if_condition_42992' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'if_condition_42992', if_condition_42992)
            # SSA begins for if statement (line 287)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_42993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 23), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 288)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'stypy_return_type', int_42993)
            # SSA join for if statement (line 287)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'solout(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'solout' in the type store
            # Getting the type of 'stypy_return_type' (line 284)
            stypy_return_type_42994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_42994)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'solout'
            return stypy_return_type_42994

        # Assigning a type to the variable 'solout' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'solout', solout)

        @norecursion
        def rhs(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'rhs'
            module_type_store = module_type_store.open_function_context('rhs', 290, 8, False)
            
            # Passed parameters checking function
            rhs.stypy_localization = localization
            rhs.stypy_type_of_self = None
            rhs.stypy_type_store = module_type_store
            rhs.stypy_function_name = 'rhs'
            rhs.stypy_param_names_list = ['t', 'y']
            rhs.stypy_varargs_param_name = None
            rhs.stypy_kwargs_param_name = None
            rhs.stypy_call_defaults = defaults
            rhs.stypy_call_varargs = varargs
            rhs.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'rhs', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'rhs', localization, ['t', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'rhs(...)' code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 291)
            list_42995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 291)
            # Adding element type (line 291)
            
            # Obtaining the type of the subscript
            int_42996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 22), 'int')
            # Getting the type of 'y' (line 291)
            y_42997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 'y')
            # Obtaining the member '__getitem__' of a type (line 291)
            getitem___42998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 20), y_42997, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 291)
            subscript_call_result_42999 = invoke(stypy.reporting.localization.Localization(__file__, 291, 20), getitem___42998, int_42996)
            
            
            # Obtaining the type of the subscript
            int_43000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 29), 'int')
            # Getting the type of 'y' (line 291)
            y_43001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 27), 'y')
            # Obtaining the member '__getitem__' of a type (line 291)
            getitem___43002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 27), y_43001, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 291)
            subscript_call_result_43003 = invoke(stypy.reporting.localization.Localization(__file__, 291, 27), getitem___43002, int_43000)
            
            # Applying the binary operator '+' (line 291)
            result_add_43004 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 20), '+', subscript_call_result_42999, subscript_call_result_43003)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 19), list_42995, result_add_43004)
            # Adding element type (line 291)
            
            
            # Obtaining the type of the subscript
            int_43005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 36), 'int')
            # Getting the type of 'y' (line 291)
            y_43006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 34), 'y')
            # Obtaining the member '__getitem__' of a type (line 291)
            getitem___43007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 34), y_43006, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 291)
            subscript_call_result_43008 = invoke(stypy.reporting.localization.Localization(__file__, 291, 34), getitem___43007, int_43005)
            
            int_43009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 40), 'int')
            # Applying the binary operator '**' (line 291)
            result_pow_43010 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 34), '**', subscript_call_result_43008, int_43009)
            
            # Applying the 'usub' unary operator (line 291)
            result___neg___43011 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 33), 'usub', result_pow_43010)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 19), list_42995, result___neg___43011)
            
            # Assigning a type to the variable 'stypy_return_type' (line 291)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'stypy_return_type', list_42995)
            
            # ################# End of 'rhs(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'rhs' in the type store
            # Getting the type of 'stypy_return_type' (line 290)
            stypy_return_type_43012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_43012)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'rhs'
            return stypy_return_type_43012

        # Assigning a type to the variable 'rhs' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'rhs', rhs)
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to set_integrator(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'integrator' (line 293)
        integrator_43018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 37), 'integrator', False)
        # Processing the call keyword arguments (line 293)
        kwargs_43019 = {}
        
        # Call to ode(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'rhs' (line 293)
        rhs_43014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 17), 'rhs', False)
        # Processing the call keyword arguments (line 293)
        kwargs_43015 = {}
        # Getting the type of 'ode' (line 293)
        ode_43013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'ode', False)
        # Calling ode(args, kwargs) (line 293)
        ode_call_result_43016 = invoke(stypy.reporting.localization.Localization(__file__, 293, 13), ode_43013, *[rhs_43014], **kwargs_43015)
        
        # Obtaining the member 'set_integrator' of a type (line 293)
        set_integrator_43017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 13), ode_call_result_43016, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 293)
        set_integrator_call_result_43020 = invoke(stypy.reporting.localization.Localization(__file__, 293, 13), set_integrator_43017, *[integrator_43018], **kwargs_43019)
        
        # Assigning a type to the variable 'ig' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'ig', set_integrator_call_result_43020)
        
        # Call to set_solout(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'solout' (line 294)
        solout_43023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 22), 'solout', False)
        # Processing the call keyword arguments (line 294)
        kwargs_43024 = {}
        # Getting the type of 'ig' (line 294)
        ig_43021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'ig', False)
        # Obtaining the member 'set_solout' of a type (line 294)
        set_solout_43022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), ig_43021, 'set_solout')
        # Calling set_solout(args, kwargs) (line 294)
        set_solout_call_result_43025 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), set_solout_43022, *[solout_43023], **kwargs_43024)
        
        
        # Call to set_initial_value(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'y0' (line 295)
        y0_43028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 29), 'y0', False)
        # Getting the type of 't0' (line 295)
        t0_43029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 33), 't0', False)
        # Processing the call keyword arguments (line 295)
        kwargs_43030 = {}
        # Getting the type of 'ig' (line 295)
        ig_43026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'ig', False)
        # Obtaining the member 'set_initial_value' of a type (line 295)
        set_initial_value_43027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), ig_43026, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 295)
        set_initial_value_call_result_43031 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), set_initial_value_43027, *[y0_43028, t0_43029], **kwargs_43030)
        
        
        # Assigning a Call to a Name (line 296):
        
        # Assigning a Call to a Name (line 296):
        
        # Call to integrate(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'tend' (line 296)
        tend_43034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 27), 'tend', False)
        # Processing the call keyword arguments (line 296)
        kwargs_43035 = {}
        # Getting the type of 'ig' (line 296)
        ig_43032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 14), 'ig', False)
        # Obtaining the member 'integrate' of a type (line 296)
        integrate_43033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 14), ig_43032, 'integrate')
        # Calling integrate(args, kwargs) (line 296)
        integrate_call_result_43036 = invoke(stypy.reporting.localization.Localization(__file__, 296, 14), integrate_43033, *[tend_43034], **kwargs_43035)
        
        # Assigning a type to the variable 'ret' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'ret', integrate_call_result_43036)
        
        # Call to assert_array_equal(...): (line 297)
        # Processing the call arguments (line 297)
        
        # Obtaining the type of the subscript
        int_43038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 30), 'int')
        # Getting the type of 'ys' (line 297)
        ys_43039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 27), 'ys', False)
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___43040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 27), ys_43039, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_43041 = invoke(stypy.reporting.localization.Localization(__file__, 297, 27), getitem___43040, int_43038)
        
        # Getting the type of 'y0' (line 297)
        y0_43042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 34), 'y0', False)
        # Processing the call keyword arguments (line 297)
        kwargs_43043 = {}
        # Getting the type of 'assert_array_equal' (line 297)
        assert_array_equal_43037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 297)
        assert_array_equal_call_result_43044 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), assert_array_equal_43037, *[subscript_call_result_43041, y0_43042], **kwargs_43043)
        
        
        # Call to assert_array_equal(...): (line 298)
        # Processing the call arguments (line 298)
        
        # Obtaining the type of the subscript
        int_43046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 30), 'int')
        # Getting the type of 'ys' (line 298)
        ys_43047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 27), 'ys', False)
        # Obtaining the member '__getitem__' of a type (line 298)
        getitem___43048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 27), ys_43047, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 298)
        subscript_call_result_43049 = invoke(stypy.reporting.localization.Localization(__file__, 298, 27), getitem___43048, int_43046)
        
        # Getting the type of 'ret' (line 298)
        ret_43050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 35), 'ret', False)
        # Processing the call keyword arguments (line 298)
        kwargs_43051 = {}
        # Getting the type of 'assert_array_equal' (line 298)
        assert_array_equal_43045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 298)
        assert_array_equal_call_result_43052 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), assert_array_equal_43045, *[subscript_call_result_43049, ret_43050], **kwargs_43051)
        
        
        # Call to assert_equal(...): (line 299)
        # Processing the call arguments (line 299)
        
        # Obtaining the type of the subscript
        int_43054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 24), 'int')
        # Getting the type of 'ts' (line 299)
        ts_43055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 21), 'ts', False)
        # Obtaining the member '__getitem__' of a type (line 299)
        getitem___43056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 21), ts_43055, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 299)
        subscript_call_result_43057 = invoke(stypy.reporting.localization.Localization(__file__, 299, 21), getitem___43056, int_43054)
        
        # Getting the type of 't0' (line 299)
        t0_43058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 28), 't0', False)
        # Processing the call keyword arguments (line 299)
        kwargs_43059 = {}
        # Getting the type of 'assert_equal' (line 299)
        assert_equal_43053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 299)
        assert_equal_call_result_43060 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), assert_equal_43053, *[subscript_call_result_43057, t0_43058], **kwargs_43059)
        
        
        # Call to assert_(...): (line 300)
        # Processing the call arguments (line 300)
        
        
        # Obtaining the type of the subscript
        int_43062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 19), 'int')
        # Getting the type of 'ts' (line 300)
        ts_43063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'ts', False)
        # Obtaining the member '__getitem__' of a type (line 300)
        getitem___43064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 16), ts_43063, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
        subscript_call_result_43065 = invoke(stypy.reporting.localization.Localization(__file__, 300, 16), getitem___43064, int_43062)
        
        # Getting the type of 'tend' (line 300)
        tend_43066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 25), 'tend', False)
        float_43067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 30), 'float')
        # Applying the binary operator 'div' (line 300)
        result_div_43068 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 25), 'div', tend_43066, float_43067)
        
        # Applying the binary operator '>' (line 300)
        result_gt_43069 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 16), '>', subscript_call_result_43065, result_div_43068)
        
        # Processing the call keyword arguments (line 300)
        kwargs_43070 = {}
        # Getting the type of 'assert_' (line 300)
        assert__43061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 300)
        assert__call_result_43071 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), assert__43061, *[result_gt_43069], **kwargs_43070)
        
        
        # Call to assert_(...): (line 301)
        # Processing the call arguments (line 301)
        
        
        # Obtaining the type of the subscript
        int_43073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 19), 'int')
        # Getting the type of 'ts' (line 301)
        ts_43074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'ts', False)
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___43075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 16), ts_43074, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 301)
        subscript_call_result_43076 = invoke(stypy.reporting.localization.Localization(__file__, 301, 16), getitem___43075, int_43073)
        
        # Getting the type of 'tend' (line 301)
        tend_43077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'tend', False)
        # Applying the binary operator '<' (line 301)
        result_lt_43078 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 16), '<', subscript_call_result_43076, tend_43077)
        
        # Processing the call keyword arguments (line 301)
        kwargs_43079 = {}
        # Getting the type of 'assert_' (line 301)
        assert__43072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 301)
        assert__call_result_43080 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), assert__43072, *[result_lt_43078], **kwargs_43079)
        
        
        # ################# End of '_run_solout_break_test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_run_solout_break_test' in the type store
        # Getting the type of 'stypy_return_type' (line 276)
        stypy_return_type_43081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43081)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_run_solout_break_test'
        return stypy_return_type_43081


    @norecursion
    def test_solout_break(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_solout_break'
        module_type_store = module_type_store.open_function_context('test_solout_break', 303, 4, False)
        # Assigning a type to the variable 'self' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSolout.test_solout_break.__dict__.__setitem__('stypy_localization', localization)
        TestSolout.test_solout_break.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSolout.test_solout_break.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSolout.test_solout_break.__dict__.__setitem__('stypy_function_name', 'TestSolout.test_solout_break')
        TestSolout.test_solout_break.__dict__.__setitem__('stypy_param_names_list', [])
        TestSolout.test_solout_break.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSolout.test_solout_break.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSolout.test_solout_break.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSolout.test_solout_break.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSolout.test_solout_break.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSolout.test_solout_break.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSolout.test_solout_break', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_solout_break', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_solout_break(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 304)
        tuple_43082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 304)
        # Adding element type (line 304)
        str_43083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 27), 'str', 'dopri5')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 27), tuple_43082, str_43083)
        # Adding element type (line 304)
        str_43084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 37), 'str', 'dop853')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 27), tuple_43082, str_43084)
        
        # Testing the type of a for loop iterable (line 304)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 304, 8), tuple_43082)
        # Getting the type of the for loop variable (line 304)
        for_loop_var_43085 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 304, 8), tuple_43082)
        # Assigning a type to the variable 'integrator' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'integrator', for_loop_var_43085)
        # SSA begins for a for statement (line 304)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _run_solout_break_test(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'integrator' (line 305)
        integrator_43088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 40), 'integrator', False)
        # Processing the call keyword arguments (line 305)
        kwargs_43089 = {}
        # Getting the type of 'self' (line 305)
        self_43086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'self', False)
        # Obtaining the member '_run_solout_break_test' of a type (line 305)
        _run_solout_break_test_43087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 12), self_43086, '_run_solout_break_test')
        # Calling _run_solout_break_test(args, kwargs) (line 305)
        _run_solout_break_test_call_result_43090 = invoke(stypy.reporting.localization.Localization(__file__, 305, 12), _run_solout_break_test_43087, *[integrator_43088], **kwargs_43089)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_solout_break(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_solout_break' in the type store
        # Getting the type of 'stypy_return_type' (line 303)
        stypy_return_type_43091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43091)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_solout_break'
        return stypy_return_type_43091


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 218, 0, False)
        # Assigning a type to the variable 'self' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSolout.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSolout' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'TestSolout', TestSolout)
# Declaration of the 'TestComplexSolout' class

class TestComplexSolout(object, ):

    @norecursion
    def _run_solout_test(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_run_solout_test'
        module_type_store = module_type_store.open_function_context('_run_solout_test', 310, 4, False)
        # Assigning a type to the variable 'self' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplexSolout._run_solout_test.__dict__.__setitem__('stypy_localization', localization)
        TestComplexSolout._run_solout_test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplexSolout._run_solout_test.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplexSolout._run_solout_test.__dict__.__setitem__('stypy_function_name', 'TestComplexSolout._run_solout_test')
        TestComplexSolout._run_solout_test.__dict__.__setitem__('stypy_param_names_list', ['integrator'])
        TestComplexSolout._run_solout_test.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplexSolout._run_solout_test.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplexSolout._run_solout_test.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplexSolout._run_solout_test.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplexSolout._run_solout_test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplexSolout._run_solout_test.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplexSolout._run_solout_test', ['integrator'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_run_solout_test', localization, ['integrator'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_run_solout_test(...)' code ##################

        
        # Assigning a List to a Name (line 312):
        
        # Assigning a List to a Name (line 312):
        
        # Obtaining an instance of the builtin type 'list' (line 312)
        list_43092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 312)
        
        # Assigning a type to the variable 'ts' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'ts', list_43092)
        
        # Assigning a List to a Name (line 313):
        
        # Assigning a List to a Name (line 313):
        
        # Obtaining an instance of the builtin type 'list' (line 313)
        list_43093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 313)
        
        # Assigning a type to the variable 'ys' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'ys', list_43093)
        
        # Assigning a Num to a Name (line 314):
        
        # Assigning a Num to a Name (line 314):
        float_43094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 13), 'float')
        # Assigning a type to the variable 't0' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 't0', float_43094)
        
        # Assigning a Num to a Name (line 315):
        
        # Assigning a Num to a Name (line 315):
        float_43095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 15), 'float')
        # Assigning a type to the variable 'tend' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'tend', float_43095)
        
        # Assigning a List to a Name (line 316):
        
        # Assigning a List to a Name (line 316):
        
        # Obtaining an instance of the builtin type 'list' (line 316)
        list_43096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 316)
        # Adding element type (line 316)
        float_43097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 13), list_43096, float_43097)
        
        # Assigning a type to the variable 'y0' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'y0', list_43096)

        @norecursion
        def solout(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'solout'
            module_type_store = module_type_store.open_function_context('solout', 318, 8, False)
            
            # Passed parameters checking function
            solout.stypy_localization = localization
            solout.stypy_type_of_self = None
            solout.stypy_type_store = module_type_store
            solout.stypy_function_name = 'solout'
            solout.stypy_param_names_list = ['t', 'y']
            solout.stypy_varargs_param_name = None
            solout.stypy_kwargs_param_name = None
            solout.stypy_call_defaults = defaults
            solout.stypy_call_varargs = varargs
            solout.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'solout', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'solout', localization, ['t', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'solout(...)' code ##################

            
            # Call to append(...): (line 319)
            # Processing the call arguments (line 319)
            # Getting the type of 't' (line 319)
            t_43100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 22), 't', False)
            # Processing the call keyword arguments (line 319)
            kwargs_43101 = {}
            # Getting the type of 'ts' (line 319)
            ts_43098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'ts', False)
            # Obtaining the member 'append' of a type (line 319)
            append_43099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 12), ts_43098, 'append')
            # Calling append(args, kwargs) (line 319)
            append_call_result_43102 = invoke(stypy.reporting.localization.Localization(__file__, 319, 12), append_43099, *[t_43100], **kwargs_43101)
            
            
            # Call to append(...): (line 320)
            # Processing the call arguments (line 320)
            
            # Call to copy(...): (line 320)
            # Processing the call keyword arguments (line 320)
            kwargs_43107 = {}
            # Getting the type of 'y' (line 320)
            y_43105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 22), 'y', False)
            # Obtaining the member 'copy' of a type (line 320)
            copy_43106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 22), y_43105, 'copy')
            # Calling copy(args, kwargs) (line 320)
            copy_call_result_43108 = invoke(stypy.reporting.localization.Localization(__file__, 320, 22), copy_43106, *[], **kwargs_43107)
            
            # Processing the call keyword arguments (line 320)
            kwargs_43109 = {}
            # Getting the type of 'ys' (line 320)
            ys_43103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'ys', False)
            # Obtaining the member 'append' of a type (line 320)
            append_43104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), ys_43103, 'append')
            # Calling append(args, kwargs) (line 320)
            append_call_result_43110 = invoke(stypy.reporting.localization.Localization(__file__, 320, 12), append_43104, *[copy_call_result_43108], **kwargs_43109)
            
            
            # ################# End of 'solout(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'solout' in the type store
            # Getting the type of 'stypy_return_type' (line 318)
            stypy_return_type_43111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_43111)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'solout'
            return stypy_return_type_43111

        # Assigning a type to the variable 'solout' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'solout', solout)

        @norecursion
        def rhs(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'rhs'
            module_type_store = module_type_store.open_function_context('rhs', 322, 8, False)
            
            # Passed parameters checking function
            rhs.stypy_localization = localization
            rhs.stypy_type_of_self = None
            rhs.stypy_type_store = module_type_store
            rhs.stypy_function_name = 'rhs'
            rhs.stypy_param_names_list = ['t', 'y']
            rhs.stypy_varargs_param_name = None
            rhs.stypy_kwargs_param_name = None
            rhs.stypy_call_defaults = defaults
            rhs.stypy_call_varargs = varargs
            rhs.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'rhs', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'rhs', localization, ['t', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'rhs(...)' code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 323)
            list_43112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 323)
            # Adding element type (line 323)
            float_43113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 20), 'float')
            # Getting the type of 't' (line 323)
            t_43114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 25), 't')
            float_43115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 29), 'float')
            # Applying the binary operator '-' (line 323)
            result_sub_43116 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 25), '-', t_43114, float_43115)
            
            complex_43117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 36), 'complex')
            # Applying the binary operator '-' (line 323)
            result_sub_43118 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 34), '-', result_sub_43116, complex_43117)
            
            # Applying the binary operator 'div' (line 323)
            result_div_43119 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 20), 'div', float_43113, result_sub_43118)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 19), list_43112, result_div_43119)
            
            # Assigning a type to the variable 'stypy_return_type' (line 323)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'stypy_return_type', list_43112)
            
            # ################# End of 'rhs(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'rhs' in the type store
            # Getting the type of 'stypy_return_type' (line 322)
            stypy_return_type_43120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_43120)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'rhs'
            return stypy_return_type_43120

        # Assigning a type to the variable 'rhs' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'rhs', rhs)
        
        # Assigning a Call to a Name (line 325):
        
        # Assigning a Call to a Name (line 325):
        
        # Call to set_integrator(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'integrator' (line 325)
        integrator_43126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 45), 'integrator', False)
        # Processing the call keyword arguments (line 325)
        kwargs_43127 = {}
        
        # Call to complex_ode(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'rhs' (line 325)
        rhs_43122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 25), 'rhs', False)
        # Processing the call keyword arguments (line 325)
        kwargs_43123 = {}
        # Getting the type of 'complex_ode' (line 325)
        complex_ode_43121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 13), 'complex_ode', False)
        # Calling complex_ode(args, kwargs) (line 325)
        complex_ode_call_result_43124 = invoke(stypy.reporting.localization.Localization(__file__, 325, 13), complex_ode_43121, *[rhs_43122], **kwargs_43123)
        
        # Obtaining the member 'set_integrator' of a type (line 325)
        set_integrator_43125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 13), complex_ode_call_result_43124, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 325)
        set_integrator_call_result_43128 = invoke(stypy.reporting.localization.Localization(__file__, 325, 13), set_integrator_43125, *[integrator_43126], **kwargs_43127)
        
        # Assigning a type to the variable 'ig' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'ig', set_integrator_call_result_43128)
        
        # Call to set_solout(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'solout' (line 326)
        solout_43131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 22), 'solout', False)
        # Processing the call keyword arguments (line 326)
        kwargs_43132 = {}
        # Getting the type of 'ig' (line 326)
        ig_43129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'ig', False)
        # Obtaining the member 'set_solout' of a type (line 326)
        set_solout_43130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), ig_43129, 'set_solout')
        # Calling set_solout(args, kwargs) (line 326)
        set_solout_call_result_43133 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), set_solout_43130, *[solout_43131], **kwargs_43132)
        
        
        # Call to set_initial_value(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'y0' (line 327)
        y0_43136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 29), 'y0', False)
        # Getting the type of 't0' (line 327)
        t0_43137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 33), 't0', False)
        # Processing the call keyword arguments (line 327)
        kwargs_43138 = {}
        # Getting the type of 'ig' (line 327)
        ig_43134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'ig', False)
        # Obtaining the member 'set_initial_value' of a type (line 327)
        set_initial_value_43135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 8), ig_43134, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 327)
        set_initial_value_call_result_43139 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), set_initial_value_43135, *[y0_43136, t0_43137], **kwargs_43138)
        
        
        # Assigning a Call to a Name (line 328):
        
        # Assigning a Call to a Name (line 328):
        
        # Call to integrate(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'tend' (line 328)
        tend_43142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 27), 'tend', False)
        # Processing the call keyword arguments (line 328)
        kwargs_43143 = {}
        # Getting the type of 'ig' (line 328)
        ig_43140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 14), 'ig', False)
        # Obtaining the member 'integrate' of a type (line 328)
        integrate_43141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 14), ig_43140, 'integrate')
        # Calling integrate(args, kwargs) (line 328)
        integrate_call_result_43144 = invoke(stypy.reporting.localization.Localization(__file__, 328, 14), integrate_43141, *[tend_43142], **kwargs_43143)
        
        # Assigning a type to the variable 'ret' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'ret', integrate_call_result_43144)
        
        # Call to assert_array_equal(...): (line 329)
        # Processing the call arguments (line 329)
        
        # Obtaining the type of the subscript
        int_43146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 30), 'int')
        # Getting the type of 'ys' (line 329)
        ys_43147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 27), 'ys', False)
        # Obtaining the member '__getitem__' of a type (line 329)
        getitem___43148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 27), ys_43147, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 329)
        subscript_call_result_43149 = invoke(stypy.reporting.localization.Localization(__file__, 329, 27), getitem___43148, int_43146)
        
        # Getting the type of 'y0' (line 329)
        y0_43150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 34), 'y0', False)
        # Processing the call keyword arguments (line 329)
        kwargs_43151 = {}
        # Getting the type of 'assert_array_equal' (line 329)
        assert_array_equal_43145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 329)
        assert_array_equal_call_result_43152 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), assert_array_equal_43145, *[subscript_call_result_43149, y0_43150], **kwargs_43151)
        
        
        # Call to assert_array_equal(...): (line 330)
        # Processing the call arguments (line 330)
        
        # Obtaining the type of the subscript
        int_43154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 30), 'int')
        # Getting the type of 'ys' (line 330)
        ys_43155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 27), 'ys', False)
        # Obtaining the member '__getitem__' of a type (line 330)
        getitem___43156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 27), ys_43155, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 330)
        subscript_call_result_43157 = invoke(stypy.reporting.localization.Localization(__file__, 330, 27), getitem___43156, int_43154)
        
        # Getting the type of 'ret' (line 330)
        ret_43158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 35), 'ret', False)
        # Processing the call keyword arguments (line 330)
        kwargs_43159 = {}
        # Getting the type of 'assert_array_equal' (line 330)
        assert_array_equal_43153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 330)
        assert_array_equal_call_result_43160 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), assert_array_equal_43153, *[subscript_call_result_43157, ret_43158], **kwargs_43159)
        
        
        # Call to assert_equal(...): (line 331)
        # Processing the call arguments (line 331)
        
        # Obtaining the type of the subscript
        int_43162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 24), 'int')
        # Getting the type of 'ts' (line 331)
        ts_43163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 21), 'ts', False)
        # Obtaining the member '__getitem__' of a type (line 331)
        getitem___43164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 21), ts_43163, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 331)
        subscript_call_result_43165 = invoke(stypy.reporting.localization.Localization(__file__, 331, 21), getitem___43164, int_43162)
        
        # Getting the type of 't0' (line 331)
        t0_43166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 28), 't0', False)
        # Processing the call keyword arguments (line 331)
        kwargs_43167 = {}
        # Getting the type of 'assert_equal' (line 331)
        assert_equal_43161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 331)
        assert_equal_call_result_43168 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), assert_equal_43161, *[subscript_call_result_43165, t0_43166], **kwargs_43167)
        
        
        # Call to assert_equal(...): (line 332)
        # Processing the call arguments (line 332)
        
        # Obtaining the type of the subscript
        int_43170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 24), 'int')
        # Getting the type of 'ts' (line 332)
        ts_43171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 21), 'ts', False)
        # Obtaining the member '__getitem__' of a type (line 332)
        getitem___43172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 21), ts_43171, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 332)
        subscript_call_result_43173 = invoke(stypy.reporting.localization.Localization(__file__, 332, 21), getitem___43172, int_43170)
        
        # Getting the type of 'tend' (line 332)
        tend_43174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 29), 'tend', False)
        # Processing the call keyword arguments (line 332)
        kwargs_43175 = {}
        # Getting the type of 'assert_equal' (line 332)
        assert_equal_43169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 332)
        assert_equal_call_result_43176 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), assert_equal_43169, *[subscript_call_result_43173, tend_43174], **kwargs_43175)
        
        
        # ################# End of '_run_solout_test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_run_solout_test' in the type store
        # Getting the type of 'stypy_return_type' (line 310)
        stypy_return_type_43177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43177)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_run_solout_test'
        return stypy_return_type_43177


    @norecursion
    def test_solout(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_solout'
        module_type_store = module_type_store.open_function_context('test_solout', 334, 4, False)
        # Assigning a type to the variable 'self' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplexSolout.test_solout.__dict__.__setitem__('stypy_localization', localization)
        TestComplexSolout.test_solout.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplexSolout.test_solout.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplexSolout.test_solout.__dict__.__setitem__('stypy_function_name', 'TestComplexSolout.test_solout')
        TestComplexSolout.test_solout.__dict__.__setitem__('stypy_param_names_list', [])
        TestComplexSolout.test_solout.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplexSolout.test_solout.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplexSolout.test_solout.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplexSolout.test_solout.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplexSolout.test_solout.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplexSolout.test_solout.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplexSolout.test_solout', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_solout', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_solout(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 335)
        tuple_43178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 335)
        # Adding element type (line 335)
        str_43179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 27), 'str', 'dopri5')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 27), tuple_43178, str_43179)
        # Adding element type (line 335)
        str_43180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 37), 'str', 'dop853')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 27), tuple_43178, str_43180)
        
        # Testing the type of a for loop iterable (line 335)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 335, 8), tuple_43178)
        # Getting the type of the for loop variable (line 335)
        for_loop_var_43181 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 335, 8), tuple_43178)
        # Assigning a type to the variable 'integrator' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'integrator', for_loop_var_43181)
        # SSA begins for a for statement (line 335)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _run_solout_test(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'integrator' (line 336)
        integrator_43184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 34), 'integrator', False)
        # Processing the call keyword arguments (line 336)
        kwargs_43185 = {}
        # Getting the type of 'self' (line 336)
        self_43182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'self', False)
        # Obtaining the member '_run_solout_test' of a type (line 336)
        _run_solout_test_43183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), self_43182, '_run_solout_test')
        # Calling _run_solout_test(args, kwargs) (line 336)
        _run_solout_test_call_result_43186 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), _run_solout_test_43183, *[integrator_43184], **kwargs_43185)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_solout(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_solout' in the type store
        # Getting the type of 'stypy_return_type' (line 334)
        stypy_return_type_43187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43187)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_solout'
        return stypy_return_type_43187


    @norecursion
    def _run_solout_break_test(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_run_solout_break_test'
        module_type_store = module_type_store.open_function_context('_run_solout_break_test', 338, 4, False)
        # Assigning a type to the variable 'self' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplexSolout._run_solout_break_test.__dict__.__setitem__('stypy_localization', localization)
        TestComplexSolout._run_solout_break_test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplexSolout._run_solout_break_test.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplexSolout._run_solout_break_test.__dict__.__setitem__('stypy_function_name', 'TestComplexSolout._run_solout_break_test')
        TestComplexSolout._run_solout_break_test.__dict__.__setitem__('stypy_param_names_list', ['integrator'])
        TestComplexSolout._run_solout_break_test.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplexSolout._run_solout_break_test.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplexSolout._run_solout_break_test.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplexSolout._run_solout_break_test.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplexSolout._run_solout_break_test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplexSolout._run_solout_break_test.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplexSolout._run_solout_break_test', ['integrator'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_run_solout_break_test', localization, ['integrator'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_run_solout_break_test(...)' code ##################

        
        # Assigning a List to a Name (line 340):
        
        # Assigning a List to a Name (line 340):
        
        # Obtaining an instance of the builtin type 'list' (line 340)
        list_43188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 340)
        
        # Assigning a type to the variable 'ts' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'ts', list_43188)
        
        # Assigning a List to a Name (line 341):
        
        # Assigning a List to a Name (line 341):
        
        # Obtaining an instance of the builtin type 'list' (line 341)
        list_43189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 341)
        
        # Assigning a type to the variable 'ys' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'ys', list_43189)
        
        # Assigning a Num to a Name (line 342):
        
        # Assigning a Num to a Name (line 342):
        float_43190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 13), 'float')
        # Assigning a type to the variable 't0' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 't0', float_43190)
        
        # Assigning a Num to a Name (line 343):
        
        # Assigning a Num to a Name (line 343):
        float_43191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 15), 'float')
        # Assigning a type to the variable 'tend' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'tend', float_43191)
        
        # Assigning a List to a Name (line 344):
        
        # Assigning a List to a Name (line 344):
        
        # Obtaining an instance of the builtin type 'list' (line 344)
        list_43192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 344)
        # Adding element type (line 344)
        float_43193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 13), list_43192, float_43193)
        
        # Assigning a type to the variable 'y0' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'y0', list_43192)

        @norecursion
        def solout(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'solout'
            module_type_store = module_type_store.open_function_context('solout', 346, 8, False)
            
            # Passed parameters checking function
            solout.stypy_localization = localization
            solout.stypy_type_of_self = None
            solout.stypy_type_store = module_type_store
            solout.stypy_function_name = 'solout'
            solout.stypy_param_names_list = ['t', 'y']
            solout.stypy_varargs_param_name = None
            solout.stypy_kwargs_param_name = None
            solout.stypy_call_defaults = defaults
            solout.stypy_call_varargs = varargs
            solout.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'solout', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'solout', localization, ['t', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'solout(...)' code ##################

            
            # Call to append(...): (line 347)
            # Processing the call arguments (line 347)
            # Getting the type of 't' (line 347)
            t_43196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 22), 't', False)
            # Processing the call keyword arguments (line 347)
            kwargs_43197 = {}
            # Getting the type of 'ts' (line 347)
            ts_43194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'ts', False)
            # Obtaining the member 'append' of a type (line 347)
            append_43195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 12), ts_43194, 'append')
            # Calling append(args, kwargs) (line 347)
            append_call_result_43198 = invoke(stypy.reporting.localization.Localization(__file__, 347, 12), append_43195, *[t_43196], **kwargs_43197)
            
            
            # Call to append(...): (line 348)
            # Processing the call arguments (line 348)
            
            # Call to copy(...): (line 348)
            # Processing the call keyword arguments (line 348)
            kwargs_43203 = {}
            # Getting the type of 'y' (line 348)
            y_43201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 22), 'y', False)
            # Obtaining the member 'copy' of a type (line 348)
            copy_43202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 22), y_43201, 'copy')
            # Calling copy(args, kwargs) (line 348)
            copy_call_result_43204 = invoke(stypy.reporting.localization.Localization(__file__, 348, 22), copy_43202, *[], **kwargs_43203)
            
            # Processing the call keyword arguments (line 348)
            kwargs_43205 = {}
            # Getting the type of 'ys' (line 348)
            ys_43199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'ys', False)
            # Obtaining the member 'append' of a type (line 348)
            append_43200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 12), ys_43199, 'append')
            # Calling append(args, kwargs) (line 348)
            append_call_result_43206 = invoke(stypy.reporting.localization.Localization(__file__, 348, 12), append_43200, *[copy_call_result_43204], **kwargs_43205)
            
            
            
            # Getting the type of 't' (line 349)
            t_43207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 't')
            # Getting the type of 'tend' (line 349)
            tend_43208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 19), 'tend')
            float_43209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 24), 'float')
            # Applying the binary operator 'div' (line 349)
            result_div_43210 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 19), 'div', tend_43208, float_43209)
            
            # Applying the binary operator '>' (line 349)
            result_gt_43211 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 15), '>', t_43207, result_div_43210)
            
            # Testing the type of an if condition (line 349)
            if_condition_43212 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 349, 12), result_gt_43211)
            # Assigning a type to the variable 'if_condition_43212' (line 349)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'if_condition_43212', if_condition_43212)
            # SSA begins for if statement (line 349)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_43213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 23), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 350)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'stypy_return_type', int_43213)
            # SSA join for if statement (line 349)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'solout(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'solout' in the type store
            # Getting the type of 'stypy_return_type' (line 346)
            stypy_return_type_43214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_43214)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'solout'
            return stypy_return_type_43214

        # Assigning a type to the variable 'solout' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'solout', solout)

        @norecursion
        def rhs(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'rhs'
            module_type_store = module_type_store.open_function_context('rhs', 352, 8, False)
            
            # Passed parameters checking function
            rhs.stypy_localization = localization
            rhs.stypy_type_of_self = None
            rhs.stypy_type_store = module_type_store
            rhs.stypy_function_name = 'rhs'
            rhs.stypy_param_names_list = ['t', 'y']
            rhs.stypy_varargs_param_name = None
            rhs.stypy_kwargs_param_name = None
            rhs.stypy_call_defaults = defaults
            rhs.stypy_call_varargs = varargs
            rhs.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'rhs', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'rhs', localization, ['t', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'rhs(...)' code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 353)
            list_43215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 353)
            # Adding element type (line 353)
            float_43216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 20), 'float')
            # Getting the type of 't' (line 353)
            t_43217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 25), 't')
            float_43218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 29), 'float')
            # Applying the binary operator '-' (line 353)
            result_sub_43219 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 25), '-', t_43217, float_43218)
            
            complex_43220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 36), 'complex')
            # Applying the binary operator '-' (line 353)
            result_sub_43221 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 34), '-', result_sub_43219, complex_43220)
            
            # Applying the binary operator 'div' (line 353)
            result_div_43222 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 20), 'div', float_43216, result_sub_43221)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 19), list_43215, result_div_43222)
            
            # Assigning a type to the variable 'stypy_return_type' (line 353)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'stypy_return_type', list_43215)
            
            # ################# End of 'rhs(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'rhs' in the type store
            # Getting the type of 'stypy_return_type' (line 352)
            stypy_return_type_43223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_43223)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'rhs'
            return stypy_return_type_43223

        # Assigning a type to the variable 'rhs' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'rhs', rhs)
        
        # Assigning a Call to a Name (line 355):
        
        # Assigning a Call to a Name (line 355):
        
        # Call to set_integrator(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'integrator' (line 355)
        integrator_43229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 45), 'integrator', False)
        # Processing the call keyword arguments (line 355)
        kwargs_43230 = {}
        
        # Call to complex_ode(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'rhs' (line 355)
        rhs_43225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 25), 'rhs', False)
        # Processing the call keyword arguments (line 355)
        kwargs_43226 = {}
        # Getting the type of 'complex_ode' (line 355)
        complex_ode_43224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 13), 'complex_ode', False)
        # Calling complex_ode(args, kwargs) (line 355)
        complex_ode_call_result_43227 = invoke(stypy.reporting.localization.Localization(__file__, 355, 13), complex_ode_43224, *[rhs_43225], **kwargs_43226)
        
        # Obtaining the member 'set_integrator' of a type (line 355)
        set_integrator_43228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 13), complex_ode_call_result_43227, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 355)
        set_integrator_call_result_43231 = invoke(stypy.reporting.localization.Localization(__file__, 355, 13), set_integrator_43228, *[integrator_43229], **kwargs_43230)
        
        # Assigning a type to the variable 'ig' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'ig', set_integrator_call_result_43231)
        
        # Call to set_solout(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'solout' (line 356)
        solout_43234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 22), 'solout', False)
        # Processing the call keyword arguments (line 356)
        kwargs_43235 = {}
        # Getting the type of 'ig' (line 356)
        ig_43232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'ig', False)
        # Obtaining the member 'set_solout' of a type (line 356)
        set_solout_43233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), ig_43232, 'set_solout')
        # Calling set_solout(args, kwargs) (line 356)
        set_solout_call_result_43236 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), set_solout_43233, *[solout_43234], **kwargs_43235)
        
        
        # Call to set_initial_value(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'y0' (line 357)
        y0_43239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 29), 'y0', False)
        # Getting the type of 't0' (line 357)
        t0_43240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 33), 't0', False)
        # Processing the call keyword arguments (line 357)
        kwargs_43241 = {}
        # Getting the type of 'ig' (line 357)
        ig_43237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'ig', False)
        # Obtaining the member 'set_initial_value' of a type (line 357)
        set_initial_value_43238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), ig_43237, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 357)
        set_initial_value_call_result_43242 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), set_initial_value_43238, *[y0_43239, t0_43240], **kwargs_43241)
        
        
        # Assigning a Call to a Name (line 358):
        
        # Assigning a Call to a Name (line 358):
        
        # Call to integrate(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'tend' (line 358)
        tend_43245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 27), 'tend', False)
        # Processing the call keyword arguments (line 358)
        kwargs_43246 = {}
        # Getting the type of 'ig' (line 358)
        ig_43243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 14), 'ig', False)
        # Obtaining the member 'integrate' of a type (line 358)
        integrate_43244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 14), ig_43243, 'integrate')
        # Calling integrate(args, kwargs) (line 358)
        integrate_call_result_43247 = invoke(stypy.reporting.localization.Localization(__file__, 358, 14), integrate_43244, *[tend_43245], **kwargs_43246)
        
        # Assigning a type to the variable 'ret' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'ret', integrate_call_result_43247)
        
        # Call to assert_array_equal(...): (line 359)
        # Processing the call arguments (line 359)
        
        # Obtaining the type of the subscript
        int_43249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 30), 'int')
        # Getting the type of 'ys' (line 359)
        ys_43250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 27), 'ys', False)
        # Obtaining the member '__getitem__' of a type (line 359)
        getitem___43251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 27), ys_43250, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 359)
        subscript_call_result_43252 = invoke(stypy.reporting.localization.Localization(__file__, 359, 27), getitem___43251, int_43249)
        
        # Getting the type of 'y0' (line 359)
        y0_43253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 34), 'y0', False)
        # Processing the call keyword arguments (line 359)
        kwargs_43254 = {}
        # Getting the type of 'assert_array_equal' (line 359)
        assert_array_equal_43248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 359)
        assert_array_equal_call_result_43255 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), assert_array_equal_43248, *[subscript_call_result_43252, y0_43253], **kwargs_43254)
        
        
        # Call to assert_array_equal(...): (line 360)
        # Processing the call arguments (line 360)
        
        # Obtaining the type of the subscript
        int_43257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 30), 'int')
        # Getting the type of 'ys' (line 360)
        ys_43258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 27), 'ys', False)
        # Obtaining the member '__getitem__' of a type (line 360)
        getitem___43259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 27), ys_43258, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 360)
        subscript_call_result_43260 = invoke(stypy.reporting.localization.Localization(__file__, 360, 27), getitem___43259, int_43257)
        
        # Getting the type of 'ret' (line 360)
        ret_43261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 35), 'ret', False)
        # Processing the call keyword arguments (line 360)
        kwargs_43262 = {}
        # Getting the type of 'assert_array_equal' (line 360)
        assert_array_equal_43256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 360)
        assert_array_equal_call_result_43263 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), assert_array_equal_43256, *[subscript_call_result_43260, ret_43261], **kwargs_43262)
        
        
        # Call to assert_equal(...): (line 361)
        # Processing the call arguments (line 361)
        
        # Obtaining the type of the subscript
        int_43265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 24), 'int')
        # Getting the type of 'ts' (line 361)
        ts_43266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 21), 'ts', False)
        # Obtaining the member '__getitem__' of a type (line 361)
        getitem___43267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 21), ts_43266, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 361)
        subscript_call_result_43268 = invoke(stypy.reporting.localization.Localization(__file__, 361, 21), getitem___43267, int_43265)
        
        # Getting the type of 't0' (line 361)
        t0_43269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 28), 't0', False)
        # Processing the call keyword arguments (line 361)
        kwargs_43270 = {}
        # Getting the type of 'assert_equal' (line 361)
        assert_equal_43264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 361)
        assert_equal_call_result_43271 = invoke(stypy.reporting.localization.Localization(__file__, 361, 8), assert_equal_43264, *[subscript_call_result_43268, t0_43269], **kwargs_43270)
        
        
        # Call to assert_(...): (line 362)
        # Processing the call arguments (line 362)
        
        
        # Obtaining the type of the subscript
        int_43273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 19), 'int')
        # Getting the type of 'ts' (line 362)
        ts_43274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'ts', False)
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___43275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 16), ts_43274, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_43276 = invoke(stypy.reporting.localization.Localization(__file__, 362, 16), getitem___43275, int_43273)
        
        # Getting the type of 'tend' (line 362)
        tend_43277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 25), 'tend', False)
        float_43278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 30), 'float')
        # Applying the binary operator 'div' (line 362)
        result_div_43279 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 25), 'div', tend_43277, float_43278)
        
        # Applying the binary operator '>' (line 362)
        result_gt_43280 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 16), '>', subscript_call_result_43276, result_div_43279)
        
        # Processing the call keyword arguments (line 362)
        kwargs_43281 = {}
        # Getting the type of 'assert_' (line 362)
        assert__43272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 362)
        assert__call_result_43282 = invoke(stypy.reporting.localization.Localization(__file__, 362, 8), assert__43272, *[result_gt_43280], **kwargs_43281)
        
        
        # Call to assert_(...): (line 363)
        # Processing the call arguments (line 363)
        
        
        # Obtaining the type of the subscript
        int_43284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 19), 'int')
        # Getting the type of 'ts' (line 363)
        ts_43285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'ts', False)
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___43286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 16), ts_43285, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_43287 = invoke(stypy.reporting.localization.Localization(__file__, 363, 16), getitem___43286, int_43284)
        
        # Getting the type of 'tend' (line 363)
        tend_43288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'tend', False)
        # Applying the binary operator '<' (line 363)
        result_lt_43289 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 16), '<', subscript_call_result_43287, tend_43288)
        
        # Processing the call keyword arguments (line 363)
        kwargs_43290 = {}
        # Getting the type of 'assert_' (line 363)
        assert__43283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 363)
        assert__call_result_43291 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), assert__43283, *[result_lt_43289], **kwargs_43290)
        
        
        # ################# End of '_run_solout_break_test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_run_solout_break_test' in the type store
        # Getting the type of 'stypy_return_type' (line 338)
        stypy_return_type_43292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43292)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_run_solout_break_test'
        return stypy_return_type_43292


    @norecursion
    def test_solout_break(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_solout_break'
        module_type_store = module_type_store.open_function_context('test_solout_break', 365, 4, False)
        # Assigning a type to the variable 'self' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplexSolout.test_solout_break.__dict__.__setitem__('stypy_localization', localization)
        TestComplexSolout.test_solout_break.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplexSolout.test_solout_break.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplexSolout.test_solout_break.__dict__.__setitem__('stypy_function_name', 'TestComplexSolout.test_solout_break')
        TestComplexSolout.test_solout_break.__dict__.__setitem__('stypy_param_names_list', [])
        TestComplexSolout.test_solout_break.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplexSolout.test_solout_break.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplexSolout.test_solout_break.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplexSolout.test_solout_break.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplexSolout.test_solout_break.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplexSolout.test_solout_break.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplexSolout.test_solout_break', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_solout_break', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_solout_break(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 366)
        tuple_43293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 366)
        # Adding element type (line 366)
        str_43294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 27), 'str', 'dopri5')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 27), tuple_43293, str_43294)
        # Adding element type (line 366)
        str_43295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 37), 'str', 'dop853')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 27), tuple_43293, str_43295)
        
        # Testing the type of a for loop iterable (line 366)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 366, 8), tuple_43293)
        # Getting the type of the for loop variable (line 366)
        for_loop_var_43296 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 366, 8), tuple_43293)
        # Assigning a type to the variable 'integrator' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'integrator', for_loop_var_43296)
        # SSA begins for a for statement (line 366)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _run_solout_break_test(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'integrator' (line 367)
        integrator_43299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 40), 'integrator', False)
        # Processing the call keyword arguments (line 367)
        kwargs_43300 = {}
        # Getting the type of 'self' (line 367)
        self_43297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'self', False)
        # Obtaining the member '_run_solout_break_test' of a type (line 367)
        _run_solout_break_test_43298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 12), self_43297, '_run_solout_break_test')
        # Calling _run_solout_break_test(args, kwargs) (line 367)
        _run_solout_break_test_call_result_43301 = invoke(stypy.reporting.localization.Localization(__file__, 367, 12), _run_solout_break_test_43298, *[integrator_43299], **kwargs_43300)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_solout_break(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_solout_break' in the type store
        # Getting the type of 'stypy_return_type' (line 365)
        stypy_return_type_43302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43302)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_solout_break'
        return stypy_return_type_43302


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplexSolout.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestComplexSolout' (line 308)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 0), 'TestComplexSolout', TestComplexSolout)
# Declaration of the 'ODE' class

class ODE:
    str_43303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, (-1)), 'str', '\n    ODE problem\n    ')
    
    # Assigning a Name to a Name (line 379):
    
    # Assigning a Name to a Name (line 380):
    
    # Assigning a Num to a Name (line 381):
    
    # Assigning a List to a Name (line 382):
    
    # Assigning a Name to a Name (line 384):
    
    # Assigning a Name to a Name (line 385):
    
    # Assigning a Num to a Name (line 387):
    
    # Assigning a Num to a Name (line 388):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 375, 0, False)
        # Assigning a type to the variable 'self' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODE.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ODE' (line 375)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 0), 'ODE', ODE)

# Assigning a Name to a Name (line 379):
# Getting the type of 'False' (line 379)
False_43304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'False')
# Getting the type of 'ODE'
ODE_43305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ODE')
# Setting the type of the member 'stiff' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ODE_43305, 'stiff', False_43304)

# Assigning a Name to a Name (line 380):
# Getting the type of 'False' (line 380)
False_43306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'False')
# Getting the type of 'ODE'
ODE_43307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ODE')
# Setting the type of the member 'cmplx' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ODE_43307, 'cmplx', False_43306)

# Assigning a Num to a Name (line 381):
int_43308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 13), 'int')
# Getting the type of 'ODE'
ODE_43309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ODE')
# Setting the type of the member 'stop_t' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ODE_43309, 'stop_t', int_43308)

# Assigning a List to a Name (line 382):

# Obtaining an instance of the builtin type 'list' (line 382)
list_43310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 382)

# Getting the type of 'ODE'
ODE_43311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ODE')
# Setting the type of the member 'z0' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ODE_43311, 'z0', list_43310)

# Assigning a Name to a Name (line 384):
# Getting the type of 'None' (line 384)
None_43312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'None')
# Getting the type of 'ODE'
ODE_43313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ODE')
# Setting the type of the member 'lband' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ODE_43313, 'lband', None_43312)

# Assigning a Name to a Name (line 385):
# Getting the type of 'None' (line 385)
None_43314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'None')
# Getting the type of 'ODE'
ODE_43315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ODE')
# Setting the type of the member 'uband' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ODE_43315, 'uband', None_43314)

# Assigning a Num to a Name (line 387):
float_43316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 11), 'float')
# Getting the type of 'ODE'
ODE_43317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ODE')
# Setting the type of the member 'atol' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ODE_43317, 'atol', float_43316)

# Assigning a Num to a Name (line 388):
float_43318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 11), 'float')
# Getting the type of 'ODE'
ODE_43319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ODE')
# Setting the type of the member 'rtol' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ODE_43319, 'rtol', float_43318)
# Declaration of the 'SimpleOscillator' class
# Getting the type of 'ODE' (line 391)
ODE_43320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 23), 'ODE')

class SimpleOscillator(ODE_43320, ):
    str_43321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, (-1)), 'str', '\n    Free vibration of a simple oscillator::\n        m \\ddot{u} + k u = 0, u(0) = u_0 \\dot{u}(0) \\dot{u}_0\n    Solution::\n        u(t) = u_0*cos(sqrt(k/m)*t)+\\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)\n    ')
    
    # Assigning a BinOp to a Name (line 398):
    
    # Assigning a Call to a Name (line 399):
    
    # Assigning a Num to a Name (line 401):
    
    # Assigning a Num to a Name (line 402):

    @norecursion
    def f(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 404, 4, False)
        # Assigning a type to the variable 'self' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SimpleOscillator.f.__dict__.__setitem__('stypy_localization', localization)
        SimpleOscillator.f.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SimpleOscillator.f.__dict__.__setitem__('stypy_type_store', module_type_store)
        SimpleOscillator.f.__dict__.__setitem__('stypy_function_name', 'SimpleOscillator.f')
        SimpleOscillator.f.__dict__.__setitem__('stypy_param_names_list', ['z', 't'])
        SimpleOscillator.f.__dict__.__setitem__('stypy_varargs_param_name', None)
        SimpleOscillator.f.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SimpleOscillator.f.__dict__.__setitem__('stypy_call_defaults', defaults)
        SimpleOscillator.f.__dict__.__setitem__('stypy_call_varargs', varargs)
        SimpleOscillator.f.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SimpleOscillator.f.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SimpleOscillator.f', ['z', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['z', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Assigning a Call to a Name (line 405):
        
        # Assigning a Call to a Name (line 405):
        
        # Call to zeros(...): (line 405)
        # Processing the call arguments (line 405)
        
        # Obtaining an instance of the builtin type 'tuple' (line 405)
        tuple_43323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 405)
        # Adding element type (line 405)
        int_43324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 21), tuple_43323, int_43324)
        # Adding element type (line 405)
        int_43325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 21), tuple_43323, int_43325)
        
        # Getting the type of 'float' (line 405)
        float_43326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 28), 'float', False)
        # Processing the call keyword arguments (line 405)
        kwargs_43327 = {}
        # Getting the type of 'zeros' (line 405)
        zeros_43322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 14), 'zeros', False)
        # Calling zeros(args, kwargs) (line 405)
        zeros_call_result_43328 = invoke(stypy.reporting.localization.Localization(__file__, 405, 14), zeros_43322, *[tuple_43323, float_43326], **kwargs_43327)
        
        # Assigning a type to the variable 'tmp' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tmp', zeros_call_result_43328)
        
        # Assigning a Num to a Subscript (line 406):
        
        # Assigning a Num to a Subscript (line 406):
        float_43329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 20), 'float')
        # Getting the type of 'tmp' (line 406)
        tmp_43330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'tmp')
        
        # Obtaining an instance of the builtin type 'tuple' (line 406)
        tuple_43331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 406)
        # Adding element type (line 406)
        int_43332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 12), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 12), tuple_43331, int_43332)
        # Adding element type (line 406)
        int_43333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 12), tuple_43331, int_43333)
        
        # Storing an element on a container (line 406)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 8), tmp_43330, (tuple_43331, float_43329))
        
        # Assigning a BinOp to a Subscript (line 407):
        
        # Assigning a BinOp to a Subscript (line 407):
        
        # Getting the type of 'self' (line 407)
        self_43334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 21), 'self')
        # Obtaining the member 'k' of a type (line 407)
        k_43335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 21), self_43334, 'k')
        # Applying the 'usub' unary operator (line 407)
        result___neg___43336 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 20), 'usub', k_43335)
        
        # Getting the type of 'self' (line 407)
        self_43337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 30), 'self')
        # Obtaining the member 'm' of a type (line 407)
        m_43338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 30), self_43337, 'm')
        # Applying the binary operator 'div' (line 407)
        result_div_43339 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 20), 'div', result___neg___43336, m_43338)
        
        # Getting the type of 'tmp' (line 407)
        tmp_43340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'tmp')
        
        # Obtaining an instance of the builtin type 'tuple' (line 407)
        tuple_43341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 407)
        # Adding element type (line 407)
        int_43342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 12), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), tuple_43341, int_43342)
        # Adding element type (line 407)
        int_43343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), tuple_43341, int_43343)
        
        # Storing an element on a container (line 407)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 8), tmp_43340, (tuple_43341, result_div_43339))
        
        # Call to dot(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'tmp' (line 408)
        tmp_43345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 19), 'tmp', False)
        # Getting the type of 'z' (line 408)
        z_43346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 24), 'z', False)
        # Processing the call keyword arguments (line 408)
        kwargs_43347 = {}
        # Getting the type of 'dot' (line 408)
        dot_43344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 15), 'dot', False)
        # Calling dot(args, kwargs) (line 408)
        dot_call_result_43348 = invoke(stypy.reporting.localization.Localization(__file__, 408, 15), dot_43344, *[tmp_43345, z_43346], **kwargs_43347)
        
        # Assigning a type to the variable 'stypy_return_type' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'stypy_return_type', dot_call_result_43348)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 404)
        stypy_return_type_43349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43349)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_43349


    @norecursion
    def verify(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'verify'
        module_type_store = module_type_store.open_function_context('verify', 410, 4, False)
        # Assigning a type to the variable 'self' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SimpleOscillator.verify.__dict__.__setitem__('stypy_localization', localization)
        SimpleOscillator.verify.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SimpleOscillator.verify.__dict__.__setitem__('stypy_type_store', module_type_store)
        SimpleOscillator.verify.__dict__.__setitem__('stypy_function_name', 'SimpleOscillator.verify')
        SimpleOscillator.verify.__dict__.__setitem__('stypy_param_names_list', ['zs', 't'])
        SimpleOscillator.verify.__dict__.__setitem__('stypy_varargs_param_name', None)
        SimpleOscillator.verify.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SimpleOscillator.verify.__dict__.__setitem__('stypy_call_defaults', defaults)
        SimpleOscillator.verify.__dict__.__setitem__('stypy_call_varargs', varargs)
        SimpleOscillator.verify.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SimpleOscillator.verify.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SimpleOscillator.verify', ['zs', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'verify', localization, ['zs', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'verify(...)' code ##################

        
        # Assigning a Call to a Name (line 411):
        
        # Assigning a Call to a Name (line 411):
        
        # Call to sqrt(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'self' (line 411)
        self_43351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 21), 'self', False)
        # Obtaining the member 'k' of a type (line 411)
        k_43352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 21), self_43351, 'k')
        # Getting the type of 'self' (line 411)
        self_43353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 30), 'self', False)
        # Obtaining the member 'm' of a type (line 411)
        m_43354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 30), self_43353, 'm')
        # Applying the binary operator 'div' (line 411)
        result_div_43355 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 21), 'div', k_43352, m_43354)
        
        # Processing the call keyword arguments (line 411)
        kwargs_43356 = {}
        # Getting the type of 'sqrt' (line 411)
        sqrt_43350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 16), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 411)
        sqrt_call_result_43357 = invoke(stypy.reporting.localization.Localization(__file__, 411, 16), sqrt_43350, *[result_div_43355], **kwargs_43356)
        
        # Assigning a type to the variable 'omega' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'omega', sqrt_call_result_43357)
        
        # Assigning a BinOp to a Name (line 412):
        
        # Assigning a BinOp to a Name (line 412):
        
        # Obtaining the type of the subscript
        int_43358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 20), 'int')
        # Getting the type of 'self' (line 412)
        self_43359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'self')
        # Obtaining the member 'z0' of a type (line 412)
        z0_43360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 12), self_43359, 'z0')
        # Obtaining the member '__getitem__' of a type (line 412)
        getitem___43361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 12), z0_43360, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 412)
        subscript_call_result_43362 = invoke(stypy.reporting.localization.Localization(__file__, 412, 12), getitem___43361, int_43358)
        
        
        # Call to cos(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'omega' (line 412)
        omega_43364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 27), 'omega', False)
        # Getting the type of 't' (line 412)
        t_43365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 33), 't', False)
        # Applying the binary operator '*' (line 412)
        result_mul_43366 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 27), '*', omega_43364, t_43365)
        
        # Processing the call keyword arguments (line 412)
        kwargs_43367 = {}
        # Getting the type of 'cos' (line 412)
        cos_43363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 23), 'cos', False)
        # Calling cos(args, kwargs) (line 412)
        cos_call_result_43368 = invoke(stypy.reporting.localization.Localization(__file__, 412, 23), cos_43363, *[result_mul_43366], **kwargs_43367)
        
        # Applying the binary operator '*' (line 412)
        result_mul_43369 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 12), '*', subscript_call_result_43362, cos_call_result_43368)
        
        
        # Obtaining the type of the subscript
        int_43370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 46), 'int')
        # Getting the type of 'self' (line 412)
        self_43371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 38), 'self')
        # Obtaining the member 'z0' of a type (line 412)
        z0_43372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 38), self_43371, 'z0')
        # Obtaining the member '__getitem__' of a type (line 412)
        getitem___43373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 38), z0_43372, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 412)
        subscript_call_result_43374 = invoke(stypy.reporting.localization.Localization(__file__, 412, 38), getitem___43373, int_43370)
        
        
        # Call to sin(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'omega' (line 412)
        omega_43376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 53), 'omega', False)
        # Getting the type of 't' (line 412)
        t_43377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 59), 't', False)
        # Applying the binary operator '*' (line 412)
        result_mul_43378 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 53), '*', omega_43376, t_43377)
        
        # Processing the call keyword arguments (line 412)
        kwargs_43379 = {}
        # Getting the type of 'sin' (line 412)
        sin_43375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 49), 'sin', False)
        # Calling sin(args, kwargs) (line 412)
        sin_call_result_43380 = invoke(stypy.reporting.localization.Localization(__file__, 412, 49), sin_43375, *[result_mul_43378], **kwargs_43379)
        
        # Applying the binary operator '*' (line 412)
        result_mul_43381 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 38), '*', subscript_call_result_43374, sin_call_result_43380)
        
        # Getting the type of 'omega' (line 412)
        omega_43382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 62), 'omega')
        # Applying the binary operator 'div' (line 412)
        result_div_43383 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 61), 'div', result_mul_43381, omega_43382)
        
        # Applying the binary operator '+' (line 412)
        result_add_43384 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 12), '+', result_mul_43369, result_div_43383)
        
        # Assigning a type to the variable 'u' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'u', result_add_43384)
        
        # Call to allclose(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'u' (line 413)
        u_43386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 24), 'u', False)
        
        # Obtaining the type of the subscript
        slice_43387 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 413, 27), None, None, None)
        int_43388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 33), 'int')
        # Getting the type of 'zs' (line 413)
        zs_43389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 27), 'zs', False)
        # Obtaining the member '__getitem__' of a type (line 413)
        getitem___43390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 27), zs_43389, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 413)
        subscript_call_result_43391 = invoke(stypy.reporting.localization.Localization(__file__, 413, 27), getitem___43390, (slice_43387, int_43388))
        
        # Processing the call keyword arguments (line 413)
        # Getting the type of 'self' (line 413)
        self_43392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 42), 'self', False)
        # Obtaining the member 'atol' of a type (line 413)
        atol_43393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 42), self_43392, 'atol')
        keyword_43394 = atol_43393
        # Getting the type of 'self' (line 413)
        self_43395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 58), 'self', False)
        # Obtaining the member 'rtol' of a type (line 413)
        rtol_43396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 58), self_43395, 'rtol')
        keyword_43397 = rtol_43396
        kwargs_43398 = {'rtol': keyword_43397, 'atol': keyword_43394}
        # Getting the type of 'allclose' (line 413)
        allclose_43385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 15), 'allclose', False)
        # Calling allclose(args, kwargs) (line 413)
        allclose_call_result_43399 = invoke(stypy.reporting.localization.Localization(__file__, 413, 15), allclose_43385, *[u_43386, subscript_call_result_43391], **kwargs_43398)
        
        # Assigning a type to the variable 'stypy_return_type' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'stypy_return_type', allclose_call_result_43399)
        
        # ################# End of 'verify(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'verify' in the type store
        # Getting the type of 'stypy_return_type' (line 410)
        stypy_return_type_43400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43400)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'verify'
        return stypy_return_type_43400


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 391, 0, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SimpleOscillator.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SimpleOscillator' (line 391)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 0), 'SimpleOscillator', SimpleOscillator)

# Assigning a BinOp to a Name (line 398):
int_43401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 13), 'int')
float_43402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 17), 'float')
# Applying the binary operator '+' (line 398)
result_add_43403 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 13), '+', int_43401, float_43402)

# Getting the type of 'SimpleOscillator'
SimpleOscillator_43404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SimpleOscillator')
# Setting the type of the member 'stop_t' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SimpleOscillator_43404, 'stop_t', result_add_43403)

# Assigning a Call to a Name (line 399):

# Call to array(...): (line 399)
# Processing the call arguments (line 399)

# Obtaining an instance of the builtin type 'list' (line 399)
list_43406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 399)
# Adding element type (line 399)
float_43407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 15), list_43406, float_43407)
# Adding element type (line 399)
float_43408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 15), list_43406, float_43408)

# Getting the type of 'float' (line 399)
float_43409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 27), 'float', False)
# Processing the call keyword arguments (line 399)
kwargs_43410 = {}
# Getting the type of 'array' (line 399)
array_43405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 9), 'array', False)
# Calling array(args, kwargs) (line 399)
array_call_result_43411 = invoke(stypy.reporting.localization.Localization(__file__, 399, 9), array_43405, *[list_43406, float_43409], **kwargs_43410)

# Getting the type of 'SimpleOscillator'
SimpleOscillator_43412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SimpleOscillator')
# Setting the type of the member 'z0' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SimpleOscillator_43412, 'z0', array_call_result_43411)

# Assigning a Num to a Name (line 401):
float_43413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 8), 'float')
# Getting the type of 'SimpleOscillator'
SimpleOscillator_43414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SimpleOscillator')
# Setting the type of the member 'k' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SimpleOscillator_43414, 'k', float_43413)

# Assigning a Num to a Name (line 402):
float_43415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 8), 'float')
# Getting the type of 'SimpleOscillator'
SimpleOscillator_43416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SimpleOscillator')
# Setting the type of the member 'm' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SimpleOscillator_43416, 'm', float_43415)
# Declaration of the 'ComplexExp' class
# Getting the type of 'ODE' (line 416)
ODE_43417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 17), 'ODE')

class ComplexExp(ODE_43417, ):
    str_43418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 4), 'str', 'The equation :lm:`\\dot u = i u`')
    
    # Assigning a BinOp to a Name (line 418):
    
    # Assigning a Call to a Name (line 419):
    
    # Assigning a Name to a Name (line 420):

    @norecursion
    def f(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 422, 4, False)
        # Assigning a type to the variable 'self' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ComplexExp.f.__dict__.__setitem__('stypy_localization', localization)
        ComplexExp.f.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ComplexExp.f.__dict__.__setitem__('stypy_type_store', module_type_store)
        ComplexExp.f.__dict__.__setitem__('stypy_function_name', 'ComplexExp.f')
        ComplexExp.f.__dict__.__setitem__('stypy_param_names_list', ['z', 't'])
        ComplexExp.f.__dict__.__setitem__('stypy_varargs_param_name', None)
        ComplexExp.f.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ComplexExp.f.__dict__.__setitem__('stypy_call_defaults', defaults)
        ComplexExp.f.__dict__.__setitem__('stypy_call_varargs', varargs)
        ComplexExp.f.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ComplexExp.f.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ComplexExp.f', ['z', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['z', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        complex_43419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 15), 'complex')
        # Getting the type of 'z' (line 423)
        z_43420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 18), 'z')
        # Applying the binary operator '*' (line 423)
        result_mul_43421 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 15), '*', complex_43419, z_43420)
        
        # Assigning a type to the variable 'stypy_return_type' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'stypy_return_type', result_mul_43421)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 422)
        stypy_return_type_43422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43422)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_43422


    @norecursion
    def jac(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac'
        module_type_store = module_type_store.open_function_context('jac', 425, 4, False)
        # Assigning a type to the variable 'self' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ComplexExp.jac.__dict__.__setitem__('stypy_localization', localization)
        ComplexExp.jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ComplexExp.jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        ComplexExp.jac.__dict__.__setitem__('stypy_function_name', 'ComplexExp.jac')
        ComplexExp.jac.__dict__.__setitem__('stypy_param_names_list', ['z', 't'])
        ComplexExp.jac.__dict__.__setitem__('stypy_varargs_param_name', None)
        ComplexExp.jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ComplexExp.jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        ComplexExp.jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        ComplexExp.jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ComplexExp.jac.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ComplexExp.jac', ['z', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac', localization, ['z', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac(...)' code ##################

        complex_43423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 15), 'complex')
        
        # Call to eye(...): (line 426)
        # Processing the call arguments (line 426)
        int_43425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 22), 'int')
        # Processing the call keyword arguments (line 426)
        kwargs_43426 = {}
        # Getting the type of 'eye' (line 426)
        eye_43424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 18), 'eye', False)
        # Calling eye(args, kwargs) (line 426)
        eye_call_result_43427 = invoke(stypy.reporting.localization.Localization(__file__, 426, 18), eye_43424, *[int_43425], **kwargs_43426)
        
        # Applying the binary operator '*' (line 426)
        result_mul_43428 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 15), '*', complex_43423, eye_call_result_43427)
        
        # Assigning a type to the variable 'stypy_return_type' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'stypy_return_type', result_mul_43428)
        
        # ################# End of 'jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac' in the type store
        # Getting the type of 'stypy_return_type' (line 425)
        stypy_return_type_43429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43429)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac'
        return stypy_return_type_43429


    @norecursion
    def verify(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'verify'
        module_type_store = module_type_store.open_function_context('verify', 428, 4, False)
        # Assigning a type to the variable 'self' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ComplexExp.verify.__dict__.__setitem__('stypy_localization', localization)
        ComplexExp.verify.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ComplexExp.verify.__dict__.__setitem__('stypy_type_store', module_type_store)
        ComplexExp.verify.__dict__.__setitem__('stypy_function_name', 'ComplexExp.verify')
        ComplexExp.verify.__dict__.__setitem__('stypy_param_names_list', ['zs', 't'])
        ComplexExp.verify.__dict__.__setitem__('stypy_varargs_param_name', None)
        ComplexExp.verify.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ComplexExp.verify.__dict__.__setitem__('stypy_call_defaults', defaults)
        ComplexExp.verify.__dict__.__setitem__('stypy_call_varargs', varargs)
        ComplexExp.verify.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ComplexExp.verify.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ComplexExp.verify', ['zs', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'verify', localization, ['zs', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'verify(...)' code ##################

        
        # Assigning a BinOp to a Name (line 429):
        
        # Assigning a BinOp to a Name (line 429):
        # Getting the type of 'self' (line 429)
        self_43430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'self')
        # Obtaining the member 'z0' of a type (line 429)
        z0_43431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 12), self_43430, 'z0')
        
        # Call to exp(...): (line 429)
        # Processing the call arguments (line 429)
        complex_43433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 26), 'complex')
        # Getting the type of 't' (line 429)
        t_43434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 29), 't', False)
        # Applying the binary operator '*' (line 429)
        result_mul_43435 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 26), '*', complex_43433, t_43434)
        
        # Processing the call keyword arguments (line 429)
        kwargs_43436 = {}
        # Getting the type of 'exp' (line 429)
        exp_43432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 22), 'exp', False)
        # Calling exp(args, kwargs) (line 429)
        exp_call_result_43437 = invoke(stypy.reporting.localization.Localization(__file__, 429, 22), exp_43432, *[result_mul_43435], **kwargs_43436)
        
        # Applying the binary operator '*' (line 429)
        result_mul_43438 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 12), '*', z0_43431, exp_call_result_43437)
        
        # Assigning a type to the variable 'u' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'u', result_mul_43438)
        
        # Call to allclose(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'u' (line 430)
        u_43440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 24), 'u', False)
        # Getting the type of 'zs' (line 430)
        zs_43441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 27), 'zs', False)
        # Processing the call keyword arguments (line 430)
        # Getting the type of 'self' (line 430)
        self_43442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 36), 'self', False)
        # Obtaining the member 'atol' of a type (line 430)
        atol_43443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 36), self_43442, 'atol')
        keyword_43444 = atol_43443
        # Getting the type of 'self' (line 430)
        self_43445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 52), 'self', False)
        # Obtaining the member 'rtol' of a type (line 430)
        rtol_43446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 52), self_43445, 'rtol')
        keyword_43447 = rtol_43446
        kwargs_43448 = {'rtol': keyword_43447, 'atol': keyword_43444}
        # Getting the type of 'allclose' (line 430)
        allclose_43439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 15), 'allclose', False)
        # Calling allclose(args, kwargs) (line 430)
        allclose_call_result_43449 = invoke(stypy.reporting.localization.Localization(__file__, 430, 15), allclose_43439, *[u_43440, zs_43441], **kwargs_43448)
        
        # Assigning a type to the variable 'stypy_return_type' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'stypy_return_type', allclose_call_result_43449)
        
        # ################# End of 'verify(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'verify' in the type store
        # Getting the type of 'stypy_return_type' (line 428)
        stypy_return_type_43450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43450)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'verify'
        return stypy_return_type_43450


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 416, 0, False)
        # Assigning a type to the variable 'self' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ComplexExp.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ComplexExp' (line 416)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 0), 'ComplexExp', ComplexExp)

# Assigning a BinOp to a Name (line 418):
float_43451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 13), 'float')
# Getting the type of 'pi' (line 418)
pi_43452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 18), 'pi')
# Applying the binary operator '*' (line 418)
result_mul_43453 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 13), '*', float_43451, pi_43452)

# Getting the type of 'ComplexExp'
ComplexExp_43454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ComplexExp')
# Setting the type of the member 'stop_t' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ComplexExp_43454, 'stop_t', result_mul_43453)

# Assigning a Call to a Name (line 419):

# Call to exp(...): (line 419)
# Processing the call arguments (line 419)

# Obtaining an instance of the builtin type 'list' (line 419)
list_43456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 419)
# Adding element type (line 419)
complex_43457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 14), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 13), list_43456, complex_43457)
# Adding element type (line 419)
complex_43458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 18), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 13), list_43456, complex_43458)
# Adding element type (line 419)
complex_43459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 22), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 13), list_43456, complex_43459)
# Adding element type (line 419)
complex_43460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 26), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 13), list_43456, complex_43460)
# Adding element type (line 419)
complex_43461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 30), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 13), list_43456, complex_43461)

# Processing the call keyword arguments (line 419)
kwargs_43462 = {}
# Getting the type of 'exp' (line 419)
exp_43455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 9), 'exp', False)
# Calling exp(args, kwargs) (line 419)
exp_call_result_43463 = invoke(stypy.reporting.localization.Localization(__file__, 419, 9), exp_43455, *[list_43456], **kwargs_43462)

# Getting the type of 'ComplexExp'
ComplexExp_43464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ComplexExp')
# Setting the type of the member 'z0' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ComplexExp_43464, 'z0', exp_call_result_43463)

# Assigning a Name to a Name (line 420):
# Getting the type of 'True' (line 420)
True_43465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'True')
# Getting the type of 'ComplexExp'
ComplexExp_43466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ComplexExp')
# Setting the type of the member 'cmplx' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ComplexExp_43466, 'cmplx', True_43465)
# Declaration of the 'Pi' class
# Getting the type of 'ODE' (line 433)
ODE_43467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 9), 'ODE')

class Pi(ODE_43467, ):
    str_43468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 4), 'str', 'Integrate 1/(t + 1j) from t=-10 to t=10')
    
    # Assigning a Num to a Name (line 435):
    
    # Assigning a List to a Name (line 436):
    
    # Assigning a Name to a Name (line 437):

    @norecursion
    def f(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 439, 4, False)
        # Assigning a type to the variable 'self' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Pi.f.__dict__.__setitem__('stypy_localization', localization)
        Pi.f.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Pi.f.__dict__.__setitem__('stypy_type_store', module_type_store)
        Pi.f.__dict__.__setitem__('stypy_function_name', 'Pi.f')
        Pi.f.__dict__.__setitem__('stypy_param_names_list', ['z', 't'])
        Pi.f.__dict__.__setitem__('stypy_varargs_param_name', None)
        Pi.f.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Pi.f.__dict__.__setitem__('stypy_call_defaults', defaults)
        Pi.f.__dict__.__setitem__('stypy_call_varargs', varargs)
        Pi.f.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Pi.f.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Pi.f', ['z', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['z', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Call to array(...): (line 440)
        # Processing the call arguments (line 440)
        
        # Obtaining an instance of the builtin type 'list' (line 440)
        list_43470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 440)
        # Adding element type (line 440)
        float_43471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 22), 'float')
        # Getting the type of 't' (line 440)
        t_43472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 26), 't', False)
        int_43473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 30), 'int')
        # Applying the binary operator '-' (line 440)
        result_sub_43474 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 26), '-', t_43472, int_43473)
        
        complex_43475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 35), 'complex')
        # Applying the binary operator '+' (line 440)
        result_add_43476 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 33), '+', result_sub_43474, complex_43475)
        
        # Applying the binary operator 'div' (line 440)
        result_div_43477 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 22), 'div', float_43471, result_add_43476)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 21), list_43470, result_div_43477)
        
        # Processing the call keyword arguments (line 440)
        kwargs_43478 = {}
        # Getting the type of 'array' (line 440)
        array_43469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'array', False)
        # Calling array(args, kwargs) (line 440)
        array_call_result_43479 = invoke(stypy.reporting.localization.Localization(__file__, 440, 15), array_43469, *[list_43470], **kwargs_43478)
        
        # Assigning a type to the variable 'stypy_return_type' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'stypy_return_type', array_call_result_43479)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 439)
        stypy_return_type_43480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43480)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_43480


    @norecursion
    def verify(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'verify'
        module_type_store = module_type_store.open_function_context('verify', 442, 4, False)
        # Assigning a type to the variable 'self' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Pi.verify.__dict__.__setitem__('stypy_localization', localization)
        Pi.verify.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Pi.verify.__dict__.__setitem__('stypy_type_store', module_type_store)
        Pi.verify.__dict__.__setitem__('stypy_function_name', 'Pi.verify')
        Pi.verify.__dict__.__setitem__('stypy_param_names_list', ['zs', 't'])
        Pi.verify.__dict__.__setitem__('stypy_varargs_param_name', None)
        Pi.verify.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Pi.verify.__dict__.__setitem__('stypy_call_defaults', defaults)
        Pi.verify.__dict__.__setitem__('stypy_call_varargs', varargs)
        Pi.verify.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Pi.verify.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Pi.verify', ['zs', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'verify', localization, ['zs', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'verify(...)' code ##################

        
        # Assigning a BinOp to a Name (line 443):
        
        # Assigning a BinOp to a Name (line 443):
        complex_43481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 12), 'complex')
        
        # Call to arctan(...): (line 443)
        # Processing the call arguments (line 443)
        int_43484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 28), 'int')
        # Processing the call keyword arguments (line 443)
        kwargs_43485 = {}
        # Getting the type of 'np' (line 443)
        np_43482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 18), 'np', False)
        # Obtaining the member 'arctan' of a type (line 443)
        arctan_43483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 18), np_43482, 'arctan')
        # Calling arctan(args, kwargs) (line 443)
        arctan_call_result_43486 = invoke(stypy.reporting.localization.Localization(__file__, 443, 18), arctan_43483, *[int_43484], **kwargs_43485)
        
        # Applying the binary operator '*' (line 443)
        result_mul_43487 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 12), '*', complex_43481, arctan_call_result_43486)
        
        # Assigning a type to the variable 'u' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'u', result_mul_43487)
        
        # Call to allclose(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of 'u' (line 444)
        u_43489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 24), 'u', False)
        
        # Obtaining the type of the subscript
        int_43490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 30), 'int')
        slice_43491 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 444, 27), None, None, None)
        # Getting the type of 'zs' (line 444)
        zs_43492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 27), 'zs', False)
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___43493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 27), zs_43492, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 444)
        subscript_call_result_43494 = invoke(stypy.reporting.localization.Localization(__file__, 444, 27), getitem___43493, (int_43490, slice_43491))
        
        # Processing the call keyword arguments (line 444)
        # Getting the type of 'self' (line 444)
        self_43495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 43), 'self', False)
        # Obtaining the member 'atol' of a type (line 444)
        atol_43496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 43), self_43495, 'atol')
        keyword_43497 = atol_43496
        # Getting the type of 'self' (line 444)
        self_43498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 59), 'self', False)
        # Obtaining the member 'rtol' of a type (line 444)
        rtol_43499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 59), self_43498, 'rtol')
        keyword_43500 = rtol_43499
        kwargs_43501 = {'rtol': keyword_43500, 'atol': keyword_43497}
        # Getting the type of 'allclose' (line 444)
        allclose_43488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 15), 'allclose', False)
        # Calling allclose(args, kwargs) (line 444)
        allclose_call_result_43502 = invoke(stypy.reporting.localization.Localization(__file__, 444, 15), allclose_43488, *[u_43489, subscript_call_result_43494], **kwargs_43501)
        
        # Assigning a type to the variable 'stypy_return_type' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'stypy_return_type', allclose_call_result_43502)
        
        # ################# End of 'verify(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'verify' in the type store
        # Getting the type of 'stypy_return_type' (line 442)
        stypy_return_type_43503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43503)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'verify'
        return stypy_return_type_43503


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 433, 0, False)
        # Assigning a type to the variable 'self' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Pi.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Pi' (line 433)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 0), 'Pi', Pi)

# Assigning a Num to a Name (line 435):
int_43504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 13), 'int')
# Getting the type of 'Pi'
Pi_43505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Pi')
# Setting the type of the member 'stop_t' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Pi_43505, 'stop_t', int_43504)

# Assigning a List to a Name (line 436):

# Obtaining an instance of the builtin type 'list' (line 436)
list_43506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 436)
# Adding element type (line 436)
int_43507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 9), list_43506, int_43507)

# Getting the type of 'Pi'
Pi_43508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Pi')
# Setting the type of the member 'z0' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Pi_43508, 'z0', list_43506)

# Assigning a Name to a Name (line 437):
# Getting the type of 'True' (line 437)
True_43509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'True')
# Getting the type of 'Pi'
Pi_43510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Pi')
# Setting the type of the member 'cmplx' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Pi_43510, 'cmplx', True_43509)
# Declaration of the 'CoupledDecay' class
# Getting the type of 'ODE' (line 447)
ODE_43511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 19), 'ODE')

class CoupledDecay(ODE_43511, ):
    str_43512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, (-1)), 'str', '\n    3 coupled decays suited for banded treatment\n    (banded mode makes it necessary when N>>3)\n    ')
    
    # Assigning a Name to a Name (line 453):
    
    # Assigning a Num to a Name (line 454):
    
    # Assigning a List to a Name (line 455):
    
    # Assigning a Num to a Name (line 456):
    
    # Assigning a Num to a Name (line 457):
    
    # Assigning a List to a Name (line 459):

    @norecursion
    def f(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 461, 4, False)
        # Assigning a type to the variable 'self' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CoupledDecay.f.__dict__.__setitem__('stypy_localization', localization)
        CoupledDecay.f.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CoupledDecay.f.__dict__.__setitem__('stypy_type_store', module_type_store)
        CoupledDecay.f.__dict__.__setitem__('stypy_function_name', 'CoupledDecay.f')
        CoupledDecay.f.__dict__.__setitem__('stypy_param_names_list', ['z', 't'])
        CoupledDecay.f.__dict__.__setitem__('stypy_varargs_param_name', None)
        CoupledDecay.f.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CoupledDecay.f.__dict__.__setitem__('stypy_call_defaults', defaults)
        CoupledDecay.f.__dict__.__setitem__('stypy_call_varargs', varargs)
        CoupledDecay.f.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CoupledDecay.f.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CoupledDecay.f', ['z', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['z', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Assigning a Attribute to a Name (line 462):
        
        # Assigning a Attribute to a Name (line 462):
        # Getting the type of 'self' (line 462)
        self_43513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 15), 'self')
        # Obtaining the member 'lmbd' of a type (line 462)
        lmbd_43514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 15), self_43513, 'lmbd')
        # Assigning a type to the variable 'lmbd' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'lmbd', lmbd_43514)
        
        # Call to array(...): (line 463)
        # Processing the call arguments (line 463)
        
        # Obtaining an instance of the builtin type 'list' (line 463)
        list_43517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 463)
        # Adding element type (line 463)
        
        
        # Obtaining the type of the subscript
        int_43518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 31), 'int')
        # Getting the type of 'lmbd' (line 463)
        lmbd_43519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 26), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 463)
        getitem___43520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 26), lmbd_43519, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 463)
        subscript_call_result_43521 = invoke(stypy.reporting.localization.Localization(__file__, 463, 26), getitem___43520, int_43518)
        
        # Applying the 'usub' unary operator (line 463)
        result___neg___43522 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 25), 'usub', subscript_call_result_43521)
        
        
        # Obtaining the type of the subscript
        int_43523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 36), 'int')
        # Getting the type of 'z' (line 463)
        z_43524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 34), 'z', False)
        # Obtaining the member '__getitem__' of a type (line 463)
        getitem___43525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 34), z_43524, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 463)
        subscript_call_result_43526 = invoke(stypy.reporting.localization.Localization(__file__, 463, 34), getitem___43525, int_43523)
        
        # Applying the binary operator '*' (line 463)
        result_mul_43527 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 25), '*', result___neg___43522, subscript_call_result_43526)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 24), list_43517, result_mul_43527)
        # Adding element type (line 463)
        
        
        # Obtaining the type of the subscript
        int_43528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 31), 'int')
        # Getting the type of 'lmbd' (line 464)
        lmbd_43529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 26), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 464)
        getitem___43530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 26), lmbd_43529, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 464)
        subscript_call_result_43531 = invoke(stypy.reporting.localization.Localization(__file__, 464, 26), getitem___43530, int_43528)
        
        # Applying the 'usub' unary operator (line 464)
        result___neg___43532 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 25), 'usub', subscript_call_result_43531)
        
        
        # Obtaining the type of the subscript
        int_43533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 36), 'int')
        # Getting the type of 'z' (line 464)
        z_43534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 34), 'z', False)
        # Obtaining the member '__getitem__' of a type (line 464)
        getitem___43535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 34), z_43534, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 464)
        subscript_call_result_43536 = invoke(stypy.reporting.localization.Localization(__file__, 464, 34), getitem___43535, int_43533)
        
        # Applying the binary operator '*' (line 464)
        result_mul_43537 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 25), '*', result___neg___43532, subscript_call_result_43536)
        
        
        # Obtaining the type of the subscript
        int_43538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 46), 'int')
        # Getting the type of 'lmbd' (line 464)
        lmbd_43539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 41), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 464)
        getitem___43540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 41), lmbd_43539, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 464)
        subscript_call_result_43541 = invoke(stypy.reporting.localization.Localization(__file__, 464, 41), getitem___43540, int_43538)
        
        
        # Obtaining the type of the subscript
        int_43542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 51), 'int')
        # Getting the type of 'z' (line 464)
        z_43543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 49), 'z', False)
        # Obtaining the member '__getitem__' of a type (line 464)
        getitem___43544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 49), z_43543, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 464)
        subscript_call_result_43545 = invoke(stypy.reporting.localization.Localization(__file__, 464, 49), getitem___43544, int_43542)
        
        # Applying the binary operator '*' (line 464)
        result_mul_43546 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 41), '*', subscript_call_result_43541, subscript_call_result_43545)
        
        # Applying the binary operator '+' (line 464)
        result_add_43547 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 25), '+', result_mul_43537, result_mul_43546)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 24), list_43517, result_add_43547)
        # Adding element type (line 463)
        
        
        # Obtaining the type of the subscript
        int_43548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 31), 'int')
        # Getting the type of 'lmbd' (line 465)
        lmbd_43549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 26), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 465)
        getitem___43550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 26), lmbd_43549, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 465)
        subscript_call_result_43551 = invoke(stypy.reporting.localization.Localization(__file__, 465, 26), getitem___43550, int_43548)
        
        # Applying the 'usub' unary operator (line 465)
        result___neg___43552 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 25), 'usub', subscript_call_result_43551)
        
        
        # Obtaining the type of the subscript
        int_43553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 36), 'int')
        # Getting the type of 'z' (line 465)
        z_43554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 34), 'z', False)
        # Obtaining the member '__getitem__' of a type (line 465)
        getitem___43555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 34), z_43554, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 465)
        subscript_call_result_43556 = invoke(stypy.reporting.localization.Localization(__file__, 465, 34), getitem___43555, int_43553)
        
        # Applying the binary operator '*' (line 465)
        result_mul_43557 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 25), '*', result___neg___43552, subscript_call_result_43556)
        
        
        # Obtaining the type of the subscript
        int_43558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 46), 'int')
        # Getting the type of 'lmbd' (line 465)
        lmbd_43559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 41), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 465)
        getitem___43560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 41), lmbd_43559, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 465)
        subscript_call_result_43561 = invoke(stypy.reporting.localization.Localization(__file__, 465, 41), getitem___43560, int_43558)
        
        
        # Obtaining the type of the subscript
        int_43562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 51), 'int')
        # Getting the type of 'z' (line 465)
        z_43563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 49), 'z', False)
        # Obtaining the member '__getitem__' of a type (line 465)
        getitem___43564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 49), z_43563, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 465)
        subscript_call_result_43565 = invoke(stypy.reporting.localization.Localization(__file__, 465, 49), getitem___43564, int_43562)
        
        # Applying the binary operator '*' (line 465)
        result_mul_43566 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 41), '*', subscript_call_result_43561, subscript_call_result_43565)
        
        # Applying the binary operator '+' (line 465)
        result_add_43567 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 25), '+', result_mul_43557, result_mul_43566)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 24), list_43517, result_add_43567)
        
        # Processing the call keyword arguments (line 463)
        kwargs_43568 = {}
        # Getting the type of 'np' (line 463)
        np_43515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 463)
        array_43516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 15), np_43515, 'array')
        # Calling array(args, kwargs) (line 463)
        array_call_result_43569 = invoke(stypy.reporting.localization.Localization(__file__, 463, 15), array_43516, *[list_43517], **kwargs_43568)
        
        # Assigning a type to the variable 'stypy_return_type' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'stypy_return_type', array_call_result_43569)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 461)
        stypy_return_type_43570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43570)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_43570


    @norecursion
    def jac(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac'
        module_type_store = module_type_store.open_function_context('jac', 467, 4, False)
        # Assigning a type to the variable 'self' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CoupledDecay.jac.__dict__.__setitem__('stypy_localization', localization)
        CoupledDecay.jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CoupledDecay.jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        CoupledDecay.jac.__dict__.__setitem__('stypy_function_name', 'CoupledDecay.jac')
        CoupledDecay.jac.__dict__.__setitem__('stypy_param_names_list', ['z', 't'])
        CoupledDecay.jac.__dict__.__setitem__('stypy_varargs_param_name', None)
        CoupledDecay.jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CoupledDecay.jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        CoupledDecay.jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        CoupledDecay.jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CoupledDecay.jac.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CoupledDecay.jac', ['z', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac', localization, ['z', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac(...)' code ##################

        
        # Assigning a Attribute to a Name (line 480):
        
        # Assigning a Attribute to a Name (line 480):
        # Getting the type of 'self' (line 480)
        self_43571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 15), 'self')
        # Obtaining the member 'lmbd' of a type (line 480)
        lmbd_43572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 15), self_43571, 'lmbd')
        # Assigning a type to the variable 'lmbd' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'lmbd', lmbd_43572)
        
        # Assigning a Call to a Name (line 481):
        
        # Assigning a Call to a Name (line 481):
        
        # Call to zeros(...): (line 481)
        # Processing the call arguments (line 481)
        
        # Obtaining an instance of the builtin type 'tuple' (line 481)
        tuple_43575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 481)
        # Adding element type (line 481)
        # Getting the type of 'self' (line 481)
        self_43576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 22), 'self', False)
        # Obtaining the member 'lband' of a type (line 481)
        lband_43577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 22), self_43576, 'lband')
        # Getting the type of 'self' (line 481)
        self_43578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 35), 'self', False)
        # Obtaining the member 'uband' of a type (line 481)
        uband_43579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 35), self_43578, 'uband')
        # Applying the binary operator '+' (line 481)
        result_add_43580 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 22), '+', lband_43577, uband_43579)
        
        int_43581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 48), 'int')
        # Applying the binary operator '+' (line 481)
        result_add_43582 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 46), '+', result_add_43580, int_43581)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 22), tuple_43575, result_add_43582)
        # Adding element type (line 481)
        int_43583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 22), tuple_43575, int_43583)
        
        # Processing the call keyword arguments (line 481)
        str_43584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 61), 'str', 'F')
        keyword_43585 = str_43584
        kwargs_43586 = {'order': keyword_43585}
        # Getting the type of 'np' (line 481)
        np_43573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 481)
        zeros_43574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 12), np_43573, 'zeros')
        # Calling zeros(args, kwargs) (line 481)
        zeros_call_result_43587 = invoke(stypy.reporting.localization.Localization(__file__, 481, 12), zeros_43574, *[tuple_43575], **kwargs_43586)
        
        # Assigning a type to the variable 'j' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'j', zeros_call_result_43587)

        @norecursion
        def set_j(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'set_j'
            module_type_store = module_type_store.open_function_context('set_j', 483, 8, False)
            
            # Passed parameters checking function
            set_j.stypy_localization = localization
            set_j.stypy_type_of_self = None
            set_j.stypy_type_store = module_type_store
            set_j.stypy_function_name = 'set_j'
            set_j.stypy_param_names_list = ['ri', 'ci', 'val']
            set_j.stypy_varargs_param_name = None
            set_j.stypy_kwargs_param_name = None
            set_j.stypy_call_defaults = defaults
            set_j.stypy_call_varargs = varargs
            set_j.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'set_j', ['ri', 'ci', 'val'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'set_j', localization, ['ri', 'ci', 'val'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'set_j(...)' code ##################

            
            # Assigning a Name to a Subscript (line 484):
            
            # Assigning a Name to a Subscript (line 484):
            # Getting the type of 'val' (line 484)
            val_43588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 42), 'val')
            # Getting the type of 'j' (line 484)
            j_43589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'j')
            
            # Obtaining an instance of the builtin type 'tuple' (line 484)
            tuple_43590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 14), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 484)
            # Adding element type (line 484)
            # Getting the type of 'self' (line 484)
            self_43591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 14), 'self')
            # Obtaining the member 'uband' of a type (line 484)
            uband_43592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 14), self_43591, 'uband')
            # Getting the type of 'ri' (line 484)
            ri_43593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 27), 'ri')
            # Applying the binary operator '+' (line 484)
            result_add_43594 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 14), '+', uband_43592, ri_43593)
            
            # Getting the type of 'ci' (line 484)
            ci_43595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 32), 'ci')
            # Applying the binary operator '-' (line 484)
            result_sub_43596 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 30), '-', result_add_43594, ci_43595)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 14), tuple_43590, result_sub_43596)
            # Adding element type (line 484)
            # Getting the type of 'ci' (line 484)
            ci_43597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 36), 'ci')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 14), tuple_43590, ci_43597)
            
            # Storing an element on a container (line 484)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), j_43589, (tuple_43590, val_43588))
            
            # ################# End of 'set_j(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'set_j' in the type store
            # Getting the type of 'stypy_return_type' (line 483)
            stypy_return_type_43598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_43598)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'set_j'
            return stypy_return_type_43598

        # Assigning a type to the variable 'set_j' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'set_j', set_j)
        
        # Call to set_j(...): (line 485)
        # Processing the call arguments (line 485)
        int_43600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 14), 'int')
        int_43601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 17), 'int')
        
        
        # Obtaining the type of the subscript
        int_43602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 26), 'int')
        # Getting the type of 'lmbd' (line 485)
        lmbd_43603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 21), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 485)
        getitem___43604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 21), lmbd_43603, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 485)
        subscript_call_result_43605 = invoke(stypy.reporting.localization.Localization(__file__, 485, 21), getitem___43604, int_43602)
        
        # Applying the 'usub' unary operator (line 485)
        result___neg___43606 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 20), 'usub', subscript_call_result_43605)
        
        # Processing the call keyword arguments (line 485)
        kwargs_43607 = {}
        # Getting the type of 'set_j' (line 485)
        set_j_43599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'set_j', False)
        # Calling set_j(args, kwargs) (line 485)
        set_j_call_result_43608 = invoke(stypy.reporting.localization.Localization(__file__, 485, 8), set_j_43599, *[int_43600, int_43601, result___neg___43606], **kwargs_43607)
        
        
        # Call to set_j(...): (line 486)
        # Processing the call arguments (line 486)
        int_43610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 14), 'int')
        int_43611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 17), 'int')
        
        # Obtaining the type of the subscript
        int_43612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 25), 'int')
        # Getting the type of 'lmbd' (line 486)
        lmbd_43613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 20), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 486)
        getitem___43614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 20), lmbd_43613, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 486)
        subscript_call_result_43615 = invoke(stypy.reporting.localization.Localization(__file__, 486, 20), getitem___43614, int_43612)
        
        # Processing the call keyword arguments (line 486)
        kwargs_43616 = {}
        # Getting the type of 'set_j' (line 486)
        set_j_43609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'set_j', False)
        # Calling set_j(args, kwargs) (line 486)
        set_j_call_result_43617 = invoke(stypy.reporting.localization.Localization(__file__, 486, 8), set_j_43609, *[int_43610, int_43611, subscript_call_result_43615], **kwargs_43616)
        
        
        # Call to set_j(...): (line 487)
        # Processing the call arguments (line 487)
        int_43619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 14), 'int')
        int_43620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 17), 'int')
        
        
        # Obtaining the type of the subscript
        int_43621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 26), 'int')
        # Getting the type of 'lmbd' (line 487)
        lmbd_43622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 21), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 487)
        getitem___43623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 21), lmbd_43622, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 487)
        subscript_call_result_43624 = invoke(stypy.reporting.localization.Localization(__file__, 487, 21), getitem___43623, int_43621)
        
        # Applying the 'usub' unary operator (line 487)
        result___neg___43625 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 20), 'usub', subscript_call_result_43624)
        
        # Processing the call keyword arguments (line 487)
        kwargs_43626 = {}
        # Getting the type of 'set_j' (line 487)
        set_j_43618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'set_j', False)
        # Calling set_j(args, kwargs) (line 487)
        set_j_call_result_43627 = invoke(stypy.reporting.localization.Localization(__file__, 487, 8), set_j_43618, *[int_43619, int_43620, result___neg___43625], **kwargs_43626)
        
        
        # Call to set_j(...): (line 488)
        # Processing the call arguments (line 488)
        int_43629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 14), 'int')
        int_43630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 17), 'int')
        
        # Obtaining the type of the subscript
        int_43631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 25), 'int')
        # Getting the type of 'lmbd' (line 488)
        lmbd_43632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 20), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 488)
        getitem___43633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 20), lmbd_43632, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 488)
        subscript_call_result_43634 = invoke(stypy.reporting.localization.Localization(__file__, 488, 20), getitem___43633, int_43631)
        
        # Processing the call keyword arguments (line 488)
        kwargs_43635 = {}
        # Getting the type of 'set_j' (line 488)
        set_j_43628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'set_j', False)
        # Calling set_j(args, kwargs) (line 488)
        set_j_call_result_43636 = invoke(stypy.reporting.localization.Localization(__file__, 488, 8), set_j_43628, *[int_43629, int_43630, subscript_call_result_43634], **kwargs_43635)
        
        
        # Call to set_j(...): (line 489)
        # Processing the call arguments (line 489)
        int_43638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 14), 'int')
        int_43639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 17), 'int')
        
        
        # Obtaining the type of the subscript
        int_43640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 26), 'int')
        # Getting the type of 'lmbd' (line 489)
        lmbd_43641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 21), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 489)
        getitem___43642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 21), lmbd_43641, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 489)
        subscript_call_result_43643 = invoke(stypy.reporting.localization.Localization(__file__, 489, 21), getitem___43642, int_43640)
        
        # Applying the 'usub' unary operator (line 489)
        result___neg___43644 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 20), 'usub', subscript_call_result_43643)
        
        # Processing the call keyword arguments (line 489)
        kwargs_43645 = {}
        # Getting the type of 'set_j' (line 489)
        set_j_43637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'set_j', False)
        # Calling set_j(args, kwargs) (line 489)
        set_j_call_result_43646 = invoke(stypy.reporting.localization.Localization(__file__, 489, 8), set_j_43637, *[int_43638, int_43639, result___neg___43644], **kwargs_43645)
        
        # Getting the type of 'j' (line 490)
        j_43647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 15), 'j')
        # Assigning a type to the variable 'stypy_return_type' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'stypy_return_type', j_43647)
        
        # ################# End of 'jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac' in the type store
        # Getting the type of 'stypy_return_type' (line 467)
        stypy_return_type_43648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43648)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac'
        return stypy_return_type_43648


    @norecursion
    def verify(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'verify'
        module_type_store = module_type_store.open_function_context('verify', 492, 4, False)
        # Assigning a type to the variable 'self' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CoupledDecay.verify.__dict__.__setitem__('stypy_localization', localization)
        CoupledDecay.verify.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CoupledDecay.verify.__dict__.__setitem__('stypy_type_store', module_type_store)
        CoupledDecay.verify.__dict__.__setitem__('stypy_function_name', 'CoupledDecay.verify')
        CoupledDecay.verify.__dict__.__setitem__('stypy_param_names_list', ['zs', 't'])
        CoupledDecay.verify.__dict__.__setitem__('stypy_varargs_param_name', None)
        CoupledDecay.verify.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CoupledDecay.verify.__dict__.__setitem__('stypy_call_defaults', defaults)
        CoupledDecay.verify.__dict__.__setitem__('stypy_call_varargs', varargs)
        CoupledDecay.verify.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CoupledDecay.verify.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CoupledDecay.verify', ['zs', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'verify', localization, ['zs', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'verify(...)' code ##################

        
        # Assigning a Call to a Name (line 494):
        
        # Assigning a Call to a Name (line 494):
        
        # Call to array(...): (line 494)
        # Processing the call arguments (line 494)
        # Getting the type of 'self' (line 494)
        self_43651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 24), 'self', False)
        # Obtaining the member 'lmbd' of a type (line 494)
        lmbd_43652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 24), self_43651, 'lmbd')
        # Processing the call keyword arguments (line 494)
        kwargs_43653 = {}
        # Getting the type of 'np' (line 494)
        np_43649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 494)
        array_43650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 15), np_43649, 'array')
        # Calling array(args, kwargs) (line 494)
        array_call_result_43654 = invoke(stypy.reporting.localization.Localization(__file__, 494, 15), array_43650, *[lmbd_43652], **kwargs_43653)
        
        # Assigning a type to the variable 'lmbd' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'lmbd', array_call_result_43654)
        
        # Assigning a BinOp to a Name (line 495):
        
        # Assigning a BinOp to a Name (line 495):
        
        # Obtaining the type of the subscript
        int_43655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 19), 'int')
        # Getting the type of 'lmbd' (line 495)
        lmbd_43656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 14), 'lmbd')
        # Obtaining the member '__getitem__' of a type (line 495)
        getitem___43657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 14), lmbd_43656, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 495)
        subscript_call_result_43658 = invoke(stypy.reporting.localization.Localization(__file__, 495, 14), getitem___43657, int_43655)
        
        
        # Obtaining the type of the subscript
        int_43659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 29), 'int')
        # Getting the type of 'lmbd' (line 495)
        lmbd_43660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 24), 'lmbd')
        # Obtaining the member '__getitem__' of a type (line 495)
        getitem___43661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 24), lmbd_43660, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 495)
        subscript_call_result_43662 = invoke(stypy.reporting.localization.Localization(__file__, 495, 24), getitem___43661, int_43659)
        
        # Applying the binary operator '-' (line 495)
        result_sub_43663 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 14), '-', subscript_call_result_43658, subscript_call_result_43662)
        
        # Assigning a type to the variable 'd10' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'd10', result_sub_43663)
        
        # Assigning a BinOp to a Name (line 496):
        
        # Assigning a BinOp to a Name (line 496):
        
        # Obtaining the type of the subscript
        int_43664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 19), 'int')
        # Getting the type of 'lmbd' (line 496)
        lmbd_43665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 14), 'lmbd')
        # Obtaining the member '__getitem__' of a type (line 496)
        getitem___43666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 14), lmbd_43665, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 496)
        subscript_call_result_43667 = invoke(stypy.reporting.localization.Localization(__file__, 496, 14), getitem___43666, int_43664)
        
        
        # Obtaining the type of the subscript
        int_43668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 29), 'int')
        # Getting the type of 'lmbd' (line 496)
        lmbd_43669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 24), 'lmbd')
        # Obtaining the member '__getitem__' of a type (line 496)
        getitem___43670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 24), lmbd_43669, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 496)
        subscript_call_result_43671 = invoke(stypy.reporting.localization.Localization(__file__, 496, 24), getitem___43670, int_43668)
        
        # Applying the binary operator '-' (line 496)
        result_sub_43672 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 14), '-', subscript_call_result_43667, subscript_call_result_43671)
        
        # Assigning a type to the variable 'd21' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'd21', result_sub_43672)
        
        # Assigning a BinOp to a Name (line 497):
        
        # Assigning a BinOp to a Name (line 497):
        
        # Obtaining the type of the subscript
        int_43673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 19), 'int')
        # Getting the type of 'lmbd' (line 497)
        lmbd_43674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 14), 'lmbd')
        # Obtaining the member '__getitem__' of a type (line 497)
        getitem___43675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 14), lmbd_43674, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 497)
        subscript_call_result_43676 = invoke(stypy.reporting.localization.Localization(__file__, 497, 14), getitem___43675, int_43673)
        
        
        # Obtaining the type of the subscript
        int_43677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 29), 'int')
        # Getting the type of 'lmbd' (line 497)
        lmbd_43678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 24), 'lmbd')
        # Obtaining the member '__getitem__' of a type (line 497)
        getitem___43679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 24), lmbd_43678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 497)
        subscript_call_result_43680 = invoke(stypy.reporting.localization.Localization(__file__, 497, 24), getitem___43679, int_43677)
        
        # Applying the binary operator '-' (line 497)
        result_sub_43681 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 14), '-', subscript_call_result_43676, subscript_call_result_43680)
        
        # Assigning a type to the variable 'd20' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'd20', result_sub_43681)
        
        # Assigning a Call to a Name (line 498):
        
        # Assigning a Call to a Name (line 498):
        
        # Call to exp(...): (line 498)
        # Processing the call arguments (line 498)
        
        
        # Obtaining the type of the subscript
        int_43684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 26), 'int')
        # Getting the type of 'lmbd' (line 498)
        lmbd_43685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 21), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 498)
        getitem___43686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 21), lmbd_43685, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 498)
        subscript_call_result_43687 = invoke(stypy.reporting.localization.Localization(__file__, 498, 21), getitem___43686, int_43684)
        
        # Applying the 'usub' unary operator (line 498)
        result___neg___43688 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 20), 'usub', subscript_call_result_43687)
        
        # Getting the type of 't' (line 498)
        t_43689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 31), 't', False)
        # Applying the binary operator '*' (line 498)
        result_mul_43690 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 20), '*', result___neg___43688, t_43689)
        
        # Processing the call keyword arguments (line 498)
        kwargs_43691 = {}
        # Getting the type of 'np' (line 498)
        np_43682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 13), 'np', False)
        # Obtaining the member 'exp' of a type (line 498)
        exp_43683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 13), np_43682, 'exp')
        # Calling exp(args, kwargs) (line 498)
        exp_call_result_43692 = invoke(stypy.reporting.localization.Localization(__file__, 498, 13), exp_43683, *[result_mul_43690], **kwargs_43691)
        
        # Assigning a type to the variable 'e0' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'e0', exp_call_result_43692)
        
        # Assigning a Call to a Name (line 499):
        
        # Assigning a Call to a Name (line 499):
        
        # Call to exp(...): (line 499)
        # Processing the call arguments (line 499)
        
        
        # Obtaining the type of the subscript
        int_43695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 26), 'int')
        # Getting the type of 'lmbd' (line 499)
        lmbd_43696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 21), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 499)
        getitem___43697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 21), lmbd_43696, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 499)
        subscript_call_result_43698 = invoke(stypy.reporting.localization.Localization(__file__, 499, 21), getitem___43697, int_43695)
        
        # Applying the 'usub' unary operator (line 499)
        result___neg___43699 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 20), 'usub', subscript_call_result_43698)
        
        # Getting the type of 't' (line 499)
        t_43700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 31), 't', False)
        # Applying the binary operator '*' (line 499)
        result_mul_43701 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 20), '*', result___neg___43699, t_43700)
        
        # Processing the call keyword arguments (line 499)
        kwargs_43702 = {}
        # Getting the type of 'np' (line 499)
        np_43693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 13), 'np', False)
        # Obtaining the member 'exp' of a type (line 499)
        exp_43694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 13), np_43693, 'exp')
        # Calling exp(args, kwargs) (line 499)
        exp_call_result_43703 = invoke(stypy.reporting.localization.Localization(__file__, 499, 13), exp_43694, *[result_mul_43701], **kwargs_43702)
        
        # Assigning a type to the variable 'e1' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'e1', exp_call_result_43703)
        
        # Assigning a Call to a Name (line 500):
        
        # Assigning a Call to a Name (line 500):
        
        # Call to exp(...): (line 500)
        # Processing the call arguments (line 500)
        
        
        # Obtaining the type of the subscript
        int_43706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 26), 'int')
        # Getting the type of 'lmbd' (line 500)
        lmbd_43707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 21), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 500)
        getitem___43708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 21), lmbd_43707, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 500)
        subscript_call_result_43709 = invoke(stypy.reporting.localization.Localization(__file__, 500, 21), getitem___43708, int_43706)
        
        # Applying the 'usub' unary operator (line 500)
        result___neg___43710 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 20), 'usub', subscript_call_result_43709)
        
        # Getting the type of 't' (line 500)
        t_43711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 31), 't', False)
        # Applying the binary operator '*' (line 500)
        result_mul_43712 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 20), '*', result___neg___43710, t_43711)
        
        # Processing the call keyword arguments (line 500)
        kwargs_43713 = {}
        # Getting the type of 'np' (line 500)
        np_43704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 13), 'np', False)
        # Obtaining the member 'exp' of a type (line 500)
        exp_43705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 13), np_43704, 'exp')
        # Calling exp(args, kwargs) (line 500)
        exp_call_result_43714 = invoke(stypy.reporting.localization.Localization(__file__, 500, 13), exp_43705, *[result_mul_43712], **kwargs_43713)
        
        # Assigning a type to the variable 'e2' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'e2', exp_call_result_43714)
        
        # Assigning a Call to a Name (line 501):
        
        # Assigning a Call to a Name (line 501):
        
        # Call to transpose(...): (line 501)
        # Processing the call keyword arguments (line 501)
        kwargs_43810 = {}
        
        # Call to vstack(...): (line 501)
        # Processing the call arguments (line 501)
        
        # Obtaining an instance of the builtin type 'tuple' (line 502)
        tuple_43717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 502)
        # Adding element type (line 502)
        
        # Obtaining the type of the subscript
        int_43718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 20), 'int')
        # Getting the type of 'self' (line 502)
        self_43719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'self', False)
        # Obtaining the member 'z0' of a type (line 502)
        z0_43720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 12), self_43719, 'z0')
        # Obtaining the member '__getitem__' of a type (line 502)
        getitem___43721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 12), z0_43720, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 502)
        subscript_call_result_43722 = invoke(stypy.reporting.localization.Localization(__file__, 502, 12), getitem___43721, int_43718)
        
        # Getting the type of 'e0' (line 502)
        e0_43723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 25), 'e0', False)
        # Applying the binary operator '*' (line 502)
        result_mul_43724 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 12), '*', subscript_call_result_43722, e0_43723)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 12), tuple_43717, result_mul_43724)
        # Adding element type (line 502)
        
        # Obtaining the type of the subscript
        int_43725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 20), 'int')
        # Getting the type of 'self' (line 503)
        self_43726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'self', False)
        # Obtaining the member 'z0' of a type (line 503)
        z0_43727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 12), self_43726, 'z0')
        # Obtaining the member '__getitem__' of a type (line 503)
        getitem___43728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 12), z0_43727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 503)
        subscript_call_result_43729 = invoke(stypy.reporting.localization.Localization(__file__, 503, 12), getitem___43728, int_43725)
        
        # Getting the type of 'e1' (line 503)
        e1_43730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 25), 'e1', False)
        # Applying the binary operator '*' (line 503)
        result_mul_43731 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 12), '*', subscript_call_result_43729, e1_43730)
        
        
        # Obtaining the type of the subscript
        int_43732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 38), 'int')
        # Getting the type of 'self' (line 503)
        self_43733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 30), 'self', False)
        # Obtaining the member 'z0' of a type (line 503)
        z0_43734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 30), self_43733, 'z0')
        # Obtaining the member '__getitem__' of a type (line 503)
        getitem___43735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 30), z0_43734, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 503)
        subscript_call_result_43736 = invoke(stypy.reporting.localization.Localization(__file__, 503, 30), getitem___43735, int_43732)
        
        
        # Obtaining the type of the subscript
        int_43737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 48), 'int')
        # Getting the type of 'lmbd' (line 503)
        lmbd_43738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 43), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 503)
        getitem___43739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 43), lmbd_43738, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 503)
        subscript_call_result_43740 = invoke(stypy.reporting.localization.Localization(__file__, 503, 43), getitem___43739, int_43737)
        
        # Applying the binary operator '*' (line 503)
        result_mul_43741 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 30), '*', subscript_call_result_43736, subscript_call_result_43740)
        
        # Getting the type of 'd10' (line 503)
        d10_43742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 53), 'd10', False)
        # Applying the binary operator 'div' (line 503)
        result_div_43743 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 51), 'div', result_mul_43741, d10_43742)
        
        # Getting the type of 'e0' (line 503)
        e0_43744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 60), 'e0', False)
        # Getting the type of 'e1' (line 503)
        e1_43745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 65), 'e1', False)
        # Applying the binary operator '-' (line 503)
        result_sub_43746 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 60), '-', e0_43744, e1_43745)
        
        # Applying the binary operator '*' (line 503)
        result_mul_43747 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 57), '*', result_div_43743, result_sub_43746)
        
        # Applying the binary operator '+' (line 503)
        result_add_43748 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 12), '+', result_mul_43731, result_mul_43747)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 12), tuple_43717, result_add_43748)
        # Adding element type (line 502)
        
        # Obtaining the type of the subscript
        int_43749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 20), 'int')
        # Getting the type of 'self' (line 504)
        self_43750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'self', False)
        # Obtaining the member 'z0' of a type (line 504)
        z0_43751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 12), self_43750, 'z0')
        # Obtaining the member '__getitem__' of a type (line 504)
        getitem___43752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 12), z0_43751, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 504)
        subscript_call_result_43753 = invoke(stypy.reporting.localization.Localization(__file__, 504, 12), getitem___43752, int_43749)
        
        # Getting the type of 'e2' (line 504)
        e2_43754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 25), 'e2', False)
        # Applying the binary operator '*' (line 504)
        result_mul_43755 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 12), '*', subscript_call_result_43753, e2_43754)
        
        
        # Obtaining the type of the subscript
        int_43756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 38), 'int')
        # Getting the type of 'self' (line 504)
        self_43757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 30), 'self', False)
        # Obtaining the member 'z0' of a type (line 504)
        z0_43758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 30), self_43757, 'z0')
        # Obtaining the member '__getitem__' of a type (line 504)
        getitem___43759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 30), z0_43758, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 504)
        subscript_call_result_43760 = invoke(stypy.reporting.localization.Localization(__file__, 504, 30), getitem___43759, int_43756)
        
        
        # Obtaining the type of the subscript
        int_43761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 48), 'int')
        # Getting the type of 'lmbd' (line 504)
        lmbd_43762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 43), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 504)
        getitem___43763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 43), lmbd_43762, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 504)
        subscript_call_result_43764 = invoke(stypy.reporting.localization.Localization(__file__, 504, 43), getitem___43763, int_43761)
        
        # Applying the binary operator '*' (line 504)
        result_mul_43765 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 30), '*', subscript_call_result_43760, subscript_call_result_43764)
        
        # Getting the type of 'd21' (line 504)
        d21_43766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 53), 'd21', False)
        # Applying the binary operator 'div' (line 504)
        result_div_43767 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 51), 'div', result_mul_43765, d21_43766)
        
        # Getting the type of 'e1' (line 504)
        e1_43768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 60), 'e1', False)
        # Getting the type of 'e2' (line 504)
        e2_43769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 65), 'e2', False)
        # Applying the binary operator '-' (line 504)
        result_sub_43770 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 60), '-', e1_43768, e2_43769)
        
        # Applying the binary operator '*' (line 504)
        result_mul_43771 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 57), '*', result_div_43767, result_sub_43770)
        
        # Applying the binary operator '+' (line 504)
        result_add_43772 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 12), '+', result_mul_43755, result_mul_43771)
        
        
        # Obtaining the type of the subscript
        int_43773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 17), 'int')
        # Getting the type of 'lmbd' (line 505)
        lmbd_43774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 505)
        getitem___43775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 12), lmbd_43774, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 505)
        subscript_call_result_43776 = invoke(stypy.reporting.localization.Localization(__file__, 505, 12), getitem___43775, int_43773)
        
        
        # Obtaining the type of the subscript
        int_43777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 27), 'int')
        # Getting the type of 'lmbd' (line 505)
        lmbd_43778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 22), 'lmbd', False)
        # Obtaining the member '__getitem__' of a type (line 505)
        getitem___43779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 22), lmbd_43778, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 505)
        subscript_call_result_43780 = invoke(stypy.reporting.localization.Localization(__file__, 505, 22), getitem___43779, int_43777)
        
        # Applying the binary operator '*' (line 505)
        result_mul_43781 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 12), '*', subscript_call_result_43776, subscript_call_result_43780)
        
        
        # Obtaining the type of the subscript
        int_43782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 40), 'int')
        # Getting the type of 'self' (line 505)
        self_43783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 32), 'self', False)
        # Obtaining the member 'z0' of a type (line 505)
        z0_43784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 32), self_43783, 'z0')
        # Obtaining the member '__getitem__' of a type (line 505)
        getitem___43785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 32), z0_43784, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 505)
        subscript_call_result_43786 = invoke(stypy.reporting.localization.Localization(__file__, 505, 32), getitem___43785, int_43782)
        
        # Applying the binary operator '*' (line 505)
        result_mul_43787 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 30), '*', result_mul_43781, subscript_call_result_43786)
        
        # Getting the type of 'd10' (line 505)
        d10_43788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 45), 'd10', False)
        # Applying the binary operator 'div' (line 505)
        result_div_43789 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 43), 'div', result_mul_43787, d10_43788)
        
        int_43790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 13), 'int')
        # Getting the type of 'd20' (line 506)
        d20_43791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 17), 'd20', False)
        # Applying the binary operator 'div' (line 506)
        result_div_43792 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 13), 'div', int_43790, d20_43791)
        
        # Getting the type of 'e0' (line 506)
        e0_43793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 24), 'e0', False)
        # Getting the type of 'e2' (line 506)
        e2_43794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 29), 'e2', False)
        # Applying the binary operator '-' (line 506)
        result_sub_43795 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 24), '-', e0_43793, e2_43794)
        
        # Applying the binary operator '*' (line 506)
        result_mul_43796 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 21), '*', result_div_43792, result_sub_43795)
        
        int_43797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 35), 'int')
        # Getting the type of 'd21' (line 506)
        d21_43798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 39), 'd21', False)
        # Applying the binary operator 'div' (line 506)
        result_div_43799 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 35), 'div', int_43797, d21_43798)
        
        # Getting the type of 'e1' (line 506)
        e1_43800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 46), 'e1', False)
        # Getting the type of 'e2' (line 506)
        e2_43801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 51), 'e2', False)
        # Applying the binary operator '-' (line 506)
        result_sub_43802 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 46), '-', e1_43800, e2_43801)
        
        # Applying the binary operator '*' (line 506)
        result_mul_43803 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 43), '*', result_div_43799, result_sub_43802)
        
        # Applying the binary operator '-' (line 506)
        result_sub_43804 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 13), '-', result_mul_43796, result_mul_43803)
        
        # Applying the binary operator '*' (line 505)
        result_mul_43805 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 49), '*', result_div_43789, result_sub_43804)
        
        # Applying the binary operator '+' (line 504)
        result_add_43806 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 69), '+', result_add_43772, result_mul_43805)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 12), tuple_43717, result_add_43806)
        
        # Processing the call keyword arguments (line 501)
        kwargs_43807 = {}
        # Getting the type of 'np' (line 501)
        np_43715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'np', False)
        # Obtaining the member 'vstack' of a type (line 501)
        vstack_43716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 12), np_43715, 'vstack')
        # Calling vstack(args, kwargs) (line 501)
        vstack_call_result_43808 = invoke(stypy.reporting.localization.Localization(__file__, 501, 12), vstack_43716, *[tuple_43717], **kwargs_43807)
        
        # Obtaining the member 'transpose' of a type (line 501)
        transpose_43809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 12), vstack_call_result_43808, 'transpose')
        # Calling transpose(args, kwargs) (line 501)
        transpose_call_result_43811 = invoke(stypy.reporting.localization.Localization(__file__, 501, 12), transpose_43809, *[], **kwargs_43810)
        
        # Assigning a type to the variable 'u' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'u', transpose_call_result_43811)
        
        # Call to allclose(...): (line 507)
        # Processing the call arguments (line 507)
        # Getting the type of 'u' (line 507)
        u_43813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 24), 'u', False)
        # Getting the type of 'zs' (line 507)
        zs_43814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 27), 'zs', False)
        # Processing the call keyword arguments (line 507)
        # Getting the type of 'self' (line 507)
        self_43815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 36), 'self', False)
        # Obtaining the member 'atol' of a type (line 507)
        atol_43816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 36), self_43815, 'atol')
        keyword_43817 = atol_43816
        # Getting the type of 'self' (line 507)
        self_43818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 52), 'self', False)
        # Obtaining the member 'rtol' of a type (line 507)
        rtol_43819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 52), self_43818, 'rtol')
        keyword_43820 = rtol_43819
        kwargs_43821 = {'rtol': keyword_43820, 'atol': keyword_43817}
        # Getting the type of 'allclose' (line 507)
        allclose_43812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 15), 'allclose', False)
        # Calling allclose(args, kwargs) (line 507)
        allclose_call_result_43822 = invoke(stypy.reporting.localization.Localization(__file__, 507, 15), allclose_43812, *[u_43813, zs_43814], **kwargs_43821)
        
        # Assigning a type to the variable 'stypy_return_type' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'stypy_return_type', allclose_call_result_43822)
        
        # ################# End of 'verify(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'verify' in the type store
        # Getting the type of 'stypy_return_type' (line 492)
        stypy_return_type_43823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43823)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'verify'
        return stypy_return_type_43823


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 447, 0, False)
        # Assigning a type to the variable 'self' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CoupledDecay.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CoupledDecay' (line 447)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 0), 'CoupledDecay', CoupledDecay)

# Assigning a Name to a Name (line 453):
# Getting the type of 'True' (line 453)
True_43824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'True')
# Getting the type of 'CoupledDecay'
CoupledDecay_43825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CoupledDecay')
# Setting the type of the member 'stiff' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CoupledDecay_43825, 'stiff', True_43824)

# Assigning a Num to a Name (line 454):
float_43826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 13), 'float')
# Getting the type of 'CoupledDecay'
CoupledDecay_43827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CoupledDecay')
# Setting the type of the member 'stop_t' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CoupledDecay_43827, 'stop_t', float_43826)

# Assigning a List to a Name (line 455):

# Obtaining an instance of the builtin type 'list' (line 455)
list_43828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 455)
# Adding element type (line 455)
float_43829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 9), list_43828, float_43829)
# Adding element type (line 455)
float_43830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 9), list_43828, float_43830)
# Adding element type (line 455)
float_43831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 9), list_43828, float_43831)

# Getting the type of 'CoupledDecay'
CoupledDecay_43832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CoupledDecay')
# Setting the type of the member 'z0' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CoupledDecay_43832, 'z0', list_43828)

# Assigning a Num to a Name (line 456):
int_43833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 12), 'int')
# Getting the type of 'CoupledDecay'
CoupledDecay_43834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CoupledDecay')
# Setting the type of the member 'lband' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CoupledDecay_43834, 'lband', int_43833)

# Assigning a Num to a Name (line 457):
int_43835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 12), 'int')
# Getting the type of 'CoupledDecay'
CoupledDecay_43836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CoupledDecay')
# Setting the type of the member 'uband' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CoupledDecay_43836, 'uband', int_43835)

# Assigning a List to a Name (line 459):

# Obtaining an instance of the builtin type 'list' (line 459)
list_43837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 459)
# Adding element type (line 459)
float_43838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 12), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 11), list_43837, float_43838)
# Adding element type (line 459)
float_43839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 11), list_43837, float_43839)
# Adding element type (line 459)
float_43840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 11), list_43837, float_43840)

# Getting the type of 'CoupledDecay'
CoupledDecay_43841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CoupledDecay')
# Setting the type of the member 'lmbd' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CoupledDecay_43841, 'lmbd', list_43837)

# Assigning a List to a Name (line 510):

# Assigning a List to a Name (line 510):

# Obtaining an instance of the builtin type 'list' (line 510)
list_43842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 510)
# Adding element type (line 510)
# Getting the type of 'SimpleOscillator' (line 510)
SimpleOscillator_43843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'SimpleOscillator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 11), list_43842, SimpleOscillator_43843)
# Adding element type (line 510)
# Getting the type of 'ComplexExp' (line 510)
ComplexExp_43844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 30), 'ComplexExp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 11), list_43842, ComplexExp_43844)
# Adding element type (line 510)
# Getting the type of 'Pi' (line 510)
Pi_43845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 42), 'Pi')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 11), list_43842, Pi_43845)
# Adding element type (line 510)
# Getting the type of 'CoupledDecay' (line 510)
CoupledDecay_43846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 46), 'CoupledDecay')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 11), list_43842, CoupledDecay_43846)

# Assigning a type to the variable 'PROBLEMS' (line 510)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 0), 'PROBLEMS', list_43842)

@norecursion
def f(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f'
    module_type_store = module_type_store.open_function_context('f', 515, 0, False)
    
    # Passed parameters checking function
    f.stypy_localization = localization
    f.stypy_type_of_self = None
    f.stypy_type_store = module_type_store
    f.stypy_function_name = 'f'
    f.stypy_param_names_list = ['t', 'x']
    f.stypy_varargs_param_name = None
    f.stypy_kwargs_param_name = None
    f.stypy_call_defaults = defaults
    f.stypy_call_varargs = varargs
    f.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f', ['t', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f', localization, ['t', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f(...)' code ##################

    
    # Assigning a List to a Name (line 516):
    
    # Assigning a List to a Name (line 516):
    
    # Obtaining an instance of the builtin type 'list' (line 516)
    list_43847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 516)
    # Adding element type (line 516)
    
    # Obtaining the type of the subscript
    int_43848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 14), 'int')
    # Getting the type of 'x' (line 516)
    x_43849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'x')
    # Obtaining the member '__getitem__' of a type (line 516)
    getitem___43850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 12), x_43849, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 516)
    subscript_call_result_43851 = invoke(stypy.reporting.localization.Localization(__file__, 516, 12), getitem___43850, int_43848)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 11), list_43847, subscript_call_result_43851)
    # Adding element type (line 516)
    
    
    # Obtaining the type of the subscript
    int_43852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 21), 'int')
    # Getting the type of 'x' (line 516)
    x_43853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 19), 'x')
    # Obtaining the member '__getitem__' of a type (line 516)
    getitem___43854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 19), x_43853, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 516)
    subscript_call_result_43855 = invoke(stypy.reporting.localization.Localization(__file__, 516, 19), getitem___43854, int_43852)
    
    # Applying the 'usub' unary operator (line 516)
    result___neg___43856 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 18), 'usub', subscript_call_result_43855)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 11), list_43847, result___neg___43856)
    
    # Assigning a type to the variable 'dxdt' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'dxdt', list_43847)
    # Getting the type of 'dxdt' (line 517)
    dxdt_43857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 11), 'dxdt')
    # Assigning a type to the variable 'stypy_return_type' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'stypy_return_type', dxdt_43857)
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 515)
    stypy_return_type_43858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_43858)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_43858

# Assigning a type to the variable 'f' (line 515)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 0), 'f', f)

@norecursion
def jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'jac'
    module_type_store = module_type_store.open_function_context('jac', 520, 0, False)
    
    # Passed parameters checking function
    jac.stypy_localization = localization
    jac.stypy_type_of_self = None
    jac.stypy_type_store = module_type_store
    jac.stypy_function_name = 'jac'
    jac.stypy_param_names_list = ['t', 'x']
    jac.stypy_varargs_param_name = None
    jac.stypy_kwargs_param_name = None
    jac.stypy_call_defaults = defaults
    jac.stypy_call_varargs = varargs
    jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'jac', ['t', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'jac', localization, ['t', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'jac(...)' code ##################

    
    # Assigning a Call to a Name (line 521):
    
    # Assigning a Call to a Name (line 521):
    
    # Call to array(...): (line 521)
    # Processing the call arguments (line 521)
    
    # Obtaining an instance of the builtin type 'list' (line 521)
    list_43860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 521)
    # Adding element type (line 521)
    
    # Obtaining an instance of the builtin type 'list' (line 521)
    list_43861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 521)
    # Adding element type (line 521)
    float_43862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 15), list_43861, float_43862)
    # Adding element type (line 521)
    float_43863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 15), list_43861, float_43863)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 14), list_43860, list_43861)
    # Adding element type (line 521)
    
    # Obtaining an instance of the builtin type 'list' (line 522)
    list_43864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 522)
    # Adding element type (line 522)
    float_43865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 15), list_43864, float_43865)
    # Adding element type (line 522)
    float_43866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 15), list_43864, float_43866)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 14), list_43860, list_43864)
    
    # Processing the call keyword arguments (line 521)
    kwargs_43867 = {}
    # Getting the type of 'array' (line 521)
    array_43859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'array', False)
    # Calling array(args, kwargs) (line 521)
    array_call_result_43868 = invoke(stypy.reporting.localization.Localization(__file__, 521, 8), array_43859, *[list_43860], **kwargs_43867)
    
    # Assigning a type to the variable 'j' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'j', array_call_result_43868)
    # Getting the type of 'j' (line 523)
    j_43869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 11), 'j')
    # Assigning a type to the variable 'stypy_return_type' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'stypy_return_type', j_43869)
    
    # ################# End of 'jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'jac' in the type store
    # Getting the type of 'stypy_return_type' (line 520)
    stypy_return_type_43870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_43870)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'jac'
    return stypy_return_type_43870

# Assigning a type to the variable 'jac' (line 520)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 0), 'jac', jac)

@norecursion
def f1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f1'
    module_type_store = module_type_store.open_function_context('f1', 526, 0, False)
    
    # Passed parameters checking function
    f1.stypy_localization = localization
    f1.stypy_type_of_self = None
    f1.stypy_type_store = module_type_store
    f1.stypy_function_name = 'f1'
    f1.stypy_param_names_list = ['t', 'x', 'omega']
    f1.stypy_varargs_param_name = None
    f1.stypy_kwargs_param_name = None
    f1.stypy_call_defaults = defaults
    f1.stypy_call_varargs = varargs
    f1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f1', ['t', 'x', 'omega'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f1', localization, ['t', 'x', 'omega'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f1(...)' code ##################

    
    # Assigning a List to a Name (line 527):
    
    # Assigning a List to a Name (line 527):
    
    # Obtaining an instance of the builtin type 'list' (line 527)
    list_43871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 527)
    # Adding element type (line 527)
    # Getting the type of 'omega' (line 527)
    omega_43872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'omega')
    
    # Obtaining the type of the subscript
    int_43873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 20), 'int')
    # Getting the type of 'x' (line 527)
    x_43874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 18), 'x')
    # Obtaining the member '__getitem__' of a type (line 527)
    getitem___43875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 18), x_43874, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 527)
    subscript_call_result_43876 = invoke(stypy.reporting.localization.Localization(__file__, 527, 18), getitem___43875, int_43873)
    
    # Applying the binary operator '*' (line 527)
    result_mul_43877 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 12), '*', omega_43872, subscript_call_result_43876)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 11), list_43871, result_mul_43877)
    # Adding element type (line 527)
    
    # Getting the type of 'omega' (line 527)
    omega_43878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 25), 'omega')
    # Applying the 'usub' unary operator (line 527)
    result___neg___43879 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 24), 'usub', omega_43878)
    
    
    # Obtaining the type of the subscript
    int_43880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 33), 'int')
    # Getting the type of 'x' (line 527)
    x_43881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 31), 'x')
    # Obtaining the member '__getitem__' of a type (line 527)
    getitem___43882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 31), x_43881, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 527)
    subscript_call_result_43883 = invoke(stypy.reporting.localization.Localization(__file__, 527, 31), getitem___43882, int_43880)
    
    # Applying the binary operator '*' (line 527)
    result_mul_43884 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 24), '*', result___neg___43879, subscript_call_result_43883)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 11), list_43871, result_mul_43884)
    
    # Assigning a type to the variable 'dxdt' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'dxdt', list_43871)
    # Getting the type of 'dxdt' (line 528)
    dxdt_43885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 11), 'dxdt')
    # Assigning a type to the variable 'stypy_return_type' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'stypy_return_type', dxdt_43885)
    
    # ################# End of 'f1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f1' in the type store
    # Getting the type of 'stypy_return_type' (line 526)
    stypy_return_type_43886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_43886)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f1'
    return stypy_return_type_43886

# Assigning a type to the variable 'f1' (line 526)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 0), 'f1', f1)

@norecursion
def jac1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'jac1'
    module_type_store = module_type_store.open_function_context('jac1', 531, 0, False)
    
    # Passed parameters checking function
    jac1.stypy_localization = localization
    jac1.stypy_type_of_self = None
    jac1.stypy_type_store = module_type_store
    jac1.stypy_function_name = 'jac1'
    jac1.stypy_param_names_list = ['t', 'x', 'omega']
    jac1.stypy_varargs_param_name = None
    jac1.stypy_kwargs_param_name = None
    jac1.stypy_call_defaults = defaults
    jac1.stypy_call_varargs = varargs
    jac1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'jac1', ['t', 'x', 'omega'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'jac1', localization, ['t', 'x', 'omega'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'jac1(...)' code ##################

    
    # Assigning a Call to a Name (line 532):
    
    # Assigning a Call to a Name (line 532):
    
    # Call to array(...): (line 532)
    # Processing the call arguments (line 532)
    
    # Obtaining an instance of the builtin type 'list' (line 532)
    list_43888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 532)
    # Adding element type (line 532)
    
    # Obtaining an instance of the builtin type 'list' (line 532)
    list_43889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 532)
    # Adding element type (line 532)
    float_43890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 15), list_43889, float_43890)
    # Adding element type (line 532)
    # Getting the type of 'omega' (line 532)
    omega_43891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 21), 'omega', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 15), list_43889, omega_43891)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 14), list_43888, list_43889)
    # Adding element type (line 532)
    
    # Obtaining an instance of the builtin type 'list' (line 533)
    list_43892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 533)
    # Adding element type (line 533)
    
    # Getting the type of 'omega' (line 533)
    omega_43893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 17), 'omega', False)
    # Applying the 'usub' unary operator (line 533)
    result___neg___43894 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 16), 'usub', omega_43893)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 15), list_43892, result___neg___43894)
    # Adding element type (line 533)
    float_43895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 15), list_43892, float_43895)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 14), list_43888, list_43892)
    
    # Processing the call keyword arguments (line 532)
    kwargs_43896 = {}
    # Getting the type of 'array' (line 532)
    array_43887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'array', False)
    # Calling array(args, kwargs) (line 532)
    array_call_result_43897 = invoke(stypy.reporting.localization.Localization(__file__, 532, 8), array_43887, *[list_43888], **kwargs_43896)
    
    # Assigning a type to the variable 'j' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'j', array_call_result_43897)
    # Getting the type of 'j' (line 534)
    j_43898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 11), 'j')
    # Assigning a type to the variable 'stypy_return_type' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'stypy_return_type', j_43898)
    
    # ################# End of 'jac1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'jac1' in the type store
    # Getting the type of 'stypy_return_type' (line 531)
    stypy_return_type_43899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_43899)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'jac1'
    return stypy_return_type_43899

# Assigning a type to the variable 'jac1' (line 531)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 0), 'jac1', jac1)

@norecursion
def f2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f2'
    module_type_store = module_type_store.open_function_context('f2', 537, 0, False)
    
    # Passed parameters checking function
    f2.stypy_localization = localization
    f2.stypy_type_of_self = None
    f2.stypy_type_store = module_type_store
    f2.stypy_function_name = 'f2'
    f2.stypy_param_names_list = ['t', 'x', 'omega1', 'omega2']
    f2.stypy_varargs_param_name = None
    f2.stypy_kwargs_param_name = None
    f2.stypy_call_defaults = defaults
    f2.stypy_call_varargs = varargs
    f2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f2', ['t', 'x', 'omega1', 'omega2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f2', localization, ['t', 'x', 'omega1', 'omega2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f2(...)' code ##################

    
    # Assigning a List to a Name (line 538):
    
    # Assigning a List to a Name (line 538):
    
    # Obtaining an instance of the builtin type 'list' (line 538)
    list_43900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 538)
    # Adding element type (line 538)
    # Getting the type of 'omega1' (line 538)
    omega1_43901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'omega1')
    
    # Obtaining the type of the subscript
    int_43902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 21), 'int')
    # Getting the type of 'x' (line 538)
    x_43903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 19), 'x')
    # Obtaining the member '__getitem__' of a type (line 538)
    getitem___43904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 19), x_43903, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 538)
    subscript_call_result_43905 = invoke(stypy.reporting.localization.Localization(__file__, 538, 19), getitem___43904, int_43902)
    
    # Applying the binary operator '*' (line 538)
    result_mul_43906 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 12), '*', omega1_43901, subscript_call_result_43905)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 11), list_43900, result_mul_43906)
    # Adding element type (line 538)
    
    # Getting the type of 'omega2' (line 538)
    omega2_43907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 26), 'omega2')
    # Applying the 'usub' unary operator (line 538)
    result___neg___43908 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 25), 'usub', omega2_43907)
    
    
    # Obtaining the type of the subscript
    int_43909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 35), 'int')
    # Getting the type of 'x' (line 538)
    x_43910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 33), 'x')
    # Obtaining the member '__getitem__' of a type (line 538)
    getitem___43911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 33), x_43910, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 538)
    subscript_call_result_43912 = invoke(stypy.reporting.localization.Localization(__file__, 538, 33), getitem___43911, int_43909)
    
    # Applying the binary operator '*' (line 538)
    result_mul_43913 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 25), '*', result___neg___43908, subscript_call_result_43912)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 11), list_43900, result_mul_43913)
    
    # Assigning a type to the variable 'dxdt' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'dxdt', list_43900)
    # Getting the type of 'dxdt' (line 539)
    dxdt_43914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 11), 'dxdt')
    # Assigning a type to the variable 'stypy_return_type' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'stypy_return_type', dxdt_43914)
    
    # ################# End of 'f2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f2' in the type store
    # Getting the type of 'stypy_return_type' (line 537)
    stypy_return_type_43915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_43915)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f2'
    return stypy_return_type_43915

# Assigning a type to the variable 'f2' (line 537)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 0), 'f2', f2)

@norecursion
def jac2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'jac2'
    module_type_store = module_type_store.open_function_context('jac2', 542, 0, False)
    
    # Passed parameters checking function
    jac2.stypy_localization = localization
    jac2.stypy_type_of_self = None
    jac2.stypy_type_store = module_type_store
    jac2.stypy_function_name = 'jac2'
    jac2.stypy_param_names_list = ['t', 'x', 'omega1', 'omega2']
    jac2.stypy_varargs_param_name = None
    jac2.stypy_kwargs_param_name = None
    jac2.stypy_call_defaults = defaults
    jac2.stypy_call_varargs = varargs
    jac2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'jac2', ['t', 'x', 'omega1', 'omega2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'jac2', localization, ['t', 'x', 'omega1', 'omega2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'jac2(...)' code ##################

    
    # Assigning a Call to a Name (line 543):
    
    # Assigning a Call to a Name (line 543):
    
    # Call to array(...): (line 543)
    # Processing the call arguments (line 543)
    
    # Obtaining an instance of the builtin type 'list' (line 543)
    list_43917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 543)
    # Adding element type (line 543)
    
    # Obtaining an instance of the builtin type 'list' (line 543)
    list_43918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 543)
    # Adding element type (line 543)
    float_43919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 15), list_43918, float_43919)
    # Adding element type (line 543)
    # Getting the type of 'omega1' (line 543)
    omega1_43920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 21), 'omega1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 15), list_43918, omega1_43920)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 14), list_43917, list_43918)
    # Adding element type (line 543)
    
    # Obtaining an instance of the builtin type 'list' (line 544)
    list_43921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 544)
    # Adding element type (line 544)
    
    # Getting the type of 'omega2' (line 544)
    omega2_43922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 17), 'omega2', False)
    # Applying the 'usub' unary operator (line 544)
    result___neg___43923 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 16), 'usub', omega2_43922)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 15), list_43921, result___neg___43923)
    # Adding element type (line 544)
    float_43924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 15), list_43921, float_43924)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 14), list_43917, list_43921)
    
    # Processing the call keyword arguments (line 543)
    kwargs_43925 = {}
    # Getting the type of 'array' (line 543)
    array_43916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'array', False)
    # Calling array(args, kwargs) (line 543)
    array_call_result_43926 = invoke(stypy.reporting.localization.Localization(__file__, 543, 8), array_43916, *[list_43917], **kwargs_43925)
    
    # Assigning a type to the variable 'j' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'j', array_call_result_43926)
    # Getting the type of 'j' (line 545)
    j_43927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 11), 'j')
    # Assigning a type to the variable 'stypy_return_type' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'stypy_return_type', j_43927)
    
    # ################# End of 'jac2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'jac2' in the type store
    # Getting the type of 'stypy_return_type' (line 542)
    stypy_return_type_43928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_43928)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'jac2'
    return stypy_return_type_43928

# Assigning a type to the variable 'jac2' (line 542)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 0), 'jac2', jac2)

@norecursion
def fv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fv'
    module_type_store = module_type_store.open_function_context('fv', 548, 0, False)
    
    # Passed parameters checking function
    fv.stypy_localization = localization
    fv.stypy_type_of_self = None
    fv.stypy_type_store = module_type_store
    fv.stypy_function_name = 'fv'
    fv.stypy_param_names_list = ['t', 'x', 'omega']
    fv.stypy_varargs_param_name = None
    fv.stypy_kwargs_param_name = None
    fv.stypy_call_defaults = defaults
    fv.stypy_call_varargs = varargs
    fv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fv', ['t', 'x', 'omega'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fv', localization, ['t', 'x', 'omega'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fv(...)' code ##################

    
    # Assigning a List to a Name (line 549):
    
    # Assigning a List to a Name (line 549):
    
    # Obtaining an instance of the builtin type 'list' (line 549)
    list_43929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 549)
    # Adding element type (line 549)
    
    # Obtaining the type of the subscript
    int_43930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 18), 'int')
    # Getting the type of 'omega' (line 549)
    omega_43931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'omega')
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___43932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 12), omega_43931, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_43933 = invoke(stypy.reporting.localization.Localization(__file__, 549, 12), getitem___43932, int_43930)
    
    
    # Obtaining the type of the subscript
    int_43934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 23), 'int')
    # Getting the type of 'x' (line 549)
    x_43935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 21), 'x')
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___43936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 21), x_43935, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_43937 = invoke(stypy.reporting.localization.Localization(__file__, 549, 21), getitem___43936, int_43934)
    
    # Applying the binary operator '*' (line 549)
    result_mul_43938 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 12), '*', subscript_call_result_43933, subscript_call_result_43937)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 11), list_43929, result_mul_43938)
    # Adding element type (line 549)
    
    
    # Obtaining the type of the subscript
    int_43939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 34), 'int')
    # Getting the type of 'omega' (line 549)
    omega_43940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 28), 'omega')
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___43941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 28), omega_43940, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_43942 = invoke(stypy.reporting.localization.Localization(__file__, 549, 28), getitem___43941, int_43939)
    
    # Applying the 'usub' unary operator (line 549)
    result___neg___43943 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 27), 'usub', subscript_call_result_43942)
    
    
    # Obtaining the type of the subscript
    int_43944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 39), 'int')
    # Getting the type of 'x' (line 549)
    x_43945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 37), 'x')
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___43946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 37), x_43945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_43947 = invoke(stypy.reporting.localization.Localization(__file__, 549, 37), getitem___43946, int_43944)
    
    # Applying the binary operator '*' (line 549)
    result_mul_43948 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 27), '*', result___neg___43943, subscript_call_result_43947)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 11), list_43929, result_mul_43948)
    
    # Assigning a type to the variable 'dxdt' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'dxdt', list_43929)
    # Getting the type of 'dxdt' (line 550)
    dxdt_43949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 11), 'dxdt')
    # Assigning a type to the variable 'stypy_return_type' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'stypy_return_type', dxdt_43949)
    
    # ################# End of 'fv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fv' in the type store
    # Getting the type of 'stypy_return_type' (line 548)
    stypy_return_type_43950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_43950)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fv'
    return stypy_return_type_43950

# Assigning a type to the variable 'fv' (line 548)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 0), 'fv', fv)

@norecursion
def jacv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'jacv'
    module_type_store = module_type_store.open_function_context('jacv', 553, 0, False)
    
    # Passed parameters checking function
    jacv.stypy_localization = localization
    jacv.stypy_type_of_self = None
    jacv.stypy_type_store = module_type_store
    jacv.stypy_function_name = 'jacv'
    jacv.stypy_param_names_list = ['t', 'x', 'omega']
    jacv.stypy_varargs_param_name = None
    jacv.stypy_kwargs_param_name = None
    jacv.stypy_call_defaults = defaults
    jacv.stypy_call_varargs = varargs
    jacv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'jacv', ['t', 'x', 'omega'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'jacv', localization, ['t', 'x', 'omega'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'jacv(...)' code ##################

    
    # Assigning a Call to a Name (line 554):
    
    # Assigning a Call to a Name (line 554):
    
    # Call to array(...): (line 554)
    # Processing the call arguments (line 554)
    
    # Obtaining an instance of the builtin type 'list' (line 554)
    list_43952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 554)
    # Adding element type (line 554)
    
    # Obtaining an instance of the builtin type 'list' (line 554)
    list_43953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 554)
    # Adding element type (line 554)
    float_43954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 15), list_43953, float_43954)
    # Adding element type (line 554)
    
    # Obtaining the type of the subscript
    int_43955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 27), 'int')
    # Getting the type of 'omega' (line 554)
    omega_43956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 21), 'omega', False)
    # Obtaining the member '__getitem__' of a type (line 554)
    getitem___43957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 21), omega_43956, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 554)
    subscript_call_result_43958 = invoke(stypy.reporting.localization.Localization(__file__, 554, 21), getitem___43957, int_43955)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 15), list_43953, subscript_call_result_43958)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 14), list_43952, list_43953)
    # Adding element type (line 554)
    
    # Obtaining an instance of the builtin type 'list' (line 555)
    list_43959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 555)
    # Adding element type (line 555)
    
    
    # Obtaining the type of the subscript
    int_43960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 23), 'int')
    # Getting the type of 'omega' (line 555)
    omega_43961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 17), 'omega', False)
    # Obtaining the member '__getitem__' of a type (line 555)
    getitem___43962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 17), omega_43961, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 555)
    subscript_call_result_43963 = invoke(stypy.reporting.localization.Localization(__file__, 555, 17), getitem___43962, int_43960)
    
    # Applying the 'usub' unary operator (line 555)
    result___neg___43964 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 16), 'usub', subscript_call_result_43963)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 555, 15), list_43959, result___neg___43964)
    # Adding element type (line 555)
    float_43965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 27), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 555, 15), list_43959, float_43965)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 14), list_43952, list_43959)
    
    # Processing the call keyword arguments (line 554)
    kwargs_43966 = {}
    # Getting the type of 'array' (line 554)
    array_43951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'array', False)
    # Calling array(args, kwargs) (line 554)
    array_call_result_43967 = invoke(stypy.reporting.localization.Localization(__file__, 554, 8), array_43951, *[list_43952], **kwargs_43966)
    
    # Assigning a type to the variable 'j' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'j', array_call_result_43967)
    # Getting the type of 'j' (line 556)
    j_43968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 11), 'j')
    # Assigning a type to the variable 'stypy_return_type' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'stypy_return_type', j_43968)
    
    # ################# End of 'jacv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'jacv' in the type store
    # Getting the type of 'stypy_return_type' (line 553)
    stypy_return_type_43969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_43969)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'jacv'
    return stypy_return_type_43969

# Assigning a type to the variable 'jacv' (line 553)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 0), 'jacv', jacv)
# Declaration of the 'ODECheckParameterUse' class

class ODECheckParameterUse(object, ):
    str_43970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 4), 'str', 'Call an ode-class solver with several cases of parameter use.')
    
    # Assigning a Str to a Name (line 565):
    
    # Assigning a Name to a Name (line 566):

    @norecursion
    def _get_solver(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_solver'
        module_type_store = module_type_store.open_function_context('_get_solver', 568, 4, False)
        # Assigning a type to the variable 'self' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ODECheckParameterUse._get_solver.__dict__.__setitem__('stypy_localization', localization)
        ODECheckParameterUse._get_solver.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ODECheckParameterUse._get_solver.__dict__.__setitem__('stypy_type_store', module_type_store)
        ODECheckParameterUse._get_solver.__dict__.__setitem__('stypy_function_name', 'ODECheckParameterUse._get_solver')
        ODECheckParameterUse._get_solver.__dict__.__setitem__('stypy_param_names_list', ['f', 'jac'])
        ODECheckParameterUse._get_solver.__dict__.__setitem__('stypy_varargs_param_name', None)
        ODECheckParameterUse._get_solver.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ODECheckParameterUse._get_solver.__dict__.__setitem__('stypy_call_defaults', defaults)
        ODECheckParameterUse._get_solver.__dict__.__setitem__('stypy_call_varargs', varargs)
        ODECheckParameterUse._get_solver.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ODECheckParameterUse._get_solver.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODECheckParameterUse._get_solver', ['f', 'jac'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_solver', localization, ['f', 'jac'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_solver(...)' code ##################

        
        # Assigning a Call to a Name (line 569):
        
        # Assigning a Call to a Name (line 569):
        
        # Call to ode(...): (line 569)
        # Processing the call arguments (line 569)
        # Getting the type of 'f' (line 569)
        f_43972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 21), 'f', False)
        # Getting the type of 'jac' (line 569)
        jac_43973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 24), 'jac', False)
        # Processing the call keyword arguments (line 569)
        kwargs_43974 = {}
        # Getting the type of 'ode' (line 569)
        ode_43971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 17), 'ode', False)
        # Calling ode(args, kwargs) (line 569)
        ode_call_result_43975 = invoke(stypy.reporting.localization.Localization(__file__, 569, 17), ode_43971, *[f_43972, jac_43973], **kwargs_43974)
        
        # Assigning a type to the variable 'solver' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'solver', ode_call_result_43975)
        
        # Getting the type of 'self' (line 570)
        self_43976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 11), 'self')
        # Obtaining the member 'solver_uses_jac' of a type (line 570)
        solver_uses_jac_43977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 11), self_43976, 'solver_uses_jac')
        # Testing the type of an if condition (line 570)
        if_condition_43978 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 570, 8), solver_uses_jac_43977)
        # Assigning a type to the variable 'if_condition_43978' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'if_condition_43978', if_condition_43978)
        # SSA begins for if statement (line 570)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_integrator(...): (line 571)
        # Processing the call arguments (line 571)
        # Getting the type of 'self' (line 571)
        self_43981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 34), 'self', False)
        # Obtaining the member 'solver_name' of a type (line 571)
        solver_name_43982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 34), self_43981, 'solver_name')
        # Processing the call keyword arguments (line 571)
        float_43983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 57), 'float')
        keyword_43984 = float_43983
        float_43985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 68), 'float')
        keyword_43986 = float_43985
        # Getting the type of 'self' (line 572)
        self_43987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 48), 'self', False)
        # Obtaining the member 'solver_uses_jac' of a type (line 572)
        solver_uses_jac_43988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 48), self_43987, 'solver_uses_jac')
        keyword_43989 = solver_uses_jac_43988
        kwargs_43990 = {'with_jacobian': keyword_43989, 'rtol': keyword_43986, 'atol': keyword_43984}
        # Getting the type of 'solver' (line 571)
        solver_43979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 12), 'solver', False)
        # Obtaining the member 'set_integrator' of a type (line 571)
        set_integrator_43980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 12), solver_43979, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 571)
        set_integrator_call_result_43991 = invoke(stypy.reporting.localization.Localization(__file__, 571, 12), set_integrator_43980, *[solver_name_43982], **kwargs_43990)
        
        # SSA branch for the else part of an if statement (line 570)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_integrator(...): (line 577)
        # Processing the call arguments (line 577)
        # Getting the type of 'self' (line 577)
        self_43994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 34), 'self', False)
        # Obtaining the member 'solver_name' of a type (line 577)
        solver_name_43995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 34), self_43994, 'solver_name')
        # Processing the call keyword arguments (line 577)
        float_43996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 57), 'float')
        keyword_43997 = float_43996
        float_43998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 68), 'float')
        keyword_43999 = float_43998
        kwargs_44000 = {'rtol': keyword_43999, 'atol': keyword_43997}
        # Getting the type of 'solver' (line 577)
        solver_43992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 12), 'solver', False)
        # Obtaining the member 'set_integrator' of a type (line 577)
        set_integrator_43993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 12), solver_43992, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 577)
        set_integrator_call_result_44001 = invoke(stypy.reporting.localization.Localization(__file__, 577, 12), set_integrator_43993, *[solver_name_43995], **kwargs_44000)
        
        # SSA join for if statement (line 570)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'solver' (line 578)
        solver_44002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 15), 'solver')
        # Assigning a type to the variable 'stypy_return_type' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'stypy_return_type', solver_44002)
        
        # ################# End of '_get_solver(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_solver' in the type store
        # Getting the type of 'stypy_return_type' (line 568)
        stypy_return_type_44003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44003)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_solver'
        return stypy_return_type_44003


    @norecursion
    def _check_solver(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_solver'
        module_type_store = module_type_store.open_function_context('_check_solver', 580, 4, False)
        # Assigning a type to the variable 'self' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ODECheckParameterUse._check_solver.__dict__.__setitem__('stypy_localization', localization)
        ODECheckParameterUse._check_solver.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ODECheckParameterUse._check_solver.__dict__.__setitem__('stypy_type_store', module_type_store)
        ODECheckParameterUse._check_solver.__dict__.__setitem__('stypy_function_name', 'ODECheckParameterUse._check_solver')
        ODECheckParameterUse._check_solver.__dict__.__setitem__('stypy_param_names_list', ['solver'])
        ODECheckParameterUse._check_solver.__dict__.__setitem__('stypy_varargs_param_name', None)
        ODECheckParameterUse._check_solver.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ODECheckParameterUse._check_solver.__dict__.__setitem__('stypy_call_defaults', defaults)
        ODECheckParameterUse._check_solver.__dict__.__setitem__('stypy_call_varargs', varargs)
        ODECheckParameterUse._check_solver.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ODECheckParameterUse._check_solver.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODECheckParameterUse._check_solver', ['solver'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_solver', localization, ['solver'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_solver(...)' code ##################

        
        # Assigning a List to a Name (line 581):
        
        # Assigning a List to a Name (line 581):
        
        # Obtaining an instance of the builtin type 'list' (line 581)
        list_44004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 581)
        # Adding element type (line 581)
        float_44005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 581, 13), list_44004, float_44005)
        # Adding element type (line 581)
        float_44006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 581, 13), list_44004, float_44006)
        
        # Assigning a type to the variable 'ic' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'ic', list_44004)
        
        # Call to set_initial_value(...): (line 582)
        # Processing the call arguments (line 582)
        # Getting the type of 'ic' (line 582)
        ic_44009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 33), 'ic', False)
        float_44010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 37), 'float')
        # Processing the call keyword arguments (line 582)
        kwargs_44011 = {}
        # Getting the type of 'solver' (line 582)
        solver_44007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'solver', False)
        # Obtaining the member 'set_initial_value' of a type (line 582)
        set_initial_value_44008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 8), solver_44007, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 582)
        set_initial_value_call_result_44012 = invoke(stypy.reporting.localization.Localization(__file__, 582, 8), set_initial_value_44008, *[ic_44009, float_44010], **kwargs_44011)
        
        
        # Call to integrate(...): (line 583)
        # Processing the call arguments (line 583)
        # Getting the type of 'pi' (line 583)
        pi_44015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 25), 'pi', False)
        # Processing the call keyword arguments (line 583)
        kwargs_44016 = {}
        # Getting the type of 'solver' (line 583)
        solver_44013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'solver', False)
        # Obtaining the member 'integrate' of a type (line 583)
        integrate_44014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 8), solver_44013, 'integrate')
        # Calling integrate(args, kwargs) (line 583)
        integrate_call_result_44017 = invoke(stypy.reporting.localization.Localization(__file__, 583, 8), integrate_44014, *[pi_44015], **kwargs_44016)
        
        
        # Call to assert_array_almost_equal(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 'solver' (line 584)
        solver_44019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 34), 'solver', False)
        # Obtaining the member 'y' of a type (line 584)
        y_44020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 34), solver_44019, 'y')
        
        # Obtaining an instance of the builtin type 'list' (line 584)
        list_44021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 584)
        # Adding element type (line 584)
        float_44022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 44), list_44021, float_44022)
        # Adding element type (line 584)
        float_44023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 44), list_44021, float_44023)
        
        # Processing the call keyword arguments (line 584)
        kwargs_44024 = {}
        # Getting the type of 'assert_array_almost_equal' (line 584)
        assert_array_almost_equal_44018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 584)
        assert_array_almost_equal_call_result_44025 = invoke(stypy.reporting.localization.Localization(__file__, 584, 8), assert_array_almost_equal_44018, *[y_44020, list_44021], **kwargs_44024)
        
        
        # ################# End of '_check_solver(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_solver' in the type store
        # Getting the type of 'stypy_return_type' (line 580)
        stypy_return_type_44026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44026)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_solver'
        return stypy_return_type_44026


    @norecursion
    def test_no_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_no_params'
        module_type_store = module_type_store.open_function_context('test_no_params', 586, 4, False)
        # Assigning a type to the variable 'self' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ODECheckParameterUse.test_no_params.__dict__.__setitem__('stypy_localization', localization)
        ODECheckParameterUse.test_no_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ODECheckParameterUse.test_no_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        ODECheckParameterUse.test_no_params.__dict__.__setitem__('stypy_function_name', 'ODECheckParameterUse.test_no_params')
        ODECheckParameterUse.test_no_params.__dict__.__setitem__('stypy_param_names_list', [])
        ODECheckParameterUse.test_no_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        ODECheckParameterUse.test_no_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ODECheckParameterUse.test_no_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        ODECheckParameterUse.test_no_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        ODECheckParameterUse.test_no_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ODECheckParameterUse.test_no_params.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODECheckParameterUse.test_no_params', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_no_params', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_no_params(...)' code ##################

        
        # Assigning a Call to a Name (line 587):
        
        # Assigning a Call to a Name (line 587):
        
        # Call to _get_solver(...): (line 587)
        # Processing the call arguments (line 587)
        # Getting the type of 'f' (line 587)
        f_44029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 34), 'f', False)
        # Getting the type of 'jac' (line 587)
        jac_44030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 37), 'jac', False)
        # Processing the call keyword arguments (line 587)
        kwargs_44031 = {}
        # Getting the type of 'self' (line 587)
        self_44027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 17), 'self', False)
        # Obtaining the member '_get_solver' of a type (line 587)
        _get_solver_44028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 17), self_44027, '_get_solver')
        # Calling _get_solver(args, kwargs) (line 587)
        _get_solver_call_result_44032 = invoke(stypy.reporting.localization.Localization(__file__, 587, 17), _get_solver_44028, *[f_44029, jac_44030], **kwargs_44031)
        
        # Assigning a type to the variable 'solver' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'solver', _get_solver_call_result_44032)
        
        # Call to _check_solver(...): (line 588)
        # Processing the call arguments (line 588)
        # Getting the type of 'solver' (line 588)
        solver_44035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 27), 'solver', False)
        # Processing the call keyword arguments (line 588)
        kwargs_44036 = {}
        # Getting the type of 'self' (line 588)
        self_44033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'self', False)
        # Obtaining the member '_check_solver' of a type (line 588)
        _check_solver_44034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), self_44033, '_check_solver')
        # Calling _check_solver(args, kwargs) (line 588)
        _check_solver_call_result_44037 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), _check_solver_44034, *[solver_44035], **kwargs_44036)
        
        
        # ################# End of 'test_no_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_no_params' in the type store
        # Getting the type of 'stypy_return_type' (line 586)
        stypy_return_type_44038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44038)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_no_params'
        return stypy_return_type_44038


    @norecursion
    def test_one_scalar_param(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_one_scalar_param'
        module_type_store = module_type_store.open_function_context('test_one_scalar_param', 590, 4, False)
        # Assigning a type to the variable 'self' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ODECheckParameterUse.test_one_scalar_param.__dict__.__setitem__('stypy_localization', localization)
        ODECheckParameterUse.test_one_scalar_param.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ODECheckParameterUse.test_one_scalar_param.__dict__.__setitem__('stypy_type_store', module_type_store)
        ODECheckParameterUse.test_one_scalar_param.__dict__.__setitem__('stypy_function_name', 'ODECheckParameterUse.test_one_scalar_param')
        ODECheckParameterUse.test_one_scalar_param.__dict__.__setitem__('stypy_param_names_list', [])
        ODECheckParameterUse.test_one_scalar_param.__dict__.__setitem__('stypy_varargs_param_name', None)
        ODECheckParameterUse.test_one_scalar_param.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ODECheckParameterUse.test_one_scalar_param.__dict__.__setitem__('stypy_call_defaults', defaults)
        ODECheckParameterUse.test_one_scalar_param.__dict__.__setitem__('stypy_call_varargs', varargs)
        ODECheckParameterUse.test_one_scalar_param.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ODECheckParameterUse.test_one_scalar_param.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODECheckParameterUse.test_one_scalar_param', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_one_scalar_param', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_one_scalar_param(...)' code ##################

        
        # Assigning a Call to a Name (line 591):
        
        # Assigning a Call to a Name (line 591):
        
        # Call to _get_solver(...): (line 591)
        # Processing the call arguments (line 591)
        # Getting the type of 'f1' (line 591)
        f1_44041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 34), 'f1', False)
        # Getting the type of 'jac1' (line 591)
        jac1_44042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 38), 'jac1', False)
        # Processing the call keyword arguments (line 591)
        kwargs_44043 = {}
        # Getting the type of 'self' (line 591)
        self_44039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 17), 'self', False)
        # Obtaining the member '_get_solver' of a type (line 591)
        _get_solver_44040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 17), self_44039, '_get_solver')
        # Calling _get_solver(args, kwargs) (line 591)
        _get_solver_call_result_44044 = invoke(stypy.reporting.localization.Localization(__file__, 591, 17), _get_solver_44040, *[f1_44041, jac1_44042], **kwargs_44043)
        
        # Assigning a type to the variable 'solver' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'solver', _get_solver_call_result_44044)
        
        # Assigning a Num to a Name (line 592):
        
        # Assigning a Num to a Name (line 592):
        float_44045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 16), 'float')
        # Assigning a type to the variable 'omega' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'omega', float_44045)
        
        # Call to set_f_params(...): (line 593)
        # Processing the call arguments (line 593)
        # Getting the type of 'omega' (line 593)
        omega_44048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 28), 'omega', False)
        # Processing the call keyword arguments (line 593)
        kwargs_44049 = {}
        # Getting the type of 'solver' (line 593)
        solver_44046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'solver', False)
        # Obtaining the member 'set_f_params' of a type (line 593)
        set_f_params_44047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 8), solver_44046, 'set_f_params')
        # Calling set_f_params(args, kwargs) (line 593)
        set_f_params_call_result_44050 = invoke(stypy.reporting.localization.Localization(__file__, 593, 8), set_f_params_44047, *[omega_44048], **kwargs_44049)
        
        
        # Getting the type of 'self' (line 594)
        self_44051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 11), 'self')
        # Obtaining the member 'solver_uses_jac' of a type (line 594)
        solver_uses_jac_44052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 11), self_44051, 'solver_uses_jac')
        # Testing the type of an if condition (line 594)
        if_condition_44053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 594, 8), solver_uses_jac_44052)
        # Assigning a type to the variable 'if_condition_44053' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'if_condition_44053', if_condition_44053)
        # SSA begins for if statement (line 594)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_jac_params(...): (line 595)
        # Processing the call arguments (line 595)
        # Getting the type of 'omega' (line 595)
        omega_44056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 34), 'omega', False)
        # Processing the call keyword arguments (line 595)
        kwargs_44057 = {}
        # Getting the type of 'solver' (line 595)
        solver_44054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'solver', False)
        # Obtaining the member 'set_jac_params' of a type (line 595)
        set_jac_params_44055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 12), solver_44054, 'set_jac_params')
        # Calling set_jac_params(args, kwargs) (line 595)
        set_jac_params_call_result_44058 = invoke(stypy.reporting.localization.Localization(__file__, 595, 12), set_jac_params_44055, *[omega_44056], **kwargs_44057)
        
        # SSA join for if statement (line 594)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _check_solver(...): (line 596)
        # Processing the call arguments (line 596)
        # Getting the type of 'solver' (line 596)
        solver_44061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 27), 'solver', False)
        # Processing the call keyword arguments (line 596)
        kwargs_44062 = {}
        # Getting the type of 'self' (line 596)
        self_44059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'self', False)
        # Obtaining the member '_check_solver' of a type (line 596)
        _check_solver_44060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 8), self_44059, '_check_solver')
        # Calling _check_solver(args, kwargs) (line 596)
        _check_solver_call_result_44063 = invoke(stypy.reporting.localization.Localization(__file__, 596, 8), _check_solver_44060, *[solver_44061], **kwargs_44062)
        
        
        # ################# End of 'test_one_scalar_param(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_one_scalar_param' in the type store
        # Getting the type of 'stypy_return_type' (line 590)
        stypy_return_type_44064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44064)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_one_scalar_param'
        return stypy_return_type_44064


    @norecursion
    def test_two_scalar_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_two_scalar_params'
        module_type_store = module_type_store.open_function_context('test_two_scalar_params', 598, 4, False)
        # Assigning a type to the variable 'self' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ODECheckParameterUse.test_two_scalar_params.__dict__.__setitem__('stypy_localization', localization)
        ODECheckParameterUse.test_two_scalar_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ODECheckParameterUse.test_two_scalar_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        ODECheckParameterUse.test_two_scalar_params.__dict__.__setitem__('stypy_function_name', 'ODECheckParameterUse.test_two_scalar_params')
        ODECheckParameterUse.test_two_scalar_params.__dict__.__setitem__('stypy_param_names_list', [])
        ODECheckParameterUse.test_two_scalar_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        ODECheckParameterUse.test_two_scalar_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ODECheckParameterUse.test_two_scalar_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        ODECheckParameterUse.test_two_scalar_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        ODECheckParameterUse.test_two_scalar_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ODECheckParameterUse.test_two_scalar_params.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODECheckParameterUse.test_two_scalar_params', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_two_scalar_params', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_two_scalar_params(...)' code ##################

        
        # Assigning a Call to a Name (line 599):
        
        # Assigning a Call to a Name (line 599):
        
        # Call to _get_solver(...): (line 599)
        # Processing the call arguments (line 599)
        # Getting the type of 'f2' (line 599)
        f2_44067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 34), 'f2', False)
        # Getting the type of 'jac2' (line 599)
        jac2_44068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 38), 'jac2', False)
        # Processing the call keyword arguments (line 599)
        kwargs_44069 = {}
        # Getting the type of 'self' (line 599)
        self_44065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 17), 'self', False)
        # Obtaining the member '_get_solver' of a type (line 599)
        _get_solver_44066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 17), self_44065, '_get_solver')
        # Calling _get_solver(args, kwargs) (line 599)
        _get_solver_call_result_44070 = invoke(stypy.reporting.localization.Localization(__file__, 599, 17), _get_solver_44066, *[f2_44067, jac2_44068], **kwargs_44069)
        
        # Assigning a type to the variable 'solver' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'solver', _get_solver_call_result_44070)
        
        # Assigning a Num to a Name (line 600):
        
        # Assigning a Num to a Name (line 600):
        float_44071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 17), 'float')
        # Assigning a type to the variable 'omega1' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'omega1', float_44071)
        
        # Assigning a Num to a Name (line 601):
        
        # Assigning a Num to a Name (line 601):
        float_44072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 17), 'float')
        # Assigning a type to the variable 'omega2' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'omega2', float_44072)
        
        # Call to set_f_params(...): (line 602)
        # Processing the call arguments (line 602)
        # Getting the type of 'omega1' (line 602)
        omega1_44075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 28), 'omega1', False)
        # Getting the type of 'omega2' (line 602)
        omega2_44076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 36), 'omega2', False)
        # Processing the call keyword arguments (line 602)
        kwargs_44077 = {}
        # Getting the type of 'solver' (line 602)
        solver_44073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'solver', False)
        # Obtaining the member 'set_f_params' of a type (line 602)
        set_f_params_44074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 8), solver_44073, 'set_f_params')
        # Calling set_f_params(args, kwargs) (line 602)
        set_f_params_call_result_44078 = invoke(stypy.reporting.localization.Localization(__file__, 602, 8), set_f_params_44074, *[omega1_44075, omega2_44076], **kwargs_44077)
        
        
        # Getting the type of 'self' (line 603)
        self_44079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 11), 'self')
        # Obtaining the member 'solver_uses_jac' of a type (line 603)
        solver_uses_jac_44080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 11), self_44079, 'solver_uses_jac')
        # Testing the type of an if condition (line 603)
        if_condition_44081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 603, 8), solver_uses_jac_44080)
        # Assigning a type to the variable 'if_condition_44081' (line 603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'if_condition_44081', if_condition_44081)
        # SSA begins for if statement (line 603)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_jac_params(...): (line 604)
        # Processing the call arguments (line 604)
        # Getting the type of 'omega1' (line 604)
        omega1_44084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 34), 'omega1', False)
        # Getting the type of 'omega2' (line 604)
        omega2_44085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 42), 'omega2', False)
        # Processing the call keyword arguments (line 604)
        kwargs_44086 = {}
        # Getting the type of 'solver' (line 604)
        solver_44082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'solver', False)
        # Obtaining the member 'set_jac_params' of a type (line 604)
        set_jac_params_44083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 12), solver_44082, 'set_jac_params')
        # Calling set_jac_params(args, kwargs) (line 604)
        set_jac_params_call_result_44087 = invoke(stypy.reporting.localization.Localization(__file__, 604, 12), set_jac_params_44083, *[omega1_44084, omega2_44085], **kwargs_44086)
        
        # SSA join for if statement (line 603)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _check_solver(...): (line 605)
        # Processing the call arguments (line 605)
        # Getting the type of 'solver' (line 605)
        solver_44090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 27), 'solver', False)
        # Processing the call keyword arguments (line 605)
        kwargs_44091 = {}
        # Getting the type of 'self' (line 605)
        self_44088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'self', False)
        # Obtaining the member '_check_solver' of a type (line 605)
        _check_solver_44089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 8), self_44088, '_check_solver')
        # Calling _check_solver(args, kwargs) (line 605)
        _check_solver_call_result_44092 = invoke(stypy.reporting.localization.Localization(__file__, 605, 8), _check_solver_44089, *[solver_44090], **kwargs_44091)
        
        
        # ################# End of 'test_two_scalar_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_two_scalar_params' in the type store
        # Getting the type of 'stypy_return_type' (line 598)
        stypy_return_type_44093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44093)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_two_scalar_params'
        return stypy_return_type_44093


    @norecursion
    def test_vector_param(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vector_param'
        module_type_store = module_type_store.open_function_context('test_vector_param', 607, 4, False)
        # Assigning a type to the variable 'self' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ODECheckParameterUse.test_vector_param.__dict__.__setitem__('stypy_localization', localization)
        ODECheckParameterUse.test_vector_param.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ODECheckParameterUse.test_vector_param.__dict__.__setitem__('stypy_type_store', module_type_store)
        ODECheckParameterUse.test_vector_param.__dict__.__setitem__('stypy_function_name', 'ODECheckParameterUse.test_vector_param')
        ODECheckParameterUse.test_vector_param.__dict__.__setitem__('stypy_param_names_list', [])
        ODECheckParameterUse.test_vector_param.__dict__.__setitem__('stypy_varargs_param_name', None)
        ODECheckParameterUse.test_vector_param.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ODECheckParameterUse.test_vector_param.__dict__.__setitem__('stypy_call_defaults', defaults)
        ODECheckParameterUse.test_vector_param.__dict__.__setitem__('stypy_call_varargs', varargs)
        ODECheckParameterUse.test_vector_param.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ODECheckParameterUse.test_vector_param.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODECheckParameterUse.test_vector_param', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vector_param', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vector_param(...)' code ##################

        
        # Assigning a Call to a Name (line 608):
        
        # Assigning a Call to a Name (line 608):
        
        # Call to _get_solver(...): (line 608)
        # Processing the call arguments (line 608)
        # Getting the type of 'fv' (line 608)
        fv_44096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 34), 'fv', False)
        # Getting the type of 'jacv' (line 608)
        jacv_44097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 38), 'jacv', False)
        # Processing the call keyword arguments (line 608)
        kwargs_44098 = {}
        # Getting the type of 'self' (line 608)
        self_44094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 17), 'self', False)
        # Obtaining the member '_get_solver' of a type (line 608)
        _get_solver_44095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 17), self_44094, '_get_solver')
        # Calling _get_solver(args, kwargs) (line 608)
        _get_solver_call_result_44099 = invoke(stypy.reporting.localization.Localization(__file__, 608, 17), _get_solver_44095, *[fv_44096, jacv_44097], **kwargs_44098)
        
        # Assigning a type to the variable 'solver' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'solver', _get_solver_call_result_44099)
        
        # Assigning a List to a Name (line 609):
        
        # Assigning a List to a Name (line 609):
        
        # Obtaining an instance of the builtin type 'list' (line 609)
        list_44100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 609)
        # Adding element type (line 609)
        float_44101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 609, 16), list_44100, float_44101)
        # Adding element type (line 609)
        float_44102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 609, 16), list_44100, float_44102)
        
        # Assigning a type to the variable 'omega' (line 609)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'omega', list_44100)
        
        # Call to set_f_params(...): (line 610)
        # Processing the call arguments (line 610)
        # Getting the type of 'omega' (line 610)
        omega_44105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 28), 'omega', False)
        # Processing the call keyword arguments (line 610)
        kwargs_44106 = {}
        # Getting the type of 'solver' (line 610)
        solver_44103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'solver', False)
        # Obtaining the member 'set_f_params' of a type (line 610)
        set_f_params_44104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 8), solver_44103, 'set_f_params')
        # Calling set_f_params(args, kwargs) (line 610)
        set_f_params_call_result_44107 = invoke(stypy.reporting.localization.Localization(__file__, 610, 8), set_f_params_44104, *[omega_44105], **kwargs_44106)
        
        
        # Getting the type of 'self' (line 611)
        self_44108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 11), 'self')
        # Obtaining the member 'solver_uses_jac' of a type (line 611)
        solver_uses_jac_44109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 11), self_44108, 'solver_uses_jac')
        # Testing the type of an if condition (line 611)
        if_condition_44110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 611, 8), solver_uses_jac_44109)
        # Assigning a type to the variable 'if_condition_44110' (line 611)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'if_condition_44110', if_condition_44110)
        # SSA begins for if statement (line 611)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_jac_params(...): (line 612)
        # Processing the call arguments (line 612)
        # Getting the type of 'omega' (line 612)
        omega_44113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 34), 'omega', False)
        # Processing the call keyword arguments (line 612)
        kwargs_44114 = {}
        # Getting the type of 'solver' (line 612)
        solver_44111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'solver', False)
        # Obtaining the member 'set_jac_params' of a type (line 612)
        set_jac_params_44112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 12), solver_44111, 'set_jac_params')
        # Calling set_jac_params(args, kwargs) (line 612)
        set_jac_params_call_result_44115 = invoke(stypy.reporting.localization.Localization(__file__, 612, 12), set_jac_params_44112, *[omega_44113], **kwargs_44114)
        
        # SSA join for if statement (line 611)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _check_solver(...): (line 613)
        # Processing the call arguments (line 613)
        # Getting the type of 'solver' (line 613)
        solver_44118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 27), 'solver', False)
        # Processing the call keyword arguments (line 613)
        kwargs_44119 = {}
        # Getting the type of 'self' (line 613)
        self_44116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'self', False)
        # Obtaining the member '_check_solver' of a type (line 613)
        _check_solver_44117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 8), self_44116, '_check_solver')
        # Calling _check_solver(args, kwargs) (line 613)
        _check_solver_call_result_44120 = invoke(stypy.reporting.localization.Localization(__file__, 613, 8), _check_solver_44117, *[solver_44118], **kwargs_44119)
        
        
        # ################# End of 'test_vector_param(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vector_param' in the type store
        # Getting the type of 'stypy_return_type' (line 607)
        stypy_return_type_44121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44121)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vector_param'
        return stypy_return_type_44121


    @norecursion
    def test_warns_on_failure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_warns_on_failure'
        module_type_store = module_type_store.open_function_context('test_warns_on_failure', 615, 4, False)
        # Assigning a type to the variable 'self' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ODECheckParameterUse.test_warns_on_failure.__dict__.__setitem__('stypy_localization', localization)
        ODECheckParameterUse.test_warns_on_failure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ODECheckParameterUse.test_warns_on_failure.__dict__.__setitem__('stypy_type_store', module_type_store)
        ODECheckParameterUse.test_warns_on_failure.__dict__.__setitem__('stypy_function_name', 'ODECheckParameterUse.test_warns_on_failure')
        ODECheckParameterUse.test_warns_on_failure.__dict__.__setitem__('stypy_param_names_list', [])
        ODECheckParameterUse.test_warns_on_failure.__dict__.__setitem__('stypy_varargs_param_name', None)
        ODECheckParameterUse.test_warns_on_failure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ODECheckParameterUse.test_warns_on_failure.__dict__.__setitem__('stypy_call_defaults', defaults)
        ODECheckParameterUse.test_warns_on_failure.__dict__.__setitem__('stypy_call_varargs', varargs)
        ODECheckParameterUse.test_warns_on_failure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ODECheckParameterUse.test_warns_on_failure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODECheckParameterUse.test_warns_on_failure', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_warns_on_failure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_warns_on_failure(...)' code ##################

        
        # Assigning a Call to a Name (line 618):
        
        # Assigning a Call to a Name (line 618):
        
        # Call to _get_solver(...): (line 618)
        # Processing the call arguments (line 618)
        # Getting the type of 'f' (line 618)
        f_44124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 34), 'f', False)
        # Getting the type of 'jac' (line 618)
        jac_44125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 37), 'jac', False)
        # Processing the call keyword arguments (line 618)
        kwargs_44126 = {}
        # Getting the type of 'self' (line 618)
        self_44122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 17), 'self', False)
        # Obtaining the member '_get_solver' of a type (line 618)
        _get_solver_44123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 17), self_44122, '_get_solver')
        # Calling _get_solver(args, kwargs) (line 618)
        _get_solver_call_result_44127 = invoke(stypy.reporting.localization.Localization(__file__, 618, 17), _get_solver_44123, *[f_44124, jac_44125], **kwargs_44126)
        
        # Assigning a type to the variable 'solver' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'solver', _get_solver_call_result_44127)
        
        # Call to set_integrator(...): (line 619)
        # Processing the call arguments (line 619)
        # Getting the type of 'self' (line 619)
        self_44130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 30), 'self', False)
        # Obtaining the member 'solver_name' of a type (line 619)
        solver_name_44131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 30), self_44130, 'solver_name')
        # Processing the call keyword arguments (line 619)
        int_44132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 55), 'int')
        keyword_44133 = int_44132
        kwargs_44134 = {'nsteps': keyword_44133}
        # Getting the type of 'solver' (line 619)
        solver_44128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'solver', False)
        # Obtaining the member 'set_integrator' of a type (line 619)
        set_integrator_44129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 8), solver_44128, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 619)
        set_integrator_call_result_44135 = invoke(stypy.reporting.localization.Localization(__file__, 619, 8), set_integrator_44129, *[solver_name_44131], **kwargs_44134)
        
        
        # Assigning a List to a Name (line 620):
        
        # Assigning a List to a Name (line 620):
        
        # Obtaining an instance of the builtin type 'list' (line 620)
        list_44136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 620)
        # Adding element type (line 620)
        float_44137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 13), list_44136, float_44137)
        # Adding element type (line 620)
        float_44138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 13), list_44136, float_44138)
        
        # Assigning a type to the variable 'ic' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'ic', list_44136)
        
        # Call to set_initial_value(...): (line 621)
        # Processing the call arguments (line 621)
        # Getting the type of 'ic' (line 621)
        ic_44141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 33), 'ic', False)
        float_44142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 37), 'float')
        # Processing the call keyword arguments (line 621)
        kwargs_44143 = {}
        # Getting the type of 'solver' (line 621)
        solver_44139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'solver', False)
        # Obtaining the member 'set_initial_value' of a type (line 621)
        set_initial_value_44140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 8), solver_44139, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 621)
        set_initial_value_call_result_44144 = invoke(stypy.reporting.localization.Localization(__file__, 621, 8), set_initial_value_44140, *[ic_44141, float_44142], **kwargs_44143)
        
        
        # Call to _assert_warns(...): (line 622)
        # Processing the call arguments (line 622)
        # Getting the type of 'UserWarning' (line 622)
        UserWarning_44146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 22), 'UserWarning', False)
        # Getting the type of 'solver' (line 622)
        solver_44147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 35), 'solver', False)
        # Obtaining the member 'integrate' of a type (line 622)
        integrate_44148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 35), solver_44147, 'integrate')
        # Getting the type of 'pi' (line 622)
        pi_44149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 53), 'pi', False)
        # Processing the call keyword arguments (line 622)
        kwargs_44150 = {}
        # Getting the type of '_assert_warns' (line 622)
        _assert_warns_44145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), '_assert_warns', False)
        # Calling _assert_warns(args, kwargs) (line 622)
        _assert_warns_call_result_44151 = invoke(stypy.reporting.localization.Localization(__file__, 622, 8), _assert_warns_44145, *[UserWarning_44146, integrate_44148, pi_44149], **kwargs_44150)
        
        
        # ################# End of 'test_warns_on_failure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_warns_on_failure' in the type store
        # Getting the type of 'stypy_return_type' (line 615)
        stypy_return_type_44152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44152)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_warns_on_failure'
        return stypy_return_type_44152


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 559, 0, False)
        # Assigning a type to the variable 'self' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODECheckParameterUse.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ODECheckParameterUse' (line 559)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 0), 'ODECheckParameterUse', ODECheckParameterUse)

# Assigning a Str to a Name (line 565):
str_44153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 18), 'str', '')
# Getting the type of 'ODECheckParameterUse'
ODECheckParameterUse_44154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ODECheckParameterUse')
# Setting the type of the member 'solver_name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ODECheckParameterUse_44154, 'solver_name', str_44153)

# Assigning a Name to a Name (line 566):
# Getting the type of 'False' (line 566)
False_44155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 22), 'False')
# Getting the type of 'ODECheckParameterUse'
ODECheckParameterUse_44156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ODECheckParameterUse')
# Setting the type of the member 'solver_uses_jac' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ODECheckParameterUse_44156, 'solver_uses_jac', False_44155)
# Declaration of the 'TestDOPRI5CheckParameterUse' class
# Getting the type of 'ODECheckParameterUse' (line 625)
ODECheckParameterUse_44157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 34), 'ODECheckParameterUse')

class TestDOPRI5CheckParameterUse(ODECheckParameterUse_44157, ):
    
    # Assigning a Str to a Name (line 626):
    
    # Assigning a Name to a Name (line 627):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 625, 0, False)
        # Assigning a type to the variable 'self' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDOPRI5CheckParameterUse.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDOPRI5CheckParameterUse' (line 625)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 0), 'TestDOPRI5CheckParameterUse', TestDOPRI5CheckParameterUse)

# Assigning a Str to a Name (line 626):
str_44158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 18), 'str', 'dopri5')
# Getting the type of 'TestDOPRI5CheckParameterUse'
TestDOPRI5CheckParameterUse_44159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDOPRI5CheckParameterUse')
# Setting the type of the member 'solver_name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDOPRI5CheckParameterUse_44159, 'solver_name', str_44158)

# Assigning a Name to a Name (line 627):
# Getting the type of 'False' (line 627)
False_44160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 22), 'False')
# Getting the type of 'TestDOPRI5CheckParameterUse'
TestDOPRI5CheckParameterUse_44161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDOPRI5CheckParameterUse')
# Setting the type of the member 'solver_uses_jac' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDOPRI5CheckParameterUse_44161, 'solver_uses_jac', False_44160)
# Declaration of the 'TestDOP853CheckParameterUse' class
# Getting the type of 'ODECheckParameterUse' (line 630)
ODECheckParameterUse_44162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 34), 'ODECheckParameterUse')

class TestDOP853CheckParameterUse(ODECheckParameterUse_44162, ):
    
    # Assigning a Str to a Name (line 631):
    
    # Assigning a Name to a Name (line 632):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 630, 0, False)
        # Assigning a type to the variable 'self' (line 631)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDOP853CheckParameterUse.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDOP853CheckParameterUse' (line 630)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 0), 'TestDOP853CheckParameterUse', TestDOP853CheckParameterUse)

# Assigning a Str to a Name (line 631):
str_44163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 18), 'str', 'dop853')
# Getting the type of 'TestDOP853CheckParameterUse'
TestDOP853CheckParameterUse_44164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDOP853CheckParameterUse')
# Setting the type of the member 'solver_name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDOP853CheckParameterUse_44164, 'solver_name', str_44163)

# Assigning a Name to a Name (line 632):
# Getting the type of 'False' (line 632)
False_44165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 22), 'False')
# Getting the type of 'TestDOP853CheckParameterUse'
TestDOP853CheckParameterUse_44166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDOP853CheckParameterUse')
# Setting the type of the member 'solver_uses_jac' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDOP853CheckParameterUse_44166, 'solver_uses_jac', False_44165)
# Declaration of the 'TestVODECheckParameterUse' class
# Getting the type of 'ODECheckParameterUse' (line 635)
ODECheckParameterUse_44167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 32), 'ODECheckParameterUse')

class TestVODECheckParameterUse(ODECheckParameterUse_44167, ):
    
    # Assigning a Str to a Name (line 636):
    
    # Assigning a Name to a Name (line 637):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 635, 0, False)
        # Assigning a type to the variable 'self' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestVODECheckParameterUse.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestVODECheckParameterUse' (line 635)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 0), 'TestVODECheckParameterUse', TestVODECheckParameterUse)

# Assigning a Str to a Name (line 636):
str_44168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 18), 'str', 'vode')
# Getting the type of 'TestVODECheckParameterUse'
TestVODECheckParameterUse_44169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestVODECheckParameterUse')
# Setting the type of the member 'solver_name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestVODECheckParameterUse_44169, 'solver_name', str_44168)

# Assigning a Name to a Name (line 637):
# Getting the type of 'True' (line 637)
True_44170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 22), 'True')
# Getting the type of 'TestVODECheckParameterUse'
TestVODECheckParameterUse_44171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestVODECheckParameterUse')
# Setting the type of the member 'solver_uses_jac' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestVODECheckParameterUse_44171, 'solver_uses_jac', True_44170)
# Declaration of the 'TestZVODECheckParameterUse' class
# Getting the type of 'ODECheckParameterUse' (line 640)
ODECheckParameterUse_44172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 33), 'ODECheckParameterUse')

class TestZVODECheckParameterUse(ODECheckParameterUse_44172, ):
    
    # Assigning a Str to a Name (line 641):
    
    # Assigning a Name to a Name (line 642):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 640, 0, False)
        # Assigning a type to the variable 'self' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZVODECheckParameterUse.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestZVODECheckParameterUse' (line 640)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 0), 'TestZVODECheckParameterUse', TestZVODECheckParameterUse)

# Assigning a Str to a Name (line 641):
str_44173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 18), 'str', 'zvode')
# Getting the type of 'TestZVODECheckParameterUse'
TestZVODECheckParameterUse_44174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestZVODECheckParameterUse')
# Setting the type of the member 'solver_name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestZVODECheckParameterUse_44174, 'solver_name', str_44173)

# Assigning a Name to a Name (line 642):
# Getting the type of 'True' (line 642)
True_44175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 22), 'True')
# Getting the type of 'TestZVODECheckParameterUse'
TestZVODECheckParameterUse_44176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestZVODECheckParameterUse')
# Setting the type of the member 'solver_uses_jac' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestZVODECheckParameterUse_44176, 'solver_uses_jac', True_44175)
# Declaration of the 'TestLSODACheckParameterUse' class
# Getting the type of 'ODECheckParameterUse' (line 645)
ODECheckParameterUse_44177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 33), 'ODECheckParameterUse')

class TestLSODACheckParameterUse(ODECheckParameterUse_44177, ):
    
    # Assigning a Str to a Name (line 646):
    
    # Assigning a Name to a Name (line 647):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 645, 0, False)
        # Assigning a type to the variable 'self' (line 646)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSODACheckParameterUse.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLSODACheckParameterUse' (line 645)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 0), 'TestLSODACheckParameterUse', TestLSODACheckParameterUse)

# Assigning a Str to a Name (line 646):
str_44178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 18), 'str', 'lsoda')
# Getting the type of 'TestLSODACheckParameterUse'
TestLSODACheckParameterUse_44179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestLSODACheckParameterUse')
# Setting the type of the member 'solver_name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestLSODACheckParameterUse_44179, 'solver_name', str_44178)

# Assigning a Name to a Name (line 647):
# Getting the type of 'True' (line 647)
True_44180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 22), 'True')
# Getting the type of 'TestLSODACheckParameterUse'
TestLSODACheckParameterUse_44181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestLSODACheckParameterUse')
# Setting the type of the member 'solver_uses_jac' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestLSODACheckParameterUse_44181, 'solver_uses_jac', True_44180)

@norecursion
def test_odeint_trivial_time(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_odeint_trivial_time'
    module_type_store = module_type_store.open_function_context('test_odeint_trivial_time', 650, 0, False)
    
    # Passed parameters checking function
    test_odeint_trivial_time.stypy_localization = localization
    test_odeint_trivial_time.stypy_type_of_self = None
    test_odeint_trivial_time.stypy_type_store = module_type_store
    test_odeint_trivial_time.stypy_function_name = 'test_odeint_trivial_time'
    test_odeint_trivial_time.stypy_param_names_list = []
    test_odeint_trivial_time.stypy_varargs_param_name = None
    test_odeint_trivial_time.stypy_kwargs_param_name = None
    test_odeint_trivial_time.stypy_call_defaults = defaults
    test_odeint_trivial_time.stypy_call_varargs = varargs
    test_odeint_trivial_time.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_odeint_trivial_time', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_odeint_trivial_time', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_odeint_trivial_time(...)' code ##################

    
    # Assigning a Num to a Name (line 653):
    
    # Assigning a Num to a Name (line 653):
    int_44182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 9), 'int')
    # Assigning a type to the variable 'y0' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'y0', int_44182)
    
    # Assigning a List to a Name (line 654):
    
    # Assigning a List to a Name (line 654):
    
    # Obtaining an instance of the builtin type 'list' (line 654)
    list_44183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 654)
    # Adding element type (line 654)
    int_44184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 8), list_44183, int_44184)
    
    # Assigning a type to the variable 't' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 't', list_44183)
    
    # Assigning a Call to a Tuple (line 655):
    
    # Assigning a Subscript to a Name (line 655):
    
    # Obtaining the type of the subscript
    int_44185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 4), 'int')
    
    # Call to odeint(...): (line 655)
    # Processing the call arguments (line 655)

    @norecursion
    def _stypy_temp_lambda_21(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_21'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_21', 655, 21, True)
        # Passed parameters checking function
        _stypy_temp_lambda_21.stypy_localization = localization
        _stypy_temp_lambda_21.stypy_type_of_self = None
        _stypy_temp_lambda_21.stypy_type_store = module_type_store
        _stypy_temp_lambda_21.stypy_function_name = '_stypy_temp_lambda_21'
        _stypy_temp_lambda_21.stypy_param_names_list = ['y', 't']
        _stypy_temp_lambda_21.stypy_varargs_param_name = None
        _stypy_temp_lambda_21.stypy_kwargs_param_name = None
        _stypy_temp_lambda_21.stypy_call_defaults = defaults
        _stypy_temp_lambda_21.stypy_call_varargs = varargs
        _stypy_temp_lambda_21.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_21', ['y', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_21', ['y', 't'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Getting the type of 'y' (line 655)
        y_44187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 35), 'y', False)
        # Applying the 'usub' unary operator (line 655)
        result___neg___44188 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 34), 'usub', y_44187)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 655)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 21), 'stypy_return_type', result___neg___44188)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_21' in the type store
        # Getting the type of 'stypy_return_type' (line 655)
        stypy_return_type_44189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 21), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44189)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_21'
        return stypy_return_type_44189

    # Assigning a type to the variable '_stypy_temp_lambda_21' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 21), '_stypy_temp_lambda_21', _stypy_temp_lambda_21)
    # Getting the type of '_stypy_temp_lambda_21' (line 655)
    _stypy_temp_lambda_21_44190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 21), '_stypy_temp_lambda_21')
    # Getting the type of 'y0' (line 655)
    y0_44191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 38), 'y0', False)
    # Getting the type of 't' (line 655)
    t_44192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 42), 't', False)
    # Processing the call keyword arguments (line 655)
    # Getting the type of 'True' (line 655)
    True_44193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 57), 'True', False)
    keyword_44194 = True_44193
    kwargs_44195 = {'full_output': keyword_44194}
    # Getting the type of 'odeint' (line 655)
    odeint_44186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 14), 'odeint', False)
    # Calling odeint(args, kwargs) (line 655)
    odeint_call_result_44196 = invoke(stypy.reporting.localization.Localization(__file__, 655, 14), odeint_44186, *[_stypy_temp_lambda_21_44190, y0_44191, t_44192], **kwargs_44195)
    
    # Obtaining the member '__getitem__' of a type (line 655)
    getitem___44197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 4), odeint_call_result_44196, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 655)
    subscript_call_result_44198 = invoke(stypy.reporting.localization.Localization(__file__, 655, 4), getitem___44197, int_44185)
    
    # Assigning a type to the variable 'tuple_var_assignment_42125' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'tuple_var_assignment_42125', subscript_call_result_44198)
    
    # Assigning a Subscript to a Name (line 655):
    
    # Obtaining the type of the subscript
    int_44199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 4), 'int')
    
    # Call to odeint(...): (line 655)
    # Processing the call arguments (line 655)

    @norecursion
    def _stypy_temp_lambda_22(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_22'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_22', 655, 21, True)
        # Passed parameters checking function
        _stypy_temp_lambda_22.stypy_localization = localization
        _stypy_temp_lambda_22.stypy_type_of_self = None
        _stypy_temp_lambda_22.stypy_type_store = module_type_store
        _stypy_temp_lambda_22.stypy_function_name = '_stypy_temp_lambda_22'
        _stypy_temp_lambda_22.stypy_param_names_list = ['y', 't']
        _stypy_temp_lambda_22.stypy_varargs_param_name = None
        _stypy_temp_lambda_22.stypy_kwargs_param_name = None
        _stypy_temp_lambda_22.stypy_call_defaults = defaults
        _stypy_temp_lambda_22.stypy_call_varargs = varargs
        _stypy_temp_lambda_22.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_22', ['y', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_22', ['y', 't'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Getting the type of 'y' (line 655)
        y_44201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 35), 'y', False)
        # Applying the 'usub' unary operator (line 655)
        result___neg___44202 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 34), 'usub', y_44201)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 655)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 21), 'stypy_return_type', result___neg___44202)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_22' in the type store
        # Getting the type of 'stypy_return_type' (line 655)
        stypy_return_type_44203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 21), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44203)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_22'
        return stypy_return_type_44203

    # Assigning a type to the variable '_stypy_temp_lambda_22' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 21), '_stypy_temp_lambda_22', _stypy_temp_lambda_22)
    # Getting the type of '_stypy_temp_lambda_22' (line 655)
    _stypy_temp_lambda_22_44204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 21), '_stypy_temp_lambda_22')
    # Getting the type of 'y0' (line 655)
    y0_44205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 38), 'y0', False)
    # Getting the type of 't' (line 655)
    t_44206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 42), 't', False)
    # Processing the call keyword arguments (line 655)
    # Getting the type of 'True' (line 655)
    True_44207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 57), 'True', False)
    keyword_44208 = True_44207
    kwargs_44209 = {'full_output': keyword_44208}
    # Getting the type of 'odeint' (line 655)
    odeint_44200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 14), 'odeint', False)
    # Calling odeint(args, kwargs) (line 655)
    odeint_call_result_44210 = invoke(stypy.reporting.localization.Localization(__file__, 655, 14), odeint_44200, *[_stypy_temp_lambda_22_44204, y0_44205, t_44206], **kwargs_44209)
    
    # Obtaining the member '__getitem__' of a type (line 655)
    getitem___44211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 4), odeint_call_result_44210, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 655)
    subscript_call_result_44212 = invoke(stypy.reporting.localization.Localization(__file__, 655, 4), getitem___44211, int_44199)
    
    # Assigning a type to the variable 'tuple_var_assignment_42126' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'tuple_var_assignment_42126', subscript_call_result_44212)
    
    # Assigning a Name to a Name (line 655):
    # Getting the type of 'tuple_var_assignment_42125' (line 655)
    tuple_var_assignment_42125_44213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'tuple_var_assignment_42125')
    # Assigning a type to the variable 'y' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'y', tuple_var_assignment_42125_44213)
    
    # Assigning a Name to a Name (line 655):
    # Getting the type of 'tuple_var_assignment_42126' (line 655)
    tuple_var_assignment_42126_44214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'tuple_var_assignment_42126')
    # Assigning a type to the variable 'info' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 7), 'info', tuple_var_assignment_42126_44214)
    
    # Call to assert_array_equal(...): (line 656)
    # Processing the call arguments (line 656)
    # Getting the type of 'y' (line 656)
    y_44216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 23), 'y', False)
    
    # Call to array(...): (line 656)
    # Processing the call arguments (line 656)
    
    # Obtaining an instance of the builtin type 'list' (line 656)
    list_44219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 656)
    # Adding element type (line 656)
    
    # Obtaining an instance of the builtin type 'list' (line 656)
    list_44220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 656)
    # Adding element type (line 656)
    # Getting the type of 'y0' (line 656)
    y0_44221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 37), 'y0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 36), list_44220, y0_44221)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 35), list_44219, list_44220)
    
    # Processing the call keyword arguments (line 656)
    kwargs_44222 = {}
    # Getting the type of 'np' (line 656)
    np_44217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 26), 'np', False)
    # Obtaining the member 'array' of a type (line 656)
    array_44218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 26), np_44217, 'array')
    # Calling array(args, kwargs) (line 656)
    array_call_result_44223 = invoke(stypy.reporting.localization.Localization(__file__, 656, 26), array_44218, *[list_44219], **kwargs_44222)
    
    # Processing the call keyword arguments (line 656)
    kwargs_44224 = {}
    # Getting the type of 'assert_array_equal' (line 656)
    assert_array_equal_44215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 656)
    assert_array_equal_call_result_44225 = invoke(stypy.reporting.localization.Localization(__file__, 656, 4), assert_array_equal_44215, *[y_44216, array_call_result_44223], **kwargs_44224)
    
    
    # ################# End of 'test_odeint_trivial_time(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_odeint_trivial_time' in the type store
    # Getting the type of 'stypy_return_type' (line 650)
    stypy_return_type_44226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44226)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_odeint_trivial_time'
    return stypy_return_type_44226

# Assigning a type to the variable 'test_odeint_trivial_time' (line 650)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 0), 'test_odeint_trivial_time', test_odeint_trivial_time)

@norecursion
def test_odeint_banded_jacobian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_odeint_banded_jacobian'
    module_type_store = module_type_store.open_function_context('test_odeint_banded_jacobian', 659, 0, False)
    
    # Passed parameters checking function
    test_odeint_banded_jacobian.stypy_localization = localization
    test_odeint_banded_jacobian.stypy_type_of_self = None
    test_odeint_banded_jacobian.stypy_type_store = module_type_store
    test_odeint_banded_jacobian.stypy_function_name = 'test_odeint_banded_jacobian'
    test_odeint_banded_jacobian.stypy_param_names_list = []
    test_odeint_banded_jacobian.stypy_varargs_param_name = None
    test_odeint_banded_jacobian.stypy_kwargs_param_name = None
    test_odeint_banded_jacobian.stypy_call_defaults = defaults
    test_odeint_banded_jacobian.stypy_call_varargs = varargs
    test_odeint_banded_jacobian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_odeint_banded_jacobian', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_odeint_banded_jacobian', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_odeint_banded_jacobian(...)' code ##################


    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 662, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = ['y', 't', 'c']
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', ['y', 't', 'c'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, ['y', 't', 'c'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        
        # Call to dot(...): (line 663)
        # Processing the call arguments (line 663)
        # Getting the type of 'y' (line 663)
        y_44229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 21), 'y', False)
        # Processing the call keyword arguments (line 663)
        kwargs_44230 = {}
        # Getting the type of 'c' (line 663)
        c_44227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 15), 'c', False)
        # Obtaining the member 'dot' of a type (line 663)
        dot_44228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 15), c_44227, 'dot')
        # Calling dot(args, kwargs) (line 663)
        dot_call_result_44231 = invoke(stypy.reporting.localization.Localization(__file__, 663, 15), dot_44228, *[y_44229], **kwargs_44230)
        
        # Assigning a type to the variable 'stypy_return_type' (line 663)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'stypy_return_type', dot_call_result_44231)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 662)
        stypy_return_type_44232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44232)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_44232

    # Assigning a type to the variable 'func' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 'func', func)

    @norecursion
    def jac(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac'
        module_type_store = module_type_store.open_function_context('jac', 665, 4, False)
        
        # Passed parameters checking function
        jac.stypy_localization = localization
        jac.stypy_type_of_self = None
        jac.stypy_type_store = module_type_store
        jac.stypy_function_name = 'jac'
        jac.stypy_param_names_list = ['y', 't', 'c']
        jac.stypy_varargs_param_name = None
        jac.stypy_kwargs_param_name = None
        jac.stypy_call_defaults = defaults
        jac.stypy_call_varargs = varargs
        jac.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'jac', ['y', 't', 'c'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac', localization, ['y', 't', 'c'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac(...)' code ##################

        # Getting the type of 'c' (line 666)
        c_44233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 15), 'c')
        # Assigning a type to the variable 'stypy_return_type' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'stypy_return_type', c_44233)
        
        # ################# End of 'jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac' in the type store
        # Getting the type of 'stypy_return_type' (line 665)
        stypy_return_type_44234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44234)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac'
        return stypy_return_type_44234

    # Assigning a type to the variable 'jac' (line 665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'jac', jac)

    @norecursion
    def jac_transpose(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac_transpose'
        module_type_store = module_type_store.open_function_context('jac_transpose', 668, 4, False)
        
        # Passed parameters checking function
        jac_transpose.stypy_localization = localization
        jac_transpose.stypy_type_of_self = None
        jac_transpose.stypy_type_store = module_type_store
        jac_transpose.stypy_function_name = 'jac_transpose'
        jac_transpose.stypy_param_names_list = ['y', 't', 'c']
        jac_transpose.stypy_varargs_param_name = None
        jac_transpose.stypy_kwargs_param_name = None
        jac_transpose.stypy_call_defaults = defaults
        jac_transpose.stypy_call_varargs = varargs
        jac_transpose.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'jac_transpose', ['y', 't', 'c'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_transpose', localization, ['y', 't', 'c'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_transpose(...)' code ##################

        
        # Call to copy(...): (line 669)
        # Processing the call keyword arguments (line 669)
        str_44238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 30), 'str', 'C')
        keyword_44239 = str_44238
        kwargs_44240 = {'order': keyword_44239}
        # Getting the type of 'c' (line 669)
        c_44235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 15), 'c', False)
        # Obtaining the member 'T' of a type (line 669)
        T_44236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 15), c_44235, 'T')
        # Obtaining the member 'copy' of a type (line 669)
        copy_44237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 15), T_44236, 'copy')
        # Calling copy(args, kwargs) (line 669)
        copy_call_result_44241 = invoke(stypy.reporting.localization.Localization(__file__, 669, 15), copy_44237, *[], **kwargs_44240)
        
        # Assigning a type to the variable 'stypy_return_type' (line 669)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 8), 'stypy_return_type', copy_call_result_44241)
        
        # ################# End of 'jac_transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 668)
        stypy_return_type_44242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44242)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_transpose'
        return stypy_return_type_44242

    # Assigning a type to the variable 'jac_transpose' (line 668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 4), 'jac_transpose', jac_transpose)

    @norecursion
    def bjac_rows(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'bjac_rows'
        module_type_store = module_type_store.open_function_context('bjac_rows', 671, 4, False)
        
        # Passed parameters checking function
        bjac_rows.stypy_localization = localization
        bjac_rows.stypy_type_of_self = None
        bjac_rows.stypy_type_store = module_type_store
        bjac_rows.stypy_function_name = 'bjac_rows'
        bjac_rows.stypy_param_names_list = ['y', 't', 'c']
        bjac_rows.stypy_varargs_param_name = None
        bjac_rows.stypy_kwargs_param_name = None
        bjac_rows.stypy_call_defaults = defaults
        bjac_rows.stypy_call_varargs = varargs
        bjac_rows.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'bjac_rows', ['y', 't', 'c'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bjac_rows', localization, ['y', 't', 'c'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bjac_rows(...)' code ##################

        
        # Assigning a Call to a Name (line 672):
        
        # Assigning a Call to a Name (line 672):
        
        # Call to row_stack(...): (line 672)
        # Processing the call arguments (line 672)
        
        # Obtaining an instance of the builtin type 'tuple' (line 672)
        tuple_44245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 672)
        # Adding element type (line 672)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 672)
        tuple_44246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 672)
        # Adding element type (line 672)
        int_44247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 34), tuple_44246, int_44247)
        # Adding element type (line 672)
        
        # Call to diag(...): (line 672)
        # Processing the call arguments (line 672)
        # Getting the type of 'c' (line 672)
        c_44250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 45), 'c', False)
        int_44251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 48), 'int')
        # Processing the call keyword arguments (line 672)
        kwargs_44252 = {}
        # Getting the type of 'np' (line 672)
        np_44248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 37), 'np', False)
        # Obtaining the member 'diag' of a type (line 672)
        diag_44249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 37), np_44248, 'diag')
        # Calling diag(args, kwargs) (line 672)
        diag_call_result_44253 = invoke(stypy.reporting.localization.Localization(__file__, 672, 37), diag_44249, *[c_44250, int_44251], **kwargs_44252)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 34), tuple_44246, diag_call_result_44253)
        
        # Getting the type of 'np' (line 672)
        np_44254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 28), 'np', False)
        # Obtaining the member 'r_' of a type (line 672)
        r__44255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 28), np_44254, 'r_')
        # Obtaining the member '__getitem__' of a type (line 672)
        getitem___44256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 28), r__44255, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 672)
        subscript_call_result_44257 = invoke(stypy.reporting.localization.Localization(__file__, 672, 28), getitem___44256, tuple_44246)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 28), tuple_44245, subscript_call_result_44257)
        # Adding element type (line 672)
        
        # Call to diag(...): (line 673)
        # Processing the call arguments (line 673)
        # Getting the type of 'c' (line 673)
        c_44260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 36), 'c', False)
        # Processing the call keyword arguments (line 673)
        kwargs_44261 = {}
        # Getting the type of 'np' (line 673)
        np_44258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 28), 'np', False)
        # Obtaining the member 'diag' of a type (line 673)
        diag_44259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 28), np_44258, 'diag')
        # Calling diag(args, kwargs) (line 673)
        diag_call_result_44262 = invoke(stypy.reporting.localization.Localization(__file__, 673, 28), diag_44259, *[c_44260], **kwargs_44261)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 28), tuple_44245, diag_call_result_44262)
        # Adding element type (line 672)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 674)
        tuple_44263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 674)
        # Adding element type (line 674)
        
        # Call to diag(...): (line 674)
        # Processing the call arguments (line 674)
        # Getting the type of 'c' (line 674)
        c_44266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 42), 'c', False)
        int_44267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 45), 'int')
        # Processing the call keyword arguments (line 674)
        kwargs_44268 = {}
        # Getting the type of 'np' (line 674)
        np_44264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 34), 'np', False)
        # Obtaining the member 'diag' of a type (line 674)
        diag_44265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 34), np_44264, 'diag')
        # Calling diag(args, kwargs) (line 674)
        diag_call_result_44269 = invoke(stypy.reporting.localization.Localization(__file__, 674, 34), diag_44265, *[c_44266, int_44267], **kwargs_44268)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 674, 34), tuple_44263, diag_call_result_44269)
        # Adding element type (line 674)
        int_44270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 674, 34), tuple_44263, int_44270)
        
        # Getting the type of 'np' (line 674)
        np_44271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 28), 'np', False)
        # Obtaining the member 'r_' of a type (line 674)
        r__44272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 28), np_44271, 'r_')
        # Obtaining the member '__getitem__' of a type (line 674)
        getitem___44273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 28), r__44272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 674)
        subscript_call_result_44274 = invoke(stypy.reporting.localization.Localization(__file__, 674, 28), getitem___44273, tuple_44263)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 28), tuple_44245, subscript_call_result_44274)
        # Adding element type (line 672)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 675)
        tuple_44275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 675)
        # Adding element type (line 675)
        
        # Call to diag(...): (line 675)
        # Processing the call arguments (line 675)
        # Getting the type of 'c' (line 675)
        c_44278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 42), 'c', False)
        int_44279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 45), 'int')
        # Processing the call keyword arguments (line 675)
        kwargs_44280 = {}
        # Getting the type of 'np' (line 675)
        np_44276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 34), 'np', False)
        # Obtaining the member 'diag' of a type (line 675)
        diag_44277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 34), np_44276, 'diag')
        # Calling diag(args, kwargs) (line 675)
        diag_call_result_44281 = invoke(stypy.reporting.localization.Localization(__file__, 675, 34), diag_44277, *[c_44278, int_44279], **kwargs_44280)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 675, 34), tuple_44275, diag_call_result_44281)
        # Adding element type (line 675)
        int_44282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 675, 34), tuple_44275, int_44282)
        # Adding element type (line 675)
        int_44283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 675, 34), tuple_44275, int_44283)
        
        # Getting the type of 'np' (line 675)
        np_44284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 28), 'np', False)
        # Obtaining the member 'r_' of a type (line 675)
        r__44285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 28), np_44284, 'r_')
        # Obtaining the member '__getitem__' of a type (line 675)
        getitem___44286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 28), r__44285, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 675)
        subscript_call_result_44287 = invoke(stypy.reporting.localization.Localization(__file__, 675, 28), getitem___44286, tuple_44275)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 28), tuple_44245, subscript_call_result_44287)
        
        # Processing the call keyword arguments (line 672)
        kwargs_44288 = {}
        # Getting the type of 'np' (line 672)
        np_44243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 14), 'np', False)
        # Obtaining the member 'row_stack' of a type (line 672)
        row_stack_44244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 14), np_44243, 'row_stack')
        # Calling row_stack(args, kwargs) (line 672)
        row_stack_call_result_44289 = invoke(stypy.reporting.localization.Localization(__file__, 672, 14), row_stack_44244, *[tuple_44245], **kwargs_44288)
        
        # Assigning a type to the variable 'jac' (line 672)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'jac', row_stack_call_result_44289)
        # Getting the type of 'jac' (line 676)
        jac_44290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 15), 'jac')
        # Assigning a type to the variable 'stypy_return_type' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'stypy_return_type', jac_44290)
        
        # ################# End of 'bjac_rows(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bjac_rows' in the type store
        # Getting the type of 'stypy_return_type' (line 671)
        stypy_return_type_44291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44291)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bjac_rows'
        return stypy_return_type_44291

    # Assigning a type to the variable 'bjac_rows' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 4), 'bjac_rows', bjac_rows)

    @norecursion
    def bjac_cols(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'bjac_cols'
        module_type_store = module_type_store.open_function_context('bjac_cols', 678, 4, False)
        
        # Passed parameters checking function
        bjac_cols.stypy_localization = localization
        bjac_cols.stypy_type_of_self = None
        bjac_cols.stypy_type_store = module_type_store
        bjac_cols.stypy_function_name = 'bjac_cols'
        bjac_cols.stypy_param_names_list = ['y', 't', 'c']
        bjac_cols.stypy_varargs_param_name = None
        bjac_cols.stypy_kwargs_param_name = None
        bjac_cols.stypy_call_defaults = defaults
        bjac_cols.stypy_call_varargs = varargs
        bjac_cols.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'bjac_cols', ['y', 't', 'c'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bjac_cols', localization, ['y', 't', 'c'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bjac_cols(...)' code ##################

        
        # Call to copy(...): (line 679)
        # Processing the call keyword arguments (line 679)
        str_44300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 47), 'str', 'C')
        keyword_44301 = str_44300
        kwargs_44302 = {'order': keyword_44301}
        
        # Call to bjac_rows(...): (line 679)
        # Processing the call arguments (line 679)
        # Getting the type of 'y' (line 679)
        y_44293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 25), 'y', False)
        # Getting the type of 't' (line 679)
        t_44294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 28), 't', False)
        # Getting the type of 'c' (line 679)
        c_44295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 31), 'c', False)
        # Processing the call keyword arguments (line 679)
        kwargs_44296 = {}
        # Getting the type of 'bjac_rows' (line 679)
        bjac_rows_44292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 15), 'bjac_rows', False)
        # Calling bjac_rows(args, kwargs) (line 679)
        bjac_rows_call_result_44297 = invoke(stypy.reporting.localization.Localization(__file__, 679, 15), bjac_rows_44292, *[y_44293, t_44294, c_44295], **kwargs_44296)
        
        # Obtaining the member 'T' of a type (line 679)
        T_44298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 15), bjac_rows_call_result_44297, 'T')
        # Obtaining the member 'copy' of a type (line 679)
        copy_44299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 15), T_44298, 'copy')
        # Calling copy(args, kwargs) (line 679)
        copy_call_result_44303 = invoke(stypy.reporting.localization.Localization(__file__, 679, 15), copy_44299, *[], **kwargs_44302)
        
        # Assigning a type to the variable 'stypy_return_type' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 8), 'stypy_return_type', copy_call_result_44303)
        
        # ################# End of 'bjac_cols(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bjac_cols' in the type store
        # Getting the type of 'stypy_return_type' (line 678)
        stypy_return_type_44304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44304)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bjac_cols'
        return stypy_return_type_44304

    # Assigning a type to the variable 'bjac_cols' (line 678)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 4), 'bjac_cols', bjac_cols)
    
    # Assigning a Call to a Name (line 681):
    
    # Assigning a Call to a Name (line 681):
    
    # Call to array(...): (line 681)
    # Processing the call arguments (line 681)
    
    # Obtaining an instance of the builtin type 'list' (line 681)
    list_44306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 681)
    # Adding element type (line 681)
    
    # Obtaining an instance of the builtin type 'list' (line 681)
    list_44307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 681)
    # Adding element type (line 681)
    int_44308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 15), list_44307, int_44308)
    # Adding element type (line 681)
    float_44309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 15), list_44307, float_44309)
    # Adding element type (line 681)
    float_44310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 15), list_44307, float_44310)
    # Adding element type (line 681)
    float_44311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 15), list_44307, float_44311)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 14), list_44306, list_44307)
    # Adding element type (line 681)
    
    # Obtaining an instance of the builtin type 'list' (line 682)
    list_44312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 682)
    # Adding element type (line 682)
    float_44313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 15), list_44312, float_44313)
    # Adding element type (line 682)
    float_44314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 15), list_44312, float_44314)
    # Adding element type (line 682)
    float_44315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 15), list_44312, float_44315)
    # Adding element type (line 682)
    float_44316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 15), list_44312, float_44316)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 14), list_44306, list_44312)
    # Adding element type (line 681)
    
    # Obtaining an instance of the builtin type 'list' (line 683)
    list_44317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 683)
    # Adding element type (line 683)
    float_44318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 683, 15), list_44317, float_44318)
    # Adding element type (line 683)
    float_44319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 683, 15), list_44317, float_44319)
    # Adding element type (line 683)
    float_44320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 683, 15), list_44317, float_44320)
    # Adding element type (line 683)
    float_44321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 683, 15), list_44317, float_44321)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 14), list_44306, list_44317)
    # Adding element type (line 681)
    
    # Obtaining an instance of the builtin type 'list' (line 684)
    list_44322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 684)
    # Adding element type (line 684)
    float_44323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 684, 15), list_44322, float_44323)
    # Adding element type (line 684)
    float_44324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 684, 15), list_44322, float_44324)
    # Adding element type (line 684)
    float_44325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 684, 15), list_44322, float_44325)
    # Adding element type (line 684)
    float_44326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 684, 15), list_44322, float_44326)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 14), list_44306, list_44322)
    
    # Processing the call keyword arguments (line 681)
    kwargs_44327 = {}
    # Getting the type of 'array' (line 681)
    array_44305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'array', False)
    # Calling array(args, kwargs) (line 681)
    array_call_result_44328 = invoke(stypy.reporting.localization.Localization(__file__, 681, 8), array_44305, *[list_44306], **kwargs_44327)
    
    # Assigning a type to the variable 'c' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 4), 'c', array_call_result_44328)
    
    # Assigning a Call to a Name (line 686):
    
    # Assigning a Call to a Name (line 686):
    
    # Call to ones(...): (line 686)
    # Processing the call arguments (line 686)
    int_44331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 17), 'int')
    # Processing the call keyword arguments (line 686)
    kwargs_44332 = {}
    # Getting the type of 'np' (line 686)
    np_44329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 9), 'np', False)
    # Obtaining the member 'ones' of a type (line 686)
    ones_44330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 9), np_44329, 'ones')
    # Calling ones(args, kwargs) (line 686)
    ones_call_result_44333 = invoke(stypy.reporting.localization.Localization(__file__, 686, 9), ones_44330, *[int_44331], **kwargs_44332)
    
    # Assigning a type to the variable 'y0' (line 686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'y0', ones_call_result_44333)
    
    # Assigning a Call to a Name (line 687):
    
    # Assigning a Call to a Name (line 687):
    
    # Call to array(...): (line 687)
    # Processing the call arguments (line 687)
    
    # Obtaining an instance of the builtin type 'list' (line 687)
    list_44336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 687)
    # Adding element type (line 687)
    int_44337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 17), list_44336, int_44337)
    # Adding element type (line 687)
    int_44338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 17), list_44336, int_44338)
    # Adding element type (line 687)
    int_44339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 17), list_44336, int_44339)
    # Adding element type (line 687)
    int_44340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 17), list_44336, int_44340)
    
    # Processing the call keyword arguments (line 687)
    kwargs_44341 = {}
    # Getting the type of 'np' (line 687)
    np_44334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 687)
    array_44335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 8), np_44334, 'array')
    # Calling array(args, kwargs) (line 687)
    array_call_result_44342 = invoke(stypy.reporting.localization.Localization(__file__, 687, 8), array_44335, *[list_44336], **kwargs_44341)
    
    # Assigning a type to the variable 't' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 't', array_call_result_44342)
    
    # Assigning a Call to a Tuple (line 690):
    
    # Assigning a Subscript to a Name (line 690):
    
    # Obtaining the type of the subscript
    int_44343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 4), 'int')
    
    # Call to odeint(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of 'func' (line 690)
    func_44345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 25), 'func', False)
    # Getting the type of 'y0' (line 690)
    y0_44346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 31), 'y0', False)
    # Getting the type of 't' (line 690)
    t_44347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 35), 't', False)
    # Processing the call keyword arguments (line 690)
    
    # Obtaining an instance of the builtin type 'tuple' (line 690)
    tuple_44348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 690)
    # Adding element type (line 690)
    # Getting the type of 'c' (line 690)
    c_44349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 44), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 690, 44), tuple_44348, c_44349)
    
    keyword_44350 = tuple_44348
    # Getting the type of 'True' (line 690)
    True_44351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 61), 'True', False)
    keyword_44352 = True_44351
    float_44353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 30), 'float')
    keyword_44354 = float_44353
    float_44355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 42), 'float')
    keyword_44356 = float_44355
    int_44357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 56), 'int')
    keyword_44358 = int_44357
    # Getting the type of 'jac' (line 692)
    jac_44359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 30), 'jac', False)
    keyword_44360 = jac_44359
    kwargs_44361 = {'args': keyword_44350, 'mxstep': keyword_44358, 'full_output': keyword_44352, 'Dfun': keyword_44360, 'rtol': keyword_44356, 'atol': keyword_44354}
    # Getting the type of 'odeint' (line 690)
    odeint_44344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 18), 'odeint', False)
    # Calling odeint(args, kwargs) (line 690)
    odeint_call_result_44362 = invoke(stypy.reporting.localization.Localization(__file__, 690, 18), odeint_44344, *[func_44345, y0_44346, t_44347], **kwargs_44361)
    
    # Obtaining the member '__getitem__' of a type (line 690)
    getitem___44363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 4), odeint_call_result_44362, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 690)
    subscript_call_result_44364 = invoke(stypy.reporting.localization.Localization(__file__, 690, 4), getitem___44363, int_44343)
    
    # Assigning a type to the variable 'tuple_var_assignment_42127' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_42127', subscript_call_result_44364)
    
    # Assigning a Subscript to a Name (line 690):
    
    # Obtaining the type of the subscript
    int_44365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 4), 'int')
    
    # Call to odeint(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of 'func' (line 690)
    func_44367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 25), 'func', False)
    # Getting the type of 'y0' (line 690)
    y0_44368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 31), 'y0', False)
    # Getting the type of 't' (line 690)
    t_44369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 35), 't', False)
    # Processing the call keyword arguments (line 690)
    
    # Obtaining an instance of the builtin type 'tuple' (line 690)
    tuple_44370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 690)
    # Adding element type (line 690)
    # Getting the type of 'c' (line 690)
    c_44371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 44), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 690, 44), tuple_44370, c_44371)
    
    keyword_44372 = tuple_44370
    # Getting the type of 'True' (line 690)
    True_44373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 61), 'True', False)
    keyword_44374 = True_44373
    float_44375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 30), 'float')
    keyword_44376 = float_44375
    float_44377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 42), 'float')
    keyword_44378 = float_44377
    int_44379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 56), 'int')
    keyword_44380 = int_44379
    # Getting the type of 'jac' (line 692)
    jac_44381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 30), 'jac', False)
    keyword_44382 = jac_44381
    kwargs_44383 = {'args': keyword_44372, 'mxstep': keyword_44380, 'full_output': keyword_44374, 'Dfun': keyword_44382, 'rtol': keyword_44378, 'atol': keyword_44376}
    # Getting the type of 'odeint' (line 690)
    odeint_44366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 18), 'odeint', False)
    # Calling odeint(args, kwargs) (line 690)
    odeint_call_result_44384 = invoke(stypy.reporting.localization.Localization(__file__, 690, 18), odeint_44366, *[func_44367, y0_44368, t_44369], **kwargs_44383)
    
    # Obtaining the member '__getitem__' of a type (line 690)
    getitem___44385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 4), odeint_call_result_44384, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 690)
    subscript_call_result_44386 = invoke(stypy.reporting.localization.Localization(__file__, 690, 4), getitem___44385, int_44365)
    
    # Assigning a type to the variable 'tuple_var_assignment_42128' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_42128', subscript_call_result_44386)
    
    # Assigning a Name to a Name (line 690):
    # Getting the type of 'tuple_var_assignment_42127' (line 690)
    tuple_var_assignment_42127_44387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_42127')
    # Assigning a type to the variable 'sol1' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'sol1', tuple_var_assignment_42127_44387)
    
    # Assigning a Name to a Name (line 690):
    # Getting the type of 'tuple_var_assignment_42128' (line 690)
    tuple_var_assignment_42128_44388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_42128')
    # Assigning a type to the variable 'info1' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 10), 'info1', tuple_var_assignment_42128_44388)
    
    # Assigning a Call to a Tuple (line 695):
    
    # Assigning a Subscript to a Name (line 695):
    
    # Obtaining the type of the subscript
    int_44389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 4), 'int')
    
    # Call to odeint(...): (line 695)
    # Processing the call arguments (line 695)
    # Getting the type of 'func' (line 695)
    func_44391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 25), 'func', False)
    # Getting the type of 'y0' (line 695)
    y0_44392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 31), 'y0', False)
    # Getting the type of 't' (line 695)
    t_44393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 't', False)
    # Processing the call keyword arguments (line 695)
    
    # Obtaining an instance of the builtin type 'tuple' (line 695)
    tuple_44394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 695)
    # Adding element type (line 695)
    # Getting the type of 'c' (line 695)
    c_44395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 44), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 44), tuple_44394, c_44395)
    
    keyword_44396 = tuple_44394
    # Getting the type of 'True' (line 695)
    True_44397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 61), 'True', False)
    keyword_44398 = True_44397
    float_44399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 30), 'float')
    keyword_44400 = float_44399
    float_44401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 42), 'float')
    keyword_44402 = float_44401
    int_44403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 56), 'int')
    keyword_44404 = int_44403
    # Getting the type of 'jac_transpose' (line 697)
    jac_transpose_44405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 30), 'jac_transpose', False)
    keyword_44406 = jac_transpose_44405
    # Getting the type of 'True' (line 697)
    True_44407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 55), 'True', False)
    keyword_44408 = True_44407
    kwargs_44409 = {'args': keyword_44396, 'mxstep': keyword_44404, 'col_deriv': keyword_44408, 'full_output': keyword_44398, 'Dfun': keyword_44406, 'rtol': keyword_44402, 'atol': keyword_44400}
    # Getting the type of 'odeint' (line 695)
    odeint_44390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 18), 'odeint', False)
    # Calling odeint(args, kwargs) (line 695)
    odeint_call_result_44410 = invoke(stypy.reporting.localization.Localization(__file__, 695, 18), odeint_44390, *[func_44391, y0_44392, t_44393], **kwargs_44409)
    
    # Obtaining the member '__getitem__' of a type (line 695)
    getitem___44411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 4), odeint_call_result_44410, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 695)
    subscript_call_result_44412 = invoke(stypy.reporting.localization.Localization(__file__, 695, 4), getitem___44411, int_44389)
    
    # Assigning a type to the variable 'tuple_var_assignment_42129' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'tuple_var_assignment_42129', subscript_call_result_44412)
    
    # Assigning a Subscript to a Name (line 695):
    
    # Obtaining the type of the subscript
    int_44413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 4), 'int')
    
    # Call to odeint(...): (line 695)
    # Processing the call arguments (line 695)
    # Getting the type of 'func' (line 695)
    func_44415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 25), 'func', False)
    # Getting the type of 'y0' (line 695)
    y0_44416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 31), 'y0', False)
    # Getting the type of 't' (line 695)
    t_44417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 't', False)
    # Processing the call keyword arguments (line 695)
    
    # Obtaining an instance of the builtin type 'tuple' (line 695)
    tuple_44418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 695)
    # Adding element type (line 695)
    # Getting the type of 'c' (line 695)
    c_44419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 44), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 44), tuple_44418, c_44419)
    
    keyword_44420 = tuple_44418
    # Getting the type of 'True' (line 695)
    True_44421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 61), 'True', False)
    keyword_44422 = True_44421
    float_44423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 30), 'float')
    keyword_44424 = float_44423
    float_44425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 42), 'float')
    keyword_44426 = float_44425
    int_44427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 56), 'int')
    keyword_44428 = int_44427
    # Getting the type of 'jac_transpose' (line 697)
    jac_transpose_44429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 30), 'jac_transpose', False)
    keyword_44430 = jac_transpose_44429
    # Getting the type of 'True' (line 697)
    True_44431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 55), 'True', False)
    keyword_44432 = True_44431
    kwargs_44433 = {'args': keyword_44420, 'mxstep': keyword_44428, 'col_deriv': keyword_44432, 'full_output': keyword_44422, 'Dfun': keyword_44430, 'rtol': keyword_44426, 'atol': keyword_44424}
    # Getting the type of 'odeint' (line 695)
    odeint_44414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 18), 'odeint', False)
    # Calling odeint(args, kwargs) (line 695)
    odeint_call_result_44434 = invoke(stypy.reporting.localization.Localization(__file__, 695, 18), odeint_44414, *[func_44415, y0_44416, t_44417], **kwargs_44433)
    
    # Obtaining the member '__getitem__' of a type (line 695)
    getitem___44435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 4), odeint_call_result_44434, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 695)
    subscript_call_result_44436 = invoke(stypy.reporting.localization.Localization(__file__, 695, 4), getitem___44435, int_44413)
    
    # Assigning a type to the variable 'tuple_var_assignment_42130' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'tuple_var_assignment_42130', subscript_call_result_44436)
    
    # Assigning a Name to a Name (line 695):
    # Getting the type of 'tuple_var_assignment_42129' (line 695)
    tuple_var_assignment_42129_44437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'tuple_var_assignment_42129')
    # Assigning a type to the variable 'sol2' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'sol2', tuple_var_assignment_42129_44437)
    
    # Assigning a Name to a Name (line 695):
    # Getting the type of 'tuple_var_assignment_42130' (line 695)
    tuple_var_assignment_42130_44438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'tuple_var_assignment_42130')
    # Assigning a type to the variable 'info2' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 10), 'info2', tuple_var_assignment_42130_44438)
    
    # Assigning a Call to a Tuple (line 700):
    
    # Assigning a Subscript to a Name (line 700):
    
    # Obtaining the type of the subscript
    int_44439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 4), 'int')
    
    # Call to odeint(...): (line 700)
    # Processing the call arguments (line 700)
    # Getting the type of 'func' (line 700)
    func_44441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 25), 'func', False)
    # Getting the type of 'y0' (line 700)
    y0_44442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 31), 'y0', False)
    # Getting the type of 't' (line 700)
    t_44443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 35), 't', False)
    # Processing the call keyword arguments (line 700)
    
    # Obtaining an instance of the builtin type 'tuple' (line 700)
    tuple_44444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 700)
    # Adding element type (line 700)
    # Getting the type of 'c' (line 700)
    c_44445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 44), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 700, 44), tuple_44444, c_44445)
    
    keyword_44446 = tuple_44444
    # Getting the type of 'True' (line 700)
    True_44447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 61), 'True', False)
    keyword_44448 = True_44447
    float_44449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 30), 'float')
    keyword_44450 = float_44449
    float_44451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 42), 'float')
    keyword_44452 = float_44451
    int_44453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 56), 'int')
    keyword_44454 = int_44453
    # Getting the type of 'bjac_rows' (line 702)
    bjac_rows_44455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 30), 'bjac_rows', False)
    keyword_44456 = bjac_rows_44455
    int_44457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 44), 'int')
    keyword_44458 = int_44457
    int_44459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 50), 'int')
    keyword_44460 = int_44459
    kwargs_44461 = {'args': keyword_44446, 'mxstep': keyword_44454, 'full_output': keyword_44448, 'ml': keyword_44458, 'Dfun': keyword_44456, 'mu': keyword_44460, 'rtol': keyword_44452, 'atol': keyword_44450}
    # Getting the type of 'odeint' (line 700)
    odeint_44440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 18), 'odeint', False)
    # Calling odeint(args, kwargs) (line 700)
    odeint_call_result_44462 = invoke(stypy.reporting.localization.Localization(__file__, 700, 18), odeint_44440, *[func_44441, y0_44442, t_44443], **kwargs_44461)
    
    # Obtaining the member '__getitem__' of a type (line 700)
    getitem___44463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 4), odeint_call_result_44462, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 700)
    subscript_call_result_44464 = invoke(stypy.reporting.localization.Localization(__file__, 700, 4), getitem___44463, int_44439)
    
    # Assigning a type to the variable 'tuple_var_assignment_42131' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'tuple_var_assignment_42131', subscript_call_result_44464)
    
    # Assigning a Subscript to a Name (line 700):
    
    # Obtaining the type of the subscript
    int_44465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 4), 'int')
    
    # Call to odeint(...): (line 700)
    # Processing the call arguments (line 700)
    # Getting the type of 'func' (line 700)
    func_44467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 25), 'func', False)
    # Getting the type of 'y0' (line 700)
    y0_44468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 31), 'y0', False)
    # Getting the type of 't' (line 700)
    t_44469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 35), 't', False)
    # Processing the call keyword arguments (line 700)
    
    # Obtaining an instance of the builtin type 'tuple' (line 700)
    tuple_44470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 700)
    # Adding element type (line 700)
    # Getting the type of 'c' (line 700)
    c_44471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 44), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 700, 44), tuple_44470, c_44471)
    
    keyword_44472 = tuple_44470
    # Getting the type of 'True' (line 700)
    True_44473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 61), 'True', False)
    keyword_44474 = True_44473
    float_44475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 30), 'float')
    keyword_44476 = float_44475
    float_44477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 42), 'float')
    keyword_44478 = float_44477
    int_44479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 56), 'int')
    keyword_44480 = int_44479
    # Getting the type of 'bjac_rows' (line 702)
    bjac_rows_44481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 30), 'bjac_rows', False)
    keyword_44482 = bjac_rows_44481
    int_44483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 44), 'int')
    keyword_44484 = int_44483
    int_44485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 50), 'int')
    keyword_44486 = int_44485
    kwargs_44487 = {'args': keyword_44472, 'mxstep': keyword_44480, 'full_output': keyword_44474, 'ml': keyword_44484, 'Dfun': keyword_44482, 'mu': keyword_44486, 'rtol': keyword_44478, 'atol': keyword_44476}
    # Getting the type of 'odeint' (line 700)
    odeint_44466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 18), 'odeint', False)
    # Calling odeint(args, kwargs) (line 700)
    odeint_call_result_44488 = invoke(stypy.reporting.localization.Localization(__file__, 700, 18), odeint_44466, *[func_44467, y0_44468, t_44469], **kwargs_44487)
    
    # Obtaining the member '__getitem__' of a type (line 700)
    getitem___44489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 4), odeint_call_result_44488, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 700)
    subscript_call_result_44490 = invoke(stypy.reporting.localization.Localization(__file__, 700, 4), getitem___44489, int_44465)
    
    # Assigning a type to the variable 'tuple_var_assignment_42132' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'tuple_var_assignment_42132', subscript_call_result_44490)
    
    # Assigning a Name to a Name (line 700):
    # Getting the type of 'tuple_var_assignment_42131' (line 700)
    tuple_var_assignment_42131_44491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'tuple_var_assignment_42131')
    # Assigning a type to the variable 'sol3' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'sol3', tuple_var_assignment_42131_44491)
    
    # Assigning a Name to a Name (line 700):
    # Getting the type of 'tuple_var_assignment_42132' (line 700)
    tuple_var_assignment_42132_44492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'tuple_var_assignment_42132')
    # Assigning a type to the variable 'info3' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 10), 'info3', tuple_var_assignment_42132_44492)
    
    # Assigning a Call to a Tuple (line 705):
    
    # Assigning a Subscript to a Name (line 705):
    
    # Obtaining the type of the subscript
    int_44493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 4), 'int')
    
    # Call to odeint(...): (line 705)
    # Processing the call arguments (line 705)
    # Getting the type of 'func' (line 705)
    func_44495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 25), 'func', False)
    # Getting the type of 'y0' (line 705)
    y0_44496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 31), 'y0', False)
    # Getting the type of 't' (line 705)
    t_44497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 35), 't', False)
    # Processing the call keyword arguments (line 705)
    
    # Obtaining an instance of the builtin type 'tuple' (line 705)
    tuple_44498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 705)
    # Adding element type (line 705)
    # Getting the type of 'c' (line 705)
    c_44499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 44), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 44), tuple_44498, c_44499)
    
    keyword_44500 = tuple_44498
    # Getting the type of 'True' (line 705)
    True_44501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 61), 'True', False)
    keyword_44502 = True_44501
    float_44503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 30), 'float')
    keyword_44504 = float_44503
    float_44505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 42), 'float')
    keyword_44506 = float_44505
    int_44507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 56), 'int')
    keyword_44508 = int_44507
    # Getting the type of 'bjac_cols' (line 707)
    bjac_cols_44509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 30), 'bjac_cols', False)
    keyword_44510 = bjac_cols_44509
    int_44511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 44), 'int')
    keyword_44512 = int_44511
    int_44513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 50), 'int')
    keyword_44514 = int_44513
    # Getting the type of 'True' (line 707)
    True_44515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 63), 'True', False)
    keyword_44516 = True_44515
    kwargs_44517 = {'args': keyword_44500, 'mxstep': keyword_44508, 'col_deriv': keyword_44516, 'full_output': keyword_44502, 'ml': keyword_44512, 'Dfun': keyword_44510, 'mu': keyword_44514, 'rtol': keyword_44506, 'atol': keyword_44504}
    # Getting the type of 'odeint' (line 705)
    odeint_44494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 18), 'odeint', False)
    # Calling odeint(args, kwargs) (line 705)
    odeint_call_result_44518 = invoke(stypy.reporting.localization.Localization(__file__, 705, 18), odeint_44494, *[func_44495, y0_44496, t_44497], **kwargs_44517)
    
    # Obtaining the member '__getitem__' of a type (line 705)
    getitem___44519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 4), odeint_call_result_44518, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 705)
    subscript_call_result_44520 = invoke(stypy.reporting.localization.Localization(__file__, 705, 4), getitem___44519, int_44493)
    
    # Assigning a type to the variable 'tuple_var_assignment_42133' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'tuple_var_assignment_42133', subscript_call_result_44520)
    
    # Assigning a Subscript to a Name (line 705):
    
    # Obtaining the type of the subscript
    int_44521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 4), 'int')
    
    # Call to odeint(...): (line 705)
    # Processing the call arguments (line 705)
    # Getting the type of 'func' (line 705)
    func_44523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 25), 'func', False)
    # Getting the type of 'y0' (line 705)
    y0_44524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 31), 'y0', False)
    # Getting the type of 't' (line 705)
    t_44525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 35), 't', False)
    # Processing the call keyword arguments (line 705)
    
    # Obtaining an instance of the builtin type 'tuple' (line 705)
    tuple_44526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 705)
    # Adding element type (line 705)
    # Getting the type of 'c' (line 705)
    c_44527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 44), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 44), tuple_44526, c_44527)
    
    keyword_44528 = tuple_44526
    # Getting the type of 'True' (line 705)
    True_44529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 61), 'True', False)
    keyword_44530 = True_44529
    float_44531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 30), 'float')
    keyword_44532 = float_44531
    float_44533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 42), 'float')
    keyword_44534 = float_44533
    int_44535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 56), 'int')
    keyword_44536 = int_44535
    # Getting the type of 'bjac_cols' (line 707)
    bjac_cols_44537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 30), 'bjac_cols', False)
    keyword_44538 = bjac_cols_44537
    int_44539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 44), 'int')
    keyword_44540 = int_44539
    int_44541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 50), 'int')
    keyword_44542 = int_44541
    # Getting the type of 'True' (line 707)
    True_44543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 63), 'True', False)
    keyword_44544 = True_44543
    kwargs_44545 = {'args': keyword_44528, 'mxstep': keyword_44536, 'col_deriv': keyword_44544, 'full_output': keyword_44530, 'ml': keyword_44540, 'Dfun': keyword_44538, 'mu': keyword_44542, 'rtol': keyword_44534, 'atol': keyword_44532}
    # Getting the type of 'odeint' (line 705)
    odeint_44522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 18), 'odeint', False)
    # Calling odeint(args, kwargs) (line 705)
    odeint_call_result_44546 = invoke(stypy.reporting.localization.Localization(__file__, 705, 18), odeint_44522, *[func_44523, y0_44524, t_44525], **kwargs_44545)
    
    # Obtaining the member '__getitem__' of a type (line 705)
    getitem___44547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 4), odeint_call_result_44546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 705)
    subscript_call_result_44548 = invoke(stypy.reporting.localization.Localization(__file__, 705, 4), getitem___44547, int_44521)
    
    # Assigning a type to the variable 'tuple_var_assignment_42134' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'tuple_var_assignment_42134', subscript_call_result_44548)
    
    # Assigning a Name to a Name (line 705):
    # Getting the type of 'tuple_var_assignment_42133' (line 705)
    tuple_var_assignment_42133_44549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'tuple_var_assignment_42133')
    # Assigning a type to the variable 'sol4' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'sol4', tuple_var_assignment_42133_44549)
    
    # Assigning a Name to a Name (line 705):
    # Getting the type of 'tuple_var_assignment_42134' (line 705)
    tuple_var_assignment_42134_44550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'tuple_var_assignment_42134')
    # Assigning a type to the variable 'info4' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 10), 'info4', tuple_var_assignment_42134_44550)
    
    # Call to assert_allclose(...): (line 709)
    # Processing the call arguments (line 709)
    # Getting the type of 'sol1' (line 709)
    sol1_44552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 20), 'sol1', False)
    # Getting the type of 'sol2' (line 709)
    sol2_44553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 26), 'sol2', False)
    # Processing the call keyword arguments (line 709)
    str_44554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 40), 'str', 'sol1 != sol2')
    keyword_44555 = str_44554
    kwargs_44556 = {'err_msg': keyword_44555}
    # Getting the type of 'assert_allclose' (line 709)
    assert_allclose_44551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 709)
    assert_allclose_call_result_44557 = invoke(stypy.reporting.localization.Localization(__file__, 709, 4), assert_allclose_44551, *[sol1_44552, sol2_44553], **kwargs_44556)
    
    
    # Call to assert_allclose(...): (line 710)
    # Processing the call arguments (line 710)
    # Getting the type of 'sol1' (line 710)
    sol1_44559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 20), 'sol1', False)
    # Getting the type of 'sol3' (line 710)
    sol3_44560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 26), 'sol3', False)
    # Processing the call keyword arguments (line 710)
    float_44561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 37), 'float')
    keyword_44562 = float_44561
    str_44563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 52), 'str', 'sol1 != sol3')
    keyword_44564 = str_44563
    kwargs_44565 = {'err_msg': keyword_44564, 'atol': keyword_44562}
    # Getting the type of 'assert_allclose' (line 710)
    assert_allclose_44558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 710)
    assert_allclose_call_result_44566 = invoke(stypy.reporting.localization.Localization(__file__, 710, 4), assert_allclose_44558, *[sol1_44559, sol3_44560], **kwargs_44565)
    
    
    # Call to assert_allclose(...): (line 711)
    # Processing the call arguments (line 711)
    # Getting the type of 'sol3' (line 711)
    sol3_44568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 20), 'sol3', False)
    # Getting the type of 'sol4' (line 711)
    sol4_44569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 26), 'sol4', False)
    # Processing the call keyword arguments (line 711)
    str_44570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 40), 'str', 'sol3 != sol4')
    keyword_44571 = str_44570
    kwargs_44572 = {'err_msg': keyword_44571}
    # Getting the type of 'assert_allclose' (line 711)
    assert_allclose_44567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 711)
    assert_allclose_call_result_44573 = invoke(stypy.reporting.localization.Localization(__file__, 711, 4), assert_allclose_44567, *[sol3_44568, sol4_44569], **kwargs_44572)
    
    
    # Call to assert_array_equal(...): (line 718)
    # Processing the call arguments (line 718)
    
    # Obtaining the type of the subscript
    str_44575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 29), 'str', 'nje')
    # Getting the type of 'info1' (line 718)
    info1_44576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 23), 'info1', False)
    # Obtaining the member '__getitem__' of a type (line 718)
    getitem___44577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 23), info1_44576, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 718)
    subscript_call_result_44578 = invoke(stypy.reporting.localization.Localization(__file__, 718, 23), getitem___44577, str_44575)
    
    
    # Obtaining the type of the subscript
    str_44579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 43), 'str', 'nje')
    # Getting the type of 'info2' (line 718)
    info2_44580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 37), 'info2', False)
    # Obtaining the member '__getitem__' of a type (line 718)
    getitem___44581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 37), info2_44580, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 718)
    subscript_call_result_44582 = invoke(stypy.reporting.localization.Localization(__file__, 718, 37), getitem___44581, str_44579)
    
    # Processing the call keyword arguments (line 718)
    kwargs_44583 = {}
    # Getting the type of 'assert_array_equal' (line 718)
    assert_array_equal_44574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 718)
    assert_array_equal_call_result_44584 = invoke(stypy.reporting.localization.Localization(__file__, 718, 4), assert_array_equal_44574, *[subscript_call_result_44578, subscript_call_result_44582], **kwargs_44583)
    
    
    # Call to assert_array_equal(...): (line 719)
    # Processing the call arguments (line 719)
    
    # Obtaining the type of the subscript
    str_44586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 29), 'str', 'nje')
    # Getting the type of 'info3' (line 719)
    info3_44587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 23), 'info3', False)
    # Obtaining the member '__getitem__' of a type (line 719)
    getitem___44588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 23), info3_44587, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 719)
    subscript_call_result_44589 = invoke(stypy.reporting.localization.Localization(__file__, 719, 23), getitem___44588, str_44586)
    
    
    # Obtaining the type of the subscript
    str_44590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 43), 'str', 'nje')
    # Getting the type of 'info4' (line 719)
    info4_44591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 37), 'info4', False)
    # Obtaining the member '__getitem__' of a type (line 719)
    getitem___44592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 37), info4_44591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 719)
    subscript_call_result_44593 = invoke(stypy.reporting.localization.Localization(__file__, 719, 37), getitem___44592, str_44590)
    
    # Processing the call keyword arguments (line 719)
    kwargs_44594 = {}
    # Getting the type of 'assert_array_equal' (line 719)
    assert_array_equal_44585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 719)
    assert_array_equal_call_result_44595 = invoke(stypy.reporting.localization.Localization(__file__, 719, 4), assert_array_equal_44585, *[subscript_call_result_44589, subscript_call_result_44593], **kwargs_44594)
    
    
    # ################# End of 'test_odeint_banded_jacobian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_odeint_banded_jacobian' in the type store
    # Getting the type of 'stypy_return_type' (line 659)
    stypy_return_type_44596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44596)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_odeint_banded_jacobian'
    return stypy_return_type_44596

# Assigning a type to the variable 'test_odeint_banded_jacobian' (line 659)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 0), 'test_odeint_banded_jacobian', test_odeint_banded_jacobian)

@norecursion
def test_odeint_errors(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_odeint_errors'
    module_type_store = module_type_store.open_function_context('test_odeint_errors', 722, 0, False)
    
    # Passed parameters checking function
    test_odeint_errors.stypy_localization = localization
    test_odeint_errors.stypy_type_of_self = None
    test_odeint_errors.stypy_type_store = module_type_store
    test_odeint_errors.stypy_function_name = 'test_odeint_errors'
    test_odeint_errors.stypy_param_names_list = []
    test_odeint_errors.stypy_varargs_param_name = None
    test_odeint_errors.stypy_kwargs_param_name = None
    test_odeint_errors.stypy_call_defaults = defaults
    test_odeint_errors.stypy_call_varargs = varargs
    test_odeint_errors.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_odeint_errors', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_odeint_errors', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_odeint_errors(...)' code ##################


    @norecursion
    def sys1d(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sys1d'
        module_type_store = module_type_store.open_function_context('sys1d', 723, 4, False)
        
        # Passed parameters checking function
        sys1d.stypy_localization = localization
        sys1d.stypy_type_of_self = None
        sys1d.stypy_type_store = module_type_store
        sys1d.stypy_function_name = 'sys1d'
        sys1d.stypy_param_names_list = ['x', 't']
        sys1d.stypy_varargs_param_name = None
        sys1d.stypy_kwargs_param_name = None
        sys1d.stypy_call_defaults = defaults
        sys1d.stypy_call_varargs = varargs
        sys1d.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'sys1d', ['x', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sys1d', localization, ['x', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sys1d(...)' code ##################

        int_44597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 15), 'int')
        # Getting the type of 'x' (line 724)
        x_44598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 20), 'x')
        # Applying the binary operator '*' (line 724)
        result_mul_44599 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 15), '*', int_44597, x_44598)
        
        # Assigning a type to the variable 'stypy_return_type' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'stypy_return_type', result_mul_44599)
        
        # ################# End of 'sys1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sys1d' in the type store
        # Getting the type of 'stypy_return_type' (line 723)
        stypy_return_type_44600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44600)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sys1d'
        return stypy_return_type_44600

    # Assigning a type to the variable 'sys1d' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 4), 'sys1d', sys1d)

    @norecursion
    def bad1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'bad1'
        module_type_store = module_type_store.open_function_context('bad1', 726, 4, False)
        
        # Passed parameters checking function
        bad1.stypy_localization = localization
        bad1.stypy_type_of_self = None
        bad1.stypy_type_store = module_type_store
        bad1.stypy_function_name = 'bad1'
        bad1.stypy_param_names_list = ['x', 't']
        bad1.stypy_varargs_param_name = None
        bad1.stypy_kwargs_param_name = None
        bad1.stypy_call_defaults = defaults
        bad1.stypy_call_varargs = varargs
        bad1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'bad1', ['x', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bad1', localization, ['x', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bad1(...)' code ##################

        float_44601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 15), 'float')
        int_44602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 19), 'int')
        # Applying the binary operator 'div' (line 727)
        result_div_44603 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 15), 'div', float_44601, int_44602)
        
        # Assigning a type to the variable 'stypy_return_type' (line 727)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'stypy_return_type', result_div_44603)
        
        # ################# End of 'bad1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bad1' in the type store
        # Getting the type of 'stypy_return_type' (line 726)
        stypy_return_type_44604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44604)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bad1'
        return stypy_return_type_44604

    # Assigning a type to the variable 'bad1' (line 726)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'bad1', bad1)

    @norecursion
    def bad2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'bad2'
        module_type_store = module_type_store.open_function_context('bad2', 729, 4, False)
        
        # Passed parameters checking function
        bad2.stypy_localization = localization
        bad2.stypy_type_of_self = None
        bad2.stypy_type_store = module_type_store
        bad2.stypy_function_name = 'bad2'
        bad2.stypy_param_names_list = ['x', 't']
        bad2.stypy_varargs_param_name = None
        bad2.stypy_kwargs_param_name = None
        bad2.stypy_call_defaults = defaults
        bad2.stypy_call_varargs = varargs
        bad2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'bad2', ['x', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bad2', localization, ['x', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bad2(...)' code ##################

        str_44605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 15), 'str', 'foo')
        # Assigning a type to the variable 'stypy_return_type' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'stypy_return_type', str_44605)
        
        # ################# End of 'bad2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bad2' in the type store
        # Getting the type of 'stypy_return_type' (line 729)
        stypy_return_type_44606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44606)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bad2'
        return stypy_return_type_44606

    # Assigning a type to the variable 'bad2' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'bad2', bad2)

    @norecursion
    def bad_jac1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'bad_jac1'
        module_type_store = module_type_store.open_function_context('bad_jac1', 732, 4, False)
        
        # Passed parameters checking function
        bad_jac1.stypy_localization = localization
        bad_jac1.stypy_type_of_self = None
        bad_jac1.stypy_type_store = module_type_store
        bad_jac1.stypy_function_name = 'bad_jac1'
        bad_jac1.stypy_param_names_list = ['x', 't']
        bad_jac1.stypy_varargs_param_name = None
        bad_jac1.stypy_kwargs_param_name = None
        bad_jac1.stypy_call_defaults = defaults
        bad_jac1.stypy_call_varargs = varargs
        bad_jac1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'bad_jac1', ['x', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bad_jac1', localization, ['x', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bad_jac1(...)' code ##################

        float_44607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 15), 'float')
        int_44608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 19), 'int')
        # Applying the binary operator 'div' (line 733)
        result_div_44609 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 15), 'div', float_44607, int_44608)
        
        # Assigning a type to the variable 'stypy_return_type' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'stypy_return_type', result_div_44609)
        
        # ################# End of 'bad_jac1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bad_jac1' in the type store
        # Getting the type of 'stypy_return_type' (line 732)
        stypy_return_type_44610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44610)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bad_jac1'
        return stypy_return_type_44610

    # Assigning a type to the variable 'bad_jac1' (line 732)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'bad_jac1', bad_jac1)

    @norecursion
    def bad_jac2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'bad_jac2'
        module_type_store = module_type_store.open_function_context('bad_jac2', 735, 4, False)
        
        # Passed parameters checking function
        bad_jac2.stypy_localization = localization
        bad_jac2.stypy_type_of_self = None
        bad_jac2.stypy_type_store = module_type_store
        bad_jac2.stypy_function_name = 'bad_jac2'
        bad_jac2.stypy_param_names_list = ['x', 't']
        bad_jac2.stypy_varargs_param_name = None
        bad_jac2.stypy_kwargs_param_name = None
        bad_jac2.stypy_call_defaults = defaults
        bad_jac2.stypy_call_varargs = varargs
        bad_jac2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'bad_jac2', ['x', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bad_jac2', localization, ['x', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bad_jac2(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 736)
        list_44611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 736)
        # Adding element type (line 736)
        
        # Obtaining an instance of the builtin type 'list' (line 736)
        list_44612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 736)
        # Adding element type (line 736)
        str_44613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 17), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 736, 16), list_44612, str_44613)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 736, 15), list_44611, list_44612)
        
        # Assigning a type to the variable 'stypy_return_type' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 8), 'stypy_return_type', list_44611)
        
        # ################# End of 'bad_jac2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bad_jac2' in the type store
        # Getting the type of 'stypy_return_type' (line 735)
        stypy_return_type_44614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44614)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bad_jac2'
        return stypy_return_type_44614

    # Assigning a type to the variable 'bad_jac2' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'bad_jac2', bad_jac2)

    @norecursion
    def sys2d(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sys2d'
        module_type_store = module_type_store.open_function_context('sys2d', 738, 4, False)
        
        # Passed parameters checking function
        sys2d.stypy_localization = localization
        sys2d.stypy_type_of_self = None
        sys2d.stypy_type_store = module_type_store
        sys2d.stypy_function_name = 'sys2d'
        sys2d.stypy_param_names_list = ['x', 't']
        sys2d.stypy_varargs_param_name = None
        sys2d.stypy_kwargs_param_name = None
        sys2d.stypy_call_defaults = defaults
        sys2d.stypy_call_varargs = varargs
        sys2d.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'sys2d', ['x', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sys2d', localization, ['x', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sys2d(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 739)
        list_44615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 739)
        # Adding element type (line 739)
        int_44616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 16), 'int')
        
        # Obtaining the type of the subscript
        int_44617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 23), 'int')
        # Getting the type of 'x' (line 739)
        x_44618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 21), 'x')
        # Obtaining the member '__getitem__' of a type (line 739)
        getitem___44619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 21), x_44618, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 739)
        subscript_call_result_44620 = invoke(stypy.reporting.localization.Localization(__file__, 739, 21), getitem___44619, int_44617)
        
        # Applying the binary operator '*' (line 739)
        result_mul_44621 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 16), '*', int_44616, subscript_call_result_44620)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 739, 15), list_44615, result_mul_44621)
        # Adding element type (line 739)
        float_44622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 27), 'float')
        
        # Obtaining the type of the subscript
        int_44623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 34), 'int')
        # Getting the type of 'x' (line 739)
        x_44624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 32), 'x')
        # Obtaining the member '__getitem__' of a type (line 739)
        getitem___44625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 32), x_44624, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 739)
        subscript_call_result_44626 = invoke(stypy.reporting.localization.Localization(__file__, 739, 32), getitem___44625, int_44623)
        
        # Applying the binary operator '*' (line 739)
        result_mul_44627 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 27), '*', float_44622, subscript_call_result_44626)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 739, 15), list_44615, result_mul_44627)
        
        # Assigning a type to the variable 'stypy_return_type' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'stypy_return_type', list_44615)
        
        # ################# End of 'sys2d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sys2d' in the type store
        # Getting the type of 'stypy_return_type' (line 738)
        stypy_return_type_44628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44628)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sys2d'
        return stypy_return_type_44628

    # Assigning a type to the variable 'sys2d' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 4), 'sys2d', sys2d)

    @norecursion
    def sys2d_bad_jac(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sys2d_bad_jac'
        module_type_store = module_type_store.open_function_context('sys2d_bad_jac', 741, 4, False)
        
        # Passed parameters checking function
        sys2d_bad_jac.stypy_localization = localization
        sys2d_bad_jac.stypy_type_of_self = None
        sys2d_bad_jac.stypy_type_store = module_type_store
        sys2d_bad_jac.stypy_function_name = 'sys2d_bad_jac'
        sys2d_bad_jac.stypy_param_names_list = ['x', 't']
        sys2d_bad_jac.stypy_varargs_param_name = None
        sys2d_bad_jac.stypy_kwargs_param_name = None
        sys2d_bad_jac.stypy_call_defaults = defaults
        sys2d_bad_jac.stypy_call_varargs = varargs
        sys2d_bad_jac.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'sys2d_bad_jac', ['x', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sys2d_bad_jac', localization, ['x', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sys2d_bad_jac(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 742)
        list_44629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 742)
        # Adding element type (line 742)
        
        # Obtaining an instance of the builtin type 'list' (line 742)
        list_44630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 742)
        # Adding element type (line 742)
        float_44631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 17), 'float')
        int_44632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 21), 'int')
        # Applying the binary operator 'div' (line 742)
        result_div_44633 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 17), 'div', float_44631, int_44632)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 16), list_44630, result_div_44633)
        # Adding element type (line 742)
        int_44634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 16), list_44630, int_44634)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 15), list_44629, list_44630)
        # Adding element type (line 742)
        
        # Obtaining an instance of the builtin type 'list' (line 742)
        list_44635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 742)
        # Adding element type (line 742)
        int_44636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 28), list_44635, int_44636)
        # Adding element type (line 742)
        float_44637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 28), list_44635, float_44637)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 15), list_44629, list_44635)
        
        # Assigning a type to the variable 'stypy_return_type' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'stypy_return_type', list_44629)
        
        # ################# End of 'sys2d_bad_jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sys2d_bad_jac' in the type store
        # Getting the type of 'stypy_return_type' (line 741)
        stypy_return_type_44638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44638)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sys2d_bad_jac'
        return stypy_return_type_44638

    # Assigning a type to the variable 'sys2d_bad_jac' (line 741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 4), 'sys2d_bad_jac', sys2d_bad_jac)
    
    # Call to assert_raises(...): (line 744)
    # Processing the call arguments (line 744)
    # Getting the type of 'ZeroDivisionError' (line 744)
    ZeroDivisionError_44640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 18), 'ZeroDivisionError', False)
    # Getting the type of 'odeint' (line 744)
    odeint_44641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 37), 'odeint', False)
    # Getting the type of 'bad1' (line 744)
    bad1_44642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 45), 'bad1', False)
    float_44643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 51), 'float')
    
    # Obtaining an instance of the builtin type 'list' (line 744)
    list_44644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 56), 'list')
    # Adding type elements to the builtin type 'list' instance (line 744)
    # Adding element type (line 744)
    int_44645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 56), list_44644, int_44645)
    # Adding element type (line 744)
    int_44646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 56), list_44644, int_44646)
    
    # Processing the call keyword arguments (line 744)
    kwargs_44647 = {}
    # Getting the type of 'assert_raises' (line 744)
    assert_raises_44639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 744)
    assert_raises_call_result_44648 = invoke(stypy.reporting.localization.Localization(__file__, 744, 4), assert_raises_44639, *[ZeroDivisionError_44640, odeint_44641, bad1_44642, float_44643, list_44644], **kwargs_44647)
    
    
    # Call to assert_raises(...): (line 745)
    # Processing the call arguments (line 745)
    # Getting the type of 'ValueError' (line 745)
    ValueError_44650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 18), 'ValueError', False)
    # Getting the type of 'odeint' (line 745)
    odeint_44651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 30), 'odeint', False)
    # Getting the type of 'bad2' (line 745)
    bad2_44652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 38), 'bad2', False)
    float_44653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 44), 'float')
    
    # Obtaining an instance of the builtin type 'list' (line 745)
    list_44654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 745)
    # Adding element type (line 745)
    int_44655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 49), list_44654, int_44655)
    # Adding element type (line 745)
    int_44656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 49), list_44654, int_44656)
    
    # Processing the call keyword arguments (line 745)
    kwargs_44657 = {}
    # Getting the type of 'assert_raises' (line 745)
    assert_raises_44649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 745)
    assert_raises_call_result_44658 = invoke(stypy.reporting.localization.Localization(__file__, 745, 4), assert_raises_44649, *[ValueError_44650, odeint_44651, bad2_44652, float_44653, list_44654], **kwargs_44657)
    
    
    # Call to assert_raises(...): (line 747)
    # Processing the call arguments (line 747)
    # Getting the type of 'ZeroDivisionError' (line 747)
    ZeroDivisionError_44660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 18), 'ZeroDivisionError', False)
    # Getting the type of 'odeint' (line 747)
    odeint_44661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 37), 'odeint', False)
    # Getting the type of 'sys1d' (line 747)
    sys1d_44662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 45), 'sys1d', False)
    float_44663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 52), 'float')
    
    # Obtaining an instance of the builtin type 'list' (line 747)
    list_44664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 747)
    # Adding element type (line 747)
    int_44665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 58), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 57), list_44664, int_44665)
    # Adding element type (line 747)
    int_44666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 61), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 57), list_44664, int_44666)
    
    # Processing the call keyword arguments (line 747)
    # Getting the type of 'bad_jac1' (line 747)
    bad_jac1_44667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 70), 'bad_jac1', False)
    keyword_44668 = bad_jac1_44667
    kwargs_44669 = {'Dfun': keyword_44668}
    # Getting the type of 'assert_raises' (line 747)
    assert_raises_44659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 747)
    assert_raises_call_result_44670 = invoke(stypy.reporting.localization.Localization(__file__, 747, 4), assert_raises_44659, *[ZeroDivisionError_44660, odeint_44661, sys1d_44662, float_44663, list_44664], **kwargs_44669)
    
    
    # Call to assert_raises(...): (line 748)
    # Processing the call arguments (line 748)
    # Getting the type of 'ValueError' (line 748)
    ValueError_44672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 18), 'ValueError', False)
    # Getting the type of 'odeint' (line 748)
    odeint_44673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 30), 'odeint', False)
    # Getting the type of 'sys1d' (line 748)
    sys1d_44674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 38), 'sys1d', False)
    float_44675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 45), 'float')
    
    # Obtaining an instance of the builtin type 'list' (line 748)
    list_44676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 748)
    # Adding element type (line 748)
    int_44677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 50), list_44676, int_44677)
    # Adding element type (line 748)
    int_44678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 50), list_44676, int_44678)
    
    # Processing the call keyword arguments (line 748)
    # Getting the type of 'bad_jac2' (line 748)
    bad_jac2_44679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 63), 'bad_jac2', False)
    keyword_44680 = bad_jac2_44679
    kwargs_44681 = {'Dfun': keyword_44680}
    # Getting the type of 'assert_raises' (line 748)
    assert_raises_44671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 748)
    assert_raises_call_result_44682 = invoke(stypy.reporting.localization.Localization(__file__, 748, 4), assert_raises_44671, *[ValueError_44672, odeint_44673, sys1d_44674, float_44675, list_44676], **kwargs_44681)
    
    
    # Call to assert_raises(...): (line 750)
    # Processing the call arguments (line 750)
    # Getting the type of 'ZeroDivisionError' (line 750)
    ZeroDivisionError_44684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 18), 'ZeroDivisionError', False)
    # Getting the type of 'odeint' (line 750)
    odeint_44685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 37), 'odeint', False)
    # Getting the type of 'sys2d' (line 750)
    sys2d_44686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 45), 'sys2d', False)
    
    # Obtaining an instance of the builtin type 'list' (line 750)
    list_44687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 52), 'list')
    # Adding type elements to the builtin type 'list' instance (line 750)
    # Adding element type (line 750)
    float_44688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 53), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 52), list_44687, float_44688)
    # Adding element type (line 750)
    float_44689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 58), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 52), list_44687, float_44689)
    
    
    # Obtaining an instance of the builtin type 'list' (line 750)
    list_44690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 64), 'list')
    # Adding type elements to the builtin type 'list' instance (line 750)
    # Adding element type (line 750)
    int_44691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 65), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 64), list_44690, int_44691)
    # Adding element type (line 750)
    int_44692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 68), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 64), list_44690, int_44692)
    
    # Processing the call keyword arguments (line 750)
    # Getting the type of 'sys2d_bad_jac' (line 751)
    sys2d_bad_jac_44693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 23), 'sys2d_bad_jac', False)
    keyword_44694 = sys2d_bad_jac_44693
    kwargs_44695 = {'Dfun': keyword_44694}
    # Getting the type of 'assert_raises' (line 750)
    assert_raises_44683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 750)
    assert_raises_call_result_44696 = invoke(stypy.reporting.localization.Localization(__file__, 750, 4), assert_raises_44683, *[ZeroDivisionError_44684, odeint_44685, sys2d_44686, list_44687, list_44690], **kwargs_44695)
    
    
    # ################# End of 'test_odeint_errors(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_odeint_errors' in the type store
    # Getting the type of 'stypy_return_type' (line 722)
    stypy_return_type_44697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44697)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_odeint_errors'
    return stypy_return_type_44697

# Assigning a type to the variable 'test_odeint_errors' (line 722)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 0), 'test_odeint_errors', test_odeint_errors)

@norecursion
def test_odeint_bad_shapes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_odeint_bad_shapes'
    module_type_store = module_type_store.open_function_context('test_odeint_bad_shapes', 754, 0, False)
    
    # Passed parameters checking function
    test_odeint_bad_shapes.stypy_localization = localization
    test_odeint_bad_shapes.stypy_type_of_self = None
    test_odeint_bad_shapes.stypy_type_store = module_type_store
    test_odeint_bad_shapes.stypy_function_name = 'test_odeint_bad_shapes'
    test_odeint_bad_shapes.stypy_param_names_list = []
    test_odeint_bad_shapes.stypy_varargs_param_name = None
    test_odeint_bad_shapes.stypy_kwargs_param_name = None
    test_odeint_bad_shapes.stypy_call_defaults = defaults
    test_odeint_bad_shapes.stypy_call_varargs = varargs
    test_odeint_bad_shapes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_odeint_bad_shapes', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_odeint_bad_shapes', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_odeint_bad_shapes(...)' code ##################


    @norecursion
    def badrhs(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'badrhs'
        module_type_store = module_type_store.open_function_context('badrhs', 757, 4, False)
        
        # Passed parameters checking function
        badrhs.stypy_localization = localization
        badrhs.stypy_type_of_self = None
        badrhs.stypy_type_store = module_type_store
        badrhs.stypy_function_name = 'badrhs'
        badrhs.stypy_param_names_list = ['x', 't']
        badrhs.stypy_varargs_param_name = None
        badrhs.stypy_kwargs_param_name = None
        badrhs.stypy_call_defaults = defaults
        badrhs.stypy_call_varargs = varargs
        badrhs.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'badrhs', ['x', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'badrhs', localization, ['x', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'badrhs(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 758)
        list_44698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 758)
        # Adding element type (line 758)
        int_44699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 758, 15), list_44698, int_44699)
        # Adding element type (line 758)
        int_44700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 758, 15), list_44698, int_44700)
        
        # Assigning a type to the variable 'stypy_return_type' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'stypy_return_type', list_44698)
        
        # ################# End of 'badrhs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'badrhs' in the type store
        # Getting the type of 'stypy_return_type' (line 757)
        stypy_return_type_44701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44701)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'badrhs'
        return stypy_return_type_44701

    # Assigning a type to the variable 'badrhs' (line 757)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 4), 'badrhs', badrhs)

    @norecursion
    def sys1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sys1'
        module_type_store = module_type_store.open_function_context('sys1', 760, 4, False)
        
        # Passed parameters checking function
        sys1.stypy_localization = localization
        sys1.stypy_type_of_self = None
        sys1.stypy_type_store = module_type_store
        sys1.stypy_function_name = 'sys1'
        sys1.stypy_param_names_list = ['x', 't']
        sys1.stypy_varargs_param_name = None
        sys1.stypy_kwargs_param_name = None
        sys1.stypy_call_defaults = defaults
        sys1.stypy_call_varargs = varargs
        sys1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'sys1', ['x', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sys1', localization, ['x', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sys1(...)' code ##################

        int_44702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 15), 'int')
        # Getting the type of 'x' (line 761)
        x_44703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 20), 'x')
        # Applying the binary operator '*' (line 761)
        result_mul_44704 = python_operator(stypy.reporting.localization.Localization(__file__, 761, 15), '*', int_44702, x_44703)
        
        # Assigning a type to the variable 'stypy_return_type' (line 761)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'stypy_return_type', result_mul_44704)
        
        # ################# End of 'sys1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sys1' in the type store
        # Getting the type of 'stypy_return_type' (line 760)
        stypy_return_type_44705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44705)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sys1'
        return stypy_return_type_44705

    # Assigning a type to the variable 'sys1' (line 760)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 4), 'sys1', sys1)

    @norecursion
    def badjac(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'badjac'
        module_type_store = module_type_store.open_function_context('badjac', 763, 4, False)
        
        # Passed parameters checking function
        badjac.stypy_localization = localization
        badjac.stypy_type_of_self = None
        badjac.stypy_type_store = module_type_store
        badjac.stypy_function_name = 'badjac'
        badjac.stypy_param_names_list = ['x', 't']
        badjac.stypy_varargs_param_name = None
        badjac.stypy_kwargs_param_name = None
        badjac.stypy_call_defaults = defaults
        badjac.stypy_call_varargs = varargs
        badjac.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'badjac', ['x', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'badjac', localization, ['x', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'badjac(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 764)
        list_44706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 764)
        # Adding element type (line 764)
        
        # Obtaining an instance of the builtin type 'list' (line 764)
        list_44707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 764)
        # Adding element type (line 764)
        int_44708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 764, 16), list_44707, int_44708)
        # Adding element type (line 764)
        int_44709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 764, 16), list_44707, int_44709)
        # Adding element type (line 764)
        int_44710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 764, 16), list_44707, int_44710)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 764, 15), list_44706, list_44707)
        
        # Assigning a type to the variable 'stypy_return_type' (line 764)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 8), 'stypy_return_type', list_44706)
        
        # ################# End of 'badjac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'badjac' in the type store
        # Getting the type of 'stypy_return_type' (line 763)
        stypy_return_type_44711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44711)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'badjac'
        return stypy_return_type_44711

    # Assigning a type to the variable 'badjac' (line 763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 4), 'badjac', badjac)
    
    # Assigning a List to a Name (line 767):
    
    # Assigning a List to a Name (line 767):
    
    # Obtaining an instance of the builtin type 'list' (line 767)
    list_44712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 767)
    # Adding element type (line 767)
    
    # Obtaining an instance of the builtin type 'list' (line 767)
    list_44713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 767)
    # Adding element type (line 767)
    int_44714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 14), list_44713, int_44714)
    # Adding element type (line 767)
    int_44715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 14), list_44713, int_44715)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 13), list_44712, list_44713)
    # Adding element type (line 767)
    
    # Obtaining an instance of the builtin type 'list' (line 767)
    list_44716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 767)
    # Adding element type (line 767)
    int_44717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 22), list_44716, int_44717)
    # Adding element type (line 767)
    int_44718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 22), list_44716, int_44718)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 13), list_44712, list_44716)
    
    # Assigning a type to the variable 'bad_y0' (line 767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 4), 'bad_y0', list_44712)
    
    # Call to assert_raises(...): (line 768)
    # Processing the call arguments (line 768)
    # Getting the type of 'ValueError' (line 768)
    ValueError_44720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 18), 'ValueError', False)
    # Getting the type of 'odeint' (line 768)
    odeint_44721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 30), 'odeint', False)
    # Getting the type of 'sys1' (line 768)
    sys1_44722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 38), 'sys1', False)
    # Getting the type of 'bad_y0' (line 768)
    bad_y0_44723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 44), 'bad_y0', False)
    
    # Obtaining an instance of the builtin type 'list' (line 768)
    list_44724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 52), 'list')
    # Adding type elements to the builtin type 'list' instance (line 768)
    # Adding element type (line 768)
    int_44725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 768, 52), list_44724, int_44725)
    # Adding element type (line 768)
    int_44726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 56), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 768, 52), list_44724, int_44726)
    
    # Processing the call keyword arguments (line 768)
    kwargs_44727 = {}
    # Getting the type of 'assert_raises' (line 768)
    assert_raises_44719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 768)
    assert_raises_call_result_44728 = invoke(stypy.reporting.localization.Localization(__file__, 768, 4), assert_raises_44719, *[ValueError_44720, odeint_44721, sys1_44722, bad_y0_44723, list_44724], **kwargs_44727)
    
    
    # Assigning a List to a Name (line 771):
    
    # Assigning a List to a Name (line 771):
    
    # Obtaining an instance of the builtin type 'list' (line 771)
    list_44729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 771)
    # Adding element type (line 771)
    
    # Obtaining an instance of the builtin type 'list' (line 771)
    list_44730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 771)
    # Adding element type (line 771)
    int_44731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 771, 13), list_44730, int_44731)
    # Adding element type (line 771)
    int_44732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 771, 13), list_44730, int_44732)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 771, 12), list_44729, list_44730)
    # Adding element type (line 771)
    
    # Obtaining an instance of the builtin type 'list' (line 771)
    list_44733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 771)
    # Adding element type (line 771)
    int_44734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 771, 21), list_44733, int_44734)
    # Adding element type (line 771)
    int_44735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 771, 21), list_44733, int_44735)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 771, 12), list_44729, list_44733)
    
    # Assigning a type to the variable 'bad_t' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'bad_t', list_44729)
    
    # Call to assert_raises(...): (line 772)
    # Processing the call arguments (line 772)
    # Getting the type of 'ValueError' (line 772)
    ValueError_44737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 18), 'ValueError', False)
    # Getting the type of 'odeint' (line 772)
    odeint_44738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 30), 'odeint', False)
    # Getting the type of 'sys1' (line 772)
    sys1_44739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 38), 'sys1', False)
    
    # Obtaining an instance of the builtin type 'list' (line 772)
    list_44740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 772)
    # Adding element type (line 772)
    float_44741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 45), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 772, 44), list_44740, float_44741)
    
    # Getting the type of 'bad_t' (line 772)
    bad_t_44742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 52), 'bad_t', False)
    # Processing the call keyword arguments (line 772)
    kwargs_44743 = {}
    # Getting the type of 'assert_raises' (line 772)
    assert_raises_44736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 772)
    assert_raises_call_result_44744 = invoke(stypy.reporting.localization.Localization(__file__, 772, 4), assert_raises_44736, *[ValueError_44737, odeint_44738, sys1_44739, list_44740, bad_t_44742], **kwargs_44743)
    
    
    # Call to assert_raises(...): (line 775)
    # Processing the call arguments (line 775)
    # Getting the type of 'RuntimeError' (line 775)
    RuntimeError_44746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 18), 'RuntimeError', False)
    # Getting the type of 'odeint' (line 775)
    odeint_44747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 32), 'odeint', False)
    # Getting the type of 'badrhs' (line 775)
    badrhs_44748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 40), 'badrhs', False)
    int_44749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 48), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 775)
    list_44750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 52), 'list')
    # Adding type elements to the builtin type 'list' instance (line 775)
    # Adding element type (line 775)
    int_44751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 52), list_44750, int_44751)
    # Adding element type (line 775)
    int_44752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 56), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 52), list_44750, int_44752)
    
    # Processing the call keyword arguments (line 775)
    kwargs_44753 = {}
    # Getting the type of 'assert_raises' (line 775)
    assert_raises_44745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 775)
    assert_raises_call_result_44754 = invoke(stypy.reporting.localization.Localization(__file__, 775, 4), assert_raises_44745, *[RuntimeError_44746, odeint_44747, badrhs_44748, int_44749, list_44750], **kwargs_44753)
    
    
    # Call to assert_raises(...): (line 778)
    # Processing the call arguments (line 778)
    # Getting the type of 'RuntimeError' (line 778)
    RuntimeError_44756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 18), 'RuntimeError', False)
    # Getting the type of 'odeint' (line 778)
    odeint_44757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 32), 'odeint', False)
    # Getting the type of 'sys1' (line 778)
    sys1_44758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 40), 'sys1', False)
    
    # Obtaining an instance of the builtin type 'list' (line 778)
    list_44759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 778)
    # Adding element type (line 778)
    int_44760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 46), list_44759, int_44760)
    # Adding element type (line 778)
    int_44761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 46), list_44759, int_44761)
    
    
    # Obtaining an instance of the builtin type 'list' (line 778)
    list_44762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 56), 'list')
    # Adding type elements to the builtin type 'list' instance (line 778)
    # Adding element type (line 778)
    int_44763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 56), list_44762, int_44763)
    # Adding element type (line 778)
    int_44764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 56), list_44762, int_44764)
    
    # Processing the call keyword arguments (line 778)
    # Getting the type of 'badjac' (line 778)
    badjac_44765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 69), 'badjac', False)
    keyword_44766 = badjac_44765
    kwargs_44767 = {'Dfun': keyword_44766}
    # Getting the type of 'assert_raises' (line 778)
    assert_raises_44755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 778)
    assert_raises_call_result_44768 = invoke(stypy.reporting.localization.Localization(__file__, 778, 4), assert_raises_44755, *[RuntimeError_44756, odeint_44757, sys1_44758, list_44759, list_44762], **kwargs_44767)
    
    
    # ################# End of 'test_odeint_bad_shapes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_odeint_bad_shapes' in the type store
    # Getting the type of 'stypy_return_type' (line 754)
    stypy_return_type_44769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44769)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_odeint_bad_shapes'
    return stypy_return_type_44769

# Assigning a type to the variable 'test_odeint_bad_shapes' (line 754)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 0), 'test_odeint_bad_shapes', test_odeint_bad_shapes)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
