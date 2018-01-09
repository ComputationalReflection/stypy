
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Authors: Pearu Peterson, Pauli Virtanen, John Travers
2: '''
3: First-order ODE integrators.
4: 
5: User-friendly interface to various numerical integrators for solving a
6: system of first order ODEs with prescribed initial conditions::
7: 
8:     d y(t)[i]
9:     ---------  = f(t,y(t))[i],
10:        d t
11: 
12:     y(t=0)[i] = y0[i],
13: 
14: where::
15: 
16:     i = 0, ..., len(y0) - 1
17: 
18: class ode
19: ---------
20: 
21: A generic interface class to numeric integrators. It has the following
22: methods::
23: 
24:     integrator = ode(f, jac=None)
25:     integrator = integrator.set_integrator(name, **params)
26:     integrator = integrator.set_initial_value(y0, t0=0.0)
27:     integrator = integrator.set_f_params(*args)
28:     integrator = integrator.set_jac_params(*args)
29:     y1 = integrator.integrate(t1, step=False, relax=False)
30:     flag = integrator.successful()
31: 
32: class complex_ode
33: -----------------
34: 
35: This class has the same generic interface as ode, except it can handle complex
36: f, y and Jacobians by transparently translating them into the equivalent
37: real valued system. It supports the real valued solvers (i.e not zvode) and is
38: an alternative to ode with the zvode solver, sometimes performing better.
39: '''
40: from __future__ import division, print_function, absolute_import
41: 
42: # XXX: Integrators must have:
43: # ===========================
44: # cvode - C version of vode and vodpk with many improvements.
45: #   Get it from http://www.netlib.org/ode/cvode.tar.gz
46: #   To wrap cvode to Python, one must write extension module by
47: #   hand. Its interface is too much 'advanced C' that using f2py
48: #   would be too complicated (or impossible).
49: #
50: # How to define a new integrator:
51: # ===============================
52: #
53: # class myodeint(IntegratorBase):
54: #
55: #     runner = <odeint function> or None
56: #
57: #     def __init__(self,...):                           # required
58: #         <initialize>
59: #
60: #     def reset(self,n,has_jac):                        # optional
61: #         # n - the size of the problem (number of equations)
62: #         # has_jac - whether user has supplied its own routine for Jacobian
63: #         <allocate memory,initialize further>
64: #
65: #     def run(self,f,jac,y0,t0,t1,f_params,jac_params): # required
66: #         # this method is called to integrate from t=t0 to t=t1
67: #         # with initial condition y0. f and jac are user-supplied functions
68: #         # that define the problem. f_params,jac_params are additional
69: #         # arguments
70: #         # to these functions.
71: #         <calculate y1>
72: #         if <calculation was unsuccesful>:
73: #             self.success = 0
74: #         return t1,y1
75: #
76: #     # In addition, one can define step() and run_relax() methods (they
77: #     # take the same arguments as run()) if the integrator can support
78: #     # these features (see IntegratorBase doc strings).
79: #
80: # if myodeint.runner:
81: #     IntegratorBase.integrator_classes.append(myodeint)
82: 
83: __all__ = ['ode', 'complex_ode']
84: __version__ = "$Id$"
85: __docformat__ = "restructuredtext en"
86: 
87: import re
88: import warnings
89: 
90: from numpy import asarray, array, zeros, int32, isscalar, real, imag, vstack
91: 
92: from . import vode as _vode
93: from . import _dop
94: from . import lsoda as _lsoda
95: 
96: 
97: # ------------------------------------------------------------------------------
98: # User interface
99: # ------------------------------------------------------------------------------
100: 
101: 
102: class ode(object):
103:     '''
104:     A generic interface class to numeric integrators.
105: 
106:     Solve an equation system :math:`y'(t) = f(t,y)` with (optional) ``jac = df/dy``.
107: 
108:     *Note*: The first two arguments of ``f(t, y, ...)`` are in the
109:     opposite order of the arguments in the system definition function used
110:     by `scipy.integrate.odeint`.
111: 
112:     Parameters
113:     ----------
114:     f : callable ``f(t, y, *f_args)``
115:         Right-hand side of the differential equation. t is a scalar,
116:         ``y.shape == (n,)``.
117:         ``f_args`` is set by calling ``set_f_params(*args)``.
118:         `f` should return a scalar, array or list (not a tuple).
119:     jac : callable ``jac(t, y, *jac_args)``, optional
120:         Jacobian of the right-hand side, ``jac[i,j] = d f[i] / d y[j]``.
121:         ``jac_args`` is set by calling ``set_jac_params(*args)``.
122: 
123:     Attributes
124:     ----------
125:     t : float
126:         Current time.
127:     y : ndarray
128:         Current variable values.
129: 
130:     See also
131:     --------
132:     odeint : an integrator with a simpler interface based on lsoda from ODEPACK
133:     quad : for finding the area under a curve
134: 
135:     Notes
136:     -----
137:     Available integrators are listed below. They can be selected using
138:     the `set_integrator` method.
139: 
140:     "vode"
141: 
142:         Real-valued Variable-coefficient Ordinary Differential Equation
143:         solver, with fixed-leading-coefficient implementation. It provides
144:         implicit Adams method (for non-stiff problems) and a method based on
145:         backward differentiation formulas (BDF) (for stiff problems).
146: 
147:         Source: http://www.netlib.org/ode/vode.f
148: 
149:         .. warning::
150: 
151:            This integrator is not re-entrant. You cannot have two `ode`
152:            instances using the "vode" integrator at the same time.
153: 
154:         This integrator accepts the following parameters in `set_integrator`
155:         method of the `ode` class:
156: 
157:         - atol : float or sequence
158:           absolute tolerance for solution
159:         - rtol : float or sequence
160:           relative tolerance for solution
161:         - lband : None or int
162:         - uband : None or int
163:           Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+uband.
164:           Setting these requires your jac routine to return the jacobian
165:           in packed format, jac_packed[i-j+uband, j] = jac[i,j]. The
166:           dimension of the matrix must be (lband+uband+1, len(y)).
167:         - method: 'adams' or 'bdf'
168:           Which solver to use, Adams (non-stiff) or BDF (stiff)
169:         - with_jacobian : bool
170:           This option is only considered when the user has not supplied a
171:           Jacobian function and has not indicated (by setting either band)
172:           that the Jacobian is banded.  In this case, `with_jacobian` specifies
173:           whether the iteration method of the ODE solver's correction step is
174:           chord iteration with an internally generated full Jacobian or
175:           functional iteration with no Jacobian.
176:         - nsteps : int
177:           Maximum number of (internally defined) steps allowed during one
178:           call to the solver.
179:         - first_step : float
180:         - min_step : float
181:         - max_step : float
182:           Limits for the step sizes used by the integrator.
183:         - order : int
184:           Maximum order used by the integrator,
185:           order <= 12 for Adams, <= 5 for BDF.
186: 
187:     "zvode"
188: 
189:         Complex-valued Variable-coefficient Ordinary Differential Equation
190:         solver, with fixed-leading-coefficient implementation.  It provides
191:         implicit Adams method (for non-stiff problems) and a method based on
192:         backward differentiation formulas (BDF) (for stiff problems).
193: 
194:         Source: http://www.netlib.org/ode/zvode.f
195: 
196:         .. warning::
197: 
198:            This integrator is not re-entrant. You cannot have two `ode`
199:            instances using the "zvode" integrator at the same time.
200: 
201:         This integrator accepts the same parameters in `set_integrator`
202:         as the "vode" solver.
203: 
204:         .. note::
205: 
206:             When using ZVODE for a stiff system, it should only be used for
207:             the case in which the function f is analytic, that is, when each f(i)
208:             is an analytic function of each y(j).  Analyticity means that the
209:             partial derivative df(i)/dy(j) is a unique complex number, and this
210:             fact is critical in the way ZVODE solves the dense or banded linear
211:             systems that arise in the stiff case.  For a complex stiff ODE system
212:             in which f is not analytic, ZVODE is likely to have convergence
213:             failures, and for this problem one should instead use DVODE on the
214:             equivalent real system (in the real and imaginary parts of y).
215: 
216:     "lsoda"
217: 
218:         Real-valued Variable-coefficient Ordinary Differential Equation
219:         solver, with fixed-leading-coefficient implementation. It provides
220:         automatic method switching between implicit Adams method (for non-stiff
221:         problems) and a method based on backward differentiation formulas (BDF)
222:         (for stiff problems).
223: 
224:         Source: http://www.netlib.org/odepack
225: 
226:         .. warning::
227: 
228:            This integrator is not re-entrant. You cannot have two `ode`
229:            instances using the "lsoda" integrator at the same time.
230: 
231:         This integrator accepts the following parameters in `set_integrator`
232:         method of the `ode` class:
233: 
234:         - atol : float or sequence
235:           absolute tolerance for solution
236:         - rtol : float or sequence
237:           relative tolerance for solution
238:         - lband : None or int
239:         - uband : None or int
240:           Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+uband.
241:           Setting these requires your jac routine to return the jacobian
242:           in packed format, jac_packed[i-j+uband, j] = jac[i,j].
243:         - with_jacobian : bool
244:           *Not used.*
245:         - nsteps : int
246:           Maximum number of (internally defined) steps allowed during one
247:           call to the solver.
248:         - first_step : float
249:         - min_step : float
250:         - max_step : float
251:           Limits for the step sizes used by the integrator.
252:         - max_order_ns : int
253:           Maximum order used in the nonstiff case (default 12).
254:         - max_order_s : int
255:           Maximum order used in the stiff case (default 5).
256:         - max_hnil : int
257:           Maximum number of messages reporting too small step size (t + h = t)
258:           (default 0)
259:         - ixpr : int
260:           Whether to generate extra printing at method switches (default False).
261: 
262:     "dopri5"
263: 
264:         This is an explicit runge-kutta method of order (4)5 due to Dormand &
265:         Prince (with stepsize control and dense output).
266: 
267:         Authors:
268: 
269:             E. Hairer and G. Wanner
270:             Universite de Geneve, Dept. de Mathematiques
271:             CH-1211 Geneve 24, Switzerland
272:             e-mail:  ernst.hairer@math.unige.ch, gerhard.wanner@math.unige.ch
273: 
274:         This code is described in [HNW93]_.
275: 
276:         This integrator accepts the following parameters in set_integrator()
277:         method of the ode class:
278: 
279:         - atol : float or sequence
280:           absolute tolerance for solution
281:         - rtol : float or sequence
282:           relative tolerance for solution
283:         - nsteps : int
284:           Maximum number of (internally defined) steps allowed during one
285:           call to the solver.
286:         - first_step : float
287:         - max_step : float
288:         - safety : float
289:           Safety factor on new step selection (default 0.9)
290:         - ifactor : float
291:         - dfactor : float
292:           Maximum factor to increase/decrease step size by in one step
293:         - beta : float
294:           Beta parameter for stabilised step size control.
295:         - verbosity : int
296:           Switch for printing messages (< 0 for no messages).
297: 
298:     "dop853"
299: 
300:         This is an explicit runge-kutta method of order 8(5,3) due to Dormand
301:         & Prince (with stepsize control and dense output).
302: 
303:         Options and references the same as "dopri5".
304: 
305:     Examples
306:     --------
307: 
308:     A problem to integrate and the corresponding jacobian:
309: 
310:     >>> from scipy.integrate import ode
311:     >>>
312:     >>> y0, t0 = [1.0j, 2.0], 0
313:     >>>
314:     >>> def f(t, y, arg1):
315:     ...     return [1j*arg1*y[0] + y[1], -arg1*y[1]**2]
316:     >>> def jac(t, y, arg1):
317:     ...     return [[1j*arg1, 1], [0, -arg1*2*y[1]]]
318: 
319:     The integration:
320: 
321:     >>> r = ode(f, jac).set_integrator('zvode', method='bdf')
322:     >>> r.set_initial_value(y0, t0).set_f_params(2.0).set_jac_params(2.0)
323:     >>> t1 = 10
324:     >>> dt = 1
325:     >>> while r.successful() and r.t < t1:
326:     ...     print(r.t+dt, r.integrate(r.t+dt))
327:     1 [-0.71038232+0.23749653j  0.40000271+0.j        ]
328:     2.0 [ 0.19098503-0.52359246j  0.22222356+0.j        ]
329:     3.0 [ 0.47153208+0.52701229j  0.15384681+0.j        ]
330:     4.0 [-0.61905937+0.30726255j  0.11764744+0.j        ]
331:     5.0 [ 0.02340997-0.61418799j  0.09523835+0.j        ]
332:     6.0 [ 0.58643071+0.339819j  0.08000018+0.j      ]
333:     7.0 [-0.52070105+0.44525141j  0.06896565+0.j        ]
334:     8.0 [-0.15986733-0.61234476j  0.06060616+0.j        ]
335:     9.0 [ 0.64850462+0.15048982j  0.05405414+0.j        ]
336:     10.0 [-0.38404699+0.56382299j  0.04878055+0.j        ]
337: 
338:     References
339:     ----------
340:     .. [HNW93] E. Hairer, S.P. Norsett and G. Wanner, Solving Ordinary
341:         Differential Equations i. Nonstiff Problems. 2nd edition.
342:         Springer Series in Computational Mathematics,
343:         Springer-Verlag (1993)
344: 
345:     '''
346: 
347:     def __init__(self, f, jac=None):
348:         self.stiff = 0
349:         self.f = f
350:         self.jac = jac
351:         self.f_params = ()
352:         self.jac_params = ()
353:         self._y = []
354: 
355:     @property
356:     def y(self):
357:         return self._y
358: 
359:     def set_initial_value(self, y, t=0.0):
360:         '''Set initial conditions y(t) = y.'''
361:         if isscalar(y):
362:             y = [y]
363:         n_prev = len(self._y)
364:         if not n_prev:
365:             self.set_integrator('')  # find first available integrator
366:         self._y = asarray(y, self._integrator.scalar)
367:         self.t = t
368:         self._integrator.reset(len(self._y), self.jac is not None)
369:         return self
370: 
371:     def set_integrator(self, name, **integrator_params):
372:         '''
373:         Set integrator by name.
374: 
375:         Parameters
376:         ----------
377:         name : str
378:             Name of the integrator.
379:         integrator_params
380:             Additional parameters for the integrator.
381:         '''
382:         integrator = find_integrator(name)
383:         if integrator is None:
384:             # FIXME: this really should be raise an exception. Will that break
385:             # any code?
386:             warnings.warn('No integrator name match with %r or is not '
387:                           'available.' % name)
388:         else:
389:             self._integrator = integrator(**integrator_params)
390:             if not len(self._y):
391:                 self.t = 0.0
392:                 self._y = array([0.0], self._integrator.scalar)
393:             self._integrator.reset(len(self._y), self.jac is not None)
394:         return self
395: 
396:     def integrate(self, t, step=False, relax=False):
397:         '''Find y=y(t), set y as an initial condition, and return y.
398: 
399:         Parameters
400:         ----------
401:         t : float
402:             The endpoint of the integration step.
403:         step : bool
404:             If True, and if the integrator supports the step method,
405:             then perform a single integration step and return.
406:             This parameter is provided in order to expose internals of
407:             the implementation, and should not be changed from its default
408:             value in most cases.
409:         relax : bool
410:             If True and if the integrator supports the run_relax method,
411:             then integrate until t_1 >= t and return. ``relax`` is not
412:             referenced if ``step=True``.
413:             This parameter is provided in order to expose internals of
414:             the implementation, and should not be changed from its default
415:             value in most cases.
416: 
417:         Returns
418:         -------
419:         y : float
420:             The integrated value at t
421:         '''
422:         if step and self._integrator.supports_step:
423:             mth = self._integrator.step
424:         elif relax and self._integrator.supports_run_relax:
425:             mth = self._integrator.run_relax
426:         else:
427:             mth = self._integrator.run
428: 
429:         try:
430:             self._y, self.t = mth(self.f, self.jac or (lambda: None),
431:                                   self._y, self.t, t,
432:                                   self.f_params, self.jac_params)
433:         except SystemError:
434:             # f2py issue with tuple returns, see ticket 1187.
435:             raise ValueError('Function to integrate must not return a tuple.')
436: 
437:         return self._y
438: 
439:     def successful(self):
440:         '''Check if integration was successful.'''
441:         try:
442:             self._integrator
443:         except AttributeError:
444:             self.set_integrator('')
445:         return self._integrator.success == 1
446: 
447:     def get_return_code(self):
448:         '''Extracts the return code for the integration to enable better control
449:         if the integration fails.'''
450:         try:
451:             self._integrator
452:         except AttributeError:
453:             self.set_integrator('')
454:         return self._integrator.istate
455: 
456:     def set_f_params(self, *args):
457:         '''Set extra parameters for user-supplied function f.'''
458:         self.f_params = args
459:         return self
460: 
461:     def set_jac_params(self, *args):
462:         '''Set extra parameters for user-supplied function jac.'''
463:         self.jac_params = args
464:         return self
465: 
466:     def set_solout(self, solout):
467:         '''
468:         Set callable to be called at every successful integration step.
469: 
470:         Parameters
471:         ----------
472:         solout : callable
473:             ``solout(t, y)`` is called at each internal integrator step,
474:             t is a scalar providing the current independent position
475:             y is the current soloution ``y.shape == (n,)``
476:             solout should return -1 to stop integration
477:             otherwise it should return None or 0
478: 
479:         '''
480:         if self._integrator.supports_solout:
481:             self._integrator.set_solout(solout)
482:             if self._y is not None:
483:                 self._integrator.reset(len(self._y), self.jac is not None)
484:         else:
485:             raise ValueError("selected integrator does not support solout,"
486:                              " choose another one")
487: 
488: 
489: def _transform_banded_jac(bjac):
490:     '''
491:     Convert a real matrix of the form (for example)
492: 
493:         [0 0 A B]        [0 0 0 B]
494:         [0 0 C D]        [0 0 A D]
495:         [E F G H]   to   [0 F C H]
496:         [I J K L]        [E J G L]
497:                          [I 0 K 0]
498: 
499:     That is, every other column is shifted up one.
500:     '''
501:     # Shift every other column.
502:     newjac = zeros((bjac.shape[0] + 1, bjac.shape[1]))
503:     newjac[1:, ::2] = bjac[:, ::2]
504:     newjac[:-1, 1::2] = bjac[:, 1::2]
505:     return newjac
506: 
507: 
508: class complex_ode(ode):
509:     '''
510:     A wrapper of ode for complex systems.
511: 
512:     This functions similarly as `ode`, but re-maps a complex-valued
513:     equation system to a real-valued one before using the integrators.
514: 
515:     Parameters
516:     ----------
517:     f : callable ``f(t, y, *f_args)``
518:         Rhs of the equation. t is a scalar, ``y.shape == (n,)``.
519:         ``f_args`` is set by calling ``set_f_params(*args)``.
520:     jac : callable ``jac(t, y, *jac_args)``
521:         Jacobian of the rhs, ``jac[i,j] = d f[i] / d y[j]``.
522:         ``jac_args`` is set by calling ``set_f_params(*args)``.
523: 
524:     Attributes
525:     ----------
526:     t : float
527:         Current time.
528:     y : ndarray
529:         Current variable values.
530: 
531:     Examples
532:     --------
533:     For usage examples, see `ode`.
534: 
535:     '''
536: 
537:     def __init__(self, f, jac=None):
538:         self.cf = f
539:         self.cjac = jac
540:         if jac is None:
541:             ode.__init__(self, self._wrap, None)
542:         else:
543:             ode.__init__(self, self._wrap, self._wrap_jac)
544: 
545:     def _wrap(self, t, y, *f_args):
546:         f = self.cf(*((t, y[::2] + 1j * y[1::2]) + f_args))
547:         # self.tmp is a real-valued array containing the interleaved
548:         # real and imaginary parts of f.
549:         self.tmp[::2] = real(f)
550:         self.tmp[1::2] = imag(f)
551:         return self.tmp
552: 
553:     def _wrap_jac(self, t, y, *jac_args):
554:         # jac is the complex Jacobian computed by the user-defined function.
555:         jac = self.cjac(*((t, y[::2] + 1j * y[1::2]) + jac_args))
556: 
557:         # jac_tmp is the real version of the complex Jacobian.  Each complex
558:         # entry in jac, say 2+3j, becomes a 2x2 block of the form
559:         #     [2 -3]
560:         #     [3  2]
561:         jac_tmp = zeros((2 * jac.shape[0], 2 * jac.shape[1]))
562:         jac_tmp[1::2, 1::2] = jac_tmp[::2, ::2] = real(jac)
563:         jac_tmp[1::2, ::2] = imag(jac)
564:         jac_tmp[::2, 1::2] = -jac_tmp[1::2, ::2]
565: 
566:         ml = getattr(self._integrator, 'ml', None)
567:         mu = getattr(self._integrator, 'mu', None)
568:         if ml is not None or mu is not None:
569:             # Jacobian is banded.  The user's Jacobian function has computed
570:             # the complex Jacobian in packed format.  The corresponding
571:             # real-valued version has every other column shifted up.
572:             jac_tmp = _transform_banded_jac(jac_tmp)
573: 
574:         return jac_tmp
575: 
576:     @property
577:     def y(self):
578:         return self._y[::2] + 1j * self._y[1::2]
579: 
580:     def set_integrator(self, name, **integrator_params):
581:         '''
582:         Set integrator by name.
583: 
584:         Parameters
585:         ----------
586:         name : str
587:             Name of the integrator
588:         integrator_params
589:             Additional parameters for the integrator.
590:         '''
591:         if name == 'zvode':
592:             raise ValueError("zvode must be used with ode, not complex_ode")
593: 
594:         lband = integrator_params.get('lband')
595:         uband = integrator_params.get('uband')
596:         if lband is not None or uband is not None:
597:             # The Jacobian is banded.  Override the user-supplied bandwidths
598:             # (which are for the complex Jacobian) with the bandwidths of
599:             # the corresponding real-valued Jacobian wrapper of the complex
600:             # Jacobian.
601:             integrator_params['lband'] = 2 * (lband or 0) + 1
602:             integrator_params['uband'] = 2 * (uband or 0) + 1
603: 
604:         return ode.set_integrator(self, name, **integrator_params)
605: 
606:     def set_initial_value(self, y, t=0.0):
607:         '''Set initial conditions y(t) = y.'''
608:         y = asarray(y)
609:         self.tmp = zeros(y.size * 2, 'float')
610:         self.tmp[::2] = real(y)
611:         self.tmp[1::2] = imag(y)
612:         return ode.set_initial_value(self, self.tmp, t)
613: 
614:     def integrate(self, t, step=False, relax=False):
615:         '''Find y=y(t), set y as an initial condition, and return y.
616: 
617:         Parameters
618:         ----------
619:         t : float
620:             The endpoint of the integration step.
621:         step : bool
622:             If True, and if the integrator supports the step method,
623:             then perform a single integration step and return.
624:             This parameter is provided in order to expose internals of
625:             the implementation, and should not be changed from its default
626:             value in most cases.
627:         relax : bool
628:             If True and if the integrator supports the run_relax method,
629:             then integrate until t_1 >= t and return. ``relax`` is not
630:             referenced if ``step=True``.
631:             This parameter is provided in order to expose internals of
632:             the implementation, and should not be changed from its default
633:             value in most cases.
634: 
635:         Returns
636:         -------
637:         y : float
638:             The integrated value at t
639:         '''
640:         y = ode.integrate(self, t, step, relax)
641:         return y[::2] + 1j * y[1::2]
642: 
643:     def set_solout(self, solout):
644:         '''
645:         Set callable to be called at every successful integration step.
646: 
647:         Parameters
648:         ----------
649:         solout : callable
650:             ``solout(t, y)`` is called at each internal integrator step,
651:             t is a scalar providing the current independent position
652:             y is the current soloution ``y.shape == (n,)``
653:             solout should return -1 to stop integration
654:             otherwise it should return None or 0
655: 
656:         '''
657:         if self._integrator.supports_solout:
658:             self._integrator.set_solout(solout, complex=True)
659:         else:
660:             raise TypeError("selected integrator does not support solouta,"
661:                             + "choose another one")
662: 
663: 
664: # ------------------------------------------------------------------------------
665: # ODE integrators
666: # ------------------------------------------------------------------------------
667: 
668: def find_integrator(name):
669:     for cl in IntegratorBase.integrator_classes:
670:         if re.match(name, cl.__name__, re.I):
671:             return cl
672:     return None
673: 
674: 
675: class IntegratorConcurrencyError(RuntimeError):
676:     '''
677:     Failure due to concurrent usage of an integrator that can be used
678:     only for a single problem at a time.
679: 
680:     '''
681: 
682:     def __init__(self, name):
683:         msg = ("Integrator `%s` can be used to solve only a single problem "
684:                "at a time. If you want to integrate multiple problems, "
685:                "consider using a different integrator "
686:                "(see `ode.set_integrator`)") % name
687:         RuntimeError.__init__(self, msg)
688: 
689: 
690: class IntegratorBase(object):
691:     runner = None  # runner is None => integrator is not available
692:     success = None  # success==1 if integrator was called successfully
693:     istate = None  # istate > 0 means success, istate < 0 means failure
694:     supports_run_relax = None
695:     supports_step = None
696:     supports_solout = False
697:     integrator_classes = []
698:     scalar = float
699: 
700:     def acquire_new_handle(self):
701:         # Some of the integrators have internal state (ancient
702:         # Fortran...), and so only one instance can use them at a time.
703:         # We keep track of this, and fail when concurrent usage is tried.
704:         self.__class__.active_global_handle += 1
705:         self.handle = self.__class__.active_global_handle
706: 
707:     def check_handle(self):
708:         if self.handle is not self.__class__.active_global_handle:
709:             raise IntegratorConcurrencyError(self.__class__.__name__)
710: 
711:     def reset(self, n, has_jac):
712:         '''Prepare integrator for call: allocate memory, set flags, etc.
713:         n - number of equations.
714:         has_jac - if user has supplied function for evaluating Jacobian.
715:         '''
716: 
717:     def run(self, f, jac, y0, t0, t1, f_params, jac_params):
718:         '''Integrate from t=t0 to t=t1 using y0 as an initial condition.
719:         Return 2-tuple (y1,t1) where y1 is the result and t=t1
720:         defines the stoppage coordinate of the result.
721:         '''
722:         raise NotImplementedError('all integrators must define '
723:                                   'run(f, jac, t0, t1, y0, f_params, jac_params)')
724: 
725:     def step(self, f, jac, y0, t0, t1, f_params, jac_params):
726:         '''Make one integration step and return (y1,t1).'''
727:         raise NotImplementedError('%s does not support step() method' %
728:                                   self.__class__.__name__)
729: 
730:     def run_relax(self, f, jac, y0, t0, t1, f_params, jac_params):
731:         '''Integrate from t=t0 to t>=t1 and return (y1,t).'''
732:         raise NotImplementedError('%s does not support run_relax() method' %
733:                                   self.__class__.__name__)
734: 
735:     # XXX: __str__ method for getting visual state of the integrator
736: 
737: 
738: def _vode_banded_jac_wrapper(jacfunc, ml, jac_params):
739:     '''
740:     Wrap a banded Jacobian function with a function that pads
741:     the Jacobian with `ml` rows of zeros.
742:     '''
743: 
744:     def jac_wrapper(t, y):
745:         jac = asarray(jacfunc(t, y, *jac_params))
746:         padded_jac = vstack((jac, zeros((ml, jac.shape[1]))))
747:         return padded_jac
748: 
749:     return jac_wrapper
750: 
751: 
752: class vode(IntegratorBase):
753:     runner = getattr(_vode, 'dvode', None)
754: 
755:     messages = {-1: 'Excess work done on this call. (Perhaps wrong MF.)',
756:                 -2: 'Excess accuracy requested. (Tolerances too small.)',
757:                 -3: 'Illegal input detected. (See printed message.)',
758:                 -4: 'Repeated error test failures. (Check all input.)',
759:                 -5: 'Repeated convergence failures. (Perhaps bad'
760:                     ' Jacobian supplied or wrong choice of MF or tolerances.)',
761:                 -6: 'Error weight became zero during problem. (Solution'
762:                     ' component i vanished, and ATOL or ATOL(i) = 0.)'
763:                 }
764:     supports_run_relax = 1
765:     supports_step = 1
766:     active_global_handle = 0
767: 
768:     def __init__(self,
769:                  method='adams',
770:                  with_jacobian=False,
771:                  rtol=1e-6, atol=1e-12,
772:                  lband=None, uband=None,
773:                  order=12,
774:                  nsteps=500,
775:                  max_step=0.0,  # corresponds to infinite
776:                  min_step=0.0,
777:                  first_step=0.0,  # determined by solver
778:                  ):
779: 
780:         if re.match(method, r'adams', re.I):
781:             self.meth = 1
782:         elif re.match(method, r'bdf', re.I):
783:             self.meth = 2
784:         else:
785:             raise ValueError('Unknown integration method %s' % method)
786:         self.with_jacobian = with_jacobian
787:         self.rtol = rtol
788:         self.atol = atol
789:         self.mu = uband
790:         self.ml = lband
791: 
792:         self.order = order
793:         self.nsteps = nsteps
794:         self.max_step = max_step
795:         self.min_step = min_step
796:         self.first_step = first_step
797:         self.success = 1
798: 
799:         self.initialized = False
800: 
801:     def _determine_mf_and_set_bands(self, has_jac):
802:         '''
803:         Determine the `MF` parameter (Method Flag) for the Fortran subroutine `dvode`.
804: 
805:         In the Fortran code, the legal values of `MF` are:
806:             10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25,
807:             -11, -12, -14, -15, -21, -22, -24, -25
808:         but this python wrapper does not use negative values.
809: 
810:         Returns
811: 
812:             mf  = 10*self.meth + miter
813: 
814:         self.meth is the linear multistep method:
815:             self.meth == 1:  method="adams"
816:             self.meth == 2:  method="bdf"
817: 
818:         miter is the correction iteration method:
819:             miter == 0:  Functional iteraton; no Jacobian involved.
820:             miter == 1:  Chord iteration with user-supplied full Jacobian
821:             miter == 2:  Chord iteration with internally computed full Jacobian
822:             miter == 3:  Chord iteration with internally computed diagonal Jacobian
823:             miter == 4:  Chord iteration with user-supplied banded Jacobian
824:             miter == 5:  Chord iteration with internally computed banded Jacobian
825: 
826:         Side effects: If either self.mu or self.ml is not None and the other is None,
827:         then the one that is None is set to 0.
828:         '''
829: 
830:         jac_is_banded = self.mu is not None or self.ml is not None
831:         if jac_is_banded:
832:             if self.mu is None:
833:                 self.mu = 0
834:             if self.ml is None:
835:                 self.ml = 0
836: 
837:         # has_jac is True if the user provided a jacobian function.
838:         if has_jac:
839:             if jac_is_banded:
840:                 miter = 4
841:             else:
842:                 miter = 1
843:         else:
844:             if jac_is_banded:
845:                 if self.ml == self.mu == 0:
846:                     miter = 3  # Chord iteration with internal diagonal Jacobian.
847:                 else:
848:                     miter = 5  # Chord iteration with internal banded Jacobian.
849:             else:
850:                 # self.with_jacobian is set by the user in the call to ode.set_integrator.
851:                 if self.with_jacobian:
852:                     miter = 2  # Chord iteration with internal full Jacobian.
853:                 else:
854:                     miter = 0  # Functional iteraton; no Jacobian involved.
855: 
856:         mf = 10 * self.meth + miter
857:         return mf
858: 
859:     def reset(self, n, has_jac):
860:         mf = self._determine_mf_and_set_bands(has_jac)
861: 
862:         if mf == 10:
863:             lrw = 20 + 16 * n
864:         elif mf in [11, 12]:
865:             lrw = 22 + 16 * n + 2 * n * n
866:         elif mf == 13:
867:             lrw = 22 + 17 * n
868:         elif mf in [14, 15]:
869:             lrw = 22 + 18 * n + (3 * self.ml + 2 * self.mu) * n
870:         elif mf == 20:
871:             lrw = 20 + 9 * n
872:         elif mf in [21, 22]:
873:             lrw = 22 + 9 * n + 2 * n * n
874:         elif mf == 23:
875:             lrw = 22 + 10 * n
876:         elif mf in [24, 25]:
877:             lrw = 22 + 11 * n + (3 * self.ml + 2 * self.mu) * n
878:         else:
879:             raise ValueError('Unexpected mf=%s' % mf)
880: 
881:         if mf % 10 in [0, 3]:
882:             liw = 30
883:         else:
884:             liw = 30 + n
885: 
886:         rwork = zeros((lrw,), float)
887:         rwork[4] = self.first_step
888:         rwork[5] = self.max_step
889:         rwork[6] = self.min_step
890:         self.rwork = rwork
891: 
892:         iwork = zeros((liw,), int32)
893:         if self.ml is not None:
894:             iwork[0] = self.ml
895:         if self.mu is not None:
896:             iwork[1] = self.mu
897:         iwork[4] = self.order
898:         iwork[5] = self.nsteps
899:         iwork[6] = 2  # mxhnil
900:         self.iwork = iwork
901: 
902:         self.call_args = [self.rtol, self.atol, 1, 1,
903:                           self.rwork, self.iwork, mf]
904:         self.success = 1
905:         self.initialized = False
906: 
907:     def run(self, f, jac, y0, t0, t1, f_params, jac_params):
908:         if self.initialized:
909:             self.check_handle()
910:         else:
911:             self.initialized = True
912:             self.acquire_new_handle()
913: 
914:         if self.ml is not None and self.ml > 0:
915:             # Banded Jacobian.  Wrap the user-provided function with one
916:             # that pads the Jacobian array with the extra `self.ml` rows
917:             # required by the f2py-generated wrapper.
918:             jac = _vode_banded_jac_wrapper(jac, self.ml, jac_params)
919: 
920:         args = ((f, jac, y0, t0, t1) + tuple(self.call_args) +
921:                 (f_params, jac_params))
922:         y1, t, istate = self.runner(*args)
923:         self.istate = istate
924:         if istate < 0:
925:             unexpected_istate_msg = 'Unexpected istate={:d}'.format(istate)
926:             warnings.warn('{:s}: {:s}'.format(self.__class__.__name__,
927:                           self.messages.get(istate, unexpected_istate_msg)))
928:             self.success = 0
929:         else:
930:             self.call_args[3] = 2  # upgrade istate from 1 to 2
931:             self.istate = 2
932:         return y1, t
933: 
934:     def step(self, *args):
935:         itask = self.call_args[2]
936:         self.call_args[2] = 2
937:         r = self.run(*args)
938:         self.call_args[2] = itask
939:         return r
940: 
941:     def run_relax(self, *args):
942:         itask = self.call_args[2]
943:         self.call_args[2] = 3
944:         r = self.run(*args)
945:         self.call_args[2] = itask
946:         return r
947: 
948: 
949: if vode.runner is not None:
950:     IntegratorBase.integrator_classes.append(vode)
951: 
952: 
953: class zvode(vode):
954:     runner = getattr(_vode, 'zvode', None)
955: 
956:     supports_run_relax = 1
957:     supports_step = 1
958:     scalar = complex
959:     active_global_handle = 0
960: 
961:     def reset(self, n, has_jac):
962:         mf = self._determine_mf_and_set_bands(has_jac)
963: 
964:         if mf in (10,):
965:             lzw = 15 * n
966:         elif mf in (11, 12):
967:             lzw = 15 * n + 2 * n ** 2
968:         elif mf in (-11, -12):
969:             lzw = 15 * n + n ** 2
970:         elif mf in (13,):
971:             lzw = 16 * n
972:         elif mf in (14, 15):
973:             lzw = 17 * n + (3 * self.ml + 2 * self.mu) * n
974:         elif mf in (-14, -15):
975:             lzw = 16 * n + (2 * self.ml + self.mu) * n
976:         elif mf in (20,):
977:             lzw = 8 * n
978:         elif mf in (21, 22):
979:             lzw = 8 * n + 2 * n ** 2
980:         elif mf in (-21, -22):
981:             lzw = 8 * n + n ** 2
982:         elif mf in (23,):
983:             lzw = 9 * n
984:         elif mf in (24, 25):
985:             lzw = 10 * n + (3 * self.ml + 2 * self.mu) * n
986:         elif mf in (-24, -25):
987:             lzw = 9 * n + (2 * self.ml + self.mu) * n
988: 
989:         lrw = 20 + n
990: 
991:         if mf % 10 in (0, 3):
992:             liw = 30
993:         else:
994:             liw = 30 + n
995: 
996:         zwork = zeros((lzw,), complex)
997:         self.zwork = zwork
998: 
999:         rwork = zeros((lrw,), float)
1000:         rwork[4] = self.first_step
1001:         rwork[5] = self.max_step
1002:         rwork[6] = self.min_step
1003:         self.rwork = rwork
1004: 
1005:         iwork = zeros((liw,), int32)
1006:         if self.ml is not None:
1007:             iwork[0] = self.ml
1008:         if self.mu is not None:
1009:             iwork[1] = self.mu
1010:         iwork[4] = self.order
1011:         iwork[5] = self.nsteps
1012:         iwork[6] = 2  # mxhnil
1013:         self.iwork = iwork
1014: 
1015:         self.call_args = [self.rtol, self.atol, 1, 1,
1016:                           self.zwork, self.rwork, self.iwork, mf]
1017:         self.success = 1
1018:         self.initialized = False
1019: 
1020: 
1021: if zvode.runner is not None:
1022:     IntegratorBase.integrator_classes.append(zvode)
1023: 
1024: 
1025: class dopri5(IntegratorBase):
1026:     runner = getattr(_dop, 'dopri5', None)
1027:     name = 'dopri5'
1028:     supports_solout = True
1029: 
1030:     messages = {1: 'computation successful',
1031:                 2: 'comput. successful (interrupted by solout)',
1032:                 -1: 'input is not consistent',
1033:                 -2: 'larger nmax is needed',
1034:                 -3: 'step size becomes too small',
1035:                 -4: 'problem is probably stiff (interrupted)',
1036:                 }
1037: 
1038:     def __init__(self,
1039:                  rtol=1e-6, atol=1e-12,
1040:                  nsteps=500,
1041:                  max_step=0.0,
1042:                  first_step=0.0,  # determined by solver
1043:                  safety=0.9,
1044:                  ifactor=10.0,
1045:                  dfactor=0.2,
1046:                  beta=0.0,
1047:                  method=None,
1048:                  verbosity=-1,  # no messages if negative
1049:                  ):
1050:         self.rtol = rtol
1051:         self.atol = atol
1052:         self.nsteps = nsteps
1053:         self.max_step = max_step
1054:         self.first_step = first_step
1055:         self.safety = safety
1056:         self.ifactor = ifactor
1057:         self.dfactor = dfactor
1058:         self.beta = beta
1059:         self.verbosity = verbosity
1060:         self.success = 1
1061:         self.set_solout(None)
1062: 
1063:     def set_solout(self, solout, complex=False):
1064:         self.solout = solout
1065:         self.solout_cmplx = complex
1066:         if solout is None:
1067:             self.iout = 0
1068:         else:
1069:             self.iout = 1
1070: 
1071:     def reset(self, n, has_jac):
1072:         work = zeros((8 * n + 21,), float)
1073:         work[1] = self.safety
1074:         work[2] = self.dfactor
1075:         work[3] = self.ifactor
1076:         work[4] = self.beta
1077:         work[5] = self.max_step
1078:         work[6] = self.first_step
1079:         self.work = work
1080:         iwork = zeros((21,), int32)
1081:         iwork[0] = self.nsteps
1082:         iwork[2] = self.verbosity
1083:         self.iwork = iwork
1084:         self.call_args = [self.rtol, self.atol, self._solout,
1085:                           self.iout, self.work, self.iwork]
1086:         self.success = 1
1087: 
1088:     def run(self, f, jac, y0, t0, t1, f_params, jac_params):
1089:         x, y, iwork, istate = self.runner(*((f, t0, y0, t1) +
1090:                                           tuple(self.call_args) + (f_params,)))
1091:         self.istate = istate
1092:         if istate < 0:
1093:             unexpected_istate_msg = 'Unexpected istate={:d}'.format(istate)
1094:             warnings.warn('{:s}: {:s}'.format(self.__class__.__name__,
1095:                           self.messages.get(istate, unexpected_istate_msg)))
1096:             self.success = 0
1097:         return y, x
1098: 
1099:     def _solout(self, nr, xold, x, y, nd, icomp, con):
1100:         if self.solout is not None:
1101:             if self.solout_cmplx:
1102:                 y = y[::2] + 1j * y[1::2]
1103:             return self.solout(x, y)
1104:         else:
1105:             return 1
1106: 
1107: 
1108: if dopri5.runner is not None:
1109:     IntegratorBase.integrator_classes.append(dopri5)
1110: 
1111: 
1112: class dop853(dopri5):
1113:     runner = getattr(_dop, 'dop853', None)
1114:     name = 'dop853'
1115: 
1116:     def __init__(self,
1117:                  rtol=1e-6, atol=1e-12,
1118:                  nsteps=500,
1119:                  max_step=0.0,
1120:                  first_step=0.0,  # determined by solver
1121:                  safety=0.9,
1122:                  ifactor=6.0,
1123:                  dfactor=0.3,
1124:                  beta=0.0,
1125:                  method=None,
1126:                  verbosity=-1,  # no messages if negative
1127:                  ):
1128:         super(self.__class__, self).__init__(rtol, atol, nsteps, max_step,
1129:                                              first_step, safety, ifactor,
1130:                                              dfactor, beta, method,
1131:                                              verbosity)
1132: 
1133:     def reset(self, n, has_jac):
1134:         work = zeros((11 * n + 21,), float)
1135:         work[1] = self.safety
1136:         work[2] = self.dfactor
1137:         work[3] = self.ifactor
1138:         work[4] = self.beta
1139:         work[5] = self.max_step
1140:         work[6] = self.first_step
1141:         self.work = work
1142:         iwork = zeros((21,), int32)
1143:         iwork[0] = self.nsteps
1144:         iwork[2] = self.verbosity
1145:         self.iwork = iwork
1146:         self.call_args = [self.rtol, self.atol, self._solout,
1147:                           self.iout, self.work, self.iwork]
1148:         self.success = 1
1149: 
1150: 
1151: if dop853.runner is not None:
1152:     IntegratorBase.integrator_classes.append(dop853)
1153: 
1154: 
1155: class lsoda(IntegratorBase):
1156:     runner = getattr(_lsoda, 'lsoda', None)
1157:     active_global_handle = 0
1158: 
1159:     messages = {
1160:         2: "Integration successful.",
1161:         -1: "Excess work done on this call (perhaps wrong Dfun type).",
1162:         -2: "Excess accuracy requested (tolerances too small).",
1163:         -3: "Illegal input detected (internal error).",
1164:         -4: "Repeated error test failures (internal error).",
1165:         -5: "Repeated convergence failures (perhaps bad Jacobian or tolerances).",
1166:         -6: "Error weight became zero during problem.",
1167:         -7: "Internal workspace insufficient to finish (internal error)."
1168:     }
1169: 
1170:     def __init__(self,
1171:                  with_jacobian=False,
1172:                  rtol=1e-6, atol=1e-12,
1173:                  lband=None, uband=None,
1174:                  nsteps=500,
1175:                  max_step=0.0,  # corresponds to infinite
1176:                  min_step=0.0,
1177:                  first_step=0.0,  # determined by solver
1178:                  ixpr=0,
1179:                  max_hnil=0,
1180:                  max_order_ns=12,
1181:                  max_order_s=5,
1182:                  method=None
1183:                  ):
1184: 
1185:         self.with_jacobian = with_jacobian
1186:         self.rtol = rtol
1187:         self.atol = atol
1188:         self.mu = uband
1189:         self.ml = lband
1190: 
1191:         self.max_order_ns = max_order_ns
1192:         self.max_order_s = max_order_s
1193:         self.nsteps = nsteps
1194:         self.max_step = max_step
1195:         self.min_step = min_step
1196:         self.first_step = first_step
1197:         self.ixpr = ixpr
1198:         self.max_hnil = max_hnil
1199:         self.success = 1
1200: 
1201:         self.initialized = False
1202: 
1203:     def reset(self, n, has_jac):
1204:         # Calculate parameters for Fortran subroutine dvode.
1205:         if has_jac:
1206:             if self.mu is None and self.ml is None:
1207:                 jt = 1
1208:             else:
1209:                 if self.mu is None:
1210:                     self.mu = 0
1211:                 if self.ml is None:
1212:                     self.ml = 0
1213:                 jt = 4
1214:         else:
1215:             if self.mu is None and self.ml is None:
1216:                 jt = 2
1217:             else:
1218:                 if self.mu is None:
1219:                     self.mu = 0
1220:                 if self.ml is None:
1221:                     self.ml = 0
1222:                 jt = 5
1223:         lrn = 20 + (self.max_order_ns + 4) * n
1224:         if jt in [1, 2]:
1225:             lrs = 22 + (self.max_order_s + 4) * n + n * n
1226:         elif jt in [4, 5]:
1227:             lrs = 22 + (self.max_order_s + 5 + 2 * self.ml + self.mu) * n
1228:         else:
1229:             raise ValueError('Unexpected jt=%s' % jt)
1230:         lrw = max(lrn, lrs)
1231:         liw = 20 + n
1232:         rwork = zeros((lrw,), float)
1233:         rwork[4] = self.first_step
1234:         rwork[5] = self.max_step
1235:         rwork[6] = self.min_step
1236:         self.rwork = rwork
1237:         iwork = zeros((liw,), int32)
1238:         if self.ml is not None:
1239:             iwork[0] = self.ml
1240:         if self.mu is not None:
1241:             iwork[1] = self.mu
1242:         iwork[4] = self.ixpr
1243:         iwork[5] = self.nsteps
1244:         iwork[6] = self.max_hnil
1245:         iwork[7] = self.max_order_ns
1246:         iwork[8] = self.max_order_s
1247:         self.iwork = iwork
1248:         self.call_args = [self.rtol, self.atol, 1, 1,
1249:                           self.rwork, self.iwork, jt]
1250:         self.success = 1
1251:         self.initialized = False
1252: 
1253:     def run(self, f, jac, y0, t0, t1, f_params, jac_params):
1254:         if self.initialized:
1255:             self.check_handle()
1256:         else:
1257:             self.initialized = True
1258:             self.acquire_new_handle()
1259:         args = [f, y0, t0, t1] + self.call_args[:-1] + \
1260:                [jac, self.call_args[-1], f_params, 0, jac_params]
1261:         y1, t, istate = self.runner(*args)
1262:         self.istate = istate
1263:         if istate < 0:
1264:             unexpected_istate_msg = 'Unexpected istate={:d}'.format(istate)
1265:             warnings.warn('{:s}: {:s}'.format(self.__class__.__name__,
1266:                           self.messages.get(istate, unexpected_istate_msg)))
1267:             self.success = 0
1268:         else:
1269:             self.call_args[3] = 2  # upgrade istate from 1 to 2
1270:             self.istate = 2
1271:         return y1, t
1272: 
1273:     def step(self, *args):
1274:         itask = self.call_args[2]
1275:         self.call_args[2] = 2
1276:         r = self.run(*args)
1277:         self.call_args[2] = itask
1278:         return r
1279: 
1280:     def run_relax(self, *args):
1281:         itask = self.call_args[2]
1282:         self.call_args[2] = 3
1283:         r = self.run(*args)
1284:         self.call_args[2] = itask
1285:         return r
1286: 
1287: 
1288: if lsoda.runner:
1289:     IntegratorBase.integrator_classes.append(lsoda)
1290: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_35485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'str', '\nFirst-order ODE integrators.\n\nUser-friendly interface to various numerical integrators for solving a\nsystem of first order ODEs with prescribed initial conditions::\n\n    d y(t)[i]\n    ---------  = f(t,y(t))[i],\n       d t\n\n    y(t=0)[i] = y0[i],\n\nwhere::\n\n    i = 0, ..., len(y0) - 1\n\nclass ode\n---------\n\nA generic interface class to numeric integrators. It has the following\nmethods::\n\n    integrator = ode(f, jac=None)\n    integrator = integrator.set_integrator(name, **params)\n    integrator = integrator.set_initial_value(y0, t0=0.0)\n    integrator = integrator.set_f_params(*args)\n    integrator = integrator.set_jac_params(*args)\n    y1 = integrator.integrate(t1, step=False, relax=False)\n    flag = integrator.successful()\n\nclass complex_ode\n-----------------\n\nThis class has the same generic interface as ode, except it can handle complex\nf, y and Jacobians by transparently translating them into the equivalent\nreal valued system. It supports the real valued solvers (i.e not zvode) and is\nan alternative to ode with the zvode solver, sometimes performing better.\n')

# Assigning a List to a Name (line 83):

# Assigning a List to a Name (line 83):
__all__ = ['ode', 'complex_ode']
module_type_store.set_exportable_members(['ode', 'complex_ode'])

# Obtaining an instance of the builtin type 'list' (line 83)
list_35486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 83)
# Adding element type (line 83)
str_35487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 11), 'str', 'ode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 10), list_35486, str_35487)
# Adding element type (line 83)
str_35488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 18), 'str', 'complex_ode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 10), list_35486, str_35488)

# Assigning a type to the variable '__all__' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), '__all__', list_35486)

# Assigning a Str to a Name (line 84):

# Assigning a Str to a Name (line 84):
str_35489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 14), 'str', '$Id$')
# Assigning a type to the variable '__version__' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), '__version__', str_35489)

# Assigning a Str to a Name (line 85):

# Assigning a Str to a Name (line 85):
str_35490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), '__docformat__', str_35490)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 87, 0))

# 'import re' statement (line 87)
import re

import_module(stypy.reporting.localization.Localization(__file__, 87, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 88, 0))

# 'import warnings' statement (line 88)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 88, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 90, 0))

# 'from numpy import asarray, array, zeros, int32, isscalar, real, imag, vstack' statement (line 90)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_35491 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 90, 0), 'numpy')

if (type(import_35491) is not StypyTypeError):

    if (import_35491 != 'pyd_module'):
        __import__(import_35491)
        sys_modules_35492 = sys.modules[import_35491]
        import_from_module(stypy.reporting.localization.Localization(__file__, 90, 0), 'numpy', sys_modules_35492.module_type_store, module_type_store, ['asarray', 'array', 'zeros', 'int32', 'isscalar', 'real', 'imag', 'vstack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 90, 0), __file__, sys_modules_35492, sys_modules_35492.module_type_store, module_type_store)
    else:
        from numpy import asarray, array, zeros, int32, isscalar, real, imag, vstack

        import_from_module(stypy.reporting.localization.Localization(__file__, 90, 0), 'numpy', None, module_type_store, ['asarray', 'array', 'zeros', 'int32', 'isscalar', 'real', 'imag', 'vstack'], [asarray, array, zeros, int32, isscalar, real, imag, vstack])

else:
    # Assigning a type to the variable 'numpy' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'numpy', import_35491)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 92, 0))

# 'from scipy.integrate import _vode' statement (line 92)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_35493 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 92, 0), 'scipy.integrate')

if (type(import_35493) is not StypyTypeError):

    if (import_35493 != 'pyd_module'):
        __import__(import_35493)
        sys_modules_35494 = sys.modules[import_35493]
        import_from_module(stypy.reporting.localization.Localization(__file__, 92, 0), 'scipy.integrate', sys_modules_35494.module_type_store, module_type_store, ['vode'])
        nest_module(stypy.reporting.localization.Localization(__file__, 92, 0), __file__, sys_modules_35494, sys_modules_35494.module_type_store, module_type_store)
    else:
        from scipy.integrate import vode as _vode

        import_from_module(stypy.reporting.localization.Localization(__file__, 92, 0), 'scipy.integrate', None, module_type_store, ['vode'], [_vode])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'scipy.integrate', import_35493)

# Adding an alias
module_type_store.add_alias('_vode', 'vode')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 93, 0))

# 'from scipy.integrate import _dop' statement (line 93)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_35495 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.integrate')

if (type(import_35495) is not StypyTypeError):

    if (import_35495 != 'pyd_module'):
        __import__(import_35495)
        sys_modules_35496 = sys.modules[import_35495]
        import_from_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.integrate', sys_modules_35496.module_type_store, module_type_store, ['_dop'])
        nest_module(stypy.reporting.localization.Localization(__file__, 93, 0), __file__, sys_modules_35496, sys_modules_35496.module_type_store, module_type_store)
    else:
        from scipy.integrate import _dop

        import_from_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.integrate', None, module_type_store, ['_dop'], [_dop])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.integrate', import_35495)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 94, 0))

# 'from scipy.integrate import _lsoda' statement (line 94)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_35497 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.integrate')

if (type(import_35497) is not StypyTypeError):

    if (import_35497 != 'pyd_module'):
        __import__(import_35497)
        sys_modules_35498 = sys.modules[import_35497]
        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.integrate', sys_modules_35498.module_type_store, module_type_store, ['lsoda'])
        nest_module(stypy.reporting.localization.Localization(__file__, 94, 0), __file__, sys_modules_35498, sys_modules_35498.module_type_store, module_type_store)
    else:
        from scipy.integrate import lsoda as _lsoda

        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.integrate', None, module_type_store, ['lsoda'], [_lsoda])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.integrate', import_35497)

# Adding an alias
module_type_store.add_alias('_lsoda', 'lsoda')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

# Declaration of the 'ode' class

class ode(object, ):
    str_35499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, (-1)), 'str', '\n    A generic interface class to numeric integrators.\n\n    Solve an equation system :math:`y\'(t) = f(t,y)` with (optional) ``jac = df/dy``.\n\n    *Note*: The first two arguments of ``f(t, y, ...)`` are in the\n    opposite order of the arguments in the system definition function used\n    by `scipy.integrate.odeint`.\n\n    Parameters\n    ----------\n    f : callable ``f(t, y, *f_args)``\n        Right-hand side of the differential equation. t is a scalar,\n        ``y.shape == (n,)``.\n        ``f_args`` is set by calling ``set_f_params(*args)``.\n        `f` should return a scalar, array or list (not a tuple).\n    jac : callable ``jac(t, y, *jac_args)``, optional\n        Jacobian of the right-hand side, ``jac[i,j] = d f[i] / d y[j]``.\n        ``jac_args`` is set by calling ``set_jac_params(*args)``.\n\n    Attributes\n    ----------\n    t : float\n        Current time.\n    y : ndarray\n        Current variable values.\n\n    See also\n    --------\n    odeint : an integrator with a simpler interface based on lsoda from ODEPACK\n    quad : for finding the area under a curve\n\n    Notes\n    -----\n    Available integrators are listed below. They can be selected using\n    the `set_integrator` method.\n\n    "vode"\n\n        Real-valued Variable-coefficient Ordinary Differential Equation\n        solver, with fixed-leading-coefficient implementation. It provides\n        implicit Adams method (for non-stiff problems) and a method based on\n        backward differentiation formulas (BDF) (for stiff problems).\n\n        Source: http://www.netlib.org/ode/vode.f\n\n        .. warning::\n\n           This integrator is not re-entrant. You cannot have two `ode`\n           instances using the "vode" integrator at the same time.\n\n        This integrator accepts the following parameters in `set_integrator`\n        method of the `ode` class:\n\n        - atol : float or sequence\n          absolute tolerance for solution\n        - rtol : float or sequence\n          relative tolerance for solution\n        - lband : None or int\n        - uband : None or int\n          Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+uband.\n          Setting these requires your jac routine to return the jacobian\n          in packed format, jac_packed[i-j+uband, j] = jac[i,j]. The\n          dimension of the matrix must be (lband+uband+1, len(y)).\n        - method: \'adams\' or \'bdf\'\n          Which solver to use, Adams (non-stiff) or BDF (stiff)\n        - with_jacobian : bool\n          This option is only considered when the user has not supplied a\n          Jacobian function and has not indicated (by setting either band)\n          that the Jacobian is banded.  In this case, `with_jacobian` specifies\n          whether the iteration method of the ODE solver\'s correction step is\n          chord iteration with an internally generated full Jacobian or\n          functional iteration with no Jacobian.\n        - nsteps : int\n          Maximum number of (internally defined) steps allowed during one\n          call to the solver.\n        - first_step : float\n        - min_step : float\n        - max_step : float\n          Limits for the step sizes used by the integrator.\n        - order : int\n          Maximum order used by the integrator,\n          order <= 12 for Adams, <= 5 for BDF.\n\n    "zvode"\n\n        Complex-valued Variable-coefficient Ordinary Differential Equation\n        solver, with fixed-leading-coefficient implementation.  It provides\n        implicit Adams method (for non-stiff problems) and a method based on\n        backward differentiation formulas (BDF) (for stiff problems).\n\n        Source: http://www.netlib.org/ode/zvode.f\n\n        .. warning::\n\n           This integrator is not re-entrant. You cannot have two `ode`\n           instances using the "zvode" integrator at the same time.\n\n        This integrator accepts the same parameters in `set_integrator`\n        as the "vode" solver.\n\n        .. note::\n\n            When using ZVODE for a stiff system, it should only be used for\n            the case in which the function f is analytic, that is, when each f(i)\n            is an analytic function of each y(j).  Analyticity means that the\n            partial derivative df(i)/dy(j) is a unique complex number, and this\n            fact is critical in the way ZVODE solves the dense or banded linear\n            systems that arise in the stiff case.  For a complex stiff ODE system\n            in which f is not analytic, ZVODE is likely to have convergence\n            failures, and for this problem one should instead use DVODE on the\n            equivalent real system (in the real and imaginary parts of y).\n\n    "lsoda"\n\n        Real-valued Variable-coefficient Ordinary Differential Equation\n        solver, with fixed-leading-coefficient implementation. It provides\n        automatic method switching between implicit Adams method (for non-stiff\n        problems) and a method based on backward differentiation formulas (BDF)\n        (for stiff problems).\n\n        Source: http://www.netlib.org/odepack\n\n        .. warning::\n\n           This integrator is not re-entrant. You cannot have two `ode`\n           instances using the "lsoda" integrator at the same time.\n\n        This integrator accepts the following parameters in `set_integrator`\n        method of the `ode` class:\n\n        - atol : float or sequence\n          absolute tolerance for solution\n        - rtol : float or sequence\n          relative tolerance for solution\n        - lband : None or int\n        - uband : None or int\n          Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+uband.\n          Setting these requires your jac routine to return the jacobian\n          in packed format, jac_packed[i-j+uband, j] = jac[i,j].\n        - with_jacobian : bool\n          *Not used.*\n        - nsteps : int\n          Maximum number of (internally defined) steps allowed during one\n          call to the solver.\n        - first_step : float\n        - min_step : float\n        - max_step : float\n          Limits for the step sizes used by the integrator.\n        - max_order_ns : int\n          Maximum order used in the nonstiff case (default 12).\n        - max_order_s : int\n          Maximum order used in the stiff case (default 5).\n        - max_hnil : int\n          Maximum number of messages reporting too small step size (t + h = t)\n          (default 0)\n        - ixpr : int\n          Whether to generate extra printing at method switches (default False).\n\n    "dopri5"\n\n        This is an explicit runge-kutta method of order (4)5 due to Dormand &\n        Prince (with stepsize control and dense output).\n\n        Authors:\n\n            E. Hairer and G. Wanner\n            Universite de Geneve, Dept. de Mathematiques\n            CH-1211 Geneve 24, Switzerland\n            e-mail:  ernst.hairer@math.unige.ch, gerhard.wanner@math.unige.ch\n\n        This code is described in [HNW93]_.\n\n        This integrator accepts the following parameters in set_integrator()\n        method of the ode class:\n\n        - atol : float or sequence\n          absolute tolerance for solution\n        - rtol : float or sequence\n          relative tolerance for solution\n        - nsteps : int\n          Maximum number of (internally defined) steps allowed during one\n          call to the solver.\n        - first_step : float\n        - max_step : float\n        - safety : float\n          Safety factor on new step selection (default 0.9)\n        - ifactor : float\n        - dfactor : float\n          Maximum factor to increase/decrease step size by in one step\n        - beta : float\n          Beta parameter for stabilised step size control.\n        - verbosity : int\n          Switch for printing messages (< 0 for no messages).\n\n    "dop853"\n\n        This is an explicit runge-kutta method of order 8(5,3) due to Dormand\n        & Prince (with stepsize control and dense output).\n\n        Options and references the same as "dopri5".\n\n    Examples\n    --------\n\n    A problem to integrate and the corresponding jacobian:\n\n    >>> from scipy.integrate import ode\n    >>>\n    >>> y0, t0 = [1.0j, 2.0], 0\n    >>>\n    >>> def f(t, y, arg1):\n    ...     return [1j*arg1*y[0] + y[1], -arg1*y[1]**2]\n    >>> def jac(t, y, arg1):\n    ...     return [[1j*arg1, 1], [0, -arg1*2*y[1]]]\n\n    The integration:\n\n    >>> r = ode(f, jac).set_integrator(\'zvode\', method=\'bdf\')\n    >>> r.set_initial_value(y0, t0).set_f_params(2.0).set_jac_params(2.0)\n    >>> t1 = 10\n    >>> dt = 1\n    >>> while r.successful() and r.t < t1:\n    ...     print(r.t+dt, r.integrate(r.t+dt))\n    1 [-0.71038232+0.23749653j  0.40000271+0.j        ]\n    2.0 [ 0.19098503-0.52359246j  0.22222356+0.j        ]\n    3.0 [ 0.47153208+0.52701229j  0.15384681+0.j        ]\n    4.0 [-0.61905937+0.30726255j  0.11764744+0.j        ]\n    5.0 [ 0.02340997-0.61418799j  0.09523835+0.j        ]\n    6.0 [ 0.58643071+0.339819j  0.08000018+0.j      ]\n    7.0 [-0.52070105+0.44525141j  0.06896565+0.j        ]\n    8.0 [-0.15986733-0.61234476j  0.06060616+0.j        ]\n    9.0 [ 0.64850462+0.15048982j  0.05405414+0.j        ]\n    10.0 [-0.38404699+0.56382299j  0.04878055+0.j        ]\n\n    References\n    ----------\n    .. [HNW93] E. Hairer, S.P. Norsett and G. Wanner, Solving Ordinary\n        Differential Equations i. Nonstiff Problems. 2nd edition.\n        Springer Series in Computational Mathematics,\n        Springer-Verlag (1993)\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 347)
        None_35500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 30), 'None')
        defaults = [None_35500]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 347, 4, False)
        # Assigning a type to the variable 'self' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ode.__init__', ['f', 'jac'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['f', 'jac'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Num to a Attribute (line 348):
        
        # Assigning a Num to a Attribute (line 348):
        int_35501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 21), 'int')
        # Getting the type of 'self' (line 348)
        self_35502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'self')
        # Setting the type of the member 'stiff' of a type (line 348)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), self_35502, 'stiff', int_35501)
        
        # Assigning a Name to a Attribute (line 349):
        
        # Assigning a Name to a Attribute (line 349):
        # Getting the type of 'f' (line 349)
        f_35503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 17), 'f')
        # Getting the type of 'self' (line 349)
        self_35504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'self')
        # Setting the type of the member 'f' of a type (line 349)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), self_35504, 'f', f_35503)
        
        # Assigning a Name to a Attribute (line 350):
        
        # Assigning a Name to a Attribute (line 350):
        # Getting the type of 'jac' (line 350)
        jac_35505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 19), 'jac')
        # Getting the type of 'self' (line 350)
        self_35506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'self')
        # Setting the type of the member 'jac' of a type (line 350)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), self_35506, 'jac', jac_35505)
        
        # Assigning a Tuple to a Attribute (line 351):
        
        # Assigning a Tuple to a Attribute (line 351):
        
        # Obtaining an instance of the builtin type 'tuple' (line 351)
        tuple_35507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 351)
        
        # Getting the type of 'self' (line 351)
        self_35508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'self')
        # Setting the type of the member 'f_params' of a type (line 351)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 8), self_35508, 'f_params', tuple_35507)
        
        # Assigning a Tuple to a Attribute (line 352):
        
        # Assigning a Tuple to a Attribute (line 352):
        
        # Obtaining an instance of the builtin type 'tuple' (line 352)
        tuple_35509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 352)
        
        # Getting the type of 'self' (line 352)
        self_35510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'self')
        # Setting the type of the member 'jac_params' of a type (line 352)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), self_35510, 'jac_params', tuple_35509)
        
        # Assigning a List to a Attribute (line 353):
        
        # Assigning a List to a Attribute (line 353):
        
        # Obtaining an instance of the builtin type 'list' (line 353)
        list_35511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 353)
        
        # Getting the type of 'self' (line 353)
        self_35512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'self')
        # Setting the type of the member '_y' of a type (line 353)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 8), self_35512, '_y', list_35511)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def y(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'y'
        module_type_store = module_type_store.open_function_context('y', 355, 4, False)
        # Assigning a type to the variable 'self' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ode.y.__dict__.__setitem__('stypy_localization', localization)
        ode.y.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ode.y.__dict__.__setitem__('stypy_type_store', module_type_store)
        ode.y.__dict__.__setitem__('stypy_function_name', 'ode.y')
        ode.y.__dict__.__setitem__('stypy_param_names_list', [])
        ode.y.__dict__.__setitem__('stypy_varargs_param_name', None)
        ode.y.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ode.y.__dict__.__setitem__('stypy_call_defaults', defaults)
        ode.y.__dict__.__setitem__('stypy_call_varargs', varargs)
        ode.y.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ode.y.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ode.y', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'y', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'y(...)' code ##################

        # Getting the type of 'self' (line 357)
        self_35513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 15), 'self')
        # Obtaining the member '_y' of a type (line 357)
        _y_35514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 15), self_35513, '_y')
        # Assigning a type to the variable 'stypy_return_type' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'stypy_return_type', _y_35514)
        
        # ################# End of 'y(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'y' in the type store
        # Getting the type of 'stypy_return_type' (line 355)
        stypy_return_type_35515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35515)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'y'
        return stypy_return_type_35515


    @norecursion
    def set_initial_value(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_35516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 37), 'float')
        defaults = [float_35516]
        # Create a new context for function 'set_initial_value'
        module_type_store = module_type_store.open_function_context('set_initial_value', 359, 4, False)
        # Assigning a type to the variable 'self' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ode.set_initial_value.__dict__.__setitem__('stypy_localization', localization)
        ode.set_initial_value.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ode.set_initial_value.__dict__.__setitem__('stypy_type_store', module_type_store)
        ode.set_initial_value.__dict__.__setitem__('stypy_function_name', 'ode.set_initial_value')
        ode.set_initial_value.__dict__.__setitem__('stypy_param_names_list', ['y', 't'])
        ode.set_initial_value.__dict__.__setitem__('stypy_varargs_param_name', None)
        ode.set_initial_value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ode.set_initial_value.__dict__.__setitem__('stypy_call_defaults', defaults)
        ode.set_initial_value.__dict__.__setitem__('stypy_call_varargs', varargs)
        ode.set_initial_value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ode.set_initial_value.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ode.set_initial_value', ['y', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_initial_value', localization, ['y', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_initial_value(...)' code ##################

        str_35517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 8), 'str', 'Set initial conditions y(t) = y.')
        
        
        # Call to isscalar(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'y' (line 361)
        y_35519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'y', False)
        # Processing the call keyword arguments (line 361)
        kwargs_35520 = {}
        # Getting the type of 'isscalar' (line 361)
        isscalar_35518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 11), 'isscalar', False)
        # Calling isscalar(args, kwargs) (line 361)
        isscalar_call_result_35521 = invoke(stypy.reporting.localization.Localization(__file__, 361, 11), isscalar_35518, *[y_35519], **kwargs_35520)
        
        # Testing the type of an if condition (line 361)
        if_condition_35522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 8), isscalar_call_result_35521)
        # Assigning a type to the variable 'if_condition_35522' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'if_condition_35522', if_condition_35522)
        # SSA begins for if statement (line 361)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 362):
        
        # Assigning a List to a Name (line 362):
        
        # Obtaining an instance of the builtin type 'list' (line 362)
        list_35523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 362)
        # Adding element type (line 362)
        # Getting the type of 'y' (line 362)
        y_35524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 17), 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 16), list_35523, y_35524)
        
        # Assigning a type to the variable 'y' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'y', list_35523)
        # SSA join for if statement (line 361)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 363):
        
        # Assigning a Call to a Name (line 363):
        
        # Call to len(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'self' (line 363)
        self_35526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 21), 'self', False)
        # Obtaining the member '_y' of a type (line 363)
        _y_35527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 21), self_35526, '_y')
        # Processing the call keyword arguments (line 363)
        kwargs_35528 = {}
        # Getting the type of 'len' (line 363)
        len_35525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 17), 'len', False)
        # Calling len(args, kwargs) (line 363)
        len_call_result_35529 = invoke(stypy.reporting.localization.Localization(__file__, 363, 17), len_35525, *[_y_35527], **kwargs_35528)
        
        # Assigning a type to the variable 'n_prev' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'n_prev', len_call_result_35529)
        
        
        # Getting the type of 'n_prev' (line 364)
        n_prev_35530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 15), 'n_prev')
        # Applying the 'not' unary operator (line 364)
        result_not__35531 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 11), 'not', n_prev_35530)
        
        # Testing the type of an if condition (line 364)
        if_condition_35532 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 8), result_not__35531)
        # Assigning a type to the variable 'if_condition_35532' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'if_condition_35532', if_condition_35532)
        # SSA begins for if statement (line 364)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_integrator(...): (line 365)
        # Processing the call arguments (line 365)
        str_35535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 32), 'str', '')
        # Processing the call keyword arguments (line 365)
        kwargs_35536 = {}
        # Getting the type of 'self' (line 365)
        self_35533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'self', False)
        # Obtaining the member 'set_integrator' of a type (line 365)
        set_integrator_35534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 12), self_35533, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 365)
        set_integrator_call_result_35537 = invoke(stypy.reporting.localization.Localization(__file__, 365, 12), set_integrator_35534, *[str_35535], **kwargs_35536)
        
        # SSA join for if statement (line 364)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 366):
        
        # Assigning a Call to a Attribute (line 366):
        
        # Call to asarray(...): (line 366)
        # Processing the call arguments (line 366)
        # Getting the type of 'y' (line 366)
        y_35539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 26), 'y', False)
        # Getting the type of 'self' (line 366)
        self_35540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 29), 'self', False)
        # Obtaining the member '_integrator' of a type (line 366)
        _integrator_35541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 29), self_35540, '_integrator')
        # Obtaining the member 'scalar' of a type (line 366)
        scalar_35542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 29), _integrator_35541, 'scalar')
        # Processing the call keyword arguments (line 366)
        kwargs_35543 = {}
        # Getting the type of 'asarray' (line 366)
        asarray_35538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 18), 'asarray', False)
        # Calling asarray(args, kwargs) (line 366)
        asarray_call_result_35544 = invoke(stypy.reporting.localization.Localization(__file__, 366, 18), asarray_35538, *[y_35539, scalar_35542], **kwargs_35543)
        
        # Getting the type of 'self' (line 366)
        self_35545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'self')
        # Setting the type of the member '_y' of a type (line 366)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 8), self_35545, '_y', asarray_call_result_35544)
        
        # Assigning a Name to a Attribute (line 367):
        
        # Assigning a Name to a Attribute (line 367):
        # Getting the type of 't' (line 367)
        t_35546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 17), 't')
        # Getting the type of 'self' (line 367)
        self_35547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'self')
        # Setting the type of the member 't' of a type (line 367)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 8), self_35547, 't', t_35546)
        
        # Call to reset(...): (line 368)
        # Processing the call arguments (line 368)
        
        # Call to len(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'self' (line 368)
        self_35552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 35), 'self', False)
        # Obtaining the member '_y' of a type (line 368)
        _y_35553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 35), self_35552, '_y')
        # Processing the call keyword arguments (line 368)
        kwargs_35554 = {}
        # Getting the type of 'len' (line 368)
        len_35551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 31), 'len', False)
        # Calling len(args, kwargs) (line 368)
        len_call_result_35555 = invoke(stypy.reporting.localization.Localization(__file__, 368, 31), len_35551, *[_y_35553], **kwargs_35554)
        
        
        # Getting the type of 'self' (line 368)
        self_35556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 45), 'self', False)
        # Obtaining the member 'jac' of a type (line 368)
        jac_35557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 45), self_35556, 'jac')
        # Getting the type of 'None' (line 368)
        None_35558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 61), 'None', False)
        # Applying the binary operator 'isnot' (line 368)
        result_is_not_35559 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 45), 'isnot', jac_35557, None_35558)
        
        # Processing the call keyword arguments (line 368)
        kwargs_35560 = {}
        # Getting the type of 'self' (line 368)
        self_35548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'self', False)
        # Obtaining the member '_integrator' of a type (line 368)
        _integrator_35549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), self_35548, '_integrator')
        # Obtaining the member 'reset' of a type (line 368)
        reset_35550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), _integrator_35549, 'reset')
        # Calling reset(args, kwargs) (line 368)
        reset_call_result_35561 = invoke(stypy.reporting.localization.Localization(__file__, 368, 8), reset_35550, *[len_call_result_35555, result_is_not_35559], **kwargs_35560)
        
        # Getting the type of 'self' (line 369)
        self_35562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'stypy_return_type', self_35562)
        
        # ################# End of 'set_initial_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_initial_value' in the type store
        # Getting the type of 'stypy_return_type' (line 359)
        stypy_return_type_35563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35563)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_initial_value'
        return stypy_return_type_35563


    @norecursion
    def set_integrator(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_integrator'
        module_type_store = module_type_store.open_function_context('set_integrator', 371, 4, False)
        # Assigning a type to the variable 'self' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ode.set_integrator.__dict__.__setitem__('stypy_localization', localization)
        ode.set_integrator.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ode.set_integrator.__dict__.__setitem__('stypy_type_store', module_type_store)
        ode.set_integrator.__dict__.__setitem__('stypy_function_name', 'ode.set_integrator')
        ode.set_integrator.__dict__.__setitem__('stypy_param_names_list', ['name'])
        ode.set_integrator.__dict__.__setitem__('stypy_varargs_param_name', None)
        ode.set_integrator.__dict__.__setitem__('stypy_kwargs_param_name', 'integrator_params')
        ode.set_integrator.__dict__.__setitem__('stypy_call_defaults', defaults)
        ode.set_integrator.__dict__.__setitem__('stypy_call_varargs', varargs)
        ode.set_integrator.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ode.set_integrator.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ode.set_integrator', ['name'], None, 'integrator_params', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_integrator', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_integrator(...)' code ##################

        str_35564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, (-1)), 'str', '\n        Set integrator by name.\n\n        Parameters\n        ----------\n        name : str\n            Name of the integrator.\n        integrator_params\n            Additional parameters for the integrator.\n        ')
        
        # Assigning a Call to a Name (line 382):
        
        # Assigning a Call to a Name (line 382):
        
        # Call to find_integrator(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'name' (line 382)
        name_35566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 37), 'name', False)
        # Processing the call keyword arguments (line 382)
        kwargs_35567 = {}
        # Getting the type of 'find_integrator' (line 382)
        find_integrator_35565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 21), 'find_integrator', False)
        # Calling find_integrator(args, kwargs) (line 382)
        find_integrator_call_result_35568 = invoke(stypy.reporting.localization.Localization(__file__, 382, 21), find_integrator_35565, *[name_35566], **kwargs_35567)
        
        # Assigning a type to the variable 'integrator' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'integrator', find_integrator_call_result_35568)
        
        # Type idiom detected: calculating its left and rigth part (line 383)
        # Getting the type of 'integrator' (line 383)
        integrator_35569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 11), 'integrator')
        # Getting the type of 'None' (line 383)
        None_35570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 25), 'None')
        
        (may_be_35571, more_types_in_union_35572) = may_be_none(integrator_35569, None_35570)

        if may_be_35571:

            if more_types_in_union_35572:
                # Runtime conditional SSA (line 383)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to warn(...): (line 386)
            # Processing the call arguments (line 386)
            str_35575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 26), 'str', 'No integrator name match with %r or is not available.')
            # Getting the type of 'name' (line 387)
            name_35576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 41), 'name', False)
            # Applying the binary operator '%' (line 386)
            result_mod_35577 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 26), '%', str_35575, name_35576)
            
            # Processing the call keyword arguments (line 386)
            kwargs_35578 = {}
            # Getting the type of 'warnings' (line 386)
            warnings_35573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 386)
            warn_35574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 12), warnings_35573, 'warn')
            # Calling warn(args, kwargs) (line 386)
            warn_call_result_35579 = invoke(stypy.reporting.localization.Localization(__file__, 386, 12), warn_35574, *[result_mod_35577], **kwargs_35578)
            

            if more_types_in_union_35572:
                # Runtime conditional SSA for else branch (line 383)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_35571) or more_types_in_union_35572):
            
            # Assigning a Call to a Attribute (line 389):
            
            # Assigning a Call to a Attribute (line 389):
            
            # Call to integrator(...): (line 389)
            # Processing the call keyword arguments (line 389)
            # Getting the type of 'integrator_params' (line 389)
            integrator_params_35581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 44), 'integrator_params', False)
            kwargs_35582 = {'integrator_params_35581': integrator_params_35581}
            # Getting the type of 'integrator' (line 389)
            integrator_35580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 31), 'integrator', False)
            # Calling integrator(args, kwargs) (line 389)
            integrator_call_result_35583 = invoke(stypy.reporting.localization.Localization(__file__, 389, 31), integrator_35580, *[], **kwargs_35582)
            
            # Getting the type of 'self' (line 389)
            self_35584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'self')
            # Setting the type of the member '_integrator' of a type (line 389)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), self_35584, '_integrator', integrator_call_result_35583)
            
            
            
            # Call to len(...): (line 390)
            # Processing the call arguments (line 390)
            # Getting the type of 'self' (line 390)
            self_35586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 23), 'self', False)
            # Obtaining the member '_y' of a type (line 390)
            _y_35587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 23), self_35586, '_y')
            # Processing the call keyword arguments (line 390)
            kwargs_35588 = {}
            # Getting the type of 'len' (line 390)
            len_35585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'len', False)
            # Calling len(args, kwargs) (line 390)
            len_call_result_35589 = invoke(stypy.reporting.localization.Localization(__file__, 390, 19), len_35585, *[_y_35587], **kwargs_35588)
            
            # Applying the 'not' unary operator (line 390)
            result_not__35590 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 15), 'not', len_call_result_35589)
            
            # Testing the type of an if condition (line 390)
            if_condition_35591 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 390, 12), result_not__35590)
            # Assigning a type to the variable 'if_condition_35591' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'if_condition_35591', if_condition_35591)
            # SSA begins for if statement (line 390)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Attribute (line 391):
            
            # Assigning a Num to a Attribute (line 391):
            float_35592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 25), 'float')
            # Getting the type of 'self' (line 391)
            self_35593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 16), 'self')
            # Setting the type of the member 't' of a type (line 391)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 16), self_35593, 't', float_35592)
            
            # Assigning a Call to a Attribute (line 392):
            
            # Assigning a Call to a Attribute (line 392):
            
            # Call to array(...): (line 392)
            # Processing the call arguments (line 392)
            
            # Obtaining an instance of the builtin type 'list' (line 392)
            list_35595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 32), 'list')
            # Adding type elements to the builtin type 'list' instance (line 392)
            # Adding element type (line 392)
            float_35596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 33), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 32), list_35595, float_35596)
            
            # Getting the type of 'self' (line 392)
            self_35597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 39), 'self', False)
            # Obtaining the member '_integrator' of a type (line 392)
            _integrator_35598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 39), self_35597, '_integrator')
            # Obtaining the member 'scalar' of a type (line 392)
            scalar_35599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 39), _integrator_35598, 'scalar')
            # Processing the call keyword arguments (line 392)
            kwargs_35600 = {}
            # Getting the type of 'array' (line 392)
            array_35594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 26), 'array', False)
            # Calling array(args, kwargs) (line 392)
            array_call_result_35601 = invoke(stypy.reporting.localization.Localization(__file__, 392, 26), array_35594, *[list_35595, scalar_35599], **kwargs_35600)
            
            # Getting the type of 'self' (line 392)
            self_35602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'self')
            # Setting the type of the member '_y' of a type (line 392)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 16), self_35602, '_y', array_call_result_35601)
            # SSA join for if statement (line 390)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to reset(...): (line 393)
            # Processing the call arguments (line 393)
            
            # Call to len(...): (line 393)
            # Processing the call arguments (line 393)
            # Getting the type of 'self' (line 393)
            self_35607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 39), 'self', False)
            # Obtaining the member '_y' of a type (line 393)
            _y_35608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 39), self_35607, '_y')
            # Processing the call keyword arguments (line 393)
            kwargs_35609 = {}
            # Getting the type of 'len' (line 393)
            len_35606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 35), 'len', False)
            # Calling len(args, kwargs) (line 393)
            len_call_result_35610 = invoke(stypy.reporting.localization.Localization(__file__, 393, 35), len_35606, *[_y_35608], **kwargs_35609)
            
            
            # Getting the type of 'self' (line 393)
            self_35611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 49), 'self', False)
            # Obtaining the member 'jac' of a type (line 393)
            jac_35612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 49), self_35611, 'jac')
            # Getting the type of 'None' (line 393)
            None_35613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 65), 'None', False)
            # Applying the binary operator 'isnot' (line 393)
            result_is_not_35614 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 49), 'isnot', jac_35612, None_35613)
            
            # Processing the call keyword arguments (line 393)
            kwargs_35615 = {}
            # Getting the type of 'self' (line 393)
            self_35603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'self', False)
            # Obtaining the member '_integrator' of a type (line 393)
            _integrator_35604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 12), self_35603, '_integrator')
            # Obtaining the member 'reset' of a type (line 393)
            reset_35605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 12), _integrator_35604, 'reset')
            # Calling reset(args, kwargs) (line 393)
            reset_call_result_35616 = invoke(stypy.reporting.localization.Localization(__file__, 393, 12), reset_35605, *[len_call_result_35610, result_is_not_35614], **kwargs_35615)
            

            if (may_be_35571 and more_types_in_union_35572):
                # SSA join for if statement (line 383)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 394)
        self_35617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'stypy_return_type', self_35617)
        
        # ################# End of 'set_integrator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_integrator' in the type store
        # Getting the type of 'stypy_return_type' (line 371)
        stypy_return_type_35618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35618)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_integrator'
        return stypy_return_type_35618


    @norecursion
    def integrate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 396)
        False_35619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 32), 'False')
        # Getting the type of 'False' (line 396)
        False_35620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 45), 'False')
        defaults = [False_35619, False_35620]
        # Create a new context for function 'integrate'
        module_type_store = module_type_store.open_function_context('integrate', 396, 4, False)
        # Assigning a type to the variable 'self' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ode.integrate.__dict__.__setitem__('stypy_localization', localization)
        ode.integrate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ode.integrate.__dict__.__setitem__('stypy_type_store', module_type_store)
        ode.integrate.__dict__.__setitem__('stypy_function_name', 'ode.integrate')
        ode.integrate.__dict__.__setitem__('stypy_param_names_list', ['t', 'step', 'relax'])
        ode.integrate.__dict__.__setitem__('stypy_varargs_param_name', None)
        ode.integrate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ode.integrate.__dict__.__setitem__('stypy_call_defaults', defaults)
        ode.integrate.__dict__.__setitem__('stypy_call_varargs', varargs)
        ode.integrate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ode.integrate.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ode.integrate', ['t', 'step', 'relax'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'integrate', localization, ['t', 'step', 'relax'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'integrate(...)' code ##################

        str_35621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, (-1)), 'str', 'Find y=y(t), set y as an initial condition, and return y.\n\n        Parameters\n        ----------\n        t : float\n            The endpoint of the integration step.\n        step : bool\n            If True, and if the integrator supports the step method,\n            then perform a single integration step and return.\n            This parameter is provided in order to expose internals of\n            the implementation, and should not be changed from its default\n            value in most cases.\n        relax : bool\n            If True and if the integrator supports the run_relax method,\n            then integrate until t_1 >= t and return. ``relax`` is not\n            referenced if ``step=True``.\n            This parameter is provided in order to expose internals of\n            the implementation, and should not be changed from its default\n            value in most cases.\n\n        Returns\n        -------\n        y : float\n            The integrated value at t\n        ')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'step' (line 422)
        step_35622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 11), 'step')
        # Getting the type of 'self' (line 422)
        self_35623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 20), 'self')
        # Obtaining the member '_integrator' of a type (line 422)
        _integrator_35624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 20), self_35623, '_integrator')
        # Obtaining the member 'supports_step' of a type (line 422)
        supports_step_35625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 20), _integrator_35624, 'supports_step')
        # Applying the binary operator 'and' (line 422)
        result_and_keyword_35626 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 11), 'and', step_35622, supports_step_35625)
        
        # Testing the type of an if condition (line 422)
        if_condition_35627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 8), result_and_keyword_35626)
        # Assigning a type to the variable 'if_condition_35627' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'if_condition_35627', if_condition_35627)
        # SSA begins for if statement (line 422)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 423):
        
        # Assigning a Attribute to a Name (line 423):
        # Getting the type of 'self' (line 423)
        self_35628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 18), 'self')
        # Obtaining the member '_integrator' of a type (line 423)
        _integrator_35629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 18), self_35628, '_integrator')
        # Obtaining the member 'step' of a type (line 423)
        step_35630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 18), _integrator_35629, 'step')
        # Assigning a type to the variable 'mth' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'mth', step_35630)
        # SSA branch for the else part of an if statement (line 422)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'relax' (line 424)
        relax_35631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 13), 'relax')
        # Getting the type of 'self' (line 424)
        self_35632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 23), 'self')
        # Obtaining the member '_integrator' of a type (line 424)
        _integrator_35633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 23), self_35632, '_integrator')
        # Obtaining the member 'supports_run_relax' of a type (line 424)
        supports_run_relax_35634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 23), _integrator_35633, 'supports_run_relax')
        # Applying the binary operator 'and' (line 424)
        result_and_keyword_35635 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 13), 'and', relax_35631, supports_run_relax_35634)
        
        # Testing the type of an if condition (line 424)
        if_condition_35636 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 424, 13), result_and_keyword_35635)
        # Assigning a type to the variable 'if_condition_35636' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 13), 'if_condition_35636', if_condition_35636)
        # SSA begins for if statement (line 424)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 425):
        
        # Assigning a Attribute to a Name (line 425):
        # Getting the type of 'self' (line 425)
        self_35637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 18), 'self')
        # Obtaining the member '_integrator' of a type (line 425)
        _integrator_35638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 18), self_35637, '_integrator')
        # Obtaining the member 'run_relax' of a type (line 425)
        run_relax_35639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 18), _integrator_35638, 'run_relax')
        # Assigning a type to the variable 'mth' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'mth', run_relax_35639)
        # SSA branch for the else part of an if statement (line 424)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 427):
        
        # Assigning a Attribute to a Name (line 427):
        # Getting the type of 'self' (line 427)
        self_35640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 18), 'self')
        # Obtaining the member '_integrator' of a type (line 427)
        _integrator_35641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 18), self_35640, '_integrator')
        # Obtaining the member 'run' of a type (line 427)
        run_35642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 18), _integrator_35641, 'run')
        # Assigning a type to the variable 'mth' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'mth', run_35642)
        # SSA join for if statement (line 424)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 422)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 429)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Tuple (line 430):
        
        # Assigning a Subscript to a Name (line 430):
        
        # Obtaining the type of the subscript
        int_35643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 12), 'int')
        
        # Call to mth(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'self' (line 430)
        self_35645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 34), 'self', False)
        # Obtaining the member 'f' of a type (line 430)
        f_35646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 34), self_35645, 'f')
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 430)
        self_35647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 42), 'self', False)
        # Obtaining the member 'jac' of a type (line 430)
        jac_35648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 42), self_35647, 'jac')

        @norecursion
        def _stypy_temp_lambda_3(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_3'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_3', 430, 55, True)
            # Passed parameters checking function
            _stypy_temp_lambda_3.stypy_localization = localization
            _stypy_temp_lambda_3.stypy_type_of_self = None
            _stypy_temp_lambda_3.stypy_type_store = module_type_store
            _stypy_temp_lambda_3.stypy_function_name = '_stypy_temp_lambda_3'
            _stypy_temp_lambda_3.stypy_param_names_list = []
            _stypy_temp_lambda_3.stypy_varargs_param_name = None
            _stypy_temp_lambda_3.stypy_kwargs_param_name = None
            _stypy_temp_lambda_3.stypy_call_defaults = defaults
            _stypy_temp_lambda_3.stypy_call_varargs = varargs
            _stypy_temp_lambda_3.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_3', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_3', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 430)
            None_35649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 63), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 430)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 55), 'stypy_return_type', None_35649)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_3' in the type store
            # Getting the type of 'stypy_return_type' (line 430)
            stypy_return_type_35650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 55), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_35650)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_3'
            return stypy_return_type_35650

        # Assigning a type to the variable '_stypy_temp_lambda_3' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 55), '_stypy_temp_lambda_3', _stypy_temp_lambda_3)
        # Getting the type of '_stypy_temp_lambda_3' (line 430)
        _stypy_temp_lambda_3_35651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 55), '_stypy_temp_lambda_3')
        # Applying the binary operator 'or' (line 430)
        result_or_keyword_35652 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 42), 'or', jac_35648, _stypy_temp_lambda_3_35651)
        
        # Getting the type of 'self' (line 431)
        self_35653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 34), 'self', False)
        # Obtaining the member '_y' of a type (line 431)
        _y_35654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 34), self_35653, '_y')
        # Getting the type of 'self' (line 431)
        self_35655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 43), 'self', False)
        # Obtaining the member 't' of a type (line 431)
        t_35656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 43), self_35655, 't')
        # Getting the type of 't' (line 431)
        t_35657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 51), 't', False)
        # Getting the type of 'self' (line 432)
        self_35658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 34), 'self', False)
        # Obtaining the member 'f_params' of a type (line 432)
        f_params_35659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 34), self_35658, 'f_params')
        # Getting the type of 'self' (line 432)
        self_35660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 49), 'self', False)
        # Obtaining the member 'jac_params' of a type (line 432)
        jac_params_35661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 49), self_35660, 'jac_params')
        # Processing the call keyword arguments (line 430)
        kwargs_35662 = {}
        # Getting the type of 'mth' (line 430)
        mth_35644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 30), 'mth', False)
        # Calling mth(args, kwargs) (line 430)
        mth_call_result_35663 = invoke(stypy.reporting.localization.Localization(__file__, 430, 30), mth_35644, *[f_35646, result_or_keyword_35652, _y_35654, t_35656, t_35657, f_params_35659, jac_params_35661], **kwargs_35662)
        
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___35664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), mth_call_result_35663, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_35665 = invoke(stypy.reporting.localization.Localization(__file__, 430, 12), getitem___35664, int_35643)
        
        # Assigning a type to the variable 'tuple_var_assignment_35473' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'tuple_var_assignment_35473', subscript_call_result_35665)
        
        # Assigning a Subscript to a Name (line 430):
        
        # Obtaining the type of the subscript
        int_35666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 12), 'int')
        
        # Call to mth(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'self' (line 430)
        self_35668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 34), 'self', False)
        # Obtaining the member 'f' of a type (line 430)
        f_35669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 34), self_35668, 'f')
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 430)
        self_35670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 42), 'self', False)
        # Obtaining the member 'jac' of a type (line 430)
        jac_35671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 42), self_35670, 'jac')

        @norecursion
        def _stypy_temp_lambda_4(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_4'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_4', 430, 55, True)
            # Passed parameters checking function
            _stypy_temp_lambda_4.stypy_localization = localization
            _stypy_temp_lambda_4.stypy_type_of_self = None
            _stypy_temp_lambda_4.stypy_type_store = module_type_store
            _stypy_temp_lambda_4.stypy_function_name = '_stypy_temp_lambda_4'
            _stypy_temp_lambda_4.stypy_param_names_list = []
            _stypy_temp_lambda_4.stypy_varargs_param_name = None
            _stypy_temp_lambda_4.stypy_kwargs_param_name = None
            _stypy_temp_lambda_4.stypy_call_defaults = defaults
            _stypy_temp_lambda_4.stypy_call_varargs = varargs
            _stypy_temp_lambda_4.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_4', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_4', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 430)
            None_35672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 63), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 430)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 55), 'stypy_return_type', None_35672)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_4' in the type store
            # Getting the type of 'stypy_return_type' (line 430)
            stypy_return_type_35673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 55), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_35673)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_4'
            return stypy_return_type_35673

        # Assigning a type to the variable '_stypy_temp_lambda_4' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 55), '_stypy_temp_lambda_4', _stypy_temp_lambda_4)
        # Getting the type of '_stypy_temp_lambda_4' (line 430)
        _stypy_temp_lambda_4_35674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 55), '_stypy_temp_lambda_4')
        # Applying the binary operator 'or' (line 430)
        result_or_keyword_35675 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 42), 'or', jac_35671, _stypy_temp_lambda_4_35674)
        
        # Getting the type of 'self' (line 431)
        self_35676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 34), 'self', False)
        # Obtaining the member '_y' of a type (line 431)
        _y_35677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 34), self_35676, '_y')
        # Getting the type of 'self' (line 431)
        self_35678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 43), 'self', False)
        # Obtaining the member 't' of a type (line 431)
        t_35679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 43), self_35678, 't')
        # Getting the type of 't' (line 431)
        t_35680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 51), 't', False)
        # Getting the type of 'self' (line 432)
        self_35681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 34), 'self', False)
        # Obtaining the member 'f_params' of a type (line 432)
        f_params_35682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 34), self_35681, 'f_params')
        # Getting the type of 'self' (line 432)
        self_35683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 49), 'self', False)
        # Obtaining the member 'jac_params' of a type (line 432)
        jac_params_35684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 49), self_35683, 'jac_params')
        # Processing the call keyword arguments (line 430)
        kwargs_35685 = {}
        # Getting the type of 'mth' (line 430)
        mth_35667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 30), 'mth', False)
        # Calling mth(args, kwargs) (line 430)
        mth_call_result_35686 = invoke(stypy.reporting.localization.Localization(__file__, 430, 30), mth_35667, *[f_35669, result_or_keyword_35675, _y_35677, t_35679, t_35680, f_params_35682, jac_params_35684], **kwargs_35685)
        
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___35687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), mth_call_result_35686, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_35688 = invoke(stypy.reporting.localization.Localization(__file__, 430, 12), getitem___35687, int_35666)
        
        # Assigning a type to the variable 'tuple_var_assignment_35474' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'tuple_var_assignment_35474', subscript_call_result_35688)
        
        # Assigning a Name to a Attribute (line 430):
        # Getting the type of 'tuple_var_assignment_35473' (line 430)
        tuple_var_assignment_35473_35689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'tuple_var_assignment_35473')
        # Getting the type of 'self' (line 430)
        self_35690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'self')
        # Setting the type of the member '_y' of a type (line 430)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), self_35690, '_y', tuple_var_assignment_35473_35689)
        
        # Assigning a Name to a Attribute (line 430):
        # Getting the type of 'tuple_var_assignment_35474' (line 430)
        tuple_var_assignment_35474_35691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'tuple_var_assignment_35474')
        # Getting the type of 'self' (line 430)
        self_35692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 21), 'self')
        # Setting the type of the member 't' of a type (line 430)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 21), self_35692, 't', tuple_var_assignment_35474_35691)
        # SSA branch for the except part of a try statement (line 429)
        # SSA branch for the except 'SystemError' branch of a try statement (line 429)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 435)
        # Processing the call arguments (line 435)
        str_35694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 29), 'str', 'Function to integrate must not return a tuple.')
        # Processing the call keyword arguments (line 435)
        kwargs_35695 = {}
        # Getting the type of 'ValueError' (line 435)
        ValueError_35693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 435)
        ValueError_call_result_35696 = invoke(stypy.reporting.localization.Localization(__file__, 435, 18), ValueError_35693, *[str_35694], **kwargs_35695)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 435, 12), ValueError_call_result_35696, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 429)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 437)
        self_35697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'self')
        # Obtaining the member '_y' of a type (line 437)
        _y_35698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 15), self_35697, '_y')
        # Assigning a type to the variable 'stypy_return_type' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'stypy_return_type', _y_35698)
        
        # ################# End of 'integrate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'integrate' in the type store
        # Getting the type of 'stypy_return_type' (line 396)
        stypy_return_type_35699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35699)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'integrate'
        return stypy_return_type_35699


    @norecursion
    def successful(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'successful'
        module_type_store = module_type_store.open_function_context('successful', 439, 4, False)
        # Assigning a type to the variable 'self' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ode.successful.__dict__.__setitem__('stypy_localization', localization)
        ode.successful.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ode.successful.__dict__.__setitem__('stypy_type_store', module_type_store)
        ode.successful.__dict__.__setitem__('stypy_function_name', 'ode.successful')
        ode.successful.__dict__.__setitem__('stypy_param_names_list', [])
        ode.successful.__dict__.__setitem__('stypy_varargs_param_name', None)
        ode.successful.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ode.successful.__dict__.__setitem__('stypy_call_defaults', defaults)
        ode.successful.__dict__.__setitem__('stypy_call_varargs', varargs)
        ode.successful.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ode.successful.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ode.successful', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'successful', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'successful(...)' code ##################

        str_35700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 8), 'str', 'Check if integration was successful.')
        
        
        # SSA begins for try-except statement (line 441)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Getting the type of 'self' (line 442)
        self_35701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'self')
        # Obtaining the member '_integrator' of a type (line 442)
        _integrator_35702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 12), self_35701, '_integrator')
        # SSA branch for the except part of a try statement (line 441)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 441)
        module_type_store.open_ssa_branch('except')
        
        # Call to set_integrator(...): (line 444)
        # Processing the call arguments (line 444)
        str_35705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 32), 'str', '')
        # Processing the call keyword arguments (line 444)
        kwargs_35706 = {}
        # Getting the type of 'self' (line 444)
        self_35703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'self', False)
        # Obtaining the member 'set_integrator' of a type (line 444)
        set_integrator_35704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), self_35703, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 444)
        set_integrator_call_result_35707 = invoke(stypy.reporting.localization.Localization(__file__, 444, 12), set_integrator_35704, *[str_35705], **kwargs_35706)
        
        # SSA join for try-except statement (line 441)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 445)
        self_35708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 15), 'self')
        # Obtaining the member '_integrator' of a type (line 445)
        _integrator_35709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 15), self_35708, '_integrator')
        # Obtaining the member 'success' of a type (line 445)
        success_35710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 15), _integrator_35709, 'success')
        int_35711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 43), 'int')
        # Applying the binary operator '==' (line 445)
        result_eq_35712 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 15), '==', success_35710, int_35711)
        
        # Assigning a type to the variable 'stypy_return_type' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'stypy_return_type', result_eq_35712)
        
        # ################# End of 'successful(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'successful' in the type store
        # Getting the type of 'stypy_return_type' (line 439)
        stypy_return_type_35713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35713)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'successful'
        return stypy_return_type_35713


    @norecursion
    def get_return_code(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_return_code'
        module_type_store = module_type_store.open_function_context('get_return_code', 447, 4, False)
        # Assigning a type to the variable 'self' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ode.get_return_code.__dict__.__setitem__('stypy_localization', localization)
        ode.get_return_code.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ode.get_return_code.__dict__.__setitem__('stypy_type_store', module_type_store)
        ode.get_return_code.__dict__.__setitem__('stypy_function_name', 'ode.get_return_code')
        ode.get_return_code.__dict__.__setitem__('stypy_param_names_list', [])
        ode.get_return_code.__dict__.__setitem__('stypy_varargs_param_name', None)
        ode.get_return_code.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ode.get_return_code.__dict__.__setitem__('stypy_call_defaults', defaults)
        ode.get_return_code.__dict__.__setitem__('stypy_call_varargs', varargs)
        ode.get_return_code.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ode.get_return_code.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ode.get_return_code', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_return_code', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_return_code(...)' code ##################

        str_35714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, (-1)), 'str', 'Extracts the return code for the integration to enable better control\n        if the integration fails.')
        
        
        # SSA begins for try-except statement (line 450)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Getting the type of 'self' (line 451)
        self_35715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'self')
        # Obtaining the member '_integrator' of a type (line 451)
        _integrator_35716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 12), self_35715, '_integrator')
        # SSA branch for the except part of a try statement (line 450)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 450)
        module_type_store.open_ssa_branch('except')
        
        # Call to set_integrator(...): (line 453)
        # Processing the call arguments (line 453)
        str_35719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 32), 'str', '')
        # Processing the call keyword arguments (line 453)
        kwargs_35720 = {}
        # Getting the type of 'self' (line 453)
        self_35717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'self', False)
        # Obtaining the member 'set_integrator' of a type (line 453)
        set_integrator_35718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 12), self_35717, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 453)
        set_integrator_call_result_35721 = invoke(stypy.reporting.localization.Localization(__file__, 453, 12), set_integrator_35718, *[str_35719], **kwargs_35720)
        
        # SSA join for try-except statement (line 450)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 454)
        self_35722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 15), 'self')
        # Obtaining the member '_integrator' of a type (line 454)
        _integrator_35723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 15), self_35722, '_integrator')
        # Obtaining the member 'istate' of a type (line 454)
        istate_35724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 15), _integrator_35723, 'istate')
        # Assigning a type to the variable 'stypy_return_type' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'stypy_return_type', istate_35724)
        
        # ################# End of 'get_return_code(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_return_code' in the type store
        # Getting the type of 'stypy_return_type' (line 447)
        stypy_return_type_35725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35725)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_return_code'
        return stypy_return_type_35725


    @norecursion
    def set_f_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_f_params'
        module_type_store = module_type_store.open_function_context('set_f_params', 456, 4, False)
        # Assigning a type to the variable 'self' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ode.set_f_params.__dict__.__setitem__('stypy_localization', localization)
        ode.set_f_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ode.set_f_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        ode.set_f_params.__dict__.__setitem__('stypy_function_name', 'ode.set_f_params')
        ode.set_f_params.__dict__.__setitem__('stypy_param_names_list', [])
        ode.set_f_params.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        ode.set_f_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ode.set_f_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        ode.set_f_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        ode.set_f_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ode.set_f_params.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ode.set_f_params', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_f_params', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_f_params(...)' code ##################

        str_35726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 8), 'str', 'Set extra parameters for user-supplied function f.')
        
        # Assigning a Name to a Attribute (line 458):
        
        # Assigning a Name to a Attribute (line 458):
        # Getting the type of 'args' (line 458)
        args_35727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 24), 'args')
        # Getting the type of 'self' (line 458)
        self_35728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'self')
        # Setting the type of the member 'f_params' of a type (line 458)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), self_35728, 'f_params', args_35727)
        # Getting the type of 'self' (line 459)
        self_35729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'stypy_return_type', self_35729)
        
        # ################# End of 'set_f_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_f_params' in the type store
        # Getting the type of 'stypy_return_type' (line 456)
        stypy_return_type_35730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35730)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_f_params'
        return stypy_return_type_35730


    @norecursion
    def set_jac_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_jac_params'
        module_type_store = module_type_store.open_function_context('set_jac_params', 461, 4, False)
        # Assigning a type to the variable 'self' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ode.set_jac_params.__dict__.__setitem__('stypy_localization', localization)
        ode.set_jac_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ode.set_jac_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        ode.set_jac_params.__dict__.__setitem__('stypy_function_name', 'ode.set_jac_params')
        ode.set_jac_params.__dict__.__setitem__('stypy_param_names_list', [])
        ode.set_jac_params.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        ode.set_jac_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ode.set_jac_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        ode.set_jac_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        ode.set_jac_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ode.set_jac_params.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ode.set_jac_params', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_jac_params', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_jac_params(...)' code ##################

        str_35731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 8), 'str', 'Set extra parameters for user-supplied function jac.')
        
        # Assigning a Name to a Attribute (line 463):
        
        # Assigning a Name to a Attribute (line 463):
        # Getting the type of 'args' (line 463)
        args_35732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 26), 'args')
        # Getting the type of 'self' (line 463)
        self_35733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'self')
        # Setting the type of the member 'jac_params' of a type (line 463)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 8), self_35733, 'jac_params', args_35732)
        # Getting the type of 'self' (line 464)
        self_35734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'stypy_return_type', self_35734)
        
        # ################# End of 'set_jac_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_jac_params' in the type store
        # Getting the type of 'stypy_return_type' (line 461)
        stypy_return_type_35735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35735)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_jac_params'
        return stypy_return_type_35735


    @norecursion
    def set_solout(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_solout'
        module_type_store = module_type_store.open_function_context('set_solout', 466, 4, False)
        # Assigning a type to the variable 'self' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ode.set_solout.__dict__.__setitem__('stypy_localization', localization)
        ode.set_solout.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ode.set_solout.__dict__.__setitem__('stypy_type_store', module_type_store)
        ode.set_solout.__dict__.__setitem__('stypy_function_name', 'ode.set_solout')
        ode.set_solout.__dict__.__setitem__('stypy_param_names_list', ['solout'])
        ode.set_solout.__dict__.__setitem__('stypy_varargs_param_name', None)
        ode.set_solout.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ode.set_solout.__dict__.__setitem__('stypy_call_defaults', defaults)
        ode.set_solout.__dict__.__setitem__('stypy_call_varargs', varargs)
        ode.set_solout.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ode.set_solout.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ode.set_solout', ['solout'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_solout', localization, ['solout'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_solout(...)' code ##################

        str_35736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, (-1)), 'str', '\n        Set callable to be called at every successful integration step.\n\n        Parameters\n        ----------\n        solout : callable\n            ``solout(t, y)`` is called at each internal integrator step,\n            t is a scalar providing the current independent position\n            y is the current soloution ``y.shape == (n,)``\n            solout should return -1 to stop integration\n            otherwise it should return None or 0\n\n        ')
        
        # Getting the type of 'self' (line 480)
        self_35737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 11), 'self')
        # Obtaining the member '_integrator' of a type (line 480)
        _integrator_35738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 11), self_35737, '_integrator')
        # Obtaining the member 'supports_solout' of a type (line 480)
        supports_solout_35739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 11), _integrator_35738, 'supports_solout')
        # Testing the type of an if condition (line 480)
        if_condition_35740 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 480, 8), supports_solout_35739)
        # Assigning a type to the variable 'if_condition_35740' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'if_condition_35740', if_condition_35740)
        # SSA begins for if statement (line 480)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_solout(...): (line 481)
        # Processing the call arguments (line 481)
        # Getting the type of 'solout' (line 481)
        solout_35744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 40), 'solout', False)
        # Processing the call keyword arguments (line 481)
        kwargs_35745 = {}
        # Getting the type of 'self' (line 481)
        self_35741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'self', False)
        # Obtaining the member '_integrator' of a type (line 481)
        _integrator_35742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 12), self_35741, '_integrator')
        # Obtaining the member 'set_solout' of a type (line 481)
        set_solout_35743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 12), _integrator_35742, 'set_solout')
        # Calling set_solout(args, kwargs) (line 481)
        set_solout_call_result_35746 = invoke(stypy.reporting.localization.Localization(__file__, 481, 12), set_solout_35743, *[solout_35744], **kwargs_35745)
        
        
        
        # Getting the type of 'self' (line 482)
        self_35747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 15), 'self')
        # Obtaining the member '_y' of a type (line 482)
        _y_35748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 15), self_35747, '_y')
        # Getting the type of 'None' (line 482)
        None_35749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 30), 'None')
        # Applying the binary operator 'isnot' (line 482)
        result_is_not_35750 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 15), 'isnot', _y_35748, None_35749)
        
        # Testing the type of an if condition (line 482)
        if_condition_35751 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 482, 12), result_is_not_35750)
        # Assigning a type to the variable 'if_condition_35751' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'if_condition_35751', if_condition_35751)
        # SSA begins for if statement (line 482)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to reset(...): (line 483)
        # Processing the call arguments (line 483)
        
        # Call to len(...): (line 483)
        # Processing the call arguments (line 483)
        # Getting the type of 'self' (line 483)
        self_35756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 43), 'self', False)
        # Obtaining the member '_y' of a type (line 483)
        _y_35757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 43), self_35756, '_y')
        # Processing the call keyword arguments (line 483)
        kwargs_35758 = {}
        # Getting the type of 'len' (line 483)
        len_35755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 39), 'len', False)
        # Calling len(args, kwargs) (line 483)
        len_call_result_35759 = invoke(stypy.reporting.localization.Localization(__file__, 483, 39), len_35755, *[_y_35757], **kwargs_35758)
        
        
        # Getting the type of 'self' (line 483)
        self_35760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 53), 'self', False)
        # Obtaining the member 'jac' of a type (line 483)
        jac_35761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 53), self_35760, 'jac')
        # Getting the type of 'None' (line 483)
        None_35762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 69), 'None', False)
        # Applying the binary operator 'isnot' (line 483)
        result_is_not_35763 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 53), 'isnot', jac_35761, None_35762)
        
        # Processing the call keyword arguments (line 483)
        kwargs_35764 = {}
        # Getting the type of 'self' (line 483)
        self_35752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 16), 'self', False)
        # Obtaining the member '_integrator' of a type (line 483)
        _integrator_35753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 16), self_35752, '_integrator')
        # Obtaining the member 'reset' of a type (line 483)
        reset_35754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 16), _integrator_35753, 'reset')
        # Calling reset(args, kwargs) (line 483)
        reset_call_result_35765 = invoke(stypy.reporting.localization.Localization(__file__, 483, 16), reset_35754, *[len_call_result_35759, result_is_not_35763], **kwargs_35764)
        
        # SSA join for if statement (line 482)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 480)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 485)
        # Processing the call arguments (line 485)
        str_35767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 29), 'str', 'selected integrator does not support solout, choose another one')
        # Processing the call keyword arguments (line 485)
        kwargs_35768 = {}
        # Getting the type of 'ValueError' (line 485)
        ValueError_35766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 485)
        ValueError_call_result_35769 = invoke(stypy.reporting.localization.Localization(__file__, 485, 18), ValueError_35766, *[str_35767], **kwargs_35768)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 485, 12), ValueError_call_result_35769, 'raise parameter', BaseException)
        # SSA join for if statement (line 480)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_solout(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_solout' in the type store
        # Getting the type of 'stypy_return_type' (line 466)
        stypy_return_type_35770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35770)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_solout'
        return stypy_return_type_35770


# Assigning a type to the variable 'ode' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'ode', ode)

@norecursion
def _transform_banded_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_transform_banded_jac'
    module_type_store = module_type_store.open_function_context('_transform_banded_jac', 489, 0, False)
    
    # Passed parameters checking function
    _transform_banded_jac.stypy_localization = localization
    _transform_banded_jac.stypy_type_of_self = None
    _transform_banded_jac.stypy_type_store = module_type_store
    _transform_banded_jac.stypy_function_name = '_transform_banded_jac'
    _transform_banded_jac.stypy_param_names_list = ['bjac']
    _transform_banded_jac.stypy_varargs_param_name = None
    _transform_banded_jac.stypy_kwargs_param_name = None
    _transform_banded_jac.stypy_call_defaults = defaults
    _transform_banded_jac.stypy_call_varargs = varargs
    _transform_banded_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_transform_banded_jac', ['bjac'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_transform_banded_jac', localization, ['bjac'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_transform_banded_jac(...)' code ##################

    str_35771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, (-1)), 'str', '\n    Convert a real matrix of the form (for example)\n\n        [0 0 A B]        [0 0 0 B]\n        [0 0 C D]        [0 0 A D]\n        [E F G H]   to   [0 F C H]\n        [I J K L]        [E J G L]\n                         [I 0 K 0]\n\n    That is, every other column is shifted up one.\n    ')
    
    # Assigning a Call to a Name (line 502):
    
    # Assigning a Call to a Name (line 502):
    
    # Call to zeros(...): (line 502)
    # Processing the call arguments (line 502)
    
    # Obtaining an instance of the builtin type 'tuple' (line 502)
    tuple_35773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 502)
    # Adding element type (line 502)
    
    # Obtaining the type of the subscript
    int_35774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 31), 'int')
    # Getting the type of 'bjac' (line 502)
    bjac_35775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 20), 'bjac', False)
    # Obtaining the member 'shape' of a type (line 502)
    shape_35776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 20), bjac_35775, 'shape')
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___35777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 20), shape_35776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_35778 = invoke(stypy.reporting.localization.Localization(__file__, 502, 20), getitem___35777, int_35774)
    
    int_35779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 36), 'int')
    # Applying the binary operator '+' (line 502)
    result_add_35780 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 20), '+', subscript_call_result_35778, int_35779)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 20), tuple_35773, result_add_35780)
    # Adding element type (line 502)
    
    # Obtaining the type of the subscript
    int_35781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 50), 'int')
    # Getting the type of 'bjac' (line 502)
    bjac_35782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 39), 'bjac', False)
    # Obtaining the member 'shape' of a type (line 502)
    shape_35783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 39), bjac_35782, 'shape')
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___35784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 39), shape_35783, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_35785 = invoke(stypy.reporting.localization.Localization(__file__, 502, 39), getitem___35784, int_35781)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 20), tuple_35773, subscript_call_result_35785)
    
    # Processing the call keyword arguments (line 502)
    kwargs_35786 = {}
    # Getting the type of 'zeros' (line 502)
    zeros_35772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 13), 'zeros', False)
    # Calling zeros(args, kwargs) (line 502)
    zeros_call_result_35787 = invoke(stypy.reporting.localization.Localization(__file__, 502, 13), zeros_35772, *[tuple_35773], **kwargs_35786)
    
    # Assigning a type to the variable 'newjac' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'newjac', zeros_call_result_35787)
    
    # Assigning a Subscript to a Subscript (line 503):
    
    # Assigning a Subscript to a Subscript (line 503):
    
    # Obtaining the type of the subscript
    slice_35788 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 503, 22), None, None, None)
    int_35789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 32), 'int')
    slice_35790 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 503, 22), None, None, int_35789)
    # Getting the type of 'bjac' (line 503)
    bjac_35791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 22), 'bjac')
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___35792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 22), bjac_35791, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_35793 = invoke(stypy.reporting.localization.Localization(__file__, 503, 22), getitem___35792, (slice_35788, slice_35790))
    
    # Getting the type of 'newjac' (line 503)
    newjac_35794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'newjac')
    int_35795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 11), 'int')
    slice_35796 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 503, 4), int_35795, None, None)
    int_35797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 17), 'int')
    slice_35798 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 503, 4), None, None, int_35797)
    # Storing an element on a container (line 503)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 4), newjac_35794, ((slice_35796, slice_35798), subscript_call_result_35793))
    
    # Assigning a Subscript to a Subscript (line 504):
    
    # Assigning a Subscript to a Subscript (line 504):
    
    # Obtaining the type of the subscript
    slice_35799 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 504, 24), None, None, None)
    int_35800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 32), 'int')
    int_35801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 35), 'int')
    slice_35802 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 504, 24), int_35800, None, int_35801)
    # Getting the type of 'bjac' (line 504)
    bjac_35803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 24), 'bjac')
    # Obtaining the member '__getitem__' of a type (line 504)
    getitem___35804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 24), bjac_35803, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 504)
    subscript_call_result_35805 = invoke(stypy.reporting.localization.Localization(__file__, 504, 24), getitem___35804, (slice_35799, slice_35802))
    
    # Getting the type of 'newjac' (line 504)
    newjac_35806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'newjac')
    int_35807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 12), 'int')
    slice_35808 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 504, 4), None, int_35807, None)
    int_35809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 16), 'int')
    int_35810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 19), 'int')
    slice_35811 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 504, 4), int_35809, None, int_35810)
    # Storing an element on a container (line 504)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 4), newjac_35806, ((slice_35808, slice_35811), subscript_call_result_35805))
    # Getting the type of 'newjac' (line 505)
    newjac_35812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 11), 'newjac')
    # Assigning a type to the variable 'stypy_return_type' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'stypy_return_type', newjac_35812)
    
    # ################# End of '_transform_banded_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_transform_banded_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 489)
    stypy_return_type_35813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35813)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_transform_banded_jac'
    return stypy_return_type_35813

# Assigning a type to the variable '_transform_banded_jac' (line 489)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 0), '_transform_banded_jac', _transform_banded_jac)
# Declaration of the 'complex_ode' class
# Getting the type of 'ode' (line 508)
ode_35814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 18), 'ode')

class complex_ode(ode_35814, ):
    str_35815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, (-1)), 'str', '\n    A wrapper of ode for complex systems.\n\n    This functions similarly as `ode`, but re-maps a complex-valued\n    equation system to a real-valued one before using the integrators.\n\n    Parameters\n    ----------\n    f : callable ``f(t, y, *f_args)``\n        Rhs of the equation. t is a scalar, ``y.shape == (n,)``.\n        ``f_args`` is set by calling ``set_f_params(*args)``.\n    jac : callable ``jac(t, y, *jac_args)``\n        Jacobian of the rhs, ``jac[i,j] = d f[i] / d y[j]``.\n        ``jac_args`` is set by calling ``set_f_params(*args)``.\n\n    Attributes\n    ----------\n    t : float\n        Current time.\n    y : ndarray\n        Current variable values.\n\n    Examples\n    --------\n    For usage examples, see `ode`.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 537)
        None_35816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 30), 'None')
        defaults = [None_35816]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 537, 4, False)
        # Assigning a type to the variable 'self' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'complex_ode.__init__', ['f', 'jac'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['f', 'jac'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 538):
        
        # Assigning a Name to a Attribute (line 538):
        # Getting the type of 'f' (line 538)
        f_35817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 18), 'f')
        # Getting the type of 'self' (line 538)
        self_35818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'self')
        # Setting the type of the member 'cf' of a type (line 538)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 8), self_35818, 'cf', f_35817)
        
        # Assigning a Name to a Attribute (line 539):
        
        # Assigning a Name to a Attribute (line 539):
        # Getting the type of 'jac' (line 539)
        jac_35819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 20), 'jac')
        # Getting the type of 'self' (line 539)
        self_35820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'self')
        # Setting the type of the member 'cjac' of a type (line 539)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 8), self_35820, 'cjac', jac_35819)
        
        # Type idiom detected: calculating its left and rigth part (line 540)
        # Getting the type of 'jac' (line 540)
        jac_35821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 11), 'jac')
        # Getting the type of 'None' (line 540)
        None_35822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 18), 'None')
        
        (may_be_35823, more_types_in_union_35824) = may_be_none(jac_35821, None_35822)

        if may_be_35823:

            if more_types_in_union_35824:
                # Runtime conditional SSA (line 540)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to __init__(...): (line 541)
            # Processing the call arguments (line 541)
            # Getting the type of 'self' (line 541)
            self_35827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 25), 'self', False)
            # Getting the type of 'self' (line 541)
            self_35828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 31), 'self', False)
            # Obtaining the member '_wrap' of a type (line 541)
            _wrap_35829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 31), self_35828, '_wrap')
            # Getting the type of 'None' (line 541)
            None_35830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 43), 'None', False)
            # Processing the call keyword arguments (line 541)
            kwargs_35831 = {}
            # Getting the type of 'ode' (line 541)
            ode_35825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'ode', False)
            # Obtaining the member '__init__' of a type (line 541)
            init___35826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 12), ode_35825, '__init__')
            # Calling __init__(args, kwargs) (line 541)
            init___call_result_35832 = invoke(stypy.reporting.localization.Localization(__file__, 541, 12), init___35826, *[self_35827, _wrap_35829, None_35830], **kwargs_35831)
            

            if more_types_in_union_35824:
                # Runtime conditional SSA for else branch (line 540)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_35823) or more_types_in_union_35824):
            
            # Call to __init__(...): (line 543)
            # Processing the call arguments (line 543)
            # Getting the type of 'self' (line 543)
            self_35835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 25), 'self', False)
            # Getting the type of 'self' (line 543)
            self_35836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 31), 'self', False)
            # Obtaining the member '_wrap' of a type (line 543)
            _wrap_35837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 31), self_35836, '_wrap')
            # Getting the type of 'self' (line 543)
            self_35838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 43), 'self', False)
            # Obtaining the member '_wrap_jac' of a type (line 543)
            _wrap_jac_35839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 43), self_35838, '_wrap_jac')
            # Processing the call keyword arguments (line 543)
            kwargs_35840 = {}
            # Getting the type of 'ode' (line 543)
            ode_35833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'ode', False)
            # Obtaining the member '__init__' of a type (line 543)
            init___35834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 12), ode_35833, '__init__')
            # Calling __init__(args, kwargs) (line 543)
            init___call_result_35841 = invoke(stypy.reporting.localization.Localization(__file__, 543, 12), init___35834, *[self_35835, _wrap_35837, _wrap_jac_35839], **kwargs_35840)
            

            if (may_be_35823 and more_types_in_union_35824):
                # SSA join for if statement (line 540)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _wrap(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_wrap'
        module_type_store = module_type_store.open_function_context('_wrap', 545, 4, False)
        # Assigning a type to the variable 'self' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        complex_ode._wrap.__dict__.__setitem__('stypy_localization', localization)
        complex_ode._wrap.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        complex_ode._wrap.__dict__.__setitem__('stypy_type_store', module_type_store)
        complex_ode._wrap.__dict__.__setitem__('stypy_function_name', 'complex_ode._wrap')
        complex_ode._wrap.__dict__.__setitem__('stypy_param_names_list', ['t', 'y'])
        complex_ode._wrap.__dict__.__setitem__('stypy_varargs_param_name', 'f_args')
        complex_ode._wrap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        complex_ode._wrap.__dict__.__setitem__('stypy_call_defaults', defaults)
        complex_ode._wrap.__dict__.__setitem__('stypy_call_varargs', varargs)
        complex_ode._wrap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        complex_ode._wrap.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'complex_ode._wrap', ['t', 'y'], 'f_args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_wrap', localization, ['t', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_wrap(...)' code ##################

        
        # Assigning a Call to a Name (line 546):
        
        # Assigning a Call to a Name (line 546):
        
        # Call to cf(...): (line 546)
        
        # Obtaining an instance of the builtin type 'tuple' (line 546)
        tuple_35844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 546)
        # Adding element type (line 546)
        # Getting the type of 't' (line 546)
        t_35845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 23), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 23), tuple_35844, t_35845)
        # Adding element type (line 546)
        
        # Obtaining the type of the subscript
        int_35846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 30), 'int')
        slice_35847 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 546, 26), None, None, int_35846)
        # Getting the type of 'y' (line 546)
        y_35848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 26), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 546)
        getitem___35849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 26), y_35848, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 546)
        subscript_call_result_35850 = invoke(stypy.reporting.localization.Localization(__file__, 546, 26), getitem___35849, slice_35847)
        
        complex_35851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 35), 'complex')
        
        # Obtaining the type of the subscript
        int_35852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 42), 'int')
        int_35853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 45), 'int')
        slice_35854 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 546, 40), int_35852, None, int_35853)
        # Getting the type of 'y' (line 546)
        y_35855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 40), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 546)
        getitem___35856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 40), y_35855, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 546)
        subscript_call_result_35857 = invoke(stypy.reporting.localization.Localization(__file__, 546, 40), getitem___35856, slice_35854)
        
        # Applying the binary operator '*' (line 546)
        result_mul_35858 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 35), '*', complex_35851, subscript_call_result_35857)
        
        # Applying the binary operator '+' (line 546)
        result_add_35859 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 26), '+', subscript_call_result_35850, result_mul_35858)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 23), tuple_35844, result_add_35859)
        
        # Getting the type of 'f_args' (line 546)
        f_args_35860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 51), 'f_args', False)
        # Applying the binary operator '+' (line 546)
        result_add_35861 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 22), '+', tuple_35844, f_args_35860)
        
        # Processing the call keyword arguments (line 546)
        kwargs_35862 = {}
        # Getting the type of 'self' (line 546)
        self_35842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'self', False)
        # Obtaining the member 'cf' of a type (line 546)
        cf_35843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 12), self_35842, 'cf')
        # Calling cf(args, kwargs) (line 546)
        cf_call_result_35863 = invoke(stypy.reporting.localization.Localization(__file__, 546, 12), cf_35843, *[result_add_35861], **kwargs_35862)
        
        # Assigning a type to the variable 'f' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'f', cf_call_result_35863)
        
        # Assigning a Call to a Subscript (line 549):
        
        # Assigning a Call to a Subscript (line 549):
        
        # Call to real(...): (line 549)
        # Processing the call arguments (line 549)
        # Getting the type of 'f' (line 549)
        f_35865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 29), 'f', False)
        # Processing the call keyword arguments (line 549)
        kwargs_35866 = {}
        # Getting the type of 'real' (line 549)
        real_35864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 24), 'real', False)
        # Calling real(args, kwargs) (line 549)
        real_call_result_35867 = invoke(stypy.reporting.localization.Localization(__file__, 549, 24), real_35864, *[f_35865], **kwargs_35866)
        
        # Getting the type of 'self' (line 549)
        self_35868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'self')
        # Obtaining the member 'tmp' of a type (line 549)
        tmp_35869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 8), self_35868, 'tmp')
        int_35870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 19), 'int')
        slice_35871 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 549, 8), None, None, int_35870)
        # Storing an element on a container (line 549)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 8), tmp_35869, (slice_35871, real_call_result_35867))
        
        # Assigning a Call to a Subscript (line 550):
        
        # Assigning a Call to a Subscript (line 550):
        
        # Call to imag(...): (line 550)
        # Processing the call arguments (line 550)
        # Getting the type of 'f' (line 550)
        f_35873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 30), 'f', False)
        # Processing the call keyword arguments (line 550)
        kwargs_35874 = {}
        # Getting the type of 'imag' (line 550)
        imag_35872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 25), 'imag', False)
        # Calling imag(args, kwargs) (line 550)
        imag_call_result_35875 = invoke(stypy.reporting.localization.Localization(__file__, 550, 25), imag_35872, *[f_35873], **kwargs_35874)
        
        # Getting the type of 'self' (line 550)
        self_35876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'self')
        # Obtaining the member 'tmp' of a type (line 550)
        tmp_35877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 8), self_35876, 'tmp')
        int_35878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 17), 'int')
        int_35879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 20), 'int')
        slice_35880 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 550, 8), int_35878, None, int_35879)
        # Storing an element on a container (line 550)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 8), tmp_35877, (slice_35880, imag_call_result_35875))
        # Getting the type of 'self' (line 551)
        self_35881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 15), 'self')
        # Obtaining the member 'tmp' of a type (line 551)
        tmp_35882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 15), self_35881, 'tmp')
        # Assigning a type to the variable 'stypy_return_type' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'stypy_return_type', tmp_35882)
        
        # ################# End of '_wrap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_wrap' in the type store
        # Getting the type of 'stypy_return_type' (line 545)
        stypy_return_type_35883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35883)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_wrap'
        return stypy_return_type_35883


    @norecursion
    def _wrap_jac(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_wrap_jac'
        module_type_store = module_type_store.open_function_context('_wrap_jac', 553, 4, False)
        # Assigning a type to the variable 'self' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        complex_ode._wrap_jac.__dict__.__setitem__('stypy_localization', localization)
        complex_ode._wrap_jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        complex_ode._wrap_jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        complex_ode._wrap_jac.__dict__.__setitem__('stypy_function_name', 'complex_ode._wrap_jac')
        complex_ode._wrap_jac.__dict__.__setitem__('stypy_param_names_list', ['t', 'y'])
        complex_ode._wrap_jac.__dict__.__setitem__('stypy_varargs_param_name', 'jac_args')
        complex_ode._wrap_jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        complex_ode._wrap_jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        complex_ode._wrap_jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        complex_ode._wrap_jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        complex_ode._wrap_jac.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'complex_ode._wrap_jac', ['t', 'y'], 'jac_args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_wrap_jac', localization, ['t', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_wrap_jac(...)' code ##################

        
        # Assigning a Call to a Name (line 555):
        
        # Assigning a Call to a Name (line 555):
        
        # Call to cjac(...): (line 555)
        
        # Obtaining an instance of the builtin type 'tuple' (line 555)
        tuple_35886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 555)
        # Adding element type (line 555)
        # Getting the type of 't' (line 555)
        t_35887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 27), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 555, 27), tuple_35886, t_35887)
        # Adding element type (line 555)
        
        # Obtaining the type of the subscript
        int_35888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 34), 'int')
        slice_35889 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 555, 30), None, None, int_35888)
        # Getting the type of 'y' (line 555)
        y_35890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 30), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 555)
        getitem___35891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 30), y_35890, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 555)
        subscript_call_result_35892 = invoke(stypy.reporting.localization.Localization(__file__, 555, 30), getitem___35891, slice_35889)
        
        complex_35893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 39), 'complex')
        
        # Obtaining the type of the subscript
        int_35894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 46), 'int')
        int_35895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 49), 'int')
        slice_35896 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 555, 44), int_35894, None, int_35895)
        # Getting the type of 'y' (line 555)
        y_35897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 44), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 555)
        getitem___35898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 44), y_35897, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 555)
        subscript_call_result_35899 = invoke(stypy.reporting.localization.Localization(__file__, 555, 44), getitem___35898, slice_35896)
        
        # Applying the binary operator '*' (line 555)
        result_mul_35900 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 39), '*', complex_35893, subscript_call_result_35899)
        
        # Applying the binary operator '+' (line 555)
        result_add_35901 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 30), '+', subscript_call_result_35892, result_mul_35900)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 555, 27), tuple_35886, result_add_35901)
        
        # Getting the type of 'jac_args' (line 555)
        jac_args_35902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 55), 'jac_args', False)
        # Applying the binary operator '+' (line 555)
        result_add_35903 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 26), '+', tuple_35886, jac_args_35902)
        
        # Processing the call keyword arguments (line 555)
        kwargs_35904 = {}
        # Getting the type of 'self' (line 555)
        self_35884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 14), 'self', False)
        # Obtaining the member 'cjac' of a type (line 555)
        cjac_35885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 14), self_35884, 'cjac')
        # Calling cjac(args, kwargs) (line 555)
        cjac_call_result_35905 = invoke(stypy.reporting.localization.Localization(__file__, 555, 14), cjac_35885, *[result_add_35903], **kwargs_35904)
        
        # Assigning a type to the variable 'jac' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'jac', cjac_call_result_35905)
        
        # Assigning a Call to a Name (line 561):
        
        # Assigning a Call to a Name (line 561):
        
        # Call to zeros(...): (line 561)
        # Processing the call arguments (line 561)
        
        # Obtaining an instance of the builtin type 'tuple' (line 561)
        tuple_35907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 561)
        # Adding element type (line 561)
        int_35908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 25), 'int')
        
        # Obtaining the type of the subscript
        int_35909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 39), 'int')
        # Getting the type of 'jac' (line 561)
        jac_35910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 29), 'jac', False)
        # Obtaining the member 'shape' of a type (line 561)
        shape_35911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 29), jac_35910, 'shape')
        # Obtaining the member '__getitem__' of a type (line 561)
        getitem___35912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 29), shape_35911, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 561)
        subscript_call_result_35913 = invoke(stypy.reporting.localization.Localization(__file__, 561, 29), getitem___35912, int_35909)
        
        # Applying the binary operator '*' (line 561)
        result_mul_35914 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 25), '*', int_35908, subscript_call_result_35913)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 25), tuple_35907, result_mul_35914)
        # Adding element type (line 561)
        int_35915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 43), 'int')
        
        # Obtaining the type of the subscript
        int_35916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 57), 'int')
        # Getting the type of 'jac' (line 561)
        jac_35917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 47), 'jac', False)
        # Obtaining the member 'shape' of a type (line 561)
        shape_35918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 47), jac_35917, 'shape')
        # Obtaining the member '__getitem__' of a type (line 561)
        getitem___35919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 47), shape_35918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 561)
        subscript_call_result_35920 = invoke(stypy.reporting.localization.Localization(__file__, 561, 47), getitem___35919, int_35916)
        
        # Applying the binary operator '*' (line 561)
        result_mul_35921 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 43), '*', int_35915, subscript_call_result_35920)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 25), tuple_35907, result_mul_35921)
        
        # Processing the call keyword arguments (line 561)
        kwargs_35922 = {}
        # Getting the type of 'zeros' (line 561)
        zeros_35906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 18), 'zeros', False)
        # Calling zeros(args, kwargs) (line 561)
        zeros_call_result_35923 = invoke(stypy.reporting.localization.Localization(__file__, 561, 18), zeros_35906, *[tuple_35907], **kwargs_35922)
        
        # Assigning a type to the variable 'jac_tmp' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'jac_tmp', zeros_call_result_35923)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Subscript (line 562):
        
        # Call to real(...): (line 562)
        # Processing the call arguments (line 562)
        # Getting the type of 'jac' (line 562)
        jac_35925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 55), 'jac', False)
        # Processing the call keyword arguments (line 562)
        kwargs_35926 = {}
        # Getting the type of 'real' (line 562)
        real_35924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 50), 'real', False)
        # Calling real(args, kwargs) (line 562)
        real_call_result_35927 = invoke(stypy.reporting.localization.Localization(__file__, 562, 50), real_35924, *[jac_35925], **kwargs_35926)
        
        # Getting the type of 'jac_tmp' (line 562)
        jac_tmp_35928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 30), 'jac_tmp')
        int_35929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 40), 'int')
        slice_35930 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 562, 30), None, None, int_35929)
        int_35931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 45), 'int')
        slice_35932 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 562, 30), None, None, int_35931)
        # Storing an element on a container (line 562)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 30), jac_tmp_35928, ((slice_35930, slice_35932), real_call_result_35927))
        
        # Assigning a Subscript to a Subscript (line 562):
        
        # Obtaining the type of the subscript
        int_35933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 40), 'int')
        slice_35934 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 562, 30), None, None, int_35933)
        int_35935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 45), 'int')
        slice_35936 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 562, 30), None, None, int_35935)
        # Getting the type of 'jac_tmp' (line 562)
        jac_tmp_35937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 30), 'jac_tmp')
        # Obtaining the member '__getitem__' of a type (line 562)
        getitem___35938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 30), jac_tmp_35937, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 562)
        subscript_call_result_35939 = invoke(stypy.reporting.localization.Localization(__file__, 562, 30), getitem___35938, (slice_35934, slice_35936))
        
        # Getting the type of 'jac_tmp' (line 562)
        jac_tmp_35940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'jac_tmp')
        int_35941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 16), 'int')
        int_35942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 19), 'int')
        slice_35943 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 562, 8), int_35941, None, int_35942)
        int_35944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 22), 'int')
        int_35945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 25), 'int')
        slice_35946 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 562, 8), int_35944, None, int_35945)
        # Storing an element on a container (line 562)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 8), jac_tmp_35940, ((slice_35943, slice_35946), subscript_call_result_35939))
        
        # Assigning a Call to a Subscript (line 563):
        
        # Assigning a Call to a Subscript (line 563):
        
        # Call to imag(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'jac' (line 563)
        jac_35948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 34), 'jac', False)
        # Processing the call keyword arguments (line 563)
        kwargs_35949 = {}
        # Getting the type of 'imag' (line 563)
        imag_35947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 29), 'imag', False)
        # Calling imag(args, kwargs) (line 563)
        imag_call_result_35950 = invoke(stypy.reporting.localization.Localization(__file__, 563, 29), imag_35947, *[jac_35948], **kwargs_35949)
        
        # Getting the type of 'jac_tmp' (line 563)
        jac_tmp_35951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'jac_tmp')
        int_35952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 16), 'int')
        int_35953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 19), 'int')
        slice_35954 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 563, 8), int_35952, None, int_35953)
        int_35955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 24), 'int')
        slice_35956 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 563, 8), None, None, int_35955)
        # Storing an element on a container (line 563)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 8), jac_tmp_35951, ((slice_35954, slice_35956), imag_call_result_35950))
        
        # Assigning a UnaryOp to a Subscript (line 564):
        
        # Assigning a UnaryOp to a Subscript (line 564):
        
        
        # Obtaining the type of the subscript
        int_35957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 38), 'int')
        int_35958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 41), 'int')
        slice_35959 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 564, 30), int_35957, None, int_35958)
        int_35960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 46), 'int')
        slice_35961 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 564, 30), None, None, int_35960)
        # Getting the type of 'jac_tmp' (line 564)
        jac_tmp_35962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 30), 'jac_tmp')
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___35963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 30), jac_tmp_35962, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 564)
        subscript_call_result_35964 = invoke(stypy.reporting.localization.Localization(__file__, 564, 30), getitem___35963, (slice_35959, slice_35961))
        
        # Applying the 'usub' unary operator (line 564)
        result___neg___35965 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 29), 'usub', subscript_call_result_35964)
        
        # Getting the type of 'jac_tmp' (line 564)
        jac_tmp_35966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'jac_tmp')
        int_35967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 18), 'int')
        slice_35968 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 564, 8), None, None, int_35967)
        int_35969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 21), 'int')
        int_35970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 24), 'int')
        slice_35971 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 564, 8), int_35969, None, int_35970)
        # Storing an element on a container (line 564)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 564, 8), jac_tmp_35966, ((slice_35968, slice_35971), result___neg___35965))
        
        # Assigning a Call to a Name (line 566):
        
        # Assigning a Call to a Name (line 566):
        
        # Call to getattr(...): (line 566)
        # Processing the call arguments (line 566)
        # Getting the type of 'self' (line 566)
        self_35973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 21), 'self', False)
        # Obtaining the member '_integrator' of a type (line 566)
        _integrator_35974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 21), self_35973, '_integrator')
        str_35975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 39), 'str', 'ml')
        # Getting the type of 'None' (line 566)
        None_35976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 45), 'None', False)
        # Processing the call keyword arguments (line 566)
        kwargs_35977 = {}
        # Getting the type of 'getattr' (line 566)
        getattr_35972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 13), 'getattr', False)
        # Calling getattr(args, kwargs) (line 566)
        getattr_call_result_35978 = invoke(stypy.reporting.localization.Localization(__file__, 566, 13), getattr_35972, *[_integrator_35974, str_35975, None_35976], **kwargs_35977)
        
        # Assigning a type to the variable 'ml' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'ml', getattr_call_result_35978)
        
        # Assigning a Call to a Name (line 567):
        
        # Assigning a Call to a Name (line 567):
        
        # Call to getattr(...): (line 567)
        # Processing the call arguments (line 567)
        # Getting the type of 'self' (line 567)
        self_35980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 21), 'self', False)
        # Obtaining the member '_integrator' of a type (line 567)
        _integrator_35981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 21), self_35980, '_integrator')
        str_35982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 39), 'str', 'mu')
        # Getting the type of 'None' (line 567)
        None_35983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 45), 'None', False)
        # Processing the call keyword arguments (line 567)
        kwargs_35984 = {}
        # Getting the type of 'getattr' (line 567)
        getattr_35979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 13), 'getattr', False)
        # Calling getattr(args, kwargs) (line 567)
        getattr_call_result_35985 = invoke(stypy.reporting.localization.Localization(__file__, 567, 13), getattr_35979, *[_integrator_35981, str_35982, None_35983], **kwargs_35984)
        
        # Assigning a type to the variable 'mu' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'mu', getattr_call_result_35985)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ml' (line 568)
        ml_35986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 11), 'ml')
        # Getting the type of 'None' (line 568)
        None_35987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 21), 'None')
        # Applying the binary operator 'isnot' (line 568)
        result_is_not_35988 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 11), 'isnot', ml_35986, None_35987)
        
        
        # Getting the type of 'mu' (line 568)
        mu_35989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 29), 'mu')
        # Getting the type of 'None' (line 568)
        None_35990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 39), 'None')
        # Applying the binary operator 'isnot' (line 568)
        result_is_not_35991 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 29), 'isnot', mu_35989, None_35990)
        
        # Applying the binary operator 'or' (line 568)
        result_or_keyword_35992 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 11), 'or', result_is_not_35988, result_is_not_35991)
        
        # Testing the type of an if condition (line 568)
        if_condition_35993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 8), result_or_keyword_35992)
        # Assigning a type to the variable 'if_condition_35993' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'if_condition_35993', if_condition_35993)
        # SSA begins for if statement (line 568)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 572):
        
        # Assigning a Call to a Name (line 572):
        
        # Call to _transform_banded_jac(...): (line 572)
        # Processing the call arguments (line 572)
        # Getting the type of 'jac_tmp' (line 572)
        jac_tmp_35995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 44), 'jac_tmp', False)
        # Processing the call keyword arguments (line 572)
        kwargs_35996 = {}
        # Getting the type of '_transform_banded_jac' (line 572)
        _transform_banded_jac_35994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 22), '_transform_banded_jac', False)
        # Calling _transform_banded_jac(args, kwargs) (line 572)
        _transform_banded_jac_call_result_35997 = invoke(stypy.reporting.localization.Localization(__file__, 572, 22), _transform_banded_jac_35994, *[jac_tmp_35995], **kwargs_35996)
        
        # Assigning a type to the variable 'jac_tmp' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'jac_tmp', _transform_banded_jac_call_result_35997)
        # SSA join for if statement (line 568)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'jac_tmp' (line 574)
        jac_tmp_35998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 15), 'jac_tmp')
        # Assigning a type to the variable 'stypy_return_type' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'stypy_return_type', jac_tmp_35998)
        
        # ################# End of '_wrap_jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_wrap_jac' in the type store
        # Getting the type of 'stypy_return_type' (line 553)
        stypy_return_type_35999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35999)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_wrap_jac'
        return stypy_return_type_35999


    @norecursion
    def y(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'y'
        module_type_store = module_type_store.open_function_context('y', 576, 4, False)
        # Assigning a type to the variable 'self' (line 577)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        complex_ode.y.__dict__.__setitem__('stypy_localization', localization)
        complex_ode.y.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        complex_ode.y.__dict__.__setitem__('stypy_type_store', module_type_store)
        complex_ode.y.__dict__.__setitem__('stypy_function_name', 'complex_ode.y')
        complex_ode.y.__dict__.__setitem__('stypy_param_names_list', [])
        complex_ode.y.__dict__.__setitem__('stypy_varargs_param_name', None)
        complex_ode.y.__dict__.__setitem__('stypy_kwargs_param_name', None)
        complex_ode.y.__dict__.__setitem__('stypy_call_defaults', defaults)
        complex_ode.y.__dict__.__setitem__('stypy_call_varargs', varargs)
        complex_ode.y.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        complex_ode.y.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'complex_ode.y', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'y', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'y(...)' code ##################

        
        # Obtaining the type of the subscript
        int_36000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 25), 'int')
        slice_36001 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 578, 15), None, None, int_36000)
        # Getting the type of 'self' (line 578)
        self_36002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 15), 'self')
        # Obtaining the member '_y' of a type (line 578)
        _y_36003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 15), self_36002, '_y')
        # Obtaining the member '__getitem__' of a type (line 578)
        getitem___36004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 15), _y_36003, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 578)
        subscript_call_result_36005 = invoke(stypy.reporting.localization.Localization(__file__, 578, 15), getitem___36004, slice_36001)
        
        complex_36006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 30), 'complex')
        
        # Obtaining the type of the subscript
        int_36007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 43), 'int')
        int_36008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 46), 'int')
        slice_36009 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 578, 35), int_36007, None, int_36008)
        # Getting the type of 'self' (line 578)
        self_36010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 35), 'self')
        # Obtaining the member '_y' of a type (line 578)
        _y_36011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 35), self_36010, '_y')
        # Obtaining the member '__getitem__' of a type (line 578)
        getitem___36012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 35), _y_36011, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 578)
        subscript_call_result_36013 = invoke(stypy.reporting.localization.Localization(__file__, 578, 35), getitem___36012, slice_36009)
        
        # Applying the binary operator '*' (line 578)
        result_mul_36014 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 30), '*', complex_36006, subscript_call_result_36013)
        
        # Applying the binary operator '+' (line 578)
        result_add_36015 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 15), '+', subscript_call_result_36005, result_mul_36014)
        
        # Assigning a type to the variable 'stypy_return_type' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'stypy_return_type', result_add_36015)
        
        # ################# End of 'y(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'y' in the type store
        # Getting the type of 'stypy_return_type' (line 576)
        stypy_return_type_36016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36016)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'y'
        return stypy_return_type_36016


    @norecursion
    def set_integrator(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_integrator'
        module_type_store = module_type_store.open_function_context('set_integrator', 580, 4, False)
        # Assigning a type to the variable 'self' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        complex_ode.set_integrator.__dict__.__setitem__('stypy_localization', localization)
        complex_ode.set_integrator.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        complex_ode.set_integrator.__dict__.__setitem__('stypy_type_store', module_type_store)
        complex_ode.set_integrator.__dict__.__setitem__('stypy_function_name', 'complex_ode.set_integrator')
        complex_ode.set_integrator.__dict__.__setitem__('stypy_param_names_list', ['name'])
        complex_ode.set_integrator.__dict__.__setitem__('stypy_varargs_param_name', None)
        complex_ode.set_integrator.__dict__.__setitem__('stypy_kwargs_param_name', 'integrator_params')
        complex_ode.set_integrator.__dict__.__setitem__('stypy_call_defaults', defaults)
        complex_ode.set_integrator.__dict__.__setitem__('stypy_call_varargs', varargs)
        complex_ode.set_integrator.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        complex_ode.set_integrator.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'complex_ode.set_integrator', ['name'], None, 'integrator_params', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_integrator', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_integrator(...)' code ##################

        str_36017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, (-1)), 'str', '\n        Set integrator by name.\n\n        Parameters\n        ----------\n        name : str\n            Name of the integrator\n        integrator_params\n            Additional parameters for the integrator.\n        ')
        
        
        # Getting the type of 'name' (line 591)
        name_36018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 11), 'name')
        str_36019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 19), 'str', 'zvode')
        # Applying the binary operator '==' (line 591)
        result_eq_36020 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 11), '==', name_36018, str_36019)
        
        # Testing the type of an if condition (line 591)
        if_condition_36021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 591, 8), result_eq_36020)
        # Assigning a type to the variable 'if_condition_36021' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'if_condition_36021', if_condition_36021)
        # SSA begins for if statement (line 591)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 592)
        # Processing the call arguments (line 592)
        str_36023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 29), 'str', 'zvode must be used with ode, not complex_ode')
        # Processing the call keyword arguments (line 592)
        kwargs_36024 = {}
        # Getting the type of 'ValueError' (line 592)
        ValueError_36022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 592)
        ValueError_call_result_36025 = invoke(stypy.reporting.localization.Localization(__file__, 592, 18), ValueError_36022, *[str_36023], **kwargs_36024)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 592, 12), ValueError_call_result_36025, 'raise parameter', BaseException)
        # SSA join for if statement (line 591)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 594):
        
        # Assigning a Call to a Name (line 594):
        
        # Call to get(...): (line 594)
        # Processing the call arguments (line 594)
        str_36028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 38), 'str', 'lband')
        # Processing the call keyword arguments (line 594)
        kwargs_36029 = {}
        # Getting the type of 'integrator_params' (line 594)
        integrator_params_36026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 16), 'integrator_params', False)
        # Obtaining the member 'get' of a type (line 594)
        get_36027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 16), integrator_params_36026, 'get')
        # Calling get(args, kwargs) (line 594)
        get_call_result_36030 = invoke(stypy.reporting.localization.Localization(__file__, 594, 16), get_36027, *[str_36028], **kwargs_36029)
        
        # Assigning a type to the variable 'lband' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'lband', get_call_result_36030)
        
        # Assigning a Call to a Name (line 595):
        
        # Assigning a Call to a Name (line 595):
        
        # Call to get(...): (line 595)
        # Processing the call arguments (line 595)
        str_36033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 38), 'str', 'uband')
        # Processing the call keyword arguments (line 595)
        kwargs_36034 = {}
        # Getting the type of 'integrator_params' (line 595)
        integrator_params_36031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 16), 'integrator_params', False)
        # Obtaining the member 'get' of a type (line 595)
        get_36032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 16), integrator_params_36031, 'get')
        # Calling get(args, kwargs) (line 595)
        get_call_result_36035 = invoke(stypy.reporting.localization.Localization(__file__, 595, 16), get_36032, *[str_36033], **kwargs_36034)
        
        # Assigning a type to the variable 'uband' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'uband', get_call_result_36035)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'lband' (line 596)
        lband_36036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 11), 'lband')
        # Getting the type of 'None' (line 596)
        None_36037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 24), 'None')
        # Applying the binary operator 'isnot' (line 596)
        result_is_not_36038 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 11), 'isnot', lband_36036, None_36037)
        
        
        # Getting the type of 'uband' (line 596)
        uband_36039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 32), 'uband')
        # Getting the type of 'None' (line 596)
        None_36040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 45), 'None')
        # Applying the binary operator 'isnot' (line 596)
        result_is_not_36041 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 32), 'isnot', uband_36039, None_36040)
        
        # Applying the binary operator 'or' (line 596)
        result_or_keyword_36042 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 11), 'or', result_is_not_36038, result_is_not_36041)
        
        # Testing the type of an if condition (line 596)
        if_condition_36043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 596, 8), result_or_keyword_36042)
        # Assigning a type to the variable 'if_condition_36043' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'if_condition_36043', if_condition_36043)
        # SSA begins for if statement (line 596)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Subscript (line 601):
        
        # Assigning a BinOp to a Subscript (line 601):
        int_36044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 41), 'int')
        
        # Evaluating a boolean operation
        # Getting the type of 'lband' (line 601)
        lband_36045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 46), 'lband')
        int_36046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 55), 'int')
        # Applying the binary operator 'or' (line 601)
        result_or_keyword_36047 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 46), 'or', lband_36045, int_36046)
        
        # Applying the binary operator '*' (line 601)
        result_mul_36048 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 41), '*', int_36044, result_or_keyword_36047)
        
        int_36049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 60), 'int')
        # Applying the binary operator '+' (line 601)
        result_add_36050 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 41), '+', result_mul_36048, int_36049)
        
        # Getting the type of 'integrator_params' (line 601)
        integrator_params_36051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'integrator_params')
        str_36052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 30), 'str', 'lband')
        # Storing an element on a container (line 601)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 12), integrator_params_36051, (str_36052, result_add_36050))
        
        # Assigning a BinOp to a Subscript (line 602):
        
        # Assigning a BinOp to a Subscript (line 602):
        int_36053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 41), 'int')
        
        # Evaluating a boolean operation
        # Getting the type of 'uband' (line 602)
        uband_36054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 46), 'uband')
        int_36055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 55), 'int')
        # Applying the binary operator 'or' (line 602)
        result_or_keyword_36056 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 46), 'or', uband_36054, int_36055)
        
        # Applying the binary operator '*' (line 602)
        result_mul_36057 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 41), '*', int_36053, result_or_keyword_36056)
        
        int_36058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 60), 'int')
        # Applying the binary operator '+' (line 602)
        result_add_36059 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 41), '+', result_mul_36057, int_36058)
        
        # Getting the type of 'integrator_params' (line 602)
        integrator_params_36060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 12), 'integrator_params')
        str_36061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 30), 'str', 'uband')
        # Storing an element on a container (line 602)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 602, 12), integrator_params_36060, (str_36061, result_add_36059))
        # SSA join for if statement (line 596)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_integrator(...): (line 604)
        # Processing the call arguments (line 604)
        # Getting the type of 'self' (line 604)
        self_36064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 34), 'self', False)
        # Getting the type of 'name' (line 604)
        name_36065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 40), 'name', False)
        # Processing the call keyword arguments (line 604)
        # Getting the type of 'integrator_params' (line 604)
        integrator_params_36066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 48), 'integrator_params', False)
        kwargs_36067 = {'integrator_params_36066': integrator_params_36066}
        # Getting the type of 'ode' (line 604)
        ode_36062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 15), 'ode', False)
        # Obtaining the member 'set_integrator' of a type (line 604)
        set_integrator_36063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 15), ode_36062, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 604)
        set_integrator_call_result_36068 = invoke(stypy.reporting.localization.Localization(__file__, 604, 15), set_integrator_36063, *[self_36064, name_36065], **kwargs_36067)
        
        # Assigning a type to the variable 'stypy_return_type' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'stypy_return_type', set_integrator_call_result_36068)
        
        # ################# End of 'set_integrator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_integrator' in the type store
        # Getting the type of 'stypy_return_type' (line 580)
        stypy_return_type_36069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_integrator'
        return stypy_return_type_36069


    @norecursion
    def set_initial_value(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_36070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 37), 'float')
        defaults = [float_36070]
        # Create a new context for function 'set_initial_value'
        module_type_store = module_type_store.open_function_context('set_initial_value', 606, 4, False)
        # Assigning a type to the variable 'self' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        complex_ode.set_initial_value.__dict__.__setitem__('stypy_localization', localization)
        complex_ode.set_initial_value.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        complex_ode.set_initial_value.__dict__.__setitem__('stypy_type_store', module_type_store)
        complex_ode.set_initial_value.__dict__.__setitem__('stypy_function_name', 'complex_ode.set_initial_value')
        complex_ode.set_initial_value.__dict__.__setitem__('stypy_param_names_list', ['y', 't'])
        complex_ode.set_initial_value.__dict__.__setitem__('stypy_varargs_param_name', None)
        complex_ode.set_initial_value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        complex_ode.set_initial_value.__dict__.__setitem__('stypy_call_defaults', defaults)
        complex_ode.set_initial_value.__dict__.__setitem__('stypy_call_varargs', varargs)
        complex_ode.set_initial_value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        complex_ode.set_initial_value.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'complex_ode.set_initial_value', ['y', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_initial_value', localization, ['y', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_initial_value(...)' code ##################

        str_36071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 8), 'str', 'Set initial conditions y(t) = y.')
        
        # Assigning a Call to a Name (line 608):
        
        # Assigning a Call to a Name (line 608):
        
        # Call to asarray(...): (line 608)
        # Processing the call arguments (line 608)
        # Getting the type of 'y' (line 608)
        y_36073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 20), 'y', False)
        # Processing the call keyword arguments (line 608)
        kwargs_36074 = {}
        # Getting the type of 'asarray' (line 608)
        asarray_36072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'asarray', False)
        # Calling asarray(args, kwargs) (line 608)
        asarray_call_result_36075 = invoke(stypy.reporting.localization.Localization(__file__, 608, 12), asarray_36072, *[y_36073], **kwargs_36074)
        
        # Assigning a type to the variable 'y' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'y', asarray_call_result_36075)
        
        # Assigning a Call to a Attribute (line 609):
        
        # Assigning a Call to a Attribute (line 609):
        
        # Call to zeros(...): (line 609)
        # Processing the call arguments (line 609)
        # Getting the type of 'y' (line 609)
        y_36077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 25), 'y', False)
        # Obtaining the member 'size' of a type (line 609)
        size_36078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 25), y_36077, 'size')
        int_36079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 34), 'int')
        # Applying the binary operator '*' (line 609)
        result_mul_36080 = python_operator(stypy.reporting.localization.Localization(__file__, 609, 25), '*', size_36078, int_36079)
        
        str_36081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 37), 'str', 'float')
        # Processing the call keyword arguments (line 609)
        kwargs_36082 = {}
        # Getting the type of 'zeros' (line 609)
        zeros_36076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 19), 'zeros', False)
        # Calling zeros(args, kwargs) (line 609)
        zeros_call_result_36083 = invoke(stypy.reporting.localization.Localization(__file__, 609, 19), zeros_36076, *[result_mul_36080, str_36081], **kwargs_36082)
        
        # Getting the type of 'self' (line 609)
        self_36084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'self')
        # Setting the type of the member 'tmp' of a type (line 609)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 8), self_36084, 'tmp', zeros_call_result_36083)
        
        # Assigning a Call to a Subscript (line 610):
        
        # Assigning a Call to a Subscript (line 610):
        
        # Call to real(...): (line 610)
        # Processing the call arguments (line 610)
        # Getting the type of 'y' (line 610)
        y_36086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 29), 'y', False)
        # Processing the call keyword arguments (line 610)
        kwargs_36087 = {}
        # Getting the type of 'real' (line 610)
        real_36085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 24), 'real', False)
        # Calling real(args, kwargs) (line 610)
        real_call_result_36088 = invoke(stypy.reporting.localization.Localization(__file__, 610, 24), real_36085, *[y_36086], **kwargs_36087)
        
        # Getting the type of 'self' (line 610)
        self_36089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'self')
        # Obtaining the member 'tmp' of a type (line 610)
        tmp_36090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 8), self_36089, 'tmp')
        int_36091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 19), 'int')
        slice_36092 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 610, 8), None, None, int_36091)
        # Storing an element on a container (line 610)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 8), tmp_36090, (slice_36092, real_call_result_36088))
        
        # Assigning a Call to a Subscript (line 611):
        
        # Assigning a Call to a Subscript (line 611):
        
        # Call to imag(...): (line 611)
        # Processing the call arguments (line 611)
        # Getting the type of 'y' (line 611)
        y_36094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 30), 'y', False)
        # Processing the call keyword arguments (line 611)
        kwargs_36095 = {}
        # Getting the type of 'imag' (line 611)
        imag_36093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 25), 'imag', False)
        # Calling imag(args, kwargs) (line 611)
        imag_call_result_36096 = invoke(stypy.reporting.localization.Localization(__file__, 611, 25), imag_36093, *[y_36094], **kwargs_36095)
        
        # Getting the type of 'self' (line 611)
        self_36097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'self')
        # Obtaining the member 'tmp' of a type (line 611)
        tmp_36098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 8), self_36097, 'tmp')
        int_36099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 17), 'int')
        int_36100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 20), 'int')
        slice_36101 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 611, 8), int_36099, None, int_36100)
        # Storing an element on a container (line 611)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 8), tmp_36098, (slice_36101, imag_call_result_36096))
        
        # Call to set_initial_value(...): (line 612)
        # Processing the call arguments (line 612)
        # Getting the type of 'self' (line 612)
        self_36104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 37), 'self', False)
        # Getting the type of 'self' (line 612)
        self_36105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 43), 'self', False)
        # Obtaining the member 'tmp' of a type (line 612)
        tmp_36106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 43), self_36105, 'tmp')
        # Getting the type of 't' (line 612)
        t_36107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 53), 't', False)
        # Processing the call keyword arguments (line 612)
        kwargs_36108 = {}
        # Getting the type of 'ode' (line 612)
        ode_36102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 15), 'ode', False)
        # Obtaining the member 'set_initial_value' of a type (line 612)
        set_initial_value_36103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 15), ode_36102, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 612)
        set_initial_value_call_result_36109 = invoke(stypy.reporting.localization.Localization(__file__, 612, 15), set_initial_value_36103, *[self_36104, tmp_36106, t_36107], **kwargs_36108)
        
        # Assigning a type to the variable 'stypy_return_type' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'stypy_return_type', set_initial_value_call_result_36109)
        
        # ################# End of 'set_initial_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_initial_value' in the type store
        # Getting the type of 'stypy_return_type' (line 606)
        stypy_return_type_36110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36110)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_initial_value'
        return stypy_return_type_36110


    @norecursion
    def integrate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 614)
        False_36111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 32), 'False')
        # Getting the type of 'False' (line 614)
        False_36112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 45), 'False')
        defaults = [False_36111, False_36112]
        # Create a new context for function 'integrate'
        module_type_store = module_type_store.open_function_context('integrate', 614, 4, False)
        # Assigning a type to the variable 'self' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        complex_ode.integrate.__dict__.__setitem__('stypy_localization', localization)
        complex_ode.integrate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        complex_ode.integrate.__dict__.__setitem__('stypy_type_store', module_type_store)
        complex_ode.integrate.__dict__.__setitem__('stypy_function_name', 'complex_ode.integrate')
        complex_ode.integrate.__dict__.__setitem__('stypy_param_names_list', ['t', 'step', 'relax'])
        complex_ode.integrate.__dict__.__setitem__('stypy_varargs_param_name', None)
        complex_ode.integrate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        complex_ode.integrate.__dict__.__setitem__('stypy_call_defaults', defaults)
        complex_ode.integrate.__dict__.__setitem__('stypy_call_varargs', varargs)
        complex_ode.integrate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        complex_ode.integrate.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'complex_ode.integrate', ['t', 'step', 'relax'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'integrate', localization, ['t', 'step', 'relax'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'integrate(...)' code ##################

        str_36113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, (-1)), 'str', 'Find y=y(t), set y as an initial condition, and return y.\n\n        Parameters\n        ----------\n        t : float\n            The endpoint of the integration step.\n        step : bool\n            If True, and if the integrator supports the step method,\n            then perform a single integration step and return.\n            This parameter is provided in order to expose internals of\n            the implementation, and should not be changed from its default\n            value in most cases.\n        relax : bool\n            If True and if the integrator supports the run_relax method,\n            then integrate until t_1 >= t and return. ``relax`` is not\n            referenced if ``step=True``.\n            This parameter is provided in order to expose internals of\n            the implementation, and should not be changed from its default\n            value in most cases.\n\n        Returns\n        -------\n        y : float\n            The integrated value at t\n        ')
        
        # Assigning a Call to a Name (line 640):
        
        # Assigning a Call to a Name (line 640):
        
        # Call to integrate(...): (line 640)
        # Processing the call arguments (line 640)
        # Getting the type of 'self' (line 640)
        self_36116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 26), 'self', False)
        # Getting the type of 't' (line 640)
        t_36117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 32), 't', False)
        # Getting the type of 'step' (line 640)
        step_36118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 35), 'step', False)
        # Getting the type of 'relax' (line 640)
        relax_36119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 41), 'relax', False)
        # Processing the call keyword arguments (line 640)
        kwargs_36120 = {}
        # Getting the type of 'ode' (line 640)
        ode_36114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'ode', False)
        # Obtaining the member 'integrate' of a type (line 640)
        integrate_36115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 12), ode_36114, 'integrate')
        # Calling integrate(args, kwargs) (line 640)
        integrate_call_result_36121 = invoke(stypy.reporting.localization.Localization(__file__, 640, 12), integrate_36115, *[self_36116, t_36117, step_36118, relax_36119], **kwargs_36120)
        
        # Assigning a type to the variable 'y' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'y', integrate_call_result_36121)
        
        # Obtaining the type of the subscript
        int_36122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 19), 'int')
        slice_36123 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 641, 15), None, None, int_36122)
        # Getting the type of 'y' (line 641)
        y_36124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 15), 'y')
        # Obtaining the member '__getitem__' of a type (line 641)
        getitem___36125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 15), y_36124, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 641)
        subscript_call_result_36126 = invoke(stypy.reporting.localization.Localization(__file__, 641, 15), getitem___36125, slice_36123)
        
        complex_36127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 24), 'complex')
        
        # Obtaining the type of the subscript
        int_36128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 31), 'int')
        int_36129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 34), 'int')
        slice_36130 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 641, 29), int_36128, None, int_36129)
        # Getting the type of 'y' (line 641)
        y_36131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 29), 'y')
        # Obtaining the member '__getitem__' of a type (line 641)
        getitem___36132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 29), y_36131, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 641)
        subscript_call_result_36133 = invoke(stypy.reporting.localization.Localization(__file__, 641, 29), getitem___36132, slice_36130)
        
        # Applying the binary operator '*' (line 641)
        result_mul_36134 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 24), '*', complex_36127, subscript_call_result_36133)
        
        # Applying the binary operator '+' (line 641)
        result_add_36135 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 15), '+', subscript_call_result_36126, result_mul_36134)
        
        # Assigning a type to the variable 'stypy_return_type' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'stypy_return_type', result_add_36135)
        
        # ################# End of 'integrate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'integrate' in the type store
        # Getting the type of 'stypy_return_type' (line 614)
        stypy_return_type_36136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36136)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'integrate'
        return stypy_return_type_36136


    @norecursion
    def set_solout(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_solout'
        module_type_store = module_type_store.open_function_context('set_solout', 643, 4, False)
        # Assigning a type to the variable 'self' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        complex_ode.set_solout.__dict__.__setitem__('stypy_localization', localization)
        complex_ode.set_solout.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        complex_ode.set_solout.__dict__.__setitem__('stypy_type_store', module_type_store)
        complex_ode.set_solout.__dict__.__setitem__('stypy_function_name', 'complex_ode.set_solout')
        complex_ode.set_solout.__dict__.__setitem__('stypy_param_names_list', ['solout'])
        complex_ode.set_solout.__dict__.__setitem__('stypy_varargs_param_name', None)
        complex_ode.set_solout.__dict__.__setitem__('stypy_kwargs_param_name', None)
        complex_ode.set_solout.__dict__.__setitem__('stypy_call_defaults', defaults)
        complex_ode.set_solout.__dict__.__setitem__('stypy_call_varargs', varargs)
        complex_ode.set_solout.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        complex_ode.set_solout.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'complex_ode.set_solout', ['solout'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_solout', localization, ['solout'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_solout(...)' code ##################

        str_36137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, (-1)), 'str', '\n        Set callable to be called at every successful integration step.\n\n        Parameters\n        ----------\n        solout : callable\n            ``solout(t, y)`` is called at each internal integrator step,\n            t is a scalar providing the current independent position\n            y is the current soloution ``y.shape == (n,)``\n            solout should return -1 to stop integration\n            otherwise it should return None or 0\n\n        ')
        
        # Getting the type of 'self' (line 657)
        self_36138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 11), 'self')
        # Obtaining the member '_integrator' of a type (line 657)
        _integrator_36139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 11), self_36138, '_integrator')
        # Obtaining the member 'supports_solout' of a type (line 657)
        supports_solout_36140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 11), _integrator_36139, 'supports_solout')
        # Testing the type of an if condition (line 657)
        if_condition_36141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 657, 8), supports_solout_36140)
        # Assigning a type to the variable 'if_condition_36141' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'if_condition_36141', if_condition_36141)
        # SSA begins for if statement (line 657)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_solout(...): (line 658)
        # Processing the call arguments (line 658)
        # Getting the type of 'solout' (line 658)
        solout_36145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 40), 'solout', False)
        # Processing the call keyword arguments (line 658)
        # Getting the type of 'True' (line 658)
        True_36146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 56), 'True', False)
        keyword_36147 = True_36146
        kwargs_36148 = {'complex': keyword_36147}
        # Getting the type of 'self' (line 658)
        self_36142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'self', False)
        # Obtaining the member '_integrator' of a type (line 658)
        _integrator_36143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 12), self_36142, '_integrator')
        # Obtaining the member 'set_solout' of a type (line 658)
        set_solout_36144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 12), _integrator_36143, 'set_solout')
        # Calling set_solout(args, kwargs) (line 658)
        set_solout_call_result_36149 = invoke(stypy.reporting.localization.Localization(__file__, 658, 12), set_solout_36144, *[solout_36145], **kwargs_36148)
        
        # SSA branch for the else part of an if statement (line 657)
        module_type_store.open_ssa_branch('else')
        
        # Call to TypeError(...): (line 660)
        # Processing the call arguments (line 660)
        str_36151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 28), 'str', 'selected integrator does not support solouta,')
        str_36152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 30), 'str', 'choose another one')
        # Applying the binary operator '+' (line 660)
        result_add_36153 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 28), '+', str_36151, str_36152)
        
        # Processing the call keyword arguments (line 660)
        kwargs_36154 = {}
        # Getting the type of 'TypeError' (line 660)
        TypeError_36150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 660)
        TypeError_call_result_36155 = invoke(stypy.reporting.localization.Localization(__file__, 660, 18), TypeError_36150, *[result_add_36153], **kwargs_36154)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 660, 12), TypeError_call_result_36155, 'raise parameter', BaseException)
        # SSA join for if statement (line 657)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_solout(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_solout' in the type store
        # Getting the type of 'stypy_return_type' (line 643)
        stypy_return_type_36156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36156)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_solout'
        return stypy_return_type_36156


# Assigning a type to the variable 'complex_ode' (line 508)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 0), 'complex_ode', complex_ode)

@norecursion
def find_integrator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_integrator'
    module_type_store = module_type_store.open_function_context('find_integrator', 668, 0, False)
    
    # Passed parameters checking function
    find_integrator.stypy_localization = localization
    find_integrator.stypy_type_of_self = None
    find_integrator.stypy_type_store = module_type_store
    find_integrator.stypy_function_name = 'find_integrator'
    find_integrator.stypy_param_names_list = ['name']
    find_integrator.stypy_varargs_param_name = None
    find_integrator.stypy_kwargs_param_name = None
    find_integrator.stypy_call_defaults = defaults
    find_integrator.stypy_call_varargs = varargs
    find_integrator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_integrator', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_integrator', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_integrator(...)' code ##################

    
    # Getting the type of 'IntegratorBase' (line 669)
    IntegratorBase_36157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 14), 'IntegratorBase')
    # Obtaining the member 'integrator_classes' of a type (line 669)
    integrator_classes_36158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 14), IntegratorBase_36157, 'integrator_classes')
    # Testing the type of a for loop iterable (line 669)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 669, 4), integrator_classes_36158)
    # Getting the type of the for loop variable (line 669)
    for_loop_var_36159 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 669, 4), integrator_classes_36158)
    # Assigning a type to the variable 'cl' (line 669)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 4), 'cl', for_loop_var_36159)
    # SSA begins for a for statement (line 669)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to match(...): (line 670)
    # Processing the call arguments (line 670)
    # Getting the type of 'name' (line 670)
    name_36162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 20), 'name', False)
    # Getting the type of 'cl' (line 670)
    cl_36163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 26), 'cl', False)
    # Obtaining the member '__name__' of a type (line 670)
    name___36164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 26), cl_36163, '__name__')
    # Getting the type of 're' (line 670)
    re_36165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 39), 're', False)
    # Obtaining the member 'I' of a type (line 670)
    I_36166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 39), re_36165, 'I')
    # Processing the call keyword arguments (line 670)
    kwargs_36167 = {}
    # Getting the type of 're' (line 670)
    re_36160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 11), 're', False)
    # Obtaining the member 'match' of a type (line 670)
    match_36161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 11), re_36160, 'match')
    # Calling match(args, kwargs) (line 670)
    match_call_result_36168 = invoke(stypy.reporting.localization.Localization(__file__, 670, 11), match_36161, *[name_36162, name___36164, I_36166], **kwargs_36167)
    
    # Testing the type of an if condition (line 670)
    if_condition_36169 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 670, 8), match_call_result_36168)
    # Assigning a type to the variable 'if_condition_36169' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 8), 'if_condition_36169', if_condition_36169)
    # SSA begins for if statement (line 670)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'cl' (line 671)
    cl_36170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 19), 'cl')
    # Assigning a type to the variable 'stypy_return_type' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 12), 'stypy_return_type', cl_36170)
    # SSA join for if statement (line 670)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 672)
    None_36171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 4), 'stypy_return_type', None_36171)
    
    # ################# End of 'find_integrator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_integrator' in the type store
    # Getting the type of 'stypy_return_type' (line 668)
    stypy_return_type_36172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36172)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_integrator'
    return stypy_return_type_36172

# Assigning a type to the variable 'find_integrator' (line 668)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 0), 'find_integrator', find_integrator)
# Declaration of the 'IntegratorConcurrencyError' class
# Getting the type of 'RuntimeError' (line 675)
RuntimeError_36173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 33), 'RuntimeError')

class IntegratorConcurrencyError(RuntimeError_36173, ):
    str_36174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, (-1)), 'str', '\n    Failure due to concurrent usage of an integrator that can be used\n    only for a single problem at a time.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 682, 4, False)
        # Assigning a type to the variable 'self' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntegratorConcurrencyError.__init__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a BinOp to a Name (line 683):
        
        # Assigning a BinOp to a Name (line 683):
        str_36175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 15), 'str', 'Integrator `%s` can be used to solve only a single problem at a time. If you want to integrate multiple problems, consider using a different integrator (see `ode.set_integrator`)')
        # Getting the type of 'name' (line 686)
        name_36176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 47), 'name')
        # Applying the binary operator '%' (line 683)
        result_mod_36177 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 14), '%', str_36175, name_36176)
        
        # Assigning a type to the variable 'msg' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 8), 'msg', result_mod_36177)
        
        # Call to __init__(...): (line 687)
        # Processing the call arguments (line 687)
        # Getting the type of 'self' (line 687)
        self_36180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 30), 'self', False)
        # Getting the type of 'msg' (line 687)
        msg_36181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 36), 'msg', False)
        # Processing the call keyword arguments (line 687)
        kwargs_36182 = {}
        # Getting the type of 'RuntimeError' (line 687)
        RuntimeError_36178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 8), 'RuntimeError', False)
        # Obtaining the member '__init__' of a type (line 687)
        init___36179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 8), RuntimeError_36178, '__init__')
        # Calling __init__(args, kwargs) (line 687)
        init___call_result_36183 = invoke(stypy.reporting.localization.Localization(__file__, 687, 8), init___36179, *[self_36180, msg_36181], **kwargs_36182)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'IntegratorConcurrencyError' (line 675)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 0), 'IntegratorConcurrencyError', IntegratorConcurrencyError)
# Declaration of the 'IntegratorBase' class

class IntegratorBase(object, ):
    
    # Assigning a Name to a Name (line 691):
    
    # Assigning a Name to a Name (line 692):
    
    # Assigning a Name to a Name (line 693):
    
    # Assigning a Name to a Name (line 694):
    
    # Assigning a Name to a Name (line 695):
    
    # Assigning a Name to a Name (line 696):
    
    # Assigning a List to a Name (line 697):
    
    # Assigning a Name to a Name (line 698):

    @norecursion
    def acquire_new_handle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'acquire_new_handle'
        module_type_store = module_type_store.open_function_context('acquire_new_handle', 700, 4, False)
        # Assigning a type to the variable 'self' (line 701)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntegratorBase.acquire_new_handle.__dict__.__setitem__('stypy_localization', localization)
        IntegratorBase.acquire_new_handle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntegratorBase.acquire_new_handle.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntegratorBase.acquire_new_handle.__dict__.__setitem__('stypy_function_name', 'IntegratorBase.acquire_new_handle')
        IntegratorBase.acquire_new_handle.__dict__.__setitem__('stypy_param_names_list', [])
        IntegratorBase.acquire_new_handle.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntegratorBase.acquire_new_handle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntegratorBase.acquire_new_handle.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntegratorBase.acquire_new_handle.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntegratorBase.acquire_new_handle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntegratorBase.acquire_new_handle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntegratorBase.acquire_new_handle', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'acquire_new_handle', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'acquire_new_handle(...)' code ##################

        
        # Getting the type of 'self' (line 704)
        self_36184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'self')
        # Obtaining the member '__class__' of a type (line 704)
        class___36185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 8), self_36184, '__class__')
        # Obtaining the member 'active_global_handle' of a type (line 704)
        active_global_handle_36186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 8), class___36185, 'active_global_handle')
        int_36187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 47), 'int')
        # Applying the binary operator '+=' (line 704)
        result_iadd_36188 = python_operator(stypy.reporting.localization.Localization(__file__, 704, 8), '+=', active_global_handle_36186, int_36187)
        # Getting the type of 'self' (line 704)
        self_36189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'self')
        # Obtaining the member '__class__' of a type (line 704)
        class___36190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 8), self_36189, '__class__')
        # Setting the type of the member 'active_global_handle' of a type (line 704)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 8), class___36190, 'active_global_handle', result_iadd_36188)
        
        
        # Assigning a Attribute to a Attribute (line 705):
        
        # Assigning a Attribute to a Attribute (line 705):
        # Getting the type of 'self' (line 705)
        self_36191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 22), 'self')
        # Obtaining the member '__class__' of a type (line 705)
        class___36192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 22), self_36191, '__class__')
        # Obtaining the member 'active_global_handle' of a type (line 705)
        active_global_handle_36193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 22), class___36192, 'active_global_handle')
        # Getting the type of 'self' (line 705)
        self_36194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'self')
        # Setting the type of the member 'handle' of a type (line 705)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 8), self_36194, 'handle', active_global_handle_36193)
        
        # ################# End of 'acquire_new_handle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'acquire_new_handle' in the type store
        # Getting the type of 'stypy_return_type' (line 700)
        stypy_return_type_36195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36195)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'acquire_new_handle'
        return stypy_return_type_36195


    @norecursion
    def check_handle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_handle'
        module_type_store = module_type_store.open_function_context('check_handle', 707, 4, False)
        # Assigning a type to the variable 'self' (line 708)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntegratorBase.check_handle.__dict__.__setitem__('stypy_localization', localization)
        IntegratorBase.check_handle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntegratorBase.check_handle.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntegratorBase.check_handle.__dict__.__setitem__('stypy_function_name', 'IntegratorBase.check_handle')
        IntegratorBase.check_handle.__dict__.__setitem__('stypy_param_names_list', [])
        IntegratorBase.check_handle.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntegratorBase.check_handle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntegratorBase.check_handle.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntegratorBase.check_handle.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntegratorBase.check_handle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntegratorBase.check_handle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntegratorBase.check_handle', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_handle', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_handle(...)' code ##################

        
        
        # Getting the type of 'self' (line 708)
        self_36196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 11), 'self')
        # Obtaining the member 'handle' of a type (line 708)
        handle_36197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 11), self_36196, 'handle')
        # Getting the type of 'self' (line 708)
        self_36198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 30), 'self')
        # Obtaining the member '__class__' of a type (line 708)
        class___36199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 30), self_36198, '__class__')
        # Obtaining the member 'active_global_handle' of a type (line 708)
        active_global_handle_36200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 30), class___36199, 'active_global_handle')
        # Applying the binary operator 'isnot' (line 708)
        result_is_not_36201 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 11), 'isnot', handle_36197, active_global_handle_36200)
        
        # Testing the type of an if condition (line 708)
        if_condition_36202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 708, 8), result_is_not_36201)
        # Assigning a type to the variable 'if_condition_36202' (line 708)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'if_condition_36202', if_condition_36202)
        # SSA begins for if statement (line 708)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IntegratorConcurrencyError(...): (line 709)
        # Processing the call arguments (line 709)
        # Getting the type of 'self' (line 709)
        self_36204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 45), 'self', False)
        # Obtaining the member '__class__' of a type (line 709)
        class___36205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 45), self_36204, '__class__')
        # Obtaining the member '__name__' of a type (line 709)
        name___36206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 45), class___36205, '__name__')
        # Processing the call keyword arguments (line 709)
        kwargs_36207 = {}
        # Getting the type of 'IntegratorConcurrencyError' (line 709)
        IntegratorConcurrencyError_36203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 18), 'IntegratorConcurrencyError', False)
        # Calling IntegratorConcurrencyError(args, kwargs) (line 709)
        IntegratorConcurrencyError_call_result_36208 = invoke(stypy.reporting.localization.Localization(__file__, 709, 18), IntegratorConcurrencyError_36203, *[name___36206], **kwargs_36207)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 709, 12), IntegratorConcurrencyError_call_result_36208, 'raise parameter', BaseException)
        # SSA join for if statement (line 708)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_handle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_handle' in the type store
        # Getting the type of 'stypy_return_type' (line 707)
        stypy_return_type_36209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36209)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_handle'
        return stypy_return_type_36209


    @norecursion
    def reset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reset'
        module_type_store = module_type_store.open_function_context('reset', 711, 4, False)
        # Assigning a type to the variable 'self' (line 712)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntegratorBase.reset.__dict__.__setitem__('stypy_localization', localization)
        IntegratorBase.reset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntegratorBase.reset.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntegratorBase.reset.__dict__.__setitem__('stypy_function_name', 'IntegratorBase.reset')
        IntegratorBase.reset.__dict__.__setitem__('stypy_param_names_list', ['n', 'has_jac'])
        IntegratorBase.reset.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntegratorBase.reset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntegratorBase.reset.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntegratorBase.reset.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntegratorBase.reset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntegratorBase.reset.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntegratorBase.reset', ['n', 'has_jac'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reset', localization, ['n', 'has_jac'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reset(...)' code ##################

        str_36210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, (-1)), 'str', 'Prepare integrator for call: allocate memory, set flags, etc.\n        n - number of equations.\n        has_jac - if user has supplied function for evaluating Jacobian.\n        ')
        
        # ################# End of 'reset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset' in the type store
        # Getting the type of 'stypy_return_type' (line 711)
        stypy_return_type_36211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36211)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset'
        return stypy_return_type_36211


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 717, 4, False)
        # Assigning a type to the variable 'self' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntegratorBase.run.__dict__.__setitem__('stypy_localization', localization)
        IntegratorBase.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntegratorBase.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntegratorBase.run.__dict__.__setitem__('stypy_function_name', 'IntegratorBase.run')
        IntegratorBase.run.__dict__.__setitem__('stypy_param_names_list', ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'])
        IntegratorBase.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntegratorBase.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntegratorBase.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntegratorBase.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntegratorBase.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntegratorBase.run.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntegratorBase.run', ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        str_36212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, (-1)), 'str', 'Integrate from t=t0 to t=t1 using y0 as an initial condition.\n        Return 2-tuple (y1,t1) where y1 is the result and t=t1\n        defines the stoppage coordinate of the result.\n        ')
        
        # Call to NotImplementedError(...): (line 722)
        # Processing the call arguments (line 722)
        str_36214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 34), 'str', 'all integrators must define run(f, jac, t0, t1, y0, f_params, jac_params)')
        # Processing the call keyword arguments (line 722)
        kwargs_36215 = {}
        # Getting the type of 'NotImplementedError' (line 722)
        NotImplementedError_36213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 722)
        NotImplementedError_call_result_36216 = invoke(stypy.reporting.localization.Localization(__file__, 722, 14), NotImplementedError_36213, *[str_36214], **kwargs_36215)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 722, 8), NotImplementedError_call_result_36216, 'raise parameter', BaseException)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 717)
        stypy_return_type_36217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36217)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_36217


    @norecursion
    def step(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'step'
        module_type_store = module_type_store.open_function_context('step', 725, 4, False)
        # Assigning a type to the variable 'self' (line 726)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntegratorBase.step.__dict__.__setitem__('stypy_localization', localization)
        IntegratorBase.step.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntegratorBase.step.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntegratorBase.step.__dict__.__setitem__('stypy_function_name', 'IntegratorBase.step')
        IntegratorBase.step.__dict__.__setitem__('stypy_param_names_list', ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'])
        IntegratorBase.step.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntegratorBase.step.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntegratorBase.step.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntegratorBase.step.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntegratorBase.step.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntegratorBase.step.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntegratorBase.step', ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'step', localization, ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'step(...)' code ##################

        str_36218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 8), 'str', 'Make one integration step and return (y1,t1).')
        
        # Call to NotImplementedError(...): (line 727)
        # Processing the call arguments (line 727)
        str_36220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 34), 'str', '%s does not support step() method')
        # Getting the type of 'self' (line 728)
        self_36221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 34), 'self', False)
        # Obtaining the member '__class__' of a type (line 728)
        class___36222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 34), self_36221, '__class__')
        # Obtaining the member '__name__' of a type (line 728)
        name___36223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 34), class___36222, '__name__')
        # Applying the binary operator '%' (line 727)
        result_mod_36224 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 34), '%', str_36220, name___36223)
        
        # Processing the call keyword arguments (line 727)
        kwargs_36225 = {}
        # Getting the type of 'NotImplementedError' (line 727)
        NotImplementedError_36219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 727)
        NotImplementedError_call_result_36226 = invoke(stypy.reporting.localization.Localization(__file__, 727, 14), NotImplementedError_36219, *[result_mod_36224], **kwargs_36225)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 727, 8), NotImplementedError_call_result_36226, 'raise parameter', BaseException)
        
        # ################# End of 'step(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'step' in the type store
        # Getting the type of 'stypy_return_type' (line 725)
        stypy_return_type_36227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36227)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'step'
        return stypy_return_type_36227


    @norecursion
    def run_relax(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run_relax'
        module_type_store = module_type_store.open_function_context('run_relax', 730, 4, False)
        # Assigning a type to the variable 'self' (line 731)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntegratorBase.run_relax.__dict__.__setitem__('stypy_localization', localization)
        IntegratorBase.run_relax.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntegratorBase.run_relax.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntegratorBase.run_relax.__dict__.__setitem__('stypy_function_name', 'IntegratorBase.run_relax')
        IntegratorBase.run_relax.__dict__.__setitem__('stypy_param_names_list', ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'])
        IntegratorBase.run_relax.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntegratorBase.run_relax.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntegratorBase.run_relax.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntegratorBase.run_relax.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntegratorBase.run_relax.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntegratorBase.run_relax.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntegratorBase.run_relax', ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run_relax', localization, ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run_relax(...)' code ##################

        str_36228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 8), 'str', 'Integrate from t=t0 to t>=t1 and return (y1,t).')
        
        # Call to NotImplementedError(...): (line 732)
        # Processing the call arguments (line 732)
        str_36230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 34), 'str', '%s does not support run_relax() method')
        # Getting the type of 'self' (line 733)
        self_36231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 34), 'self', False)
        # Obtaining the member '__class__' of a type (line 733)
        class___36232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 34), self_36231, '__class__')
        # Obtaining the member '__name__' of a type (line 733)
        name___36233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 34), class___36232, '__name__')
        # Applying the binary operator '%' (line 732)
        result_mod_36234 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 34), '%', str_36230, name___36233)
        
        # Processing the call keyword arguments (line 732)
        kwargs_36235 = {}
        # Getting the type of 'NotImplementedError' (line 732)
        NotImplementedError_36229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 732)
        NotImplementedError_call_result_36236 = invoke(stypy.reporting.localization.Localization(__file__, 732, 14), NotImplementedError_36229, *[result_mod_36234], **kwargs_36235)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 732, 8), NotImplementedError_call_result_36236, 'raise parameter', BaseException)
        
        # ################# End of 'run_relax(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run_relax' in the type store
        # Getting the type of 'stypy_return_type' (line 730)
        stypy_return_type_36237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36237)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run_relax'
        return stypy_return_type_36237


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 690, 0, False)
        # Assigning a type to the variable 'self' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntegratorBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'IntegratorBase' (line 690)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 0), 'IntegratorBase', IntegratorBase)

# Assigning a Name to a Name (line 691):
# Getting the type of 'None' (line 691)
None_36238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 13), 'None')
# Getting the type of 'IntegratorBase'
IntegratorBase_36239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntegratorBase')
# Setting the type of the member 'runner' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntegratorBase_36239, 'runner', None_36238)

# Assigning a Name to a Name (line 692):
# Getting the type of 'None' (line 692)
None_36240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 14), 'None')
# Getting the type of 'IntegratorBase'
IntegratorBase_36241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntegratorBase')
# Setting the type of the member 'success' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntegratorBase_36241, 'success', None_36240)

# Assigning a Name to a Name (line 693):
# Getting the type of 'None' (line 693)
None_36242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 13), 'None')
# Getting the type of 'IntegratorBase'
IntegratorBase_36243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntegratorBase')
# Setting the type of the member 'istate' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntegratorBase_36243, 'istate', None_36242)

# Assigning a Name to a Name (line 694):
# Getting the type of 'None' (line 694)
None_36244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 25), 'None')
# Getting the type of 'IntegratorBase'
IntegratorBase_36245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntegratorBase')
# Setting the type of the member 'supports_run_relax' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntegratorBase_36245, 'supports_run_relax', None_36244)

# Assigning a Name to a Name (line 695):
# Getting the type of 'None' (line 695)
None_36246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 20), 'None')
# Getting the type of 'IntegratorBase'
IntegratorBase_36247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntegratorBase')
# Setting the type of the member 'supports_step' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntegratorBase_36247, 'supports_step', None_36246)

# Assigning a Name to a Name (line 696):
# Getting the type of 'False' (line 696)
False_36248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 22), 'False')
# Getting the type of 'IntegratorBase'
IntegratorBase_36249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntegratorBase')
# Setting the type of the member 'supports_solout' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntegratorBase_36249, 'supports_solout', False_36248)

# Assigning a List to a Name (line 697):

# Obtaining an instance of the builtin type 'list' (line 697)
list_36250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 697)

# Getting the type of 'IntegratorBase'
IntegratorBase_36251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntegratorBase')
# Setting the type of the member 'integrator_classes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntegratorBase_36251, 'integrator_classes', list_36250)

# Assigning a Name to a Name (line 698):
# Getting the type of 'float' (line 698)
float_36252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 13), 'float')
# Getting the type of 'IntegratorBase'
IntegratorBase_36253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntegratorBase')
# Setting the type of the member 'scalar' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntegratorBase_36253, 'scalar', float_36252)

@norecursion
def _vode_banded_jac_wrapper(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_vode_banded_jac_wrapper'
    module_type_store = module_type_store.open_function_context('_vode_banded_jac_wrapper', 738, 0, False)
    
    # Passed parameters checking function
    _vode_banded_jac_wrapper.stypy_localization = localization
    _vode_banded_jac_wrapper.stypy_type_of_self = None
    _vode_banded_jac_wrapper.stypy_type_store = module_type_store
    _vode_banded_jac_wrapper.stypy_function_name = '_vode_banded_jac_wrapper'
    _vode_banded_jac_wrapper.stypy_param_names_list = ['jacfunc', 'ml', 'jac_params']
    _vode_banded_jac_wrapper.stypy_varargs_param_name = None
    _vode_banded_jac_wrapper.stypy_kwargs_param_name = None
    _vode_banded_jac_wrapper.stypy_call_defaults = defaults
    _vode_banded_jac_wrapper.stypy_call_varargs = varargs
    _vode_banded_jac_wrapper.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_vode_banded_jac_wrapper', ['jacfunc', 'ml', 'jac_params'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_vode_banded_jac_wrapper', localization, ['jacfunc', 'ml', 'jac_params'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_vode_banded_jac_wrapper(...)' code ##################

    str_36254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, (-1)), 'str', '\n    Wrap a banded Jacobian function with a function that pads\n    the Jacobian with `ml` rows of zeros.\n    ')

    @norecursion
    def jac_wrapper(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac_wrapper'
        module_type_store = module_type_store.open_function_context('jac_wrapper', 744, 4, False)
        
        # Passed parameters checking function
        jac_wrapper.stypy_localization = localization
        jac_wrapper.stypy_type_of_self = None
        jac_wrapper.stypy_type_store = module_type_store
        jac_wrapper.stypy_function_name = 'jac_wrapper'
        jac_wrapper.stypy_param_names_list = ['t', 'y']
        jac_wrapper.stypy_varargs_param_name = None
        jac_wrapper.stypy_kwargs_param_name = None
        jac_wrapper.stypy_call_defaults = defaults
        jac_wrapper.stypy_call_varargs = varargs
        jac_wrapper.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapper', ['t', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_wrapper', localization, ['t', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_wrapper(...)' code ##################

        
        # Assigning a Call to a Name (line 745):
        
        # Assigning a Call to a Name (line 745):
        
        # Call to asarray(...): (line 745)
        # Processing the call arguments (line 745)
        
        # Call to jacfunc(...): (line 745)
        # Processing the call arguments (line 745)
        # Getting the type of 't' (line 745)
        t_36257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 30), 't', False)
        # Getting the type of 'y' (line 745)
        y_36258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 33), 'y', False)
        # Getting the type of 'jac_params' (line 745)
        jac_params_36259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 37), 'jac_params', False)
        # Processing the call keyword arguments (line 745)
        kwargs_36260 = {}
        # Getting the type of 'jacfunc' (line 745)
        jacfunc_36256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 22), 'jacfunc', False)
        # Calling jacfunc(args, kwargs) (line 745)
        jacfunc_call_result_36261 = invoke(stypy.reporting.localization.Localization(__file__, 745, 22), jacfunc_36256, *[t_36257, y_36258, jac_params_36259], **kwargs_36260)
        
        # Processing the call keyword arguments (line 745)
        kwargs_36262 = {}
        # Getting the type of 'asarray' (line 745)
        asarray_36255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 14), 'asarray', False)
        # Calling asarray(args, kwargs) (line 745)
        asarray_call_result_36263 = invoke(stypy.reporting.localization.Localization(__file__, 745, 14), asarray_36255, *[jacfunc_call_result_36261], **kwargs_36262)
        
        # Assigning a type to the variable 'jac' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'jac', asarray_call_result_36263)
        
        # Assigning a Call to a Name (line 746):
        
        # Assigning a Call to a Name (line 746):
        
        # Call to vstack(...): (line 746)
        # Processing the call arguments (line 746)
        
        # Obtaining an instance of the builtin type 'tuple' (line 746)
        tuple_36265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 746)
        # Adding element type (line 746)
        # Getting the type of 'jac' (line 746)
        jac_36266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 29), 'jac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 29), tuple_36265, jac_36266)
        # Adding element type (line 746)
        
        # Call to zeros(...): (line 746)
        # Processing the call arguments (line 746)
        
        # Obtaining an instance of the builtin type 'tuple' (line 746)
        tuple_36268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 746)
        # Adding element type (line 746)
        # Getting the type of 'ml' (line 746)
        ml_36269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 41), 'ml', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 41), tuple_36268, ml_36269)
        # Adding element type (line 746)
        
        # Obtaining the type of the subscript
        int_36270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 55), 'int')
        # Getting the type of 'jac' (line 746)
        jac_36271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 45), 'jac', False)
        # Obtaining the member 'shape' of a type (line 746)
        shape_36272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 45), jac_36271, 'shape')
        # Obtaining the member '__getitem__' of a type (line 746)
        getitem___36273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 45), shape_36272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 746)
        subscript_call_result_36274 = invoke(stypy.reporting.localization.Localization(__file__, 746, 45), getitem___36273, int_36270)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 41), tuple_36268, subscript_call_result_36274)
        
        # Processing the call keyword arguments (line 746)
        kwargs_36275 = {}
        # Getting the type of 'zeros' (line 746)
        zeros_36267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 34), 'zeros', False)
        # Calling zeros(args, kwargs) (line 746)
        zeros_call_result_36276 = invoke(stypy.reporting.localization.Localization(__file__, 746, 34), zeros_36267, *[tuple_36268], **kwargs_36275)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 29), tuple_36265, zeros_call_result_36276)
        
        # Processing the call keyword arguments (line 746)
        kwargs_36277 = {}
        # Getting the type of 'vstack' (line 746)
        vstack_36264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 21), 'vstack', False)
        # Calling vstack(args, kwargs) (line 746)
        vstack_call_result_36278 = invoke(stypy.reporting.localization.Localization(__file__, 746, 21), vstack_36264, *[tuple_36265], **kwargs_36277)
        
        # Assigning a type to the variable 'padded_jac' (line 746)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 8), 'padded_jac', vstack_call_result_36278)
        # Getting the type of 'padded_jac' (line 747)
        padded_jac_36279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 15), 'padded_jac')
        # Assigning a type to the variable 'stypy_return_type' (line 747)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 8), 'stypy_return_type', padded_jac_36279)
        
        # ################# End of 'jac_wrapper(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_wrapper' in the type store
        # Getting the type of 'stypy_return_type' (line 744)
        stypy_return_type_36280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36280)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_wrapper'
        return stypy_return_type_36280

    # Assigning a type to the variable 'jac_wrapper' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 4), 'jac_wrapper', jac_wrapper)
    # Getting the type of 'jac_wrapper' (line 749)
    jac_wrapper_36281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 11), 'jac_wrapper')
    # Assigning a type to the variable 'stypy_return_type' (line 749)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 4), 'stypy_return_type', jac_wrapper_36281)
    
    # ################# End of '_vode_banded_jac_wrapper(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_vode_banded_jac_wrapper' in the type store
    # Getting the type of 'stypy_return_type' (line 738)
    stypy_return_type_36282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36282)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_vode_banded_jac_wrapper'
    return stypy_return_type_36282

# Assigning a type to the variable '_vode_banded_jac_wrapper' (line 738)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 0), '_vode_banded_jac_wrapper', _vode_banded_jac_wrapper)
# Declaration of the 'vode' class
# Getting the type of 'IntegratorBase' (line 752)
IntegratorBase_36283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 11), 'IntegratorBase')

class vode(IntegratorBase_36283, ):
    
    # Assigning a Call to a Name (line 753):
    
    # Assigning a Dict to a Name (line 755):
    
    # Assigning a Num to a Name (line 764):
    
    # Assigning a Num to a Name (line 765):
    
    # Assigning a Num to a Name (line 766):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_36284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 24), 'str', 'adams')
        # Getting the type of 'False' (line 770)
        False_36285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 31), 'False')
        float_36286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 22), 'float')
        float_36287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 33), 'float')
        # Getting the type of 'None' (line 772)
        None_36288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 23), 'None')
        # Getting the type of 'None' (line 772)
        None_36289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 35), 'None')
        int_36290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 23), 'int')
        int_36291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 24), 'int')
        float_36292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 26), 'float')
        float_36293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 26), 'float')
        float_36294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 28), 'float')
        defaults = [str_36284, False_36285, float_36286, float_36287, None_36288, None_36289, int_36290, int_36291, float_36292, float_36293, float_36294]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 768, 4, False)
        # Assigning a type to the variable 'self' (line 769)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vode.__init__', ['method', 'with_jacobian', 'rtol', 'atol', 'lband', 'uband', 'order', 'nsteps', 'max_step', 'min_step', 'first_step'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['method', 'with_jacobian', 'rtol', 'atol', 'lband', 'uband', 'order', 'nsteps', 'max_step', 'min_step', 'first_step'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Call to match(...): (line 780)
        # Processing the call arguments (line 780)
        # Getting the type of 'method' (line 780)
        method_36297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 20), 'method', False)
        str_36298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 28), 'str', 'adams')
        # Getting the type of 're' (line 780)
        re_36299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 38), 're', False)
        # Obtaining the member 'I' of a type (line 780)
        I_36300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 38), re_36299, 'I')
        # Processing the call keyword arguments (line 780)
        kwargs_36301 = {}
        # Getting the type of 're' (line 780)
        re_36295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 11), 're', False)
        # Obtaining the member 'match' of a type (line 780)
        match_36296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 11), re_36295, 'match')
        # Calling match(args, kwargs) (line 780)
        match_call_result_36302 = invoke(stypy.reporting.localization.Localization(__file__, 780, 11), match_36296, *[method_36297, str_36298, I_36300], **kwargs_36301)
        
        # Testing the type of an if condition (line 780)
        if_condition_36303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 780, 8), match_call_result_36302)
        # Assigning a type to the variable 'if_condition_36303' (line 780)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'if_condition_36303', if_condition_36303)
        # SSA begins for if statement (line 780)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 781):
        
        # Assigning a Num to a Attribute (line 781):
        int_36304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 24), 'int')
        # Getting the type of 'self' (line 781)
        self_36305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 12), 'self')
        # Setting the type of the member 'meth' of a type (line 781)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 12), self_36305, 'meth', int_36304)
        # SSA branch for the else part of an if statement (line 780)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to match(...): (line 782)
        # Processing the call arguments (line 782)
        # Getting the type of 'method' (line 782)
        method_36308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 22), 'method', False)
        str_36309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 30), 'str', 'bdf')
        # Getting the type of 're' (line 782)
        re_36310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 38), 're', False)
        # Obtaining the member 'I' of a type (line 782)
        I_36311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 38), re_36310, 'I')
        # Processing the call keyword arguments (line 782)
        kwargs_36312 = {}
        # Getting the type of 're' (line 782)
        re_36306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 13), 're', False)
        # Obtaining the member 'match' of a type (line 782)
        match_36307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 13), re_36306, 'match')
        # Calling match(args, kwargs) (line 782)
        match_call_result_36313 = invoke(stypy.reporting.localization.Localization(__file__, 782, 13), match_36307, *[method_36308, str_36309, I_36311], **kwargs_36312)
        
        # Testing the type of an if condition (line 782)
        if_condition_36314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 782, 13), match_call_result_36313)
        # Assigning a type to the variable 'if_condition_36314' (line 782)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 13), 'if_condition_36314', if_condition_36314)
        # SSA begins for if statement (line 782)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 783):
        
        # Assigning a Num to a Attribute (line 783):
        int_36315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 24), 'int')
        # Getting the type of 'self' (line 783)
        self_36316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'self')
        # Setting the type of the member 'meth' of a type (line 783)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 12), self_36316, 'meth', int_36315)
        # SSA branch for the else part of an if statement (line 782)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 785)
        # Processing the call arguments (line 785)
        str_36318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 29), 'str', 'Unknown integration method %s')
        # Getting the type of 'method' (line 785)
        method_36319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 63), 'method', False)
        # Applying the binary operator '%' (line 785)
        result_mod_36320 = python_operator(stypy.reporting.localization.Localization(__file__, 785, 29), '%', str_36318, method_36319)
        
        # Processing the call keyword arguments (line 785)
        kwargs_36321 = {}
        # Getting the type of 'ValueError' (line 785)
        ValueError_36317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 785)
        ValueError_call_result_36322 = invoke(stypy.reporting.localization.Localization(__file__, 785, 18), ValueError_36317, *[result_mod_36320], **kwargs_36321)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 785, 12), ValueError_call_result_36322, 'raise parameter', BaseException)
        # SSA join for if statement (line 782)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 780)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 786):
        
        # Assigning a Name to a Attribute (line 786):
        # Getting the type of 'with_jacobian' (line 786)
        with_jacobian_36323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 29), 'with_jacobian')
        # Getting the type of 'self' (line 786)
        self_36324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'self')
        # Setting the type of the member 'with_jacobian' of a type (line 786)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 8), self_36324, 'with_jacobian', with_jacobian_36323)
        
        # Assigning a Name to a Attribute (line 787):
        
        # Assigning a Name to a Attribute (line 787):
        # Getting the type of 'rtol' (line 787)
        rtol_36325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 20), 'rtol')
        # Getting the type of 'self' (line 787)
        self_36326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 8), 'self')
        # Setting the type of the member 'rtol' of a type (line 787)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 8), self_36326, 'rtol', rtol_36325)
        
        # Assigning a Name to a Attribute (line 788):
        
        # Assigning a Name to a Attribute (line 788):
        # Getting the type of 'atol' (line 788)
        atol_36327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 20), 'atol')
        # Getting the type of 'self' (line 788)
        self_36328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 8), 'self')
        # Setting the type of the member 'atol' of a type (line 788)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 8), self_36328, 'atol', atol_36327)
        
        # Assigning a Name to a Attribute (line 789):
        
        # Assigning a Name to a Attribute (line 789):
        # Getting the type of 'uband' (line 789)
        uband_36329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 18), 'uband')
        # Getting the type of 'self' (line 789)
        self_36330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 8), 'self')
        # Setting the type of the member 'mu' of a type (line 789)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 8), self_36330, 'mu', uband_36329)
        
        # Assigning a Name to a Attribute (line 790):
        
        # Assigning a Name to a Attribute (line 790):
        # Getting the type of 'lband' (line 790)
        lband_36331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 18), 'lband')
        # Getting the type of 'self' (line 790)
        self_36332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'self')
        # Setting the type of the member 'ml' of a type (line 790)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 8), self_36332, 'ml', lband_36331)
        
        # Assigning a Name to a Attribute (line 792):
        
        # Assigning a Name to a Attribute (line 792):
        # Getting the type of 'order' (line 792)
        order_36333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 21), 'order')
        # Getting the type of 'self' (line 792)
        self_36334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 8), 'self')
        # Setting the type of the member 'order' of a type (line 792)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 8), self_36334, 'order', order_36333)
        
        # Assigning a Name to a Attribute (line 793):
        
        # Assigning a Name to a Attribute (line 793):
        # Getting the type of 'nsteps' (line 793)
        nsteps_36335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 22), 'nsteps')
        # Getting the type of 'self' (line 793)
        self_36336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 8), 'self')
        # Setting the type of the member 'nsteps' of a type (line 793)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 8), self_36336, 'nsteps', nsteps_36335)
        
        # Assigning a Name to a Attribute (line 794):
        
        # Assigning a Name to a Attribute (line 794):
        # Getting the type of 'max_step' (line 794)
        max_step_36337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 24), 'max_step')
        # Getting the type of 'self' (line 794)
        self_36338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'self')
        # Setting the type of the member 'max_step' of a type (line 794)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 8), self_36338, 'max_step', max_step_36337)
        
        # Assigning a Name to a Attribute (line 795):
        
        # Assigning a Name to a Attribute (line 795):
        # Getting the type of 'min_step' (line 795)
        min_step_36339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 24), 'min_step')
        # Getting the type of 'self' (line 795)
        self_36340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 8), 'self')
        # Setting the type of the member 'min_step' of a type (line 795)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 8), self_36340, 'min_step', min_step_36339)
        
        # Assigning a Name to a Attribute (line 796):
        
        # Assigning a Name to a Attribute (line 796):
        # Getting the type of 'first_step' (line 796)
        first_step_36341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 26), 'first_step')
        # Getting the type of 'self' (line 796)
        self_36342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 8), 'self')
        # Setting the type of the member 'first_step' of a type (line 796)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 8), self_36342, 'first_step', first_step_36341)
        
        # Assigning a Num to a Attribute (line 797):
        
        # Assigning a Num to a Attribute (line 797):
        int_36343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 23), 'int')
        # Getting the type of 'self' (line 797)
        self_36344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 8), 'self')
        # Setting the type of the member 'success' of a type (line 797)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 8), self_36344, 'success', int_36343)
        
        # Assigning a Name to a Attribute (line 799):
        
        # Assigning a Name to a Attribute (line 799):
        # Getting the type of 'False' (line 799)
        False_36345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 27), 'False')
        # Getting the type of 'self' (line 799)
        self_36346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 8), 'self')
        # Setting the type of the member 'initialized' of a type (line 799)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 8), self_36346, 'initialized', False_36345)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _determine_mf_and_set_bands(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_determine_mf_and_set_bands'
        module_type_store = module_type_store.open_function_context('_determine_mf_and_set_bands', 801, 4, False)
        # Assigning a type to the variable 'self' (line 802)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vode._determine_mf_and_set_bands.__dict__.__setitem__('stypy_localization', localization)
        vode._determine_mf_and_set_bands.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vode._determine_mf_and_set_bands.__dict__.__setitem__('stypy_type_store', module_type_store)
        vode._determine_mf_and_set_bands.__dict__.__setitem__('stypy_function_name', 'vode._determine_mf_and_set_bands')
        vode._determine_mf_and_set_bands.__dict__.__setitem__('stypy_param_names_list', ['has_jac'])
        vode._determine_mf_and_set_bands.__dict__.__setitem__('stypy_varargs_param_name', None)
        vode._determine_mf_and_set_bands.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vode._determine_mf_and_set_bands.__dict__.__setitem__('stypy_call_defaults', defaults)
        vode._determine_mf_and_set_bands.__dict__.__setitem__('stypy_call_varargs', varargs)
        vode._determine_mf_and_set_bands.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vode._determine_mf_and_set_bands.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vode._determine_mf_and_set_bands', ['has_jac'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_determine_mf_and_set_bands', localization, ['has_jac'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_determine_mf_and_set_bands(...)' code ##################

        str_36347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, (-1)), 'str', '\n        Determine the `MF` parameter (Method Flag) for the Fortran subroutine `dvode`.\n\n        In the Fortran code, the legal values of `MF` are:\n            10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25,\n            -11, -12, -14, -15, -21, -22, -24, -25\n        but this python wrapper does not use negative values.\n\n        Returns\n\n            mf  = 10*self.meth + miter\n\n        self.meth is the linear multistep method:\n            self.meth == 1:  method="adams"\n            self.meth == 2:  method="bdf"\n\n        miter is the correction iteration method:\n            miter == 0:  Functional iteraton; no Jacobian involved.\n            miter == 1:  Chord iteration with user-supplied full Jacobian\n            miter == 2:  Chord iteration with internally computed full Jacobian\n            miter == 3:  Chord iteration with internally computed diagonal Jacobian\n            miter == 4:  Chord iteration with user-supplied banded Jacobian\n            miter == 5:  Chord iteration with internally computed banded Jacobian\n\n        Side effects: If either self.mu or self.ml is not None and the other is None,\n        then the one that is None is set to 0.\n        ')
        
        # Assigning a BoolOp to a Name (line 830):
        
        # Assigning a BoolOp to a Name (line 830):
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 830)
        self_36348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 24), 'self')
        # Obtaining the member 'mu' of a type (line 830)
        mu_36349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 24), self_36348, 'mu')
        # Getting the type of 'None' (line 830)
        None_36350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 39), 'None')
        # Applying the binary operator 'isnot' (line 830)
        result_is_not_36351 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 24), 'isnot', mu_36349, None_36350)
        
        
        # Getting the type of 'self' (line 830)
        self_36352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 47), 'self')
        # Obtaining the member 'ml' of a type (line 830)
        ml_36353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 47), self_36352, 'ml')
        # Getting the type of 'None' (line 830)
        None_36354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 62), 'None')
        # Applying the binary operator 'isnot' (line 830)
        result_is_not_36355 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 47), 'isnot', ml_36353, None_36354)
        
        # Applying the binary operator 'or' (line 830)
        result_or_keyword_36356 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 24), 'or', result_is_not_36351, result_is_not_36355)
        
        # Assigning a type to the variable 'jac_is_banded' (line 830)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 8), 'jac_is_banded', result_or_keyword_36356)
        
        # Getting the type of 'jac_is_banded' (line 831)
        jac_is_banded_36357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 11), 'jac_is_banded')
        # Testing the type of an if condition (line 831)
        if_condition_36358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 831, 8), jac_is_banded_36357)
        # Assigning a type to the variable 'if_condition_36358' (line 831)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'if_condition_36358', if_condition_36358)
        # SSA begins for if statement (line 831)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 832)
        # Getting the type of 'self' (line 832)
        self_36359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 15), 'self')
        # Obtaining the member 'mu' of a type (line 832)
        mu_36360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 15), self_36359, 'mu')
        # Getting the type of 'None' (line 832)
        None_36361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 26), 'None')
        
        (may_be_36362, more_types_in_union_36363) = may_be_none(mu_36360, None_36361)

        if may_be_36362:

            if more_types_in_union_36363:
                # Runtime conditional SSA (line 832)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Attribute (line 833):
            
            # Assigning a Num to a Attribute (line 833):
            int_36364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 26), 'int')
            # Getting the type of 'self' (line 833)
            self_36365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 16), 'self')
            # Setting the type of the member 'mu' of a type (line 833)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 16), self_36365, 'mu', int_36364)

            if more_types_in_union_36363:
                # SSA join for if statement (line 832)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 834)
        # Getting the type of 'self' (line 834)
        self_36366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 15), 'self')
        # Obtaining the member 'ml' of a type (line 834)
        ml_36367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 15), self_36366, 'ml')
        # Getting the type of 'None' (line 834)
        None_36368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 26), 'None')
        
        (may_be_36369, more_types_in_union_36370) = may_be_none(ml_36367, None_36368)

        if may_be_36369:

            if more_types_in_union_36370:
                # Runtime conditional SSA (line 834)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Attribute (line 835):
            
            # Assigning a Num to a Attribute (line 835):
            int_36371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 26), 'int')
            # Getting the type of 'self' (line 835)
            self_36372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 16), 'self')
            # Setting the type of the member 'ml' of a type (line 835)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 16), self_36372, 'ml', int_36371)

            if more_types_in_union_36370:
                # SSA join for if statement (line 834)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 831)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'has_jac' (line 838)
        has_jac_36373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 11), 'has_jac')
        # Testing the type of an if condition (line 838)
        if_condition_36374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 838, 8), has_jac_36373)
        # Assigning a type to the variable 'if_condition_36374' (line 838)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'if_condition_36374', if_condition_36374)
        # SSA begins for if statement (line 838)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'jac_is_banded' (line 839)
        jac_is_banded_36375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 15), 'jac_is_banded')
        # Testing the type of an if condition (line 839)
        if_condition_36376 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 839, 12), jac_is_banded_36375)
        # Assigning a type to the variable 'if_condition_36376' (line 839)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 12), 'if_condition_36376', if_condition_36376)
        # SSA begins for if statement (line 839)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 840):
        
        # Assigning a Num to a Name (line 840):
        int_36377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 24), 'int')
        # Assigning a type to the variable 'miter' (line 840)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 16), 'miter', int_36377)
        # SSA branch for the else part of an if statement (line 839)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 842):
        
        # Assigning a Num to a Name (line 842):
        int_36378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 24), 'int')
        # Assigning a type to the variable 'miter' (line 842)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 16), 'miter', int_36378)
        # SSA join for if statement (line 839)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 838)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'jac_is_banded' (line 844)
        jac_is_banded_36379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 15), 'jac_is_banded')
        # Testing the type of an if condition (line 844)
        if_condition_36380 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 844, 12), jac_is_banded_36379)
        # Assigning a type to the variable 'if_condition_36380' (line 844)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 12), 'if_condition_36380', if_condition_36380)
        # SSA begins for if statement (line 844)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 845)
        self_36381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 19), 'self')
        # Obtaining the member 'ml' of a type (line 845)
        ml_36382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 19), self_36381, 'ml')
        # Getting the type of 'self' (line 845)
        self_36383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 30), 'self')
        # Obtaining the member 'mu' of a type (line 845)
        mu_36384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 30), self_36383, 'mu')
        # Applying the binary operator '==' (line 845)
        result_eq_36385 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 19), '==', ml_36382, mu_36384)
        int_36386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 41), 'int')
        # Applying the binary operator '==' (line 845)
        result_eq_36387 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 19), '==', mu_36384, int_36386)
        # Applying the binary operator '&' (line 845)
        result_and__36388 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 19), '&', result_eq_36385, result_eq_36387)
        
        # Testing the type of an if condition (line 845)
        if_condition_36389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 845, 16), result_and__36388)
        # Assigning a type to the variable 'if_condition_36389' (line 845)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 16), 'if_condition_36389', if_condition_36389)
        # SSA begins for if statement (line 845)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 846):
        
        # Assigning a Num to a Name (line 846):
        int_36390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 28), 'int')
        # Assigning a type to the variable 'miter' (line 846)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 20), 'miter', int_36390)
        # SSA branch for the else part of an if statement (line 845)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 848):
        
        # Assigning a Num to a Name (line 848):
        int_36391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 28), 'int')
        # Assigning a type to the variable 'miter' (line 848)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 20), 'miter', int_36391)
        # SSA join for if statement (line 845)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 844)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 851)
        self_36392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 19), 'self')
        # Obtaining the member 'with_jacobian' of a type (line 851)
        with_jacobian_36393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 19), self_36392, 'with_jacobian')
        # Testing the type of an if condition (line 851)
        if_condition_36394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 851, 16), with_jacobian_36393)
        # Assigning a type to the variable 'if_condition_36394' (line 851)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 16), 'if_condition_36394', if_condition_36394)
        # SSA begins for if statement (line 851)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 852):
        
        # Assigning a Num to a Name (line 852):
        int_36395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 28), 'int')
        # Assigning a type to the variable 'miter' (line 852)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 20), 'miter', int_36395)
        # SSA branch for the else part of an if statement (line 851)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 854):
        
        # Assigning a Num to a Name (line 854):
        int_36396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 854, 28), 'int')
        # Assigning a type to the variable 'miter' (line 854)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 20), 'miter', int_36396)
        # SSA join for if statement (line 851)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 844)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 838)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 856):
        
        # Assigning a BinOp to a Name (line 856):
        int_36397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 856, 13), 'int')
        # Getting the type of 'self' (line 856)
        self_36398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 18), 'self')
        # Obtaining the member 'meth' of a type (line 856)
        meth_36399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 18), self_36398, 'meth')
        # Applying the binary operator '*' (line 856)
        result_mul_36400 = python_operator(stypy.reporting.localization.Localization(__file__, 856, 13), '*', int_36397, meth_36399)
        
        # Getting the type of 'miter' (line 856)
        miter_36401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 30), 'miter')
        # Applying the binary operator '+' (line 856)
        result_add_36402 = python_operator(stypy.reporting.localization.Localization(__file__, 856, 13), '+', result_mul_36400, miter_36401)
        
        # Assigning a type to the variable 'mf' (line 856)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 8), 'mf', result_add_36402)
        # Getting the type of 'mf' (line 857)
        mf_36403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 15), 'mf')
        # Assigning a type to the variable 'stypy_return_type' (line 857)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 8), 'stypy_return_type', mf_36403)
        
        # ################# End of '_determine_mf_and_set_bands(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_determine_mf_and_set_bands' in the type store
        # Getting the type of 'stypy_return_type' (line 801)
        stypy_return_type_36404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36404)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_determine_mf_and_set_bands'
        return stypy_return_type_36404


    @norecursion
    def reset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reset'
        module_type_store = module_type_store.open_function_context('reset', 859, 4, False)
        # Assigning a type to the variable 'self' (line 860)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vode.reset.__dict__.__setitem__('stypy_localization', localization)
        vode.reset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vode.reset.__dict__.__setitem__('stypy_type_store', module_type_store)
        vode.reset.__dict__.__setitem__('stypy_function_name', 'vode.reset')
        vode.reset.__dict__.__setitem__('stypy_param_names_list', ['n', 'has_jac'])
        vode.reset.__dict__.__setitem__('stypy_varargs_param_name', None)
        vode.reset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vode.reset.__dict__.__setitem__('stypy_call_defaults', defaults)
        vode.reset.__dict__.__setitem__('stypy_call_varargs', varargs)
        vode.reset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vode.reset.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vode.reset', ['n', 'has_jac'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reset', localization, ['n', 'has_jac'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reset(...)' code ##################

        
        # Assigning a Call to a Name (line 860):
        
        # Assigning a Call to a Name (line 860):
        
        # Call to _determine_mf_and_set_bands(...): (line 860)
        # Processing the call arguments (line 860)
        # Getting the type of 'has_jac' (line 860)
        has_jac_36407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 46), 'has_jac', False)
        # Processing the call keyword arguments (line 860)
        kwargs_36408 = {}
        # Getting the type of 'self' (line 860)
        self_36405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 13), 'self', False)
        # Obtaining the member '_determine_mf_and_set_bands' of a type (line 860)
        _determine_mf_and_set_bands_36406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 13), self_36405, '_determine_mf_and_set_bands')
        # Calling _determine_mf_and_set_bands(args, kwargs) (line 860)
        _determine_mf_and_set_bands_call_result_36409 = invoke(stypy.reporting.localization.Localization(__file__, 860, 13), _determine_mf_and_set_bands_36406, *[has_jac_36407], **kwargs_36408)
        
        # Assigning a type to the variable 'mf' (line 860)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 8), 'mf', _determine_mf_and_set_bands_call_result_36409)
        
        
        # Getting the type of 'mf' (line 862)
        mf_36410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 11), 'mf')
        int_36411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 17), 'int')
        # Applying the binary operator '==' (line 862)
        result_eq_36412 = python_operator(stypy.reporting.localization.Localization(__file__, 862, 11), '==', mf_36410, int_36411)
        
        # Testing the type of an if condition (line 862)
        if_condition_36413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 862, 8), result_eq_36412)
        # Assigning a type to the variable 'if_condition_36413' (line 862)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 8), 'if_condition_36413', if_condition_36413)
        # SSA begins for if statement (line 862)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 863):
        
        # Assigning a BinOp to a Name (line 863):
        int_36414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 18), 'int')
        int_36415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 23), 'int')
        # Getting the type of 'n' (line 863)
        n_36416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 28), 'n')
        # Applying the binary operator '*' (line 863)
        result_mul_36417 = python_operator(stypy.reporting.localization.Localization(__file__, 863, 23), '*', int_36415, n_36416)
        
        # Applying the binary operator '+' (line 863)
        result_add_36418 = python_operator(stypy.reporting.localization.Localization(__file__, 863, 18), '+', int_36414, result_mul_36417)
        
        # Assigning a type to the variable 'lrw' (line 863)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 12), 'lrw', result_add_36418)
        # SSA branch for the else part of an if statement (line 862)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 864)
        mf_36419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'list' (line 864)
        list_36420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 864)
        # Adding element type (line 864)
        int_36421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 864, 19), list_36420, int_36421)
        # Adding element type (line 864)
        int_36422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 864, 19), list_36420, int_36422)
        
        # Applying the binary operator 'in' (line 864)
        result_contains_36423 = python_operator(stypy.reporting.localization.Localization(__file__, 864, 13), 'in', mf_36419, list_36420)
        
        # Testing the type of an if condition (line 864)
        if_condition_36424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 864, 13), result_contains_36423)
        # Assigning a type to the variable 'if_condition_36424' (line 864)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 13), 'if_condition_36424', if_condition_36424)
        # SSA begins for if statement (line 864)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 865):
        
        # Assigning a BinOp to a Name (line 865):
        int_36425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 865, 18), 'int')
        int_36426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 865, 23), 'int')
        # Getting the type of 'n' (line 865)
        n_36427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 28), 'n')
        # Applying the binary operator '*' (line 865)
        result_mul_36428 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 23), '*', int_36426, n_36427)
        
        # Applying the binary operator '+' (line 865)
        result_add_36429 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 18), '+', int_36425, result_mul_36428)
        
        int_36430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 865, 32), 'int')
        # Getting the type of 'n' (line 865)
        n_36431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 36), 'n')
        # Applying the binary operator '*' (line 865)
        result_mul_36432 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 32), '*', int_36430, n_36431)
        
        # Getting the type of 'n' (line 865)
        n_36433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 40), 'n')
        # Applying the binary operator '*' (line 865)
        result_mul_36434 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 38), '*', result_mul_36432, n_36433)
        
        # Applying the binary operator '+' (line 865)
        result_add_36435 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 30), '+', result_add_36429, result_mul_36434)
        
        # Assigning a type to the variable 'lrw' (line 865)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 12), 'lrw', result_add_36435)
        # SSA branch for the else part of an if statement (line 864)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 866)
        mf_36436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 13), 'mf')
        int_36437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 866, 19), 'int')
        # Applying the binary operator '==' (line 866)
        result_eq_36438 = python_operator(stypy.reporting.localization.Localization(__file__, 866, 13), '==', mf_36436, int_36437)
        
        # Testing the type of an if condition (line 866)
        if_condition_36439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 866, 13), result_eq_36438)
        # Assigning a type to the variable 'if_condition_36439' (line 866)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 13), 'if_condition_36439', if_condition_36439)
        # SSA begins for if statement (line 866)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 867):
        
        # Assigning a BinOp to a Name (line 867):
        int_36440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, 18), 'int')
        int_36441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, 23), 'int')
        # Getting the type of 'n' (line 867)
        n_36442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 28), 'n')
        # Applying the binary operator '*' (line 867)
        result_mul_36443 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 23), '*', int_36441, n_36442)
        
        # Applying the binary operator '+' (line 867)
        result_add_36444 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 18), '+', int_36440, result_mul_36443)
        
        # Assigning a type to the variable 'lrw' (line 867)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 12), 'lrw', result_add_36444)
        # SSA branch for the else part of an if statement (line 866)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 868)
        mf_36445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'list' (line 868)
        list_36446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 868)
        # Adding element type (line 868)
        int_36447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 868, 19), list_36446, int_36447)
        # Adding element type (line 868)
        int_36448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 868, 19), list_36446, int_36448)
        
        # Applying the binary operator 'in' (line 868)
        result_contains_36449 = python_operator(stypy.reporting.localization.Localization(__file__, 868, 13), 'in', mf_36445, list_36446)
        
        # Testing the type of an if condition (line 868)
        if_condition_36450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 868, 13), result_contains_36449)
        # Assigning a type to the variable 'if_condition_36450' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 13), 'if_condition_36450', if_condition_36450)
        # SSA begins for if statement (line 868)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 869):
        
        # Assigning a BinOp to a Name (line 869):
        int_36451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 18), 'int')
        int_36452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 23), 'int')
        # Getting the type of 'n' (line 869)
        n_36453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 28), 'n')
        # Applying the binary operator '*' (line 869)
        result_mul_36454 = python_operator(stypy.reporting.localization.Localization(__file__, 869, 23), '*', int_36452, n_36453)
        
        # Applying the binary operator '+' (line 869)
        result_add_36455 = python_operator(stypy.reporting.localization.Localization(__file__, 869, 18), '+', int_36451, result_mul_36454)
        
        int_36456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 33), 'int')
        # Getting the type of 'self' (line 869)
        self_36457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 37), 'self')
        # Obtaining the member 'ml' of a type (line 869)
        ml_36458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 37), self_36457, 'ml')
        # Applying the binary operator '*' (line 869)
        result_mul_36459 = python_operator(stypy.reporting.localization.Localization(__file__, 869, 33), '*', int_36456, ml_36458)
        
        int_36460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 47), 'int')
        # Getting the type of 'self' (line 869)
        self_36461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 51), 'self')
        # Obtaining the member 'mu' of a type (line 869)
        mu_36462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 51), self_36461, 'mu')
        # Applying the binary operator '*' (line 869)
        result_mul_36463 = python_operator(stypy.reporting.localization.Localization(__file__, 869, 47), '*', int_36460, mu_36462)
        
        # Applying the binary operator '+' (line 869)
        result_add_36464 = python_operator(stypy.reporting.localization.Localization(__file__, 869, 33), '+', result_mul_36459, result_mul_36463)
        
        # Getting the type of 'n' (line 869)
        n_36465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 62), 'n')
        # Applying the binary operator '*' (line 869)
        result_mul_36466 = python_operator(stypy.reporting.localization.Localization(__file__, 869, 32), '*', result_add_36464, n_36465)
        
        # Applying the binary operator '+' (line 869)
        result_add_36467 = python_operator(stypy.reporting.localization.Localization(__file__, 869, 30), '+', result_add_36455, result_mul_36466)
        
        # Assigning a type to the variable 'lrw' (line 869)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 12), 'lrw', result_add_36467)
        # SSA branch for the else part of an if statement (line 868)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 870)
        mf_36468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 13), 'mf')
        int_36469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 870, 19), 'int')
        # Applying the binary operator '==' (line 870)
        result_eq_36470 = python_operator(stypy.reporting.localization.Localization(__file__, 870, 13), '==', mf_36468, int_36469)
        
        # Testing the type of an if condition (line 870)
        if_condition_36471 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 870, 13), result_eq_36470)
        # Assigning a type to the variable 'if_condition_36471' (line 870)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 13), 'if_condition_36471', if_condition_36471)
        # SSA begins for if statement (line 870)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 871):
        
        # Assigning a BinOp to a Name (line 871):
        int_36472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 18), 'int')
        int_36473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 23), 'int')
        # Getting the type of 'n' (line 871)
        n_36474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 27), 'n')
        # Applying the binary operator '*' (line 871)
        result_mul_36475 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 23), '*', int_36473, n_36474)
        
        # Applying the binary operator '+' (line 871)
        result_add_36476 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 18), '+', int_36472, result_mul_36475)
        
        # Assigning a type to the variable 'lrw' (line 871)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 12), 'lrw', result_add_36476)
        # SSA branch for the else part of an if statement (line 870)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 872)
        mf_36477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'list' (line 872)
        list_36478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 872)
        # Adding element type (line 872)
        int_36479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 872, 19), list_36478, int_36479)
        # Adding element type (line 872)
        int_36480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 872, 19), list_36478, int_36480)
        
        # Applying the binary operator 'in' (line 872)
        result_contains_36481 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 13), 'in', mf_36477, list_36478)
        
        # Testing the type of an if condition (line 872)
        if_condition_36482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 872, 13), result_contains_36481)
        # Assigning a type to the variable 'if_condition_36482' (line 872)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 13), 'if_condition_36482', if_condition_36482)
        # SSA begins for if statement (line 872)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 873):
        
        # Assigning a BinOp to a Name (line 873):
        int_36483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 873, 18), 'int')
        int_36484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 873, 23), 'int')
        # Getting the type of 'n' (line 873)
        n_36485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 27), 'n')
        # Applying the binary operator '*' (line 873)
        result_mul_36486 = python_operator(stypy.reporting.localization.Localization(__file__, 873, 23), '*', int_36484, n_36485)
        
        # Applying the binary operator '+' (line 873)
        result_add_36487 = python_operator(stypy.reporting.localization.Localization(__file__, 873, 18), '+', int_36483, result_mul_36486)
        
        int_36488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 873, 31), 'int')
        # Getting the type of 'n' (line 873)
        n_36489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 35), 'n')
        # Applying the binary operator '*' (line 873)
        result_mul_36490 = python_operator(stypy.reporting.localization.Localization(__file__, 873, 31), '*', int_36488, n_36489)
        
        # Getting the type of 'n' (line 873)
        n_36491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 39), 'n')
        # Applying the binary operator '*' (line 873)
        result_mul_36492 = python_operator(stypy.reporting.localization.Localization(__file__, 873, 37), '*', result_mul_36490, n_36491)
        
        # Applying the binary operator '+' (line 873)
        result_add_36493 = python_operator(stypy.reporting.localization.Localization(__file__, 873, 29), '+', result_add_36487, result_mul_36492)
        
        # Assigning a type to the variable 'lrw' (line 873)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 12), 'lrw', result_add_36493)
        # SSA branch for the else part of an if statement (line 872)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 874)
        mf_36494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 13), 'mf')
        int_36495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 19), 'int')
        # Applying the binary operator '==' (line 874)
        result_eq_36496 = python_operator(stypy.reporting.localization.Localization(__file__, 874, 13), '==', mf_36494, int_36495)
        
        # Testing the type of an if condition (line 874)
        if_condition_36497 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 874, 13), result_eq_36496)
        # Assigning a type to the variable 'if_condition_36497' (line 874)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 874, 13), 'if_condition_36497', if_condition_36497)
        # SSA begins for if statement (line 874)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 875):
        
        # Assigning a BinOp to a Name (line 875):
        int_36498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 18), 'int')
        int_36499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 23), 'int')
        # Getting the type of 'n' (line 875)
        n_36500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 28), 'n')
        # Applying the binary operator '*' (line 875)
        result_mul_36501 = python_operator(stypy.reporting.localization.Localization(__file__, 875, 23), '*', int_36499, n_36500)
        
        # Applying the binary operator '+' (line 875)
        result_add_36502 = python_operator(stypy.reporting.localization.Localization(__file__, 875, 18), '+', int_36498, result_mul_36501)
        
        # Assigning a type to the variable 'lrw' (line 875)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 12), 'lrw', result_add_36502)
        # SSA branch for the else part of an if statement (line 874)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 876)
        mf_36503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'list' (line 876)
        list_36504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 876)
        # Adding element type (line 876)
        int_36505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 876, 19), list_36504, int_36505)
        # Adding element type (line 876)
        int_36506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 876, 19), list_36504, int_36506)
        
        # Applying the binary operator 'in' (line 876)
        result_contains_36507 = python_operator(stypy.reporting.localization.Localization(__file__, 876, 13), 'in', mf_36503, list_36504)
        
        # Testing the type of an if condition (line 876)
        if_condition_36508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 876, 13), result_contains_36507)
        # Assigning a type to the variable 'if_condition_36508' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 13), 'if_condition_36508', if_condition_36508)
        # SSA begins for if statement (line 876)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 877):
        
        # Assigning a BinOp to a Name (line 877):
        int_36509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 18), 'int')
        int_36510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 23), 'int')
        # Getting the type of 'n' (line 877)
        n_36511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 28), 'n')
        # Applying the binary operator '*' (line 877)
        result_mul_36512 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 23), '*', int_36510, n_36511)
        
        # Applying the binary operator '+' (line 877)
        result_add_36513 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 18), '+', int_36509, result_mul_36512)
        
        int_36514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 33), 'int')
        # Getting the type of 'self' (line 877)
        self_36515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 37), 'self')
        # Obtaining the member 'ml' of a type (line 877)
        ml_36516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 37), self_36515, 'ml')
        # Applying the binary operator '*' (line 877)
        result_mul_36517 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 33), '*', int_36514, ml_36516)
        
        int_36518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 47), 'int')
        # Getting the type of 'self' (line 877)
        self_36519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 51), 'self')
        # Obtaining the member 'mu' of a type (line 877)
        mu_36520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 51), self_36519, 'mu')
        # Applying the binary operator '*' (line 877)
        result_mul_36521 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 47), '*', int_36518, mu_36520)
        
        # Applying the binary operator '+' (line 877)
        result_add_36522 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 33), '+', result_mul_36517, result_mul_36521)
        
        # Getting the type of 'n' (line 877)
        n_36523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 62), 'n')
        # Applying the binary operator '*' (line 877)
        result_mul_36524 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 32), '*', result_add_36522, n_36523)
        
        # Applying the binary operator '+' (line 877)
        result_add_36525 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 30), '+', result_add_36513, result_mul_36524)
        
        # Assigning a type to the variable 'lrw' (line 877)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 12), 'lrw', result_add_36525)
        # SSA branch for the else part of an if statement (line 876)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 879)
        # Processing the call arguments (line 879)
        str_36527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 29), 'str', 'Unexpected mf=%s')
        # Getting the type of 'mf' (line 879)
        mf_36528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 50), 'mf', False)
        # Applying the binary operator '%' (line 879)
        result_mod_36529 = python_operator(stypy.reporting.localization.Localization(__file__, 879, 29), '%', str_36527, mf_36528)
        
        # Processing the call keyword arguments (line 879)
        kwargs_36530 = {}
        # Getting the type of 'ValueError' (line 879)
        ValueError_36526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 879)
        ValueError_call_result_36531 = invoke(stypy.reporting.localization.Localization(__file__, 879, 18), ValueError_36526, *[result_mod_36529], **kwargs_36530)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 879, 12), ValueError_call_result_36531, 'raise parameter', BaseException)
        # SSA join for if statement (line 876)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 874)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 872)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 870)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 868)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 866)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 864)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 862)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'mf' (line 881)
        mf_36532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 11), 'mf')
        int_36533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 16), 'int')
        # Applying the binary operator '%' (line 881)
        result_mod_36534 = python_operator(stypy.reporting.localization.Localization(__file__, 881, 11), '%', mf_36532, int_36533)
        
        
        # Obtaining an instance of the builtin type 'list' (line 881)
        list_36535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 881)
        # Adding element type (line 881)
        int_36536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 22), list_36535, int_36536)
        # Adding element type (line 881)
        int_36537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 22), list_36535, int_36537)
        
        # Applying the binary operator 'in' (line 881)
        result_contains_36538 = python_operator(stypy.reporting.localization.Localization(__file__, 881, 11), 'in', result_mod_36534, list_36535)
        
        # Testing the type of an if condition (line 881)
        if_condition_36539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 881, 8), result_contains_36538)
        # Assigning a type to the variable 'if_condition_36539' (line 881)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 8), 'if_condition_36539', if_condition_36539)
        # SSA begins for if statement (line 881)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 882):
        
        # Assigning a Num to a Name (line 882):
        int_36540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 18), 'int')
        # Assigning a type to the variable 'liw' (line 882)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 12), 'liw', int_36540)
        # SSA branch for the else part of an if statement (line 881)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 884):
        
        # Assigning a BinOp to a Name (line 884):
        int_36541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 18), 'int')
        # Getting the type of 'n' (line 884)
        n_36542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 23), 'n')
        # Applying the binary operator '+' (line 884)
        result_add_36543 = python_operator(stypy.reporting.localization.Localization(__file__, 884, 18), '+', int_36541, n_36542)
        
        # Assigning a type to the variable 'liw' (line 884)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 12), 'liw', result_add_36543)
        # SSA join for if statement (line 881)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 886):
        
        # Assigning a Call to a Name (line 886):
        
        # Call to zeros(...): (line 886)
        # Processing the call arguments (line 886)
        
        # Obtaining an instance of the builtin type 'tuple' (line 886)
        tuple_36545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 886)
        # Adding element type (line 886)
        # Getting the type of 'lrw' (line 886)
        lrw_36546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 23), 'lrw', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 886, 23), tuple_36545, lrw_36546)
        
        # Getting the type of 'float' (line 886)
        float_36547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 30), 'float', False)
        # Processing the call keyword arguments (line 886)
        kwargs_36548 = {}
        # Getting the type of 'zeros' (line 886)
        zeros_36544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 886)
        zeros_call_result_36549 = invoke(stypy.reporting.localization.Localization(__file__, 886, 16), zeros_36544, *[tuple_36545, float_36547], **kwargs_36548)
        
        # Assigning a type to the variable 'rwork' (line 886)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 886, 8), 'rwork', zeros_call_result_36549)
        
        # Assigning a Attribute to a Subscript (line 887):
        
        # Assigning a Attribute to a Subscript (line 887):
        # Getting the type of 'self' (line 887)
        self_36550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 19), 'self')
        # Obtaining the member 'first_step' of a type (line 887)
        first_step_36551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 19), self_36550, 'first_step')
        # Getting the type of 'rwork' (line 887)
        rwork_36552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 8), 'rwork')
        int_36553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 14), 'int')
        # Storing an element on a container (line 887)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 887, 8), rwork_36552, (int_36553, first_step_36551))
        
        # Assigning a Attribute to a Subscript (line 888):
        
        # Assigning a Attribute to a Subscript (line 888):
        # Getting the type of 'self' (line 888)
        self_36554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 19), 'self')
        # Obtaining the member 'max_step' of a type (line 888)
        max_step_36555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 888, 19), self_36554, 'max_step')
        # Getting the type of 'rwork' (line 888)
        rwork_36556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'rwork')
        int_36557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 14), 'int')
        # Storing an element on a container (line 888)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 8), rwork_36556, (int_36557, max_step_36555))
        
        # Assigning a Attribute to a Subscript (line 889):
        
        # Assigning a Attribute to a Subscript (line 889):
        # Getting the type of 'self' (line 889)
        self_36558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 19), 'self')
        # Obtaining the member 'min_step' of a type (line 889)
        min_step_36559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 889, 19), self_36558, 'min_step')
        # Getting the type of 'rwork' (line 889)
        rwork_36560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 8), 'rwork')
        int_36561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 14), 'int')
        # Storing an element on a container (line 889)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 889, 8), rwork_36560, (int_36561, min_step_36559))
        
        # Assigning a Name to a Attribute (line 890):
        
        # Assigning a Name to a Attribute (line 890):
        # Getting the type of 'rwork' (line 890)
        rwork_36562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 21), 'rwork')
        # Getting the type of 'self' (line 890)
        self_36563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 8), 'self')
        # Setting the type of the member 'rwork' of a type (line 890)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 890, 8), self_36563, 'rwork', rwork_36562)
        
        # Assigning a Call to a Name (line 892):
        
        # Assigning a Call to a Name (line 892):
        
        # Call to zeros(...): (line 892)
        # Processing the call arguments (line 892)
        
        # Obtaining an instance of the builtin type 'tuple' (line 892)
        tuple_36565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 892)
        # Adding element type (line 892)
        # Getting the type of 'liw' (line 892)
        liw_36566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 23), 'liw', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 892, 23), tuple_36565, liw_36566)
        
        # Getting the type of 'int32' (line 892)
        int32_36567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 30), 'int32', False)
        # Processing the call keyword arguments (line 892)
        kwargs_36568 = {}
        # Getting the type of 'zeros' (line 892)
        zeros_36564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 892)
        zeros_call_result_36569 = invoke(stypy.reporting.localization.Localization(__file__, 892, 16), zeros_36564, *[tuple_36565, int32_36567], **kwargs_36568)
        
        # Assigning a type to the variable 'iwork' (line 892)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 8), 'iwork', zeros_call_result_36569)
        
        
        # Getting the type of 'self' (line 893)
        self_36570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 11), 'self')
        # Obtaining the member 'ml' of a type (line 893)
        ml_36571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 893, 11), self_36570, 'ml')
        # Getting the type of 'None' (line 893)
        None_36572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 26), 'None')
        # Applying the binary operator 'isnot' (line 893)
        result_is_not_36573 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 11), 'isnot', ml_36571, None_36572)
        
        # Testing the type of an if condition (line 893)
        if_condition_36574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 893, 8), result_is_not_36573)
        # Assigning a type to the variable 'if_condition_36574' (line 893)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 893, 8), 'if_condition_36574', if_condition_36574)
        # SSA begins for if statement (line 893)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 894):
        
        # Assigning a Attribute to a Subscript (line 894):
        # Getting the type of 'self' (line 894)
        self_36575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 23), 'self')
        # Obtaining the member 'ml' of a type (line 894)
        ml_36576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 23), self_36575, 'ml')
        # Getting the type of 'iwork' (line 894)
        iwork_36577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 12), 'iwork')
        int_36578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 18), 'int')
        # Storing an element on a container (line 894)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 894, 12), iwork_36577, (int_36578, ml_36576))
        # SSA join for if statement (line 893)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 895)
        self_36579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 11), 'self')
        # Obtaining the member 'mu' of a type (line 895)
        mu_36580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 11), self_36579, 'mu')
        # Getting the type of 'None' (line 895)
        None_36581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 26), 'None')
        # Applying the binary operator 'isnot' (line 895)
        result_is_not_36582 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 11), 'isnot', mu_36580, None_36581)
        
        # Testing the type of an if condition (line 895)
        if_condition_36583 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 895, 8), result_is_not_36582)
        # Assigning a type to the variable 'if_condition_36583' (line 895)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 8), 'if_condition_36583', if_condition_36583)
        # SSA begins for if statement (line 895)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 896):
        
        # Assigning a Attribute to a Subscript (line 896):
        # Getting the type of 'self' (line 896)
        self_36584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 23), 'self')
        # Obtaining the member 'mu' of a type (line 896)
        mu_36585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 23), self_36584, 'mu')
        # Getting the type of 'iwork' (line 896)
        iwork_36586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 12), 'iwork')
        int_36587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 896, 18), 'int')
        # Storing an element on a container (line 896)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 896, 12), iwork_36586, (int_36587, mu_36585))
        # SSA join for if statement (line 895)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Subscript (line 897):
        
        # Assigning a Attribute to a Subscript (line 897):
        # Getting the type of 'self' (line 897)
        self_36588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 19), 'self')
        # Obtaining the member 'order' of a type (line 897)
        order_36589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 19), self_36588, 'order')
        # Getting the type of 'iwork' (line 897)
        iwork_36590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 8), 'iwork')
        int_36591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 14), 'int')
        # Storing an element on a container (line 897)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 8), iwork_36590, (int_36591, order_36589))
        
        # Assigning a Attribute to a Subscript (line 898):
        
        # Assigning a Attribute to a Subscript (line 898):
        # Getting the type of 'self' (line 898)
        self_36592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 19), 'self')
        # Obtaining the member 'nsteps' of a type (line 898)
        nsteps_36593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 898, 19), self_36592, 'nsteps')
        # Getting the type of 'iwork' (line 898)
        iwork_36594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 8), 'iwork')
        int_36595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 898, 14), 'int')
        # Storing an element on a container (line 898)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 898, 8), iwork_36594, (int_36595, nsteps_36593))
        
        # Assigning a Num to a Subscript (line 899):
        
        # Assigning a Num to a Subscript (line 899):
        int_36596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 19), 'int')
        # Getting the type of 'iwork' (line 899)
        iwork_36597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 8), 'iwork')
        int_36598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 14), 'int')
        # Storing an element on a container (line 899)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 899, 8), iwork_36597, (int_36598, int_36596))
        
        # Assigning a Name to a Attribute (line 900):
        
        # Assigning a Name to a Attribute (line 900):
        # Getting the type of 'iwork' (line 900)
        iwork_36599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 21), 'iwork')
        # Getting the type of 'self' (line 900)
        self_36600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 8), 'self')
        # Setting the type of the member 'iwork' of a type (line 900)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 8), self_36600, 'iwork', iwork_36599)
        
        # Assigning a List to a Attribute (line 902):
        
        # Assigning a List to a Attribute (line 902):
        
        # Obtaining an instance of the builtin type 'list' (line 902)
        list_36601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 902)
        # Adding element type (line 902)
        # Getting the type of 'self' (line 902)
        self_36602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 26), 'self')
        # Obtaining the member 'rtol' of a type (line 902)
        rtol_36603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 26), self_36602, 'rtol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 902, 25), list_36601, rtol_36603)
        # Adding element type (line 902)
        # Getting the type of 'self' (line 902)
        self_36604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 37), 'self')
        # Obtaining the member 'atol' of a type (line 902)
        atol_36605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 37), self_36604, 'atol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 902, 25), list_36601, atol_36605)
        # Adding element type (line 902)
        int_36606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 902, 25), list_36601, int_36606)
        # Adding element type (line 902)
        int_36607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 902, 25), list_36601, int_36607)
        # Adding element type (line 902)
        # Getting the type of 'self' (line 903)
        self_36608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 26), 'self')
        # Obtaining the member 'rwork' of a type (line 903)
        rwork_36609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 26), self_36608, 'rwork')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 902, 25), list_36601, rwork_36609)
        # Adding element type (line 902)
        # Getting the type of 'self' (line 903)
        self_36610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 38), 'self')
        # Obtaining the member 'iwork' of a type (line 903)
        iwork_36611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 38), self_36610, 'iwork')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 902, 25), list_36601, iwork_36611)
        # Adding element type (line 902)
        # Getting the type of 'mf' (line 903)
        mf_36612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 50), 'mf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 902, 25), list_36601, mf_36612)
        
        # Getting the type of 'self' (line 902)
        self_36613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 8), 'self')
        # Setting the type of the member 'call_args' of a type (line 902)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 8), self_36613, 'call_args', list_36601)
        
        # Assigning a Num to a Attribute (line 904):
        
        # Assigning a Num to a Attribute (line 904):
        int_36614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 904, 23), 'int')
        # Getting the type of 'self' (line 904)
        self_36615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 8), 'self')
        # Setting the type of the member 'success' of a type (line 904)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 904, 8), self_36615, 'success', int_36614)
        
        # Assigning a Name to a Attribute (line 905):
        
        # Assigning a Name to a Attribute (line 905):
        # Getting the type of 'False' (line 905)
        False_36616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 27), 'False')
        # Getting the type of 'self' (line 905)
        self_36617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 8), 'self')
        # Setting the type of the member 'initialized' of a type (line 905)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 8), self_36617, 'initialized', False_36616)
        
        # ################# End of 'reset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset' in the type store
        # Getting the type of 'stypy_return_type' (line 859)
        stypy_return_type_36618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36618)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset'
        return stypy_return_type_36618


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 907, 4, False)
        # Assigning a type to the variable 'self' (line 908)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 908, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vode.run.__dict__.__setitem__('stypy_localization', localization)
        vode.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vode.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        vode.run.__dict__.__setitem__('stypy_function_name', 'vode.run')
        vode.run.__dict__.__setitem__('stypy_param_names_list', ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'])
        vode.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        vode.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vode.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        vode.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        vode.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vode.run.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vode.run', ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        # Getting the type of 'self' (line 908)
        self_36619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 11), 'self')
        # Obtaining the member 'initialized' of a type (line 908)
        initialized_36620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 11), self_36619, 'initialized')
        # Testing the type of an if condition (line 908)
        if_condition_36621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 908, 8), initialized_36620)
        # Assigning a type to the variable 'if_condition_36621' (line 908)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 908, 8), 'if_condition_36621', if_condition_36621)
        # SSA begins for if statement (line 908)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_handle(...): (line 909)
        # Processing the call keyword arguments (line 909)
        kwargs_36624 = {}
        # Getting the type of 'self' (line 909)
        self_36622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 12), 'self', False)
        # Obtaining the member 'check_handle' of a type (line 909)
        check_handle_36623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 12), self_36622, 'check_handle')
        # Calling check_handle(args, kwargs) (line 909)
        check_handle_call_result_36625 = invoke(stypy.reporting.localization.Localization(__file__, 909, 12), check_handle_36623, *[], **kwargs_36624)
        
        # SSA branch for the else part of an if statement (line 908)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 911):
        
        # Assigning a Name to a Attribute (line 911):
        # Getting the type of 'True' (line 911)
        True_36626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 31), 'True')
        # Getting the type of 'self' (line 911)
        self_36627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 12), 'self')
        # Setting the type of the member 'initialized' of a type (line 911)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 911, 12), self_36627, 'initialized', True_36626)
        
        # Call to acquire_new_handle(...): (line 912)
        # Processing the call keyword arguments (line 912)
        kwargs_36630 = {}
        # Getting the type of 'self' (line 912)
        self_36628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 12), 'self', False)
        # Obtaining the member 'acquire_new_handle' of a type (line 912)
        acquire_new_handle_36629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 12), self_36628, 'acquire_new_handle')
        # Calling acquire_new_handle(args, kwargs) (line 912)
        acquire_new_handle_call_result_36631 = invoke(stypy.reporting.localization.Localization(__file__, 912, 12), acquire_new_handle_36629, *[], **kwargs_36630)
        
        # SSA join for if statement (line 908)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 914)
        self_36632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 11), 'self')
        # Obtaining the member 'ml' of a type (line 914)
        ml_36633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 914, 11), self_36632, 'ml')
        # Getting the type of 'None' (line 914)
        None_36634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 26), 'None')
        # Applying the binary operator 'isnot' (line 914)
        result_is_not_36635 = python_operator(stypy.reporting.localization.Localization(__file__, 914, 11), 'isnot', ml_36633, None_36634)
        
        
        # Getting the type of 'self' (line 914)
        self_36636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 35), 'self')
        # Obtaining the member 'ml' of a type (line 914)
        ml_36637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 914, 35), self_36636, 'ml')
        int_36638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, 45), 'int')
        # Applying the binary operator '>' (line 914)
        result_gt_36639 = python_operator(stypy.reporting.localization.Localization(__file__, 914, 35), '>', ml_36637, int_36638)
        
        # Applying the binary operator 'and' (line 914)
        result_and_keyword_36640 = python_operator(stypy.reporting.localization.Localization(__file__, 914, 11), 'and', result_is_not_36635, result_gt_36639)
        
        # Testing the type of an if condition (line 914)
        if_condition_36641 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 914, 8), result_and_keyword_36640)
        # Assigning a type to the variable 'if_condition_36641' (line 914)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 914, 8), 'if_condition_36641', if_condition_36641)
        # SSA begins for if statement (line 914)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 918):
        
        # Assigning a Call to a Name (line 918):
        
        # Call to _vode_banded_jac_wrapper(...): (line 918)
        # Processing the call arguments (line 918)
        # Getting the type of 'jac' (line 918)
        jac_36643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 43), 'jac', False)
        # Getting the type of 'self' (line 918)
        self_36644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 48), 'self', False)
        # Obtaining the member 'ml' of a type (line 918)
        ml_36645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 48), self_36644, 'ml')
        # Getting the type of 'jac_params' (line 918)
        jac_params_36646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 57), 'jac_params', False)
        # Processing the call keyword arguments (line 918)
        kwargs_36647 = {}
        # Getting the type of '_vode_banded_jac_wrapper' (line 918)
        _vode_banded_jac_wrapper_36642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 18), '_vode_banded_jac_wrapper', False)
        # Calling _vode_banded_jac_wrapper(args, kwargs) (line 918)
        _vode_banded_jac_wrapper_call_result_36648 = invoke(stypy.reporting.localization.Localization(__file__, 918, 18), _vode_banded_jac_wrapper_36642, *[jac_36643, ml_36645, jac_params_36646], **kwargs_36647)
        
        # Assigning a type to the variable 'jac' (line 918)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 12), 'jac', _vode_banded_jac_wrapper_call_result_36648)
        # SSA join for if statement (line 914)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 920):
        
        # Assigning a BinOp to a Name (line 920):
        
        # Obtaining an instance of the builtin type 'tuple' (line 920)
        tuple_36649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 920)
        # Adding element type (line 920)
        # Getting the type of 'f' (line 920)
        f_36650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 17), 'f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 920, 17), tuple_36649, f_36650)
        # Adding element type (line 920)
        # Getting the type of 'jac' (line 920)
        jac_36651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 20), 'jac')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 920, 17), tuple_36649, jac_36651)
        # Adding element type (line 920)
        # Getting the type of 'y0' (line 920)
        y0_36652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 25), 'y0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 920, 17), tuple_36649, y0_36652)
        # Adding element type (line 920)
        # Getting the type of 't0' (line 920)
        t0_36653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 29), 't0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 920, 17), tuple_36649, t0_36653)
        # Adding element type (line 920)
        # Getting the type of 't1' (line 920)
        t1_36654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 33), 't1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 920, 17), tuple_36649, t1_36654)
        
        
        # Call to tuple(...): (line 920)
        # Processing the call arguments (line 920)
        # Getting the type of 'self' (line 920)
        self_36656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 45), 'self', False)
        # Obtaining the member 'call_args' of a type (line 920)
        call_args_36657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 45), self_36656, 'call_args')
        # Processing the call keyword arguments (line 920)
        kwargs_36658 = {}
        # Getting the type of 'tuple' (line 920)
        tuple_36655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 39), 'tuple', False)
        # Calling tuple(args, kwargs) (line 920)
        tuple_call_result_36659 = invoke(stypy.reporting.localization.Localization(__file__, 920, 39), tuple_36655, *[call_args_36657], **kwargs_36658)
        
        # Applying the binary operator '+' (line 920)
        result_add_36660 = python_operator(stypy.reporting.localization.Localization(__file__, 920, 16), '+', tuple_36649, tuple_call_result_36659)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 921)
        tuple_36661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 921)
        # Adding element type (line 921)
        # Getting the type of 'f_params' (line 921)
        f_params_36662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 17), 'f_params')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 921, 17), tuple_36661, f_params_36662)
        # Adding element type (line 921)
        # Getting the type of 'jac_params' (line 921)
        jac_params_36663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 27), 'jac_params')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 921, 17), tuple_36661, jac_params_36663)
        
        # Applying the binary operator '+' (line 920)
        result_add_36664 = python_operator(stypy.reporting.localization.Localization(__file__, 920, 61), '+', result_add_36660, tuple_36661)
        
        # Assigning a type to the variable 'args' (line 920)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 8), 'args', result_add_36664)
        
        # Assigning a Call to a Tuple (line 922):
        
        # Assigning a Subscript to a Name (line 922):
        
        # Obtaining the type of the subscript
        int_36665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 8), 'int')
        
        # Call to runner(...): (line 922)
        # Getting the type of 'args' (line 922)
        args_36668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 37), 'args', False)
        # Processing the call keyword arguments (line 922)
        kwargs_36669 = {}
        # Getting the type of 'self' (line 922)
        self_36666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 24), 'self', False)
        # Obtaining the member 'runner' of a type (line 922)
        runner_36667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 24), self_36666, 'runner')
        # Calling runner(args, kwargs) (line 922)
        runner_call_result_36670 = invoke(stypy.reporting.localization.Localization(__file__, 922, 24), runner_36667, *[args_36668], **kwargs_36669)
        
        # Obtaining the member '__getitem__' of a type (line 922)
        getitem___36671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 8), runner_call_result_36670, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 922)
        subscript_call_result_36672 = invoke(stypy.reporting.localization.Localization(__file__, 922, 8), getitem___36671, int_36665)
        
        # Assigning a type to the variable 'tuple_var_assignment_35475' (line 922)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'tuple_var_assignment_35475', subscript_call_result_36672)
        
        # Assigning a Subscript to a Name (line 922):
        
        # Obtaining the type of the subscript
        int_36673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 8), 'int')
        
        # Call to runner(...): (line 922)
        # Getting the type of 'args' (line 922)
        args_36676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 37), 'args', False)
        # Processing the call keyword arguments (line 922)
        kwargs_36677 = {}
        # Getting the type of 'self' (line 922)
        self_36674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 24), 'self', False)
        # Obtaining the member 'runner' of a type (line 922)
        runner_36675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 24), self_36674, 'runner')
        # Calling runner(args, kwargs) (line 922)
        runner_call_result_36678 = invoke(stypy.reporting.localization.Localization(__file__, 922, 24), runner_36675, *[args_36676], **kwargs_36677)
        
        # Obtaining the member '__getitem__' of a type (line 922)
        getitem___36679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 8), runner_call_result_36678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 922)
        subscript_call_result_36680 = invoke(stypy.reporting.localization.Localization(__file__, 922, 8), getitem___36679, int_36673)
        
        # Assigning a type to the variable 'tuple_var_assignment_35476' (line 922)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'tuple_var_assignment_35476', subscript_call_result_36680)
        
        # Assigning a Subscript to a Name (line 922):
        
        # Obtaining the type of the subscript
        int_36681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 8), 'int')
        
        # Call to runner(...): (line 922)
        # Getting the type of 'args' (line 922)
        args_36684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 37), 'args', False)
        # Processing the call keyword arguments (line 922)
        kwargs_36685 = {}
        # Getting the type of 'self' (line 922)
        self_36682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 24), 'self', False)
        # Obtaining the member 'runner' of a type (line 922)
        runner_36683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 24), self_36682, 'runner')
        # Calling runner(args, kwargs) (line 922)
        runner_call_result_36686 = invoke(stypy.reporting.localization.Localization(__file__, 922, 24), runner_36683, *[args_36684], **kwargs_36685)
        
        # Obtaining the member '__getitem__' of a type (line 922)
        getitem___36687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 8), runner_call_result_36686, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 922)
        subscript_call_result_36688 = invoke(stypy.reporting.localization.Localization(__file__, 922, 8), getitem___36687, int_36681)
        
        # Assigning a type to the variable 'tuple_var_assignment_35477' (line 922)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'tuple_var_assignment_35477', subscript_call_result_36688)
        
        # Assigning a Name to a Name (line 922):
        # Getting the type of 'tuple_var_assignment_35475' (line 922)
        tuple_var_assignment_35475_36689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'tuple_var_assignment_35475')
        # Assigning a type to the variable 'y1' (line 922)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'y1', tuple_var_assignment_35475_36689)
        
        # Assigning a Name to a Name (line 922):
        # Getting the type of 'tuple_var_assignment_35476' (line 922)
        tuple_var_assignment_35476_36690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'tuple_var_assignment_35476')
        # Assigning a type to the variable 't' (line 922)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 12), 't', tuple_var_assignment_35476_36690)
        
        # Assigning a Name to a Name (line 922):
        # Getting the type of 'tuple_var_assignment_35477' (line 922)
        tuple_var_assignment_35477_36691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'tuple_var_assignment_35477')
        # Assigning a type to the variable 'istate' (line 922)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 15), 'istate', tuple_var_assignment_35477_36691)
        
        # Assigning a Name to a Attribute (line 923):
        
        # Assigning a Name to a Attribute (line 923):
        # Getting the type of 'istate' (line 923)
        istate_36692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 22), 'istate')
        # Getting the type of 'self' (line 923)
        self_36693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 8), 'self')
        # Setting the type of the member 'istate' of a type (line 923)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 8), self_36693, 'istate', istate_36692)
        
        
        # Getting the type of 'istate' (line 924)
        istate_36694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 11), 'istate')
        int_36695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 20), 'int')
        # Applying the binary operator '<' (line 924)
        result_lt_36696 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 11), '<', istate_36694, int_36695)
        
        # Testing the type of an if condition (line 924)
        if_condition_36697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 924, 8), result_lt_36696)
        # Assigning a type to the variable 'if_condition_36697' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 8), 'if_condition_36697', if_condition_36697)
        # SSA begins for if statement (line 924)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 925):
        
        # Assigning a Call to a Name (line 925):
        
        # Call to format(...): (line 925)
        # Processing the call arguments (line 925)
        # Getting the type of 'istate' (line 925)
        istate_36700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 68), 'istate', False)
        # Processing the call keyword arguments (line 925)
        kwargs_36701 = {}
        str_36698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 36), 'str', 'Unexpected istate={:d}')
        # Obtaining the member 'format' of a type (line 925)
        format_36699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 36), str_36698, 'format')
        # Calling format(args, kwargs) (line 925)
        format_call_result_36702 = invoke(stypy.reporting.localization.Localization(__file__, 925, 36), format_36699, *[istate_36700], **kwargs_36701)
        
        # Assigning a type to the variable 'unexpected_istate_msg' (line 925)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 12), 'unexpected_istate_msg', format_call_result_36702)
        
        # Call to warn(...): (line 926)
        # Processing the call arguments (line 926)
        
        # Call to format(...): (line 926)
        # Processing the call arguments (line 926)
        # Getting the type of 'self' (line 926)
        self_36707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 46), 'self', False)
        # Obtaining the member '__class__' of a type (line 926)
        class___36708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 46), self_36707, '__class__')
        # Obtaining the member '__name__' of a type (line 926)
        name___36709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 46), class___36708, '__name__')
        
        # Call to get(...): (line 927)
        # Processing the call arguments (line 927)
        # Getting the type of 'istate' (line 927)
        istate_36713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 44), 'istate', False)
        # Getting the type of 'unexpected_istate_msg' (line 927)
        unexpected_istate_msg_36714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 52), 'unexpected_istate_msg', False)
        # Processing the call keyword arguments (line 927)
        kwargs_36715 = {}
        # Getting the type of 'self' (line 927)
        self_36710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 26), 'self', False)
        # Obtaining the member 'messages' of a type (line 927)
        messages_36711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 26), self_36710, 'messages')
        # Obtaining the member 'get' of a type (line 927)
        get_36712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 26), messages_36711, 'get')
        # Calling get(args, kwargs) (line 927)
        get_call_result_36716 = invoke(stypy.reporting.localization.Localization(__file__, 927, 26), get_36712, *[istate_36713, unexpected_istate_msg_36714], **kwargs_36715)
        
        # Processing the call keyword arguments (line 926)
        kwargs_36717 = {}
        str_36705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 26), 'str', '{:s}: {:s}')
        # Obtaining the member 'format' of a type (line 926)
        format_36706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 26), str_36705, 'format')
        # Calling format(args, kwargs) (line 926)
        format_call_result_36718 = invoke(stypy.reporting.localization.Localization(__file__, 926, 26), format_36706, *[name___36709, get_call_result_36716], **kwargs_36717)
        
        # Processing the call keyword arguments (line 926)
        kwargs_36719 = {}
        # Getting the type of 'warnings' (line 926)
        warnings_36703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 926)
        warn_36704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 12), warnings_36703, 'warn')
        # Calling warn(args, kwargs) (line 926)
        warn_call_result_36720 = invoke(stypy.reporting.localization.Localization(__file__, 926, 12), warn_36704, *[format_call_result_36718], **kwargs_36719)
        
        
        # Assigning a Num to a Attribute (line 928):
        
        # Assigning a Num to a Attribute (line 928):
        int_36721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 928, 27), 'int')
        # Getting the type of 'self' (line 928)
        self_36722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 12), 'self')
        # Setting the type of the member 'success' of a type (line 928)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 12), self_36722, 'success', int_36721)
        # SSA branch for the else part of an if statement (line 924)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Subscript (line 930):
        
        # Assigning a Num to a Subscript (line 930):
        int_36723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 930, 32), 'int')
        # Getting the type of 'self' (line 930)
        self_36724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 12), 'self')
        # Obtaining the member 'call_args' of a type (line 930)
        call_args_36725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 12), self_36724, 'call_args')
        int_36726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 930, 27), 'int')
        # Storing an element on a container (line 930)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 930, 12), call_args_36725, (int_36726, int_36723))
        
        # Assigning a Num to a Attribute (line 931):
        
        # Assigning a Num to a Attribute (line 931):
        int_36727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 26), 'int')
        # Getting the type of 'self' (line 931)
        self_36728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 12), 'self')
        # Setting the type of the member 'istate' of a type (line 931)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 12), self_36728, 'istate', int_36727)
        # SSA join for if statement (line 924)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 932)
        tuple_36729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 932)
        # Adding element type (line 932)
        # Getting the type of 'y1' (line 932)
        y1_36730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 15), 'y1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 932, 15), tuple_36729, y1_36730)
        # Adding element type (line 932)
        # Getting the type of 't' (line 932)
        t_36731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 19), 't')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 932, 15), tuple_36729, t_36731)
        
        # Assigning a type to the variable 'stypy_return_type' (line 932)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 932, 8), 'stypy_return_type', tuple_36729)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 907)
        stypy_return_type_36732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36732)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_36732


    @norecursion
    def step(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'step'
        module_type_store = module_type_store.open_function_context('step', 934, 4, False)
        # Assigning a type to the variable 'self' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vode.step.__dict__.__setitem__('stypy_localization', localization)
        vode.step.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vode.step.__dict__.__setitem__('stypy_type_store', module_type_store)
        vode.step.__dict__.__setitem__('stypy_function_name', 'vode.step')
        vode.step.__dict__.__setitem__('stypy_param_names_list', [])
        vode.step.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        vode.step.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vode.step.__dict__.__setitem__('stypy_call_defaults', defaults)
        vode.step.__dict__.__setitem__('stypy_call_varargs', varargs)
        vode.step.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vode.step.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vode.step', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'step', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'step(...)' code ##################

        
        # Assigning a Subscript to a Name (line 935):
        
        # Assigning a Subscript to a Name (line 935):
        
        # Obtaining the type of the subscript
        int_36733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 31), 'int')
        # Getting the type of 'self' (line 935)
        self_36734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 16), 'self')
        # Obtaining the member 'call_args' of a type (line 935)
        call_args_36735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 16), self_36734, 'call_args')
        # Obtaining the member '__getitem__' of a type (line 935)
        getitem___36736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 16), call_args_36735, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 935)
        subscript_call_result_36737 = invoke(stypy.reporting.localization.Localization(__file__, 935, 16), getitem___36736, int_36733)
        
        # Assigning a type to the variable 'itask' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'itask', subscript_call_result_36737)
        
        # Assigning a Num to a Subscript (line 936):
        
        # Assigning a Num to a Subscript (line 936):
        int_36738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 28), 'int')
        # Getting the type of 'self' (line 936)
        self_36739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 8), 'self')
        # Obtaining the member 'call_args' of a type (line 936)
        call_args_36740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 936, 8), self_36739, 'call_args')
        int_36741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 23), 'int')
        # Storing an element on a container (line 936)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 936, 8), call_args_36740, (int_36741, int_36738))
        
        # Assigning a Call to a Name (line 937):
        
        # Assigning a Call to a Name (line 937):
        
        # Call to run(...): (line 937)
        # Getting the type of 'args' (line 937)
        args_36744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 22), 'args', False)
        # Processing the call keyword arguments (line 937)
        kwargs_36745 = {}
        # Getting the type of 'self' (line 937)
        self_36742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 12), 'self', False)
        # Obtaining the member 'run' of a type (line 937)
        run_36743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 937, 12), self_36742, 'run')
        # Calling run(args, kwargs) (line 937)
        run_call_result_36746 = invoke(stypy.reporting.localization.Localization(__file__, 937, 12), run_36743, *[args_36744], **kwargs_36745)
        
        # Assigning a type to the variable 'r' (line 937)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 8), 'r', run_call_result_36746)
        
        # Assigning a Name to a Subscript (line 938):
        
        # Assigning a Name to a Subscript (line 938):
        # Getting the type of 'itask' (line 938)
        itask_36747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 28), 'itask')
        # Getting the type of 'self' (line 938)
        self_36748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 8), 'self')
        # Obtaining the member 'call_args' of a type (line 938)
        call_args_36749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 938, 8), self_36748, 'call_args')
        int_36750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 23), 'int')
        # Storing an element on a container (line 938)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 938, 8), call_args_36749, (int_36750, itask_36747))
        # Getting the type of 'r' (line 939)
        r_36751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 15), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 939)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 8), 'stypy_return_type', r_36751)
        
        # ################# End of 'step(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'step' in the type store
        # Getting the type of 'stypy_return_type' (line 934)
        stypy_return_type_36752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36752)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'step'
        return stypy_return_type_36752


    @norecursion
    def run_relax(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run_relax'
        module_type_store = module_type_store.open_function_context('run_relax', 941, 4, False)
        # Assigning a type to the variable 'self' (line 942)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vode.run_relax.__dict__.__setitem__('stypy_localization', localization)
        vode.run_relax.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vode.run_relax.__dict__.__setitem__('stypy_type_store', module_type_store)
        vode.run_relax.__dict__.__setitem__('stypy_function_name', 'vode.run_relax')
        vode.run_relax.__dict__.__setitem__('stypy_param_names_list', [])
        vode.run_relax.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        vode.run_relax.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vode.run_relax.__dict__.__setitem__('stypy_call_defaults', defaults)
        vode.run_relax.__dict__.__setitem__('stypy_call_varargs', varargs)
        vode.run_relax.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vode.run_relax.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vode.run_relax', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run_relax', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run_relax(...)' code ##################

        
        # Assigning a Subscript to a Name (line 942):
        
        # Assigning a Subscript to a Name (line 942):
        
        # Obtaining the type of the subscript
        int_36753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 31), 'int')
        # Getting the type of 'self' (line 942)
        self_36754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 16), 'self')
        # Obtaining the member 'call_args' of a type (line 942)
        call_args_36755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 16), self_36754, 'call_args')
        # Obtaining the member '__getitem__' of a type (line 942)
        getitem___36756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 16), call_args_36755, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 942)
        subscript_call_result_36757 = invoke(stypy.reporting.localization.Localization(__file__, 942, 16), getitem___36756, int_36753)
        
        # Assigning a type to the variable 'itask' (line 942)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 8), 'itask', subscript_call_result_36757)
        
        # Assigning a Num to a Subscript (line 943):
        
        # Assigning a Num to a Subscript (line 943):
        int_36758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 28), 'int')
        # Getting the type of 'self' (line 943)
        self_36759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 8), 'self')
        # Obtaining the member 'call_args' of a type (line 943)
        call_args_36760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 943, 8), self_36759, 'call_args')
        int_36761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 23), 'int')
        # Storing an element on a container (line 943)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 943, 8), call_args_36760, (int_36761, int_36758))
        
        # Assigning a Call to a Name (line 944):
        
        # Assigning a Call to a Name (line 944):
        
        # Call to run(...): (line 944)
        # Getting the type of 'args' (line 944)
        args_36764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 22), 'args', False)
        # Processing the call keyword arguments (line 944)
        kwargs_36765 = {}
        # Getting the type of 'self' (line 944)
        self_36762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 12), 'self', False)
        # Obtaining the member 'run' of a type (line 944)
        run_36763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 944, 12), self_36762, 'run')
        # Calling run(args, kwargs) (line 944)
        run_call_result_36766 = invoke(stypy.reporting.localization.Localization(__file__, 944, 12), run_36763, *[args_36764], **kwargs_36765)
        
        # Assigning a type to the variable 'r' (line 944)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'r', run_call_result_36766)
        
        # Assigning a Name to a Subscript (line 945):
        
        # Assigning a Name to a Subscript (line 945):
        # Getting the type of 'itask' (line 945)
        itask_36767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 28), 'itask')
        # Getting the type of 'self' (line 945)
        self_36768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 8), 'self')
        # Obtaining the member 'call_args' of a type (line 945)
        call_args_36769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 945, 8), self_36768, 'call_args')
        int_36770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, 23), 'int')
        # Storing an element on a container (line 945)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 945, 8), call_args_36769, (int_36770, itask_36767))
        # Getting the type of 'r' (line 946)
        r_36771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 15), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 946)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 8), 'stypy_return_type', r_36771)
        
        # ################# End of 'run_relax(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run_relax' in the type store
        # Getting the type of 'stypy_return_type' (line 941)
        stypy_return_type_36772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36772)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run_relax'
        return stypy_return_type_36772


# Assigning a type to the variable 'vode' (line 752)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 0), 'vode', vode)

# Assigning a Call to a Name (line 753):

# Call to getattr(...): (line 753)
# Processing the call arguments (line 753)
# Getting the type of '_vode' (line 753)
_vode_36774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 21), '_vode', False)
str_36775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 28), 'str', 'dvode')
# Getting the type of 'None' (line 753)
None_36776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 37), 'None', False)
# Processing the call keyword arguments (line 753)
kwargs_36777 = {}
# Getting the type of 'getattr' (line 753)
getattr_36773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 13), 'getattr', False)
# Calling getattr(args, kwargs) (line 753)
getattr_call_result_36778 = invoke(stypy.reporting.localization.Localization(__file__, 753, 13), getattr_36773, *[_vode_36774, str_36775, None_36776], **kwargs_36777)

# Getting the type of 'vode'
vode_36779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'vode')
# Setting the type of the member 'runner' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), vode_36779, 'runner', getattr_call_result_36778)

# Assigning a Dict to a Name (line 755):

# Obtaining an instance of the builtin type 'dict' (line 755)
dict_36780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 755)
# Adding element type (key, value) (line 755)
int_36781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 16), 'int')
str_36782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 20), 'str', 'Excess work done on this call. (Perhaps wrong MF.)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 15), dict_36780, (int_36781, str_36782))
# Adding element type (key, value) (line 755)
int_36783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 16), 'int')
str_36784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 20), 'str', 'Excess accuracy requested. (Tolerances too small.)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 15), dict_36780, (int_36783, str_36784))
# Adding element type (key, value) (line 755)
int_36785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 16), 'int')
str_36786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 20), 'str', 'Illegal input detected. (See printed message.)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 15), dict_36780, (int_36785, str_36786))
# Adding element type (key, value) (line 755)
int_36787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 16), 'int')
str_36788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 20), 'str', 'Repeated error test failures. (Check all input.)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 15), dict_36780, (int_36787, str_36788))
# Adding element type (key, value) (line 755)
int_36789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 16), 'int')
str_36790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 20), 'str', 'Repeated convergence failures. (Perhaps bad Jacobian supplied or wrong choice of MF or tolerances.)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 15), dict_36780, (int_36789, str_36790))
# Adding element type (key, value) (line 755)
int_36791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 16), 'int')
str_36792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 20), 'str', 'Error weight became zero during problem. (Solution component i vanished, and ATOL or ATOL(i) = 0.)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 15), dict_36780, (int_36791, str_36792))

# Getting the type of 'vode'
vode_36793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'vode')
# Setting the type of the member 'messages' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), vode_36793, 'messages', dict_36780)

# Assigning a Num to a Name (line 764):
int_36794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 25), 'int')
# Getting the type of 'vode'
vode_36795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'vode')
# Setting the type of the member 'supports_run_relax' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), vode_36795, 'supports_run_relax', int_36794)

# Assigning a Num to a Name (line 765):
int_36796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 20), 'int')
# Getting the type of 'vode'
vode_36797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'vode')
# Setting the type of the member 'supports_step' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), vode_36797, 'supports_step', int_36796)

# Assigning a Num to a Name (line 766):
int_36798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 27), 'int')
# Getting the type of 'vode'
vode_36799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'vode')
# Setting the type of the member 'active_global_handle' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), vode_36799, 'active_global_handle', int_36798)


# Getting the type of 'vode' (line 949)
vode_36800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 3), 'vode')
# Obtaining the member 'runner' of a type (line 949)
runner_36801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 949, 3), vode_36800, 'runner')
# Getting the type of 'None' (line 949)
None_36802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 22), 'None')
# Applying the binary operator 'isnot' (line 949)
result_is_not_36803 = python_operator(stypy.reporting.localization.Localization(__file__, 949, 3), 'isnot', runner_36801, None_36802)

# Testing the type of an if condition (line 949)
if_condition_36804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 949, 0), result_is_not_36803)
# Assigning a type to the variable 'if_condition_36804' (line 949)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 949, 0), 'if_condition_36804', if_condition_36804)
# SSA begins for if statement (line 949)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to append(...): (line 950)
# Processing the call arguments (line 950)
# Getting the type of 'vode' (line 950)
vode_36808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 45), 'vode', False)
# Processing the call keyword arguments (line 950)
kwargs_36809 = {}
# Getting the type of 'IntegratorBase' (line 950)
IntegratorBase_36805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 4), 'IntegratorBase', False)
# Obtaining the member 'integrator_classes' of a type (line 950)
integrator_classes_36806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 950, 4), IntegratorBase_36805, 'integrator_classes')
# Obtaining the member 'append' of a type (line 950)
append_36807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 950, 4), integrator_classes_36806, 'append')
# Calling append(args, kwargs) (line 950)
append_call_result_36810 = invoke(stypy.reporting.localization.Localization(__file__, 950, 4), append_36807, *[vode_36808], **kwargs_36809)

# SSA join for if statement (line 949)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'zvode' class
# Getting the type of 'vode' (line 953)
vode_36811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 12), 'vode')

class zvode(vode_36811, ):
    
    # Assigning a Call to a Name (line 954):
    
    # Assigning a Num to a Name (line 956):
    
    # Assigning a Num to a Name (line 957):
    
    # Assigning a Name to a Name (line 958):
    
    # Assigning a Num to a Name (line 959):

    @norecursion
    def reset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reset'
        module_type_store = module_type_store.open_function_context('reset', 961, 4, False)
        # Assigning a type to the variable 'self' (line 962)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 962, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        zvode.reset.__dict__.__setitem__('stypy_localization', localization)
        zvode.reset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        zvode.reset.__dict__.__setitem__('stypy_type_store', module_type_store)
        zvode.reset.__dict__.__setitem__('stypy_function_name', 'zvode.reset')
        zvode.reset.__dict__.__setitem__('stypy_param_names_list', ['n', 'has_jac'])
        zvode.reset.__dict__.__setitem__('stypy_varargs_param_name', None)
        zvode.reset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        zvode.reset.__dict__.__setitem__('stypy_call_defaults', defaults)
        zvode.reset.__dict__.__setitem__('stypy_call_varargs', varargs)
        zvode.reset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        zvode.reset.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'zvode.reset', ['n', 'has_jac'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reset', localization, ['n', 'has_jac'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reset(...)' code ##################

        
        # Assigning a Call to a Name (line 962):
        
        # Assigning a Call to a Name (line 962):
        
        # Call to _determine_mf_and_set_bands(...): (line 962)
        # Processing the call arguments (line 962)
        # Getting the type of 'has_jac' (line 962)
        has_jac_36814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 46), 'has_jac', False)
        # Processing the call keyword arguments (line 962)
        kwargs_36815 = {}
        # Getting the type of 'self' (line 962)
        self_36812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 13), 'self', False)
        # Obtaining the member '_determine_mf_and_set_bands' of a type (line 962)
        _determine_mf_and_set_bands_36813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 13), self_36812, '_determine_mf_and_set_bands')
        # Calling _determine_mf_and_set_bands(args, kwargs) (line 962)
        _determine_mf_and_set_bands_call_result_36816 = invoke(stypy.reporting.localization.Localization(__file__, 962, 13), _determine_mf_and_set_bands_36813, *[has_jac_36814], **kwargs_36815)
        
        # Assigning a type to the variable 'mf' (line 962)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 962, 8), 'mf', _determine_mf_and_set_bands_call_result_36816)
        
        
        # Getting the type of 'mf' (line 964)
        mf_36817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 11), 'mf')
        
        # Obtaining an instance of the builtin type 'tuple' (line 964)
        tuple_36818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 964)
        # Adding element type (line 964)
        int_36819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 18), tuple_36818, int_36819)
        
        # Applying the binary operator 'in' (line 964)
        result_contains_36820 = python_operator(stypy.reporting.localization.Localization(__file__, 964, 11), 'in', mf_36817, tuple_36818)
        
        # Testing the type of an if condition (line 964)
        if_condition_36821 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 964, 8), result_contains_36820)
        # Assigning a type to the variable 'if_condition_36821' (line 964)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 8), 'if_condition_36821', if_condition_36821)
        # SSA begins for if statement (line 964)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 965):
        
        # Assigning a BinOp to a Name (line 965):
        int_36822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 18), 'int')
        # Getting the type of 'n' (line 965)
        n_36823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 23), 'n')
        # Applying the binary operator '*' (line 965)
        result_mul_36824 = python_operator(stypy.reporting.localization.Localization(__file__, 965, 18), '*', int_36822, n_36823)
        
        # Assigning a type to the variable 'lzw' (line 965)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 12), 'lzw', result_mul_36824)
        # SSA branch for the else part of an if statement (line 964)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 966)
        mf_36825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'tuple' (line 966)
        tuple_36826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 966)
        # Adding element type (line 966)
        int_36827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 966, 20), tuple_36826, int_36827)
        # Adding element type (line 966)
        int_36828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 966, 20), tuple_36826, int_36828)
        
        # Applying the binary operator 'in' (line 966)
        result_contains_36829 = python_operator(stypy.reporting.localization.Localization(__file__, 966, 13), 'in', mf_36825, tuple_36826)
        
        # Testing the type of an if condition (line 966)
        if_condition_36830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 966, 13), result_contains_36829)
        # Assigning a type to the variable 'if_condition_36830' (line 966)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 13), 'if_condition_36830', if_condition_36830)
        # SSA begins for if statement (line 966)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 967):
        
        # Assigning a BinOp to a Name (line 967):
        int_36831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 18), 'int')
        # Getting the type of 'n' (line 967)
        n_36832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 23), 'n')
        # Applying the binary operator '*' (line 967)
        result_mul_36833 = python_operator(stypy.reporting.localization.Localization(__file__, 967, 18), '*', int_36831, n_36832)
        
        int_36834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 27), 'int')
        # Getting the type of 'n' (line 967)
        n_36835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 31), 'n')
        int_36836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 36), 'int')
        # Applying the binary operator '**' (line 967)
        result_pow_36837 = python_operator(stypy.reporting.localization.Localization(__file__, 967, 31), '**', n_36835, int_36836)
        
        # Applying the binary operator '*' (line 967)
        result_mul_36838 = python_operator(stypy.reporting.localization.Localization(__file__, 967, 27), '*', int_36834, result_pow_36837)
        
        # Applying the binary operator '+' (line 967)
        result_add_36839 = python_operator(stypy.reporting.localization.Localization(__file__, 967, 18), '+', result_mul_36833, result_mul_36838)
        
        # Assigning a type to the variable 'lzw' (line 967)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 12), 'lzw', result_add_36839)
        # SSA branch for the else part of an if statement (line 966)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 968)
        mf_36840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'tuple' (line 968)
        tuple_36841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 968, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 968)
        # Adding element type (line 968)
        int_36842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 968, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 968, 20), tuple_36841, int_36842)
        # Adding element type (line 968)
        int_36843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 968, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 968, 20), tuple_36841, int_36843)
        
        # Applying the binary operator 'in' (line 968)
        result_contains_36844 = python_operator(stypy.reporting.localization.Localization(__file__, 968, 13), 'in', mf_36840, tuple_36841)
        
        # Testing the type of an if condition (line 968)
        if_condition_36845 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 968, 13), result_contains_36844)
        # Assigning a type to the variable 'if_condition_36845' (line 968)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 968, 13), 'if_condition_36845', if_condition_36845)
        # SSA begins for if statement (line 968)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 969):
        
        # Assigning a BinOp to a Name (line 969):
        int_36846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 969, 18), 'int')
        # Getting the type of 'n' (line 969)
        n_36847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 23), 'n')
        # Applying the binary operator '*' (line 969)
        result_mul_36848 = python_operator(stypy.reporting.localization.Localization(__file__, 969, 18), '*', int_36846, n_36847)
        
        # Getting the type of 'n' (line 969)
        n_36849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 27), 'n')
        int_36850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 969, 32), 'int')
        # Applying the binary operator '**' (line 969)
        result_pow_36851 = python_operator(stypy.reporting.localization.Localization(__file__, 969, 27), '**', n_36849, int_36850)
        
        # Applying the binary operator '+' (line 969)
        result_add_36852 = python_operator(stypy.reporting.localization.Localization(__file__, 969, 18), '+', result_mul_36848, result_pow_36851)
        
        # Assigning a type to the variable 'lzw' (line 969)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 12), 'lzw', result_add_36852)
        # SSA branch for the else part of an if statement (line 968)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 970)
        mf_36853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'tuple' (line 970)
        tuple_36854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 970)
        # Adding element type (line 970)
        int_36855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 970, 20), tuple_36854, int_36855)
        
        # Applying the binary operator 'in' (line 970)
        result_contains_36856 = python_operator(stypy.reporting.localization.Localization(__file__, 970, 13), 'in', mf_36853, tuple_36854)
        
        # Testing the type of an if condition (line 970)
        if_condition_36857 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 970, 13), result_contains_36856)
        # Assigning a type to the variable 'if_condition_36857' (line 970)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 13), 'if_condition_36857', if_condition_36857)
        # SSA begins for if statement (line 970)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 971):
        
        # Assigning a BinOp to a Name (line 971):
        int_36858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 18), 'int')
        # Getting the type of 'n' (line 971)
        n_36859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 23), 'n')
        # Applying the binary operator '*' (line 971)
        result_mul_36860 = python_operator(stypy.reporting.localization.Localization(__file__, 971, 18), '*', int_36858, n_36859)
        
        # Assigning a type to the variable 'lzw' (line 971)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 971, 12), 'lzw', result_mul_36860)
        # SSA branch for the else part of an if statement (line 970)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 972)
        mf_36861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'tuple' (line 972)
        tuple_36862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 972)
        # Adding element type (line 972)
        int_36863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 972, 20), tuple_36862, int_36863)
        # Adding element type (line 972)
        int_36864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 972, 20), tuple_36862, int_36864)
        
        # Applying the binary operator 'in' (line 972)
        result_contains_36865 = python_operator(stypy.reporting.localization.Localization(__file__, 972, 13), 'in', mf_36861, tuple_36862)
        
        # Testing the type of an if condition (line 972)
        if_condition_36866 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 972, 13), result_contains_36865)
        # Assigning a type to the variable 'if_condition_36866' (line 972)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 13), 'if_condition_36866', if_condition_36866)
        # SSA begins for if statement (line 972)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 973):
        
        # Assigning a BinOp to a Name (line 973):
        int_36867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 18), 'int')
        # Getting the type of 'n' (line 973)
        n_36868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 23), 'n')
        # Applying the binary operator '*' (line 973)
        result_mul_36869 = python_operator(stypy.reporting.localization.Localization(__file__, 973, 18), '*', int_36867, n_36868)
        
        int_36870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 28), 'int')
        # Getting the type of 'self' (line 973)
        self_36871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 32), 'self')
        # Obtaining the member 'ml' of a type (line 973)
        ml_36872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 32), self_36871, 'ml')
        # Applying the binary operator '*' (line 973)
        result_mul_36873 = python_operator(stypy.reporting.localization.Localization(__file__, 973, 28), '*', int_36870, ml_36872)
        
        int_36874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 42), 'int')
        # Getting the type of 'self' (line 973)
        self_36875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 46), 'self')
        # Obtaining the member 'mu' of a type (line 973)
        mu_36876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 46), self_36875, 'mu')
        # Applying the binary operator '*' (line 973)
        result_mul_36877 = python_operator(stypy.reporting.localization.Localization(__file__, 973, 42), '*', int_36874, mu_36876)
        
        # Applying the binary operator '+' (line 973)
        result_add_36878 = python_operator(stypy.reporting.localization.Localization(__file__, 973, 28), '+', result_mul_36873, result_mul_36877)
        
        # Getting the type of 'n' (line 973)
        n_36879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 57), 'n')
        # Applying the binary operator '*' (line 973)
        result_mul_36880 = python_operator(stypy.reporting.localization.Localization(__file__, 973, 27), '*', result_add_36878, n_36879)
        
        # Applying the binary operator '+' (line 973)
        result_add_36881 = python_operator(stypy.reporting.localization.Localization(__file__, 973, 18), '+', result_mul_36869, result_mul_36880)
        
        # Assigning a type to the variable 'lzw' (line 973)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 973, 12), 'lzw', result_add_36881)
        # SSA branch for the else part of an if statement (line 972)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 974)
        mf_36882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'tuple' (line 974)
        tuple_36883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 974)
        # Adding element type (line 974)
        int_36884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 974, 20), tuple_36883, int_36884)
        # Adding element type (line 974)
        int_36885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 974, 20), tuple_36883, int_36885)
        
        # Applying the binary operator 'in' (line 974)
        result_contains_36886 = python_operator(stypy.reporting.localization.Localization(__file__, 974, 13), 'in', mf_36882, tuple_36883)
        
        # Testing the type of an if condition (line 974)
        if_condition_36887 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 974, 13), result_contains_36886)
        # Assigning a type to the variable 'if_condition_36887' (line 974)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 13), 'if_condition_36887', if_condition_36887)
        # SSA begins for if statement (line 974)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 975):
        
        # Assigning a BinOp to a Name (line 975):
        int_36888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 18), 'int')
        # Getting the type of 'n' (line 975)
        n_36889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 23), 'n')
        # Applying the binary operator '*' (line 975)
        result_mul_36890 = python_operator(stypy.reporting.localization.Localization(__file__, 975, 18), '*', int_36888, n_36889)
        
        int_36891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 28), 'int')
        # Getting the type of 'self' (line 975)
        self_36892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 32), 'self')
        # Obtaining the member 'ml' of a type (line 975)
        ml_36893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 975, 32), self_36892, 'ml')
        # Applying the binary operator '*' (line 975)
        result_mul_36894 = python_operator(stypy.reporting.localization.Localization(__file__, 975, 28), '*', int_36891, ml_36893)
        
        # Getting the type of 'self' (line 975)
        self_36895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 42), 'self')
        # Obtaining the member 'mu' of a type (line 975)
        mu_36896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 975, 42), self_36895, 'mu')
        # Applying the binary operator '+' (line 975)
        result_add_36897 = python_operator(stypy.reporting.localization.Localization(__file__, 975, 28), '+', result_mul_36894, mu_36896)
        
        # Getting the type of 'n' (line 975)
        n_36898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 53), 'n')
        # Applying the binary operator '*' (line 975)
        result_mul_36899 = python_operator(stypy.reporting.localization.Localization(__file__, 975, 27), '*', result_add_36897, n_36898)
        
        # Applying the binary operator '+' (line 975)
        result_add_36900 = python_operator(stypy.reporting.localization.Localization(__file__, 975, 18), '+', result_mul_36890, result_mul_36899)
        
        # Assigning a type to the variable 'lzw' (line 975)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 12), 'lzw', result_add_36900)
        # SSA branch for the else part of an if statement (line 974)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 976)
        mf_36901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'tuple' (line 976)
        tuple_36902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 976)
        # Adding element type (line 976)
        int_36903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 976, 20), tuple_36902, int_36903)
        
        # Applying the binary operator 'in' (line 976)
        result_contains_36904 = python_operator(stypy.reporting.localization.Localization(__file__, 976, 13), 'in', mf_36901, tuple_36902)
        
        # Testing the type of an if condition (line 976)
        if_condition_36905 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 976, 13), result_contains_36904)
        # Assigning a type to the variable 'if_condition_36905' (line 976)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 13), 'if_condition_36905', if_condition_36905)
        # SSA begins for if statement (line 976)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 977):
        
        # Assigning a BinOp to a Name (line 977):
        int_36906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 18), 'int')
        # Getting the type of 'n' (line 977)
        n_36907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 22), 'n')
        # Applying the binary operator '*' (line 977)
        result_mul_36908 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 18), '*', int_36906, n_36907)
        
        # Assigning a type to the variable 'lzw' (line 977)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 977, 12), 'lzw', result_mul_36908)
        # SSA branch for the else part of an if statement (line 976)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 978)
        mf_36909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'tuple' (line 978)
        tuple_36910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 978)
        # Adding element type (line 978)
        int_36911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 978, 20), tuple_36910, int_36911)
        # Adding element type (line 978)
        int_36912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 978, 20), tuple_36910, int_36912)
        
        # Applying the binary operator 'in' (line 978)
        result_contains_36913 = python_operator(stypy.reporting.localization.Localization(__file__, 978, 13), 'in', mf_36909, tuple_36910)
        
        # Testing the type of an if condition (line 978)
        if_condition_36914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 978, 13), result_contains_36913)
        # Assigning a type to the variable 'if_condition_36914' (line 978)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 13), 'if_condition_36914', if_condition_36914)
        # SSA begins for if statement (line 978)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 979):
        
        # Assigning a BinOp to a Name (line 979):
        int_36915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 18), 'int')
        # Getting the type of 'n' (line 979)
        n_36916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 22), 'n')
        # Applying the binary operator '*' (line 979)
        result_mul_36917 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 18), '*', int_36915, n_36916)
        
        int_36918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 26), 'int')
        # Getting the type of 'n' (line 979)
        n_36919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 30), 'n')
        int_36920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 35), 'int')
        # Applying the binary operator '**' (line 979)
        result_pow_36921 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 30), '**', n_36919, int_36920)
        
        # Applying the binary operator '*' (line 979)
        result_mul_36922 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 26), '*', int_36918, result_pow_36921)
        
        # Applying the binary operator '+' (line 979)
        result_add_36923 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 18), '+', result_mul_36917, result_mul_36922)
        
        # Assigning a type to the variable 'lzw' (line 979)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 12), 'lzw', result_add_36923)
        # SSA branch for the else part of an if statement (line 978)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 980)
        mf_36924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'tuple' (line 980)
        tuple_36925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 980)
        # Adding element type (line 980)
        int_36926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 980, 20), tuple_36925, int_36926)
        # Adding element type (line 980)
        int_36927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 980, 20), tuple_36925, int_36927)
        
        # Applying the binary operator 'in' (line 980)
        result_contains_36928 = python_operator(stypy.reporting.localization.Localization(__file__, 980, 13), 'in', mf_36924, tuple_36925)
        
        # Testing the type of an if condition (line 980)
        if_condition_36929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 980, 13), result_contains_36928)
        # Assigning a type to the variable 'if_condition_36929' (line 980)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 13), 'if_condition_36929', if_condition_36929)
        # SSA begins for if statement (line 980)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 981):
        
        # Assigning a BinOp to a Name (line 981):
        int_36930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 18), 'int')
        # Getting the type of 'n' (line 981)
        n_36931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 22), 'n')
        # Applying the binary operator '*' (line 981)
        result_mul_36932 = python_operator(stypy.reporting.localization.Localization(__file__, 981, 18), '*', int_36930, n_36931)
        
        # Getting the type of 'n' (line 981)
        n_36933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 26), 'n')
        int_36934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 31), 'int')
        # Applying the binary operator '**' (line 981)
        result_pow_36935 = python_operator(stypy.reporting.localization.Localization(__file__, 981, 26), '**', n_36933, int_36934)
        
        # Applying the binary operator '+' (line 981)
        result_add_36936 = python_operator(stypy.reporting.localization.Localization(__file__, 981, 18), '+', result_mul_36932, result_pow_36935)
        
        # Assigning a type to the variable 'lzw' (line 981)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 12), 'lzw', result_add_36936)
        # SSA branch for the else part of an if statement (line 980)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 982)
        mf_36937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'tuple' (line 982)
        tuple_36938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 982, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 982)
        # Adding element type (line 982)
        int_36939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 982, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 982, 20), tuple_36938, int_36939)
        
        # Applying the binary operator 'in' (line 982)
        result_contains_36940 = python_operator(stypy.reporting.localization.Localization(__file__, 982, 13), 'in', mf_36937, tuple_36938)
        
        # Testing the type of an if condition (line 982)
        if_condition_36941 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 982, 13), result_contains_36940)
        # Assigning a type to the variable 'if_condition_36941' (line 982)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 13), 'if_condition_36941', if_condition_36941)
        # SSA begins for if statement (line 982)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 983):
        
        # Assigning a BinOp to a Name (line 983):
        int_36942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 983, 18), 'int')
        # Getting the type of 'n' (line 983)
        n_36943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 22), 'n')
        # Applying the binary operator '*' (line 983)
        result_mul_36944 = python_operator(stypy.reporting.localization.Localization(__file__, 983, 18), '*', int_36942, n_36943)
        
        # Assigning a type to the variable 'lzw' (line 983)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 983, 12), 'lzw', result_mul_36944)
        # SSA branch for the else part of an if statement (line 982)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 984)
        mf_36945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'tuple' (line 984)
        tuple_36946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 984, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 984)
        # Adding element type (line 984)
        int_36947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 984, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 984, 20), tuple_36946, int_36947)
        # Adding element type (line 984)
        int_36948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 984, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 984, 20), tuple_36946, int_36948)
        
        # Applying the binary operator 'in' (line 984)
        result_contains_36949 = python_operator(stypy.reporting.localization.Localization(__file__, 984, 13), 'in', mf_36945, tuple_36946)
        
        # Testing the type of an if condition (line 984)
        if_condition_36950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 984, 13), result_contains_36949)
        # Assigning a type to the variable 'if_condition_36950' (line 984)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 13), 'if_condition_36950', if_condition_36950)
        # SSA begins for if statement (line 984)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 985):
        
        # Assigning a BinOp to a Name (line 985):
        int_36951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 985, 18), 'int')
        # Getting the type of 'n' (line 985)
        n_36952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 23), 'n')
        # Applying the binary operator '*' (line 985)
        result_mul_36953 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 18), '*', int_36951, n_36952)
        
        int_36954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 985, 28), 'int')
        # Getting the type of 'self' (line 985)
        self_36955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 32), 'self')
        # Obtaining the member 'ml' of a type (line 985)
        ml_36956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 985, 32), self_36955, 'ml')
        # Applying the binary operator '*' (line 985)
        result_mul_36957 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 28), '*', int_36954, ml_36956)
        
        int_36958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 985, 42), 'int')
        # Getting the type of 'self' (line 985)
        self_36959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 46), 'self')
        # Obtaining the member 'mu' of a type (line 985)
        mu_36960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 985, 46), self_36959, 'mu')
        # Applying the binary operator '*' (line 985)
        result_mul_36961 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 42), '*', int_36958, mu_36960)
        
        # Applying the binary operator '+' (line 985)
        result_add_36962 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 28), '+', result_mul_36957, result_mul_36961)
        
        # Getting the type of 'n' (line 985)
        n_36963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 57), 'n')
        # Applying the binary operator '*' (line 985)
        result_mul_36964 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 27), '*', result_add_36962, n_36963)
        
        # Applying the binary operator '+' (line 985)
        result_add_36965 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 18), '+', result_mul_36953, result_mul_36964)
        
        # Assigning a type to the variable 'lzw' (line 985)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 12), 'lzw', result_add_36965)
        # SSA branch for the else part of an if statement (line 984)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mf' (line 986)
        mf_36966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 13), 'mf')
        
        # Obtaining an instance of the builtin type 'tuple' (line 986)
        tuple_36967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 986, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 986)
        # Adding element type (line 986)
        int_36968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 986, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 986, 20), tuple_36967, int_36968)
        # Adding element type (line 986)
        int_36969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 986, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 986, 20), tuple_36967, int_36969)
        
        # Applying the binary operator 'in' (line 986)
        result_contains_36970 = python_operator(stypy.reporting.localization.Localization(__file__, 986, 13), 'in', mf_36966, tuple_36967)
        
        # Testing the type of an if condition (line 986)
        if_condition_36971 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 986, 13), result_contains_36970)
        # Assigning a type to the variable 'if_condition_36971' (line 986)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 986, 13), 'if_condition_36971', if_condition_36971)
        # SSA begins for if statement (line 986)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 987):
        
        # Assigning a BinOp to a Name (line 987):
        int_36972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 987, 18), 'int')
        # Getting the type of 'n' (line 987)
        n_36973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 22), 'n')
        # Applying the binary operator '*' (line 987)
        result_mul_36974 = python_operator(stypy.reporting.localization.Localization(__file__, 987, 18), '*', int_36972, n_36973)
        
        int_36975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 987, 27), 'int')
        # Getting the type of 'self' (line 987)
        self_36976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 31), 'self')
        # Obtaining the member 'ml' of a type (line 987)
        ml_36977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 987, 31), self_36976, 'ml')
        # Applying the binary operator '*' (line 987)
        result_mul_36978 = python_operator(stypy.reporting.localization.Localization(__file__, 987, 27), '*', int_36975, ml_36977)
        
        # Getting the type of 'self' (line 987)
        self_36979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 41), 'self')
        # Obtaining the member 'mu' of a type (line 987)
        mu_36980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 987, 41), self_36979, 'mu')
        # Applying the binary operator '+' (line 987)
        result_add_36981 = python_operator(stypy.reporting.localization.Localization(__file__, 987, 27), '+', result_mul_36978, mu_36980)
        
        # Getting the type of 'n' (line 987)
        n_36982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 52), 'n')
        # Applying the binary operator '*' (line 987)
        result_mul_36983 = python_operator(stypy.reporting.localization.Localization(__file__, 987, 26), '*', result_add_36981, n_36982)
        
        # Applying the binary operator '+' (line 987)
        result_add_36984 = python_operator(stypy.reporting.localization.Localization(__file__, 987, 18), '+', result_mul_36974, result_mul_36983)
        
        # Assigning a type to the variable 'lzw' (line 987)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 987, 12), 'lzw', result_add_36984)
        # SSA join for if statement (line 986)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 984)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 982)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 980)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 978)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 976)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 974)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 972)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 970)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 968)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 966)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 964)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 989):
        
        # Assigning a BinOp to a Name (line 989):
        int_36985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 989, 14), 'int')
        # Getting the type of 'n' (line 989)
        n_36986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 19), 'n')
        # Applying the binary operator '+' (line 989)
        result_add_36987 = python_operator(stypy.reporting.localization.Localization(__file__, 989, 14), '+', int_36985, n_36986)
        
        # Assigning a type to the variable 'lrw' (line 989)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 989, 8), 'lrw', result_add_36987)
        
        
        # Getting the type of 'mf' (line 991)
        mf_36988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 11), 'mf')
        int_36989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 991, 16), 'int')
        # Applying the binary operator '%' (line 991)
        result_mod_36990 = python_operator(stypy.reporting.localization.Localization(__file__, 991, 11), '%', mf_36988, int_36989)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 991)
        tuple_36991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 991, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 991)
        # Adding element type (line 991)
        int_36992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 991, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 991, 23), tuple_36991, int_36992)
        # Adding element type (line 991)
        int_36993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 991, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 991, 23), tuple_36991, int_36993)
        
        # Applying the binary operator 'in' (line 991)
        result_contains_36994 = python_operator(stypy.reporting.localization.Localization(__file__, 991, 11), 'in', result_mod_36990, tuple_36991)
        
        # Testing the type of an if condition (line 991)
        if_condition_36995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 991, 8), result_contains_36994)
        # Assigning a type to the variable 'if_condition_36995' (line 991)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 991, 8), 'if_condition_36995', if_condition_36995)
        # SSA begins for if statement (line 991)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 992):
        
        # Assigning a Num to a Name (line 992):
        int_36996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 992, 18), 'int')
        # Assigning a type to the variable 'liw' (line 992)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 992, 12), 'liw', int_36996)
        # SSA branch for the else part of an if statement (line 991)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 994):
        
        # Assigning a BinOp to a Name (line 994):
        int_36997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 994, 18), 'int')
        # Getting the type of 'n' (line 994)
        n_36998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 23), 'n')
        # Applying the binary operator '+' (line 994)
        result_add_36999 = python_operator(stypy.reporting.localization.Localization(__file__, 994, 18), '+', int_36997, n_36998)
        
        # Assigning a type to the variable 'liw' (line 994)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 994, 12), 'liw', result_add_36999)
        # SSA join for if statement (line 991)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 996):
        
        # Assigning a Call to a Name (line 996):
        
        # Call to zeros(...): (line 996)
        # Processing the call arguments (line 996)
        
        # Obtaining an instance of the builtin type 'tuple' (line 996)
        tuple_37001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 996, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 996)
        # Adding element type (line 996)
        # Getting the type of 'lzw' (line 996)
        lzw_37002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 23), 'lzw', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 996, 23), tuple_37001, lzw_37002)
        
        # Getting the type of 'complex' (line 996)
        complex_37003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 30), 'complex', False)
        # Processing the call keyword arguments (line 996)
        kwargs_37004 = {}
        # Getting the type of 'zeros' (line 996)
        zeros_37000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 996)
        zeros_call_result_37005 = invoke(stypy.reporting.localization.Localization(__file__, 996, 16), zeros_37000, *[tuple_37001, complex_37003], **kwargs_37004)
        
        # Assigning a type to the variable 'zwork' (line 996)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 996, 8), 'zwork', zeros_call_result_37005)
        
        # Assigning a Name to a Attribute (line 997):
        
        # Assigning a Name to a Attribute (line 997):
        # Getting the type of 'zwork' (line 997)
        zwork_37006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 21), 'zwork')
        # Getting the type of 'self' (line 997)
        self_37007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 8), 'self')
        # Setting the type of the member 'zwork' of a type (line 997)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 997, 8), self_37007, 'zwork', zwork_37006)
        
        # Assigning a Call to a Name (line 999):
        
        # Assigning a Call to a Name (line 999):
        
        # Call to zeros(...): (line 999)
        # Processing the call arguments (line 999)
        
        # Obtaining an instance of the builtin type 'tuple' (line 999)
        tuple_37009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 999, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 999)
        # Adding element type (line 999)
        # Getting the type of 'lrw' (line 999)
        lrw_37010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 23), 'lrw', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 23), tuple_37009, lrw_37010)
        
        # Getting the type of 'float' (line 999)
        float_37011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 30), 'float', False)
        # Processing the call keyword arguments (line 999)
        kwargs_37012 = {}
        # Getting the type of 'zeros' (line 999)
        zeros_37008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 999)
        zeros_call_result_37013 = invoke(stypy.reporting.localization.Localization(__file__, 999, 16), zeros_37008, *[tuple_37009, float_37011], **kwargs_37012)
        
        # Assigning a type to the variable 'rwork' (line 999)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 999, 8), 'rwork', zeros_call_result_37013)
        
        # Assigning a Attribute to a Subscript (line 1000):
        
        # Assigning a Attribute to a Subscript (line 1000):
        # Getting the type of 'self' (line 1000)
        self_37014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 19), 'self')
        # Obtaining the member 'first_step' of a type (line 1000)
        first_step_37015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1000, 19), self_37014, 'first_step')
        # Getting the type of 'rwork' (line 1000)
        rwork_37016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 8), 'rwork')
        int_37017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1000, 14), 'int')
        # Storing an element on a container (line 1000)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1000, 8), rwork_37016, (int_37017, first_step_37015))
        
        # Assigning a Attribute to a Subscript (line 1001):
        
        # Assigning a Attribute to a Subscript (line 1001):
        # Getting the type of 'self' (line 1001)
        self_37018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 19), 'self')
        # Obtaining the member 'max_step' of a type (line 1001)
        max_step_37019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 19), self_37018, 'max_step')
        # Getting the type of 'rwork' (line 1001)
        rwork_37020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 8), 'rwork')
        int_37021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1001, 14), 'int')
        # Storing an element on a container (line 1001)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1001, 8), rwork_37020, (int_37021, max_step_37019))
        
        # Assigning a Attribute to a Subscript (line 1002):
        
        # Assigning a Attribute to a Subscript (line 1002):
        # Getting the type of 'self' (line 1002)
        self_37022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 19), 'self')
        # Obtaining the member 'min_step' of a type (line 1002)
        min_step_37023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1002, 19), self_37022, 'min_step')
        # Getting the type of 'rwork' (line 1002)
        rwork_37024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 8), 'rwork')
        int_37025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1002, 14), 'int')
        # Storing an element on a container (line 1002)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1002, 8), rwork_37024, (int_37025, min_step_37023))
        
        # Assigning a Name to a Attribute (line 1003):
        
        # Assigning a Name to a Attribute (line 1003):
        # Getting the type of 'rwork' (line 1003)
        rwork_37026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 21), 'rwork')
        # Getting the type of 'self' (line 1003)
        self_37027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 8), 'self')
        # Setting the type of the member 'rwork' of a type (line 1003)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1003, 8), self_37027, 'rwork', rwork_37026)
        
        # Assigning a Call to a Name (line 1005):
        
        # Assigning a Call to a Name (line 1005):
        
        # Call to zeros(...): (line 1005)
        # Processing the call arguments (line 1005)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1005)
        tuple_37029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1005, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1005)
        # Adding element type (line 1005)
        # Getting the type of 'liw' (line 1005)
        liw_37030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 23), 'liw', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1005, 23), tuple_37029, liw_37030)
        
        # Getting the type of 'int32' (line 1005)
        int32_37031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 30), 'int32', False)
        # Processing the call keyword arguments (line 1005)
        kwargs_37032 = {}
        # Getting the type of 'zeros' (line 1005)
        zeros_37028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 1005)
        zeros_call_result_37033 = invoke(stypy.reporting.localization.Localization(__file__, 1005, 16), zeros_37028, *[tuple_37029, int32_37031], **kwargs_37032)
        
        # Assigning a type to the variable 'iwork' (line 1005)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1005, 8), 'iwork', zeros_call_result_37033)
        
        
        # Getting the type of 'self' (line 1006)
        self_37034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 11), 'self')
        # Obtaining the member 'ml' of a type (line 1006)
        ml_37035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1006, 11), self_37034, 'ml')
        # Getting the type of 'None' (line 1006)
        None_37036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 26), 'None')
        # Applying the binary operator 'isnot' (line 1006)
        result_is_not_37037 = python_operator(stypy.reporting.localization.Localization(__file__, 1006, 11), 'isnot', ml_37035, None_37036)
        
        # Testing the type of an if condition (line 1006)
        if_condition_37038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1006, 8), result_is_not_37037)
        # Assigning a type to the variable 'if_condition_37038' (line 1006)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1006, 8), 'if_condition_37038', if_condition_37038)
        # SSA begins for if statement (line 1006)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 1007):
        
        # Assigning a Attribute to a Subscript (line 1007):
        # Getting the type of 'self' (line 1007)
        self_37039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 23), 'self')
        # Obtaining the member 'ml' of a type (line 1007)
        ml_37040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 23), self_37039, 'ml')
        # Getting the type of 'iwork' (line 1007)
        iwork_37041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 12), 'iwork')
        int_37042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 18), 'int')
        # Storing an element on a container (line 1007)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1007, 12), iwork_37041, (int_37042, ml_37040))
        # SSA join for if statement (line 1006)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 1008)
        self_37043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 11), 'self')
        # Obtaining the member 'mu' of a type (line 1008)
        mu_37044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1008, 11), self_37043, 'mu')
        # Getting the type of 'None' (line 1008)
        None_37045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 26), 'None')
        # Applying the binary operator 'isnot' (line 1008)
        result_is_not_37046 = python_operator(stypy.reporting.localization.Localization(__file__, 1008, 11), 'isnot', mu_37044, None_37045)
        
        # Testing the type of an if condition (line 1008)
        if_condition_37047 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1008, 8), result_is_not_37046)
        # Assigning a type to the variable 'if_condition_37047' (line 1008)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1008, 8), 'if_condition_37047', if_condition_37047)
        # SSA begins for if statement (line 1008)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 1009):
        
        # Assigning a Attribute to a Subscript (line 1009):
        # Getting the type of 'self' (line 1009)
        self_37048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 23), 'self')
        # Obtaining the member 'mu' of a type (line 1009)
        mu_37049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1009, 23), self_37048, 'mu')
        # Getting the type of 'iwork' (line 1009)
        iwork_37050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 12), 'iwork')
        int_37051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1009, 18), 'int')
        # Storing an element on a container (line 1009)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1009, 12), iwork_37050, (int_37051, mu_37049))
        # SSA join for if statement (line 1008)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Subscript (line 1010):
        
        # Assigning a Attribute to a Subscript (line 1010):
        # Getting the type of 'self' (line 1010)
        self_37052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 19), 'self')
        # Obtaining the member 'order' of a type (line 1010)
        order_37053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1010, 19), self_37052, 'order')
        # Getting the type of 'iwork' (line 1010)
        iwork_37054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 8), 'iwork')
        int_37055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1010, 14), 'int')
        # Storing an element on a container (line 1010)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1010, 8), iwork_37054, (int_37055, order_37053))
        
        # Assigning a Attribute to a Subscript (line 1011):
        
        # Assigning a Attribute to a Subscript (line 1011):
        # Getting the type of 'self' (line 1011)
        self_37056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 19), 'self')
        # Obtaining the member 'nsteps' of a type (line 1011)
        nsteps_37057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1011, 19), self_37056, 'nsteps')
        # Getting the type of 'iwork' (line 1011)
        iwork_37058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 8), 'iwork')
        int_37059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 14), 'int')
        # Storing an element on a container (line 1011)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 8), iwork_37058, (int_37059, nsteps_37057))
        
        # Assigning a Num to a Subscript (line 1012):
        
        # Assigning a Num to a Subscript (line 1012):
        int_37060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1012, 19), 'int')
        # Getting the type of 'iwork' (line 1012)
        iwork_37061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 8), 'iwork')
        int_37062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1012, 14), 'int')
        # Storing an element on a container (line 1012)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1012, 8), iwork_37061, (int_37062, int_37060))
        
        # Assigning a Name to a Attribute (line 1013):
        
        # Assigning a Name to a Attribute (line 1013):
        # Getting the type of 'iwork' (line 1013)
        iwork_37063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 21), 'iwork')
        # Getting the type of 'self' (line 1013)
        self_37064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 8), 'self')
        # Setting the type of the member 'iwork' of a type (line 1013)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1013, 8), self_37064, 'iwork', iwork_37063)
        
        # Assigning a List to a Attribute (line 1015):
        
        # Assigning a List to a Attribute (line 1015):
        
        # Obtaining an instance of the builtin type 'list' (line 1015)
        list_37065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1015, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1015)
        # Adding element type (line 1015)
        # Getting the type of 'self' (line 1015)
        self_37066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 26), 'self')
        # Obtaining the member 'rtol' of a type (line 1015)
        rtol_37067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1015, 26), self_37066, 'rtol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1015, 25), list_37065, rtol_37067)
        # Adding element type (line 1015)
        # Getting the type of 'self' (line 1015)
        self_37068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 37), 'self')
        # Obtaining the member 'atol' of a type (line 1015)
        atol_37069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1015, 37), self_37068, 'atol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1015, 25), list_37065, atol_37069)
        # Adding element type (line 1015)
        int_37070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1015, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1015, 25), list_37065, int_37070)
        # Adding element type (line 1015)
        int_37071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1015, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1015, 25), list_37065, int_37071)
        # Adding element type (line 1015)
        # Getting the type of 'self' (line 1016)
        self_37072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 26), 'self')
        # Obtaining the member 'zwork' of a type (line 1016)
        zwork_37073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1016, 26), self_37072, 'zwork')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1015, 25), list_37065, zwork_37073)
        # Adding element type (line 1015)
        # Getting the type of 'self' (line 1016)
        self_37074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 38), 'self')
        # Obtaining the member 'rwork' of a type (line 1016)
        rwork_37075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1016, 38), self_37074, 'rwork')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1015, 25), list_37065, rwork_37075)
        # Adding element type (line 1015)
        # Getting the type of 'self' (line 1016)
        self_37076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 50), 'self')
        # Obtaining the member 'iwork' of a type (line 1016)
        iwork_37077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1016, 50), self_37076, 'iwork')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1015, 25), list_37065, iwork_37077)
        # Adding element type (line 1015)
        # Getting the type of 'mf' (line 1016)
        mf_37078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 62), 'mf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1015, 25), list_37065, mf_37078)
        
        # Getting the type of 'self' (line 1015)
        self_37079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 8), 'self')
        # Setting the type of the member 'call_args' of a type (line 1015)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1015, 8), self_37079, 'call_args', list_37065)
        
        # Assigning a Num to a Attribute (line 1017):
        
        # Assigning a Num to a Attribute (line 1017):
        int_37080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, 23), 'int')
        # Getting the type of 'self' (line 1017)
        self_37081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 8), 'self')
        # Setting the type of the member 'success' of a type (line 1017)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1017, 8), self_37081, 'success', int_37080)
        
        # Assigning a Name to a Attribute (line 1018):
        
        # Assigning a Name to a Attribute (line 1018):
        # Getting the type of 'False' (line 1018)
        False_37082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1018, 27), 'False')
        # Getting the type of 'self' (line 1018)
        self_37083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1018, 8), 'self')
        # Setting the type of the member 'initialized' of a type (line 1018)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1018, 8), self_37083, 'initialized', False_37082)
        
        # ################# End of 'reset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset' in the type store
        # Getting the type of 'stypy_return_type' (line 961)
        stypy_return_type_37084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37084)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset'
        return stypy_return_type_37084


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 953, 0, False)
        # Assigning a type to the variable 'self' (line 954)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'zvode.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'zvode' (line 953)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 0), 'zvode', zvode)

# Assigning a Call to a Name (line 954):

# Call to getattr(...): (line 954)
# Processing the call arguments (line 954)
# Getting the type of '_vode' (line 954)
_vode_37086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 21), '_vode', False)
str_37087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 28), 'str', 'zvode')
# Getting the type of 'None' (line 954)
None_37088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 37), 'None', False)
# Processing the call keyword arguments (line 954)
kwargs_37089 = {}
# Getting the type of 'getattr' (line 954)
getattr_37085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 13), 'getattr', False)
# Calling getattr(args, kwargs) (line 954)
getattr_call_result_37090 = invoke(stypy.reporting.localization.Localization(__file__, 954, 13), getattr_37085, *[_vode_37086, str_37087, None_37088], **kwargs_37089)

# Getting the type of 'zvode'
zvode_37091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'zvode')
# Setting the type of the member 'runner' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), zvode_37091, 'runner', getattr_call_result_37090)

# Assigning a Num to a Name (line 956):
int_37092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 956, 25), 'int')
# Getting the type of 'zvode'
zvode_37093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'zvode')
# Setting the type of the member 'supports_run_relax' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), zvode_37093, 'supports_run_relax', int_37092)

# Assigning a Num to a Name (line 957):
int_37094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 957, 20), 'int')
# Getting the type of 'zvode'
zvode_37095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'zvode')
# Setting the type of the member 'supports_step' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), zvode_37095, 'supports_step', int_37094)

# Assigning a Name to a Name (line 958):
# Getting the type of 'complex' (line 958)
complex_37096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 13), 'complex')
# Getting the type of 'zvode'
zvode_37097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'zvode')
# Setting the type of the member 'scalar' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), zvode_37097, 'scalar', complex_37096)

# Assigning a Num to a Name (line 959):
int_37098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 27), 'int')
# Getting the type of 'zvode'
zvode_37099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'zvode')
# Setting the type of the member 'active_global_handle' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), zvode_37099, 'active_global_handle', int_37098)


# Getting the type of 'zvode' (line 1021)
zvode_37100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1021, 3), 'zvode')
# Obtaining the member 'runner' of a type (line 1021)
runner_37101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1021, 3), zvode_37100, 'runner')
# Getting the type of 'None' (line 1021)
None_37102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1021, 23), 'None')
# Applying the binary operator 'isnot' (line 1021)
result_is_not_37103 = python_operator(stypy.reporting.localization.Localization(__file__, 1021, 3), 'isnot', runner_37101, None_37102)

# Testing the type of an if condition (line 1021)
if_condition_37104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1021, 0), result_is_not_37103)
# Assigning a type to the variable 'if_condition_37104' (line 1021)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1021, 0), 'if_condition_37104', if_condition_37104)
# SSA begins for if statement (line 1021)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to append(...): (line 1022)
# Processing the call arguments (line 1022)
# Getting the type of 'zvode' (line 1022)
zvode_37108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1022, 45), 'zvode', False)
# Processing the call keyword arguments (line 1022)
kwargs_37109 = {}
# Getting the type of 'IntegratorBase' (line 1022)
IntegratorBase_37105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1022, 4), 'IntegratorBase', False)
# Obtaining the member 'integrator_classes' of a type (line 1022)
integrator_classes_37106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1022, 4), IntegratorBase_37105, 'integrator_classes')
# Obtaining the member 'append' of a type (line 1022)
append_37107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1022, 4), integrator_classes_37106, 'append')
# Calling append(args, kwargs) (line 1022)
append_call_result_37110 = invoke(stypy.reporting.localization.Localization(__file__, 1022, 4), append_37107, *[zvode_37108], **kwargs_37109)

# SSA join for if statement (line 1021)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'dopri5' class
# Getting the type of 'IntegratorBase' (line 1025)
IntegratorBase_37111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 13), 'IntegratorBase')

class dopri5(IntegratorBase_37111, ):
    
    # Assigning a Call to a Name (line 1026):
    
    # Assigning a Str to a Name (line 1027):
    
    # Assigning a Name to a Name (line 1028):
    
    # Assigning a Dict to a Name (line 1030):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_37112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 22), 'float')
        float_37113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 33), 'float')
        int_37114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1040, 24), 'int')
        float_37115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 26), 'float')
        float_37116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 28), 'float')
        float_37117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 24), 'float')
        float_37118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 25), 'float')
        float_37119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1045, 25), 'float')
        float_37120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1046, 22), 'float')
        # Getting the type of 'None' (line 1047)
        None_37121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 24), 'None')
        int_37122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1048, 27), 'int')
        defaults = [float_37112, float_37113, int_37114, float_37115, float_37116, float_37117, float_37118, float_37119, float_37120, None_37121, int_37122]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1038, 4, False)
        # Assigning a type to the variable 'self' (line 1039)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dopri5.__init__', ['rtol', 'atol', 'nsteps', 'max_step', 'first_step', 'safety', 'ifactor', 'dfactor', 'beta', 'method', 'verbosity'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['rtol', 'atol', 'nsteps', 'max_step', 'first_step', 'safety', 'ifactor', 'dfactor', 'beta', 'method', 'verbosity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 1050):
        
        # Assigning a Name to a Attribute (line 1050):
        # Getting the type of 'rtol' (line 1050)
        rtol_37123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 20), 'rtol')
        # Getting the type of 'self' (line 1050)
        self_37124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 8), 'self')
        # Setting the type of the member 'rtol' of a type (line 1050)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1050, 8), self_37124, 'rtol', rtol_37123)
        
        # Assigning a Name to a Attribute (line 1051):
        
        # Assigning a Name to a Attribute (line 1051):
        # Getting the type of 'atol' (line 1051)
        atol_37125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 20), 'atol')
        # Getting the type of 'self' (line 1051)
        self_37126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 8), 'self')
        # Setting the type of the member 'atol' of a type (line 1051)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1051, 8), self_37126, 'atol', atol_37125)
        
        # Assigning a Name to a Attribute (line 1052):
        
        # Assigning a Name to a Attribute (line 1052):
        # Getting the type of 'nsteps' (line 1052)
        nsteps_37127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 22), 'nsteps')
        # Getting the type of 'self' (line 1052)
        self_37128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 8), 'self')
        # Setting the type of the member 'nsteps' of a type (line 1052)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 8), self_37128, 'nsteps', nsteps_37127)
        
        # Assigning a Name to a Attribute (line 1053):
        
        # Assigning a Name to a Attribute (line 1053):
        # Getting the type of 'max_step' (line 1053)
        max_step_37129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 24), 'max_step')
        # Getting the type of 'self' (line 1053)
        self_37130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 8), 'self')
        # Setting the type of the member 'max_step' of a type (line 1053)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 8), self_37130, 'max_step', max_step_37129)
        
        # Assigning a Name to a Attribute (line 1054):
        
        # Assigning a Name to a Attribute (line 1054):
        # Getting the type of 'first_step' (line 1054)
        first_step_37131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 26), 'first_step')
        # Getting the type of 'self' (line 1054)
        self_37132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 8), 'self')
        # Setting the type of the member 'first_step' of a type (line 1054)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 8), self_37132, 'first_step', first_step_37131)
        
        # Assigning a Name to a Attribute (line 1055):
        
        # Assigning a Name to a Attribute (line 1055):
        # Getting the type of 'safety' (line 1055)
        safety_37133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 22), 'safety')
        # Getting the type of 'self' (line 1055)
        self_37134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 8), 'self')
        # Setting the type of the member 'safety' of a type (line 1055)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1055, 8), self_37134, 'safety', safety_37133)
        
        # Assigning a Name to a Attribute (line 1056):
        
        # Assigning a Name to a Attribute (line 1056):
        # Getting the type of 'ifactor' (line 1056)
        ifactor_37135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 23), 'ifactor')
        # Getting the type of 'self' (line 1056)
        self_37136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 8), 'self')
        # Setting the type of the member 'ifactor' of a type (line 1056)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 8), self_37136, 'ifactor', ifactor_37135)
        
        # Assigning a Name to a Attribute (line 1057):
        
        # Assigning a Name to a Attribute (line 1057):
        # Getting the type of 'dfactor' (line 1057)
        dfactor_37137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 23), 'dfactor')
        # Getting the type of 'self' (line 1057)
        self_37138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 8), 'self')
        # Setting the type of the member 'dfactor' of a type (line 1057)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1057, 8), self_37138, 'dfactor', dfactor_37137)
        
        # Assigning a Name to a Attribute (line 1058):
        
        # Assigning a Name to a Attribute (line 1058):
        # Getting the type of 'beta' (line 1058)
        beta_37139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 20), 'beta')
        # Getting the type of 'self' (line 1058)
        self_37140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 8), 'self')
        # Setting the type of the member 'beta' of a type (line 1058)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1058, 8), self_37140, 'beta', beta_37139)
        
        # Assigning a Name to a Attribute (line 1059):
        
        # Assigning a Name to a Attribute (line 1059):
        # Getting the type of 'verbosity' (line 1059)
        verbosity_37141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 25), 'verbosity')
        # Getting the type of 'self' (line 1059)
        self_37142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 8), 'self')
        # Setting the type of the member 'verbosity' of a type (line 1059)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1059, 8), self_37142, 'verbosity', verbosity_37141)
        
        # Assigning a Num to a Attribute (line 1060):
        
        # Assigning a Num to a Attribute (line 1060):
        int_37143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1060, 23), 'int')
        # Getting the type of 'self' (line 1060)
        self_37144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 8), 'self')
        # Setting the type of the member 'success' of a type (line 1060)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1060, 8), self_37144, 'success', int_37143)
        
        # Call to set_solout(...): (line 1061)
        # Processing the call arguments (line 1061)
        # Getting the type of 'None' (line 1061)
        None_37147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 24), 'None', False)
        # Processing the call keyword arguments (line 1061)
        kwargs_37148 = {}
        # Getting the type of 'self' (line 1061)
        self_37145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 8), 'self', False)
        # Obtaining the member 'set_solout' of a type (line 1061)
        set_solout_37146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1061, 8), self_37145, 'set_solout')
        # Calling set_solout(args, kwargs) (line 1061)
        set_solout_call_result_37149 = invoke(stypy.reporting.localization.Localization(__file__, 1061, 8), set_solout_37146, *[None_37147], **kwargs_37148)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_solout(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 1063)
        False_37150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 41), 'False')
        defaults = [False_37150]
        # Create a new context for function 'set_solout'
        module_type_store = module_type_store.open_function_context('set_solout', 1063, 4, False)
        # Assigning a type to the variable 'self' (line 1064)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1064, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dopri5.set_solout.__dict__.__setitem__('stypy_localization', localization)
        dopri5.set_solout.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dopri5.set_solout.__dict__.__setitem__('stypy_type_store', module_type_store)
        dopri5.set_solout.__dict__.__setitem__('stypy_function_name', 'dopri5.set_solout')
        dopri5.set_solout.__dict__.__setitem__('stypy_param_names_list', ['solout', 'complex'])
        dopri5.set_solout.__dict__.__setitem__('stypy_varargs_param_name', None)
        dopri5.set_solout.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dopri5.set_solout.__dict__.__setitem__('stypy_call_defaults', defaults)
        dopri5.set_solout.__dict__.__setitem__('stypy_call_varargs', varargs)
        dopri5.set_solout.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dopri5.set_solout.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dopri5.set_solout', ['solout', 'complex'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_solout', localization, ['solout', 'complex'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_solout(...)' code ##################

        
        # Assigning a Name to a Attribute (line 1064):
        
        # Assigning a Name to a Attribute (line 1064):
        # Getting the type of 'solout' (line 1064)
        solout_37151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 22), 'solout')
        # Getting the type of 'self' (line 1064)
        self_37152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 8), 'self')
        # Setting the type of the member 'solout' of a type (line 1064)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1064, 8), self_37152, 'solout', solout_37151)
        
        # Assigning a Name to a Attribute (line 1065):
        
        # Assigning a Name to a Attribute (line 1065):
        # Getting the type of 'complex' (line 1065)
        complex_37153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 28), 'complex')
        # Getting the type of 'self' (line 1065)
        self_37154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 8), 'self')
        # Setting the type of the member 'solout_cmplx' of a type (line 1065)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1065, 8), self_37154, 'solout_cmplx', complex_37153)
        
        # Type idiom detected: calculating its left and rigth part (line 1066)
        # Getting the type of 'solout' (line 1066)
        solout_37155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 11), 'solout')
        # Getting the type of 'None' (line 1066)
        None_37156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 21), 'None')
        
        (may_be_37157, more_types_in_union_37158) = may_be_none(solout_37155, None_37156)

        if may_be_37157:

            if more_types_in_union_37158:
                # Runtime conditional SSA (line 1066)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Attribute (line 1067):
            
            # Assigning a Num to a Attribute (line 1067):
            int_37159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1067, 24), 'int')
            # Getting the type of 'self' (line 1067)
            self_37160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 12), 'self')
            # Setting the type of the member 'iout' of a type (line 1067)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 12), self_37160, 'iout', int_37159)

            if more_types_in_union_37158:
                # Runtime conditional SSA for else branch (line 1066)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_37157) or more_types_in_union_37158):
            
            # Assigning a Num to a Attribute (line 1069):
            
            # Assigning a Num to a Attribute (line 1069):
            int_37161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 24), 'int')
            # Getting the type of 'self' (line 1069)
            self_37162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 12), 'self')
            # Setting the type of the member 'iout' of a type (line 1069)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 12), self_37162, 'iout', int_37161)

            if (may_be_37157 and more_types_in_union_37158):
                # SSA join for if statement (line 1066)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'set_solout(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_solout' in the type store
        # Getting the type of 'stypy_return_type' (line 1063)
        stypy_return_type_37163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37163)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_solout'
        return stypy_return_type_37163


    @norecursion
    def reset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reset'
        module_type_store = module_type_store.open_function_context('reset', 1071, 4, False)
        # Assigning a type to the variable 'self' (line 1072)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1072, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dopri5.reset.__dict__.__setitem__('stypy_localization', localization)
        dopri5.reset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dopri5.reset.__dict__.__setitem__('stypy_type_store', module_type_store)
        dopri5.reset.__dict__.__setitem__('stypy_function_name', 'dopri5.reset')
        dopri5.reset.__dict__.__setitem__('stypy_param_names_list', ['n', 'has_jac'])
        dopri5.reset.__dict__.__setitem__('stypy_varargs_param_name', None)
        dopri5.reset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dopri5.reset.__dict__.__setitem__('stypy_call_defaults', defaults)
        dopri5.reset.__dict__.__setitem__('stypy_call_varargs', varargs)
        dopri5.reset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dopri5.reset.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dopri5.reset', ['n', 'has_jac'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reset', localization, ['n', 'has_jac'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reset(...)' code ##################

        
        # Assigning a Call to a Name (line 1072):
        
        # Assigning a Call to a Name (line 1072):
        
        # Call to zeros(...): (line 1072)
        # Processing the call arguments (line 1072)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1072)
        tuple_37165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1072, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1072)
        # Adding element type (line 1072)
        int_37166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1072, 22), 'int')
        # Getting the type of 'n' (line 1072)
        n_37167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 26), 'n', False)
        # Applying the binary operator '*' (line 1072)
        result_mul_37168 = python_operator(stypy.reporting.localization.Localization(__file__, 1072, 22), '*', int_37166, n_37167)
        
        int_37169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1072, 30), 'int')
        # Applying the binary operator '+' (line 1072)
        result_add_37170 = python_operator(stypy.reporting.localization.Localization(__file__, 1072, 22), '+', result_mul_37168, int_37169)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1072, 22), tuple_37165, result_add_37170)
        
        # Getting the type of 'float' (line 1072)
        float_37171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 36), 'float', False)
        # Processing the call keyword arguments (line 1072)
        kwargs_37172 = {}
        # Getting the type of 'zeros' (line 1072)
        zeros_37164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 15), 'zeros', False)
        # Calling zeros(args, kwargs) (line 1072)
        zeros_call_result_37173 = invoke(stypy.reporting.localization.Localization(__file__, 1072, 15), zeros_37164, *[tuple_37165, float_37171], **kwargs_37172)
        
        # Assigning a type to the variable 'work' (line 1072)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1072, 8), 'work', zeros_call_result_37173)
        
        # Assigning a Attribute to a Subscript (line 1073):
        
        # Assigning a Attribute to a Subscript (line 1073):
        # Getting the type of 'self' (line 1073)
        self_37174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 18), 'self')
        # Obtaining the member 'safety' of a type (line 1073)
        safety_37175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 18), self_37174, 'safety')
        # Getting the type of 'work' (line 1073)
        work_37176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'work')
        int_37177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1073, 13), 'int')
        # Storing an element on a container (line 1073)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1073, 8), work_37176, (int_37177, safety_37175))
        
        # Assigning a Attribute to a Subscript (line 1074):
        
        # Assigning a Attribute to a Subscript (line 1074):
        # Getting the type of 'self' (line 1074)
        self_37178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 18), 'self')
        # Obtaining the member 'dfactor' of a type (line 1074)
        dfactor_37179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1074, 18), self_37178, 'dfactor')
        # Getting the type of 'work' (line 1074)
        work_37180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 8), 'work')
        int_37181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1074, 13), 'int')
        # Storing an element on a container (line 1074)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1074, 8), work_37180, (int_37181, dfactor_37179))
        
        # Assigning a Attribute to a Subscript (line 1075):
        
        # Assigning a Attribute to a Subscript (line 1075):
        # Getting the type of 'self' (line 1075)
        self_37182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 18), 'self')
        # Obtaining the member 'ifactor' of a type (line 1075)
        ifactor_37183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 18), self_37182, 'ifactor')
        # Getting the type of 'work' (line 1075)
        work_37184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'work')
        int_37185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 13), 'int')
        # Storing an element on a container (line 1075)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1075, 8), work_37184, (int_37185, ifactor_37183))
        
        # Assigning a Attribute to a Subscript (line 1076):
        
        # Assigning a Attribute to a Subscript (line 1076):
        # Getting the type of 'self' (line 1076)
        self_37186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 18), 'self')
        # Obtaining the member 'beta' of a type (line 1076)
        beta_37187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1076, 18), self_37186, 'beta')
        # Getting the type of 'work' (line 1076)
        work_37188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 8), 'work')
        int_37189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1076, 13), 'int')
        # Storing an element on a container (line 1076)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1076, 8), work_37188, (int_37189, beta_37187))
        
        # Assigning a Attribute to a Subscript (line 1077):
        
        # Assigning a Attribute to a Subscript (line 1077):
        # Getting the type of 'self' (line 1077)
        self_37190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 18), 'self')
        # Obtaining the member 'max_step' of a type (line 1077)
        max_step_37191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1077, 18), self_37190, 'max_step')
        # Getting the type of 'work' (line 1077)
        work_37192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 8), 'work')
        int_37193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1077, 13), 'int')
        # Storing an element on a container (line 1077)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1077, 8), work_37192, (int_37193, max_step_37191))
        
        # Assigning a Attribute to a Subscript (line 1078):
        
        # Assigning a Attribute to a Subscript (line 1078):
        # Getting the type of 'self' (line 1078)
        self_37194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 18), 'self')
        # Obtaining the member 'first_step' of a type (line 1078)
        first_step_37195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1078, 18), self_37194, 'first_step')
        # Getting the type of 'work' (line 1078)
        work_37196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 8), 'work')
        int_37197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1078, 13), 'int')
        # Storing an element on a container (line 1078)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1078, 8), work_37196, (int_37197, first_step_37195))
        
        # Assigning a Name to a Attribute (line 1079):
        
        # Assigning a Name to a Attribute (line 1079):
        # Getting the type of 'work' (line 1079)
        work_37198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 20), 'work')
        # Getting the type of 'self' (line 1079)
        self_37199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'self')
        # Setting the type of the member 'work' of a type (line 1079)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1079, 8), self_37199, 'work', work_37198)
        
        # Assigning a Call to a Name (line 1080):
        
        # Assigning a Call to a Name (line 1080):
        
        # Call to zeros(...): (line 1080)
        # Processing the call arguments (line 1080)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1080)
        tuple_37201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1080, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1080)
        # Adding element type (line 1080)
        int_37202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1080, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1080, 23), tuple_37201, int_37202)
        
        # Getting the type of 'int32' (line 1080)
        int32_37203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 29), 'int32', False)
        # Processing the call keyword arguments (line 1080)
        kwargs_37204 = {}
        # Getting the type of 'zeros' (line 1080)
        zeros_37200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 1080)
        zeros_call_result_37205 = invoke(stypy.reporting.localization.Localization(__file__, 1080, 16), zeros_37200, *[tuple_37201, int32_37203], **kwargs_37204)
        
        # Assigning a type to the variable 'iwork' (line 1080)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1080, 8), 'iwork', zeros_call_result_37205)
        
        # Assigning a Attribute to a Subscript (line 1081):
        
        # Assigning a Attribute to a Subscript (line 1081):
        # Getting the type of 'self' (line 1081)
        self_37206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 19), 'self')
        # Obtaining the member 'nsteps' of a type (line 1081)
        nsteps_37207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1081, 19), self_37206, 'nsteps')
        # Getting the type of 'iwork' (line 1081)
        iwork_37208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 8), 'iwork')
        int_37209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1081, 14), 'int')
        # Storing an element on a container (line 1081)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1081, 8), iwork_37208, (int_37209, nsteps_37207))
        
        # Assigning a Attribute to a Subscript (line 1082):
        
        # Assigning a Attribute to a Subscript (line 1082):
        # Getting the type of 'self' (line 1082)
        self_37210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1082, 19), 'self')
        # Obtaining the member 'verbosity' of a type (line 1082)
        verbosity_37211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1082, 19), self_37210, 'verbosity')
        # Getting the type of 'iwork' (line 1082)
        iwork_37212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1082, 8), 'iwork')
        int_37213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1082, 14), 'int')
        # Storing an element on a container (line 1082)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1082, 8), iwork_37212, (int_37213, verbosity_37211))
        
        # Assigning a Name to a Attribute (line 1083):
        
        # Assigning a Name to a Attribute (line 1083):
        # Getting the type of 'iwork' (line 1083)
        iwork_37214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 21), 'iwork')
        # Getting the type of 'self' (line 1083)
        self_37215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 8), 'self')
        # Setting the type of the member 'iwork' of a type (line 1083)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1083, 8), self_37215, 'iwork', iwork_37214)
        
        # Assigning a List to a Attribute (line 1084):
        
        # Assigning a List to a Attribute (line 1084):
        
        # Obtaining an instance of the builtin type 'list' (line 1084)
        list_37216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1084, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1084)
        # Adding element type (line 1084)
        # Getting the type of 'self' (line 1084)
        self_37217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 26), 'self')
        # Obtaining the member 'rtol' of a type (line 1084)
        rtol_37218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1084, 26), self_37217, 'rtol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1084, 25), list_37216, rtol_37218)
        # Adding element type (line 1084)
        # Getting the type of 'self' (line 1084)
        self_37219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 37), 'self')
        # Obtaining the member 'atol' of a type (line 1084)
        atol_37220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1084, 37), self_37219, 'atol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1084, 25), list_37216, atol_37220)
        # Adding element type (line 1084)
        # Getting the type of 'self' (line 1084)
        self_37221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 48), 'self')
        # Obtaining the member '_solout' of a type (line 1084)
        _solout_37222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1084, 48), self_37221, '_solout')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1084, 25), list_37216, _solout_37222)
        # Adding element type (line 1084)
        # Getting the type of 'self' (line 1085)
        self_37223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 26), 'self')
        # Obtaining the member 'iout' of a type (line 1085)
        iout_37224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 26), self_37223, 'iout')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1084, 25), list_37216, iout_37224)
        # Adding element type (line 1084)
        # Getting the type of 'self' (line 1085)
        self_37225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 37), 'self')
        # Obtaining the member 'work' of a type (line 1085)
        work_37226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 37), self_37225, 'work')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1084, 25), list_37216, work_37226)
        # Adding element type (line 1084)
        # Getting the type of 'self' (line 1085)
        self_37227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 48), 'self')
        # Obtaining the member 'iwork' of a type (line 1085)
        iwork_37228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 48), self_37227, 'iwork')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1084, 25), list_37216, iwork_37228)
        
        # Getting the type of 'self' (line 1084)
        self_37229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 8), 'self')
        # Setting the type of the member 'call_args' of a type (line 1084)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1084, 8), self_37229, 'call_args', list_37216)
        
        # Assigning a Num to a Attribute (line 1086):
        
        # Assigning a Num to a Attribute (line 1086):
        int_37230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1086, 23), 'int')
        # Getting the type of 'self' (line 1086)
        self_37231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 8), 'self')
        # Setting the type of the member 'success' of a type (line 1086)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1086, 8), self_37231, 'success', int_37230)
        
        # ################# End of 'reset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset' in the type store
        # Getting the type of 'stypy_return_type' (line 1071)
        stypy_return_type_37232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1071, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37232)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset'
        return stypy_return_type_37232


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 1088, 4, False)
        # Assigning a type to the variable 'self' (line 1089)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dopri5.run.__dict__.__setitem__('stypy_localization', localization)
        dopri5.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dopri5.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        dopri5.run.__dict__.__setitem__('stypy_function_name', 'dopri5.run')
        dopri5.run.__dict__.__setitem__('stypy_param_names_list', ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'])
        dopri5.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        dopri5.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dopri5.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        dopri5.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        dopri5.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dopri5.run.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dopri5.run', ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        # Assigning a Call to a Tuple (line 1089):
        
        # Assigning a Subscript to a Name (line 1089):
        
        # Obtaining the type of the subscript
        int_37233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 8), 'int')
        
        # Call to runner(...): (line 1089)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1089)
        tuple_37236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1089)
        # Adding element type (line 1089)
        # Getting the type of 'f' (line 1089)
        f_37237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 45), 'f', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37236, f_37237)
        # Adding element type (line 1089)
        # Getting the type of 't0' (line 1089)
        t0_37238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 48), 't0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37236, t0_37238)
        # Adding element type (line 1089)
        # Getting the type of 'y0' (line 1089)
        y0_37239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 52), 'y0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37236, y0_37239)
        # Adding element type (line 1089)
        # Getting the type of 't1' (line 1089)
        t1_37240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 56), 't1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37236, t1_37240)
        
        
        # Call to tuple(...): (line 1090)
        # Processing the call arguments (line 1090)
        # Getting the type of 'self' (line 1090)
        self_37242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 48), 'self', False)
        # Obtaining the member 'call_args' of a type (line 1090)
        call_args_37243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1090, 48), self_37242, 'call_args')
        # Processing the call keyword arguments (line 1090)
        kwargs_37244 = {}
        # Getting the type of 'tuple' (line 1090)
        tuple_37241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 42), 'tuple', False)
        # Calling tuple(args, kwargs) (line 1090)
        tuple_call_result_37245 = invoke(stypy.reporting.localization.Localization(__file__, 1090, 42), tuple_37241, *[call_args_37243], **kwargs_37244)
        
        # Applying the binary operator '+' (line 1089)
        result_add_37246 = python_operator(stypy.reporting.localization.Localization(__file__, 1089, 44), '+', tuple_37236, tuple_call_result_37245)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 1090)
        tuple_37247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1090, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1090)
        # Adding element type (line 1090)
        # Getting the type of 'f_params' (line 1090)
        f_params_37248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 67), 'f_params', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1090, 67), tuple_37247, f_params_37248)
        
        # Applying the binary operator '+' (line 1090)
        result_add_37249 = python_operator(stypy.reporting.localization.Localization(__file__, 1090, 64), '+', result_add_37246, tuple_37247)
        
        # Processing the call keyword arguments (line 1089)
        kwargs_37250 = {}
        # Getting the type of 'self' (line 1089)
        self_37234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 30), 'self', False)
        # Obtaining the member 'runner' of a type (line 1089)
        runner_37235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 30), self_37234, 'runner')
        # Calling runner(args, kwargs) (line 1089)
        runner_call_result_37251 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 30), runner_37235, *[result_add_37249], **kwargs_37250)
        
        # Obtaining the member '__getitem__' of a type (line 1089)
        getitem___37252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 8), runner_call_result_37251, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1089)
        subscript_call_result_37253 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 8), getitem___37252, int_37233)
        
        # Assigning a type to the variable 'tuple_var_assignment_35478' (line 1089)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'tuple_var_assignment_35478', subscript_call_result_37253)
        
        # Assigning a Subscript to a Name (line 1089):
        
        # Obtaining the type of the subscript
        int_37254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 8), 'int')
        
        # Call to runner(...): (line 1089)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1089)
        tuple_37257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1089)
        # Adding element type (line 1089)
        # Getting the type of 'f' (line 1089)
        f_37258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 45), 'f', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37257, f_37258)
        # Adding element type (line 1089)
        # Getting the type of 't0' (line 1089)
        t0_37259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 48), 't0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37257, t0_37259)
        # Adding element type (line 1089)
        # Getting the type of 'y0' (line 1089)
        y0_37260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 52), 'y0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37257, y0_37260)
        # Adding element type (line 1089)
        # Getting the type of 't1' (line 1089)
        t1_37261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 56), 't1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37257, t1_37261)
        
        
        # Call to tuple(...): (line 1090)
        # Processing the call arguments (line 1090)
        # Getting the type of 'self' (line 1090)
        self_37263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 48), 'self', False)
        # Obtaining the member 'call_args' of a type (line 1090)
        call_args_37264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1090, 48), self_37263, 'call_args')
        # Processing the call keyword arguments (line 1090)
        kwargs_37265 = {}
        # Getting the type of 'tuple' (line 1090)
        tuple_37262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 42), 'tuple', False)
        # Calling tuple(args, kwargs) (line 1090)
        tuple_call_result_37266 = invoke(stypy.reporting.localization.Localization(__file__, 1090, 42), tuple_37262, *[call_args_37264], **kwargs_37265)
        
        # Applying the binary operator '+' (line 1089)
        result_add_37267 = python_operator(stypy.reporting.localization.Localization(__file__, 1089, 44), '+', tuple_37257, tuple_call_result_37266)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 1090)
        tuple_37268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1090, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1090)
        # Adding element type (line 1090)
        # Getting the type of 'f_params' (line 1090)
        f_params_37269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 67), 'f_params', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1090, 67), tuple_37268, f_params_37269)
        
        # Applying the binary operator '+' (line 1090)
        result_add_37270 = python_operator(stypy.reporting.localization.Localization(__file__, 1090, 64), '+', result_add_37267, tuple_37268)
        
        # Processing the call keyword arguments (line 1089)
        kwargs_37271 = {}
        # Getting the type of 'self' (line 1089)
        self_37255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 30), 'self', False)
        # Obtaining the member 'runner' of a type (line 1089)
        runner_37256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 30), self_37255, 'runner')
        # Calling runner(args, kwargs) (line 1089)
        runner_call_result_37272 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 30), runner_37256, *[result_add_37270], **kwargs_37271)
        
        # Obtaining the member '__getitem__' of a type (line 1089)
        getitem___37273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 8), runner_call_result_37272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1089)
        subscript_call_result_37274 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 8), getitem___37273, int_37254)
        
        # Assigning a type to the variable 'tuple_var_assignment_35479' (line 1089)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'tuple_var_assignment_35479', subscript_call_result_37274)
        
        # Assigning a Subscript to a Name (line 1089):
        
        # Obtaining the type of the subscript
        int_37275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 8), 'int')
        
        # Call to runner(...): (line 1089)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1089)
        tuple_37278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1089)
        # Adding element type (line 1089)
        # Getting the type of 'f' (line 1089)
        f_37279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 45), 'f', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37278, f_37279)
        # Adding element type (line 1089)
        # Getting the type of 't0' (line 1089)
        t0_37280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 48), 't0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37278, t0_37280)
        # Adding element type (line 1089)
        # Getting the type of 'y0' (line 1089)
        y0_37281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 52), 'y0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37278, y0_37281)
        # Adding element type (line 1089)
        # Getting the type of 't1' (line 1089)
        t1_37282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 56), 't1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37278, t1_37282)
        
        
        # Call to tuple(...): (line 1090)
        # Processing the call arguments (line 1090)
        # Getting the type of 'self' (line 1090)
        self_37284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 48), 'self', False)
        # Obtaining the member 'call_args' of a type (line 1090)
        call_args_37285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1090, 48), self_37284, 'call_args')
        # Processing the call keyword arguments (line 1090)
        kwargs_37286 = {}
        # Getting the type of 'tuple' (line 1090)
        tuple_37283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 42), 'tuple', False)
        # Calling tuple(args, kwargs) (line 1090)
        tuple_call_result_37287 = invoke(stypy.reporting.localization.Localization(__file__, 1090, 42), tuple_37283, *[call_args_37285], **kwargs_37286)
        
        # Applying the binary operator '+' (line 1089)
        result_add_37288 = python_operator(stypy.reporting.localization.Localization(__file__, 1089, 44), '+', tuple_37278, tuple_call_result_37287)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 1090)
        tuple_37289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1090, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1090)
        # Adding element type (line 1090)
        # Getting the type of 'f_params' (line 1090)
        f_params_37290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 67), 'f_params', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1090, 67), tuple_37289, f_params_37290)
        
        # Applying the binary operator '+' (line 1090)
        result_add_37291 = python_operator(stypy.reporting.localization.Localization(__file__, 1090, 64), '+', result_add_37288, tuple_37289)
        
        # Processing the call keyword arguments (line 1089)
        kwargs_37292 = {}
        # Getting the type of 'self' (line 1089)
        self_37276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 30), 'self', False)
        # Obtaining the member 'runner' of a type (line 1089)
        runner_37277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 30), self_37276, 'runner')
        # Calling runner(args, kwargs) (line 1089)
        runner_call_result_37293 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 30), runner_37277, *[result_add_37291], **kwargs_37292)
        
        # Obtaining the member '__getitem__' of a type (line 1089)
        getitem___37294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 8), runner_call_result_37293, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1089)
        subscript_call_result_37295 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 8), getitem___37294, int_37275)
        
        # Assigning a type to the variable 'tuple_var_assignment_35480' (line 1089)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'tuple_var_assignment_35480', subscript_call_result_37295)
        
        # Assigning a Subscript to a Name (line 1089):
        
        # Obtaining the type of the subscript
        int_37296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 8), 'int')
        
        # Call to runner(...): (line 1089)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1089)
        tuple_37299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1089)
        # Adding element type (line 1089)
        # Getting the type of 'f' (line 1089)
        f_37300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 45), 'f', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37299, f_37300)
        # Adding element type (line 1089)
        # Getting the type of 't0' (line 1089)
        t0_37301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 48), 't0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37299, t0_37301)
        # Adding element type (line 1089)
        # Getting the type of 'y0' (line 1089)
        y0_37302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 52), 'y0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37299, y0_37302)
        # Adding element type (line 1089)
        # Getting the type of 't1' (line 1089)
        t1_37303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 56), 't1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 45), tuple_37299, t1_37303)
        
        
        # Call to tuple(...): (line 1090)
        # Processing the call arguments (line 1090)
        # Getting the type of 'self' (line 1090)
        self_37305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 48), 'self', False)
        # Obtaining the member 'call_args' of a type (line 1090)
        call_args_37306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1090, 48), self_37305, 'call_args')
        # Processing the call keyword arguments (line 1090)
        kwargs_37307 = {}
        # Getting the type of 'tuple' (line 1090)
        tuple_37304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 42), 'tuple', False)
        # Calling tuple(args, kwargs) (line 1090)
        tuple_call_result_37308 = invoke(stypy.reporting.localization.Localization(__file__, 1090, 42), tuple_37304, *[call_args_37306], **kwargs_37307)
        
        # Applying the binary operator '+' (line 1089)
        result_add_37309 = python_operator(stypy.reporting.localization.Localization(__file__, 1089, 44), '+', tuple_37299, tuple_call_result_37308)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 1090)
        tuple_37310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1090, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1090)
        # Adding element type (line 1090)
        # Getting the type of 'f_params' (line 1090)
        f_params_37311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 67), 'f_params', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1090, 67), tuple_37310, f_params_37311)
        
        # Applying the binary operator '+' (line 1090)
        result_add_37312 = python_operator(stypy.reporting.localization.Localization(__file__, 1090, 64), '+', result_add_37309, tuple_37310)
        
        # Processing the call keyword arguments (line 1089)
        kwargs_37313 = {}
        # Getting the type of 'self' (line 1089)
        self_37297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 30), 'self', False)
        # Obtaining the member 'runner' of a type (line 1089)
        runner_37298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 30), self_37297, 'runner')
        # Calling runner(args, kwargs) (line 1089)
        runner_call_result_37314 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 30), runner_37298, *[result_add_37312], **kwargs_37313)
        
        # Obtaining the member '__getitem__' of a type (line 1089)
        getitem___37315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 8), runner_call_result_37314, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1089)
        subscript_call_result_37316 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 8), getitem___37315, int_37296)
        
        # Assigning a type to the variable 'tuple_var_assignment_35481' (line 1089)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'tuple_var_assignment_35481', subscript_call_result_37316)
        
        # Assigning a Name to a Name (line 1089):
        # Getting the type of 'tuple_var_assignment_35478' (line 1089)
        tuple_var_assignment_35478_37317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'tuple_var_assignment_35478')
        # Assigning a type to the variable 'x' (line 1089)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'x', tuple_var_assignment_35478_37317)
        
        # Assigning a Name to a Name (line 1089):
        # Getting the type of 'tuple_var_assignment_35479' (line 1089)
        tuple_var_assignment_35479_37318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'tuple_var_assignment_35479')
        # Assigning a type to the variable 'y' (line 1089)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 11), 'y', tuple_var_assignment_35479_37318)
        
        # Assigning a Name to a Name (line 1089):
        # Getting the type of 'tuple_var_assignment_35480' (line 1089)
        tuple_var_assignment_35480_37319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'tuple_var_assignment_35480')
        # Assigning a type to the variable 'iwork' (line 1089)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 14), 'iwork', tuple_var_assignment_35480_37319)
        
        # Assigning a Name to a Name (line 1089):
        # Getting the type of 'tuple_var_assignment_35481' (line 1089)
        tuple_var_assignment_35481_37320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'tuple_var_assignment_35481')
        # Assigning a type to the variable 'istate' (line 1089)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 21), 'istate', tuple_var_assignment_35481_37320)
        
        # Assigning a Name to a Attribute (line 1091):
        
        # Assigning a Name to a Attribute (line 1091):
        # Getting the type of 'istate' (line 1091)
        istate_37321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 22), 'istate')
        # Getting the type of 'self' (line 1091)
        self_37322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 8), 'self')
        # Setting the type of the member 'istate' of a type (line 1091)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1091, 8), self_37322, 'istate', istate_37321)
        
        
        # Getting the type of 'istate' (line 1092)
        istate_37323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 11), 'istate')
        int_37324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, 20), 'int')
        # Applying the binary operator '<' (line 1092)
        result_lt_37325 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 11), '<', istate_37323, int_37324)
        
        # Testing the type of an if condition (line 1092)
        if_condition_37326 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1092, 8), result_lt_37325)
        # Assigning a type to the variable 'if_condition_37326' (line 1092)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1092, 8), 'if_condition_37326', if_condition_37326)
        # SSA begins for if statement (line 1092)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1093):
        
        # Assigning a Call to a Name (line 1093):
        
        # Call to format(...): (line 1093)
        # Processing the call arguments (line 1093)
        # Getting the type of 'istate' (line 1093)
        istate_37329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 68), 'istate', False)
        # Processing the call keyword arguments (line 1093)
        kwargs_37330 = {}
        str_37327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1093, 36), 'str', 'Unexpected istate={:d}')
        # Obtaining the member 'format' of a type (line 1093)
        format_37328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1093, 36), str_37327, 'format')
        # Calling format(args, kwargs) (line 1093)
        format_call_result_37331 = invoke(stypy.reporting.localization.Localization(__file__, 1093, 36), format_37328, *[istate_37329], **kwargs_37330)
        
        # Assigning a type to the variable 'unexpected_istate_msg' (line 1093)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 12), 'unexpected_istate_msg', format_call_result_37331)
        
        # Call to warn(...): (line 1094)
        # Processing the call arguments (line 1094)
        
        # Call to format(...): (line 1094)
        # Processing the call arguments (line 1094)
        # Getting the type of 'self' (line 1094)
        self_37336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 46), 'self', False)
        # Obtaining the member '__class__' of a type (line 1094)
        class___37337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 46), self_37336, '__class__')
        # Obtaining the member '__name__' of a type (line 1094)
        name___37338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 46), class___37337, '__name__')
        
        # Call to get(...): (line 1095)
        # Processing the call arguments (line 1095)
        # Getting the type of 'istate' (line 1095)
        istate_37342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 44), 'istate', False)
        # Getting the type of 'unexpected_istate_msg' (line 1095)
        unexpected_istate_msg_37343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 52), 'unexpected_istate_msg', False)
        # Processing the call keyword arguments (line 1095)
        kwargs_37344 = {}
        # Getting the type of 'self' (line 1095)
        self_37339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 26), 'self', False)
        # Obtaining the member 'messages' of a type (line 1095)
        messages_37340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1095, 26), self_37339, 'messages')
        # Obtaining the member 'get' of a type (line 1095)
        get_37341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1095, 26), messages_37340, 'get')
        # Calling get(args, kwargs) (line 1095)
        get_call_result_37345 = invoke(stypy.reporting.localization.Localization(__file__, 1095, 26), get_37341, *[istate_37342, unexpected_istate_msg_37343], **kwargs_37344)
        
        # Processing the call keyword arguments (line 1094)
        kwargs_37346 = {}
        str_37334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1094, 26), 'str', '{:s}: {:s}')
        # Obtaining the member 'format' of a type (line 1094)
        format_37335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 26), str_37334, 'format')
        # Calling format(args, kwargs) (line 1094)
        format_call_result_37347 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 26), format_37335, *[name___37338, get_call_result_37345], **kwargs_37346)
        
        # Processing the call keyword arguments (line 1094)
        kwargs_37348 = {}
        # Getting the type of 'warnings' (line 1094)
        warnings_37332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 1094)
        warn_37333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 12), warnings_37332, 'warn')
        # Calling warn(args, kwargs) (line 1094)
        warn_call_result_37349 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 12), warn_37333, *[format_call_result_37347], **kwargs_37348)
        
        
        # Assigning a Num to a Attribute (line 1096):
        
        # Assigning a Num to a Attribute (line 1096):
        int_37350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1096, 27), 'int')
        # Getting the type of 'self' (line 1096)
        self_37351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 12), 'self')
        # Setting the type of the member 'success' of a type (line 1096)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1096, 12), self_37351, 'success', int_37350)
        # SSA join for if statement (line 1092)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 1097)
        tuple_37352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1097, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1097)
        # Adding element type (line 1097)
        # Getting the type of 'y' (line 1097)
        y_37353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 15), 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1097, 15), tuple_37352, y_37353)
        # Adding element type (line 1097)
        # Getting the type of 'x' (line 1097)
        x_37354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 18), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1097, 15), tuple_37352, x_37354)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1097)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1097, 8), 'stypy_return_type', tuple_37352)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 1088)
        stypy_return_type_37355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37355)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_37355


    @norecursion
    def _solout(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_solout'
        module_type_store = module_type_store.open_function_context('_solout', 1099, 4, False)
        # Assigning a type to the variable 'self' (line 1100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dopri5._solout.__dict__.__setitem__('stypy_localization', localization)
        dopri5._solout.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dopri5._solout.__dict__.__setitem__('stypy_type_store', module_type_store)
        dopri5._solout.__dict__.__setitem__('stypy_function_name', 'dopri5._solout')
        dopri5._solout.__dict__.__setitem__('stypy_param_names_list', ['nr', 'xold', 'x', 'y', 'nd', 'icomp', 'con'])
        dopri5._solout.__dict__.__setitem__('stypy_varargs_param_name', None)
        dopri5._solout.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dopri5._solout.__dict__.__setitem__('stypy_call_defaults', defaults)
        dopri5._solout.__dict__.__setitem__('stypy_call_varargs', varargs)
        dopri5._solout.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dopri5._solout.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dopri5._solout', ['nr', 'xold', 'x', 'y', 'nd', 'icomp', 'con'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_solout', localization, ['nr', 'xold', 'x', 'y', 'nd', 'icomp', 'con'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_solout(...)' code ##################

        
        
        # Getting the type of 'self' (line 1100)
        self_37356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 11), 'self')
        # Obtaining the member 'solout' of a type (line 1100)
        solout_37357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1100, 11), self_37356, 'solout')
        # Getting the type of 'None' (line 1100)
        None_37358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 30), 'None')
        # Applying the binary operator 'isnot' (line 1100)
        result_is_not_37359 = python_operator(stypy.reporting.localization.Localization(__file__, 1100, 11), 'isnot', solout_37357, None_37358)
        
        # Testing the type of an if condition (line 1100)
        if_condition_37360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1100, 8), result_is_not_37359)
        # Assigning a type to the variable 'if_condition_37360' (line 1100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1100, 8), 'if_condition_37360', if_condition_37360)
        # SSA begins for if statement (line 1100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 1101)
        self_37361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 15), 'self')
        # Obtaining the member 'solout_cmplx' of a type (line 1101)
        solout_cmplx_37362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1101, 15), self_37361, 'solout_cmplx')
        # Testing the type of an if condition (line 1101)
        if_condition_37363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1101, 12), solout_cmplx_37362)
        # Assigning a type to the variable 'if_condition_37363' (line 1101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1101, 12), 'if_condition_37363', if_condition_37363)
        # SSA begins for if statement (line 1101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 1102):
        
        # Assigning a BinOp to a Name (line 1102):
        
        # Obtaining the type of the subscript
        int_37364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, 24), 'int')
        slice_37365 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1102, 20), None, None, int_37364)
        # Getting the type of 'y' (line 1102)
        y_37366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 20), 'y')
        # Obtaining the member '__getitem__' of a type (line 1102)
        getitem___37367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1102, 20), y_37366, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1102)
        subscript_call_result_37368 = invoke(stypy.reporting.localization.Localization(__file__, 1102, 20), getitem___37367, slice_37365)
        
        complex_37369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, 29), 'complex')
        
        # Obtaining the type of the subscript
        int_37370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, 36), 'int')
        int_37371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, 39), 'int')
        slice_37372 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1102, 34), int_37370, None, int_37371)
        # Getting the type of 'y' (line 1102)
        y_37373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 34), 'y')
        # Obtaining the member '__getitem__' of a type (line 1102)
        getitem___37374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1102, 34), y_37373, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1102)
        subscript_call_result_37375 = invoke(stypy.reporting.localization.Localization(__file__, 1102, 34), getitem___37374, slice_37372)
        
        # Applying the binary operator '*' (line 1102)
        result_mul_37376 = python_operator(stypy.reporting.localization.Localization(__file__, 1102, 29), '*', complex_37369, subscript_call_result_37375)
        
        # Applying the binary operator '+' (line 1102)
        result_add_37377 = python_operator(stypy.reporting.localization.Localization(__file__, 1102, 20), '+', subscript_call_result_37368, result_mul_37376)
        
        # Assigning a type to the variable 'y' (line 1102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1102, 16), 'y', result_add_37377)
        # SSA join for if statement (line 1101)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to solout(...): (line 1103)
        # Processing the call arguments (line 1103)
        # Getting the type of 'x' (line 1103)
        x_37380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 31), 'x', False)
        # Getting the type of 'y' (line 1103)
        y_37381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 34), 'y', False)
        # Processing the call keyword arguments (line 1103)
        kwargs_37382 = {}
        # Getting the type of 'self' (line 1103)
        self_37378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 19), 'self', False)
        # Obtaining the member 'solout' of a type (line 1103)
        solout_37379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1103, 19), self_37378, 'solout')
        # Calling solout(args, kwargs) (line 1103)
        solout_call_result_37383 = invoke(stypy.reporting.localization.Localization(__file__, 1103, 19), solout_37379, *[x_37380, y_37381], **kwargs_37382)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1103, 12), 'stypy_return_type', solout_call_result_37383)
        # SSA branch for the else part of an if statement (line 1100)
        module_type_store.open_ssa_branch('else')
        int_37384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1105, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 1105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1105, 12), 'stypy_return_type', int_37384)
        # SSA join for if statement (line 1100)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_solout(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_solout' in the type store
        # Getting the type of 'stypy_return_type' (line 1099)
        stypy_return_type_37385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37385)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_solout'
        return stypy_return_type_37385


# Assigning a type to the variable 'dopri5' (line 1025)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1025, 0), 'dopri5', dopri5)

# Assigning a Call to a Name (line 1026):

# Call to getattr(...): (line 1026)
# Processing the call arguments (line 1026)
# Getting the type of '_dop' (line 1026)
_dop_37387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 21), '_dop', False)
str_37388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1026, 27), 'str', 'dopri5')
# Getting the type of 'None' (line 1026)
None_37389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 37), 'None', False)
# Processing the call keyword arguments (line 1026)
kwargs_37390 = {}
# Getting the type of 'getattr' (line 1026)
getattr_37386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 13), 'getattr', False)
# Calling getattr(args, kwargs) (line 1026)
getattr_call_result_37391 = invoke(stypy.reporting.localization.Localization(__file__, 1026, 13), getattr_37386, *[_dop_37387, str_37388, None_37389], **kwargs_37390)

# Getting the type of 'dopri5'
dopri5_37392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dopri5')
# Setting the type of the member 'runner' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dopri5_37392, 'runner', getattr_call_result_37391)

# Assigning a Str to a Name (line 1027):
str_37393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1027, 11), 'str', 'dopri5')
# Getting the type of 'dopri5'
dopri5_37394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dopri5')
# Setting the type of the member 'name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dopri5_37394, 'name', str_37393)

# Assigning a Name to a Name (line 1028):
# Getting the type of 'True' (line 1028)
True_37395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 22), 'True')
# Getting the type of 'dopri5'
dopri5_37396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dopri5')
# Setting the type of the member 'supports_solout' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dopri5_37396, 'supports_solout', True_37395)

# Assigning a Dict to a Name (line 1030):

# Obtaining an instance of the builtin type 'dict' (line 1030)
dict_37397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1030)
# Adding element type (key, value) (line 1030)
int_37398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 16), 'int')
str_37399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 19), 'str', 'computation successful')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1030, 15), dict_37397, (int_37398, str_37399))
# Adding element type (key, value) (line 1030)
int_37400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 16), 'int')
str_37401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 19), 'str', 'comput. successful (interrupted by solout)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1030, 15), dict_37397, (int_37400, str_37401))
# Adding element type (key, value) (line 1030)
int_37402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 16), 'int')
str_37403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 20), 'str', 'input is not consistent')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1030, 15), dict_37397, (int_37402, str_37403))
# Adding element type (key, value) (line 1030)
int_37404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 16), 'int')
str_37405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 20), 'str', 'larger nmax is needed')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1030, 15), dict_37397, (int_37404, str_37405))
# Adding element type (key, value) (line 1030)
int_37406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 16), 'int')
str_37407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 20), 'str', 'step size becomes too small')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1030, 15), dict_37397, (int_37406, str_37407))
# Adding element type (key, value) (line 1030)
int_37408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 16), 'int')
str_37409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 20), 'str', 'problem is probably stiff (interrupted)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1030, 15), dict_37397, (int_37408, str_37409))

# Getting the type of 'dopri5'
dopri5_37410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dopri5')
# Setting the type of the member 'messages' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dopri5_37410, 'messages', dict_37397)


# Getting the type of 'dopri5' (line 1108)
dopri5_37411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 3), 'dopri5')
# Obtaining the member 'runner' of a type (line 1108)
runner_37412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1108, 3), dopri5_37411, 'runner')
# Getting the type of 'None' (line 1108)
None_37413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 24), 'None')
# Applying the binary operator 'isnot' (line 1108)
result_is_not_37414 = python_operator(stypy.reporting.localization.Localization(__file__, 1108, 3), 'isnot', runner_37412, None_37413)

# Testing the type of an if condition (line 1108)
if_condition_37415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1108, 0), result_is_not_37414)
# Assigning a type to the variable 'if_condition_37415' (line 1108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 0), 'if_condition_37415', if_condition_37415)
# SSA begins for if statement (line 1108)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to append(...): (line 1109)
# Processing the call arguments (line 1109)
# Getting the type of 'dopri5' (line 1109)
dopri5_37419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 45), 'dopri5', False)
# Processing the call keyword arguments (line 1109)
kwargs_37420 = {}
# Getting the type of 'IntegratorBase' (line 1109)
IntegratorBase_37416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 4), 'IntegratorBase', False)
# Obtaining the member 'integrator_classes' of a type (line 1109)
integrator_classes_37417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1109, 4), IntegratorBase_37416, 'integrator_classes')
# Obtaining the member 'append' of a type (line 1109)
append_37418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1109, 4), integrator_classes_37417, 'append')
# Calling append(args, kwargs) (line 1109)
append_call_result_37421 = invoke(stypy.reporting.localization.Localization(__file__, 1109, 4), append_37418, *[dopri5_37419], **kwargs_37420)

# SSA join for if statement (line 1108)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'dop853' class
# Getting the type of 'dopri5' (line 1112)
dopri5_37422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 13), 'dopri5')

class dop853(dopri5_37422, ):
    
    # Assigning a Call to a Name (line 1113):
    
    # Assigning a Str to a Name (line 1114):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_37423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 22), 'float')
        float_37424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 33), 'float')
        int_37425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1118, 24), 'int')
        float_37426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 26), 'float')
        float_37427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1120, 28), 'float')
        float_37428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1121, 24), 'float')
        float_37429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1122, 25), 'float')
        float_37430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1123, 25), 'float')
        float_37431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 22), 'float')
        # Getting the type of 'None' (line 1125)
        None_37432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 24), 'None')
        int_37433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 27), 'int')
        defaults = [float_37423, float_37424, int_37425, float_37426, float_37427, float_37428, float_37429, float_37430, float_37431, None_37432, int_37433]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1116, 4, False)
        # Assigning a type to the variable 'self' (line 1117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dop853.__init__', ['rtol', 'atol', 'nsteps', 'max_step', 'first_step', 'safety', 'ifactor', 'dfactor', 'beta', 'method', 'verbosity'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['rtol', 'atol', 'nsteps', 'max_step', 'first_step', 'safety', 'ifactor', 'dfactor', 'beta', 'method', 'verbosity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 1128)
        # Processing the call arguments (line 1128)
        # Getting the type of 'rtol' (line 1128)
        rtol_37441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 45), 'rtol', False)
        # Getting the type of 'atol' (line 1128)
        atol_37442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 51), 'atol', False)
        # Getting the type of 'nsteps' (line 1128)
        nsteps_37443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 57), 'nsteps', False)
        # Getting the type of 'max_step' (line 1128)
        max_step_37444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 65), 'max_step', False)
        # Getting the type of 'first_step' (line 1129)
        first_step_37445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 45), 'first_step', False)
        # Getting the type of 'safety' (line 1129)
        safety_37446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 57), 'safety', False)
        # Getting the type of 'ifactor' (line 1129)
        ifactor_37447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 65), 'ifactor', False)
        # Getting the type of 'dfactor' (line 1130)
        dfactor_37448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 45), 'dfactor', False)
        # Getting the type of 'beta' (line 1130)
        beta_37449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 54), 'beta', False)
        # Getting the type of 'method' (line 1130)
        method_37450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 60), 'method', False)
        # Getting the type of 'verbosity' (line 1131)
        verbosity_37451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 45), 'verbosity', False)
        # Processing the call keyword arguments (line 1128)
        kwargs_37452 = {}
        
        # Call to super(...): (line 1128)
        # Processing the call arguments (line 1128)
        # Getting the type of 'self' (line 1128)
        self_37435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 14), 'self', False)
        # Obtaining the member '__class__' of a type (line 1128)
        class___37436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1128, 14), self_37435, '__class__')
        # Getting the type of 'self' (line 1128)
        self_37437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 30), 'self', False)
        # Processing the call keyword arguments (line 1128)
        kwargs_37438 = {}
        # Getting the type of 'super' (line 1128)
        super_37434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 8), 'super', False)
        # Calling super(args, kwargs) (line 1128)
        super_call_result_37439 = invoke(stypy.reporting.localization.Localization(__file__, 1128, 8), super_37434, *[class___37436, self_37437], **kwargs_37438)
        
        # Obtaining the member '__init__' of a type (line 1128)
        init___37440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1128, 8), super_call_result_37439, '__init__')
        # Calling __init__(args, kwargs) (line 1128)
        init___call_result_37453 = invoke(stypy.reporting.localization.Localization(__file__, 1128, 8), init___37440, *[rtol_37441, atol_37442, nsteps_37443, max_step_37444, first_step_37445, safety_37446, ifactor_37447, dfactor_37448, beta_37449, method_37450, verbosity_37451], **kwargs_37452)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def reset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reset'
        module_type_store = module_type_store.open_function_context('reset', 1133, 4, False)
        # Assigning a type to the variable 'self' (line 1134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dop853.reset.__dict__.__setitem__('stypy_localization', localization)
        dop853.reset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dop853.reset.__dict__.__setitem__('stypy_type_store', module_type_store)
        dop853.reset.__dict__.__setitem__('stypy_function_name', 'dop853.reset')
        dop853.reset.__dict__.__setitem__('stypy_param_names_list', ['n', 'has_jac'])
        dop853.reset.__dict__.__setitem__('stypy_varargs_param_name', None)
        dop853.reset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dop853.reset.__dict__.__setitem__('stypy_call_defaults', defaults)
        dop853.reset.__dict__.__setitem__('stypy_call_varargs', varargs)
        dop853.reset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dop853.reset.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dop853.reset', ['n', 'has_jac'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reset', localization, ['n', 'has_jac'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reset(...)' code ##################

        
        # Assigning a Call to a Name (line 1134):
        
        # Assigning a Call to a Name (line 1134):
        
        # Call to zeros(...): (line 1134)
        # Processing the call arguments (line 1134)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1134)
        tuple_37455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1134, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1134)
        # Adding element type (line 1134)
        int_37456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1134, 22), 'int')
        # Getting the type of 'n' (line 1134)
        n_37457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 27), 'n', False)
        # Applying the binary operator '*' (line 1134)
        result_mul_37458 = python_operator(stypy.reporting.localization.Localization(__file__, 1134, 22), '*', int_37456, n_37457)
        
        int_37459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1134, 31), 'int')
        # Applying the binary operator '+' (line 1134)
        result_add_37460 = python_operator(stypy.reporting.localization.Localization(__file__, 1134, 22), '+', result_mul_37458, int_37459)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1134, 22), tuple_37455, result_add_37460)
        
        # Getting the type of 'float' (line 1134)
        float_37461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 37), 'float', False)
        # Processing the call keyword arguments (line 1134)
        kwargs_37462 = {}
        # Getting the type of 'zeros' (line 1134)
        zeros_37454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 15), 'zeros', False)
        # Calling zeros(args, kwargs) (line 1134)
        zeros_call_result_37463 = invoke(stypy.reporting.localization.Localization(__file__, 1134, 15), zeros_37454, *[tuple_37455, float_37461], **kwargs_37462)
        
        # Assigning a type to the variable 'work' (line 1134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1134, 8), 'work', zeros_call_result_37463)
        
        # Assigning a Attribute to a Subscript (line 1135):
        
        # Assigning a Attribute to a Subscript (line 1135):
        # Getting the type of 'self' (line 1135)
        self_37464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 18), 'self')
        # Obtaining the member 'safety' of a type (line 1135)
        safety_37465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1135, 18), self_37464, 'safety')
        # Getting the type of 'work' (line 1135)
        work_37466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 8), 'work')
        int_37467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, 13), 'int')
        # Storing an element on a container (line 1135)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 8), work_37466, (int_37467, safety_37465))
        
        # Assigning a Attribute to a Subscript (line 1136):
        
        # Assigning a Attribute to a Subscript (line 1136):
        # Getting the type of 'self' (line 1136)
        self_37468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 18), 'self')
        # Obtaining the member 'dfactor' of a type (line 1136)
        dfactor_37469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1136, 18), self_37468, 'dfactor')
        # Getting the type of 'work' (line 1136)
        work_37470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 8), 'work')
        int_37471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1136, 13), 'int')
        # Storing an element on a container (line 1136)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1136, 8), work_37470, (int_37471, dfactor_37469))
        
        # Assigning a Attribute to a Subscript (line 1137):
        
        # Assigning a Attribute to a Subscript (line 1137):
        # Getting the type of 'self' (line 1137)
        self_37472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 18), 'self')
        # Obtaining the member 'ifactor' of a type (line 1137)
        ifactor_37473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1137, 18), self_37472, 'ifactor')
        # Getting the type of 'work' (line 1137)
        work_37474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 8), 'work')
        int_37475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1137, 13), 'int')
        # Storing an element on a container (line 1137)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1137, 8), work_37474, (int_37475, ifactor_37473))
        
        # Assigning a Attribute to a Subscript (line 1138):
        
        # Assigning a Attribute to a Subscript (line 1138):
        # Getting the type of 'self' (line 1138)
        self_37476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 18), 'self')
        # Obtaining the member 'beta' of a type (line 1138)
        beta_37477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1138, 18), self_37476, 'beta')
        # Getting the type of 'work' (line 1138)
        work_37478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 8), 'work')
        int_37479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1138, 13), 'int')
        # Storing an element on a container (line 1138)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1138, 8), work_37478, (int_37479, beta_37477))
        
        # Assigning a Attribute to a Subscript (line 1139):
        
        # Assigning a Attribute to a Subscript (line 1139):
        # Getting the type of 'self' (line 1139)
        self_37480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 18), 'self')
        # Obtaining the member 'max_step' of a type (line 1139)
        max_step_37481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1139, 18), self_37480, 'max_step')
        # Getting the type of 'work' (line 1139)
        work_37482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 8), 'work')
        int_37483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 13), 'int')
        # Storing an element on a container (line 1139)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1139, 8), work_37482, (int_37483, max_step_37481))
        
        # Assigning a Attribute to a Subscript (line 1140):
        
        # Assigning a Attribute to a Subscript (line 1140):
        # Getting the type of 'self' (line 1140)
        self_37484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 18), 'self')
        # Obtaining the member 'first_step' of a type (line 1140)
        first_step_37485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1140, 18), self_37484, 'first_step')
        # Getting the type of 'work' (line 1140)
        work_37486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 8), 'work')
        int_37487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1140, 13), 'int')
        # Storing an element on a container (line 1140)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1140, 8), work_37486, (int_37487, first_step_37485))
        
        # Assigning a Name to a Attribute (line 1141):
        
        # Assigning a Name to a Attribute (line 1141):
        # Getting the type of 'work' (line 1141)
        work_37488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 20), 'work')
        # Getting the type of 'self' (line 1141)
        self_37489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 8), 'self')
        # Setting the type of the member 'work' of a type (line 1141)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1141, 8), self_37489, 'work', work_37488)
        
        # Assigning a Call to a Name (line 1142):
        
        # Assigning a Call to a Name (line 1142):
        
        # Call to zeros(...): (line 1142)
        # Processing the call arguments (line 1142)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1142)
        tuple_37491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1142, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1142)
        # Adding element type (line 1142)
        int_37492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1142, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1142, 23), tuple_37491, int_37492)
        
        # Getting the type of 'int32' (line 1142)
        int32_37493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 29), 'int32', False)
        # Processing the call keyword arguments (line 1142)
        kwargs_37494 = {}
        # Getting the type of 'zeros' (line 1142)
        zeros_37490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 1142)
        zeros_call_result_37495 = invoke(stypy.reporting.localization.Localization(__file__, 1142, 16), zeros_37490, *[tuple_37491, int32_37493], **kwargs_37494)
        
        # Assigning a type to the variable 'iwork' (line 1142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1142, 8), 'iwork', zeros_call_result_37495)
        
        # Assigning a Attribute to a Subscript (line 1143):
        
        # Assigning a Attribute to a Subscript (line 1143):
        # Getting the type of 'self' (line 1143)
        self_37496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 19), 'self')
        # Obtaining the member 'nsteps' of a type (line 1143)
        nsteps_37497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1143, 19), self_37496, 'nsteps')
        # Getting the type of 'iwork' (line 1143)
        iwork_37498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 8), 'iwork')
        int_37499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1143, 14), 'int')
        # Storing an element on a container (line 1143)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1143, 8), iwork_37498, (int_37499, nsteps_37497))
        
        # Assigning a Attribute to a Subscript (line 1144):
        
        # Assigning a Attribute to a Subscript (line 1144):
        # Getting the type of 'self' (line 1144)
        self_37500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 19), 'self')
        # Obtaining the member 'verbosity' of a type (line 1144)
        verbosity_37501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1144, 19), self_37500, 'verbosity')
        # Getting the type of 'iwork' (line 1144)
        iwork_37502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 8), 'iwork')
        int_37503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1144, 14), 'int')
        # Storing an element on a container (line 1144)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1144, 8), iwork_37502, (int_37503, verbosity_37501))
        
        # Assigning a Name to a Attribute (line 1145):
        
        # Assigning a Name to a Attribute (line 1145):
        # Getting the type of 'iwork' (line 1145)
        iwork_37504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 21), 'iwork')
        # Getting the type of 'self' (line 1145)
        self_37505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 8), 'self')
        # Setting the type of the member 'iwork' of a type (line 1145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1145, 8), self_37505, 'iwork', iwork_37504)
        
        # Assigning a List to a Attribute (line 1146):
        
        # Assigning a List to a Attribute (line 1146):
        
        # Obtaining an instance of the builtin type 'list' (line 1146)
        list_37506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1146, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1146)
        # Adding element type (line 1146)
        # Getting the type of 'self' (line 1146)
        self_37507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 26), 'self')
        # Obtaining the member 'rtol' of a type (line 1146)
        rtol_37508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1146, 26), self_37507, 'rtol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1146, 25), list_37506, rtol_37508)
        # Adding element type (line 1146)
        # Getting the type of 'self' (line 1146)
        self_37509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 37), 'self')
        # Obtaining the member 'atol' of a type (line 1146)
        atol_37510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1146, 37), self_37509, 'atol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1146, 25), list_37506, atol_37510)
        # Adding element type (line 1146)
        # Getting the type of 'self' (line 1146)
        self_37511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 48), 'self')
        # Obtaining the member '_solout' of a type (line 1146)
        _solout_37512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1146, 48), self_37511, '_solout')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1146, 25), list_37506, _solout_37512)
        # Adding element type (line 1146)
        # Getting the type of 'self' (line 1147)
        self_37513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 26), 'self')
        # Obtaining the member 'iout' of a type (line 1147)
        iout_37514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1147, 26), self_37513, 'iout')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1146, 25), list_37506, iout_37514)
        # Adding element type (line 1146)
        # Getting the type of 'self' (line 1147)
        self_37515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 37), 'self')
        # Obtaining the member 'work' of a type (line 1147)
        work_37516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1147, 37), self_37515, 'work')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1146, 25), list_37506, work_37516)
        # Adding element type (line 1146)
        # Getting the type of 'self' (line 1147)
        self_37517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 48), 'self')
        # Obtaining the member 'iwork' of a type (line 1147)
        iwork_37518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1147, 48), self_37517, 'iwork')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1146, 25), list_37506, iwork_37518)
        
        # Getting the type of 'self' (line 1146)
        self_37519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'self')
        # Setting the type of the member 'call_args' of a type (line 1146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1146, 8), self_37519, 'call_args', list_37506)
        
        # Assigning a Num to a Attribute (line 1148):
        
        # Assigning a Num to a Attribute (line 1148):
        int_37520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1148, 23), 'int')
        # Getting the type of 'self' (line 1148)
        self_37521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 8), 'self')
        # Setting the type of the member 'success' of a type (line 1148)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1148, 8), self_37521, 'success', int_37520)
        
        # ################# End of 'reset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset' in the type store
        # Getting the type of 'stypy_return_type' (line 1133)
        stypy_return_type_37522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37522)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset'
        return stypy_return_type_37522


# Assigning a type to the variable 'dop853' (line 1112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1112, 0), 'dop853', dop853)

# Assigning a Call to a Name (line 1113):

# Call to getattr(...): (line 1113)
# Processing the call arguments (line 1113)
# Getting the type of '_dop' (line 1113)
_dop_37524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 21), '_dop', False)
str_37525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1113, 27), 'str', 'dop853')
# Getting the type of 'None' (line 1113)
None_37526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 37), 'None', False)
# Processing the call keyword arguments (line 1113)
kwargs_37527 = {}
# Getting the type of 'getattr' (line 1113)
getattr_37523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 13), 'getattr', False)
# Calling getattr(args, kwargs) (line 1113)
getattr_call_result_37528 = invoke(stypy.reporting.localization.Localization(__file__, 1113, 13), getattr_37523, *[_dop_37524, str_37525, None_37526], **kwargs_37527)

# Getting the type of 'dop853'
dop853_37529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dop853')
# Setting the type of the member 'runner' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dop853_37529, 'runner', getattr_call_result_37528)

# Assigning a Str to a Name (line 1114):
str_37530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1114, 11), 'str', 'dop853')
# Getting the type of 'dop853'
dop853_37531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dop853')
# Setting the type of the member 'name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dop853_37531, 'name', str_37530)


# Getting the type of 'dop853' (line 1151)
dop853_37532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 3), 'dop853')
# Obtaining the member 'runner' of a type (line 1151)
runner_37533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1151, 3), dop853_37532, 'runner')
# Getting the type of 'None' (line 1151)
None_37534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 24), 'None')
# Applying the binary operator 'isnot' (line 1151)
result_is_not_37535 = python_operator(stypy.reporting.localization.Localization(__file__, 1151, 3), 'isnot', runner_37533, None_37534)

# Testing the type of an if condition (line 1151)
if_condition_37536 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1151, 0), result_is_not_37535)
# Assigning a type to the variable 'if_condition_37536' (line 1151)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1151, 0), 'if_condition_37536', if_condition_37536)
# SSA begins for if statement (line 1151)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to append(...): (line 1152)
# Processing the call arguments (line 1152)
# Getting the type of 'dop853' (line 1152)
dop853_37540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 45), 'dop853', False)
# Processing the call keyword arguments (line 1152)
kwargs_37541 = {}
# Getting the type of 'IntegratorBase' (line 1152)
IntegratorBase_37537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 4), 'IntegratorBase', False)
# Obtaining the member 'integrator_classes' of a type (line 1152)
integrator_classes_37538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1152, 4), IntegratorBase_37537, 'integrator_classes')
# Obtaining the member 'append' of a type (line 1152)
append_37539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1152, 4), integrator_classes_37538, 'append')
# Calling append(args, kwargs) (line 1152)
append_call_result_37542 = invoke(stypy.reporting.localization.Localization(__file__, 1152, 4), append_37539, *[dop853_37540], **kwargs_37541)

# SSA join for if statement (line 1151)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'lsoda' class
# Getting the type of 'IntegratorBase' (line 1155)
IntegratorBase_37543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 12), 'IntegratorBase')

class lsoda(IntegratorBase_37543, ):
    
    # Assigning a Call to a Name (line 1156):
    
    # Assigning a Num to a Name (line 1157):
    
    # Assigning a Dict to a Name (line 1159):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 1171)
        False_37544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 31), 'False')
        float_37545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1172, 22), 'float')
        float_37546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1172, 33), 'float')
        # Getting the type of 'None' (line 1173)
        None_37547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 23), 'None')
        # Getting the type of 'None' (line 1173)
        None_37548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 35), 'None')
        int_37549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1174, 24), 'int')
        float_37550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1175, 26), 'float')
        float_37551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1176, 26), 'float')
        float_37552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1177, 28), 'float')
        int_37553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1178, 22), 'int')
        int_37554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1179, 26), 'int')
        int_37555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1180, 30), 'int')
        int_37556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1181, 29), 'int')
        # Getting the type of 'None' (line 1182)
        None_37557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 24), 'None')
        defaults = [False_37544, float_37545, float_37546, None_37547, None_37548, int_37549, float_37550, float_37551, float_37552, int_37553, int_37554, int_37555, int_37556, None_37557]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1170, 4, False)
        # Assigning a type to the variable 'self' (line 1171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lsoda.__init__', ['with_jacobian', 'rtol', 'atol', 'lband', 'uband', 'nsteps', 'max_step', 'min_step', 'first_step', 'ixpr', 'max_hnil', 'max_order_ns', 'max_order_s', 'method'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['with_jacobian', 'rtol', 'atol', 'lband', 'uband', 'nsteps', 'max_step', 'min_step', 'first_step', 'ixpr', 'max_hnil', 'max_order_ns', 'max_order_s', 'method'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 1185):
        
        # Assigning a Name to a Attribute (line 1185):
        # Getting the type of 'with_jacobian' (line 1185)
        with_jacobian_37558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 29), 'with_jacobian')
        # Getting the type of 'self' (line 1185)
        self_37559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 8), 'self')
        # Setting the type of the member 'with_jacobian' of a type (line 1185)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1185, 8), self_37559, 'with_jacobian', with_jacobian_37558)
        
        # Assigning a Name to a Attribute (line 1186):
        
        # Assigning a Name to a Attribute (line 1186):
        # Getting the type of 'rtol' (line 1186)
        rtol_37560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 20), 'rtol')
        # Getting the type of 'self' (line 1186)
        self_37561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 8), 'self')
        # Setting the type of the member 'rtol' of a type (line 1186)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1186, 8), self_37561, 'rtol', rtol_37560)
        
        # Assigning a Name to a Attribute (line 1187):
        
        # Assigning a Name to a Attribute (line 1187):
        # Getting the type of 'atol' (line 1187)
        atol_37562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 20), 'atol')
        # Getting the type of 'self' (line 1187)
        self_37563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 8), 'self')
        # Setting the type of the member 'atol' of a type (line 1187)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1187, 8), self_37563, 'atol', atol_37562)
        
        # Assigning a Name to a Attribute (line 1188):
        
        # Assigning a Name to a Attribute (line 1188):
        # Getting the type of 'uband' (line 1188)
        uband_37564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 18), 'uband')
        # Getting the type of 'self' (line 1188)
        self_37565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'self')
        # Setting the type of the member 'mu' of a type (line 1188)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1188, 8), self_37565, 'mu', uband_37564)
        
        # Assigning a Name to a Attribute (line 1189):
        
        # Assigning a Name to a Attribute (line 1189):
        # Getting the type of 'lband' (line 1189)
        lband_37566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 18), 'lband')
        # Getting the type of 'self' (line 1189)
        self_37567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 8), 'self')
        # Setting the type of the member 'ml' of a type (line 1189)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1189, 8), self_37567, 'ml', lband_37566)
        
        # Assigning a Name to a Attribute (line 1191):
        
        # Assigning a Name to a Attribute (line 1191):
        # Getting the type of 'max_order_ns' (line 1191)
        max_order_ns_37568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 28), 'max_order_ns')
        # Getting the type of 'self' (line 1191)
        self_37569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 8), 'self')
        # Setting the type of the member 'max_order_ns' of a type (line 1191)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1191, 8), self_37569, 'max_order_ns', max_order_ns_37568)
        
        # Assigning a Name to a Attribute (line 1192):
        
        # Assigning a Name to a Attribute (line 1192):
        # Getting the type of 'max_order_s' (line 1192)
        max_order_s_37570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 27), 'max_order_s')
        # Getting the type of 'self' (line 1192)
        self_37571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 8), 'self')
        # Setting the type of the member 'max_order_s' of a type (line 1192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1192, 8), self_37571, 'max_order_s', max_order_s_37570)
        
        # Assigning a Name to a Attribute (line 1193):
        
        # Assigning a Name to a Attribute (line 1193):
        # Getting the type of 'nsteps' (line 1193)
        nsteps_37572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 22), 'nsteps')
        # Getting the type of 'self' (line 1193)
        self_37573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 8), 'self')
        # Setting the type of the member 'nsteps' of a type (line 1193)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1193, 8), self_37573, 'nsteps', nsteps_37572)
        
        # Assigning a Name to a Attribute (line 1194):
        
        # Assigning a Name to a Attribute (line 1194):
        # Getting the type of 'max_step' (line 1194)
        max_step_37574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 24), 'max_step')
        # Getting the type of 'self' (line 1194)
        self_37575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 8), 'self')
        # Setting the type of the member 'max_step' of a type (line 1194)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1194, 8), self_37575, 'max_step', max_step_37574)
        
        # Assigning a Name to a Attribute (line 1195):
        
        # Assigning a Name to a Attribute (line 1195):
        # Getting the type of 'min_step' (line 1195)
        min_step_37576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 24), 'min_step')
        # Getting the type of 'self' (line 1195)
        self_37577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 8), 'self')
        # Setting the type of the member 'min_step' of a type (line 1195)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1195, 8), self_37577, 'min_step', min_step_37576)
        
        # Assigning a Name to a Attribute (line 1196):
        
        # Assigning a Name to a Attribute (line 1196):
        # Getting the type of 'first_step' (line 1196)
        first_step_37578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 26), 'first_step')
        # Getting the type of 'self' (line 1196)
        self_37579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 8), 'self')
        # Setting the type of the member 'first_step' of a type (line 1196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1196, 8), self_37579, 'first_step', first_step_37578)
        
        # Assigning a Name to a Attribute (line 1197):
        
        # Assigning a Name to a Attribute (line 1197):
        # Getting the type of 'ixpr' (line 1197)
        ixpr_37580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 20), 'ixpr')
        # Getting the type of 'self' (line 1197)
        self_37581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 8), 'self')
        # Setting the type of the member 'ixpr' of a type (line 1197)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1197, 8), self_37581, 'ixpr', ixpr_37580)
        
        # Assigning a Name to a Attribute (line 1198):
        
        # Assigning a Name to a Attribute (line 1198):
        # Getting the type of 'max_hnil' (line 1198)
        max_hnil_37582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 24), 'max_hnil')
        # Getting the type of 'self' (line 1198)
        self_37583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 8), 'self')
        # Setting the type of the member 'max_hnil' of a type (line 1198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1198, 8), self_37583, 'max_hnil', max_hnil_37582)
        
        # Assigning a Num to a Attribute (line 1199):
        
        # Assigning a Num to a Attribute (line 1199):
        int_37584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1199, 23), 'int')
        # Getting the type of 'self' (line 1199)
        self_37585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 8), 'self')
        # Setting the type of the member 'success' of a type (line 1199)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1199, 8), self_37585, 'success', int_37584)
        
        # Assigning a Name to a Attribute (line 1201):
        
        # Assigning a Name to a Attribute (line 1201):
        # Getting the type of 'False' (line 1201)
        False_37586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 27), 'False')
        # Getting the type of 'self' (line 1201)
        self_37587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 8), 'self')
        # Setting the type of the member 'initialized' of a type (line 1201)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1201, 8), self_37587, 'initialized', False_37586)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def reset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reset'
        module_type_store = module_type_store.open_function_context('reset', 1203, 4, False)
        # Assigning a type to the variable 'self' (line 1204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1204, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lsoda.reset.__dict__.__setitem__('stypy_localization', localization)
        lsoda.reset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lsoda.reset.__dict__.__setitem__('stypy_type_store', module_type_store)
        lsoda.reset.__dict__.__setitem__('stypy_function_name', 'lsoda.reset')
        lsoda.reset.__dict__.__setitem__('stypy_param_names_list', ['n', 'has_jac'])
        lsoda.reset.__dict__.__setitem__('stypy_varargs_param_name', None)
        lsoda.reset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lsoda.reset.__dict__.__setitem__('stypy_call_defaults', defaults)
        lsoda.reset.__dict__.__setitem__('stypy_call_varargs', varargs)
        lsoda.reset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lsoda.reset.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lsoda.reset', ['n', 'has_jac'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reset', localization, ['n', 'has_jac'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reset(...)' code ##################

        
        # Getting the type of 'has_jac' (line 1205)
        has_jac_37588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 11), 'has_jac')
        # Testing the type of an if condition (line 1205)
        if_condition_37589 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1205, 8), has_jac_37588)
        # Assigning a type to the variable 'if_condition_37589' (line 1205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 8), 'if_condition_37589', if_condition_37589)
        # SSA begins for if statement (line 1205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 1206)
        self_37590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 15), 'self')
        # Obtaining the member 'mu' of a type (line 1206)
        mu_37591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1206, 15), self_37590, 'mu')
        # Getting the type of 'None' (line 1206)
        None_37592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 26), 'None')
        # Applying the binary operator 'is' (line 1206)
        result_is__37593 = python_operator(stypy.reporting.localization.Localization(__file__, 1206, 15), 'is', mu_37591, None_37592)
        
        
        # Getting the type of 'self' (line 1206)
        self_37594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 35), 'self')
        # Obtaining the member 'ml' of a type (line 1206)
        ml_37595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1206, 35), self_37594, 'ml')
        # Getting the type of 'None' (line 1206)
        None_37596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 46), 'None')
        # Applying the binary operator 'is' (line 1206)
        result_is__37597 = python_operator(stypy.reporting.localization.Localization(__file__, 1206, 35), 'is', ml_37595, None_37596)
        
        # Applying the binary operator 'and' (line 1206)
        result_and_keyword_37598 = python_operator(stypy.reporting.localization.Localization(__file__, 1206, 15), 'and', result_is__37593, result_is__37597)
        
        # Testing the type of an if condition (line 1206)
        if_condition_37599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1206, 12), result_and_keyword_37598)
        # Assigning a type to the variable 'if_condition_37599' (line 1206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1206, 12), 'if_condition_37599', if_condition_37599)
        # SSA begins for if statement (line 1206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 1207):
        
        # Assigning a Num to a Name (line 1207):
        int_37600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1207, 21), 'int')
        # Assigning a type to the variable 'jt' (line 1207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1207, 16), 'jt', int_37600)
        # SSA branch for the else part of an if statement (line 1206)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 1209)
        # Getting the type of 'self' (line 1209)
        self_37601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 19), 'self')
        # Obtaining the member 'mu' of a type (line 1209)
        mu_37602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1209, 19), self_37601, 'mu')
        # Getting the type of 'None' (line 1209)
        None_37603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 30), 'None')
        
        (may_be_37604, more_types_in_union_37605) = may_be_none(mu_37602, None_37603)

        if may_be_37604:

            if more_types_in_union_37605:
                # Runtime conditional SSA (line 1209)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Attribute (line 1210):
            
            # Assigning a Num to a Attribute (line 1210):
            int_37606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1210, 30), 'int')
            # Getting the type of 'self' (line 1210)
            self_37607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 20), 'self')
            # Setting the type of the member 'mu' of a type (line 1210)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1210, 20), self_37607, 'mu', int_37606)

            if more_types_in_union_37605:
                # SSA join for if statement (line 1209)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 1211)
        # Getting the type of 'self' (line 1211)
        self_37608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 19), 'self')
        # Obtaining the member 'ml' of a type (line 1211)
        ml_37609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1211, 19), self_37608, 'ml')
        # Getting the type of 'None' (line 1211)
        None_37610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 30), 'None')
        
        (may_be_37611, more_types_in_union_37612) = may_be_none(ml_37609, None_37610)

        if may_be_37611:

            if more_types_in_union_37612:
                # Runtime conditional SSA (line 1211)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Attribute (line 1212):
            
            # Assigning a Num to a Attribute (line 1212):
            int_37613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1212, 30), 'int')
            # Getting the type of 'self' (line 1212)
            self_37614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1212, 20), 'self')
            # Setting the type of the member 'ml' of a type (line 1212)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1212, 20), self_37614, 'ml', int_37613)

            if more_types_in_union_37612:
                # SSA join for if statement (line 1211)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Num to a Name (line 1213):
        
        # Assigning a Num to a Name (line 1213):
        int_37615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1213, 21), 'int')
        # Assigning a type to the variable 'jt' (line 1213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1213, 16), 'jt', int_37615)
        # SSA join for if statement (line 1206)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1205)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 1215)
        self_37616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 15), 'self')
        # Obtaining the member 'mu' of a type (line 1215)
        mu_37617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1215, 15), self_37616, 'mu')
        # Getting the type of 'None' (line 1215)
        None_37618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 26), 'None')
        # Applying the binary operator 'is' (line 1215)
        result_is__37619 = python_operator(stypy.reporting.localization.Localization(__file__, 1215, 15), 'is', mu_37617, None_37618)
        
        
        # Getting the type of 'self' (line 1215)
        self_37620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 35), 'self')
        # Obtaining the member 'ml' of a type (line 1215)
        ml_37621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1215, 35), self_37620, 'ml')
        # Getting the type of 'None' (line 1215)
        None_37622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 46), 'None')
        # Applying the binary operator 'is' (line 1215)
        result_is__37623 = python_operator(stypy.reporting.localization.Localization(__file__, 1215, 35), 'is', ml_37621, None_37622)
        
        # Applying the binary operator 'and' (line 1215)
        result_and_keyword_37624 = python_operator(stypy.reporting.localization.Localization(__file__, 1215, 15), 'and', result_is__37619, result_is__37623)
        
        # Testing the type of an if condition (line 1215)
        if_condition_37625 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1215, 12), result_and_keyword_37624)
        # Assigning a type to the variable 'if_condition_37625' (line 1215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1215, 12), 'if_condition_37625', if_condition_37625)
        # SSA begins for if statement (line 1215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 1216):
        
        # Assigning a Num to a Name (line 1216):
        int_37626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1216, 21), 'int')
        # Assigning a type to the variable 'jt' (line 1216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1216, 16), 'jt', int_37626)
        # SSA branch for the else part of an if statement (line 1215)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 1218)
        # Getting the type of 'self' (line 1218)
        self_37627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 19), 'self')
        # Obtaining the member 'mu' of a type (line 1218)
        mu_37628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1218, 19), self_37627, 'mu')
        # Getting the type of 'None' (line 1218)
        None_37629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 30), 'None')
        
        (may_be_37630, more_types_in_union_37631) = may_be_none(mu_37628, None_37629)

        if may_be_37630:

            if more_types_in_union_37631:
                # Runtime conditional SSA (line 1218)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Attribute (line 1219):
            
            # Assigning a Num to a Attribute (line 1219):
            int_37632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1219, 30), 'int')
            # Getting the type of 'self' (line 1219)
            self_37633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 20), 'self')
            # Setting the type of the member 'mu' of a type (line 1219)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1219, 20), self_37633, 'mu', int_37632)

            if more_types_in_union_37631:
                # SSA join for if statement (line 1218)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 1220)
        # Getting the type of 'self' (line 1220)
        self_37634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1220, 19), 'self')
        # Obtaining the member 'ml' of a type (line 1220)
        ml_37635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1220, 19), self_37634, 'ml')
        # Getting the type of 'None' (line 1220)
        None_37636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1220, 30), 'None')
        
        (may_be_37637, more_types_in_union_37638) = may_be_none(ml_37635, None_37636)

        if may_be_37637:

            if more_types_in_union_37638:
                # Runtime conditional SSA (line 1220)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Attribute (line 1221):
            
            # Assigning a Num to a Attribute (line 1221):
            int_37639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1221, 30), 'int')
            # Getting the type of 'self' (line 1221)
            self_37640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 20), 'self')
            # Setting the type of the member 'ml' of a type (line 1221)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1221, 20), self_37640, 'ml', int_37639)

            if more_types_in_union_37638:
                # SSA join for if statement (line 1220)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Num to a Name (line 1222):
        
        # Assigning a Num to a Name (line 1222):
        int_37641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1222, 21), 'int')
        # Assigning a type to the variable 'jt' (line 1222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1222, 16), 'jt', int_37641)
        # SSA join for if statement (line 1215)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1205)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1223):
        
        # Assigning a BinOp to a Name (line 1223):
        int_37642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1223, 14), 'int')
        # Getting the type of 'self' (line 1223)
        self_37643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 20), 'self')
        # Obtaining the member 'max_order_ns' of a type (line 1223)
        max_order_ns_37644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1223, 20), self_37643, 'max_order_ns')
        int_37645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1223, 40), 'int')
        # Applying the binary operator '+' (line 1223)
        result_add_37646 = python_operator(stypy.reporting.localization.Localization(__file__, 1223, 20), '+', max_order_ns_37644, int_37645)
        
        # Getting the type of 'n' (line 1223)
        n_37647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 45), 'n')
        # Applying the binary operator '*' (line 1223)
        result_mul_37648 = python_operator(stypy.reporting.localization.Localization(__file__, 1223, 19), '*', result_add_37646, n_37647)
        
        # Applying the binary operator '+' (line 1223)
        result_add_37649 = python_operator(stypy.reporting.localization.Localization(__file__, 1223, 14), '+', int_37642, result_mul_37648)
        
        # Assigning a type to the variable 'lrn' (line 1223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1223, 8), 'lrn', result_add_37649)
        
        
        # Getting the type of 'jt' (line 1224)
        jt_37650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 11), 'jt')
        
        # Obtaining an instance of the builtin type 'list' (line 1224)
        list_37651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1224, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1224)
        # Adding element type (line 1224)
        int_37652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1224, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1224, 17), list_37651, int_37652)
        # Adding element type (line 1224)
        int_37653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1224, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1224, 17), list_37651, int_37653)
        
        # Applying the binary operator 'in' (line 1224)
        result_contains_37654 = python_operator(stypy.reporting.localization.Localization(__file__, 1224, 11), 'in', jt_37650, list_37651)
        
        # Testing the type of an if condition (line 1224)
        if_condition_37655 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1224, 8), result_contains_37654)
        # Assigning a type to the variable 'if_condition_37655' (line 1224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1224, 8), 'if_condition_37655', if_condition_37655)
        # SSA begins for if statement (line 1224)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 1225):
        
        # Assigning a BinOp to a Name (line 1225):
        int_37656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1225, 18), 'int')
        # Getting the type of 'self' (line 1225)
        self_37657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 24), 'self')
        # Obtaining the member 'max_order_s' of a type (line 1225)
        max_order_s_37658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1225, 24), self_37657, 'max_order_s')
        int_37659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1225, 43), 'int')
        # Applying the binary operator '+' (line 1225)
        result_add_37660 = python_operator(stypy.reporting.localization.Localization(__file__, 1225, 24), '+', max_order_s_37658, int_37659)
        
        # Getting the type of 'n' (line 1225)
        n_37661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 48), 'n')
        # Applying the binary operator '*' (line 1225)
        result_mul_37662 = python_operator(stypy.reporting.localization.Localization(__file__, 1225, 23), '*', result_add_37660, n_37661)
        
        # Applying the binary operator '+' (line 1225)
        result_add_37663 = python_operator(stypy.reporting.localization.Localization(__file__, 1225, 18), '+', int_37656, result_mul_37662)
        
        # Getting the type of 'n' (line 1225)
        n_37664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 52), 'n')
        # Getting the type of 'n' (line 1225)
        n_37665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 56), 'n')
        # Applying the binary operator '*' (line 1225)
        result_mul_37666 = python_operator(stypy.reporting.localization.Localization(__file__, 1225, 52), '*', n_37664, n_37665)
        
        # Applying the binary operator '+' (line 1225)
        result_add_37667 = python_operator(stypy.reporting.localization.Localization(__file__, 1225, 50), '+', result_add_37663, result_mul_37666)
        
        # Assigning a type to the variable 'lrs' (line 1225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1225, 12), 'lrs', result_add_37667)
        # SSA branch for the else part of an if statement (line 1224)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'jt' (line 1226)
        jt_37668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 13), 'jt')
        
        # Obtaining an instance of the builtin type 'list' (line 1226)
        list_37669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1226, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1226)
        # Adding element type (line 1226)
        int_37670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1226, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1226, 19), list_37669, int_37670)
        # Adding element type (line 1226)
        int_37671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1226, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1226, 19), list_37669, int_37671)
        
        # Applying the binary operator 'in' (line 1226)
        result_contains_37672 = python_operator(stypy.reporting.localization.Localization(__file__, 1226, 13), 'in', jt_37668, list_37669)
        
        # Testing the type of an if condition (line 1226)
        if_condition_37673 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1226, 13), result_contains_37672)
        # Assigning a type to the variable 'if_condition_37673' (line 1226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1226, 13), 'if_condition_37673', if_condition_37673)
        # SSA begins for if statement (line 1226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 1227):
        
        # Assigning a BinOp to a Name (line 1227):
        int_37674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 18), 'int')
        # Getting the type of 'self' (line 1227)
        self_37675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 24), 'self')
        # Obtaining the member 'max_order_s' of a type (line 1227)
        max_order_s_37676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 24), self_37675, 'max_order_s')
        int_37677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 43), 'int')
        # Applying the binary operator '+' (line 1227)
        result_add_37678 = python_operator(stypy.reporting.localization.Localization(__file__, 1227, 24), '+', max_order_s_37676, int_37677)
        
        int_37679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 47), 'int')
        # Getting the type of 'self' (line 1227)
        self_37680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 51), 'self')
        # Obtaining the member 'ml' of a type (line 1227)
        ml_37681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 51), self_37680, 'ml')
        # Applying the binary operator '*' (line 1227)
        result_mul_37682 = python_operator(stypy.reporting.localization.Localization(__file__, 1227, 47), '*', int_37679, ml_37681)
        
        # Applying the binary operator '+' (line 1227)
        result_add_37683 = python_operator(stypy.reporting.localization.Localization(__file__, 1227, 45), '+', result_add_37678, result_mul_37682)
        
        # Getting the type of 'self' (line 1227)
        self_37684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 61), 'self')
        # Obtaining the member 'mu' of a type (line 1227)
        mu_37685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 61), self_37684, 'mu')
        # Applying the binary operator '+' (line 1227)
        result_add_37686 = python_operator(stypy.reporting.localization.Localization(__file__, 1227, 59), '+', result_add_37683, mu_37685)
        
        # Getting the type of 'n' (line 1227)
        n_37687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 72), 'n')
        # Applying the binary operator '*' (line 1227)
        result_mul_37688 = python_operator(stypy.reporting.localization.Localization(__file__, 1227, 23), '*', result_add_37686, n_37687)
        
        # Applying the binary operator '+' (line 1227)
        result_add_37689 = python_operator(stypy.reporting.localization.Localization(__file__, 1227, 18), '+', int_37674, result_mul_37688)
        
        # Assigning a type to the variable 'lrs' (line 1227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 12), 'lrs', result_add_37689)
        # SSA branch for the else part of an if statement (line 1226)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 1229)
        # Processing the call arguments (line 1229)
        str_37691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1229, 29), 'str', 'Unexpected jt=%s')
        # Getting the type of 'jt' (line 1229)
        jt_37692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 50), 'jt', False)
        # Applying the binary operator '%' (line 1229)
        result_mod_37693 = python_operator(stypy.reporting.localization.Localization(__file__, 1229, 29), '%', str_37691, jt_37692)
        
        # Processing the call keyword arguments (line 1229)
        kwargs_37694 = {}
        # Getting the type of 'ValueError' (line 1229)
        ValueError_37690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1229)
        ValueError_call_result_37695 = invoke(stypy.reporting.localization.Localization(__file__, 1229, 18), ValueError_37690, *[result_mod_37693], **kwargs_37694)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1229, 12), ValueError_call_result_37695, 'raise parameter', BaseException)
        # SSA join for if statement (line 1226)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1224)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1230):
        
        # Assigning a Call to a Name (line 1230):
        
        # Call to max(...): (line 1230)
        # Processing the call arguments (line 1230)
        # Getting the type of 'lrn' (line 1230)
        lrn_37697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 18), 'lrn', False)
        # Getting the type of 'lrs' (line 1230)
        lrs_37698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 23), 'lrs', False)
        # Processing the call keyword arguments (line 1230)
        kwargs_37699 = {}
        # Getting the type of 'max' (line 1230)
        max_37696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 14), 'max', False)
        # Calling max(args, kwargs) (line 1230)
        max_call_result_37700 = invoke(stypy.reporting.localization.Localization(__file__, 1230, 14), max_37696, *[lrn_37697, lrs_37698], **kwargs_37699)
        
        # Assigning a type to the variable 'lrw' (line 1230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1230, 8), 'lrw', max_call_result_37700)
        
        # Assigning a BinOp to a Name (line 1231):
        
        # Assigning a BinOp to a Name (line 1231):
        int_37701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1231, 14), 'int')
        # Getting the type of 'n' (line 1231)
        n_37702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 19), 'n')
        # Applying the binary operator '+' (line 1231)
        result_add_37703 = python_operator(stypy.reporting.localization.Localization(__file__, 1231, 14), '+', int_37701, n_37702)
        
        # Assigning a type to the variable 'liw' (line 1231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1231, 8), 'liw', result_add_37703)
        
        # Assigning a Call to a Name (line 1232):
        
        # Assigning a Call to a Name (line 1232):
        
        # Call to zeros(...): (line 1232)
        # Processing the call arguments (line 1232)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1232)
        tuple_37705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1232)
        # Adding element type (line 1232)
        # Getting the type of 'lrw' (line 1232)
        lrw_37706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 23), 'lrw', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1232, 23), tuple_37705, lrw_37706)
        
        # Getting the type of 'float' (line 1232)
        float_37707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 30), 'float', False)
        # Processing the call keyword arguments (line 1232)
        kwargs_37708 = {}
        # Getting the type of 'zeros' (line 1232)
        zeros_37704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 1232)
        zeros_call_result_37709 = invoke(stypy.reporting.localization.Localization(__file__, 1232, 16), zeros_37704, *[tuple_37705, float_37707], **kwargs_37708)
        
        # Assigning a type to the variable 'rwork' (line 1232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 8), 'rwork', zeros_call_result_37709)
        
        # Assigning a Attribute to a Subscript (line 1233):
        
        # Assigning a Attribute to a Subscript (line 1233):
        # Getting the type of 'self' (line 1233)
        self_37710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 19), 'self')
        # Obtaining the member 'first_step' of a type (line 1233)
        first_step_37711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1233, 19), self_37710, 'first_step')
        # Getting the type of 'rwork' (line 1233)
        rwork_37712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 8), 'rwork')
        int_37713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1233, 14), 'int')
        # Storing an element on a container (line 1233)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1233, 8), rwork_37712, (int_37713, first_step_37711))
        
        # Assigning a Attribute to a Subscript (line 1234):
        
        # Assigning a Attribute to a Subscript (line 1234):
        # Getting the type of 'self' (line 1234)
        self_37714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 19), 'self')
        # Obtaining the member 'max_step' of a type (line 1234)
        max_step_37715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1234, 19), self_37714, 'max_step')
        # Getting the type of 'rwork' (line 1234)
        rwork_37716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 8), 'rwork')
        int_37717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1234, 14), 'int')
        # Storing an element on a container (line 1234)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1234, 8), rwork_37716, (int_37717, max_step_37715))
        
        # Assigning a Attribute to a Subscript (line 1235):
        
        # Assigning a Attribute to a Subscript (line 1235):
        # Getting the type of 'self' (line 1235)
        self_37718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 19), 'self')
        # Obtaining the member 'min_step' of a type (line 1235)
        min_step_37719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1235, 19), self_37718, 'min_step')
        # Getting the type of 'rwork' (line 1235)
        rwork_37720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 8), 'rwork')
        int_37721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 14), 'int')
        # Storing an element on a container (line 1235)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1235, 8), rwork_37720, (int_37721, min_step_37719))
        
        # Assigning a Name to a Attribute (line 1236):
        
        # Assigning a Name to a Attribute (line 1236):
        # Getting the type of 'rwork' (line 1236)
        rwork_37722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 21), 'rwork')
        # Getting the type of 'self' (line 1236)
        self_37723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 8), 'self')
        # Setting the type of the member 'rwork' of a type (line 1236)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 8), self_37723, 'rwork', rwork_37722)
        
        # Assigning a Call to a Name (line 1237):
        
        # Assigning a Call to a Name (line 1237):
        
        # Call to zeros(...): (line 1237)
        # Processing the call arguments (line 1237)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1237)
        tuple_37725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1237, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1237)
        # Adding element type (line 1237)
        # Getting the type of 'liw' (line 1237)
        liw_37726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 23), 'liw', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1237, 23), tuple_37725, liw_37726)
        
        # Getting the type of 'int32' (line 1237)
        int32_37727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 30), 'int32', False)
        # Processing the call keyword arguments (line 1237)
        kwargs_37728 = {}
        # Getting the type of 'zeros' (line 1237)
        zeros_37724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 1237)
        zeros_call_result_37729 = invoke(stypy.reporting.localization.Localization(__file__, 1237, 16), zeros_37724, *[tuple_37725, int32_37727], **kwargs_37728)
        
        # Assigning a type to the variable 'iwork' (line 1237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1237, 8), 'iwork', zeros_call_result_37729)
        
        
        # Getting the type of 'self' (line 1238)
        self_37730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 11), 'self')
        # Obtaining the member 'ml' of a type (line 1238)
        ml_37731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1238, 11), self_37730, 'ml')
        # Getting the type of 'None' (line 1238)
        None_37732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 26), 'None')
        # Applying the binary operator 'isnot' (line 1238)
        result_is_not_37733 = python_operator(stypy.reporting.localization.Localization(__file__, 1238, 11), 'isnot', ml_37731, None_37732)
        
        # Testing the type of an if condition (line 1238)
        if_condition_37734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1238, 8), result_is_not_37733)
        # Assigning a type to the variable 'if_condition_37734' (line 1238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1238, 8), 'if_condition_37734', if_condition_37734)
        # SSA begins for if statement (line 1238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 1239):
        
        # Assigning a Attribute to a Subscript (line 1239):
        # Getting the type of 'self' (line 1239)
        self_37735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 23), 'self')
        # Obtaining the member 'ml' of a type (line 1239)
        ml_37736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1239, 23), self_37735, 'ml')
        # Getting the type of 'iwork' (line 1239)
        iwork_37737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 12), 'iwork')
        int_37738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1239, 18), 'int')
        # Storing an element on a container (line 1239)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1239, 12), iwork_37737, (int_37738, ml_37736))
        # SSA join for if statement (line 1238)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 1240)
        self_37739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 11), 'self')
        # Obtaining the member 'mu' of a type (line 1240)
        mu_37740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1240, 11), self_37739, 'mu')
        # Getting the type of 'None' (line 1240)
        None_37741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 26), 'None')
        # Applying the binary operator 'isnot' (line 1240)
        result_is_not_37742 = python_operator(stypy.reporting.localization.Localization(__file__, 1240, 11), 'isnot', mu_37740, None_37741)
        
        # Testing the type of an if condition (line 1240)
        if_condition_37743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1240, 8), result_is_not_37742)
        # Assigning a type to the variable 'if_condition_37743' (line 1240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1240, 8), 'if_condition_37743', if_condition_37743)
        # SSA begins for if statement (line 1240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 1241):
        
        # Assigning a Attribute to a Subscript (line 1241):
        # Getting the type of 'self' (line 1241)
        self_37744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 23), 'self')
        # Obtaining the member 'mu' of a type (line 1241)
        mu_37745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1241, 23), self_37744, 'mu')
        # Getting the type of 'iwork' (line 1241)
        iwork_37746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 12), 'iwork')
        int_37747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1241, 18), 'int')
        # Storing an element on a container (line 1241)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1241, 12), iwork_37746, (int_37747, mu_37745))
        # SSA join for if statement (line 1240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Subscript (line 1242):
        
        # Assigning a Attribute to a Subscript (line 1242):
        # Getting the type of 'self' (line 1242)
        self_37748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 19), 'self')
        # Obtaining the member 'ixpr' of a type (line 1242)
        ixpr_37749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1242, 19), self_37748, 'ixpr')
        # Getting the type of 'iwork' (line 1242)
        iwork_37750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 8), 'iwork')
        int_37751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1242, 14), 'int')
        # Storing an element on a container (line 1242)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1242, 8), iwork_37750, (int_37751, ixpr_37749))
        
        # Assigning a Attribute to a Subscript (line 1243):
        
        # Assigning a Attribute to a Subscript (line 1243):
        # Getting the type of 'self' (line 1243)
        self_37752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1243, 19), 'self')
        # Obtaining the member 'nsteps' of a type (line 1243)
        nsteps_37753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1243, 19), self_37752, 'nsteps')
        # Getting the type of 'iwork' (line 1243)
        iwork_37754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1243, 8), 'iwork')
        int_37755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1243, 14), 'int')
        # Storing an element on a container (line 1243)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1243, 8), iwork_37754, (int_37755, nsteps_37753))
        
        # Assigning a Attribute to a Subscript (line 1244):
        
        # Assigning a Attribute to a Subscript (line 1244):
        # Getting the type of 'self' (line 1244)
        self_37756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 19), 'self')
        # Obtaining the member 'max_hnil' of a type (line 1244)
        max_hnil_37757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1244, 19), self_37756, 'max_hnil')
        # Getting the type of 'iwork' (line 1244)
        iwork_37758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 8), 'iwork')
        int_37759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1244, 14), 'int')
        # Storing an element on a container (line 1244)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1244, 8), iwork_37758, (int_37759, max_hnil_37757))
        
        # Assigning a Attribute to a Subscript (line 1245):
        
        # Assigning a Attribute to a Subscript (line 1245):
        # Getting the type of 'self' (line 1245)
        self_37760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 19), 'self')
        # Obtaining the member 'max_order_ns' of a type (line 1245)
        max_order_ns_37761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1245, 19), self_37760, 'max_order_ns')
        # Getting the type of 'iwork' (line 1245)
        iwork_37762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 8), 'iwork')
        int_37763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1245, 14), 'int')
        # Storing an element on a container (line 1245)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1245, 8), iwork_37762, (int_37763, max_order_ns_37761))
        
        # Assigning a Attribute to a Subscript (line 1246):
        
        # Assigning a Attribute to a Subscript (line 1246):
        # Getting the type of 'self' (line 1246)
        self_37764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1246, 19), 'self')
        # Obtaining the member 'max_order_s' of a type (line 1246)
        max_order_s_37765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1246, 19), self_37764, 'max_order_s')
        # Getting the type of 'iwork' (line 1246)
        iwork_37766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1246, 8), 'iwork')
        int_37767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1246, 14), 'int')
        # Storing an element on a container (line 1246)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1246, 8), iwork_37766, (int_37767, max_order_s_37765))
        
        # Assigning a Name to a Attribute (line 1247):
        
        # Assigning a Name to a Attribute (line 1247):
        # Getting the type of 'iwork' (line 1247)
        iwork_37768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 21), 'iwork')
        # Getting the type of 'self' (line 1247)
        self_37769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 8), 'self')
        # Setting the type of the member 'iwork' of a type (line 1247)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1247, 8), self_37769, 'iwork', iwork_37768)
        
        # Assigning a List to a Attribute (line 1248):
        
        # Assigning a List to a Attribute (line 1248):
        
        # Obtaining an instance of the builtin type 'list' (line 1248)
        list_37770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1248, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1248)
        # Adding element type (line 1248)
        # Getting the type of 'self' (line 1248)
        self_37771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 26), 'self')
        # Obtaining the member 'rtol' of a type (line 1248)
        rtol_37772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1248, 26), self_37771, 'rtol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1248, 25), list_37770, rtol_37772)
        # Adding element type (line 1248)
        # Getting the type of 'self' (line 1248)
        self_37773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 37), 'self')
        # Obtaining the member 'atol' of a type (line 1248)
        atol_37774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1248, 37), self_37773, 'atol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1248, 25), list_37770, atol_37774)
        # Adding element type (line 1248)
        int_37775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1248, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1248, 25), list_37770, int_37775)
        # Adding element type (line 1248)
        int_37776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1248, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1248, 25), list_37770, int_37776)
        # Adding element type (line 1248)
        # Getting the type of 'self' (line 1249)
        self_37777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1249, 26), 'self')
        # Obtaining the member 'rwork' of a type (line 1249)
        rwork_37778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1249, 26), self_37777, 'rwork')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1248, 25), list_37770, rwork_37778)
        # Adding element type (line 1248)
        # Getting the type of 'self' (line 1249)
        self_37779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1249, 38), 'self')
        # Obtaining the member 'iwork' of a type (line 1249)
        iwork_37780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1249, 38), self_37779, 'iwork')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1248, 25), list_37770, iwork_37780)
        # Adding element type (line 1248)
        # Getting the type of 'jt' (line 1249)
        jt_37781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1249, 50), 'jt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1248, 25), list_37770, jt_37781)
        
        # Getting the type of 'self' (line 1248)
        self_37782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 8), 'self')
        # Setting the type of the member 'call_args' of a type (line 1248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1248, 8), self_37782, 'call_args', list_37770)
        
        # Assigning a Num to a Attribute (line 1250):
        
        # Assigning a Num to a Attribute (line 1250):
        int_37783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1250, 23), 'int')
        # Getting the type of 'self' (line 1250)
        self_37784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 8), 'self')
        # Setting the type of the member 'success' of a type (line 1250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1250, 8), self_37784, 'success', int_37783)
        
        # Assigning a Name to a Attribute (line 1251):
        
        # Assigning a Name to a Attribute (line 1251):
        # Getting the type of 'False' (line 1251)
        False_37785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 27), 'False')
        # Getting the type of 'self' (line 1251)
        self_37786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 8), 'self')
        # Setting the type of the member 'initialized' of a type (line 1251)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1251, 8), self_37786, 'initialized', False_37785)
        
        # ################# End of 'reset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset' in the type store
        # Getting the type of 'stypy_return_type' (line 1203)
        stypy_return_type_37787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37787)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset'
        return stypy_return_type_37787


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 1253, 4, False)
        # Assigning a type to the variable 'self' (line 1254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lsoda.run.__dict__.__setitem__('stypy_localization', localization)
        lsoda.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lsoda.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        lsoda.run.__dict__.__setitem__('stypy_function_name', 'lsoda.run')
        lsoda.run.__dict__.__setitem__('stypy_param_names_list', ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'])
        lsoda.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        lsoda.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lsoda.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        lsoda.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        lsoda.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lsoda.run.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lsoda.run', ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, ['f', 'jac', 'y0', 't0', 't1', 'f_params', 'jac_params'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        # Getting the type of 'self' (line 1254)
        self_37788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1254, 11), 'self')
        # Obtaining the member 'initialized' of a type (line 1254)
        initialized_37789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1254, 11), self_37788, 'initialized')
        # Testing the type of an if condition (line 1254)
        if_condition_37790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1254, 8), initialized_37789)
        # Assigning a type to the variable 'if_condition_37790' (line 1254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1254, 8), 'if_condition_37790', if_condition_37790)
        # SSA begins for if statement (line 1254)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_handle(...): (line 1255)
        # Processing the call keyword arguments (line 1255)
        kwargs_37793 = {}
        # Getting the type of 'self' (line 1255)
        self_37791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1255, 12), 'self', False)
        # Obtaining the member 'check_handle' of a type (line 1255)
        check_handle_37792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1255, 12), self_37791, 'check_handle')
        # Calling check_handle(args, kwargs) (line 1255)
        check_handle_call_result_37794 = invoke(stypy.reporting.localization.Localization(__file__, 1255, 12), check_handle_37792, *[], **kwargs_37793)
        
        # SSA branch for the else part of an if statement (line 1254)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 1257):
        
        # Assigning a Name to a Attribute (line 1257):
        # Getting the type of 'True' (line 1257)
        True_37795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 31), 'True')
        # Getting the type of 'self' (line 1257)
        self_37796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 12), 'self')
        # Setting the type of the member 'initialized' of a type (line 1257)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1257, 12), self_37796, 'initialized', True_37795)
        
        # Call to acquire_new_handle(...): (line 1258)
        # Processing the call keyword arguments (line 1258)
        kwargs_37799 = {}
        # Getting the type of 'self' (line 1258)
        self_37797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 12), 'self', False)
        # Obtaining the member 'acquire_new_handle' of a type (line 1258)
        acquire_new_handle_37798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1258, 12), self_37797, 'acquire_new_handle')
        # Calling acquire_new_handle(args, kwargs) (line 1258)
        acquire_new_handle_call_result_37800 = invoke(stypy.reporting.localization.Localization(__file__, 1258, 12), acquire_new_handle_37798, *[], **kwargs_37799)
        
        # SSA join for if statement (line 1254)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1259):
        
        # Assigning a BinOp to a Name (line 1259):
        
        # Obtaining an instance of the builtin type 'list' (line 1259)
        list_37801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1259, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1259)
        # Adding element type (line 1259)
        # Getting the type of 'f' (line 1259)
        f_37802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 16), 'f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1259, 15), list_37801, f_37802)
        # Adding element type (line 1259)
        # Getting the type of 'y0' (line 1259)
        y0_37803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 19), 'y0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1259, 15), list_37801, y0_37803)
        # Adding element type (line 1259)
        # Getting the type of 't0' (line 1259)
        t0_37804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 23), 't0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1259, 15), list_37801, t0_37804)
        # Adding element type (line 1259)
        # Getting the type of 't1' (line 1259)
        t1_37805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 27), 't1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1259, 15), list_37801, t1_37805)
        
        
        # Obtaining the type of the subscript
        int_37806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1259, 49), 'int')
        slice_37807 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1259, 33), None, int_37806, None)
        # Getting the type of 'self' (line 1259)
        self_37808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 33), 'self')
        # Obtaining the member 'call_args' of a type (line 1259)
        call_args_37809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1259, 33), self_37808, 'call_args')
        # Obtaining the member '__getitem__' of a type (line 1259)
        getitem___37810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1259, 33), call_args_37809, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1259)
        subscript_call_result_37811 = invoke(stypy.reporting.localization.Localization(__file__, 1259, 33), getitem___37810, slice_37807)
        
        # Applying the binary operator '+' (line 1259)
        result_add_37812 = python_operator(stypy.reporting.localization.Localization(__file__, 1259, 15), '+', list_37801, subscript_call_result_37811)
        
        
        # Obtaining an instance of the builtin type 'list' (line 1260)
        list_37813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1260, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1260)
        # Adding element type (line 1260)
        # Getting the type of 'jac' (line 1260)
        jac_37814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1260, 16), 'jac')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1260, 15), list_37813, jac_37814)
        # Adding element type (line 1260)
        
        # Obtaining the type of the subscript
        int_37815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1260, 36), 'int')
        # Getting the type of 'self' (line 1260)
        self_37816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1260, 21), 'self')
        # Obtaining the member 'call_args' of a type (line 1260)
        call_args_37817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1260, 21), self_37816, 'call_args')
        # Obtaining the member '__getitem__' of a type (line 1260)
        getitem___37818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1260, 21), call_args_37817, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1260)
        subscript_call_result_37819 = invoke(stypy.reporting.localization.Localization(__file__, 1260, 21), getitem___37818, int_37815)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1260, 15), list_37813, subscript_call_result_37819)
        # Adding element type (line 1260)
        # Getting the type of 'f_params' (line 1260)
        f_params_37820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1260, 41), 'f_params')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1260, 15), list_37813, f_params_37820)
        # Adding element type (line 1260)
        int_37821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1260, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1260, 15), list_37813, int_37821)
        # Adding element type (line 1260)
        # Getting the type of 'jac_params' (line 1260)
        jac_params_37822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1260, 54), 'jac_params')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1260, 15), list_37813, jac_params_37822)
        
        # Applying the binary operator '+' (line 1259)
        result_add_37823 = python_operator(stypy.reporting.localization.Localization(__file__, 1259, 53), '+', result_add_37812, list_37813)
        
        # Assigning a type to the variable 'args' (line 1259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1259, 8), 'args', result_add_37823)
        
        # Assigning a Call to a Tuple (line 1261):
        
        # Assigning a Subscript to a Name (line 1261):
        
        # Obtaining the type of the subscript
        int_37824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1261, 8), 'int')
        
        # Call to runner(...): (line 1261)
        # Getting the type of 'args' (line 1261)
        args_37827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 37), 'args', False)
        # Processing the call keyword arguments (line 1261)
        kwargs_37828 = {}
        # Getting the type of 'self' (line 1261)
        self_37825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 24), 'self', False)
        # Obtaining the member 'runner' of a type (line 1261)
        runner_37826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1261, 24), self_37825, 'runner')
        # Calling runner(args, kwargs) (line 1261)
        runner_call_result_37829 = invoke(stypy.reporting.localization.Localization(__file__, 1261, 24), runner_37826, *[args_37827], **kwargs_37828)
        
        # Obtaining the member '__getitem__' of a type (line 1261)
        getitem___37830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1261, 8), runner_call_result_37829, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1261)
        subscript_call_result_37831 = invoke(stypy.reporting.localization.Localization(__file__, 1261, 8), getitem___37830, int_37824)
        
        # Assigning a type to the variable 'tuple_var_assignment_35482' (line 1261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1261, 8), 'tuple_var_assignment_35482', subscript_call_result_37831)
        
        # Assigning a Subscript to a Name (line 1261):
        
        # Obtaining the type of the subscript
        int_37832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1261, 8), 'int')
        
        # Call to runner(...): (line 1261)
        # Getting the type of 'args' (line 1261)
        args_37835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 37), 'args', False)
        # Processing the call keyword arguments (line 1261)
        kwargs_37836 = {}
        # Getting the type of 'self' (line 1261)
        self_37833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 24), 'self', False)
        # Obtaining the member 'runner' of a type (line 1261)
        runner_37834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1261, 24), self_37833, 'runner')
        # Calling runner(args, kwargs) (line 1261)
        runner_call_result_37837 = invoke(stypy.reporting.localization.Localization(__file__, 1261, 24), runner_37834, *[args_37835], **kwargs_37836)
        
        # Obtaining the member '__getitem__' of a type (line 1261)
        getitem___37838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1261, 8), runner_call_result_37837, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1261)
        subscript_call_result_37839 = invoke(stypy.reporting.localization.Localization(__file__, 1261, 8), getitem___37838, int_37832)
        
        # Assigning a type to the variable 'tuple_var_assignment_35483' (line 1261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1261, 8), 'tuple_var_assignment_35483', subscript_call_result_37839)
        
        # Assigning a Subscript to a Name (line 1261):
        
        # Obtaining the type of the subscript
        int_37840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1261, 8), 'int')
        
        # Call to runner(...): (line 1261)
        # Getting the type of 'args' (line 1261)
        args_37843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 37), 'args', False)
        # Processing the call keyword arguments (line 1261)
        kwargs_37844 = {}
        # Getting the type of 'self' (line 1261)
        self_37841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 24), 'self', False)
        # Obtaining the member 'runner' of a type (line 1261)
        runner_37842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1261, 24), self_37841, 'runner')
        # Calling runner(args, kwargs) (line 1261)
        runner_call_result_37845 = invoke(stypy.reporting.localization.Localization(__file__, 1261, 24), runner_37842, *[args_37843], **kwargs_37844)
        
        # Obtaining the member '__getitem__' of a type (line 1261)
        getitem___37846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1261, 8), runner_call_result_37845, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1261)
        subscript_call_result_37847 = invoke(stypy.reporting.localization.Localization(__file__, 1261, 8), getitem___37846, int_37840)
        
        # Assigning a type to the variable 'tuple_var_assignment_35484' (line 1261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1261, 8), 'tuple_var_assignment_35484', subscript_call_result_37847)
        
        # Assigning a Name to a Name (line 1261):
        # Getting the type of 'tuple_var_assignment_35482' (line 1261)
        tuple_var_assignment_35482_37848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 8), 'tuple_var_assignment_35482')
        # Assigning a type to the variable 'y1' (line 1261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1261, 8), 'y1', tuple_var_assignment_35482_37848)
        
        # Assigning a Name to a Name (line 1261):
        # Getting the type of 'tuple_var_assignment_35483' (line 1261)
        tuple_var_assignment_35483_37849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 8), 'tuple_var_assignment_35483')
        # Assigning a type to the variable 't' (line 1261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1261, 12), 't', tuple_var_assignment_35483_37849)
        
        # Assigning a Name to a Name (line 1261):
        # Getting the type of 'tuple_var_assignment_35484' (line 1261)
        tuple_var_assignment_35484_37850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 8), 'tuple_var_assignment_35484')
        # Assigning a type to the variable 'istate' (line 1261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1261, 15), 'istate', tuple_var_assignment_35484_37850)
        
        # Assigning a Name to a Attribute (line 1262):
        
        # Assigning a Name to a Attribute (line 1262):
        # Getting the type of 'istate' (line 1262)
        istate_37851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1262, 22), 'istate')
        # Getting the type of 'self' (line 1262)
        self_37852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1262, 8), 'self')
        # Setting the type of the member 'istate' of a type (line 1262)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1262, 8), self_37852, 'istate', istate_37851)
        
        
        # Getting the type of 'istate' (line 1263)
        istate_37853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1263, 11), 'istate')
        int_37854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1263, 20), 'int')
        # Applying the binary operator '<' (line 1263)
        result_lt_37855 = python_operator(stypy.reporting.localization.Localization(__file__, 1263, 11), '<', istate_37853, int_37854)
        
        # Testing the type of an if condition (line 1263)
        if_condition_37856 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1263, 8), result_lt_37855)
        # Assigning a type to the variable 'if_condition_37856' (line 1263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1263, 8), 'if_condition_37856', if_condition_37856)
        # SSA begins for if statement (line 1263)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1264):
        
        # Assigning a Call to a Name (line 1264):
        
        # Call to format(...): (line 1264)
        # Processing the call arguments (line 1264)
        # Getting the type of 'istate' (line 1264)
        istate_37859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1264, 68), 'istate', False)
        # Processing the call keyword arguments (line 1264)
        kwargs_37860 = {}
        str_37857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1264, 36), 'str', 'Unexpected istate={:d}')
        # Obtaining the member 'format' of a type (line 1264)
        format_37858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1264, 36), str_37857, 'format')
        # Calling format(args, kwargs) (line 1264)
        format_call_result_37861 = invoke(stypy.reporting.localization.Localization(__file__, 1264, 36), format_37858, *[istate_37859], **kwargs_37860)
        
        # Assigning a type to the variable 'unexpected_istate_msg' (line 1264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1264, 12), 'unexpected_istate_msg', format_call_result_37861)
        
        # Call to warn(...): (line 1265)
        # Processing the call arguments (line 1265)
        
        # Call to format(...): (line 1265)
        # Processing the call arguments (line 1265)
        # Getting the type of 'self' (line 1265)
        self_37866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 46), 'self', False)
        # Obtaining the member '__class__' of a type (line 1265)
        class___37867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 46), self_37866, '__class__')
        # Obtaining the member '__name__' of a type (line 1265)
        name___37868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 46), class___37867, '__name__')
        
        # Call to get(...): (line 1266)
        # Processing the call arguments (line 1266)
        # Getting the type of 'istate' (line 1266)
        istate_37872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 44), 'istate', False)
        # Getting the type of 'unexpected_istate_msg' (line 1266)
        unexpected_istate_msg_37873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 52), 'unexpected_istate_msg', False)
        # Processing the call keyword arguments (line 1266)
        kwargs_37874 = {}
        # Getting the type of 'self' (line 1266)
        self_37869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 26), 'self', False)
        # Obtaining the member 'messages' of a type (line 1266)
        messages_37870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1266, 26), self_37869, 'messages')
        # Obtaining the member 'get' of a type (line 1266)
        get_37871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1266, 26), messages_37870, 'get')
        # Calling get(args, kwargs) (line 1266)
        get_call_result_37875 = invoke(stypy.reporting.localization.Localization(__file__, 1266, 26), get_37871, *[istate_37872, unexpected_istate_msg_37873], **kwargs_37874)
        
        # Processing the call keyword arguments (line 1265)
        kwargs_37876 = {}
        str_37864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1265, 26), 'str', '{:s}: {:s}')
        # Obtaining the member 'format' of a type (line 1265)
        format_37865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 26), str_37864, 'format')
        # Calling format(args, kwargs) (line 1265)
        format_call_result_37877 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 26), format_37865, *[name___37868, get_call_result_37875], **kwargs_37876)
        
        # Processing the call keyword arguments (line 1265)
        kwargs_37878 = {}
        # Getting the type of 'warnings' (line 1265)
        warnings_37862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 1265)
        warn_37863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 12), warnings_37862, 'warn')
        # Calling warn(args, kwargs) (line 1265)
        warn_call_result_37879 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 12), warn_37863, *[format_call_result_37877], **kwargs_37878)
        
        
        # Assigning a Num to a Attribute (line 1267):
        
        # Assigning a Num to a Attribute (line 1267):
        int_37880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1267, 27), 'int')
        # Getting the type of 'self' (line 1267)
        self_37881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1267, 12), 'self')
        # Setting the type of the member 'success' of a type (line 1267)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1267, 12), self_37881, 'success', int_37880)
        # SSA branch for the else part of an if statement (line 1263)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Subscript (line 1269):
        
        # Assigning a Num to a Subscript (line 1269):
        int_37882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1269, 32), 'int')
        # Getting the type of 'self' (line 1269)
        self_37883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1269, 12), 'self')
        # Obtaining the member 'call_args' of a type (line 1269)
        call_args_37884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1269, 12), self_37883, 'call_args')
        int_37885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1269, 27), 'int')
        # Storing an element on a container (line 1269)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1269, 12), call_args_37884, (int_37885, int_37882))
        
        # Assigning a Num to a Attribute (line 1270):
        
        # Assigning a Num to a Attribute (line 1270):
        int_37886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1270, 26), 'int')
        # Getting the type of 'self' (line 1270)
        self_37887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 12), 'self')
        # Setting the type of the member 'istate' of a type (line 1270)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1270, 12), self_37887, 'istate', int_37886)
        # SSA join for if statement (line 1263)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 1271)
        tuple_37888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1271, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1271)
        # Adding element type (line 1271)
        # Getting the type of 'y1' (line 1271)
        y1_37889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1271, 15), 'y1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1271, 15), tuple_37888, y1_37889)
        # Adding element type (line 1271)
        # Getting the type of 't' (line 1271)
        t_37890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1271, 19), 't')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1271, 15), tuple_37888, t_37890)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1271, 8), 'stypy_return_type', tuple_37888)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 1253)
        stypy_return_type_37891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37891)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_37891


    @norecursion
    def step(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'step'
        module_type_store = module_type_store.open_function_context('step', 1273, 4, False)
        # Assigning a type to the variable 'self' (line 1274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1274, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lsoda.step.__dict__.__setitem__('stypy_localization', localization)
        lsoda.step.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lsoda.step.__dict__.__setitem__('stypy_type_store', module_type_store)
        lsoda.step.__dict__.__setitem__('stypy_function_name', 'lsoda.step')
        lsoda.step.__dict__.__setitem__('stypy_param_names_list', [])
        lsoda.step.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        lsoda.step.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lsoda.step.__dict__.__setitem__('stypy_call_defaults', defaults)
        lsoda.step.__dict__.__setitem__('stypy_call_varargs', varargs)
        lsoda.step.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lsoda.step.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lsoda.step', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'step', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'step(...)' code ##################

        
        # Assigning a Subscript to a Name (line 1274):
        
        # Assigning a Subscript to a Name (line 1274):
        
        # Obtaining the type of the subscript
        int_37892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1274, 31), 'int')
        # Getting the type of 'self' (line 1274)
        self_37893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1274, 16), 'self')
        # Obtaining the member 'call_args' of a type (line 1274)
        call_args_37894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1274, 16), self_37893, 'call_args')
        # Obtaining the member '__getitem__' of a type (line 1274)
        getitem___37895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1274, 16), call_args_37894, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1274)
        subscript_call_result_37896 = invoke(stypy.reporting.localization.Localization(__file__, 1274, 16), getitem___37895, int_37892)
        
        # Assigning a type to the variable 'itask' (line 1274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1274, 8), 'itask', subscript_call_result_37896)
        
        # Assigning a Num to a Subscript (line 1275):
        
        # Assigning a Num to a Subscript (line 1275):
        int_37897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1275, 28), 'int')
        # Getting the type of 'self' (line 1275)
        self_37898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1275, 8), 'self')
        # Obtaining the member 'call_args' of a type (line 1275)
        call_args_37899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1275, 8), self_37898, 'call_args')
        int_37900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1275, 23), 'int')
        # Storing an element on a container (line 1275)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1275, 8), call_args_37899, (int_37900, int_37897))
        
        # Assigning a Call to a Name (line 1276):
        
        # Assigning a Call to a Name (line 1276):
        
        # Call to run(...): (line 1276)
        # Getting the type of 'args' (line 1276)
        args_37903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1276, 22), 'args', False)
        # Processing the call keyword arguments (line 1276)
        kwargs_37904 = {}
        # Getting the type of 'self' (line 1276)
        self_37901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1276, 12), 'self', False)
        # Obtaining the member 'run' of a type (line 1276)
        run_37902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1276, 12), self_37901, 'run')
        # Calling run(args, kwargs) (line 1276)
        run_call_result_37905 = invoke(stypy.reporting.localization.Localization(__file__, 1276, 12), run_37902, *[args_37903], **kwargs_37904)
        
        # Assigning a type to the variable 'r' (line 1276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1276, 8), 'r', run_call_result_37905)
        
        # Assigning a Name to a Subscript (line 1277):
        
        # Assigning a Name to a Subscript (line 1277):
        # Getting the type of 'itask' (line 1277)
        itask_37906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1277, 28), 'itask')
        # Getting the type of 'self' (line 1277)
        self_37907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1277, 8), 'self')
        # Obtaining the member 'call_args' of a type (line 1277)
        call_args_37908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1277, 8), self_37907, 'call_args')
        int_37909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1277, 23), 'int')
        # Storing an element on a container (line 1277)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1277, 8), call_args_37908, (int_37909, itask_37906))
        # Getting the type of 'r' (line 1278)
        r_37910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1278, 15), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 1278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1278, 8), 'stypy_return_type', r_37910)
        
        # ################# End of 'step(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'step' in the type store
        # Getting the type of 'stypy_return_type' (line 1273)
        stypy_return_type_37911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1273, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37911)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'step'
        return stypy_return_type_37911


    @norecursion
    def run_relax(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run_relax'
        module_type_store = module_type_store.open_function_context('run_relax', 1280, 4, False)
        # Assigning a type to the variable 'self' (line 1281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1281, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lsoda.run_relax.__dict__.__setitem__('stypy_localization', localization)
        lsoda.run_relax.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lsoda.run_relax.__dict__.__setitem__('stypy_type_store', module_type_store)
        lsoda.run_relax.__dict__.__setitem__('stypy_function_name', 'lsoda.run_relax')
        lsoda.run_relax.__dict__.__setitem__('stypy_param_names_list', [])
        lsoda.run_relax.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        lsoda.run_relax.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lsoda.run_relax.__dict__.__setitem__('stypy_call_defaults', defaults)
        lsoda.run_relax.__dict__.__setitem__('stypy_call_varargs', varargs)
        lsoda.run_relax.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lsoda.run_relax.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lsoda.run_relax', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run_relax', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run_relax(...)' code ##################

        
        # Assigning a Subscript to a Name (line 1281):
        
        # Assigning a Subscript to a Name (line 1281):
        
        # Obtaining the type of the subscript
        int_37912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1281, 31), 'int')
        # Getting the type of 'self' (line 1281)
        self_37913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1281, 16), 'self')
        # Obtaining the member 'call_args' of a type (line 1281)
        call_args_37914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1281, 16), self_37913, 'call_args')
        # Obtaining the member '__getitem__' of a type (line 1281)
        getitem___37915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1281, 16), call_args_37914, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1281)
        subscript_call_result_37916 = invoke(stypy.reporting.localization.Localization(__file__, 1281, 16), getitem___37915, int_37912)
        
        # Assigning a type to the variable 'itask' (line 1281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1281, 8), 'itask', subscript_call_result_37916)
        
        # Assigning a Num to a Subscript (line 1282):
        
        # Assigning a Num to a Subscript (line 1282):
        int_37917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1282, 28), 'int')
        # Getting the type of 'self' (line 1282)
        self_37918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1282, 8), 'self')
        # Obtaining the member 'call_args' of a type (line 1282)
        call_args_37919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1282, 8), self_37918, 'call_args')
        int_37920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1282, 23), 'int')
        # Storing an element on a container (line 1282)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1282, 8), call_args_37919, (int_37920, int_37917))
        
        # Assigning a Call to a Name (line 1283):
        
        # Assigning a Call to a Name (line 1283):
        
        # Call to run(...): (line 1283)
        # Getting the type of 'args' (line 1283)
        args_37923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 22), 'args', False)
        # Processing the call keyword arguments (line 1283)
        kwargs_37924 = {}
        # Getting the type of 'self' (line 1283)
        self_37921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 12), 'self', False)
        # Obtaining the member 'run' of a type (line 1283)
        run_37922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1283, 12), self_37921, 'run')
        # Calling run(args, kwargs) (line 1283)
        run_call_result_37925 = invoke(stypy.reporting.localization.Localization(__file__, 1283, 12), run_37922, *[args_37923], **kwargs_37924)
        
        # Assigning a type to the variable 'r' (line 1283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1283, 8), 'r', run_call_result_37925)
        
        # Assigning a Name to a Subscript (line 1284):
        
        # Assigning a Name to a Subscript (line 1284):
        # Getting the type of 'itask' (line 1284)
        itask_37926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 28), 'itask')
        # Getting the type of 'self' (line 1284)
        self_37927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 8), 'self')
        # Obtaining the member 'call_args' of a type (line 1284)
        call_args_37928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1284, 8), self_37927, 'call_args')
        int_37929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1284, 23), 'int')
        # Storing an element on a container (line 1284)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1284, 8), call_args_37928, (int_37929, itask_37926))
        # Getting the type of 'r' (line 1285)
        r_37930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 15), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 1285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 8), 'stypy_return_type', r_37930)
        
        # ################# End of 'run_relax(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run_relax' in the type store
        # Getting the type of 'stypy_return_type' (line 1280)
        stypy_return_type_37931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1280, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37931)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run_relax'
        return stypy_return_type_37931


# Assigning a type to the variable 'lsoda' (line 1155)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1155, 0), 'lsoda', lsoda)

# Assigning a Call to a Name (line 1156):

# Call to getattr(...): (line 1156)
# Processing the call arguments (line 1156)
# Getting the type of '_lsoda' (line 1156)
_lsoda_37933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 21), '_lsoda', False)
str_37934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1156, 29), 'str', 'lsoda')
# Getting the type of 'None' (line 1156)
None_37935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 38), 'None', False)
# Processing the call keyword arguments (line 1156)
kwargs_37936 = {}
# Getting the type of 'getattr' (line 1156)
getattr_37932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 13), 'getattr', False)
# Calling getattr(args, kwargs) (line 1156)
getattr_call_result_37937 = invoke(stypy.reporting.localization.Localization(__file__, 1156, 13), getattr_37932, *[_lsoda_37933, str_37934, None_37935], **kwargs_37936)

# Getting the type of 'lsoda'
lsoda_37938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lsoda')
# Setting the type of the member 'runner' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lsoda_37938, 'runner', getattr_call_result_37937)

# Assigning a Num to a Name (line 1157):
int_37939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1157, 27), 'int')
# Getting the type of 'lsoda'
lsoda_37940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lsoda')
# Setting the type of the member 'active_global_handle' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lsoda_37940, 'active_global_handle', int_37939)

# Assigning a Dict to a Name (line 1159):

# Obtaining an instance of the builtin type 'dict' (line 1159)
dict_37941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1159, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1159)
# Adding element type (key, value) (line 1159)
int_37942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1160, 8), 'int')
str_37943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1160, 11), 'str', 'Integration successful.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1159, 15), dict_37941, (int_37942, str_37943))
# Adding element type (key, value) (line 1159)
int_37944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1161, 8), 'int')
str_37945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1161, 12), 'str', 'Excess work done on this call (perhaps wrong Dfun type).')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1159, 15), dict_37941, (int_37944, str_37945))
# Adding element type (key, value) (line 1159)
int_37946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1162, 8), 'int')
str_37947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1162, 12), 'str', 'Excess accuracy requested (tolerances too small).')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1159, 15), dict_37941, (int_37946, str_37947))
# Adding element type (key, value) (line 1159)
int_37948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1163, 8), 'int')
str_37949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1163, 12), 'str', 'Illegal input detected (internal error).')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1159, 15), dict_37941, (int_37948, str_37949))
# Adding element type (key, value) (line 1159)
int_37950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1164, 8), 'int')
str_37951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1164, 12), 'str', 'Repeated error test failures (internal error).')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1159, 15), dict_37941, (int_37950, str_37951))
# Adding element type (key, value) (line 1159)
int_37952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1165, 8), 'int')
str_37953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1165, 12), 'str', 'Repeated convergence failures (perhaps bad Jacobian or tolerances).')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1159, 15), dict_37941, (int_37952, str_37953))
# Adding element type (key, value) (line 1159)
int_37954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1166, 8), 'int')
str_37955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1166, 12), 'str', 'Error weight became zero during problem.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1159, 15), dict_37941, (int_37954, str_37955))
# Adding element type (key, value) (line 1159)
int_37956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1167, 8), 'int')
str_37957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1167, 12), 'str', 'Internal workspace insufficient to finish (internal error).')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1159, 15), dict_37941, (int_37956, str_37957))

# Getting the type of 'lsoda'
lsoda_37958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lsoda')
# Setting the type of the member 'messages' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lsoda_37958, 'messages', dict_37941)

# Getting the type of 'lsoda' (line 1288)
lsoda_37959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 3), 'lsoda')
# Obtaining the member 'runner' of a type (line 1288)
runner_37960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1288, 3), lsoda_37959, 'runner')
# Testing the type of an if condition (line 1288)
if_condition_37961 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1288, 0), runner_37960)
# Assigning a type to the variable 'if_condition_37961' (line 1288)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1288, 0), 'if_condition_37961', if_condition_37961)
# SSA begins for if statement (line 1288)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to append(...): (line 1289)
# Processing the call arguments (line 1289)
# Getting the type of 'lsoda' (line 1289)
lsoda_37965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 45), 'lsoda', False)
# Processing the call keyword arguments (line 1289)
kwargs_37966 = {}
# Getting the type of 'IntegratorBase' (line 1289)
IntegratorBase_37962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 4), 'IntegratorBase', False)
# Obtaining the member 'integrator_classes' of a type (line 1289)
integrator_classes_37963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1289, 4), IntegratorBase_37962, 'integrator_classes')
# Obtaining the member 'append' of a type (line 1289)
append_37964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1289, 4), integrator_classes_37963, 'append')
# Calling append(args, kwargs) (line 1289)
append_call_result_37967 = invoke(stypy.reporting.localization.Localization(__file__, 1289, 4), append_37964, *[lsoda_37965], **kwargs_37966)

# SSA join for if statement (line 1288)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
