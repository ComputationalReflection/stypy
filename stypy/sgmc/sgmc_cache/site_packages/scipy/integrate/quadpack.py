
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Author: Travis Oliphant 2001
2: # Author: Nathan Woods 2013 (nquad &c)
3: from __future__ import division, print_function, absolute_import
4: 
5: import sys
6: import warnings
7: from functools import partial
8: 
9: from . import _quadpack
10: import numpy
11: from numpy import Inf
12: 
13: __all__ = ['quad', 'dblquad', 'tplquad', 'nquad', 'quad_explain',
14:            'IntegrationWarning']
15: 
16: 
17: error = _quadpack.error
18: 
19: class IntegrationWarning(UserWarning):
20:     '''
21:     Warning on issues during integration.
22:     '''
23:     pass
24: 
25: 
26: def quad_explain(output=sys.stdout):
27:     '''
28:     Print extra information about integrate.quad() parameters and returns.
29: 
30:     Parameters
31:     ----------
32:     output : instance with "write" method, optional
33:         Information about `quad` is passed to ``output.write()``.
34:         Default is ``sys.stdout``.
35: 
36:     Returns
37:     -------
38:     None
39: 
40:     '''
41:     output.write(quad.__doc__)
42: 
43: 
44: def quad(func, a, b, args=(), full_output=0, epsabs=1.49e-8, epsrel=1.49e-8,
45:          limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50,
46:          limlst=50):
47:     '''
48:     Compute a definite integral.
49: 
50:     Integrate func from `a` to `b` (possibly infinite interval) using a
51:     technique from the Fortran library QUADPACK.
52: 
53:     Parameters
54:     ----------
55:     func : {function, scipy.LowLevelCallable}
56:         A Python function or method to integrate.  If `func` takes many
57:         arguments, it is integrated along the axis corresponding to the
58:         first argument.
59: 
60:         If the user desires improved integration performance, then `f` may
61:         be a `scipy.LowLevelCallable` with one of the signatures::
62: 
63:             double func(double x)
64:             double func(double x, void *user_data)
65:             double func(int n, double *xx)
66:             double func(int n, double *xx, void *user_data)
67: 
68:         The ``user_data`` is the data contained in the `scipy.LowLevelCallable`.
69:         In the call forms with ``xx``,  ``n`` is the length of the ``xx`` 
70:         array which contains ``xx[0] == x`` and the rest of the items are
71:         numbers contained in the ``args`` argument of quad.
72: 
73:         In addition, certain ctypes call signatures are supported for 
74:         backward compatibility, but those should not be used in new code.
75:     a : float
76:         Lower limit of integration (use -numpy.inf for -infinity).
77:     b : float
78:         Upper limit of integration (use numpy.inf for +infinity).
79:     args : tuple, optional
80:         Extra arguments to pass to `func`.
81:     full_output : int, optional
82:         Non-zero to return a dictionary of integration information.
83:         If non-zero, warning messages are also suppressed and the
84:         message is appended to the output tuple.
85: 
86:     Returns
87:     -------
88:     y : float
89:         The integral of func from `a` to `b`.
90:     abserr : float
91:         An estimate of the absolute error in the result.
92:     infodict : dict
93:         A dictionary containing additional information.
94:         Run scipy.integrate.quad_explain() for more information.
95:     message
96:         A convergence message.
97:     explain
98:         Appended only with 'cos' or 'sin' weighting and infinite
99:         integration limits, it contains an explanation of the codes in
100:         infodict['ierlst']
101: 
102:     Other Parameters
103:     ----------------
104:     epsabs : float or int, optional
105:         Absolute error tolerance.
106:     epsrel : float or int, optional
107:         Relative error tolerance.
108:     limit : float or int, optional
109:         An upper bound on the number of subintervals used in the adaptive
110:         algorithm.
111:     points : (sequence of floats,ints), optional
112:         A sequence of break points in the bounded integration interval
113:         where local difficulties of the integrand may occur (e.g.,
114:         singularities, discontinuities). The sequence does not have
115:         to be sorted.
116:     weight : float or int, optional
117:         String indicating weighting function. Full explanation for this
118:         and the remaining arguments can be found below.
119:     wvar : optional
120:         Variables for use with weighting functions.
121:     wopts : optional
122:         Optional input for reusing Chebyshev moments.
123:     maxp1 : float or int, optional
124:         An upper bound on the number of Chebyshev moments.
125:     limlst : int, optional
126:         Upper bound on the number of cycles (>=3) for use with a sinusoidal
127:         weighting and an infinite end-point.
128: 
129:     See Also
130:     --------
131:     dblquad : double integral
132:     tplquad : triple integral
133:     nquad : n-dimensional integrals (uses `quad` recursively)
134:     fixed_quad : fixed-order Gaussian quadrature
135:     quadrature : adaptive Gaussian quadrature
136:     odeint : ODE integrator
137:     ode : ODE integrator
138:     simps : integrator for sampled data
139:     romb : integrator for sampled data
140:     scipy.special : for coefficients and roots of orthogonal polynomials
141: 
142:     Notes
143:     -----
144: 
145:     **Extra information for quad() inputs and outputs**
146: 
147:     If full_output is non-zero, then the third output argument
148:     (infodict) is a dictionary with entries as tabulated below.  For
149:     infinite limits, the range is transformed to (0,1) and the
150:     optional outputs are given with respect to this transformed range.
151:     Let M be the input argument limit and let K be infodict['last'].
152:     The entries are:
153: 
154:     'neval'
155:         The number of function evaluations.
156:     'last'
157:         The number, K, of subintervals produced in the subdivision process.
158:     'alist'
159:         A rank-1 array of length M, the first K elements of which are the
160:         left end points of the subintervals in the partition of the
161:         integration range.
162:     'blist'
163:         A rank-1 array of length M, the first K elements of which are the
164:         right end points of the subintervals.
165:     'rlist'
166:         A rank-1 array of length M, the first K elements of which are the
167:         integral approximations on the subintervals.
168:     'elist'
169:         A rank-1 array of length M, the first K elements of which are the
170:         moduli of the absolute error estimates on the subintervals.
171:     'iord'
172:         A rank-1 integer array of length M, the first L elements of
173:         which are pointers to the error estimates over the subintervals
174:         with ``L=K`` if ``K<=M/2+2`` or ``L=M+1-K`` otherwise. Let I be the
175:         sequence ``infodict['iord']`` and let E be the sequence
176:         ``infodict['elist']``.  Then ``E[I[1]], ..., E[I[L]]`` forms a
177:         decreasing sequence.
178: 
179:     If the input argument points is provided (i.e. it is not None),
180:     the following additional outputs are placed in the output
181:     dictionary.  Assume the points sequence is of length P.
182: 
183:     'pts'
184:         A rank-1 array of length P+2 containing the integration limits
185:         and the break points of the intervals in ascending order.
186:         This is an array giving the subintervals over which integration
187:         will occur.
188:     'level'
189:         A rank-1 integer array of length M (=limit), containing the
190:         subdivision levels of the subintervals, i.e., if (aa,bb) is a
191:         subinterval of ``(pts[1], pts[2])`` where ``pts[0]`` and ``pts[2]``
192:         are adjacent elements of ``infodict['pts']``, then (aa,bb) has level l
193:         if ``|bb-aa| = |pts[2]-pts[1]| * 2**(-l)``.
194:     'ndin'
195:         A rank-1 integer array of length P+2.  After the first integration
196:         over the intervals (pts[1], pts[2]), the error estimates over some
197:         of the intervals may have been increased artificially in order to
198:         put their subdivision forward.  This array has ones in slots
199:         corresponding to the subintervals for which this happens.
200: 
201:     **Weighting the integrand**
202: 
203:     The input variables, *weight* and *wvar*, are used to weight the
204:     integrand by a select list of functions.  Different integration
205:     methods are used to compute the integral with these weighting
206:     functions.  The possible values of weight and the corresponding
207:     weighting functions are.
208: 
209:     ==========  ===================================   =====================
210:     ``weight``  Weight function used                  ``wvar``
211:     ==========  ===================================   =====================
212:     'cos'       cos(w*x)                              wvar = w
213:     'sin'       sin(w*x)                              wvar = w
214:     'alg'       g(x) = ((x-a)**alpha)*((b-x)**beta)   wvar = (alpha, beta)
215:     'alg-loga'  g(x)*log(x-a)                         wvar = (alpha, beta)
216:     'alg-logb'  g(x)*log(b-x)                         wvar = (alpha, beta)
217:     'alg-log'   g(x)*log(x-a)*log(b-x)                wvar = (alpha, beta)
218:     'cauchy'    1/(x-c)                               wvar = c
219:     ==========  ===================================   =====================
220: 
221:     wvar holds the parameter w, (alpha, beta), or c depending on the weight
222:     selected.  In these expressions, a and b are the integration limits.
223: 
224:     For the 'cos' and 'sin' weighting, additional inputs and outputs are
225:     available.
226: 
227:     For finite integration limits, the integration is performed using a
228:     Clenshaw-Curtis method which uses Chebyshev moments.  For repeated
229:     calculations, these moments are saved in the output dictionary:
230: 
231:     'momcom'
232:         The maximum level of Chebyshev moments that have been computed,
233:         i.e., if ``M_c`` is ``infodict['momcom']`` then the moments have been
234:         computed for intervals of length ``|b-a| * 2**(-l)``,
235:         ``l=0,1,...,M_c``.
236:     'nnlog'
237:         A rank-1 integer array of length M(=limit), containing the
238:         subdivision levels of the subintervals, i.e., an element of this
239:         array is equal to l if the corresponding subinterval is
240:         ``|b-a|* 2**(-l)``.
241:     'chebmo'
242:         A rank-2 array of shape (25, maxp1) containing the computed
243:         Chebyshev moments.  These can be passed on to an integration
244:         over the same interval by passing this array as the second
245:         element of the sequence wopts and passing infodict['momcom'] as
246:         the first element.
247: 
248:     If one of the integration limits is infinite, then a Fourier integral is
249:     computed (assuming w neq 0).  If full_output is 1 and a numerical error
250:     is encountered, besides the error message attached to the output tuple,
251:     a dictionary is also appended to the output tuple which translates the
252:     error codes in the array ``info['ierlst']`` to English messages.  The
253:     output information dictionary contains the following entries instead of
254:     'last', 'alist', 'blist', 'rlist', and 'elist':
255: 
256:     'lst'
257:         The number of subintervals needed for the integration (call it ``K_f``).
258:     'rslst'
259:         A rank-1 array of length M_f=limlst, whose first ``K_f`` elements
260:         contain the integral contribution over the interval
261:         ``(a+(k-1)c, a+kc)`` where ``c = (2*floor(|w|) + 1) * pi / |w|``
262:         and ``k=1,2,...,K_f``.
263:     'erlst'
264:         A rank-1 array of length ``M_f`` containing the error estimate
265:         corresponding to the interval in the same position in
266:         ``infodict['rslist']``.
267:     'ierlst'
268:         A rank-1 integer array of length ``M_f`` containing an error flag
269:         corresponding to the interval in the same position in
270:         ``infodict['rslist']``.  See the explanation dictionary (last entry
271:         in the output tuple) for the meaning of the codes.
272: 
273:     Examples
274:     --------
275:     Calculate :math:`\\int^4_0 x^2 dx` and compare with an analytic result
276: 
277:     >>> from scipy import integrate
278:     >>> x2 = lambda x: x**2
279:     >>> integrate.quad(x2, 0, 4)
280:     (21.333333333333332, 2.3684757858670003e-13)
281:     >>> print(4**3 / 3.)  # analytical result
282:     21.3333333333
283: 
284:     Calculate :math:`\\int^\\infty_0 e^{-x} dx`
285: 
286:     >>> invexp = lambda x: np.exp(-x)
287:     >>> integrate.quad(invexp, 0, np.inf)
288:     (1.0, 5.842605999138044e-11)
289: 
290:     >>> f = lambda x,a : a*x
291:     >>> y, err = integrate.quad(f, 0, 1, args=(1,))
292:     >>> y
293:     0.5
294:     >>> y, err = integrate.quad(f, 0, 1, args=(3,))
295:     >>> y
296:     1.5
297: 
298:     Calculate :math:`\\int^1_0 x^2 + y^2 dx` with ctypes, holding
299:     y parameter as 1::
300: 
301:         testlib.c =>
302:             double func(int n, double args[n]){
303:                 return args[0]*args[0] + args[1]*args[1];}
304:         compile to library testlib.*
305: 
306:     ::
307: 
308:        from scipy import integrate
309:        import ctypes
310:        lib = ctypes.CDLL('/home/.../testlib.*') #use absolute path
311:        lib.func.restype = ctypes.c_double
312:        lib.func.argtypes = (ctypes.c_int,ctypes.c_double)
313:        integrate.quad(lib.func,0,1,(1))
314:        #(1.3333333333333333, 1.4802973661668752e-14)
315:        print((1.0**3/3.0 + 1.0) - (0.0**3/3.0 + 0.0)) #Analytic result
316:        # 1.3333333333333333
317: 
318:     '''
319:     if not isinstance(args, tuple):
320:         args = (args,)
321:     if (weight is None):
322:         retval = _quad(func, a, b, args, full_output, epsabs, epsrel, limit,
323:                        points)
324:     else:
325:         retval = _quad_weight(func, a, b, args, full_output, epsabs, epsrel,
326:                               limlst, limit, maxp1, weight, wvar, wopts)
327: 
328:     ier = retval[-1]
329:     if ier == 0:
330:         return retval[:-1]
331: 
332:     msgs = {80: "A Python error occurred possibly while calling the function.",
333:              1: "The maximum number of subdivisions (%d) has been achieved.\n  If increasing the limit yields no improvement it is advised to analyze \n  the integrand in order to determine the difficulties.  If the position of a \n  local difficulty can be determined (singularity, discontinuity) one will \n  probably gain from splitting up the interval and calling the integrator \n  on the subranges.  Perhaps a special-purpose integrator should be used." % limit,
334:              2: "The occurrence of roundoff error is detected, which prevents \n  the requested tolerance from being achieved.  The error may be \n  underestimated.",
335:              3: "Extremely bad integrand behavior occurs at some points of the\n  integration interval.",
336:              4: "The algorithm does not converge.  Roundoff error is detected\n  in the extrapolation table.  It is assumed that the requested tolerance\n  cannot be achieved, and that the returned result (if full_output = 1) is \n  the best which can be obtained.",
337:              5: "The integral is probably divergent, or slowly convergent.",
338:              6: "The input is invalid.",
339:              7: "Abnormal termination of the routine.  The estimates for result\n  and error are less reliable.  It is assumed that the requested accuracy\n  has not been achieved.",
340:             'unknown': "Unknown error."}
341: 
342:     if weight in ['cos','sin'] and (b == Inf or a == -Inf):
343:         msgs[1] = "The maximum number of cycles allowed has been achieved., e.e.\n  of subintervals (a+(k-1)c, a+kc) where c = (2*int(abs(omega)+1))\n  *pi/abs(omega), for k = 1, 2, ..., lst.  One can allow more cycles by increasing the value of limlst.  Look at info['ierlst'] with full_output=1."
344:         msgs[4] = "The extrapolation table constructed for convergence acceleration\n  of the series formed by the integral contributions over the cycles, \n  does not converge to within the requested accuracy.  Look at \n  info['ierlst'] with full_output=1."
345:         msgs[7] = "Bad integrand behavior occurs within one or more of the cycles.\n  Location and type of the difficulty involved can be determined from \n  the vector info['ierlist'] obtained with full_output=1."
346:         explain = {1: "The maximum number of subdivisions (= limit) has been \n  achieved on this cycle.",
347:                    2: "The occurrence of roundoff error is detected and prevents\n  the tolerance imposed on this cycle from being achieved.",
348:                    3: "Extremely bad integrand behavior occurs at some points of\n  this cycle.",
349:                    4: "The integral over this cycle does not converge (to within the required accuracy) due to roundoff in the extrapolation procedure invoked on this cycle.  It is assumed that the result on this interval is the best which can be obtained.",
350:                    5: "The integral over this cycle is probably divergent or slowly convergent."}
351: 
352:     try:
353:         msg = msgs[ier]
354:     except KeyError:
355:         msg = msgs['unknown']
356: 
357:     if ier in [1,2,3,4,5,7]:
358:         if full_output:
359:             if weight in ['cos','sin'] and (b == Inf or a == Inf):
360:                 return retval[:-1] + (msg, explain)
361:             else:
362:                 return retval[:-1] + (msg,)
363:         else:
364:             warnings.warn(msg, IntegrationWarning)
365:             return retval[:-1]
366:     else:
367:         raise ValueError(msg)
368: 
369: 
370: def _quad(func,a,b,args,full_output,epsabs,epsrel,limit,points):
371:     infbounds = 0
372:     if (b != Inf and a != -Inf):
373:         pass   # standard integration
374:     elif (b == Inf and a != -Inf):
375:         infbounds = 1
376:         bound = a
377:     elif (b == Inf and a == -Inf):
378:         infbounds = 2
379:         bound = 0     # ignored
380:     elif (b != Inf and a == -Inf):
381:         infbounds = -1
382:         bound = b
383:     else:
384:         raise RuntimeError("Infinity comparisons don't work for you.")
385: 
386:     if points is None:
387:         if infbounds == 0:
388:             return _quadpack._qagse(func,a,b,args,full_output,epsabs,epsrel,limit)
389:         else:
390:             return _quadpack._qagie(func,bound,infbounds,args,full_output,epsabs,epsrel,limit)
391:     else:
392:         if infbounds != 0:
393:             raise ValueError("Infinity inputs cannot be used with break points.")
394:         else:
395:             nl = len(points)
396:             the_points = numpy.zeros((nl+2,), float)
397:             the_points[:nl] = points
398:             return _quadpack._qagpe(func,a,b,the_points,args,full_output,epsabs,epsrel,limit)
399: 
400: 
401: def _quad_weight(func,a,b,args,full_output,epsabs,epsrel,limlst,limit,maxp1,weight,wvar,wopts):
402: 
403:     if weight not in ['cos','sin','alg','alg-loga','alg-logb','alg-log','cauchy']:
404:         raise ValueError("%s not a recognized weighting function." % weight)
405: 
406:     strdict = {'cos':1,'sin':2,'alg':1,'alg-loga':2,'alg-logb':3,'alg-log':4}
407: 
408:     if weight in ['cos','sin']:
409:         integr = strdict[weight]
410:         if (b != Inf and a != -Inf):  # finite limits
411:             if wopts is None:         # no precomputed chebyshev moments
412:                 return _quadpack._qawoe(func, a, b, wvar, integr, args, full_output,
413:                                         epsabs, epsrel, limit, maxp1,1)
414:             else:                     # precomputed chebyshev moments
415:                 momcom = wopts[0]
416:                 chebcom = wopts[1]
417:                 return _quadpack._qawoe(func, a, b, wvar, integr, args, full_output,
418:                                         epsabs, epsrel, limit, maxp1, 2, momcom, chebcom)
419: 
420:         elif (b == Inf and a != -Inf):
421:             return _quadpack._qawfe(func, a, wvar, integr, args, full_output,
422:                                     epsabs,limlst,limit,maxp1)
423:         elif (b != Inf and a == -Inf):  # remap function and interval
424:             if weight == 'cos':
425:                 def thefunc(x,*myargs):
426:                     y = -x
427:                     func = myargs[0]
428:                     myargs = (y,) + myargs[1:]
429:                     return func(*myargs)
430:             else:
431:                 def thefunc(x,*myargs):
432:                     y = -x
433:                     func = myargs[0]
434:                     myargs = (y,) + myargs[1:]
435:                     return -func(*myargs)
436:             args = (func,) + args
437:             return _quadpack._qawfe(thefunc, -b, wvar, integr, args,
438:                                     full_output, epsabs, limlst, limit, maxp1)
439:         else:
440:             raise ValueError("Cannot integrate with this weight from -Inf to +Inf.")
441:     else:
442:         if a in [-Inf,Inf] or b in [-Inf,Inf]:
443:             raise ValueError("Cannot integrate with this weight over an infinite interval.")
444: 
445:         if weight[:3] == 'alg':
446:             integr = strdict[weight]
447:             return _quadpack._qawse(func, a, b, wvar, integr, args,
448:                                     full_output, epsabs, epsrel, limit)
449:         else:  # weight == 'cauchy'
450:             return _quadpack._qawce(func, a, b, wvar, args, full_output,
451:                                     epsabs, epsrel, limit)
452: 
453: 
454: def dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8, epsrel=1.49e-8):
455:     '''
456:     Compute a double integral.
457: 
458:     Return the double (definite) integral of ``func(y, x)`` from ``x = a..b``
459:     and ``y = gfun(x)..hfun(x)``.
460: 
461:     Parameters
462:     ----------
463:     func : callable
464:         A Python function or method of at least two variables: y must be the
465:         first argument and x the second argument.
466:     a, b : float
467:         The limits of integration in x: `a` < `b`
468:     gfun : callable
469:         The lower boundary curve in y which is a function taking a single
470:         floating point argument (x) and returning a floating point result: a
471:         lambda function can be useful here.
472:     hfun : callable
473:         The upper boundary curve in y (same requirements as `gfun`).
474:     args : sequence, optional
475:         Extra arguments to pass to `func`.
476:     epsabs : float, optional
477:         Absolute tolerance passed directly to the inner 1-D quadrature
478:         integration. Default is 1.49e-8.
479:     epsrel : float, optional
480:         Relative tolerance of the inner 1-D integrals. Default is 1.49e-8.
481: 
482:     Returns
483:     -------
484:     y : float
485:         The resultant integral.
486:     abserr : float
487:         An estimate of the error.
488: 
489:     See also
490:     --------
491:     quad : single integral
492:     tplquad : triple integral
493:     nquad : N-dimensional integrals
494:     fixed_quad : fixed-order Gaussian quadrature
495:     quadrature : adaptive Gaussian quadrature
496:     odeint : ODE integrator
497:     ode : ODE integrator
498:     simps : integrator for sampled data
499:     romb : integrator for sampled data
500:     scipy.special : for coefficients and roots of orthogonal polynomials
501: 
502:     '''
503:     def temp_ranges(*args):
504:         return [gfun(args[0]), hfun(args[0])]
505:     return nquad(func, [temp_ranges, [a, b]], args=args, 
506:             opts={"epsabs": epsabs, "epsrel": epsrel})
507: 
508: 
509: def tplquad(func, a, b, gfun, hfun, qfun, rfun, args=(), epsabs=1.49e-8,
510:             epsrel=1.49e-8):
511:     '''
512:     Compute a triple (definite) integral.
513: 
514:     Return the triple integral of ``func(z, y, x)`` from ``x = a..b``,
515:     ``y = gfun(x)..hfun(x)``, and ``z = qfun(x,y)..rfun(x,y)``.
516: 
517:     Parameters
518:     ----------
519:     func : function
520:         A Python function or method of at least three variables in the
521:         order (z, y, x).
522:     a, b : float
523:         The limits of integration in x: `a` < `b`
524:     gfun : function
525:         The lower boundary curve in y which is a function taking a single
526:         floating point argument (x) and returning a floating point result:
527:         a lambda function can be useful here.
528:     hfun : function
529:         The upper boundary curve in y (same requirements as `gfun`).
530:     qfun : function
531:         The lower boundary surface in z.  It must be a function that takes
532:         two floats in the order (x, y) and returns a float.
533:     rfun : function
534:         The upper boundary surface in z. (Same requirements as `qfun`.)
535:     args : tuple, optional
536:         Extra arguments to pass to `func`.
537:     epsabs : float, optional
538:         Absolute tolerance passed directly to the innermost 1-D quadrature
539:         integration. Default is 1.49e-8.
540:     epsrel : float, optional
541:         Relative tolerance of the innermost 1-D integrals. Default is 1.49e-8.
542: 
543:     Returns
544:     -------
545:     y : float
546:         The resultant integral.
547:     abserr : float
548:         An estimate of the error.
549: 
550:     See Also
551:     --------
552:     quad: Adaptive quadrature using QUADPACK
553:     quadrature: Adaptive Gaussian quadrature
554:     fixed_quad: Fixed-order Gaussian quadrature
555:     dblquad: Double integrals
556:     nquad : N-dimensional integrals
557:     romb: Integrators for sampled data
558:     simps: Integrators for sampled data
559:     ode: ODE integrators
560:     odeint: ODE integrators
561:     scipy.special: For coefficients and roots of orthogonal polynomials
562: 
563:     '''
564:     # f(z, y, x)
565:     # qfun/rfun (x, y)
566:     # gfun/hfun(x)
567:     # nquad will hand (y, x, t0, ...) to ranges0
568:     # nquad will hand (x, t0, ...) to ranges1
569:     # Stupid different API...
570: 
571:     def ranges0(*args):
572:         return [qfun(args[1], args[0]), rfun(args[1], args[0])]
573: 
574:     def ranges1(*args):
575:         return [gfun(args[0]), hfun(args[0])]
576: 
577:     ranges = [ranges0, ranges1, [a, b]]
578:     return nquad(func, ranges, args=args, 
579:             opts={"epsabs": epsabs, "epsrel": epsrel})
580: 
581: 
582: def nquad(func, ranges, args=None, opts=None, full_output=False):
583:     '''
584:     Integration over multiple variables.
585: 
586:     Wraps `quad` to enable integration over multiple variables.
587:     Various options allow improved integration of discontinuous functions, as
588:     well as the use of weighted integration, and generally finer control of the
589:     integration process.
590: 
591:     Parameters
592:     ----------
593:     func : {callable, scipy.LowLevelCallable}
594:         The function to be integrated. Has arguments of ``x0, ... xn``,
595:         ``t0, tm``, where integration is carried out over ``x0, ... xn``, which
596:         must be floats.  Function signature should be
597:         ``func(x0, x1, ..., xn, t0, t1, ..., tm)``.  Integration is carried out
598:         in order.  That is, integration over ``x0`` is the innermost integral,
599:         and ``xn`` is the outermost.
600: 
601:         If the user desires improved integration performance, then `f` may
602:         be a `scipy.LowLevelCallable` with one of the signatures::
603: 
604:             double func(int n, double *xx)
605:             double func(int n, double *xx, void *user_data)
606: 
607:         where ``n`` is the number of extra parameters and args is an array
608:         of doubles of the additional parameters, the ``xx`` array contains the 
609:         coordinates. The ``user_data`` is the data contained in the
610:         `scipy.LowLevelCallable`.
611:     ranges : iterable object
612:         Each element of ranges may be either a sequence  of 2 numbers, or else
613:         a callable that returns such a sequence.  ``ranges[0]`` corresponds to
614:         integration over x0, and so on.  If an element of ranges is a callable,
615:         then it will be called with all of the integration arguments available,
616:         as well as any parametric arguments. e.g. if 
617:         ``func = f(x0, x1, x2, t0, t1)``, then ``ranges[0]`` may be defined as
618:         either ``(a, b)`` or else as ``(a, b) = range0(x1, x2, t0, t1)``.
619:     args : iterable object, optional
620:         Additional arguments ``t0, ..., tn``, required by `func`, `ranges`, and
621:         ``opts``.
622:     opts : iterable object or dict, optional
623:         Options to be passed to `quad`.  May be empty, a dict, or
624:         a sequence of dicts or functions that return a dict.  If empty, the
625:         default options from scipy.integrate.quad are used.  If a dict, the same
626:         options are used for all levels of integraion.  If a sequence, then each
627:         element of the sequence corresponds to a particular integration. e.g.
628:         opts[0] corresponds to integration over x0, and so on. If a callable, 
629:         the signature must be the same as for ``ranges``. The available
630:         options together with their default values are:
631: 
632:           - epsabs = 1.49e-08
633:           - epsrel = 1.49e-08
634:           - limit  = 50
635:           - points = None
636:           - weight = None
637:           - wvar   = None
638:           - wopts  = None
639: 
640:         For more information on these options, see `quad` and `quad_explain`.
641: 
642:     full_output : bool, optional
643:         Partial implementation of ``full_output`` from scipy.integrate.quad. 
644:         The number of integrand function evaluations ``neval`` can be obtained 
645:         by setting ``full_output=True`` when calling nquad.
646: 
647:     Returns
648:     -------
649:     result : float
650:         The result of the integration.
651:     abserr : float
652:         The maximum of the estimates of the absolute error in the various
653:         integration results.
654:     out_dict : dict, optional
655:         A dict containing additional information on the integration. 
656: 
657:     See Also
658:     --------
659:     quad : 1-dimensional numerical integration
660:     dblquad, tplquad : double and triple integrals
661:     fixed_quad : fixed-order Gaussian quadrature
662:     quadrature : adaptive Gaussian quadrature
663: 
664:     Examples
665:     --------
666:     >>> from scipy import integrate
667:     >>> func = lambda x0,x1,x2,x3 : x0**2 + x1*x2 - x3**3 + np.sin(x0) + (
668:     ...                                 1 if (x0-.2*x3-.5-.25*x1>0) else 0)
669:     >>> points = [[lambda x1,x2,x3 : 0.2*x3 + 0.5 + 0.25*x1], [], [], []]
670:     >>> def opts0(*args, **kwargs):
671:     ...     return {'points':[0.2*args[2] + 0.5 + 0.25*args[0]]}
672:     >>> integrate.nquad(func, [[0,1], [-1,1], [.13,.8], [-.15,1]],
673:     ...                 opts=[opts0,{},{},{}], full_output=True)
674:     (1.5267454070738633, 2.9437360001402324e-14, {'neval': 388962})
675: 
676:     >>> scale = .1
677:     >>> def func2(x0, x1, x2, x3, t0, t1):
678:     ...     return x0*x1*x3**2 + np.sin(x2) + 1 + (1 if x0+t1*x1-t0>0 else 0)
679:     >>> def lim0(x1, x2, x3, t0, t1):
680:     ...     return [scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) - 1,
681:     ...             scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) + 1]
682:     >>> def lim1(x2, x3, t0, t1):
683:     ...     return [scale * (t0*x2 + t1*x3) - 1,
684:     ...             scale * (t0*x2 + t1*x3) + 1]
685:     >>> def lim2(x3, t0, t1):
686:     ...     return [scale * (x3 + t0**2*t1**3) - 1,
687:     ...             scale * (x3 + t0**2*t1**3) + 1]
688:     >>> def lim3(t0, t1):
689:     ...     return [scale * (t0+t1) - 1, scale * (t0+t1) + 1]
690:     >>> def opts0(x1, x2, x3, t0, t1):
691:     ...     return {'points' : [t0 - t1*x1]}
692:     >>> def opts1(x2, x3, t0, t1):
693:     ...     return {}
694:     >>> def opts2(x3, t0, t1):
695:     ...     return {}
696:     >>> def opts3(t0, t1):
697:     ...     return {}
698:     >>> integrate.nquad(func2, [lim0, lim1, lim2, lim3], args=(0,0),
699:     ...                 opts=[opts0, opts1, opts2, opts3])
700:     (25.066666666666666, 2.7829590483937256e-13)
701: 
702:     '''
703:     depth = len(ranges)
704:     ranges = [rng if callable(rng) else _RangeFunc(rng) for rng in ranges]
705:     if args is None:
706:         args = ()
707:     if opts is None:
708:         opts = [dict([])] * depth
709: 
710:     if isinstance(opts, dict):
711:         opts = [_OptFunc(opts)] * depth
712:     else:
713:         opts = [opt if callable(opt) else _OptFunc(opt) for opt in opts]
714:     return _NQuad(func, ranges, opts, full_output).integrate(*args)
715: 
716: 
717: class _RangeFunc(object):
718:     def __init__(self, range_):
719:         self.range_ = range_
720: 
721:     def __call__(self, *args):
722:         '''Return stored value.
723: 
724:         *args needed because range_ can be float or func, and is called with
725:         variable number of parameters.
726:         '''
727:         return self.range_
728: 
729: 
730: class _OptFunc(object):
731:     def __init__(self, opt):
732:         self.opt = opt
733: 
734:     def __call__(self, *args):
735:         '''Return stored dict.'''
736:         return self.opt
737: 
738: 
739: class _NQuad(object):
740:     def __init__(self, func, ranges, opts, full_output):
741:         self.abserr = 0
742:         self.func = func
743:         self.ranges = ranges
744:         self.opts = opts
745:         self.maxdepth = len(ranges)
746:         self.full_output = full_output
747:         if self.full_output:
748:             self.out_dict = {'neval': 0}
749: 
750:     def integrate(self, *args, **kwargs):
751:         depth = kwargs.pop('depth', 0)
752:         if kwargs:
753:             raise ValueError('unexpected kwargs')
754: 
755:         # Get the integration range and options for this depth.
756:         ind = -(depth + 1)
757:         fn_range = self.ranges[ind]
758:         low, high = fn_range(*args)
759:         fn_opt = self.opts[ind]
760:         opt = dict(fn_opt(*args))
761: 
762:         if 'points' in opt:
763:             opt['points'] = [x for x in opt['points'] if low <= x <= high]
764:         if depth + 1 == self.maxdepth:
765:             f = self.func
766:         else:
767:             f = partial(self.integrate, depth=depth+1)
768:         quad_r = quad(f, low, high, args=args, full_output=self.full_output,
769:                       **opt)
770:         value = quad_r[0]
771:         abserr = quad_r[1]
772:         if self.full_output:
773:             infodict = quad_r[2]
774:             # The 'neval' parameter in full_output returns the total
775:             # number of times the integrand function was evaluated.
776:             # Therefore, only the innermost integration loop counts.
777:             if depth + 1 == self.maxdepth:
778:                 self.out_dict['neval'] += infodict['neval']
779:         self.abserr = max(self.abserr, abserr)
780:         if depth > 0:
781:             return value
782:         else:
783:             # Final result of n-D integration with error
784:             if self.full_output:
785:                 return value, self.abserr, self.out_dict
786:             else:
787:                 return value, self.abserr
788: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import warnings' statement (line 6)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from functools import partial' statement (line 7)
try:
    from functools import partial

except:
    partial = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'functools', None, module_type_store, ['partial'], [partial])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.integrate import _quadpack' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_29040 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate')

if (type(import_29040) is not StypyTypeError):

    if (import_29040 != 'pyd_module'):
        __import__(import_29040)
        sys_modules_29041 = sys.modules[import_29040]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate', sys_modules_29041.module_type_store, module_type_store, ['_quadpack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_29041, sys_modules_29041.module_type_store, module_type_store)
    else:
        from scipy.integrate import _quadpack

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate', None, module_type_store, ['_quadpack'], [_quadpack])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate', import_29040)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_29042 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_29042) is not StypyTypeError):

    if (import_29042 != 'pyd_module'):
        __import__(import_29042)
        sys_modules_29043 = sys.modules[import_29042]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', sys_modules_29043.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_29042)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy import Inf' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_29044 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_29044) is not StypyTypeError):

    if (import_29044 != 'pyd_module'):
        __import__(import_29044)
        sys_modules_29045 = sys.modules[import_29044]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', sys_modules_29045.module_type_store, module_type_store, ['Inf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_29045, sys_modules_29045.module_type_store, module_type_store)
    else:
        from numpy import Inf

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', None, module_type_store, ['Inf'], [Inf])

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_29044)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')


# Assigning a List to a Name (line 13):

# Assigning a List to a Name (line 13):
__all__ = ['quad', 'dblquad', 'tplquad', 'nquad', 'quad_explain', 'IntegrationWarning']
module_type_store.set_exportable_members(['quad', 'dblquad', 'tplquad', 'nquad', 'quad_explain', 'IntegrationWarning'])

# Obtaining an instance of the builtin type 'list' (line 13)
list_29046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
str_29047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'quad')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_29046, str_29047)
# Adding element type (line 13)
str_29048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'str', 'dblquad')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_29046, str_29048)
# Adding element type (line 13)
str_29049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 30), 'str', 'tplquad')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_29046, str_29049)
# Adding element type (line 13)
str_29050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 41), 'str', 'nquad')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_29046, str_29050)
# Adding element type (line 13)
str_29051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 50), 'str', 'quad_explain')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_29046, str_29051)
# Adding element type (line 13)
str_29052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'IntegrationWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_29046, str_29052)

# Assigning a type to the variable '__all__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__all__', list_29046)

# Assigning a Attribute to a Name (line 17):

# Assigning a Attribute to a Name (line 17):
# Getting the type of '_quadpack' (line 17)
_quadpack_29053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), '_quadpack')
# Obtaining the member 'error' of a type (line 17)
error_29054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), _quadpack_29053, 'error')
# Assigning a type to the variable 'error' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'error', error_29054)
# Declaration of the 'IntegrationWarning' class
# Getting the type of 'UserWarning' (line 19)
UserWarning_29055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'UserWarning')

class IntegrationWarning(UserWarning_29055, ):
    str_29056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', '\n    Warning on issues during integration.\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 0, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntegrationWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'IntegrationWarning' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'IntegrationWarning', IntegrationWarning)

@norecursion
def quad_explain(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'sys' (line 26)
    sys_29057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 24), 'sys')
    # Obtaining the member 'stdout' of a type (line 26)
    stdout_29058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 24), sys_29057, 'stdout')
    defaults = [stdout_29058]
    # Create a new context for function 'quad_explain'
    module_type_store = module_type_store.open_function_context('quad_explain', 26, 0, False)
    
    # Passed parameters checking function
    quad_explain.stypy_localization = localization
    quad_explain.stypy_type_of_self = None
    quad_explain.stypy_type_store = module_type_store
    quad_explain.stypy_function_name = 'quad_explain'
    quad_explain.stypy_param_names_list = ['output']
    quad_explain.stypy_varargs_param_name = None
    quad_explain.stypy_kwargs_param_name = None
    quad_explain.stypy_call_defaults = defaults
    quad_explain.stypy_call_varargs = varargs
    quad_explain.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'quad_explain', ['output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'quad_explain', localization, ['output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'quad_explain(...)' code ##################

    str_29059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', '\n    Print extra information about integrate.quad() parameters and returns.\n\n    Parameters\n    ----------\n    output : instance with "write" method, optional\n        Information about `quad` is passed to ``output.write()``.\n        Default is ``sys.stdout``.\n\n    Returns\n    -------\n    None\n\n    ')
    
    # Call to write(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'quad' (line 41)
    quad_29062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'quad', False)
    # Obtaining the member '__doc__' of a type (line 41)
    doc___29063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 17), quad_29062, '__doc__')
    # Processing the call keyword arguments (line 41)
    kwargs_29064 = {}
    # Getting the type of 'output' (line 41)
    output_29060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'output', False)
    # Obtaining the member 'write' of a type (line 41)
    write_29061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), output_29060, 'write')
    # Calling write(args, kwargs) (line 41)
    write_call_result_29065 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), write_29061, *[doc___29063], **kwargs_29064)
    
    
    # ################# End of 'quad_explain(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'quad_explain' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_29066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29066)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'quad_explain'
    return stypy_return_type_29066

# Assigning a type to the variable 'quad_explain' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'quad_explain', quad_explain)

@norecursion
def quad(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_29067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    
    int_29068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 42), 'int')
    float_29069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 52), 'float')
    float_29070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 68), 'float')
    int_29071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 15), 'int')
    # Getting the type of 'None' (line 45)
    None_29072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'None')
    # Getting the type of 'None' (line 45)
    None_29073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 39), 'None')
    # Getting the type of 'None' (line 45)
    None_29074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 50), 'None')
    # Getting the type of 'None' (line 45)
    None_29075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 62), 'None')
    int_29076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 74), 'int')
    int_29077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 16), 'int')
    defaults = [tuple_29067, int_29068, float_29069, float_29070, int_29071, None_29072, None_29073, None_29074, None_29075, int_29076, int_29077]
    # Create a new context for function 'quad'
    module_type_store = module_type_store.open_function_context('quad', 44, 0, False)
    
    # Passed parameters checking function
    quad.stypy_localization = localization
    quad.stypy_type_of_self = None
    quad.stypy_type_store = module_type_store
    quad.stypy_function_name = 'quad'
    quad.stypy_param_names_list = ['func', 'a', 'b', 'args', 'full_output', 'epsabs', 'epsrel', 'limit', 'points', 'weight', 'wvar', 'wopts', 'maxp1', 'limlst']
    quad.stypy_varargs_param_name = None
    quad.stypy_kwargs_param_name = None
    quad.stypy_call_defaults = defaults
    quad.stypy_call_varargs = varargs
    quad.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'quad', ['func', 'a', 'b', 'args', 'full_output', 'epsabs', 'epsrel', 'limit', 'points', 'weight', 'wvar', 'wopts', 'maxp1', 'limlst'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'quad', localization, ['func', 'a', 'b', 'args', 'full_output', 'epsabs', 'epsrel', 'limit', 'points', 'weight', 'wvar', 'wopts', 'maxp1', 'limlst'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'quad(...)' code ##################

    str_29078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, (-1)), 'str', "\n    Compute a definite integral.\n\n    Integrate func from `a` to `b` (possibly infinite interval) using a\n    technique from the Fortran library QUADPACK.\n\n    Parameters\n    ----------\n    func : {function, scipy.LowLevelCallable}\n        A Python function or method to integrate.  If `func` takes many\n        arguments, it is integrated along the axis corresponding to the\n        first argument.\n\n        If the user desires improved integration performance, then `f` may\n        be a `scipy.LowLevelCallable` with one of the signatures::\n\n            double func(double x)\n            double func(double x, void *user_data)\n            double func(int n, double *xx)\n            double func(int n, double *xx, void *user_data)\n\n        The ``user_data`` is the data contained in the `scipy.LowLevelCallable`.\n        In the call forms with ``xx``,  ``n`` is the length of the ``xx`` \n        array which contains ``xx[0] == x`` and the rest of the items are\n        numbers contained in the ``args`` argument of quad.\n\n        In addition, certain ctypes call signatures are supported for \n        backward compatibility, but those should not be used in new code.\n    a : float\n        Lower limit of integration (use -numpy.inf for -infinity).\n    b : float\n        Upper limit of integration (use numpy.inf for +infinity).\n    args : tuple, optional\n        Extra arguments to pass to `func`.\n    full_output : int, optional\n        Non-zero to return a dictionary of integration information.\n        If non-zero, warning messages are also suppressed and the\n        message is appended to the output tuple.\n\n    Returns\n    -------\n    y : float\n        The integral of func from `a` to `b`.\n    abserr : float\n        An estimate of the absolute error in the result.\n    infodict : dict\n        A dictionary containing additional information.\n        Run scipy.integrate.quad_explain() for more information.\n    message\n        A convergence message.\n    explain\n        Appended only with 'cos' or 'sin' weighting and infinite\n        integration limits, it contains an explanation of the codes in\n        infodict['ierlst']\n\n    Other Parameters\n    ----------------\n    epsabs : float or int, optional\n        Absolute error tolerance.\n    epsrel : float or int, optional\n        Relative error tolerance.\n    limit : float or int, optional\n        An upper bound on the number of subintervals used in the adaptive\n        algorithm.\n    points : (sequence of floats,ints), optional\n        A sequence of break points in the bounded integration interval\n        where local difficulties of the integrand may occur (e.g.,\n        singularities, discontinuities). The sequence does not have\n        to be sorted.\n    weight : float or int, optional\n        String indicating weighting function. Full explanation for this\n        and the remaining arguments can be found below.\n    wvar : optional\n        Variables for use with weighting functions.\n    wopts : optional\n        Optional input for reusing Chebyshev moments.\n    maxp1 : float or int, optional\n        An upper bound on the number of Chebyshev moments.\n    limlst : int, optional\n        Upper bound on the number of cycles (>=3) for use with a sinusoidal\n        weighting and an infinite end-point.\n\n    See Also\n    --------\n    dblquad : double integral\n    tplquad : triple integral\n    nquad : n-dimensional integrals (uses `quad` recursively)\n    fixed_quad : fixed-order Gaussian quadrature\n    quadrature : adaptive Gaussian quadrature\n    odeint : ODE integrator\n    ode : ODE integrator\n    simps : integrator for sampled data\n    romb : integrator for sampled data\n    scipy.special : for coefficients and roots of orthogonal polynomials\n\n    Notes\n    -----\n\n    **Extra information for quad() inputs and outputs**\n\n    If full_output is non-zero, then the third output argument\n    (infodict) is a dictionary with entries as tabulated below.  For\n    infinite limits, the range is transformed to (0,1) and the\n    optional outputs are given with respect to this transformed range.\n    Let M be the input argument limit and let K be infodict['last'].\n    The entries are:\n\n    'neval'\n        The number of function evaluations.\n    'last'\n        The number, K, of subintervals produced in the subdivision process.\n    'alist'\n        A rank-1 array of length M, the first K elements of which are the\n        left end points of the subintervals in the partition of the\n        integration range.\n    'blist'\n        A rank-1 array of length M, the first K elements of which are the\n        right end points of the subintervals.\n    'rlist'\n        A rank-1 array of length M, the first K elements of which are the\n        integral approximations on the subintervals.\n    'elist'\n        A rank-1 array of length M, the first K elements of which are the\n        moduli of the absolute error estimates on the subintervals.\n    'iord'\n        A rank-1 integer array of length M, the first L elements of\n        which are pointers to the error estimates over the subintervals\n        with ``L=K`` if ``K<=M/2+2`` or ``L=M+1-K`` otherwise. Let I be the\n        sequence ``infodict['iord']`` and let E be the sequence\n        ``infodict['elist']``.  Then ``E[I[1]], ..., E[I[L]]`` forms a\n        decreasing sequence.\n\n    If the input argument points is provided (i.e. it is not None),\n    the following additional outputs are placed in the output\n    dictionary.  Assume the points sequence is of length P.\n\n    'pts'\n        A rank-1 array of length P+2 containing the integration limits\n        and the break points of the intervals in ascending order.\n        This is an array giving the subintervals over which integration\n        will occur.\n    'level'\n        A rank-1 integer array of length M (=limit), containing the\n        subdivision levels of the subintervals, i.e., if (aa,bb) is a\n        subinterval of ``(pts[1], pts[2])`` where ``pts[0]`` and ``pts[2]``\n        are adjacent elements of ``infodict['pts']``, then (aa,bb) has level l\n        if ``|bb-aa| = |pts[2]-pts[1]| * 2**(-l)``.\n    'ndin'\n        A rank-1 integer array of length P+2.  After the first integration\n        over the intervals (pts[1], pts[2]), the error estimates over some\n        of the intervals may have been increased artificially in order to\n        put their subdivision forward.  This array has ones in slots\n        corresponding to the subintervals for which this happens.\n\n    **Weighting the integrand**\n\n    The input variables, *weight* and *wvar*, are used to weight the\n    integrand by a select list of functions.  Different integration\n    methods are used to compute the integral with these weighting\n    functions.  The possible values of weight and the corresponding\n    weighting functions are.\n\n    ==========  ===================================   =====================\n    ``weight``  Weight function used                  ``wvar``\n    ==========  ===================================   =====================\n    'cos'       cos(w*x)                              wvar = w\n    'sin'       sin(w*x)                              wvar = w\n    'alg'       g(x) = ((x-a)**alpha)*((b-x)**beta)   wvar = (alpha, beta)\n    'alg-loga'  g(x)*log(x-a)                         wvar = (alpha, beta)\n    'alg-logb'  g(x)*log(b-x)                         wvar = (alpha, beta)\n    'alg-log'   g(x)*log(x-a)*log(b-x)                wvar = (alpha, beta)\n    'cauchy'    1/(x-c)                               wvar = c\n    ==========  ===================================   =====================\n\n    wvar holds the parameter w, (alpha, beta), or c depending on the weight\n    selected.  In these expressions, a and b are the integration limits.\n\n    For the 'cos' and 'sin' weighting, additional inputs and outputs are\n    available.\n\n    For finite integration limits, the integration is performed using a\n    Clenshaw-Curtis method which uses Chebyshev moments.  For repeated\n    calculations, these moments are saved in the output dictionary:\n\n    'momcom'\n        The maximum level of Chebyshev moments that have been computed,\n        i.e., if ``M_c`` is ``infodict['momcom']`` then the moments have been\n        computed for intervals of length ``|b-a| * 2**(-l)``,\n        ``l=0,1,...,M_c``.\n    'nnlog'\n        A rank-1 integer array of length M(=limit), containing the\n        subdivision levels of the subintervals, i.e., an element of this\n        array is equal to l if the corresponding subinterval is\n        ``|b-a|* 2**(-l)``.\n    'chebmo'\n        A rank-2 array of shape (25, maxp1) containing the computed\n        Chebyshev moments.  These can be passed on to an integration\n        over the same interval by passing this array as the second\n        element of the sequence wopts and passing infodict['momcom'] as\n        the first element.\n\n    If one of the integration limits is infinite, then a Fourier integral is\n    computed (assuming w neq 0).  If full_output is 1 and a numerical error\n    is encountered, besides the error message attached to the output tuple,\n    a dictionary is also appended to the output tuple which translates the\n    error codes in the array ``info['ierlst']`` to English messages.  The\n    output information dictionary contains the following entries instead of\n    'last', 'alist', 'blist', 'rlist', and 'elist':\n\n    'lst'\n        The number of subintervals needed for the integration (call it ``K_f``).\n    'rslst'\n        A rank-1 array of length M_f=limlst, whose first ``K_f`` elements\n        contain the integral contribution over the interval\n        ``(a+(k-1)c, a+kc)`` where ``c = (2*floor(|w|) + 1) * pi / |w|``\n        and ``k=1,2,...,K_f``.\n    'erlst'\n        A rank-1 array of length ``M_f`` containing the error estimate\n        corresponding to the interval in the same position in\n        ``infodict['rslist']``.\n    'ierlst'\n        A rank-1 integer array of length ``M_f`` containing an error flag\n        corresponding to the interval in the same position in\n        ``infodict['rslist']``.  See the explanation dictionary (last entry\n        in the output tuple) for the meaning of the codes.\n\n    Examples\n    --------\n    Calculate :math:`\\int^4_0 x^2 dx` and compare with an analytic result\n\n    >>> from scipy import integrate\n    >>> x2 = lambda x: x**2\n    >>> integrate.quad(x2, 0, 4)\n    (21.333333333333332, 2.3684757858670003e-13)\n    >>> print(4**3 / 3.)  # analytical result\n    21.3333333333\n\n    Calculate :math:`\\int^\\infty_0 e^{-x} dx`\n\n    >>> invexp = lambda x: np.exp(-x)\n    >>> integrate.quad(invexp, 0, np.inf)\n    (1.0, 5.842605999138044e-11)\n\n    >>> f = lambda x,a : a*x\n    >>> y, err = integrate.quad(f, 0, 1, args=(1,))\n    >>> y\n    0.5\n    >>> y, err = integrate.quad(f, 0, 1, args=(3,))\n    >>> y\n    1.5\n\n    Calculate :math:`\\int^1_0 x^2 + y^2 dx` with ctypes, holding\n    y parameter as 1::\n\n        testlib.c =>\n            double func(int n, double args[n]){\n                return args[0]*args[0] + args[1]*args[1];}\n        compile to library testlib.*\n\n    ::\n\n       from scipy import integrate\n       import ctypes\n       lib = ctypes.CDLL('/home/.../testlib.*') #use absolute path\n       lib.func.restype = ctypes.c_double\n       lib.func.argtypes = (ctypes.c_int,ctypes.c_double)\n       integrate.quad(lib.func,0,1,(1))\n       #(1.3333333333333333, 1.4802973661668752e-14)\n       print((1.0**3/3.0 + 1.0) - (0.0**3/3.0 + 0.0)) #Analytic result\n       # 1.3333333333333333\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 319)
    # Getting the type of 'tuple' (line 319)
    tuple_29079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 28), 'tuple')
    # Getting the type of 'args' (line 319)
    args_29080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 22), 'args')
    
    (may_be_29081, more_types_in_union_29082) = may_not_be_subtype(tuple_29079, args_29080)

    if may_be_29081:

        if more_types_in_union_29082:
            # Runtime conditional SSA (line 319)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'args' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'args', remove_subtype_from_union(args_29080, tuple))
        
        # Assigning a Tuple to a Name (line 320):
        
        # Assigning a Tuple to a Name (line 320):
        
        # Obtaining an instance of the builtin type 'tuple' (line 320)
        tuple_29083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 320)
        # Adding element type (line 320)
        # Getting the type of 'args' (line 320)
        args_29084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 16), tuple_29083, args_29084)
        
        # Assigning a type to the variable 'args' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'args', tuple_29083)

        if more_types_in_union_29082:
            # SSA join for if statement (line 319)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 321)
    # Getting the type of 'weight' (line 321)
    weight_29085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'weight')
    # Getting the type of 'None' (line 321)
    None_29086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 18), 'None')
    
    (may_be_29087, more_types_in_union_29088) = may_be_none(weight_29085, None_29086)

    if may_be_29087:

        if more_types_in_union_29088:
            # Runtime conditional SSA (line 321)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 322):
        
        # Assigning a Call to a Name (line 322):
        
        # Call to _quad(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'func' (line 322)
        func_29090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 23), 'func', False)
        # Getting the type of 'a' (line 322)
        a_29091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 29), 'a', False)
        # Getting the type of 'b' (line 322)
        b_29092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 32), 'b', False)
        # Getting the type of 'args' (line 322)
        args_29093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 35), 'args', False)
        # Getting the type of 'full_output' (line 322)
        full_output_29094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 41), 'full_output', False)
        # Getting the type of 'epsabs' (line 322)
        epsabs_29095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 54), 'epsabs', False)
        # Getting the type of 'epsrel' (line 322)
        epsrel_29096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 62), 'epsrel', False)
        # Getting the type of 'limit' (line 322)
        limit_29097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 70), 'limit', False)
        # Getting the type of 'points' (line 323)
        points_29098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 23), 'points', False)
        # Processing the call keyword arguments (line 322)
        kwargs_29099 = {}
        # Getting the type of '_quad' (line 322)
        _quad_29089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 17), '_quad', False)
        # Calling _quad(args, kwargs) (line 322)
        _quad_call_result_29100 = invoke(stypy.reporting.localization.Localization(__file__, 322, 17), _quad_29089, *[func_29090, a_29091, b_29092, args_29093, full_output_29094, epsabs_29095, epsrel_29096, limit_29097, points_29098], **kwargs_29099)
        
        # Assigning a type to the variable 'retval' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'retval', _quad_call_result_29100)

        if more_types_in_union_29088:
            # Runtime conditional SSA for else branch (line 321)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_29087) or more_types_in_union_29088):
        
        # Assigning a Call to a Name (line 325):
        
        # Assigning a Call to a Name (line 325):
        
        # Call to _quad_weight(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'func' (line 325)
        func_29102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 30), 'func', False)
        # Getting the type of 'a' (line 325)
        a_29103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 36), 'a', False)
        # Getting the type of 'b' (line 325)
        b_29104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 39), 'b', False)
        # Getting the type of 'args' (line 325)
        args_29105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 42), 'args', False)
        # Getting the type of 'full_output' (line 325)
        full_output_29106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 48), 'full_output', False)
        # Getting the type of 'epsabs' (line 325)
        epsabs_29107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 61), 'epsabs', False)
        # Getting the type of 'epsrel' (line 325)
        epsrel_29108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 69), 'epsrel', False)
        # Getting the type of 'limlst' (line 326)
        limlst_29109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 30), 'limlst', False)
        # Getting the type of 'limit' (line 326)
        limit_29110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 38), 'limit', False)
        # Getting the type of 'maxp1' (line 326)
        maxp1_29111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 45), 'maxp1', False)
        # Getting the type of 'weight' (line 326)
        weight_29112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 52), 'weight', False)
        # Getting the type of 'wvar' (line 326)
        wvar_29113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 60), 'wvar', False)
        # Getting the type of 'wopts' (line 326)
        wopts_29114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 66), 'wopts', False)
        # Processing the call keyword arguments (line 325)
        kwargs_29115 = {}
        # Getting the type of '_quad_weight' (line 325)
        _quad_weight_29101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 17), '_quad_weight', False)
        # Calling _quad_weight(args, kwargs) (line 325)
        _quad_weight_call_result_29116 = invoke(stypy.reporting.localization.Localization(__file__, 325, 17), _quad_weight_29101, *[func_29102, a_29103, b_29104, args_29105, full_output_29106, epsabs_29107, epsrel_29108, limlst_29109, limit_29110, maxp1_29111, weight_29112, wvar_29113, wopts_29114], **kwargs_29115)
        
        # Assigning a type to the variable 'retval' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'retval', _quad_weight_call_result_29116)

        if (may_be_29087 and more_types_in_union_29088):
            # SSA join for if statement (line 321)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Subscript to a Name (line 328):
    
    # Assigning a Subscript to a Name (line 328):
    
    # Obtaining the type of the subscript
    int_29117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 17), 'int')
    # Getting the type of 'retval' (line 328)
    retval_29118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 10), 'retval')
    # Obtaining the member '__getitem__' of a type (line 328)
    getitem___29119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 10), retval_29118, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 328)
    subscript_call_result_29120 = invoke(stypy.reporting.localization.Localization(__file__, 328, 10), getitem___29119, int_29117)
    
    # Assigning a type to the variable 'ier' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'ier', subscript_call_result_29120)
    
    
    # Getting the type of 'ier' (line 329)
    ier_29121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 7), 'ier')
    int_29122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 14), 'int')
    # Applying the binary operator '==' (line 329)
    result_eq_29123 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 7), '==', ier_29121, int_29122)
    
    # Testing the type of an if condition (line 329)
    if_condition_29124 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 329, 4), result_eq_29123)
    # Assigning a type to the variable 'if_condition_29124' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'if_condition_29124', if_condition_29124)
    # SSA begins for if statement (line 329)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_29125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 23), 'int')
    slice_29126 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 330, 15), None, int_29125, None)
    # Getting the type of 'retval' (line 330)
    retval_29127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 15), 'retval')
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___29128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 15), retval_29127, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_29129 = invoke(stypy.reporting.localization.Localization(__file__, 330, 15), getitem___29128, slice_29126)
    
    # Assigning a type to the variable 'stypy_return_type' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'stypy_return_type', subscript_call_result_29129)
    # SSA join for if statement (line 329)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 332):
    
    # Assigning a Dict to a Name (line 332):
    
    # Obtaining an instance of the builtin type 'dict' (line 332)
    dict_29130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 332)
    # Adding element type (key, value) (line 332)
    int_29131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 12), 'int')
    str_29132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 16), 'str', 'A Python error occurred possibly while calling the function.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 11), dict_29130, (int_29131, str_29132))
    # Adding element type (key, value) (line 332)
    int_29133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 13), 'int')
    str_29134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 16), 'str', 'The maximum number of subdivisions (%d) has been achieved.\n  If increasing the limit yields no improvement it is advised to analyze \n  the integrand in order to determine the difficulties.  If the position of a \n  local difficulty can be determined (singularity, discontinuity) one will \n  probably gain from splitting up the interval and calling the integrator \n  on the subranges.  Perhaps a special-purpose integrator should be used.')
    # Getting the type of 'limit' (line 333)
    limit_29135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 462), 'limit')
    # Applying the binary operator '%' (line 333)
    result_mod_29136 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 16), '%', str_29134, limit_29135)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 11), dict_29130, (int_29133, result_mod_29136))
    # Adding element type (key, value) (line 332)
    int_29137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 13), 'int')
    str_29138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 16), 'str', 'The occurrence of roundoff error is detected, which prevents \n  the requested tolerance from being achieved.  The error may be \n  underestimated.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 11), dict_29130, (int_29137, str_29138))
    # Adding element type (key, value) (line 332)
    int_29139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 13), 'int')
    str_29140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 16), 'str', 'Extremely bad integrand behavior occurs at some points of the\n  integration interval.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 11), dict_29130, (int_29139, str_29140))
    # Adding element type (key, value) (line 332)
    int_29141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 13), 'int')
    str_29142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 16), 'str', 'The algorithm does not converge.  Roundoff error is detected\n  in the extrapolation table.  It is assumed that the requested tolerance\n  cannot be achieved, and that the returned result (if full_output = 1) is \n  the best which can be obtained.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 11), dict_29130, (int_29141, str_29142))
    # Adding element type (key, value) (line 332)
    int_29143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 13), 'int')
    str_29144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 16), 'str', 'The integral is probably divergent, or slowly convergent.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 11), dict_29130, (int_29143, str_29144))
    # Adding element type (key, value) (line 332)
    int_29145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 13), 'int')
    str_29146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 16), 'str', 'The input is invalid.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 11), dict_29130, (int_29145, str_29146))
    # Adding element type (key, value) (line 332)
    int_29147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 13), 'int')
    str_29148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 16), 'str', 'Abnormal termination of the routine.  The estimates for result\n  and error are less reliable.  It is assumed that the requested accuracy\n  has not been achieved.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 11), dict_29130, (int_29147, str_29148))
    # Adding element type (key, value) (line 332)
    str_29149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 12), 'str', 'unknown')
    str_29150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 23), 'str', 'Unknown error.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 11), dict_29130, (str_29149, str_29150))
    
    # Assigning a type to the variable 'msgs' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'msgs', dict_29130)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'weight' (line 342)
    weight_29151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 7), 'weight')
    
    # Obtaining an instance of the builtin type 'list' (line 342)
    list_29152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 342)
    # Adding element type (line 342)
    str_29153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 18), 'str', 'cos')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 17), list_29152, str_29153)
    # Adding element type (line 342)
    str_29154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 24), 'str', 'sin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 17), list_29152, str_29154)
    
    # Applying the binary operator 'in' (line 342)
    result_contains_29155 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 7), 'in', weight_29151, list_29152)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 342)
    b_29156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 36), 'b')
    # Getting the type of 'Inf' (line 342)
    Inf_29157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 41), 'Inf')
    # Applying the binary operator '==' (line 342)
    result_eq_29158 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 36), '==', b_29156, Inf_29157)
    
    
    # Getting the type of 'a' (line 342)
    a_29159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 48), 'a')
    
    # Getting the type of 'Inf' (line 342)
    Inf_29160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 54), 'Inf')
    # Applying the 'usub' unary operator (line 342)
    result___neg___29161 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 53), 'usub', Inf_29160)
    
    # Applying the binary operator '==' (line 342)
    result_eq_29162 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 48), '==', a_29159, result___neg___29161)
    
    # Applying the binary operator 'or' (line 342)
    result_or_keyword_29163 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 36), 'or', result_eq_29158, result_eq_29162)
    
    # Applying the binary operator 'and' (line 342)
    result_and_keyword_29164 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 7), 'and', result_contains_29155, result_or_keyword_29163)
    
    # Testing the type of an if condition (line 342)
    if_condition_29165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 4), result_and_keyword_29164)
    # Assigning a type to the variable 'if_condition_29165' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'if_condition_29165', if_condition_29165)
    # SSA begins for if statement (line 342)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 343):
    
    # Assigning a Str to a Subscript (line 343):
    str_29166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 18), 'str', "The maximum number of cycles allowed has been achieved., e.e.\n  of subintervals (a+(k-1)c, a+kc) where c = (2*int(abs(omega)+1))\n  *pi/abs(omega), for k = 1, 2, ..., lst.  One can allow more cycles by increasing the value of limlst.  Look at info['ierlst'] with full_output=1.")
    # Getting the type of 'msgs' (line 343)
    msgs_29167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'msgs')
    int_29168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 13), 'int')
    # Storing an element on a container (line 343)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 8), msgs_29167, (int_29168, str_29166))
    
    # Assigning a Str to a Subscript (line 344):
    
    # Assigning a Str to a Subscript (line 344):
    str_29169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 18), 'str', "The extrapolation table constructed for convergence acceleration\n  of the series formed by the integral contributions over the cycles, \n  does not converge to within the requested accuracy.  Look at \n  info['ierlst'] with full_output=1.")
    # Getting the type of 'msgs' (line 344)
    msgs_29170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'msgs')
    int_29171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 13), 'int')
    # Storing an element on a container (line 344)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 8), msgs_29170, (int_29171, str_29169))
    
    # Assigning a Str to a Subscript (line 345):
    
    # Assigning a Str to a Subscript (line 345):
    str_29172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 18), 'str', "Bad integrand behavior occurs within one or more of the cycles.\n  Location and type of the difficulty involved can be determined from \n  the vector info['ierlist'] obtained with full_output=1.")
    # Getting the type of 'msgs' (line 345)
    msgs_29173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'msgs')
    int_29174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 13), 'int')
    # Storing an element on a container (line 345)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 8), msgs_29173, (int_29174, str_29172))
    
    # Assigning a Dict to a Name (line 346):
    
    # Assigning a Dict to a Name (line 346):
    
    # Obtaining an instance of the builtin type 'dict' (line 346)
    dict_29175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 18), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 346)
    # Adding element type (key, value) (line 346)
    int_29176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 19), 'int')
    str_29177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 22), 'str', 'The maximum number of subdivisions (= limit) has been \n  achieved on this cycle.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 18), dict_29175, (int_29176, str_29177))
    # Adding element type (key, value) (line 346)
    int_29178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 19), 'int')
    str_29179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 22), 'str', 'The occurrence of roundoff error is detected and prevents\n  the tolerance imposed on this cycle from being achieved.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 18), dict_29175, (int_29178, str_29179))
    # Adding element type (key, value) (line 346)
    int_29180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 19), 'int')
    str_29181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 22), 'str', 'Extremely bad integrand behavior occurs at some points of\n  this cycle.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 18), dict_29175, (int_29180, str_29181))
    # Adding element type (key, value) (line 346)
    int_29182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 19), 'int')
    str_29183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 22), 'str', 'The integral over this cycle does not converge (to within the required accuracy) due to roundoff in the extrapolation procedure invoked on this cycle.  It is assumed that the result on this interval is the best which can be obtained.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 18), dict_29175, (int_29182, str_29183))
    # Adding element type (key, value) (line 346)
    int_29184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 19), 'int')
    str_29185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 22), 'str', 'The integral over this cycle is probably divergent or slowly convergent.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 18), dict_29175, (int_29184, str_29185))
    
    # Assigning a type to the variable 'explain' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'explain', dict_29175)
    # SSA join for if statement (line 342)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 352)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 353):
    
    # Assigning a Subscript to a Name (line 353):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ier' (line 353)
    ier_29186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 19), 'ier')
    # Getting the type of 'msgs' (line 353)
    msgs_29187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 14), 'msgs')
    # Obtaining the member '__getitem__' of a type (line 353)
    getitem___29188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 14), msgs_29187, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 353)
    subscript_call_result_29189 = invoke(stypy.reporting.localization.Localization(__file__, 353, 14), getitem___29188, ier_29186)
    
    # Assigning a type to the variable 'msg' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'msg', subscript_call_result_29189)
    # SSA branch for the except part of a try statement (line 352)
    # SSA branch for the except 'KeyError' branch of a try statement (line 352)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Subscript to a Name (line 355):
    
    # Assigning a Subscript to a Name (line 355):
    
    # Obtaining the type of the subscript
    str_29190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 19), 'str', 'unknown')
    # Getting the type of 'msgs' (line 355)
    msgs_29191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 14), 'msgs')
    # Obtaining the member '__getitem__' of a type (line 355)
    getitem___29192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 14), msgs_29191, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 355)
    subscript_call_result_29193 = invoke(stypy.reporting.localization.Localization(__file__, 355, 14), getitem___29192, str_29190)
    
    # Assigning a type to the variable 'msg' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'msg', subscript_call_result_29193)
    # SSA join for try-except statement (line 352)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ier' (line 357)
    ier_29194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 7), 'ier')
    
    # Obtaining an instance of the builtin type 'list' (line 357)
    list_29195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 357)
    # Adding element type (line 357)
    int_29196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 14), list_29195, int_29196)
    # Adding element type (line 357)
    int_29197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 14), list_29195, int_29197)
    # Adding element type (line 357)
    int_29198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 14), list_29195, int_29198)
    # Adding element type (line 357)
    int_29199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 14), list_29195, int_29199)
    # Adding element type (line 357)
    int_29200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 14), list_29195, int_29200)
    # Adding element type (line 357)
    int_29201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 14), list_29195, int_29201)
    
    # Applying the binary operator 'in' (line 357)
    result_contains_29202 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 7), 'in', ier_29194, list_29195)
    
    # Testing the type of an if condition (line 357)
    if_condition_29203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 4), result_contains_29202)
    # Assigning a type to the variable 'if_condition_29203' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'if_condition_29203', if_condition_29203)
    # SSA begins for if statement (line 357)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'full_output' (line 358)
    full_output_29204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 11), 'full_output')
    # Testing the type of an if condition (line 358)
    if_condition_29205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 358, 8), full_output_29204)
    # Assigning a type to the variable 'if_condition_29205' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'if_condition_29205', if_condition_29205)
    # SSA begins for if statement (line 358)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'weight' (line 359)
    weight_29206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 15), 'weight')
    
    # Obtaining an instance of the builtin type 'list' (line 359)
    list_29207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 359)
    # Adding element type (line 359)
    str_29208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 26), 'str', 'cos')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 25), list_29207, str_29208)
    # Adding element type (line 359)
    str_29209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 32), 'str', 'sin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 25), list_29207, str_29209)
    
    # Applying the binary operator 'in' (line 359)
    result_contains_29210 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 15), 'in', weight_29206, list_29207)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 359)
    b_29211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 44), 'b')
    # Getting the type of 'Inf' (line 359)
    Inf_29212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 49), 'Inf')
    # Applying the binary operator '==' (line 359)
    result_eq_29213 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 44), '==', b_29211, Inf_29212)
    
    
    # Getting the type of 'a' (line 359)
    a_29214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 56), 'a')
    # Getting the type of 'Inf' (line 359)
    Inf_29215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 61), 'Inf')
    # Applying the binary operator '==' (line 359)
    result_eq_29216 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 56), '==', a_29214, Inf_29215)
    
    # Applying the binary operator 'or' (line 359)
    result_or_keyword_29217 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 44), 'or', result_eq_29213, result_eq_29216)
    
    # Applying the binary operator 'and' (line 359)
    result_and_keyword_29218 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 15), 'and', result_contains_29210, result_or_keyword_29217)
    
    # Testing the type of an if condition (line 359)
    if_condition_29219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 359, 12), result_and_keyword_29218)
    # Assigning a type to the variable 'if_condition_29219' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'if_condition_29219', if_condition_29219)
    # SSA begins for if statement (line 359)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_29220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 31), 'int')
    slice_29221 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 360, 23), None, int_29220, None)
    # Getting the type of 'retval' (line 360)
    retval_29222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'retval')
    # Obtaining the member '__getitem__' of a type (line 360)
    getitem___29223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 23), retval_29222, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 360)
    subscript_call_result_29224 = invoke(stypy.reporting.localization.Localization(__file__, 360, 23), getitem___29223, slice_29221)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 360)
    tuple_29225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 360)
    # Adding element type (line 360)
    # Getting the type of 'msg' (line 360)
    msg_29226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 38), 'msg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 38), tuple_29225, msg_29226)
    # Adding element type (line 360)
    # Getting the type of 'explain' (line 360)
    explain_29227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 43), 'explain')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 38), tuple_29225, explain_29227)
    
    # Applying the binary operator '+' (line 360)
    result_add_29228 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 23), '+', subscript_call_result_29224, tuple_29225)
    
    # Assigning a type to the variable 'stypy_return_type' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 16), 'stypy_return_type', result_add_29228)
    # SSA branch for the else part of an if statement (line 359)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining the type of the subscript
    int_29229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 31), 'int')
    slice_29230 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 362, 23), None, int_29229, None)
    # Getting the type of 'retval' (line 362)
    retval_29231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 23), 'retval')
    # Obtaining the member '__getitem__' of a type (line 362)
    getitem___29232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 23), retval_29231, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 362)
    subscript_call_result_29233 = invoke(stypy.reporting.localization.Localization(__file__, 362, 23), getitem___29232, slice_29230)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 362)
    tuple_29234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 362)
    # Adding element type (line 362)
    # Getting the type of 'msg' (line 362)
    msg_29235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 38), 'msg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 38), tuple_29234, msg_29235)
    
    # Applying the binary operator '+' (line 362)
    result_add_29236 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 23), '+', subscript_call_result_29233, tuple_29234)
    
    # Assigning a type to the variable 'stypy_return_type' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'stypy_return_type', result_add_29236)
    # SSA join for if statement (line 359)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 358)
    module_type_store.open_ssa_branch('else')
    
    # Call to warn(...): (line 364)
    # Processing the call arguments (line 364)
    # Getting the type of 'msg' (line 364)
    msg_29239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 26), 'msg', False)
    # Getting the type of 'IntegrationWarning' (line 364)
    IntegrationWarning_29240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 31), 'IntegrationWarning', False)
    # Processing the call keyword arguments (line 364)
    kwargs_29241 = {}
    # Getting the type of 'warnings' (line 364)
    warnings_29237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 364)
    warn_29238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 12), warnings_29237, 'warn')
    # Calling warn(args, kwargs) (line 364)
    warn_call_result_29242 = invoke(stypy.reporting.localization.Localization(__file__, 364, 12), warn_29238, *[msg_29239, IntegrationWarning_29240], **kwargs_29241)
    
    
    # Obtaining the type of the subscript
    int_29243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 27), 'int')
    slice_29244 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 365, 19), None, int_29243, None)
    # Getting the type of 'retval' (line 365)
    retval_29245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 19), 'retval')
    # Obtaining the member '__getitem__' of a type (line 365)
    getitem___29246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 19), retval_29245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 365)
    subscript_call_result_29247 = invoke(stypy.reporting.localization.Localization(__file__, 365, 19), getitem___29246, slice_29244)
    
    # Assigning a type to the variable 'stypy_return_type' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'stypy_return_type', subscript_call_result_29247)
    # SSA join for if statement (line 358)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 357)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'msg' (line 367)
    msg_29249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 25), 'msg', False)
    # Processing the call keyword arguments (line 367)
    kwargs_29250 = {}
    # Getting the type of 'ValueError' (line 367)
    ValueError_29248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 367)
    ValueError_call_result_29251 = invoke(stypy.reporting.localization.Localization(__file__, 367, 14), ValueError_29248, *[msg_29249], **kwargs_29250)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 367, 8), ValueError_call_result_29251, 'raise parameter', BaseException)
    # SSA join for if statement (line 357)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'quad(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'quad' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_29252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29252)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'quad'
    return stypy_return_type_29252

# Assigning a type to the variable 'quad' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'quad', quad)

@norecursion
def _quad(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_quad'
    module_type_store = module_type_store.open_function_context('_quad', 370, 0, False)
    
    # Passed parameters checking function
    _quad.stypy_localization = localization
    _quad.stypy_type_of_self = None
    _quad.stypy_type_store = module_type_store
    _quad.stypy_function_name = '_quad'
    _quad.stypy_param_names_list = ['func', 'a', 'b', 'args', 'full_output', 'epsabs', 'epsrel', 'limit', 'points']
    _quad.stypy_varargs_param_name = None
    _quad.stypy_kwargs_param_name = None
    _quad.stypy_call_defaults = defaults
    _quad.stypy_call_varargs = varargs
    _quad.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_quad', ['func', 'a', 'b', 'args', 'full_output', 'epsabs', 'epsrel', 'limit', 'points'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_quad', localization, ['func', 'a', 'b', 'args', 'full_output', 'epsabs', 'epsrel', 'limit', 'points'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_quad(...)' code ##################

    
    # Assigning a Num to a Name (line 371):
    
    # Assigning a Num to a Name (line 371):
    int_29253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 16), 'int')
    # Assigning a type to the variable 'infbounds' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'infbounds', int_29253)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 372)
    b_29254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'b')
    # Getting the type of 'Inf' (line 372)
    Inf_29255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 13), 'Inf')
    # Applying the binary operator '!=' (line 372)
    result_ne_29256 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 8), '!=', b_29254, Inf_29255)
    
    
    # Getting the type of 'a' (line 372)
    a_29257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 21), 'a')
    
    # Getting the type of 'Inf' (line 372)
    Inf_29258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 27), 'Inf')
    # Applying the 'usub' unary operator (line 372)
    result___neg___29259 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 26), 'usub', Inf_29258)
    
    # Applying the binary operator '!=' (line 372)
    result_ne_29260 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 21), '!=', a_29257, result___neg___29259)
    
    # Applying the binary operator 'and' (line 372)
    result_and_keyword_29261 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 8), 'and', result_ne_29256, result_ne_29260)
    
    # Testing the type of an if condition (line 372)
    if_condition_29262 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 4), result_and_keyword_29261)
    # Assigning a type to the variable 'if_condition_29262' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'if_condition_29262', if_condition_29262)
    # SSA begins for if statement (line 372)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 372)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 374)
    b_29263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 10), 'b')
    # Getting the type of 'Inf' (line 374)
    Inf_29264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 15), 'Inf')
    # Applying the binary operator '==' (line 374)
    result_eq_29265 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 10), '==', b_29263, Inf_29264)
    
    
    # Getting the type of 'a' (line 374)
    a_29266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'a')
    
    # Getting the type of 'Inf' (line 374)
    Inf_29267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 29), 'Inf')
    # Applying the 'usub' unary operator (line 374)
    result___neg___29268 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 28), 'usub', Inf_29267)
    
    # Applying the binary operator '!=' (line 374)
    result_ne_29269 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 23), '!=', a_29266, result___neg___29268)
    
    # Applying the binary operator 'and' (line 374)
    result_and_keyword_29270 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 10), 'and', result_eq_29265, result_ne_29269)
    
    # Testing the type of an if condition (line 374)
    if_condition_29271 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 374, 9), result_and_keyword_29270)
    # Assigning a type to the variable 'if_condition_29271' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 9), 'if_condition_29271', if_condition_29271)
    # SSA begins for if statement (line 374)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 375):
    
    # Assigning a Num to a Name (line 375):
    int_29272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 20), 'int')
    # Assigning a type to the variable 'infbounds' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'infbounds', int_29272)
    
    # Assigning a Name to a Name (line 376):
    
    # Assigning a Name to a Name (line 376):
    # Getting the type of 'a' (line 376)
    a_29273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 16), 'a')
    # Assigning a type to the variable 'bound' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'bound', a_29273)
    # SSA branch for the else part of an if statement (line 374)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 377)
    b_29274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 10), 'b')
    # Getting the type of 'Inf' (line 377)
    Inf_29275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 15), 'Inf')
    # Applying the binary operator '==' (line 377)
    result_eq_29276 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 10), '==', b_29274, Inf_29275)
    
    
    # Getting the type of 'a' (line 377)
    a_29277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 23), 'a')
    
    # Getting the type of 'Inf' (line 377)
    Inf_29278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 29), 'Inf')
    # Applying the 'usub' unary operator (line 377)
    result___neg___29279 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 28), 'usub', Inf_29278)
    
    # Applying the binary operator '==' (line 377)
    result_eq_29280 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 23), '==', a_29277, result___neg___29279)
    
    # Applying the binary operator 'and' (line 377)
    result_and_keyword_29281 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 10), 'and', result_eq_29276, result_eq_29280)
    
    # Testing the type of an if condition (line 377)
    if_condition_29282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 377, 9), result_and_keyword_29281)
    # Assigning a type to the variable 'if_condition_29282' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 9), 'if_condition_29282', if_condition_29282)
    # SSA begins for if statement (line 377)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 378):
    
    # Assigning a Num to a Name (line 378):
    int_29283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 20), 'int')
    # Assigning a type to the variable 'infbounds' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'infbounds', int_29283)
    
    # Assigning a Num to a Name (line 379):
    
    # Assigning a Num to a Name (line 379):
    int_29284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 16), 'int')
    # Assigning a type to the variable 'bound' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'bound', int_29284)
    # SSA branch for the else part of an if statement (line 377)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 380)
    b_29285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 10), 'b')
    # Getting the type of 'Inf' (line 380)
    Inf_29286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 15), 'Inf')
    # Applying the binary operator '!=' (line 380)
    result_ne_29287 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 10), '!=', b_29285, Inf_29286)
    
    
    # Getting the type of 'a' (line 380)
    a_29288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 23), 'a')
    
    # Getting the type of 'Inf' (line 380)
    Inf_29289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 29), 'Inf')
    # Applying the 'usub' unary operator (line 380)
    result___neg___29290 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 28), 'usub', Inf_29289)
    
    # Applying the binary operator '==' (line 380)
    result_eq_29291 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 23), '==', a_29288, result___neg___29290)
    
    # Applying the binary operator 'and' (line 380)
    result_and_keyword_29292 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 10), 'and', result_ne_29287, result_eq_29291)
    
    # Testing the type of an if condition (line 380)
    if_condition_29293 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 380, 9), result_and_keyword_29292)
    # Assigning a type to the variable 'if_condition_29293' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 9), 'if_condition_29293', if_condition_29293)
    # SSA begins for if statement (line 380)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 381):
    
    # Assigning a Num to a Name (line 381):
    int_29294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 20), 'int')
    # Assigning a type to the variable 'infbounds' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'infbounds', int_29294)
    
    # Assigning a Name to a Name (line 382):
    
    # Assigning a Name to a Name (line 382):
    # Getting the type of 'b' (line 382)
    b_29295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 16), 'b')
    # Assigning a type to the variable 'bound' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'bound', b_29295)
    # SSA branch for the else part of an if statement (line 380)
    module_type_store.open_ssa_branch('else')
    
    # Call to RuntimeError(...): (line 384)
    # Processing the call arguments (line 384)
    str_29297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 27), 'str', "Infinity comparisons don't work for you.")
    # Processing the call keyword arguments (line 384)
    kwargs_29298 = {}
    # Getting the type of 'RuntimeError' (line 384)
    RuntimeError_29296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 384)
    RuntimeError_call_result_29299 = invoke(stypy.reporting.localization.Localization(__file__, 384, 14), RuntimeError_29296, *[str_29297], **kwargs_29298)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 384, 8), RuntimeError_call_result_29299, 'raise parameter', BaseException)
    # SSA join for if statement (line 380)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 377)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 374)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 372)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 386)
    # Getting the type of 'points' (line 386)
    points_29300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 7), 'points')
    # Getting the type of 'None' (line 386)
    None_29301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 17), 'None')
    
    (may_be_29302, more_types_in_union_29303) = may_be_none(points_29300, None_29301)

    if may_be_29302:

        if more_types_in_union_29303:
            # Runtime conditional SSA (line 386)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'infbounds' (line 387)
        infbounds_29304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 11), 'infbounds')
        int_29305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 24), 'int')
        # Applying the binary operator '==' (line 387)
        result_eq_29306 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 11), '==', infbounds_29304, int_29305)
        
        # Testing the type of an if condition (line 387)
        if_condition_29307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 387, 8), result_eq_29306)
        # Assigning a type to the variable 'if_condition_29307' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'if_condition_29307', if_condition_29307)
        # SSA begins for if statement (line 387)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _qagse(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'func' (line 388)
        func_29310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 36), 'func', False)
        # Getting the type of 'a' (line 388)
        a_29311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 41), 'a', False)
        # Getting the type of 'b' (line 388)
        b_29312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 43), 'b', False)
        # Getting the type of 'args' (line 388)
        args_29313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 45), 'args', False)
        # Getting the type of 'full_output' (line 388)
        full_output_29314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 50), 'full_output', False)
        # Getting the type of 'epsabs' (line 388)
        epsabs_29315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 62), 'epsabs', False)
        # Getting the type of 'epsrel' (line 388)
        epsrel_29316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 69), 'epsrel', False)
        # Getting the type of 'limit' (line 388)
        limit_29317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 76), 'limit', False)
        # Processing the call keyword arguments (line 388)
        kwargs_29318 = {}
        # Getting the type of '_quadpack' (line 388)
        _quadpack_29308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 19), '_quadpack', False)
        # Obtaining the member '_qagse' of a type (line 388)
        _qagse_29309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 19), _quadpack_29308, '_qagse')
        # Calling _qagse(args, kwargs) (line 388)
        _qagse_call_result_29319 = invoke(stypy.reporting.localization.Localization(__file__, 388, 19), _qagse_29309, *[func_29310, a_29311, b_29312, args_29313, full_output_29314, epsabs_29315, epsrel_29316, limit_29317], **kwargs_29318)
        
        # Assigning a type to the variable 'stypy_return_type' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'stypy_return_type', _qagse_call_result_29319)
        # SSA branch for the else part of an if statement (line 387)
        module_type_store.open_ssa_branch('else')
        
        # Call to _qagie(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'func' (line 390)
        func_29322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 36), 'func', False)
        # Getting the type of 'bound' (line 390)
        bound_29323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 41), 'bound', False)
        # Getting the type of 'infbounds' (line 390)
        infbounds_29324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 47), 'infbounds', False)
        # Getting the type of 'args' (line 390)
        args_29325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 57), 'args', False)
        # Getting the type of 'full_output' (line 390)
        full_output_29326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 62), 'full_output', False)
        # Getting the type of 'epsabs' (line 390)
        epsabs_29327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 74), 'epsabs', False)
        # Getting the type of 'epsrel' (line 390)
        epsrel_29328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 81), 'epsrel', False)
        # Getting the type of 'limit' (line 390)
        limit_29329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 88), 'limit', False)
        # Processing the call keyword arguments (line 390)
        kwargs_29330 = {}
        # Getting the type of '_quadpack' (line 390)
        _quadpack_29320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), '_quadpack', False)
        # Obtaining the member '_qagie' of a type (line 390)
        _qagie_29321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 19), _quadpack_29320, '_qagie')
        # Calling _qagie(args, kwargs) (line 390)
        _qagie_call_result_29331 = invoke(stypy.reporting.localization.Localization(__file__, 390, 19), _qagie_29321, *[func_29322, bound_29323, infbounds_29324, args_29325, full_output_29326, epsabs_29327, epsrel_29328, limit_29329], **kwargs_29330)
        
        # Assigning a type to the variable 'stypy_return_type' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'stypy_return_type', _qagie_call_result_29331)
        # SSA join for if statement (line 387)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_29303:
            # Runtime conditional SSA for else branch (line 386)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_29302) or more_types_in_union_29303):
        
        
        # Getting the type of 'infbounds' (line 392)
        infbounds_29332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 11), 'infbounds')
        int_29333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 24), 'int')
        # Applying the binary operator '!=' (line 392)
        result_ne_29334 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 11), '!=', infbounds_29332, int_29333)
        
        # Testing the type of an if condition (line 392)
        if_condition_29335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 392, 8), result_ne_29334)
        # Assigning a type to the variable 'if_condition_29335' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'if_condition_29335', if_condition_29335)
        # SSA begins for if statement (line 392)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 393)
        # Processing the call arguments (line 393)
        str_29337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 29), 'str', 'Infinity inputs cannot be used with break points.')
        # Processing the call keyword arguments (line 393)
        kwargs_29338 = {}
        # Getting the type of 'ValueError' (line 393)
        ValueError_29336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 393)
        ValueError_call_result_29339 = invoke(stypy.reporting.localization.Localization(__file__, 393, 18), ValueError_29336, *[str_29337], **kwargs_29338)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 393, 12), ValueError_call_result_29339, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 392)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 395):
        
        # Assigning a Call to a Name (line 395):
        
        # Call to len(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'points' (line 395)
        points_29341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 21), 'points', False)
        # Processing the call keyword arguments (line 395)
        kwargs_29342 = {}
        # Getting the type of 'len' (line 395)
        len_29340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 17), 'len', False)
        # Calling len(args, kwargs) (line 395)
        len_call_result_29343 = invoke(stypy.reporting.localization.Localization(__file__, 395, 17), len_29340, *[points_29341], **kwargs_29342)
        
        # Assigning a type to the variable 'nl' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'nl', len_call_result_29343)
        
        # Assigning a Call to a Name (line 396):
        
        # Assigning a Call to a Name (line 396):
        
        # Call to zeros(...): (line 396)
        # Processing the call arguments (line 396)
        
        # Obtaining an instance of the builtin type 'tuple' (line 396)
        tuple_29346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 396)
        # Adding element type (line 396)
        # Getting the type of 'nl' (line 396)
        nl_29347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 38), 'nl', False)
        int_29348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 41), 'int')
        # Applying the binary operator '+' (line 396)
        result_add_29349 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 38), '+', nl_29347, int_29348)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 38), tuple_29346, result_add_29349)
        
        # Getting the type of 'float' (line 396)
        float_29350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 46), 'float', False)
        # Processing the call keyword arguments (line 396)
        kwargs_29351 = {}
        # Getting the type of 'numpy' (line 396)
        numpy_29344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 25), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 396)
        zeros_29345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 25), numpy_29344, 'zeros')
        # Calling zeros(args, kwargs) (line 396)
        zeros_call_result_29352 = invoke(stypy.reporting.localization.Localization(__file__, 396, 25), zeros_29345, *[tuple_29346, float_29350], **kwargs_29351)
        
        # Assigning a type to the variable 'the_points' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'the_points', zeros_call_result_29352)
        
        # Assigning a Name to a Subscript (line 397):
        
        # Assigning a Name to a Subscript (line 397):
        # Getting the type of 'points' (line 397)
        points_29353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 30), 'points')
        # Getting the type of 'the_points' (line 397)
        the_points_29354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'the_points')
        # Getting the type of 'nl' (line 397)
        nl_29355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 24), 'nl')
        slice_29356 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 397, 12), None, nl_29355, None)
        # Storing an element on a container (line 397)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 12), the_points_29354, (slice_29356, points_29353))
        
        # Call to _qagpe(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'func' (line 398)
        func_29359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 36), 'func', False)
        # Getting the type of 'a' (line 398)
        a_29360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 41), 'a', False)
        # Getting the type of 'b' (line 398)
        b_29361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 43), 'b', False)
        # Getting the type of 'the_points' (line 398)
        the_points_29362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 45), 'the_points', False)
        # Getting the type of 'args' (line 398)
        args_29363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 56), 'args', False)
        # Getting the type of 'full_output' (line 398)
        full_output_29364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 61), 'full_output', False)
        # Getting the type of 'epsabs' (line 398)
        epsabs_29365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 73), 'epsabs', False)
        # Getting the type of 'epsrel' (line 398)
        epsrel_29366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 80), 'epsrel', False)
        # Getting the type of 'limit' (line 398)
        limit_29367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 87), 'limit', False)
        # Processing the call keyword arguments (line 398)
        kwargs_29368 = {}
        # Getting the type of '_quadpack' (line 398)
        _quadpack_29357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), '_quadpack', False)
        # Obtaining the member '_qagpe' of a type (line 398)
        _qagpe_29358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 19), _quadpack_29357, '_qagpe')
        # Calling _qagpe(args, kwargs) (line 398)
        _qagpe_call_result_29369 = invoke(stypy.reporting.localization.Localization(__file__, 398, 19), _qagpe_29358, *[func_29359, a_29360, b_29361, the_points_29362, args_29363, full_output_29364, epsabs_29365, epsrel_29366, limit_29367], **kwargs_29368)
        
        # Assigning a type to the variable 'stypy_return_type' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'stypy_return_type', _qagpe_call_result_29369)
        # SSA join for if statement (line 392)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_29302 and more_types_in_union_29303):
            # SSA join for if statement (line 386)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_quad(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_quad' in the type store
    # Getting the type of 'stypy_return_type' (line 370)
    stypy_return_type_29370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29370)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_quad'
    return stypy_return_type_29370

# Assigning a type to the variable '_quad' (line 370)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 0), '_quad', _quad)

@norecursion
def _quad_weight(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_quad_weight'
    module_type_store = module_type_store.open_function_context('_quad_weight', 401, 0, False)
    
    # Passed parameters checking function
    _quad_weight.stypy_localization = localization
    _quad_weight.stypy_type_of_self = None
    _quad_weight.stypy_type_store = module_type_store
    _quad_weight.stypy_function_name = '_quad_weight'
    _quad_weight.stypy_param_names_list = ['func', 'a', 'b', 'args', 'full_output', 'epsabs', 'epsrel', 'limlst', 'limit', 'maxp1', 'weight', 'wvar', 'wopts']
    _quad_weight.stypy_varargs_param_name = None
    _quad_weight.stypy_kwargs_param_name = None
    _quad_weight.stypy_call_defaults = defaults
    _quad_weight.stypy_call_varargs = varargs
    _quad_weight.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_quad_weight', ['func', 'a', 'b', 'args', 'full_output', 'epsabs', 'epsrel', 'limlst', 'limit', 'maxp1', 'weight', 'wvar', 'wopts'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_quad_weight', localization, ['func', 'a', 'b', 'args', 'full_output', 'epsabs', 'epsrel', 'limlst', 'limit', 'maxp1', 'weight', 'wvar', 'wopts'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_quad_weight(...)' code ##################

    
    
    # Getting the type of 'weight' (line 403)
    weight_29371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 7), 'weight')
    
    # Obtaining an instance of the builtin type 'list' (line 403)
    list_29372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 403)
    # Adding element type (line 403)
    str_29373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 22), 'str', 'cos')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 21), list_29372, str_29373)
    # Adding element type (line 403)
    str_29374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 28), 'str', 'sin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 21), list_29372, str_29374)
    # Adding element type (line 403)
    str_29375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 34), 'str', 'alg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 21), list_29372, str_29375)
    # Adding element type (line 403)
    str_29376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 40), 'str', 'alg-loga')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 21), list_29372, str_29376)
    # Adding element type (line 403)
    str_29377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 51), 'str', 'alg-logb')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 21), list_29372, str_29377)
    # Adding element type (line 403)
    str_29378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 62), 'str', 'alg-log')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 21), list_29372, str_29378)
    # Adding element type (line 403)
    str_29379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 72), 'str', 'cauchy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 21), list_29372, str_29379)
    
    # Applying the binary operator 'notin' (line 403)
    result_contains_29380 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 7), 'notin', weight_29371, list_29372)
    
    # Testing the type of an if condition (line 403)
    if_condition_29381 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 4), result_contains_29380)
    # Assigning a type to the variable 'if_condition_29381' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'if_condition_29381', if_condition_29381)
    # SSA begins for if statement (line 403)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 404)
    # Processing the call arguments (line 404)
    str_29383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 25), 'str', '%s not a recognized weighting function.')
    # Getting the type of 'weight' (line 404)
    weight_29384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 69), 'weight', False)
    # Applying the binary operator '%' (line 404)
    result_mod_29385 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 25), '%', str_29383, weight_29384)
    
    # Processing the call keyword arguments (line 404)
    kwargs_29386 = {}
    # Getting the type of 'ValueError' (line 404)
    ValueError_29382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 404)
    ValueError_call_result_29387 = invoke(stypy.reporting.localization.Localization(__file__, 404, 14), ValueError_29382, *[result_mod_29385], **kwargs_29386)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 404, 8), ValueError_call_result_29387, 'raise parameter', BaseException)
    # SSA join for if statement (line 403)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 406):
    
    # Assigning a Dict to a Name (line 406):
    
    # Obtaining an instance of the builtin type 'dict' (line 406)
    dict_29388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 406)
    # Adding element type (key, value) (line 406)
    str_29389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 15), 'str', 'cos')
    int_29390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 21), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 14), dict_29388, (str_29389, int_29390))
    # Adding element type (key, value) (line 406)
    str_29391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 23), 'str', 'sin')
    int_29392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 29), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 14), dict_29388, (str_29391, int_29392))
    # Adding element type (key, value) (line 406)
    str_29393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 31), 'str', 'alg')
    int_29394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 37), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 14), dict_29388, (str_29393, int_29394))
    # Adding element type (key, value) (line 406)
    str_29395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 39), 'str', 'alg-loga')
    int_29396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 50), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 14), dict_29388, (str_29395, int_29396))
    # Adding element type (key, value) (line 406)
    str_29397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 52), 'str', 'alg-logb')
    int_29398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 63), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 14), dict_29388, (str_29397, int_29398))
    # Adding element type (key, value) (line 406)
    str_29399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 65), 'str', 'alg-log')
    int_29400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 75), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 14), dict_29388, (str_29399, int_29400))
    
    # Assigning a type to the variable 'strdict' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'strdict', dict_29388)
    
    
    # Getting the type of 'weight' (line 408)
    weight_29401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 7), 'weight')
    
    # Obtaining an instance of the builtin type 'list' (line 408)
    list_29402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 408)
    # Adding element type (line 408)
    str_29403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 18), 'str', 'cos')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 17), list_29402, str_29403)
    # Adding element type (line 408)
    str_29404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 24), 'str', 'sin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 17), list_29402, str_29404)
    
    # Applying the binary operator 'in' (line 408)
    result_contains_29405 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 7), 'in', weight_29401, list_29402)
    
    # Testing the type of an if condition (line 408)
    if_condition_29406 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 408, 4), result_contains_29405)
    # Assigning a type to the variable 'if_condition_29406' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'if_condition_29406', if_condition_29406)
    # SSA begins for if statement (line 408)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 409):
    
    # Assigning a Subscript to a Name (line 409):
    
    # Obtaining the type of the subscript
    # Getting the type of 'weight' (line 409)
    weight_29407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 25), 'weight')
    # Getting the type of 'strdict' (line 409)
    strdict_29408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 17), 'strdict')
    # Obtaining the member '__getitem__' of a type (line 409)
    getitem___29409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 17), strdict_29408, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 409)
    subscript_call_result_29410 = invoke(stypy.reporting.localization.Localization(__file__, 409, 17), getitem___29409, weight_29407)
    
    # Assigning a type to the variable 'integr' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'integr', subscript_call_result_29410)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 410)
    b_29411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'b')
    # Getting the type of 'Inf' (line 410)
    Inf_29412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 17), 'Inf')
    # Applying the binary operator '!=' (line 410)
    result_ne_29413 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 12), '!=', b_29411, Inf_29412)
    
    
    # Getting the type of 'a' (line 410)
    a_29414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'a')
    
    # Getting the type of 'Inf' (line 410)
    Inf_29415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 31), 'Inf')
    # Applying the 'usub' unary operator (line 410)
    result___neg___29416 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 30), 'usub', Inf_29415)
    
    # Applying the binary operator '!=' (line 410)
    result_ne_29417 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 25), '!=', a_29414, result___neg___29416)
    
    # Applying the binary operator 'and' (line 410)
    result_and_keyword_29418 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 12), 'and', result_ne_29413, result_ne_29417)
    
    # Testing the type of an if condition (line 410)
    if_condition_29419 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 410, 8), result_and_keyword_29418)
    # Assigning a type to the variable 'if_condition_29419' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'if_condition_29419', if_condition_29419)
    # SSA begins for if statement (line 410)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 411)
    # Getting the type of 'wopts' (line 411)
    wopts_29420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 15), 'wopts')
    # Getting the type of 'None' (line 411)
    None_29421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 24), 'None')
    
    (may_be_29422, more_types_in_union_29423) = may_be_none(wopts_29420, None_29421)

    if may_be_29422:

        if more_types_in_union_29423:
            # Runtime conditional SSA (line 411)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to _qawoe(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'func' (line 412)
        func_29426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 40), 'func', False)
        # Getting the type of 'a' (line 412)
        a_29427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 46), 'a', False)
        # Getting the type of 'b' (line 412)
        b_29428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 49), 'b', False)
        # Getting the type of 'wvar' (line 412)
        wvar_29429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 52), 'wvar', False)
        # Getting the type of 'integr' (line 412)
        integr_29430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 58), 'integr', False)
        # Getting the type of 'args' (line 412)
        args_29431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 66), 'args', False)
        # Getting the type of 'full_output' (line 412)
        full_output_29432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 72), 'full_output', False)
        # Getting the type of 'epsabs' (line 413)
        epsabs_29433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 40), 'epsabs', False)
        # Getting the type of 'epsrel' (line 413)
        epsrel_29434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 48), 'epsrel', False)
        # Getting the type of 'limit' (line 413)
        limit_29435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 56), 'limit', False)
        # Getting the type of 'maxp1' (line 413)
        maxp1_29436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 63), 'maxp1', False)
        int_29437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 69), 'int')
        # Processing the call keyword arguments (line 412)
        kwargs_29438 = {}
        # Getting the type of '_quadpack' (line 412)
        _quadpack_29424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 23), '_quadpack', False)
        # Obtaining the member '_qawoe' of a type (line 412)
        _qawoe_29425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 23), _quadpack_29424, '_qawoe')
        # Calling _qawoe(args, kwargs) (line 412)
        _qawoe_call_result_29439 = invoke(stypy.reporting.localization.Localization(__file__, 412, 23), _qawoe_29425, *[func_29426, a_29427, b_29428, wvar_29429, integr_29430, args_29431, full_output_29432, epsabs_29433, epsrel_29434, limit_29435, maxp1_29436, int_29437], **kwargs_29438)
        
        # Assigning a type to the variable 'stypy_return_type' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 16), 'stypy_return_type', _qawoe_call_result_29439)

        if more_types_in_union_29423:
            # Runtime conditional SSA for else branch (line 411)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_29422) or more_types_in_union_29423):
        
        # Assigning a Subscript to a Name (line 415):
        
        # Assigning a Subscript to a Name (line 415):
        
        # Obtaining the type of the subscript
        int_29440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 31), 'int')
        # Getting the type of 'wopts' (line 415)
        wopts_29441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 25), 'wopts')
        # Obtaining the member '__getitem__' of a type (line 415)
        getitem___29442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 25), wopts_29441, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 415)
        subscript_call_result_29443 = invoke(stypy.reporting.localization.Localization(__file__, 415, 25), getitem___29442, int_29440)
        
        # Assigning a type to the variable 'momcom' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 16), 'momcom', subscript_call_result_29443)
        
        # Assigning a Subscript to a Name (line 416):
        
        # Assigning a Subscript to a Name (line 416):
        
        # Obtaining the type of the subscript
        int_29444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 32), 'int')
        # Getting the type of 'wopts' (line 416)
        wopts_29445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 26), 'wopts')
        # Obtaining the member '__getitem__' of a type (line 416)
        getitem___29446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 26), wopts_29445, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 416)
        subscript_call_result_29447 = invoke(stypy.reporting.localization.Localization(__file__, 416, 26), getitem___29446, int_29444)
        
        # Assigning a type to the variable 'chebcom' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 16), 'chebcom', subscript_call_result_29447)
        
        # Call to _qawoe(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 'func' (line 417)
        func_29450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 40), 'func', False)
        # Getting the type of 'a' (line 417)
        a_29451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 46), 'a', False)
        # Getting the type of 'b' (line 417)
        b_29452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 49), 'b', False)
        # Getting the type of 'wvar' (line 417)
        wvar_29453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 52), 'wvar', False)
        # Getting the type of 'integr' (line 417)
        integr_29454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 58), 'integr', False)
        # Getting the type of 'args' (line 417)
        args_29455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 66), 'args', False)
        # Getting the type of 'full_output' (line 417)
        full_output_29456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 72), 'full_output', False)
        # Getting the type of 'epsabs' (line 418)
        epsabs_29457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 40), 'epsabs', False)
        # Getting the type of 'epsrel' (line 418)
        epsrel_29458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 48), 'epsrel', False)
        # Getting the type of 'limit' (line 418)
        limit_29459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 56), 'limit', False)
        # Getting the type of 'maxp1' (line 418)
        maxp1_29460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 63), 'maxp1', False)
        int_29461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 70), 'int')
        # Getting the type of 'momcom' (line 418)
        momcom_29462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 73), 'momcom', False)
        # Getting the type of 'chebcom' (line 418)
        chebcom_29463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 81), 'chebcom', False)
        # Processing the call keyword arguments (line 417)
        kwargs_29464 = {}
        # Getting the type of '_quadpack' (line 417)
        _quadpack_29448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 23), '_quadpack', False)
        # Obtaining the member '_qawoe' of a type (line 417)
        _qawoe_29449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 23), _quadpack_29448, '_qawoe')
        # Calling _qawoe(args, kwargs) (line 417)
        _qawoe_call_result_29465 = invoke(stypy.reporting.localization.Localization(__file__, 417, 23), _qawoe_29449, *[func_29450, a_29451, b_29452, wvar_29453, integr_29454, args_29455, full_output_29456, epsabs_29457, epsrel_29458, limit_29459, maxp1_29460, int_29461, momcom_29462, chebcom_29463], **kwargs_29464)
        
        # Assigning a type to the variable 'stypy_return_type' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 16), 'stypy_return_type', _qawoe_call_result_29465)

        if (may_be_29422 and more_types_in_union_29423):
            # SSA join for if statement (line 411)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 410)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 420)
    b_29466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 14), 'b')
    # Getting the type of 'Inf' (line 420)
    Inf_29467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 19), 'Inf')
    # Applying the binary operator '==' (line 420)
    result_eq_29468 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 14), '==', b_29466, Inf_29467)
    
    
    # Getting the type of 'a' (line 420)
    a_29469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 27), 'a')
    
    # Getting the type of 'Inf' (line 420)
    Inf_29470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 33), 'Inf')
    # Applying the 'usub' unary operator (line 420)
    result___neg___29471 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 32), 'usub', Inf_29470)
    
    # Applying the binary operator '!=' (line 420)
    result_ne_29472 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 27), '!=', a_29469, result___neg___29471)
    
    # Applying the binary operator 'and' (line 420)
    result_and_keyword_29473 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 14), 'and', result_eq_29468, result_ne_29472)
    
    # Testing the type of an if condition (line 420)
    if_condition_29474 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 13), result_and_keyword_29473)
    # Assigning a type to the variable 'if_condition_29474' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 13), 'if_condition_29474', if_condition_29474)
    # SSA begins for if statement (line 420)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _qawfe(...): (line 421)
    # Processing the call arguments (line 421)
    # Getting the type of 'func' (line 421)
    func_29477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 36), 'func', False)
    # Getting the type of 'a' (line 421)
    a_29478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 42), 'a', False)
    # Getting the type of 'wvar' (line 421)
    wvar_29479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 45), 'wvar', False)
    # Getting the type of 'integr' (line 421)
    integr_29480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 51), 'integr', False)
    # Getting the type of 'args' (line 421)
    args_29481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 59), 'args', False)
    # Getting the type of 'full_output' (line 421)
    full_output_29482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 65), 'full_output', False)
    # Getting the type of 'epsabs' (line 422)
    epsabs_29483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 36), 'epsabs', False)
    # Getting the type of 'limlst' (line 422)
    limlst_29484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 43), 'limlst', False)
    # Getting the type of 'limit' (line 422)
    limit_29485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 50), 'limit', False)
    # Getting the type of 'maxp1' (line 422)
    maxp1_29486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 56), 'maxp1', False)
    # Processing the call keyword arguments (line 421)
    kwargs_29487 = {}
    # Getting the type of '_quadpack' (line 421)
    _quadpack_29475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 19), '_quadpack', False)
    # Obtaining the member '_qawfe' of a type (line 421)
    _qawfe_29476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 19), _quadpack_29475, '_qawfe')
    # Calling _qawfe(args, kwargs) (line 421)
    _qawfe_call_result_29488 = invoke(stypy.reporting.localization.Localization(__file__, 421, 19), _qawfe_29476, *[func_29477, a_29478, wvar_29479, integr_29480, args_29481, full_output_29482, epsabs_29483, limlst_29484, limit_29485, maxp1_29486], **kwargs_29487)
    
    # Assigning a type to the variable 'stypy_return_type' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'stypy_return_type', _qawfe_call_result_29488)
    # SSA branch for the else part of an if statement (line 420)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 423)
    b_29489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 14), 'b')
    # Getting the type of 'Inf' (line 423)
    Inf_29490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 19), 'Inf')
    # Applying the binary operator '!=' (line 423)
    result_ne_29491 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 14), '!=', b_29489, Inf_29490)
    
    
    # Getting the type of 'a' (line 423)
    a_29492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 27), 'a')
    
    # Getting the type of 'Inf' (line 423)
    Inf_29493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 33), 'Inf')
    # Applying the 'usub' unary operator (line 423)
    result___neg___29494 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 32), 'usub', Inf_29493)
    
    # Applying the binary operator '==' (line 423)
    result_eq_29495 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 27), '==', a_29492, result___neg___29494)
    
    # Applying the binary operator 'and' (line 423)
    result_and_keyword_29496 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 14), 'and', result_ne_29491, result_eq_29495)
    
    # Testing the type of an if condition (line 423)
    if_condition_29497 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 13), result_and_keyword_29496)
    # Assigning a type to the variable 'if_condition_29497' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 13), 'if_condition_29497', if_condition_29497)
    # SSA begins for if statement (line 423)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'weight' (line 424)
    weight_29498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 15), 'weight')
    str_29499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 25), 'str', 'cos')
    # Applying the binary operator '==' (line 424)
    result_eq_29500 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 15), '==', weight_29498, str_29499)
    
    # Testing the type of an if condition (line 424)
    if_condition_29501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 424, 12), result_eq_29500)
    # Assigning a type to the variable 'if_condition_29501' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'if_condition_29501', if_condition_29501)
    # SSA begins for if statement (line 424)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

    @norecursion
    def thefunc(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'thefunc'
        module_type_store = module_type_store.open_function_context('thefunc', 425, 16, False)
        
        # Passed parameters checking function
        thefunc.stypy_localization = localization
        thefunc.stypy_type_of_self = None
        thefunc.stypy_type_store = module_type_store
        thefunc.stypy_function_name = 'thefunc'
        thefunc.stypy_param_names_list = ['x']
        thefunc.stypy_varargs_param_name = 'myargs'
        thefunc.stypy_kwargs_param_name = None
        thefunc.stypy_call_defaults = defaults
        thefunc.stypy_call_varargs = varargs
        thefunc.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'thefunc', ['x'], 'myargs', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'thefunc', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'thefunc(...)' code ##################

        
        # Assigning a UnaryOp to a Name (line 426):
        
        # Assigning a UnaryOp to a Name (line 426):
        
        # Getting the type of 'x' (line 426)
        x_29502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 25), 'x')
        # Applying the 'usub' unary operator (line 426)
        result___neg___29503 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 24), 'usub', x_29502)
        
        # Assigning a type to the variable 'y' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 20), 'y', result___neg___29503)
        
        # Assigning a Subscript to a Name (line 427):
        
        # Assigning a Subscript to a Name (line 427):
        
        # Obtaining the type of the subscript
        int_29504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 34), 'int')
        # Getting the type of 'myargs' (line 427)
        myargs_29505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 27), 'myargs')
        # Obtaining the member '__getitem__' of a type (line 427)
        getitem___29506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 27), myargs_29505, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 427)
        subscript_call_result_29507 = invoke(stypy.reporting.localization.Localization(__file__, 427, 27), getitem___29506, int_29504)
        
        # Assigning a type to the variable 'func' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 20), 'func', subscript_call_result_29507)
        
        # Assigning a BinOp to a Name (line 428):
        
        # Assigning a BinOp to a Name (line 428):
        
        # Obtaining an instance of the builtin type 'tuple' (line 428)
        tuple_29508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 428)
        # Adding element type (line 428)
        # Getting the type of 'y' (line 428)
        y_29509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 30), 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 30), tuple_29508, y_29509)
        
        
        # Obtaining the type of the subscript
        int_29510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 43), 'int')
        slice_29511 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 428, 36), int_29510, None, None)
        # Getting the type of 'myargs' (line 428)
        myargs_29512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 36), 'myargs')
        # Obtaining the member '__getitem__' of a type (line 428)
        getitem___29513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 36), myargs_29512, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 428)
        subscript_call_result_29514 = invoke(stypy.reporting.localization.Localization(__file__, 428, 36), getitem___29513, slice_29511)
        
        # Applying the binary operator '+' (line 428)
        result_add_29515 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 29), '+', tuple_29508, subscript_call_result_29514)
        
        # Assigning a type to the variable 'myargs' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 20), 'myargs', result_add_29515)
        
        # Call to func(...): (line 429)
        # Getting the type of 'myargs' (line 429)
        myargs_29517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 33), 'myargs', False)
        # Processing the call keyword arguments (line 429)
        kwargs_29518 = {}
        # Getting the type of 'func' (line 429)
        func_29516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 27), 'func', False)
        # Calling func(args, kwargs) (line 429)
        func_call_result_29519 = invoke(stypy.reporting.localization.Localization(__file__, 429, 27), func_29516, *[myargs_29517], **kwargs_29518)
        
        # Assigning a type to the variable 'stypy_return_type' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 20), 'stypy_return_type', func_call_result_29519)
        
        # ################# End of 'thefunc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'thefunc' in the type store
        # Getting the type of 'stypy_return_type' (line 425)
        stypy_return_type_29520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29520)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'thefunc'
        return stypy_return_type_29520

    # Assigning a type to the variable 'thefunc' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'thefunc', thefunc)
    # SSA branch for the else part of an if statement (line 424)
    module_type_store.open_ssa_branch('else')

    @norecursion
    def thefunc(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'thefunc'
        module_type_store = module_type_store.open_function_context('thefunc', 431, 16, False)
        
        # Passed parameters checking function
        thefunc.stypy_localization = localization
        thefunc.stypy_type_of_self = None
        thefunc.stypy_type_store = module_type_store
        thefunc.stypy_function_name = 'thefunc'
        thefunc.stypy_param_names_list = ['x']
        thefunc.stypy_varargs_param_name = 'myargs'
        thefunc.stypy_kwargs_param_name = None
        thefunc.stypy_call_defaults = defaults
        thefunc.stypy_call_varargs = varargs
        thefunc.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'thefunc', ['x'], 'myargs', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'thefunc', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'thefunc(...)' code ##################

        
        # Assigning a UnaryOp to a Name (line 432):
        
        # Assigning a UnaryOp to a Name (line 432):
        
        # Getting the type of 'x' (line 432)
        x_29521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 25), 'x')
        # Applying the 'usub' unary operator (line 432)
        result___neg___29522 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 24), 'usub', x_29521)
        
        # Assigning a type to the variable 'y' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 20), 'y', result___neg___29522)
        
        # Assigning a Subscript to a Name (line 433):
        
        # Assigning a Subscript to a Name (line 433):
        
        # Obtaining the type of the subscript
        int_29523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 34), 'int')
        # Getting the type of 'myargs' (line 433)
        myargs_29524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 27), 'myargs')
        # Obtaining the member '__getitem__' of a type (line 433)
        getitem___29525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 27), myargs_29524, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 433)
        subscript_call_result_29526 = invoke(stypy.reporting.localization.Localization(__file__, 433, 27), getitem___29525, int_29523)
        
        # Assigning a type to the variable 'func' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'func', subscript_call_result_29526)
        
        # Assigning a BinOp to a Name (line 434):
        
        # Assigning a BinOp to a Name (line 434):
        
        # Obtaining an instance of the builtin type 'tuple' (line 434)
        tuple_29527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 434)
        # Adding element type (line 434)
        # Getting the type of 'y' (line 434)
        y_29528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 30), 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 30), tuple_29527, y_29528)
        
        
        # Obtaining the type of the subscript
        int_29529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 43), 'int')
        slice_29530 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 434, 36), int_29529, None, None)
        # Getting the type of 'myargs' (line 434)
        myargs_29531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 36), 'myargs')
        # Obtaining the member '__getitem__' of a type (line 434)
        getitem___29532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 36), myargs_29531, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 434)
        subscript_call_result_29533 = invoke(stypy.reporting.localization.Localization(__file__, 434, 36), getitem___29532, slice_29530)
        
        # Applying the binary operator '+' (line 434)
        result_add_29534 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 29), '+', tuple_29527, subscript_call_result_29533)
        
        # Assigning a type to the variable 'myargs' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 20), 'myargs', result_add_29534)
        
        
        # Call to func(...): (line 435)
        # Getting the type of 'myargs' (line 435)
        myargs_29536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 34), 'myargs', False)
        # Processing the call keyword arguments (line 435)
        kwargs_29537 = {}
        # Getting the type of 'func' (line 435)
        func_29535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 28), 'func', False)
        # Calling func(args, kwargs) (line 435)
        func_call_result_29538 = invoke(stypy.reporting.localization.Localization(__file__, 435, 28), func_29535, *[myargs_29536], **kwargs_29537)
        
        # Applying the 'usub' unary operator (line 435)
        result___neg___29539 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 27), 'usub', func_call_result_29538)
        
        # Assigning a type to the variable 'stypy_return_type' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'stypy_return_type', result___neg___29539)
        
        # ################# End of 'thefunc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'thefunc' in the type store
        # Getting the type of 'stypy_return_type' (line 431)
        stypy_return_type_29540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 16), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29540)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'thefunc'
        return stypy_return_type_29540

    # Assigning a type to the variable 'thefunc' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 16), 'thefunc', thefunc)
    # SSA join for if statement (line 424)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 436):
    
    # Assigning a BinOp to a Name (line 436):
    
    # Obtaining an instance of the builtin type 'tuple' (line 436)
    tuple_29541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 436)
    # Adding element type (line 436)
    # Getting the type of 'func' (line 436)
    func_29542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 20), 'func')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 20), tuple_29541, func_29542)
    
    # Getting the type of 'args' (line 436)
    args_29543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 29), 'args')
    # Applying the binary operator '+' (line 436)
    result_add_29544 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 19), '+', tuple_29541, args_29543)
    
    # Assigning a type to the variable 'args' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'args', result_add_29544)
    
    # Call to _qawfe(...): (line 437)
    # Processing the call arguments (line 437)
    # Getting the type of 'thefunc' (line 437)
    thefunc_29547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 36), 'thefunc', False)
    
    # Getting the type of 'b' (line 437)
    b_29548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 46), 'b', False)
    # Applying the 'usub' unary operator (line 437)
    result___neg___29549 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 45), 'usub', b_29548)
    
    # Getting the type of 'wvar' (line 437)
    wvar_29550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 49), 'wvar', False)
    # Getting the type of 'integr' (line 437)
    integr_29551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 55), 'integr', False)
    # Getting the type of 'args' (line 437)
    args_29552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 63), 'args', False)
    # Getting the type of 'full_output' (line 438)
    full_output_29553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 36), 'full_output', False)
    # Getting the type of 'epsabs' (line 438)
    epsabs_29554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 49), 'epsabs', False)
    # Getting the type of 'limlst' (line 438)
    limlst_29555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 57), 'limlst', False)
    # Getting the type of 'limit' (line 438)
    limit_29556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 65), 'limit', False)
    # Getting the type of 'maxp1' (line 438)
    maxp1_29557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 72), 'maxp1', False)
    # Processing the call keyword arguments (line 437)
    kwargs_29558 = {}
    # Getting the type of '_quadpack' (line 437)
    _quadpack_29545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 19), '_quadpack', False)
    # Obtaining the member '_qawfe' of a type (line 437)
    _qawfe_29546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 19), _quadpack_29545, '_qawfe')
    # Calling _qawfe(args, kwargs) (line 437)
    _qawfe_call_result_29559 = invoke(stypy.reporting.localization.Localization(__file__, 437, 19), _qawfe_29546, *[thefunc_29547, result___neg___29549, wvar_29550, integr_29551, args_29552, full_output_29553, epsabs_29554, limlst_29555, limit_29556, maxp1_29557], **kwargs_29558)
    
    # Assigning a type to the variable 'stypy_return_type' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'stypy_return_type', _qawfe_call_result_29559)
    # SSA branch for the else part of an if statement (line 423)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 440)
    # Processing the call arguments (line 440)
    str_29561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 29), 'str', 'Cannot integrate with this weight from -Inf to +Inf.')
    # Processing the call keyword arguments (line 440)
    kwargs_29562 = {}
    # Getting the type of 'ValueError' (line 440)
    ValueError_29560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 440)
    ValueError_call_result_29563 = invoke(stypy.reporting.localization.Localization(__file__, 440, 18), ValueError_29560, *[str_29561], **kwargs_29562)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 440, 12), ValueError_call_result_29563, 'raise parameter', BaseException)
    # SSA join for if statement (line 423)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 420)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 410)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 408)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'a' (line 442)
    a_29564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 11), 'a')
    
    # Obtaining an instance of the builtin type 'list' (line 442)
    list_29565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 442)
    # Adding element type (line 442)
    
    # Getting the type of 'Inf' (line 442)
    Inf_29566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 18), 'Inf')
    # Applying the 'usub' unary operator (line 442)
    result___neg___29567 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 17), 'usub', Inf_29566)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 16), list_29565, result___neg___29567)
    # Adding element type (line 442)
    # Getting the type of 'Inf' (line 442)
    Inf_29568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 22), 'Inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 16), list_29565, Inf_29568)
    
    # Applying the binary operator 'in' (line 442)
    result_contains_29569 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 11), 'in', a_29564, list_29565)
    
    
    # Getting the type of 'b' (line 442)
    b_29570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 30), 'b')
    
    # Obtaining an instance of the builtin type 'list' (line 442)
    list_29571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 442)
    # Adding element type (line 442)
    
    # Getting the type of 'Inf' (line 442)
    Inf_29572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 37), 'Inf')
    # Applying the 'usub' unary operator (line 442)
    result___neg___29573 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 36), 'usub', Inf_29572)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 35), list_29571, result___neg___29573)
    # Adding element type (line 442)
    # Getting the type of 'Inf' (line 442)
    Inf_29574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 41), 'Inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 35), list_29571, Inf_29574)
    
    # Applying the binary operator 'in' (line 442)
    result_contains_29575 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 30), 'in', b_29570, list_29571)
    
    # Applying the binary operator 'or' (line 442)
    result_or_keyword_29576 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 11), 'or', result_contains_29569, result_contains_29575)
    
    # Testing the type of an if condition (line 442)
    if_condition_29577 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 442, 8), result_or_keyword_29576)
    # Assigning a type to the variable 'if_condition_29577' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'if_condition_29577', if_condition_29577)
    # SSA begins for if statement (line 442)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 443)
    # Processing the call arguments (line 443)
    str_29579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 29), 'str', 'Cannot integrate with this weight over an infinite interval.')
    # Processing the call keyword arguments (line 443)
    kwargs_29580 = {}
    # Getting the type of 'ValueError' (line 443)
    ValueError_29578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 443)
    ValueError_call_result_29581 = invoke(stypy.reporting.localization.Localization(__file__, 443, 18), ValueError_29578, *[str_29579], **kwargs_29580)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 443, 12), ValueError_call_result_29581, 'raise parameter', BaseException)
    # SSA join for if statement (line 442)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_29582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 19), 'int')
    slice_29583 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 445, 11), None, int_29582, None)
    # Getting the type of 'weight' (line 445)
    weight_29584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 11), 'weight')
    # Obtaining the member '__getitem__' of a type (line 445)
    getitem___29585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 11), weight_29584, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 445)
    subscript_call_result_29586 = invoke(stypy.reporting.localization.Localization(__file__, 445, 11), getitem___29585, slice_29583)
    
    str_29587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 25), 'str', 'alg')
    # Applying the binary operator '==' (line 445)
    result_eq_29588 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 11), '==', subscript_call_result_29586, str_29587)
    
    # Testing the type of an if condition (line 445)
    if_condition_29589 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 445, 8), result_eq_29588)
    # Assigning a type to the variable 'if_condition_29589' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'if_condition_29589', if_condition_29589)
    # SSA begins for if statement (line 445)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 446):
    
    # Assigning a Subscript to a Name (line 446):
    
    # Obtaining the type of the subscript
    # Getting the type of 'weight' (line 446)
    weight_29590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 29), 'weight')
    # Getting the type of 'strdict' (line 446)
    strdict_29591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 21), 'strdict')
    # Obtaining the member '__getitem__' of a type (line 446)
    getitem___29592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 21), strdict_29591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 446)
    subscript_call_result_29593 = invoke(stypy.reporting.localization.Localization(__file__, 446, 21), getitem___29592, weight_29590)
    
    # Assigning a type to the variable 'integr' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'integr', subscript_call_result_29593)
    
    # Call to _qawse(...): (line 447)
    # Processing the call arguments (line 447)
    # Getting the type of 'func' (line 447)
    func_29596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 36), 'func', False)
    # Getting the type of 'a' (line 447)
    a_29597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 42), 'a', False)
    # Getting the type of 'b' (line 447)
    b_29598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 45), 'b', False)
    # Getting the type of 'wvar' (line 447)
    wvar_29599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 48), 'wvar', False)
    # Getting the type of 'integr' (line 447)
    integr_29600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 54), 'integr', False)
    # Getting the type of 'args' (line 447)
    args_29601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 62), 'args', False)
    # Getting the type of 'full_output' (line 448)
    full_output_29602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 36), 'full_output', False)
    # Getting the type of 'epsabs' (line 448)
    epsabs_29603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 49), 'epsabs', False)
    # Getting the type of 'epsrel' (line 448)
    epsrel_29604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 57), 'epsrel', False)
    # Getting the type of 'limit' (line 448)
    limit_29605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 65), 'limit', False)
    # Processing the call keyword arguments (line 447)
    kwargs_29606 = {}
    # Getting the type of '_quadpack' (line 447)
    _quadpack_29594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 19), '_quadpack', False)
    # Obtaining the member '_qawse' of a type (line 447)
    _qawse_29595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 19), _quadpack_29594, '_qawse')
    # Calling _qawse(args, kwargs) (line 447)
    _qawse_call_result_29607 = invoke(stypy.reporting.localization.Localization(__file__, 447, 19), _qawse_29595, *[func_29596, a_29597, b_29598, wvar_29599, integr_29600, args_29601, full_output_29602, epsabs_29603, epsrel_29604, limit_29605], **kwargs_29606)
    
    # Assigning a type to the variable 'stypy_return_type' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'stypy_return_type', _qawse_call_result_29607)
    # SSA branch for the else part of an if statement (line 445)
    module_type_store.open_ssa_branch('else')
    
    # Call to _qawce(...): (line 450)
    # Processing the call arguments (line 450)
    # Getting the type of 'func' (line 450)
    func_29610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 36), 'func', False)
    # Getting the type of 'a' (line 450)
    a_29611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 42), 'a', False)
    # Getting the type of 'b' (line 450)
    b_29612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 45), 'b', False)
    # Getting the type of 'wvar' (line 450)
    wvar_29613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 48), 'wvar', False)
    # Getting the type of 'args' (line 450)
    args_29614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 54), 'args', False)
    # Getting the type of 'full_output' (line 450)
    full_output_29615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 60), 'full_output', False)
    # Getting the type of 'epsabs' (line 451)
    epsabs_29616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 36), 'epsabs', False)
    # Getting the type of 'epsrel' (line 451)
    epsrel_29617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 44), 'epsrel', False)
    # Getting the type of 'limit' (line 451)
    limit_29618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 52), 'limit', False)
    # Processing the call keyword arguments (line 450)
    kwargs_29619 = {}
    # Getting the type of '_quadpack' (line 450)
    _quadpack_29608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 19), '_quadpack', False)
    # Obtaining the member '_qawce' of a type (line 450)
    _qawce_29609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 19), _quadpack_29608, '_qawce')
    # Calling _qawce(args, kwargs) (line 450)
    _qawce_call_result_29620 = invoke(stypy.reporting.localization.Localization(__file__, 450, 19), _qawce_29609, *[func_29610, a_29611, b_29612, wvar_29613, args_29614, full_output_29615, epsabs_29616, epsrel_29617, limit_29618], **kwargs_29619)
    
    # Assigning a type to the variable 'stypy_return_type' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'stypy_return_type', _qawce_call_result_29620)
    # SSA join for if statement (line 445)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 408)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_quad_weight(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_quad_weight' in the type store
    # Getting the type of 'stypy_return_type' (line 401)
    stypy_return_type_29621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29621)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_quad_weight'
    return stypy_return_type_29621

# Assigning a type to the variable '_quad_weight' (line 401)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 0), '_quad_weight', _quad_weight)

@norecursion
def dblquad(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 454)
    tuple_29622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 454)
    
    float_29623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 52), 'float')
    float_29624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 68), 'float')
    defaults = [tuple_29622, float_29623, float_29624]
    # Create a new context for function 'dblquad'
    module_type_store = module_type_store.open_function_context('dblquad', 454, 0, False)
    
    # Passed parameters checking function
    dblquad.stypy_localization = localization
    dblquad.stypy_type_of_self = None
    dblquad.stypy_type_store = module_type_store
    dblquad.stypy_function_name = 'dblquad'
    dblquad.stypy_param_names_list = ['func', 'a', 'b', 'gfun', 'hfun', 'args', 'epsabs', 'epsrel']
    dblquad.stypy_varargs_param_name = None
    dblquad.stypy_kwargs_param_name = None
    dblquad.stypy_call_defaults = defaults
    dblquad.stypy_call_varargs = varargs
    dblquad.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dblquad', ['func', 'a', 'b', 'gfun', 'hfun', 'args', 'epsabs', 'epsrel'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dblquad', localization, ['func', 'a', 'b', 'gfun', 'hfun', 'args', 'epsabs', 'epsrel'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dblquad(...)' code ##################

    str_29625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, (-1)), 'str', '\n    Compute a double integral.\n\n    Return the double (definite) integral of ``func(y, x)`` from ``x = a..b``\n    and ``y = gfun(x)..hfun(x)``.\n\n    Parameters\n    ----------\n    func : callable\n        A Python function or method of at least two variables: y must be the\n        first argument and x the second argument.\n    a, b : float\n        The limits of integration in x: `a` < `b`\n    gfun : callable\n        The lower boundary curve in y which is a function taking a single\n        floating point argument (x) and returning a floating point result: a\n        lambda function can be useful here.\n    hfun : callable\n        The upper boundary curve in y (same requirements as `gfun`).\n    args : sequence, optional\n        Extra arguments to pass to `func`.\n    epsabs : float, optional\n        Absolute tolerance passed directly to the inner 1-D quadrature\n        integration. Default is 1.49e-8.\n    epsrel : float, optional\n        Relative tolerance of the inner 1-D integrals. Default is 1.49e-8.\n\n    Returns\n    -------\n    y : float\n        The resultant integral.\n    abserr : float\n        An estimate of the error.\n\n    See also\n    --------\n    quad : single integral\n    tplquad : triple integral\n    nquad : N-dimensional integrals\n    fixed_quad : fixed-order Gaussian quadrature\n    quadrature : adaptive Gaussian quadrature\n    odeint : ODE integrator\n    ode : ODE integrator\n    simps : integrator for sampled data\n    romb : integrator for sampled data\n    scipy.special : for coefficients and roots of orthogonal polynomials\n\n    ')

    @norecursion
    def temp_ranges(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'temp_ranges'
        module_type_store = module_type_store.open_function_context('temp_ranges', 503, 4, False)
        
        # Passed parameters checking function
        temp_ranges.stypy_localization = localization
        temp_ranges.stypy_type_of_self = None
        temp_ranges.stypy_type_store = module_type_store
        temp_ranges.stypy_function_name = 'temp_ranges'
        temp_ranges.stypy_param_names_list = []
        temp_ranges.stypy_varargs_param_name = 'args'
        temp_ranges.stypy_kwargs_param_name = None
        temp_ranges.stypy_call_defaults = defaults
        temp_ranges.stypy_call_varargs = varargs
        temp_ranges.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'temp_ranges', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'temp_ranges', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'temp_ranges(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 504)
        list_29626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 504)
        # Adding element type (line 504)
        
        # Call to gfun(...): (line 504)
        # Processing the call arguments (line 504)
        
        # Obtaining the type of the subscript
        int_29628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 26), 'int')
        # Getting the type of 'args' (line 504)
        args_29629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 21), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 504)
        getitem___29630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 21), args_29629, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 504)
        subscript_call_result_29631 = invoke(stypy.reporting.localization.Localization(__file__, 504, 21), getitem___29630, int_29628)
        
        # Processing the call keyword arguments (line 504)
        kwargs_29632 = {}
        # Getting the type of 'gfun' (line 504)
        gfun_29627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 16), 'gfun', False)
        # Calling gfun(args, kwargs) (line 504)
        gfun_call_result_29633 = invoke(stypy.reporting.localization.Localization(__file__, 504, 16), gfun_29627, *[subscript_call_result_29631], **kwargs_29632)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 15), list_29626, gfun_call_result_29633)
        # Adding element type (line 504)
        
        # Call to hfun(...): (line 504)
        # Processing the call arguments (line 504)
        
        # Obtaining the type of the subscript
        int_29635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 41), 'int')
        # Getting the type of 'args' (line 504)
        args_29636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 36), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 504)
        getitem___29637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 36), args_29636, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 504)
        subscript_call_result_29638 = invoke(stypy.reporting.localization.Localization(__file__, 504, 36), getitem___29637, int_29635)
        
        # Processing the call keyword arguments (line 504)
        kwargs_29639 = {}
        # Getting the type of 'hfun' (line 504)
        hfun_29634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 31), 'hfun', False)
        # Calling hfun(args, kwargs) (line 504)
        hfun_call_result_29640 = invoke(stypy.reporting.localization.Localization(__file__, 504, 31), hfun_29634, *[subscript_call_result_29638], **kwargs_29639)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 15), list_29626, hfun_call_result_29640)
        
        # Assigning a type to the variable 'stypy_return_type' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'stypy_return_type', list_29626)
        
        # ################# End of 'temp_ranges(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'temp_ranges' in the type store
        # Getting the type of 'stypy_return_type' (line 503)
        stypy_return_type_29641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29641)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'temp_ranges'
        return stypy_return_type_29641

    # Assigning a type to the variable 'temp_ranges' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'temp_ranges', temp_ranges)
    
    # Call to nquad(...): (line 505)
    # Processing the call arguments (line 505)
    # Getting the type of 'func' (line 505)
    func_29643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 17), 'func', False)
    
    # Obtaining an instance of the builtin type 'list' (line 505)
    list_29644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 505)
    # Adding element type (line 505)
    # Getting the type of 'temp_ranges' (line 505)
    temp_ranges_29645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 24), 'temp_ranges', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 23), list_29644, temp_ranges_29645)
    # Adding element type (line 505)
    
    # Obtaining an instance of the builtin type 'list' (line 505)
    list_29646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 505)
    # Adding element type (line 505)
    # Getting the type of 'a' (line 505)
    a_29647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 38), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 37), list_29646, a_29647)
    # Adding element type (line 505)
    # Getting the type of 'b' (line 505)
    b_29648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 41), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 37), list_29646, b_29648)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 23), list_29644, list_29646)
    
    # Processing the call keyword arguments (line 505)
    # Getting the type of 'args' (line 505)
    args_29649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 51), 'args', False)
    keyword_29650 = args_29649
    
    # Obtaining an instance of the builtin type 'dict' (line 506)
    dict_29651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 506)
    # Adding element type (key, value) (line 506)
    str_29652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 18), 'str', 'epsabs')
    # Getting the type of 'epsabs' (line 506)
    epsabs_29653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 28), 'epsabs', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 17), dict_29651, (str_29652, epsabs_29653))
    # Adding element type (key, value) (line 506)
    str_29654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 36), 'str', 'epsrel')
    # Getting the type of 'epsrel' (line 506)
    epsrel_29655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 46), 'epsrel', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 17), dict_29651, (str_29654, epsrel_29655))
    
    keyword_29656 = dict_29651
    kwargs_29657 = {'args': keyword_29650, 'opts': keyword_29656}
    # Getting the type of 'nquad' (line 505)
    nquad_29642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 11), 'nquad', False)
    # Calling nquad(args, kwargs) (line 505)
    nquad_call_result_29658 = invoke(stypy.reporting.localization.Localization(__file__, 505, 11), nquad_29642, *[func_29643, list_29644], **kwargs_29657)
    
    # Assigning a type to the variable 'stypy_return_type' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'stypy_return_type', nquad_call_result_29658)
    
    # ################# End of 'dblquad(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dblquad' in the type store
    # Getting the type of 'stypy_return_type' (line 454)
    stypy_return_type_29659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29659)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dblquad'
    return stypy_return_type_29659

# Assigning a type to the variable 'dblquad' (line 454)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 0), 'dblquad', dblquad)

@norecursion
def tplquad(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 509)
    tuple_29660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 509)
    
    float_29661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 64), 'float')
    float_29662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 19), 'float')
    defaults = [tuple_29660, float_29661, float_29662]
    # Create a new context for function 'tplquad'
    module_type_store = module_type_store.open_function_context('tplquad', 509, 0, False)
    
    # Passed parameters checking function
    tplquad.stypy_localization = localization
    tplquad.stypy_type_of_self = None
    tplquad.stypy_type_store = module_type_store
    tplquad.stypy_function_name = 'tplquad'
    tplquad.stypy_param_names_list = ['func', 'a', 'b', 'gfun', 'hfun', 'qfun', 'rfun', 'args', 'epsabs', 'epsrel']
    tplquad.stypy_varargs_param_name = None
    tplquad.stypy_kwargs_param_name = None
    tplquad.stypy_call_defaults = defaults
    tplquad.stypy_call_varargs = varargs
    tplquad.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tplquad', ['func', 'a', 'b', 'gfun', 'hfun', 'qfun', 'rfun', 'args', 'epsabs', 'epsrel'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tplquad', localization, ['func', 'a', 'b', 'gfun', 'hfun', 'qfun', 'rfun', 'args', 'epsabs', 'epsrel'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tplquad(...)' code ##################

    str_29663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, (-1)), 'str', '\n    Compute a triple (definite) integral.\n\n    Return the triple integral of ``func(z, y, x)`` from ``x = a..b``,\n    ``y = gfun(x)..hfun(x)``, and ``z = qfun(x,y)..rfun(x,y)``.\n\n    Parameters\n    ----------\n    func : function\n        A Python function or method of at least three variables in the\n        order (z, y, x).\n    a, b : float\n        The limits of integration in x: `a` < `b`\n    gfun : function\n        The lower boundary curve in y which is a function taking a single\n        floating point argument (x) and returning a floating point result:\n        a lambda function can be useful here.\n    hfun : function\n        The upper boundary curve in y (same requirements as `gfun`).\n    qfun : function\n        The lower boundary surface in z.  It must be a function that takes\n        two floats in the order (x, y) and returns a float.\n    rfun : function\n        The upper boundary surface in z. (Same requirements as `qfun`.)\n    args : tuple, optional\n        Extra arguments to pass to `func`.\n    epsabs : float, optional\n        Absolute tolerance passed directly to the innermost 1-D quadrature\n        integration. Default is 1.49e-8.\n    epsrel : float, optional\n        Relative tolerance of the innermost 1-D integrals. Default is 1.49e-8.\n\n    Returns\n    -------\n    y : float\n        The resultant integral.\n    abserr : float\n        An estimate of the error.\n\n    See Also\n    --------\n    quad: Adaptive quadrature using QUADPACK\n    quadrature: Adaptive Gaussian quadrature\n    fixed_quad: Fixed-order Gaussian quadrature\n    dblquad: Double integrals\n    nquad : N-dimensional integrals\n    romb: Integrators for sampled data\n    simps: Integrators for sampled data\n    ode: ODE integrators\n    odeint: ODE integrators\n    scipy.special: For coefficients and roots of orthogonal polynomials\n\n    ')

    @norecursion
    def ranges0(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'ranges0'
        module_type_store = module_type_store.open_function_context('ranges0', 571, 4, False)
        
        # Passed parameters checking function
        ranges0.stypy_localization = localization
        ranges0.stypy_type_of_self = None
        ranges0.stypy_type_store = module_type_store
        ranges0.stypy_function_name = 'ranges0'
        ranges0.stypy_param_names_list = []
        ranges0.stypy_varargs_param_name = 'args'
        ranges0.stypy_kwargs_param_name = None
        ranges0.stypy_call_defaults = defaults
        ranges0.stypy_call_varargs = varargs
        ranges0.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'ranges0', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ranges0', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ranges0(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 572)
        list_29664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 572)
        # Adding element type (line 572)
        
        # Call to qfun(...): (line 572)
        # Processing the call arguments (line 572)
        
        # Obtaining the type of the subscript
        int_29666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 26), 'int')
        # Getting the type of 'args' (line 572)
        args_29667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 21), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 572)
        getitem___29668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 21), args_29667, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 572)
        subscript_call_result_29669 = invoke(stypy.reporting.localization.Localization(__file__, 572, 21), getitem___29668, int_29666)
        
        
        # Obtaining the type of the subscript
        int_29670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 35), 'int')
        # Getting the type of 'args' (line 572)
        args_29671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 30), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 572)
        getitem___29672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 30), args_29671, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 572)
        subscript_call_result_29673 = invoke(stypy.reporting.localization.Localization(__file__, 572, 30), getitem___29672, int_29670)
        
        # Processing the call keyword arguments (line 572)
        kwargs_29674 = {}
        # Getting the type of 'qfun' (line 572)
        qfun_29665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 16), 'qfun', False)
        # Calling qfun(args, kwargs) (line 572)
        qfun_call_result_29675 = invoke(stypy.reporting.localization.Localization(__file__, 572, 16), qfun_29665, *[subscript_call_result_29669, subscript_call_result_29673], **kwargs_29674)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 15), list_29664, qfun_call_result_29675)
        # Adding element type (line 572)
        
        # Call to rfun(...): (line 572)
        # Processing the call arguments (line 572)
        
        # Obtaining the type of the subscript
        int_29677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 50), 'int')
        # Getting the type of 'args' (line 572)
        args_29678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 45), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 572)
        getitem___29679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 45), args_29678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 572)
        subscript_call_result_29680 = invoke(stypy.reporting.localization.Localization(__file__, 572, 45), getitem___29679, int_29677)
        
        
        # Obtaining the type of the subscript
        int_29681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 59), 'int')
        # Getting the type of 'args' (line 572)
        args_29682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 54), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 572)
        getitem___29683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 54), args_29682, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 572)
        subscript_call_result_29684 = invoke(stypy.reporting.localization.Localization(__file__, 572, 54), getitem___29683, int_29681)
        
        # Processing the call keyword arguments (line 572)
        kwargs_29685 = {}
        # Getting the type of 'rfun' (line 572)
        rfun_29676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 40), 'rfun', False)
        # Calling rfun(args, kwargs) (line 572)
        rfun_call_result_29686 = invoke(stypy.reporting.localization.Localization(__file__, 572, 40), rfun_29676, *[subscript_call_result_29680, subscript_call_result_29684], **kwargs_29685)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 15), list_29664, rfun_call_result_29686)
        
        # Assigning a type to the variable 'stypy_return_type' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'stypy_return_type', list_29664)
        
        # ################# End of 'ranges0(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ranges0' in the type store
        # Getting the type of 'stypy_return_type' (line 571)
        stypy_return_type_29687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29687)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ranges0'
        return stypy_return_type_29687

    # Assigning a type to the variable 'ranges0' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 4), 'ranges0', ranges0)

    @norecursion
    def ranges1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'ranges1'
        module_type_store = module_type_store.open_function_context('ranges1', 574, 4, False)
        
        # Passed parameters checking function
        ranges1.stypy_localization = localization
        ranges1.stypy_type_of_self = None
        ranges1.stypy_type_store = module_type_store
        ranges1.stypy_function_name = 'ranges1'
        ranges1.stypy_param_names_list = []
        ranges1.stypy_varargs_param_name = 'args'
        ranges1.stypy_kwargs_param_name = None
        ranges1.stypy_call_defaults = defaults
        ranges1.stypy_call_varargs = varargs
        ranges1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'ranges1', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ranges1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ranges1(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 575)
        list_29688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 575)
        # Adding element type (line 575)
        
        # Call to gfun(...): (line 575)
        # Processing the call arguments (line 575)
        
        # Obtaining the type of the subscript
        int_29690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 26), 'int')
        # Getting the type of 'args' (line 575)
        args_29691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 21), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 575)
        getitem___29692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 21), args_29691, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 575)
        subscript_call_result_29693 = invoke(stypy.reporting.localization.Localization(__file__, 575, 21), getitem___29692, int_29690)
        
        # Processing the call keyword arguments (line 575)
        kwargs_29694 = {}
        # Getting the type of 'gfun' (line 575)
        gfun_29689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'gfun', False)
        # Calling gfun(args, kwargs) (line 575)
        gfun_call_result_29695 = invoke(stypy.reporting.localization.Localization(__file__, 575, 16), gfun_29689, *[subscript_call_result_29693], **kwargs_29694)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 15), list_29688, gfun_call_result_29695)
        # Adding element type (line 575)
        
        # Call to hfun(...): (line 575)
        # Processing the call arguments (line 575)
        
        # Obtaining the type of the subscript
        int_29697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 41), 'int')
        # Getting the type of 'args' (line 575)
        args_29698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 36), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 575)
        getitem___29699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 36), args_29698, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 575)
        subscript_call_result_29700 = invoke(stypy.reporting.localization.Localization(__file__, 575, 36), getitem___29699, int_29697)
        
        # Processing the call keyword arguments (line 575)
        kwargs_29701 = {}
        # Getting the type of 'hfun' (line 575)
        hfun_29696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 31), 'hfun', False)
        # Calling hfun(args, kwargs) (line 575)
        hfun_call_result_29702 = invoke(stypy.reporting.localization.Localization(__file__, 575, 31), hfun_29696, *[subscript_call_result_29700], **kwargs_29701)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 15), list_29688, hfun_call_result_29702)
        
        # Assigning a type to the variable 'stypy_return_type' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'stypy_return_type', list_29688)
        
        # ################# End of 'ranges1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ranges1' in the type store
        # Getting the type of 'stypy_return_type' (line 574)
        stypy_return_type_29703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29703)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ranges1'
        return stypy_return_type_29703

    # Assigning a type to the variable 'ranges1' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'ranges1', ranges1)
    
    # Assigning a List to a Name (line 577):
    
    # Assigning a List to a Name (line 577):
    
    # Obtaining an instance of the builtin type 'list' (line 577)
    list_29704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 577)
    # Adding element type (line 577)
    # Getting the type of 'ranges0' (line 577)
    ranges0_29705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 14), 'ranges0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 13), list_29704, ranges0_29705)
    # Adding element type (line 577)
    # Getting the type of 'ranges1' (line 577)
    ranges1_29706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 23), 'ranges1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 13), list_29704, ranges1_29706)
    # Adding element type (line 577)
    
    # Obtaining an instance of the builtin type 'list' (line 577)
    list_29707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 577)
    # Adding element type (line 577)
    # Getting the type of 'a' (line 577)
    a_29708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 33), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 32), list_29707, a_29708)
    # Adding element type (line 577)
    # Getting the type of 'b' (line 577)
    b_29709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 36), 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 32), list_29707, b_29709)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 13), list_29704, list_29707)
    
    # Assigning a type to the variable 'ranges' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'ranges', list_29704)
    
    # Call to nquad(...): (line 578)
    # Processing the call arguments (line 578)
    # Getting the type of 'func' (line 578)
    func_29711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 17), 'func', False)
    # Getting the type of 'ranges' (line 578)
    ranges_29712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 23), 'ranges', False)
    # Processing the call keyword arguments (line 578)
    # Getting the type of 'args' (line 578)
    args_29713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 36), 'args', False)
    keyword_29714 = args_29713
    
    # Obtaining an instance of the builtin type 'dict' (line 579)
    dict_29715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 579)
    # Adding element type (key, value) (line 579)
    str_29716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 18), 'str', 'epsabs')
    # Getting the type of 'epsabs' (line 579)
    epsabs_29717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 28), 'epsabs', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 17), dict_29715, (str_29716, epsabs_29717))
    # Adding element type (key, value) (line 579)
    str_29718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 36), 'str', 'epsrel')
    # Getting the type of 'epsrel' (line 579)
    epsrel_29719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 46), 'epsrel', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 17), dict_29715, (str_29718, epsrel_29719))
    
    keyword_29720 = dict_29715
    kwargs_29721 = {'args': keyword_29714, 'opts': keyword_29720}
    # Getting the type of 'nquad' (line 578)
    nquad_29710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 11), 'nquad', False)
    # Calling nquad(args, kwargs) (line 578)
    nquad_call_result_29722 = invoke(stypy.reporting.localization.Localization(__file__, 578, 11), nquad_29710, *[func_29711, ranges_29712], **kwargs_29721)
    
    # Assigning a type to the variable 'stypy_return_type' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'stypy_return_type', nquad_call_result_29722)
    
    # ################# End of 'tplquad(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tplquad' in the type store
    # Getting the type of 'stypy_return_type' (line 509)
    stypy_return_type_29723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29723)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tplquad'
    return stypy_return_type_29723

# Assigning a type to the variable 'tplquad' (line 509)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 0), 'tplquad', tplquad)

@norecursion
def nquad(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 582)
    None_29724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 29), 'None')
    # Getting the type of 'None' (line 582)
    None_29725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 40), 'None')
    # Getting the type of 'False' (line 582)
    False_29726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 58), 'False')
    defaults = [None_29724, None_29725, False_29726]
    # Create a new context for function 'nquad'
    module_type_store = module_type_store.open_function_context('nquad', 582, 0, False)
    
    # Passed parameters checking function
    nquad.stypy_localization = localization
    nquad.stypy_type_of_self = None
    nquad.stypy_type_store = module_type_store
    nquad.stypy_function_name = 'nquad'
    nquad.stypy_param_names_list = ['func', 'ranges', 'args', 'opts', 'full_output']
    nquad.stypy_varargs_param_name = None
    nquad.stypy_kwargs_param_name = None
    nquad.stypy_call_defaults = defaults
    nquad.stypy_call_varargs = varargs
    nquad.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nquad', ['func', 'ranges', 'args', 'opts', 'full_output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nquad', localization, ['func', 'ranges', 'args', 'opts', 'full_output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nquad(...)' code ##################

    str_29727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, (-1)), 'str', "\n    Integration over multiple variables.\n\n    Wraps `quad` to enable integration over multiple variables.\n    Various options allow improved integration of discontinuous functions, as\n    well as the use of weighted integration, and generally finer control of the\n    integration process.\n\n    Parameters\n    ----------\n    func : {callable, scipy.LowLevelCallable}\n        The function to be integrated. Has arguments of ``x0, ... xn``,\n        ``t0, tm``, where integration is carried out over ``x0, ... xn``, which\n        must be floats.  Function signature should be\n        ``func(x0, x1, ..., xn, t0, t1, ..., tm)``.  Integration is carried out\n        in order.  That is, integration over ``x0`` is the innermost integral,\n        and ``xn`` is the outermost.\n\n        If the user desires improved integration performance, then `f` may\n        be a `scipy.LowLevelCallable` with one of the signatures::\n\n            double func(int n, double *xx)\n            double func(int n, double *xx, void *user_data)\n\n        where ``n`` is the number of extra parameters and args is an array\n        of doubles of the additional parameters, the ``xx`` array contains the \n        coordinates. The ``user_data`` is the data contained in the\n        `scipy.LowLevelCallable`.\n    ranges : iterable object\n        Each element of ranges may be either a sequence  of 2 numbers, or else\n        a callable that returns such a sequence.  ``ranges[0]`` corresponds to\n        integration over x0, and so on.  If an element of ranges is a callable,\n        then it will be called with all of the integration arguments available,\n        as well as any parametric arguments. e.g. if \n        ``func = f(x0, x1, x2, t0, t1)``, then ``ranges[0]`` may be defined as\n        either ``(a, b)`` or else as ``(a, b) = range0(x1, x2, t0, t1)``.\n    args : iterable object, optional\n        Additional arguments ``t0, ..., tn``, required by `func`, `ranges`, and\n        ``opts``.\n    opts : iterable object or dict, optional\n        Options to be passed to `quad`.  May be empty, a dict, or\n        a sequence of dicts or functions that return a dict.  If empty, the\n        default options from scipy.integrate.quad are used.  If a dict, the same\n        options are used for all levels of integraion.  If a sequence, then each\n        element of the sequence corresponds to a particular integration. e.g.\n        opts[0] corresponds to integration over x0, and so on. If a callable, \n        the signature must be the same as for ``ranges``. The available\n        options together with their default values are:\n\n          - epsabs = 1.49e-08\n          - epsrel = 1.49e-08\n          - limit  = 50\n          - points = None\n          - weight = None\n          - wvar   = None\n          - wopts  = None\n\n        For more information on these options, see `quad` and `quad_explain`.\n\n    full_output : bool, optional\n        Partial implementation of ``full_output`` from scipy.integrate.quad. \n        The number of integrand function evaluations ``neval`` can be obtained \n        by setting ``full_output=True`` when calling nquad.\n\n    Returns\n    -------\n    result : float\n        The result of the integration.\n    abserr : float\n        The maximum of the estimates of the absolute error in the various\n        integration results.\n    out_dict : dict, optional\n        A dict containing additional information on the integration. \n\n    See Also\n    --------\n    quad : 1-dimensional numerical integration\n    dblquad, tplquad : double and triple integrals\n    fixed_quad : fixed-order Gaussian quadrature\n    quadrature : adaptive Gaussian quadrature\n\n    Examples\n    --------\n    >>> from scipy import integrate\n    >>> func = lambda x0,x1,x2,x3 : x0**2 + x1*x2 - x3**3 + np.sin(x0) + (\n    ...                                 1 if (x0-.2*x3-.5-.25*x1>0) else 0)\n    >>> points = [[lambda x1,x2,x3 : 0.2*x3 + 0.5 + 0.25*x1], [], [], []]\n    >>> def opts0(*args, **kwargs):\n    ...     return {'points':[0.2*args[2] + 0.5 + 0.25*args[0]]}\n    >>> integrate.nquad(func, [[0,1], [-1,1], [.13,.8], [-.15,1]],\n    ...                 opts=[opts0,{},{},{}], full_output=True)\n    (1.5267454070738633, 2.9437360001402324e-14, {'neval': 388962})\n\n    >>> scale = .1\n    >>> def func2(x0, x1, x2, x3, t0, t1):\n    ...     return x0*x1*x3**2 + np.sin(x2) + 1 + (1 if x0+t1*x1-t0>0 else 0)\n    >>> def lim0(x1, x2, x3, t0, t1):\n    ...     return [scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) - 1,\n    ...             scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) + 1]\n    >>> def lim1(x2, x3, t0, t1):\n    ...     return [scale * (t0*x2 + t1*x3) - 1,\n    ...             scale * (t0*x2 + t1*x3) + 1]\n    >>> def lim2(x3, t0, t1):\n    ...     return [scale * (x3 + t0**2*t1**3) - 1,\n    ...             scale * (x3 + t0**2*t1**3) + 1]\n    >>> def lim3(t0, t1):\n    ...     return [scale * (t0+t1) - 1, scale * (t0+t1) + 1]\n    >>> def opts0(x1, x2, x3, t0, t1):\n    ...     return {'points' : [t0 - t1*x1]}\n    >>> def opts1(x2, x3, t0, t1):\n    ...     return {}\n    >>> def opts2(x3, t0, t1):\n    ...     return {}\n    >>> def opts3(t0, t1):\n    ...     return {}\n    >>> integrate.nquad(func2, [lim0, lim1, lim2, lim3], args=(0,0),\n    ...                 opts=[opts0, opts1, opts2, opts3])\n    (25.066666666666666, 2.7829590483937256e-13)\n\n    ")
    
    # Assigning a Call to a Name (line 703):
    
    # Assigning a Call to a Name (line 703):
    
    # Call to len(...): (line 703)
    # Processing the call arguments (line 703)
    # Getting the type of 'ranges' (line 703)
    ranges_29729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 16), 'ranges', False)
    # Processing the call keyword arguments (line 703)
    kwargs_29730 = {}
    # Getting the type of 'len' (line 703)
    len_29728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 12), 'len', False)
    # Calling len(args, kwargs) (line 703)
    len_call_result_29731 = invoke(stypy.reporting.localization.Localization(__file__, 703, 12), len_29728, *[ranges_29729], **kwargs_29730)
    
    # Assigning a type to the variable 'depth' (line 703)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 4), 'depth', len_call_result_29731)
    
    # Assigning a ListComp to a Name (line 704):
    
    # Assigning a ListComp to a Name (line 704):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ranges' (line 704)
    ranges_29742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 67), 'ranges')
    comprehension_29743 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 704, 14), ranges_29742)
    # Assigning a type to the variable 'rng' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 14), 'rng', comprehension_29743)
    
    
    # Call to callable(...): (line 704)
    # Processing the call arguments (line 704)
    # Getting the type of 'rng' (line 704)
    rng_29733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 30), 'rng', False)
    # Processing the call keyword arguments (line 704)
    kwargs_29734 = {}
    # Getting the type of 'callable' (line 704)
    callable_29732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 21), 'callable', False)
    # Calling callable(args, kwargs) (line 704)
    callable_call_result_29735 = invoke(stypy.reporting.localization.Localization(__file__, 704, 21), callable_29732, *[rng_29733], **kwargs_29734)
    
    # Testing the type of an if expression (line 704)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 704, 14), callable_call_result_29735)
    # SSA begins for if expression (line 704)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'rng' (line 704)
    rng_29736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 14), 'rng')
    # SSA branch for the else part of an if expression (line 704)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to _RangeFunc(...): (line 704)
    # Processing the call arguments (line 704)
    # Getting the type of 'rng' (line 704)
    rng_29738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 51), 'rng', False)
    # Processing the call keyword arguments (line 704)
    kwargs_29739 = {}
    # Getting the type of '_RangeFunc' (line 704)
    _RangeFunc_29737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 40), '_RangeFunc', False)
    # Calling _RangeFunc(args, kwargs) (line 704)
    _RangeFunc_call_result_29740 = invoke(stypy.reporting.localization.Localization(__file__, 704, 40), _RangeFunc_29737, *[rng_29738], **kwargs_29739)
    
    # SSA join for if expression (line 704)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_29741 = union_type.UnionType.add(rng_29736, _RangeFunc_call_result_29740)
    
    list_29744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 704, 14), list_29744, if_exp_29741)
    # Assigning a type to the variable 'ranges' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'ranges', list_29744)
    
    # Type idiom detected: calculating its left and rigth part (line 705)
    # Getting the type of 'args' (line 705)
    args_29745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 7), 'args')
    # Getting the type of 'None' (line 705)
    None_29746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 15), 'None')
    
    (may_be_29747, more_types_in_union_29748) = may_be_none(args_29745, None_29746)

    if may_be_29747:

        if more_types_in_union_29748:
            # Runtime conditional SSA (line 705)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Tuple to a Name (line 706):
        
        # Assigning a Tuple to a Name (line 706):
        
        # Obtaining an instance of the builtin type 'tuple' (line 706)
        tuple_29749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 706)
        
        # Assigning a type to the variable 'args' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'args', tuple_29749)

        if more_types_in_union_29748:
            # SSA join for if statement (line 705)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 707)
    # Getting the type of 'opts' (line 707)
    opts_29750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 7), 'opts')
    # Getting the type of 'None' (line 707)
    None_29751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 15), 'None')
    
    (may_be_29752, more_types_in_union_29753) = may_be_none(opts_29750, None_29751)

    if may_be_29752:

        if more_types_in_union_29753:
            # Runtime conditional SSA (line 707)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 708):
        
        # Assigning a BinOp to a Name (line 708):
        
        # Obtaining an instance of the builtin type 'list' (line 708)
        list_29754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 708)
        # Adding element type (line 708)
        
        # Call to dict(...): (line 708)
        # Processing the call arguments (line 708)
        
        # Obtaining an instance of the builtin type 'list' (line 708)
        list_29756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 708)
        
        # Processing the call keyword arguments (line 708)
        kwargs_29757 = {}
        # Getting the type of 'dict' (line 708)
        dict_29755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 16), 'dict', False)
        # Calling dict(args, kwargs) (line 708)
        dict_call_result_29758 = invoke(stypy.reporting.localization.Localization(__file__, 708, 16), dict_29755, *[list_29756], **kwargs_29757)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 15), list_29754, dict_call_result_29758)
        
        # Getting the type of 'depth' (line 708)
        depth_29759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 28), 'depth')
        # Applying the binary operator '*' (line 708)
        result_mul_29760 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 15), '*', list_29754, depth_29759)
        
        # Assigning a type to the variable 'opts' (line 708)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'opts', result_mul_29760)

        if more_types_in_union_29753:
            # SSA join for if statement (line 707)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 710)
    # Getting the type of 'dict' (line 710)
    dict_29761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 24), 'dict')
    # Getting the type of 'opts' (line 710)
    opts_29762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 18), 'opts')
    
    (may_be_29763, more_types_in_union_29764) = may_be_subtype(dict_29761, opts_29762)

    if may_be_29763:

        if more_types_in_union_29764:
            # Runtime conditional SSA (line 710)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'opts' (line 710)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'opts', remove_not_subtype_from_union(opts_29762, dict))
        
        # Assigning a BinOp to a Name (line 711):
        
        # Assigning a BinOp to a Name (line 711):
        
        # Obtaining an instance of the builtin type 'list' (line 711)
        list_29765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 711)
        # Adding element type (line 711)
        
        # Call to _OptFunc(...): (line 711)
        # Processing the call arguments (line 711)
        # Getting the type of 'opts' (line 711)
        opts_29767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 25), 'opts', False)
        # Processing the call keyword arguments (line 711)
        kwargs_29768 = {}
        # Getting the type of '_OptFunc' (line 711)
        _OptFunc_29766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 16), '_OptFunc', False)
        # Calling _OptFunc(args, kwargs) (line 711)
        _OptFunc_call_result_29769 = invoke(stypy.reporting.localization.Localization(__file__, 711, 16), _OptFunc_29766, *[opts_29767], **kwargs_29768)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 15), list_29765, _OptFunc_call_result_29769)
        
        # Getting the type of 'depth' (line 711)
        depth_29770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 34), 'depth')
        # Applying the binary operator '*' (line 711)
        result_mul_29771 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 15), '*', list_29765, depth_29770)
        
        # Assigning a type to the variable 'opts' (line 711)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'opts', result_mul_29771)

        if more_types_in_union_29764:
            # Runtime conditional SSA for else branch (line 710)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_29763) or more_types_in_union_29764):
        # Assigning a type to the variable 'opts' (line 710)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'opts', remove_subtype_from_union(opts_29762, dict))
        
        # Assigning a ListComp to a Name (line 713):
        
        # Assigning a ListComp to a Name (line 713):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'opts' (line 713)
        opts_29782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 67), 'opts')
        comprehension_29783 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 16), opts_29782)
        # Assigning a type to the variable 'opt' (line 713)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 16), 'opt', comprehension_29783)
        
        
        # Call to callable(...): (line 713)
        # Processing the call arguments (line 713)
        # Getting the type of 'opt' (line 713)
        opt_29773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 32), 'opt', False)
        # Processing the call keyword arguments (line 713)
        kwargs_29774 = {}
        # Getting the type of 'callable' (line 713)
        callable_29772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 23), 'callable', False)
        # Calling callable(args, kwargs) (line 713)
        callable_call_result_29775 = invoke(stypy.reporting.localization.Localization(__file__, 713, 23), callable_29772, *[opt_29773], **kwargs_29774)
        
        # Testing the type of an if expression (line 713)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 713, 16), callable_call_result_29775)
        # SSA begins for if expression (line 713)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'opt' (line 713)
        opt_29776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 16), 'opt')
        # SSA branch for the else part of an if expression (line 713)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to _OptFunc(...): (line 713)
        # Processing the call arguments (line 713)
        # Getting the type of 'opt' (line 713)
        opt_29778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 51), 'opt', False)
        # Processing the call keyword arguments (line 713)
        kwargs_29779 = {}
        # Getting the type of '_OptFunc' (line 713)
        _OptFunc_29777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 42), '_OptFunc', False)
        # Calling _OptFunc(args, kwargs) (line 713)
        _OptFunc_call_result_29780 = invoke(stypy.reporting.localization.Localization(__file__, 713, 42), _OptFunc_29777, *[opt_29778], **kwargs_29779)
        
        # SSA join for if expression (line 713)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_29781 = union_type.UnionType.add(opt_29776, _OptFunc_call_result_29780)
        
        list_29784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 16), list_29784, if_exp_29781)
        # Assigning a type to the variable 'opts' (line 713)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'opts', list_29784)

        if (may_be_29763 and more_types_in_union_29764):
            # SSA join for if statement (line 710)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to integrate(...): (line 714)
    # Getting the type of 'args' (line 714)
    args_29793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 62), 'args', False)
    # Processing the call keyword arguments (line 714)
    kwargs_29794 = {}
    
    # Call to _NQuad(...): (line 714)
    # Processing the call arguments (line 714)
    # Getting the type of 'func' (line 714)
    func_29786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 18), 'func', False)
    # Getting the type of 'ranges' (line 714)
    ranges_29787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 24), 'ranges', False)
    # Getting the type of 'opts' (line 714)
    opts_29788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 32), 'opts', False)
    # Getting the type of 'full_output' (line 714)
    full_output_29789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 38), 'full_output', False)
    # Processing the call keyword arguments (line 714)
    kwargs_29790 = {}
    # Getting the type of '_NQuad' (line 714)
    _NQuad_29785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 11), '_NQuad', False)
    # Calling _NQuad(args, kwargs) (line 714)
    _NQuad_call_result_29791 = invoke(stypy.reporting.localization.Localization(__file__, 714, 11), _NQuad_29785, *[func_29786, ranges_29787, opts_29788, full_output_29789], **kwargs_29790)
    
    # Obtaining the member 'integrate' of a type (line 714)
    integrate_29792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 11), _NQuad_call_result_29791, 'integrate')
    # Calling integrate(args, kwargs) (line 714)
    integrate_call_result_29795 = invoke(stypy.reporting.localization.Localization(__file__, 714, 11), integrate_29792, *[args_29793], **kwargs_29794)
    
    # Assigning a type to the variable 'stypy_return_type' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'stypy_return_type', integrate_call_result_29795)
    
    # ################# End of 'nquad(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nquad' in the type store
    # Getting the type of 'stypy_return_type' (line 582)
    stypy_return_type_29796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29796)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nquad'
    return stypy_return_type_29796

# Assigning a type to the variable 'nquad' (line 582)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 0), 'nquad', nquad)
# Declaration of the '_RangeFunc' class

class _RangeFunc(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 718, 4, False)
        # Assigning a type to the variable 'self' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_RangeFunc.__init__', ['range_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['range_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 719):
        
        # Assigning a Name to a Attribute (line 719):
        # Getting the type of 'range_' (line 719)
        range__29797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 22), 'range_')
        # Getting the type of 'self' (line 719)
        self_29798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'self')
        # Setting the type of the member 'range_' of a type (line 719)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 8), self_29798, 'range_', range__29797)
        
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
        module_type_store = module_type_store.open_function_context('__call__', 721, 4, False)
        # Assigning a type to the variable 'self' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _RangeFunc.__call__.__dict__.__setitem__('stypy_localization', localization)
        _RangeFunc.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _RangeFunc.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _RangeFunc.__call__.__dict__.__setitem__('stypy_function_name', '_RangeFunc.__call__')
        _RangeFunc.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        _RangeFunc.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        _RangeFunc.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _RangeFunc.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _RangeFunc.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _RangeFunc.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _RangeFunc.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_RangeFunc.__call__', [], 'args', None, defaults, varargs, kwargs)

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

        str_29799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, (-1)), 'str', 'Return stored value.\n\n        *args needed because range_ can be float or func, and is called with\n        variable number of parameters.\n        ')
        # Getting the type of 'self' (line 727)
        self_29800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 15), 'self')
        # Obtaining the member 'range_' of a type (line 727)
        range__29801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 15), self_29800, 'range_')
        # Assigning a type to the variable 'stypy_return_type' (line 727)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'stypy_return_type', range__29801)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 721)
        stypy_return_type_29802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29802)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_29802


# Assigning a type to the variable '_RangeFunc' (line 717)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 0), '_RangeFunc', _RangeFunc)
# Declaration of the '_OptFunc' class

class _OptFunc(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 731, 4, False)
        # Assigning a type to the variable 'self' (line 732)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_OptFunc.__init__', ['opt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['opt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 732):
        
        # Assigning a Name to a Attribute (line 732):
        # Getting the type of 'opt' (line 732)
        opt_29803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 19), 'opt')
        # Getting the type of 'self' (line 732)
        self_29804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 8), 'self')
        # Setting the type of the member 'opt' of a type (line 732)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 8), self_29804, 'opt', opt_29803)
        
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
        module_type_store = module_type_store.open_function_context('__call__', 734, 4, False)
        # Assigning a type to the variable 'self' (line 735)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _OptFunc.__call__.__dict__.__setitem__('stypy_localization', localization)
        _OptFunc.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _OptFunc.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _OptFunc.__call__.__dict__.__setitem__('stypy_function_name', '_OptFunc.__call__')
        _OptFunc.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        _OptFunc.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        _OptFunc.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _OptFunc.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _OptFunc.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _OptFunc.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _OptFunc.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_OptFunc.__call__', [], 'args', None, defaults, varargs, kwargs)

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

        str_29805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 8), 'str', 'Return stored dict.')
        # Getting the type of 'self' (line 736)
        self_29806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 15), 'self')
        # Obtaining the member 'opt' of a type (line 736)
        opt_29807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 15), self_29806, 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 8), 'stypy_return_type', opt_29807)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 734)
        stypy_return_type_29808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29808)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_29808


# Assigning a type to the variable '_OptFunc' (line 730)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 0), '_OptFunc', _OptFunc)
# Declaration of the '_NQuad' class

class _NQuad(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 740, 4, False)
        # Assigning a type to the variable 'self' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_NQuad.__init__', ['func', 'ranges', 'opts', 'full_output'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['func', 'ranges', 'opts', 'full_output'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Num to a Attribute (line 741):
        
        # Assigning a Num to a Attribute (line 741):
        int_29809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 22), 'int')
        # Getting the type of 'self' (line 741)
        self_29810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'self')
        # Setting the type of the member 'abserr' of a type (line 741)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 8), self_29810, 'abserr', int_29809)
        
        # Assigning a Name to a Attribute (line 742):
        
        # Assigning a Name to a Attribute (line 742):
        # Getting the type of 'func' (line 742)
        func_29811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 20), 'func')
        # Getting the type of 'self' (line 742)
        self_29812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'self')
        # Setting the type of the member 'func' of a type (line 742)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 8), self_29812, 'func', func_29811)
        
        # Assigning a Name to a Attribute (line 743):
        
        # Assigning a Name to a Attribute (line 743):
        # Getting the type of 'ranges' (line 743)
        ranges_29813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 22), 'ranges')
        # Getting the type of 'self' (line 743)
        self_29814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'self')
        # Setting the type of the member 'ranges' of a type (line 743)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 8), self_29814, 'ranges', ranges_29813)
        
        # Assigning a Name to a Attribute (line 744):
        
        # Assigning a Name to a Attribute (line 744):
        # Getting the type of 'opts' (line 744)
        opts_29815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 20), 'opts')
        # Getting the type of 'self' (line 744)
        self_29816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'self')
        # Setting the type of the member 'opts' of a type (line 744)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 8), self_29816, 'opts', opts_29815)
        
        # Assigning a Call to a Attribute (line 745):
        
        # Assigning a Call to a Attribute (line 745):
        
        # Call to len(...): (line 745)
        # Processing the call arguments (line 745)
        # Getting the type of 'ranges' (line 745)
        ranges_29818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 28), 'ranges', False)
        # Processing the call keyword arguments (line 745)
        kwargs_29819 = {}
        # Getting the type of 'len' (line 745)
        len_29817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 24), 'len', False)
        # Calling len(args, kwargs) (line 745)
        len_call_result_29820 = invoke(stypy.reporting.localization.Localization(__file__, 745, 24), len_29817, *[ranges_29818], **kwargs_29819)
        
        # Getting the type of 'self' (line 745)
        self_29821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'self')
        # Setting the type of the member 'maxdepth' of a type (line 745)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 8), self_29821, 'maxdepth', len_call_result_29820)
        
        # Assigning a Name to a Attribute (line 746):
        
        # Assigning a Name to a Attribute (line 746):
        # Getting the type of 'full_output' (line 746)
        full_output_29822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 27), 'full_output')
        # Getting the type of 'self' (line 746)
        self_29823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 8), 'self')
        # Setting the type of the member 'full_output' of a type (line 746)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 8), self_29823, 'full_output', full_output_29822)
        
        # Getting the type of 'self' (line 747)
        self_29824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 11), 'self')
        # Obtaining the member 'full_output' of a type (line 747)
        full_output_29825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 11), self_29824, 'full_output')
        # Testing the type of an if condition (line 747)
        if_condition_29826 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 747, 8), full_output_29825)
        # Assigning a type to the variable 'if_condition_29826' (line 747)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 8), 'if_condition_29826', if_condition_29826)
        # SSA begins for if statement (line 747)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Dict to a Attribute (line 748):
        
        # Assigning a Dict to a Attribute (line 748):
        
        # Obtaining an instance of the builtin type 'dict' (line 748)
        dict_29827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 748)
        # Adding element type (key, value) (line 748)
        str_29828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 29), 'str', 'neval')
        int_29829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 38), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 28), dict_29827, (str_29828, int_29829))
        
        # Getting the type of 'self' (line 748)
        self_29830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 12), 'self')
        # Setting the type of the member 'out_dict' of a type (line 748)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 12), self_29830, 'out_dict', dict_29827)
        # SSA join for if statement (line 747)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def integrate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'integrate'
        module_type_store = module_type_store.open_function_context('integrate', 750, 4, False)
        # Assigning a type to the variable 'self' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _NQuad.integrate.__dict__.__setitem__('stypy_localization', localization)
        _NQuad.integrate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _NQuad.integrate.__dict__.__setitem__('stypy_type_store', module_type_store)
        _NQuad.integrate.__dict__.__setitem__('stypy_function_name', '_NQuad.integrate')
        _NQuad.integrate.__dict__.__setitem__('stypy_param_names_list', [])
        _NQuad.integrate.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        _NQuad.integrate.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        _NQuad.integrate.__dict__.__setitem__('stypy_call_defaults', defaults)
        _NQuad.integrate.__dict__.__setitem__('stypy_call_varargs', varargs)
        _NQuad.integrate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _NQuad.integrate.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_NQuad.integrate', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'integrate', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'integrate(...)' code ##################

        
        # Assigning a Call to a Name (line 751):
        
        # Assigning a Call to a Name (line 751):
        
        # Call to pop(...): (line 751)
        # Processing the call arguments (line 751)
        str_29833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 27), 'str', 'depth')
        int_29834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 36), 'int')
        # Processing the call keyword arguments (line 751)
        kwargs_29835 = {}
        # Getting the type of 'kwargs' (line 751)
        kwargs_29831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 16), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 751)
        pop_29832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 16), kwargs_29831, 'pop')
        # Calling pop(args, kwargs) (line 751)
        pop_call_result_29836 = invoke(stypy.reporting.localization.Localization(__file__, 751, 16), pop_29832, *[str_29833, int_29834], **kwargs_29835)
        
        # Assigning a type to the variable 'depth' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 8), 'depth', pop_call_result_29836)
        
        # Getting the type of 'kwargs' (line 752)
        kwargs_29837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 11), 'kwargs')
        # Testing the type of an if condition (line 752)
        if_condition_29838 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 752, 8), kwargs_29837)
        # Assigning a type to the variable 'if_condition_29838' (line 752)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 8), 'if_condition_29838', if_condition_29838)
        # SSA begins for if statement (line 752)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 753)
        # Processing the call arguments (line 753)
        str_29840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 29), 'str', 'unexpected kwargs')
        # Processing the call keyword arguments (line 753)
        kwargs_29841 = {}
        # Getting the type of 'ValueError' (line 753)
        ValueError_29839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 753)
        ValueError_call_result_29842 = invoke(stypy.reporting.localization.Localization(__file__, 753, 18), ValueError_29839, *[str_29840], **kwargs_29841)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 753, 12), ValueError_call_result_29842, 'raise parameter', BaseException)
        # SSA join for if statement (line 752)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a UnaryOp to a Name (line 756):
        
        # Assigning a UnaryOp to a Name (line 756):
        
        # Getting the type of 'depth' (line 756)
        depth_29843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 16), 'depth')
        int_29844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 24), 'int')
        # Applying the binary operator '+' (line 756)
        result_add_29845 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 16), '+', depth_29843, int_29844)
        
        # Applying the 'usub' unary operator (line 756)
        result___neg___29846 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 14), 'usub', result_add_29845)
        
        # Assigning a type to the variable 'ind' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'ind', result___neg___29846)
        
        # Assigning a Subscript to a Name (line 757):
        
        # Assigning a Subscript to a Name (line 757):
        
        # Obtaining the type of the subscript
        # Getting the type of 'ind' (line 757)
        ind_29847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 31), 'ind')
        # Getting the type of 'self' (line 757)
        self_29848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 19), 'self')
        # Obtaining the member 'ranges' of a type (line 757)
        ranges_29849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 19), self_29848, 'ranges')
        # Obtaining the member '__getitem__' of a type (line 757)
        getitem___29850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 19), ranges_29849, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 757)
        subscript_call_result_29851 = invoke(stypy.reporting.localization.Localization(__file__, 757, 19), getitem___29850, ind_29847)
        
        # Assigning a type to the variable 'fn_range' (line 757)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 8), 'fn_range', subscript_call_result_29851)
        
        # Assigning a Call to a Tuple (line 758):
        
        # Assigning a Subscript to a Name (line 758):
        
        # Obtaining the type of the subscript
        int_29852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 8), 'int')
        
        # Call to fn_range(...): (line 758)
        # Getting the type of 'args' (line 758)
        args_29854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 30), 'args', False)
        # Processing the call keyword arguments (line 758)
        kwargs_29855 = {}
        # Getting the type of 'fn_range' (line 758)
        fn_range_29853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 20), 'fn_range', False)
        # Calling fn_range(args, kwargs) (line 758)
        fn_range_call_result_29856 = invoke(stypy.reporting.localization.Localization(__file__, 758, 20), fn_range_29853, *[args_29854], **kwargs_29855)
        
        # Obtaining the member '__getitem__' of a type (line 758)
        getitem___29857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 8), fn_range_call_result_29856, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 758)
        subscript_call_result_29858 = invoke(stypy.reporting.localization.Localization(__file__, 758, 8), getitem___29857, int_29852)
        
        # Assigning a type to the variable 'tuple_var_assignment_29038' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'tuple_var_assignment_29038', subscript_call_result_29858)
        
        # Assigning a Subscript to a Name (line 758):
        
        # Obtaining the type of the subscript
        int_29859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 8), 'int')
        
        # Call to fn_range(...): (line 758)
        # Getting the type of 'args' (line 758)
        args_29861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 30), 'args', False)
        # Processing the call keyword arguments (line 758)
        kwargs_29862 = {}
        # Getting the type of 'fn_range' (line 758)
        fn_range_29860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 20), 'fn_range', False)
        # Calling fn_range(args, kwargs) (line 758)
        fn_range_call_result_29863 = invoke(stypy.reporting.localization.Localization(__file__, 758, 20), fn_range_29860, *[args_29861], **kwargs_29862)
        
        # Obtaining the member '__getitem__' of a type (line 758)
        getitem___29864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 8), fn_range_call_result_29863, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 758)
        subscript_call_result_29865 = invoke(stypy.reporting.localization.Localization(__file__, 758, 8), getitem___29864, int_29859)
        
        # Assigning a type to the variable 'tuple_var_assignment_29039' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'tuple_var_assignment_29039', subscript_call_result_29865)
        
        # Assigning a Name to a Name (line 758):
        # Getting the type of 'tuple_var_assignment_29038' (line 758)
        tuple_var_assignment_29038_29866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'tuple_var_assignment_29038')
        # Assigning a type to the variable 'low' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'low', tuple_var_assignment_29038_29866)
        
        # Assigning a Name to a Name (line 758):
        # Getting the type of 'tuple_var_assignment_29039' (line 758)
        tuple_var_assignment_29039_29867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'tuple_var_assignment_29039')
        # Assigning a type to the variable 'high' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 13), 'high', tuple_var_assignment_29039_29867)
        
        # Assigning a Subscript to a Name (line 759):
        
        # Assigning a Subscript to a Name (line 759):
        
        # Obtaining the type of the subscript
        # Getting the type of 'ind' (line 759)
        ind_29868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 27), 'ind')
        # Getting the type of 'self' (line 759)
        self_29869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 17), 'self')
        # Obtaining the member 'opts' of a type (line 759)
        opts_29870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 17), self_29869, 'opts')
        # Obtaining the member '__getitem__' of a type (line 759)
        getitem___29871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 17), opts_29870, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 759)
        subscript_call_result_29872 = invoke(stypy.reporting.localization.Localization(__file__, 759, 17), getitem___29871, ind_29868)
        
        # Assigning a type to the variable 'fn_opt' (line 759)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'fn_opt', subscript_call_result_29872)
        
        # Assigning a Call to a Name (line 760):
        
        # Assigning a Call to a Name (line 760):
        
        # Call to dict(...): (line 760)
        # Processing the call arguments (line 760)
        
        # Call to fn_opt(...): (line 760)
        # Getting the type of 'args' (line 760)
        args_29875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 27), 'args', False)
        # Processing the call keyword arguments (line 760)
        kwargs_29876 = {}
        # Getting the type of 'fn_opt' (line 760)
        fn_opt_29874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 19), 'fn_opt', False)
        # Calling fn_opt(args, kwargs) (line 760)
        fn_opt_call_result_29877 = invoke(stypy.reporting.localization.Localization(__file__, 760, 19), fn_opt_29874, *[args_29875], **kwargs_29876)
        
        # Processing the call keyword arguments (line 760)
        kwargs_29878 = {}
        # Getting the type of 'dict' (line 760)
        dict_29873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 14), 'dict', False)
        # Calling dict(args, kwargs) (line 760)
        dict_call_result_29879 = invoke(stypy.reporting.localization.Localization(__file__, 760, 14), dict_29873, *[fn_opt_call_result_29877], **kwargs_29878)
        
        # Assigning a type to the variable 'opt' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'opt', dict_call_result_29879)
        
        
        str_29880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 11), 'str', 'points')
        # Getting the type of 'opt' (line 762)
        opt_29881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 23), 'opt')
        # Applying the binary operator 'in' (line 762)
        result_contains_29882 = python_operator(stypy.reporting.localization.Localization(__file__, 762, 11), 'in', str_29880, opt_29881)
        
        # Testing the type of an if condition (line 762)
        if_condition_29883 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 762, 8), result_contains_29882)
        # Assigning a type to the variable 'if_condition_29883' (line 762)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'if_condition_29883', if_condition_29883)
        # SSA begins for if statement (line 762)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a ListComp to a Subscript (line 763):
        
        # Assigning a ListComp to a Subscript (line 763):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining the type of the subscript
        str_29891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 44), 'str', 'points')
        # Getting the type of 'opt' (line 763)
        opt_29892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 40), 'opt')
        # Obtaining the member '__getitem__' of a type (line 763)
        getitem___29893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 40), opt_29892, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 763)
        subscript_call_result_29894 = invoke(stypy.reporting.localization.Localization(__file__, 763, 40), getitem___29893, str_29891)
        
        comprehension_29895 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 29), subscript_call_result_29894)
        # Assigning a type to the variable 'x' (line 763)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 29), 'x', comprehension_29895)
        
        # Getting the type of 'low' (line 763)
        low_29885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 57), 'low')
        # Getting the type of 'x' (line 763)
        x_29886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 64), 'x')
        # Applying the binary operator '<=' (line 763)
        result_le_29887 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 57), '<=', low_29885, x_29886)
        # Getting the type of 'high' (line 763)
        high_29888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 69), 'high')
        # Applying the binary operator '<=' (line 763)
        result_le_29889 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 57), '<=', x_29886, high_29888)
        # Applying the binary operator '&' (line 763)
        result_and__29890 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 57), '&', result_le_29887, result_le_29889)
        
        # Getting the type of 'x' (line 763)
        x_29884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 29), 'x')
        list_29896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 29), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 29), list_29896, x_29884)
        # Getting the type of 'opt' (line 763)
        opt_29897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'opt')
        str_29898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 16), 'str', 'points')
        # Storing an element on a container (line 763)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 12), opt_29897, (str_29898, list_29896))
        # SSA join for if statement (line 762)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'depth' (line 764)
        depth_29899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 11), 'depth')
        int_29900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 19), 'int')
        # Applying the binary operator '+' (line 764)
        result_add_29901 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 11), '+', depth_29899, int_29900)
        
        # Getting the type of 'self' (line 764)
        self_29902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 24), 'self')
        # Obtaining the member 'maxdepth' of a type (line 764)
        maxdepth_29903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 24), self_29902, 'maxdepth')
        # Applying the binary operator '==' (line 764)
        result_eq_29904 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 11), '==', result_add_29901, maxdepth_29903)
        
        # Testing the type of an if condition (line 764)
        if_condition_29905 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 764, 8), result_eq_29904)
        # Assigning a type to the variable 'if_condition_29905' (line 764)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 8), 'if_condition_29905', if_condition_29905)
        # SSA begins for if statement (line 764)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 765):
        
        # Assigning a Attribute to a Name (line 765):
        # Getting the type of 'self' (line 765)
        self_29906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), 'self')
        # Obtaining the member 'func' of a type (line 765)
        func_29907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 16), self_29906, 'func')
        # Assigning a type to the variable 'f' (line 765)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 12), 'f', func_29907)
        # SSA branch for the else part of an if statement (line 764)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 767):
        
        # Assigning a Call to a Name (line 767):
        
        # Call to partial(...): (line 767)
        # Processing the call arguments (line 767)
        # Getting the type of 'self' (line 767)
        self_29909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 24), 'self', False)
        # Obtaining the member 'integrate' of a type (line 767)
        integrate_29910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 24), self_29909, 'integrate')
        # Processing the call keyword arguments (line 767)
        # Getting the type of 'depth' (line 767)
        depth_29911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 46), 'depth', False)
        int_29912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 52), 'int')
        # Applying the binary operator '+' (line 767)
        result_add_29913 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 46), '+', depth_29911, int_29912)
        
        keyword_29914 = result_add_29913
        kwargs_29915 = {'depth': keyword_29914}
        # Getting the type of 'partial' (line 767)
        partial_29908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 16), 'partial', False)
        # Calling partial(args, kwargs) (line 767)
        partial_call_result_29916 = invoke(stypy.reporting.localization.Localization(__file__, 767, 16), partial_29908, *[integrate_29910], **kwargs_29915)
        
        # Assigning a type to the variable 'f' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'f', partial_call_result_29916)
        # SSA join for if statement (line 764)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 768):
        
        # Assigning a Call to a Name (line 768):
        
        # Call to quad(...): (line 768)
        # Processing the call arguments (line 768)
        # Getting the type of 'f' (line 768)
        f_29918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 22), 'f', False)
        # Getting the type of 'low' (line 768)
        low_29919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 25), 'low', False)
        # Getting the type of 'high' (line 768)
        high_29920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 30), 'high', False)
        # Processing the call keyword arguments (line 768)
        # Getting the type of 'args' (line 768)
        args_29921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 41), 'args', False)
        keyword_29922 = args_29921
        # Getting the type of 'self' (line 768)
        self_29923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 59), 'self', False)
        # Obtaining the member 'full_output' of a type (line 768)
        full_output_29924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 59), self_29923, 'full_output')
        keyword_29925 = full_output_29924
        # Getting the type of 'opt' (line 769)
        opt_29926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 24), 'opt', False)
        kwargs_29927 = {'opt_29926': opt_29926, 'args': keyword_29922, 'full_output': keyword_29925}
        # Getting the type of 'quad' (line 768)
        quad_29917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 17), 'quad', False)
        # Calling quad(args, kwargs) (line 768)
        quad_call_result_29928 = invoke(stypy.reporting.localization.Localization(__file__, 768, 17), quad_29917, *[f_29918, low_29919, high_29920], **kwargs_29927)
        
        # Assigning a type to the variable 'quad_r' (line 768)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'quad_r', quad_call_result_29928)
        
        # Assigning a Subscript to a Name (line 770):
        
        # Assigning a Subscript to a Name (line 770):
        
        # Obtaining the type of the subscript
        int_29929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 23), 'int')
        # Getting the type of 'quad_r' (line 770)
        quad_r_29930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 16), 'quad_r')
        # Obtaining the member '__getitem__' of a type (line 770)
        getitem___29931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 16), quad_r_29930, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 770)
        subscript_call_result_29932 = invoke(stypy.reporting.localization.Localization(__file__, 770, 16), getitem___29931, int_29929)
        
        # Assigning a type to the variable 'value' (line 770)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 8), 'value', subscript_call_result_29932)
        
        # Assigning a Subscript to a Name (line 771):
        
        # Assigning a Subscript to a Name (line 771):
        
        # Obtaining the type of the subscript
        int_29933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 24), 'int')
        # Getting the type of 'quad_r' (line 771)
        quad_r_29934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 17), 'quad_r')
        # Obtaining the member '__getitem__' of a type (line 771)
        getitem___29935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 17), quad_r_29934, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 771)
        subscript_call_result_29936 = invoke(stypy.reporting.localization.Localization(__file__, 771, 17), getitem___29935, int_29933)
        
        # Assigning a type to the variable 'abserr' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'abserr', subscript_call_result_29936)
        
        # Getting the type of 'self' (line 772)
        self_29937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 11), 'self')
        # Obtaining the member 'full_output' of a type (line 772)
        full_output_29938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 11), self_29937, 'full_output')
        # Testing the type of an if condition (line 772)
        if_condition_29939 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 772, 8), full_output_29938)
        # Assigning a type to the variable 'if_condition_29939' (line 772)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 8), 'if_condition_29939', if_condition_29939)
        # SSA begins for if statement (line 772)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 773):
        
        # Assigning a Subscript to a Name (line 773):
        
        # Obtaining the type of the subscript
        int_29940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 30), 'int')
        # Getting the type of 'quad_r' (line 773)
        quad_r_29941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 23), 'quad_r')
        # Obtaining the member '__getitem__' of a type (line 773)
        getitem___29942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 23), quad_r_29941, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 773)
        subscript_call_result_29943 = invoke(stypy.reporting.localization.Localization(__file__, 773, 23), getitem___29942, int_29940)
        
        # Assigning a type to the variable 'infodict' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 12), 'infodict', subscript_call_result_29943)
        
        
        # Getting the type of 'depth' (line 777)
        depth_29944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 15), 'depth')
        int_29945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 23), 'int')
        # Applying the binary operator '+' (line 777)
        result_add_29946 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 15), '+', depth_29944, int_29945)
        
        # Getting the type of 'self' (line 777)
        self_29947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 28), 'self')
        # Obtaining the member 'maxdepth' of a type (line 777)
        maxdepth_29948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 28), self_29947, 'maxdepth')
        # Applying the binary operator '==' (line 777)
        result_eq_29949 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 15), '==', result_add_29946, maxdepth_29948)
        
        # Testing the type of an if condition (line 777)
        if_condition_29950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 777, 12), result_eq_29949)
        # Assigning a type to the variable 'if_condition_29950' (line 777)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 12), 'if_condition_29950', if_condition_29950)
        # SSA begins for if statement (line 777)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 778)
        self_29951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 16), 'self')
        # Obtaining the member 'out_dict' of a type (line 778)
        out_dict_29952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 16), self_29951, 'out_dict')
        
        # Obtaining the type of the subscript
        str_29953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 30), 'str', 'neval')
        # Getting the type of 'self' (line 778)
        self_29954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 16), 'self')
        # Obtaining the member 'out_dict' of a type (line 778)
        out_dict_29955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 16), self_29954, 'out_dict')
        # Obtaining the member '__getitem__' of a type (line 778)
        getitem___29956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 16), out_dict_29955, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 778)
        subscript_call_result_29957 = invoke(stypy.reporting.localization.Localization(__file__, 778, 16), getitem___29956, str_29953)
        
        
        # Obtaining the type of the subscript
        str_29958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 51), 'str', 'neval')
        # Getting the type of 'infodict' (line 778)
        infodict_29959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 42), 'infodict')
        # Obtaining the member '__getitem__' of a type (line 778)
        getitem___29960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 42), infodict_29959, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 778)
        subscript_call_result_29961 = invoke(stypy.reporting.localization.Localization(__file__, 778, 42), getitem___29960, str_29958)
        
        # Applying the binary operator '+=' (line 778)
        result_iadd_29962 = python_operator(stypy.reporting.localization.Localization(__file__, 778, 16), '+=', subscript_call_result_29957, subscript_call_result_29961)
        # Getting the type of 'self' (line 778)
        self_29963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 16), 'self')
        # Obtaining the member 'out_dict' of a type (line 778)
        out_dict_29964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 16), self_29963, 'out_dict')
        str_29965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 30), 'str', 'neval')
        # Storing an element on a container (line 778)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 16), out_dict_29964, (str_29965, result_iadd_29962))
        
        # SSA join for if statement (line 777)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 772)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 779):
        
        # Assigning a Call to a Attribute (line 779):
        
        # Call to max(...): (line 779)
        # Processing the call arguments (line 779)
        # Getting the type of 'self' (line 779)
        self_29967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 26), 'self', False)
        # Obtaining the member 'abserr' of a type (line 779)
        abserr_29968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 26), self_29967, 'abserr')
        # Getting the type of 'abserr' (line 779)
        abserr_29969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 39), 'abserr', False)
        # Processing the call keyword arguments (line 779)
        kwargs_29970 = {}
        # Getting the type of 'max' (line 779)
        max_29966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 22), 'max', False)
        # Calling max(args, kwargs) (line 779)
        max_call_result_29971 = invoke(stypy.reporting.localization.Localization(__file__, 779, 22), max_29966, *[abserr_29968, abserr_29969], **kwargs_29970)
        
        # Getting the type of 'self' (line 779)
        self_29972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'self')
        # Setting the type of the member 'abserr' of a type (line 779)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 8), self_29972, 'abserr', max_call_result_29971)
        
        
        # Getting the type of 'depth' (line 780)
        depth_29973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 11), 'depth')
        int_29974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 19), 'int')
        # Applying the binary operator '>' (line 780)
        result_gt_29975 = python_operator(stypy.reporting.localization.Localization(__file__, 780, 11), '>', depth_29973, int_29974)
        
        # Testing the type of an if condition (line 780)
        if_condition_29976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 780, 8), result_gt_29975)
        # Assigning a type to the variable 'if_condition_29976' (line 780)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'if_condition_29976', if_condition_29976)
        # SSA begins for if statement (line 780)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'value' (line 781)
        value_29977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 19), 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 781)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 12), 'stypy_return_type', value_29977)
        # SSA branch for the else part of an if statement (line 780)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 784)
        self_29978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 15), 'self')
        # Obtaining the member 'full_output' of a type (line 784)
        full_output_29979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 15), self_29978, 'full_output')
        # Testing the type of an if condition (line 784)
        if_condition_29980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 784, 12), full_output_29979)
        # Assigning a type to the variable 'if_condition_29980' (line 784)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 12), 'if_condition_29980', if_condition_29980)
        # SSA begins for if statement (line 784)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 785)
        tuple_29981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 785)
        # Adding element type (line 785)
        # Getting the type of 'value' (line 785)
        value_29982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 23), 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 785, 23), tuple_29981, value_29982)
        # Adding element type (line 785)
        # Getting the type of 'self' (line 785)
        self_29983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 30), 'self')
        # Obtaining the member 'abserr' of a type (line 785)
        abserr_29984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 30), self_29983, 'abserr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 785, 23), tuple_29981, abserr_29984)
        # Adding element type (line 785)
        # Getting the type of 'self' (line 785)
        self_29985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 43), 'self')
        # Obtaining the member 'out_dict' of a type (line 785)
        out_dict_29986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 43), self_29985, 'out_dict')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 785, 23), tuple_29981, out_dict_29986)
        
        # Assigning a type to the variable 'stypy_return_type' (line 785)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 16), 'stypy_return_type', tuple_29981)
        # SSA branch for the else part of an if statement (line 784)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'tuple' (line 787)
        tuple_29987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 787)
        # Adding element type (line 787)
        # Getting the type of 'value' (line 787)
        value_29988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 23), 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 23), tuple_29987, value_29988)
        # Adding element type (line 787)
        # Getting the type of 'self' (line 787)
        self_29989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 30), 'self')
        # Obtaining the member 'abserr' of a type (line 787)
        abserr_29990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 30), self_29989, 'abserr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 23), tuple_29987, abserr_29990)
        
        # Assigning a type to the variable 'stypy_return_type' (line 787)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 16), 'stypy_return_type', tuple_29987)
        # SSA join for if statement (line 784)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 780)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'integrate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'integrate' in the type store
        # Getting the type of 'stypy_return_type' (line 750)
        stypy_return_type_29991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29991)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'integrate'
        return stypy_return_type_29991


# Assigning a type to the variable '_NQuad' (line 739)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 0), '_NQuad', _NQuad)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
