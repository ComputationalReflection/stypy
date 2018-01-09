
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Compute the action of the matrix exponential.
2: '''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: 
8: import scipy.linalg
9: import scipy.sparse.linalg
10: from scipy.sparse.linalg import LinearOperator, aslinearoperator
11: 
12: __all__ = ['expm_multiply']
13: 
14: 
15: def _exact_inf_norm(A):
16:     # A compatibility function which should eventually disappear.
17:     if scipy.sparse.isspmatrix(A):
18:         return max(abs(A).sum(axis=1).flat)
19:     else:
20:         return np.linalg.norm(A, np.inf)
21: 
22: 
23: def _exact_1_norm(A):
24:     # A compatibility function which should eventually disappear.
25:     if scipy.sparse.isspmatrix(A):
26:         return max(abs(A).sum(axis=0).flat)
27:     else:
28:         return np.linalg.norm(A, 1)
29: 
30: 
31: def _trace(A):
32:     # A compatibility function which should eventually disappear.
33:     if scipy.sparse.isspmatrix(A):
34:         return A.diagonal().sum()
35:     else:
36:         return np.trace(A)
37: 
38: 
39: def _ident_like(A):
40:     # A compatibility function which should eventually disappear.
41:     if scipy.sparse.isspmatrix(A):
42:         return scipy.sparse.construct.eye(A.shape[0], A.shape[1],
43:                 dtype=A.dtype, format=A.format)
44:     else:
45:         return np.eye(A.shape[0], A.shape[1], dtype=A.dtype)
46: 
47: 
48: def expm_multiply(A, B, start=None, stop=None, num=None, endpoint=None):
49:     '''
50:     Compute the action of the matrix exponential of A on B.
51: 
52:     Parameters
53:     ----------
54:     A : transposable linear operator
55:         The operator whose exponential is of interest.
56:     B : ndarray
57:         The matrix or vector to be multiplied by the matrix exponential of A.
58:     start : scalar, optional
59:         The starting time point of the sequence.
60:     stop : scalar, optional
61:         The end time point of the sequence, unless `endpoint` is set to False.
62:         In that case, the sequence consists of all but the last of ``num + 1``
63:         evenly spaced time points, so that `stop` is excluded.
64:         Note that the step size changes when `endpoint` is False.
65:     num : int, optional
66:         Number of time points to use.
67:     endpoint : bool, optional
68:         If True, `stop` is the last time point.  Otherwise, it is not included.
69: 
70:     Returns
71:     -------
72:     expm_A_B : ndarray
73:          The result of the action :math:`e^{t_k A} B`.
74: 
75:     Notes
76:     -----
77:     The optional arguments defining the sequence of evenly spaced time points
78:     are compatible with the arguments of `numpy.linspace`.
79: 
80:     The output ndarray shape is somewhat complicated so I explain it here.
81:     The ndim of the output could be either 1, 2, or 3.
82:     It would be 1 if you are computing the expm action on a single vector
83:     at a single time point.
84:     It would be 2 if you are computing the expm action on a vector
85:     at multiple time points, or if you are computing the expm action
86:     on a matrix at a single time point.
87:     It would be 3 if you want the action on a matrix with multiple
88:     columns at multiple time points.
89:     If multiple time points are requested, expm_A_B[0] will always
90:     be the action of the expm at the first time point,
91:     regardless of whether the action is on a vector or a matrix.
92: 
93:     References
94:     ----------
95:     .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2011)
96:            "Computing the Action of the Matrix Exponential,
97:            with an Application to Exponential Integrators."
98:            SIAM Journal on Scientific Computing,
99:            33 (2). pp. 488-511. ISSN 1064-8275
100:            http://eprints.ma.man.ac.uk/1591/
101: 
102:     .. [2] Nicholas J. Higham and Awad H. Al-Mohy (2010)
103:            "Computing Matrix Functions."
104:            Acta Numerica,
105:            19. 159-208. ISSN 0962-4929
106:            http://eprints.ma.man.ac.uk/1451/
107: 
108:     Examples
109:     --------
110:     >>> from scipy.sparse import csc_matrix
111:     >>> from scipy.sparse.linalg import expm, expm_multiply
112:     >>> A = csc_matrix([[1, 0], [0, 1]])
113:     >>> A.todense()
114:     matrix([[1, 0],
115:             [0, 1]], dtype=int64)
116:     >>> B = np.array([np.exp(-1.), np.exp(-2.)])
117:     >>> B
118:     array([ 0.36787944,  0.13533528])
119:     >>> expm_multiply(A, B, start=1, stop=2, num=3, endpoint=True)
120:     array([[ 1.        ,  0.36787944],
121:            [ 1.64872127,  0.60653066],
122:            [ 2.71828183,  1.        ]])
123:     >>> expm(A).dot(B)                  # Verify 1st timestep
124:     array([ 1.        ,  0.36787944])
125:     >>> expm(1.5*A).dot(B)              # Verify 2nd timestep
126:     array([ 1.64872127,  0.60653066])
127:     >>> expm(2*A).dot(B)                # Verify 3rd timestep
128:     array([ 2.71828183,  1.        ])
129:     '''
130:     if all(arg is None for arg in (start, stop, num, endpoint)):
131:         X = _expm_multiply_simple(A, B)
132:     else:
133:         X, status = _expm_multiply_interval(A, B, start, stop, num, endpoint)
134:     return X
135: 
136: 
137: def _expm_multiply_simple(A, B, t=1.0, balance=False):
138:     '''
139:     Compute the action of the matrix exponential at a single time point.
140: 
141:     Parameters
142:     ----------
143:     A : transposable linear operator
144:         The operator whose exponential is of interest.
145:     B : ndarray
146:         The matrix to be multiplied by the matrix exponential of A.
147:     t : float
148:         A time point.
149:     balance : bool
150:         Indicates whether or not to apply balancing.
151: 
152:     Returns
153:     -------
154:     F : ndarray
155:         :math:`e^{t A} B`
156: 
157:     Notes
158:     -----
159:     This is algorithm (3.2) in Al-Mohy and Higham (2011).
160: 
161:     '''
162:     if balance:
163:         raise NotImplementedError
164:     if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
165:         raise ValueError('expected A to be like a square matrix')
166:     if A.shape[1] != B.shape[0]:
167:         raise ValueError('the matrices A and B have incompatible shapes')
168:     ident = _ident_like(A)
169:     n = A.shape[0]
170:     if len(B.shape) == 1:
171:         n0 = 1
172:     elif len(B.shape) == 2:
173:         n0 = B.shape[1]
174:     else:
175:         raise ValueError('expected B to be like a matrix or a vector')
176:     u_d = 2**-53
177:     tol = u_d
178:     mu = _trace(A) / float(n)
179:     A = A - mu * ident
180:     A_1_norm = _exact_1_norm(A)
181:     if t*A_1_norm == 0:
182:         m_star, s = 0, 1
183:     else:
184:         ell = 2
185:         norm_info = LazyOperatorNormInfo(t*A, A_1_norm=t*A_1_norm, ell=ell)
186:         m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
187:     return _expm_multiply_simple_core(A, B, t, mu, m_star, s, tol, balance)
188: 
189: 
190: def _expm_multiply_simple_core(A, B, t, mu, m_star, s, tol=None, balance=False):
191:     '''
192:     A helper function.
193:     '''
194:     if balance:
195:         raise NotImplementedError
196:     if tol is None:
197:         u_d = 2 ** -53
198:         tol = u_d
199:     F = B
200:     eta = np.exp(t*mu / float(s))
201:     for i in range(s):
202:         c1 = _exact_inf_norm(B)
203:         for j in range(m_star):
204:             coeff = t / float(s*(j+1))
205:             B = coeff * A.dot(B)
206:             c2 = _exact_inf_norm(B)
207:             F = F + B
208:             if c1 + c2 <= tol * _exact_inf_norm(F):
209:                 break
210:             c1 = c2
211:         F = eta * F
212:         B = F
213:     return F
214: 
215: # This table helps to compute bounds.
216: # They seem to have been difficult to calculate, involving symbolic
217: # manipulation of equations, followed by numerical root finding.
218: _theta = {
219:         # The first 30 values are from table A.3 of Computing Matrix Functions.
220:         1: 2.29e-16,
221:         2: 2.58e-8,
222:         3: 1.39e-5,
223:         4: 3.40e-4,
224:         5: 2.40e-3,
225:         6: 9.07e-3,
226:         7: 2.38e-2,
227:         8: 5.00e-2,
228:         9: 8.96e-2,
229:         10: 1.44e-1,
230:         # 11
231:         11: 2.14e-1,
232:         12: 3.00e-1,
233:         13: 4.00e-1,
234:         14: 5.14e-1,
235:         15: 6.41e-1,
236:         16: 7.81e-1,
237:         17: 9.31e-1,
238:         18: 1.09,
239:         19: 1.26,
240:         20: 1.44,
241:         # 21
242:         21: 1.62,
243:         22: 1.82,
244:         23: 2.01,
245:         24: 2.22,
246:         25: 2.43,
247:         26: 2.64,
248:         27: 2.86,
249:         28: 3.08,
250:         29: 3.31,
251:         30: 3.54,
252:         # The rest are from table 3.1 of
253:         # Computing the Action of the Matrix Exponential.
254:         35: 4.7,
255:         40: 6.0,
256:         45: 7.2,
257:         50: 8.5,
258:         55: 9.9,
259:         }
260: 
261: 
262: def _onenormest_matrix_power(A, p,
263:         t=2, itmax=5, compute_v=False, compute_w=False):
264:     '''
265:     Efficiently estimate the 1-norm of A^p.
266: 
267:     Parameters
268:     ----------
269:     A : ndarray
270:         Matrix whose 1-norm of a power is to be computed.
271:     p : int
272:         Non-negative integer power.
273:     t : int, optional
274:         A positive parameter controlling the tradeoff between
275:         accuracy versus time and memory usage.
276:         Larger values take longer and use more memory
277:         but give more accurate output.
278:     itmax : int, optional
279:         Use at most this many iterations.
280:     compute_v : bool, optional
281:         Request a norm-maximizing linear operator input vector if True.
282:     compute_w : bool, optional
283:         Request a norm-maximizing linear operator output vector if True.
284: 
285:     Returns
286:     -------
287:     est : float
288:         An underestimate of the 1-norm of the sparse matrix.
289:     v : ndarray, optional
290:         The vector such that ||Av||_1 == est*||v||_1.
291:         It can be thought of as an input to the linear operator
292:         that gives an output with particularly large norm.
293:     w : ndarray, optional
294:         The vector Av which has relatively large 1-norm.
295:         It can be thought of as an output of the linear operator
296:         that is relatively large in norm compared to the input.
297: 
298:     '''
299:     #XXX Eventually turn this into an API function in the  _onenormest module,
300:     #XXX and remove its underscore,
301:     #XXX but wait until expm_multiply goes into scipy.
302:     return scipy.sparse.linalg.onenormest(aslinearoperator(A) ** p)
303: 
304: class LazyOperatorNormInfo:
305:     '''
306:     Information about an operator is lazily computed.
307: 
308:     The information includes the exact 1-norm of the operator,
309:     in addition to estimates of 1-norms of powers of the operator.
310:     This uses the notation of Computing the Action (2011).
311:     This class is specialized enough to probably not be of general interest
312:     outside of this module.
313: 
314:     '''
315:     def __init__(self, A, A_1_norm=None, ell=2, scale=1):
316:         '''
317:         Provide the operator and some norm-related information.
318: 
319:         Parameters
320:         ----------
321:         A : linear operator
322:             The operator of interest.
323:         A_1_norm : float, optional
324:             The exact 1-norm of A.
325:         ell : int, optional
326:             A technical parameter controlling norm estimation quality.
327:         scale : int, optional
328:             If specified, return the norms of scale*A instead of A.
329: 
330:         '''
331:         self._A = A
332:         self._A_1_norm = A_1_norm
333:         self._ell = ell
334:         self._d = {}
335:         self._scale = scale
336: 
337:     def set_scale(self,scale):
338:         '''
339:         Set the scale parameter.
340:         '''
341:         self._scale = scale
342: 
343:     def onenorm(self):
344:         '''
345:         Compute the exact 1-norm.
346:         '''
347:         if self._A_1_norm is None:
348:             self._A_1_norm = _exact_1_norm(self._A)
349:         return self._scale*self._A_1_norm
350: 
351:     def d(self, p):
352:         '''
353:         Lazily estimate d_p(A) ~= || A^p ||^(1/p) where ||.|| is the 1-norm.
354:         '''
355:         if p not in self._d:
356:             est = _onenormest_matrix_power(self._A, p, self._ell)
357:             self._d[p] = est ** (1.0 / p)
358:         return self._scale*self._d[p]
359: 
360:     def alpha(self, p):
361:         '''
362:         Lazily compute max(d(p), d(p+1)).
363:         '''
364:         return max(self.d(p), self.d(p+1))
365: 
366: def _compute_cost_div_m(m, p, norm_info):
367:     '''
368:     A helper function for computing bounds.
369: 
370:     This is equation (3.10).
371:     It measures cost in terms of the number of required matrix products.
372: 
373:     Parameters
374:     ----------
375:     m : int
376:         A valid key of _theta.
377:     p : int
378:         A matrix power.
379:     norm_info : LazyOperatorNormInfo
380:         Information about 1-norms of related operators.
381: 
382:     Returns
383:     -------
384:     cost_div_m : int
385:         Required number of matrix products divided by m.
386: 
387:     '''
388:     return int(np.ceil(norm_info.alpha(p) / _theta[m]))
389: 
390: 
391: def _compute_p_max(m_max):
392:     '''
393:     Compute the largest positive integer p such that p*(p-1) <= m_max + 1.
394: 
395:     Do this in a slightly dumb way, but safe and not too slow.
396: 
397:     Parameters
398:     ----------
399:     m_max : int
400:         A count related to bounds.
401: 
402:     '''
403:     sqrt_m_max = np.sqrt(m_max)
404:     p_low = int(np.floor(sqrt_m_max))
405:     p_high = int(np.ceil(sqrt_m_max + 1))
406:     return max(p for p in range(p_low, p_high+1) if p*(p-1) <= m_max + 1)
407: 
408: 
409: def _fragment_3_1(norm_info, n0, tol, m_max=55, ell=2):
410:     '''
411:     A helper function for the _expm_multiply_* functions.
412: 
413:     Parameters
414:     ----------
415:     norm_info : LazyOperatorNormInfo
416:         Information about norms of certain linear operators of interest.
417:     n0 : int
418:         Number of columns in the _expm_multiply_* B matrix.
419:     tol : float
420:         Expected to be
421:         :math:`2^{-24}` for single precision or
422:         :math:`2^{-53}` for double precision.
423:     m_max : int
424:         A value related to a bound.
425:     ell : int
426:         The number of columns used in the 1-norm approximation.
427:         This is usually taken to be small, maybe between 1 and 5.
428: 
429:     Returns
430:     -------
431:     best_m : int
432:         Related to bounds for error control.
433:     best_s : int
434:         Amount of scaling.
435: 
436:     Notes
437:     -----
438:     This is code fragment (3.1) in Al-Mohy and Higham (2011).
439:     The discussion of default values for m_max and ell
440:     is given between the definitions of equation (3.11)
441:     and the definition of equation (3.12).
442: 
443:     '''
444:     if ell < 1:
445:         raise ValueError('expected ell to be a positive integer')
446:     best_m = None
447:     best_s = None
448:     if _condition_3_13(norm_info.onenorm(), n0, m_max, ell):
449:         for m, theta in _theta.items():
450:             s = int(np.ceil(norm_info.onenorm() / theta))
451:             if best_m is None or m * s < best_m * best_s:
452:                 best_m = m
453:                 best_s = s
454:     else:
455:         # Equation (3.11).
456:         for p in range(2, _compute_p_max(m_max) + 1):
457:             for m in range(p*(p-1)-1, m_max+1):
458:                 if m in _theta:
459:                     s = _compute_cost_div_m(m, p, norm_info)
460:                     if best_m is None or m * s < best_m * best_s:
461:                         best_m = m
462:                         best_s = s
463:         best_s = max(best_s, 1)
464:     return best_m, best_s
465: 
466: 
467: def _condition_3_13(A_1_norm, n0, m_max, ell):
468:     '''
469:     A helper function for the _expm_multiply_* functions.
470: 
471:     Parameters
472:     ----------
473:     A_1_norm : float
474:         The precomputed 1-norm of A.
475:     n0 : int
476:         Number of columns in the _expm_multiply_* B matrix.
477:     m_max : int
478:         A value related to a bound.
479:     ell : int
480:         The number of columns used in the 1-norm approximation.
481:         This is usually taken to be small, maybe between 1 and 5.
482: 
483:     Returns
484:     -------
485:     value : bool
486:         Indicates whether or not the condition has been met.
487: 
488:     Notes
489:     -----
490:     This is condition (3.13) in Al-Mohy and Higham (2011).
491: 
492:     '''
493: 
494:     # This is the rhs of equation (3.12).
495:     p_max = _compute_p_max(m_max)
496:     a = 2 * ell * p_max * (p_max + 3)
497: 
498:     # Evaluate the condition (3.13).
499:     b = _theta[m_max] / float(n0 * m_max)
500:     return A_1_norm <= a * b
501: 
502: 
503: def _expm_multiply_interval(A, B, start=None, stop=None,
504:         num=None, endpoint=None, balance=False, status_only=False):
505:     '''
506:     Compute the action of the matrix exponential at multiple time points.
507: 
508:     Parameters
509:     ----------
510:     A : transposable linear operator
511:         The operator whose exponential is of interest.
512:     B : ndarray
513:         The matrix to be multiplied by the matrix exponential of A.
514:     start : scalar, optional
515:         The starting time point of the sequence.
516:     stop : scalar, optional
517:         The end time point of the sequence, unless `endpoint` is set to False.
518:         In that case, the sequence consists of all but the last of ``num + 1``
519:         evenly spaced time points, so that `stop` is excluded.
520:         Note that the step size changes when `endpoint` is False.
521:     num : int, optional
522:         Number of time points to use.
523:     endpoint : bool, optional
524:         If True, `stop` is the last time point.  Otherwise, it is not included.
525:     balance : bool
526:         Indicates whether or not to apply balancing.
527:     status_only : bool
528:         A flag that is set to True for some debugging and testing operations.
529: 
530:     Returns
531:     -------
532:     F : ndarray
533:         :math:`e^{t_k A} B`
534:     status : int
535:         An integer status for testing and debugging.
536: 
537:     Notes
538:     -----
539:     This is algorithm (5.2) in Al-Mohy and Higham (2011).
540: 
541:     There seems to be a typo, where line 15 of the algorithm should be
542:     moved to line 6.5 (between lines 6 and 7).
543: 
544:     '''
545:     if balance:
546:         raise NotImplementedError
547:     if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
548:         raise ValueError('expected A to be like a square matrix')
549:     if A.shape[1] != B.shape[0]:
550:         raise ValueError('the matrices A and B have incompatible shapes')
551:     ident = _ident_like(A)
552:     n = A.shape[0]
553:     if len(B.shape) == 1:
554:         n0 = 1
555:     elif len(B.shape) == 2:
556:         n0 = B.shape[1]
557:     else:
558:         raise ValueError('expected B to be like a matrix or a vector')
559:     u_d = 2**-53
560:     tol = u_d
561:     mu = _trace(A) / float(n)
562: 
563:     # Get the linspace samples, attempting to preserve the linspace defaults.
564:     linspace_kwargs = {'retstep': True}
565:     if num is not None:
566:         linspace_kwargs['num'] = num
567:     if endpoint is not None:
568:         linspace_kwargs['endpoint'] = endpoint
569:     samples, step = np.linspace(start, stop, **linspace_kwargs)
570: 
571:     # Convert the linspace output to the notation used by the publication.
572:     nsamples = len(samples)
573:     if nsamples < 2:
574:         raise ValueError('at least two time points are required')
575:     q = nsamples - 1
576:     h = step
577:     t_0 = samples[0]
578:     t_q = samples[q]
579: 
580:     # Define the output ndarray.
581:     # Use an ndim=3 shape, such that the last two indices
582:     # are the ones that may be involved in level 3 BLAS operations.
583:     X_shape = (nsamples,) + B.shape
584:     X = np.empty(X_shape, dtype=np.result_type(A.dtype, B.dtype, float))
585:     t = t_q - t_0
586:     A = A - mu * ident
587:     A_1_norm = _exact_1_norm(A)
588:     ell = 2
589:     norm_info = LazyOperatorNormInfo(t*A, A_1_norm=t*A_1_norm, ell=ell)
590:     if t*A_1_norm == 0:
591:         m_star, s = 0, 1
592:     else:
593:         m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
594: 
595:     # Compute the expm action up to the initial time point.
596:     X[0] = _expm_multiply_simple_core(A, B, t_0, mu, m_star, s)
597: 
598:     # Compute the expm action at the rest of the time points.
599:     if q <= s:
600:         if status_only:
601:             return 0
602:         else:
603:             return _expm_multiply_interval_core_0(A, X,
604:                     h, mu, q, norm_info, tol, ell,n0)
605:     elif not (q % s):
606:         if status_only:
607:             return 1
608:         else:
609:             return _expm_multiply_interval_core_1(A, X,
610:                     h, mu, m_star, s, q, tol)
611:     elif (q % s):
612:         if status_only:
613:             return 2
614:         else:
615:             return _expm_multiply_interval_core_2(A, X,
616:                     h, mu, m_star, s, q, tol)
617:     else:
618:         raise Exception('internal error')
619: 
620: 
621: def _expm_multiply_interval_core_0(A, X, h, mu, q, norm_info, tol, ell, n0):
622:     '''
623:     A helper function, for the case q <= s.
624:     '''
625: 
626:     # Compute the new values of m_star and s which should be applied
627:     # over intervals of size t/q
628:     if norm_info.onenorm() == 0:
629:         m_star, s = 0, 1
630:     else:
631:         norm_info.set_scale(1./q)
632:         m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
633:         norm_info.set_scale(1)
634: 
635:     for k in range(q):
636:         X[k+1] = _expm_multiply_simple_core(A, X[k], h, mu, m_star, s)
637:     return X, 0
638: 
639: 
640: def _expm_multiply_interval_core_1(A, X, h, mu, m_star, s, q, tol):
641:     '''
642:     A helper function, for the case q > s and q % s == 0.
643:     '''
644:     d = q // s
645:     input_shape = X.shape[1:]
646:     K_shape = (m_star + 1, ) + input_shape
647:     K = np.empty(K_shape, dtype=X.dtype)
648:     for i in range(s):
649:         Z = X[i*d]
650:         K[0] = Z
651:         high_p = 0
652:         for k in range(1, d+1):
653:             F = K[0]
654:             c1 = _exact_inf_norm(F)
655:             for p in range(1, m_star+1):
656:                 if p > high_p:
657:                     K[p] = h * A.dot(K[p-1]) / float(p)
658:                 coeff = float(pow(k, p))
659:                 F = F + coeff * K[p]
660:                 inf_norm_K_p_1 = _exact_inf_norm(K[p])
661:                 c2 = coeff * inf_norm_K_p_1
662:                 if c1 + c2 <= tol * _exact_inf_norm(F):
663:                     break
664:                 c1 = c2
665:             X[k + i*d] = np.exp(k*h*mu) * F
666:     return X, 1
667: 
668: 
669: def _expm_multiply_interval_core_2(A, X, h, mu, m_star, s, q, tol):
670:     '''
671:     A helper function, for the case q > s and q % s > 0.
672:     '''
673:     d = q // s
674:     j = q // d
675:     r = q - d * j
676:     input_shape = X.shape[1:]
677:     K_shape = (m_star + 1, ) + input_shape
678:     K = np.empty(K_shape, dtype=X.dtype)
679:     for i in range(j + 1):
680:         Z = X[i*d]
681:         K[0] = Z
682:         high_p = 0
683:         if i < j:
684:             effective_d = d
685:         else:
686:             effective_d = r
687:         for k in range(1, effective_d+1):
688:             F = K[0]
689:             c1 = _exact_inf_norm(F)
690:             for p in range(1, m_star+1):
691:                 if p == high_p + 1:
692:                     K[p] = h * A.dot(K[p-1]) / float(p)
693:                     high_p = p
694:                 coeff = float(pow(k, p))
695:                 F = F + coeff * K[p]
696:                 inf_norm_K_p_1 = _exact_inf_norm(K[p])
697:                 c2 = coeff * inf_norm_K_p_1
698:                 if c1 + c2 <= tol * _exact_inf_norm(F):
699:                     break
700:                 c1 = c2
701:             X[k + i*d] = np.exp(k*h*mu) * F
702:     return X, 2
703: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_388592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'Compute the action of the matrix exponential.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_388593 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_388593) is not StypyTypeError):

    if (import_388593 != 'pyd_module'):
        __import__(import_388593)
        sys_modules_388594 = sys.modules[import_388593]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_388594.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_388593)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import scipy.linalg' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_388595 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg')

if (type(import_388595) is not StypyTypeError):

    if (import_388595 != 'pyd_module'):
        __import__(import_388595)
        sys_modules_388596 = sys.modules[import_388595]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg', sys_modules_388596.module_type_store, module_type_store)
    else:
        import scipy.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg', scipy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.linalg' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg', import_388595)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import scipy.sparse.linalg' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_388597 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg')

if (type(import_388597) is not StypyTypeError):

    if (import_388597 != 'pyd_module'):
        __import__(import_388597)
        sys_modules_388598 = sys.modules[import_388597]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg', sys_modules_388598.module_type_store, module_type_store)
    else:
        import scipy.sparse.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg', scipy.sparse.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg', import_388597)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.sparse.linalg import LinearOperator, aslinearoperator' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_388599 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg')

if (type(import_388599) is not StypyTypeError):

    if (import_388599 != 'pyd_module'):
        __import__(import_388599)
        sys_modules_388600 = sys.modules[import_388599]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', sys_modules_388600.module_type_store, module_type_store, ['LinearOperator', 'aslinearoperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_388600, sys_modules_388600.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import LinearOperator, aslinearoperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', None, module_type_store, ['LinearOperator', 'aslinearoperator'], [LinearOperator, aslinearoperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', import_388599)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')


# Assigning a List to a Name (line 12):

# Assigning a List to a Name (line 12):
__all__ = ['expm_multiply']
module_type_store.set_exportable_members(['expm_multiply'])

# Obtaining an instance of the builtin type 'list' (line 12)
list_388601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_388602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'expm_multiply')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_388601, str_388602)

# Assigning a type to the variable '__all__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__all__', list_388601)

@norecursion
def _exact_inf_norm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_exact_inf_norm'
    module_type_store = module_type_store.open_function_context('_exact_inf_norm', 15, 0, False)
    
    # Passed parameters checking function
    _exact_inf_norm.stypy_localization = localization
    _exact_inf_norm.stypy_type_of_self = None
    _exact_inf_norm.stypy_type_store = module_type_store
    _exact_inf_norm.stypy_function_name = '_exact_inf_norm'
    _exact_inf_norm.stypy_param_names_list = ['A']
    _exact_inf_norm.stypy_varargs_param_name = None
    _exact_inf_norm.stypy_kwargs_param_name = None
    _exact_inf_norm.stypy_call_defaults = defaults
    _exact_inf_norm.stypy_call_varargs = varargs
    _exact_inf_norm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_exact_inf_norm', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_exact_inf_norm', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_exact_inf_norm(...)' code ##################

    
    
    # Call to isspmatrix(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'A' (line 17)
    A_388606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 31), 'A', False)
    # Processing the call keyword arguments (line 17)
    kwargs_388607 = {}
    # Getting the type of 'scipy' (line 17)
    scipy_388603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 7), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 17)
    sparse_388604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 7), scipy_388603, 'sparse')
    # Obtaining the member 'isspmatrix' of a type (line 17)
    isspmatrix_388605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 7), sparse_388604, 'isspmatrix')
    # Calling isspmatrix(args, kwargs) (line 17)
    isspmatrix_call_result_388608 = invoke(stypy.reporting.localization.Localization(__file__, 17, 7), isspmatrix_388605, *[A_388606], **kwargs_388607)
    
    # Testing the type of an if condition (line 17)
    if_condition_388609 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 4), isspmatrix_call_result_388608)
    # Assigning a type to the variable 'if_condition_388609' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'if_condition_388609', if_condition_388609)
    # SSA begins for if statement (line 17)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to max(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to sum(...): (line 18)
    # Processing the call keyword arguments (line 18)
    int_388616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'int')
    keyword_388617 = int_388616
    kwargs_388618 = {'axis': keyword_388617}
    
    # Call to abs(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'A' (line 18)
    A_388612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'A', False)
    # Processing the call keyword arguments (line 18)
    kwargs_388613 = {}
    # Getting the type of 'abs' (line 18)
    abs_388611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 19), 'abs', False)
    # Calling abs(args, kwargs) (line 18)
    abs_call_result_388614 = invoke(stypy.reporting.localization.Localization(__file__, 18, 19), abs_388611, *[A_388612], **kwargs_388613)
    
    # Obtaining the member 'sum' of a type (line 18)
    sum_388615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 19), abs_call_result_388614, 'sum')
    # Calling sum(args, kwargs) (line 18)
    sum_call_result_388619 = invoke(stypy.reporting.localization.Localization(__file__, 18, 19), sum_388615, *[], **kwargs_388618)
    
    # Obtaining the member 'flat' of a type (line 18)
    flat_388620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 19), sum_call_result_388619, 'flat')
    # Processing the call keyword arguments (line 18)
    kwargs_388621 = {}
    # Getting the type of 'max' (line 18)
    max_388610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'max', False)
    # Calling max(args, kwargs) (line 18)
    max_call_result_388622 = invoke(stypy.reporting.localization.Localization(__file__, 18, 15), max_388610, *[flat_388620], **kwargs_388621)
    
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'stypy_return_type', max_call_result_388622)
    # SSA branch for the else part of an if statement (line 17)
    module_type_store.open_ssa_branch('else')
    
    # Call to norm(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'A' (line 20)
    A_388626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 30), 'A', False)
    # Getting the type of 'np' (line 20)
    np_388627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 33), 'np', False)
    # Obtaining the member 'inf' of a type (line 20)
    inf_388628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 33), np_388627, 'inf')
    # Processing the call keyword arguments (line 20)
    kwargs_388629 = {}
    # Getting the type of 'np' (line 20)
    np_388623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'np', False)
    # Obtaining the member 'linalg' of a type (line 20)
    linalg_388624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 15), np_388623, 'linalg')
    # Obtaining the member 'norm' of a type (line 20)
    norm_388625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 15), linalg_388624, 'norm')
    # Calling norm(args, kwargs) (line 20)
    norm_call_result_388630 = invoke(stypy.reporting.localization.Localization(__file__, 20, 15), norm_388625, *[A_388626, inf_388628], **kwargs_388629)
    
    # Assigning a type to the variable 'stypy_return_type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'stypy_return_type', norm_call_result_388630)
    # SSA join for if statement (line 17)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_exact_inf_norm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_exact_inf_norm' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_388631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_388631)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_exact_inf_norm'
    return stypy_return_type_388631

# Assigning a type to the variable '_exact_inf_norm' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '_exact_inf_norm', _exact_inf_norm)

@norecursion
def _exact_1_norm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_exact_1_norm'
    module_type_store = module_type_store.open_function_context('_exact_1_norm', 23, 0, False)
    
    # Passed parameters checking function
    _exact_1_norm.stypy_localization = localization
    _exact_1_norm.stypy_type_of_self = None
    _exact_1_norm.stypy_type_store = module_type_store
    _exact_1_norm.stypy_function_name = '_exact_1_norm'
    _exact_1_norm.stypy_param_names_list = ['A']
    _exact_1_norm.stypy_varargs_param_name = None
    _exact_1_norm.stypy_kwargs_param_name = None
    _exact_1_norm.stypy_call_defaults = defaults
    _exact_1_norm.stypy_call_varargs = varargs
    _exact_1_norm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_exact_1_norm', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_exact_1_norm', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_exact_1_norm(...)' code ##################

    
    
    # Call to isspmatrix(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'A' (line 25)
    A_388635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'A', False)
    # Processing the call keyword arguments (line 25)
    kwargs_388636 = {}
    # Getting the type of 'scipy' (line 25)
    scipy_388632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 7), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 25)
    sparse_388633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 7), scipy_388632, 'sparse')
    # Obtaining the member 'isspmatrix' of a type (line 25)
    isspmatrix_388634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 7), sparse_388633, 'isspmatrix')
    # Calling isspmatrix(args, kwargs) (line 25)
    isspmatrix_call_result_388637 = invoke(stypy.reporting.localization.Localization(__file__, 25, 7), isspmatrix_388634, *[A_388635], **kwargs_388636)
    
    # Testing the type of an if condition (line 25)
    if_condition_388638 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 4), isspmatrix_call_result_388637)
    # Assigning a type to the variable 'if_condition_388638' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'if_condition_388638', if_condition_388638)
    # SSA begins for if statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to max(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Call to sum(...): (line 26)
    # Processing the call keyword arguments (line 26)
    int_388645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 35), 'int')
    keyword_388646 = int_388645
    kwargs_388647 = {'axis': keyword_388646}
    
    # Call to abs(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'A' (line 26)
    A_388641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'A', False)
    # Processing the call keyword arguments (line 26)
    kwargs_388642 = {}
    # Getting the type of 'abs' (line 26)
    abs_388640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 19), 'abs', False)
    # Calling abs(args, kwargs) (line 26)
    abs_call_result_388643 = invoke(stypy.reporting.localization.Localization(__file__, 26, 19), abs_388640, *[A_388641], **kwargs_388642)
    
    # Obtaining the member 'sum' of a type (line 26)
    sum_388644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 19), abs_call_result_388643, 'sum')
    # Calling sum(args, kwargs) (line 26)
    sum_call_result_388648 = invoke(stypy.reporting.localization.Localization(__file__, 26, 19), sum_388644, *[], **kwargs_388647)
    
    # Obtaining the member 'flat' of a type (line 26)
    flat_388649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 19), sum_call_result_388648, 'flat')
    # Processing the call keyword arguments (line 26)
    kwargs_388650 = {}
    # Getting the type of 'max' (line 26)
    max_388639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'max', False)
    # Calling max(args, kwargs) (line 26)
    max_call_result_388651 = invoke(stypy.reporting.localization.Localization(__file__, 26, 15), max_388639, *[flat_388649], **kwargs_388650)
    
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type', max_call_result_388651)
    # SSA branch for the else part of an if statement (line 25)
    module_type_store.open_ssa_branch('else')
    
    # Call to norm(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'A' (line 28)
    A_388655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'A', False)
    int_388656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 33), 'int')
    # Processing the call keyword arguments (line 28)
    kwargs_388657 = {}
    # Getting the type of 'np' (line 28)
    np_388652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'np', False)
    # Obtaining the member 'linalg' of a type (line 28)
    linalg_388653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 15), np_388652, 'linalg')
    # Obtaining the member 'norm' of a type (line 28)
    norm_388654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 15), linalg_388653, 'norm')
    # Calling norm(args, kwargs) (line 28)
    norm_call_result_388658 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), norm_388654, *[A_388655, int_388656], **kwargs_388657)
    
    # Assigning a type to the variable 'stypy_return_type' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', norm_call_result_388658)
    # SSA join for if statement (line 25)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_exact_1_norm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_exact_1_norm' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_388659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_388659)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_exact_1_norm'
    return stypy_return_type_388659

# Assigning a type to the variable '_exact_1_norm' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), '_exact_1_norm', _exact_1_norm)

@norecursion
def _trace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_trace'
    module_type_store = module_type_store.open_function_context('_trace', 31, 0, False)
    
    # Passed parameters checking function
    _trace.stypy_localization = localization
    _trace.stypy_type_of_self = None
    _trace.stypy_type_store = module_type_store
    _trace.stypy_function_name = '_trace'
    _trace.stypy_param_names_list = ['A']
    _trace.stypy_varargs_param_name = None
    _trace.stypy_kwargs_param_name = None
    _trace.stypy_call_defaults = defaults
    _trace.stypy_call_varargs = varargs
    _trace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_trace', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_trace', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_trace(...)' code ##################

    
    
    # Call to isspmatrix(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'A' (line 33)
    A_388663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 31), 'A', False)
    # Processing the call keyword arguments (line 33)
    kwargs_388664 = {}
    # Getting the type of 'scipy' (line 33)
    scipy_388660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 33)
    sparse_388661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 7), scipy_388660, 'sparse')
    # Obtaining the member 'isspmatrix' of a type (line 33)
    isspmatrix_388662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 7), sparse_388661, 'isspmatrix')
    # Calling isspmatrix(args, kwargs) (line 33)
    isspmatrix_call_result_388665 = invoke(stypy.reporting.localization.Localization(__file__, 33, 7), isspmatrix_388662, *[A_388663], **kwargs_388664)
    
    # Testing the type of an if condition (line 33)
    if_condition_388666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 4), isspmatrix_call_result_388665)
    # Assigning a type to the variable 'if_condition_388666' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'if_condition_388666', if_condition_388666)
    # SSA begins for if statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to sum(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_388672 = {}
    
    # Call to diagonal(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_388669 = {}
    # Getting the type of 'A' (line 34)
    A_388667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'A', False)
    # Obtaining the member 'diagonal' of a type (line 34)
    diagonal_388668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), A_388667, 'diagonal')
    # Calling diagonal(args, kwargs) (line 34)
    diagonal_call_result_388670 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), diagonal_388668, *[], **kwargs_388669)
    
    # Obtaining the member 'sum' of a type (line 34)
    sum_388671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), diagonal_call_result_388670, 'sum')
    # Calling sum(args, kwargs) (line 34)
    sum_call_result_388673 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), sum_388671, *[], **kwargs_388672)
    
    # Assigning a type to the variable 'stypy_return_type' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', sum_call_result_388673)
    # SSA branch for the else part of an if statement (line 33)
    module_type_store.open_ssa_branch('else')
    
    # Call to trace(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'A' (line 36)
    A_388676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'A', False)
    # Processing the call keyword arguments (line 36)
    kwargs_388677 = {}
    # Getting the type of 'np' (line 36)
    np_388674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'np', False)
    # Obtaining the member 'trace' of a type (line 36)
    trace_388675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 15), np_388674, 'trace')
    # Calling trace(args, kwargs) (line 36)
    trace_call_result_388678 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), trace_388675, *[A_388676], **kwargs_388677)
    
    # Assigning a type to the variable 'stypy_return_type' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', trace_call_result_388678)
    # SSA join for if statement (line 33)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_trace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_trace' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_388679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_388679)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_trace'
    return stypy_return_type_388679

# Assigning a type to the variable '_trace' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), '_trace', _trace)

@norecursion
def _ident_like(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_ident_like'
    module_type_store = module_type_store.open_function_context('_ident_like', 39, 0, False)
    
    # Passed parameters checking function
    _ident_like.stypy_localization = localization
    _ident_like.stypy_type_of_self = None
    _ident_like.stypy_type_store = module_type_store
    _ident_like.stypy_function_name = '_ident_like'
    _ident_like.stypy_param_names_list = ['A']
    _ident_like.stypy_varargs_param_name = None
    _ident_like.stypy_kwargs_param_name = None
    _ident_like.stypy_call_defaults = defaults
    _ident_like.stypy_call_varargs = varargs
    _ident_like.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_ident_like', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_ident_like', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_ident_like(...)' code ##################

    
    
    # Call to isspmatrix(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'A' (line 41)
    A_388683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'A', False)
    # Processing the call keyword arguments (line 41)
    kwargs_388684 = {}
    # Getting the type of 'scipy' (line 41)
    scipy_388680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 7), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 41)
    sparse_388681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 7), scipy_388680, 'sparse')
    # Obtaining the member 'isspmatrix' of a type (line 41)
    isspmatrix_388682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 7), sparse_388681, 'isspmatrix')
    # Calling isspmatrix(args, kwargs) (line 41)
    isspmatrix_call_result_388685 = invoke(stypy.reporting.localization.Localization(__file__, 41, 7), isspmatrix_388682, *[A_388683], **kwargs_388684)
    
    # Testing the type of an if condition (line 41)
    if_condition_388686 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 4), isspmatrix_call_result_388685)
    # Assigning a type to the variable 'if_condition_388686' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'if_condition_388686', if_condition_388686)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to eye(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Obtaining the type of the subscript
    int_388691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 50), 'int')
    # Getting the type of 'A' (line 42)
    A_388692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 42), 'A', False)
    # Obtaining the member 'shape' of a type (line 42)
    shape_388693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 42), A_388692, 'shape')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___388694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 42), shape_388693, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_388695 = invoke(stypy.reporting.localization.Localization(__file__, 42, 42), getitem___388694, int_388691)
    
    
    # Obtaining the type of the subscript
    int_388696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 62), 'int')
    # Getting the type of 'A' (line 42)
    A_388697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 54), 'A', False)
    # Obtaining the member 'shape' of a type (line 42)
    shape_388698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 54), A_388697, 'shape')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___388699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 54), shape_388698, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_388700 = invoke(stypy.reporting.localization.Localization(__file__, 42, 54), getitem___388699, int_388696)
    
    # Processing the call keyword arguments (line 42)
    # Getting the type of 'A' (line 43)
    A_388701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'A', False)
    # Obtaining the member 'dtype' of a type (line 43)
    dtype_388702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 22), A_388701, 'dtype')
    keyword_388703 = dtype_388702
    # Getting the type of 'A' (line 43)
    A_388704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 38), 'A', False)
    # Obtaining the member 'format' of a type (line 43)
    format_388705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 38), A_388704, 'format')
    keyword_388706 = format_388705
    kwargs_388707 = {'dtype': keyword_388703, 'format': keyword_388706}
    # Getting the type of 'scipy' (line 42)
    scipy_388687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 42)
    sparse_388688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 15), scipy_388687, 'sparse')
    # Obtaining the member 'construct' of a type (line 42)
    construct_388689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 15), sparse_388688, 'construct')
    # Obtaining the member 'eye' of a type (line 42)
    eye_388690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 15), construct_388689, 'eye')
    # Calling eye(args, kwargs) (line 42)
    eye_call_result_388708 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), eye_388690, *[subscript_call_result_388695, subscript_call_result_388700], **kwargs_388707)
    
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', eye_call_result_388708)
    # SSA branch for the else part of an if statement (line 41)
    module_type_store.open_ssa_branch('else')
    
    # Call to eye(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Obtaining the type of the subscript
    int_388711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 30), 'int')
    # Getting the type of 'A' (line 45)
    A_388712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 22), 'A', False)
    # Obtaining the member 'shape' of a type (line 45)
    shape_388713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 22), A_388712, 'shape')
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___388714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 22), shape_388713, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_388715 = invoke(stypy.reporting.localization.Localization(__file__, 45, 22), getitem___388714, int_388711)
    
    
    # Obtaining the type of the subscript
    int_388716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 42), 'int')
    # Getting the type of 'A' (line 45)
    A_388717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), 'A', False)
    # Obtaining the member 'shape' of a type (line 45)
    shape_388718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 34), A_388717, 'shape')
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___388719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 34), shape_388718, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_388720 = invoke(stypy.reporting.localization.Localization(__file__, 45, 34), getitem___388719, int_388716)
    
    # Processing the call keyword arguments (line 45)
    # Getting the type of 'A' (line 45)
    A_388721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 52), 'A', False)
    # Obtaining the member 'dtype' of a type (line 45)
    dtype_388722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 52), A_388721, 'dtype')
    keyword_388723 = dtype_388722
    kwargs_388724 = {'dtype': keyword_388723}
    # Getting the type of 'np' (line 45)
    np_388709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'np', False)
    # Obtaining the member 'eye' of a type (line 45)
    eye_388710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 15), np_388709, 'eye')
    # Calling eye(args, kwargs) (line 45)
    eye_call_result_388725 = invoke(stypy.reporting.localization.Localization(__file__, 45, 15), eye_388710, *[subscript_call_result_388715, subscript_call_result_388720], **kwargs_388724)
    
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', eye_call_result_388725)
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_ident_like(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_ident_like' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_388726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_388726)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_ident_like'
    return stypy_return_type_388726

# Assigning a type to the variable '_ident_like' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), '_ident_like', _ident_like)

@norecursion
def expm_multiply(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 48)
    None_388727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'None')
    # Getting the type of 'None' (line 48)
    None_388728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 41), 'None')
    # Getting the type of 'None' (line 48)
    None_388729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 51), 'None')
    # Getting the type of 'None' (line 48)
    None_388730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 66), 'None')
    defaults = [None_388727, None_388728, None_388729, None_388730]
    # Create a new context for function 'expm_multiply'
    module_type_store = module_type_store.open_function_context('expm_multiply', 48, 0, False)
    
    # Passed parameters checking function
    expm_multiply.stypy_localization = localization
    expm_multiply.stypy_type_of_self = None
    expm_multiply.stypy_type_store = module_type_store
    expm_multiply.stypy_function_name = 'expm_multiply'
    expm_multiply.stypy_param_names_list = ['A', 'B', 'start', 'stop', 'num', 'endpoint']
    expm_multiply.stypy_varargs_param_name = None
    expm_multiply.stypy_kwargs_param_name = None
    expm_multiply.stypy_call_defaults = defaults
    expm_multiply.stypy_call_varargs = varargs
    expm_multiply.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expm_multiply', ['A', 'B', 'start', 'stop', 'num', 'endpoint'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expm_multiply', localization, ['A', 'B', 'start', 'stop', 'num', 'endpoint'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expm_multiply(...)' code ##################

    str_388731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, (-1)), 'str', '\n    Compute the action of the matrix exponential of A on B.\n\n    Parameters\n    ----------\n    A : transposable linear operator\n        The operator whose exponential is of interest.\n    B : ndarray\n        The matrix or vector to be multiplied by the matrix exponential of A.\n    start : scalar, optional\n        The starting time point of the sequence.\n    stop : scalar, optional\n        The end time point of the sequence, unless `endpoint` is set to False.\n        In that case, the sequence consists of all but the last of ``num + 1``\n        evenly spaced time points, so that `stop` is excluded.\n        Note that the step size changes when `endpoint` is False.\n    num : int, optional\n        Number of time points to use.\n    endpoint : bool, optional\n        If True, `stop` is the last time point.  Otherwise, it is not included.\n\n    Returns\n    -------\n    expm_A_B : ndarray\n         The result of the action :math:`e^{t_k A} B`.\n\n    Notes\n    -----\n    The optional arguments defining the sequence of evenly spaced time points\n    are compatible with the arguments of `numpy.linspace`.\n\n    The output ndarray shape is somewhat complicated so I explain it here.\n    The ndim of the output could be either 1, 2, or 3.\n    It would be 1 if you are computing the expm action on a single vector\n    at a single time point.\n    It would be 2 if you are computing the expm action on a vector\n    at multiple time points, or if you are computing the expm action\n    on a matrix at a single time point.\n    It would be 3 if you want the action on a matrix with multiple\n    columns at multiple time points.\n    If multiple time points are requested, expm_A_B[0] will always\n    be the action of the expm at the first time point,\n    regardless of whether the action is on a vector or a matrix.\n\n    References\n    ----------\n    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2011)\n           "Computing the Action of the Matrix Exponential,\n           with an Application to Exponential Integrators."\n           SIAM Journal on Scientific Computing,\n           33 (2). pp. 488-511. ISSN 1064-8275\n           http://eprints.ma.man.ac.uk/1591/\n\n    .. [2] Nicholas J. Higham and Awad H. Al-Mohy (2010)\n           "Computing Matrix Functions."\n           Acta Numerica,\n           19. 159-208. ISSN 0962-4929\n           http://eprints.ma.man.ac.uk/1451/\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import expm, expm_multiply\n    >>> A = csc_matrix([[1, 0], [0, 1]])\n    >>> A.todense()\n    matrix([[1, 0],\n            [0, 1]], dtype=int64)\n    >>> B = np.array([np.exp(-1.), np.exp(-2.)])\n    >>> B\n    array([ 0.36787944,  0.13533528])\n    >>> expm_multiply(A, B, start=1, stop=2, num=3, endpoint=True)\n    array([[ 1.        ,  0.36787944],\n           [ 1.64872127,  0.60653066],\n           [ 2.71828183,  1.        ]])\n    >>> expm(A).dot(B)                  # Verify 1st timestep\n    array([ 1.        ,  0.36787944])\n    >>> expm(1.5*A).dot(B)              # Verify 2nd timestep\n    array([ 1.64872127,  0.60653066])\n    >>> expm(2*A).dot(B)                # Verify 3rd timestep\n    array([ 2.71828183,  1.        ])\n    ')
    
    
    # Call to all(...): (line 130)
    # Processing the call arguments (line 130)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 130, 11, True)
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'tuple' (line 130)
    tuple_388736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 130)
    # Adding element type (line 130)
    # Getting the type of 'start' (line 130)
    start_388737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'start', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 35), tuple_388736, start_388737)
    # Adding element type (line 130)
    # Getting the type of 'stop' (line 130)
    stop_388738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 42), 'stop', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 35), tuple_388736, stop_388738)
    # Adding element type (line 130)
    # Getting the type of 'num' (line 130)
    num_388739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 48), 'num', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 35), tuple_388736, num_388739)
    # Adding element type (line 130)
    # Getting the type of 'endpoint' (line 130)
    endpoint_388740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 53), 'endpoint', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 35), tuple_388736, endpoint_388740)
    
    comprehension_388741 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 11), tuple_388736)
    # Assigning a type to the variable 'arg' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'arg', comprehension_388741)
    
    # Getting the type of 'arg' (line 130)
    arg_388733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'arg', False)
    # Getting the type of 'None' (line 130)
    None_388734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'None', False)
    # Applying the binary operator 'is' (line 130)
    result_is__388735 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), 'is', arg_388733, None_388734)
    
    list_388742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 11), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 11), list_388742, result_is__388735)
    # Processing the call keyword arguments (line 130)
    kwargs_388743 = {}
    # Getting the type of 'all' (line 130)
    all_388732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 7), 'all', False)
    # Calling all(args, kwargs) (line 130)
    all_call_result_388744 = invoke(stypy.reporting.localization.Localization(__file__, 130, 7), all_388732, *[list_388742], **kwargs_388743)
    
    # Testing the type of an if condition (line 130)
    if_condition_388745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 4), all_call_result_388744)
    # Assigning a type to the variable 'if_condition_388745' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'if_condition_388745', if_condition_388745)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to _expm_multiply_simple(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'A' (line 131)
    A_388747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 34), 'A', False)
    # Getting the type of 'B' (line 131)
    B_388748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 37), 'B', False)
    # Processing the call keyword arguments (line 131)
    kwargs_388749 = {}
    # Getting the type of '_expm_multiply_simple' (line 131)
    _expm_multiply_simple_388746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), '_expm_multiply_simple', False)
    # Calling _expm_multiply_simple(args, kwargs) (line 131)
    _expm_multiply_simple_call_result_388750 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), _expm_multiply_simple_388746, *[A_388747, B_388748], **kwargs_388749)
    
    # Assigning a type to the variable 'X' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'X', _expm_multiply_simple_call_result_388750)
    # SSA branch for the else part of an if statement (line 130)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 133):
    
    # Assigning a Subscript to a Name (line 133):
    
    # Obtaining the type of the subscript
    int_388751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'int')
    
    # Call to _expm_multiply_interval(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'A' (line 133)
    A_388753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 44), 'A', False)
    # Getting the type of 'B' (line 133)
    B_388754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 47), 'B', False)
    # Getting the type of 'start' (line 133)
    start_388755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 50), 'start', False)
    # Getting the type of 'stop' (line 133)
    stop_388756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 57), 'stop', False)
    # Getting the type of 'num' (line 133)
    num_388757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 63), 'num', False)
    # Getting the type of 'endpoint' (line 133)
    endpoint_388758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 68), 'endpoint', False)
    # Processing the call keyword arguments (line 133)
    kwargs_388759 = {}
    # Getting the type of '_expm_multiply_interval' (line 133)
    _expm_multiply_interval_388752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 20), '_expm_multiply_interval', False)
    # Calling _expm_multiply_interval(args, kwargs) (line 133)
    _expm_multiply_interval_call_result_388760 = invoke(stypy.reporting.localization.Localization(__file__, 133, 20), _expm_multiply_interval_388752, *[A_388753, B_388754, start_388755, stop_388756, num_388757, endpoint_388758], **kwargs_388759)
    
    # Obtaining the member '__getitem__' of a type (line 133)
    getitem___388761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), _expm_multiply_interval_call_result_388760, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 133)
    subscript_call_result_388762 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), getitem___388761, int_388751)
    
    # Assigning a type to the variable 'tuple_var_assignment_388576' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_388576', subscript_call_result_388762)
    
    # Assigning a Subscript to a Name (line 133):
    
    # Obtaining the type of the subscript
    int_388763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'int')
    
    # Call to _expm_multiply_interval(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'A' (line 133)
    A_388765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 44), 'A', False)
    # Getting the type of 'B' (line 133)
    B_388766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 47), 'B', False)
    # Getting the type of 'start' (line 133)
    start_388767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 50), 'start', False)
    # Getting the type of 'stop' (line 133)
    stop_388768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 57), 'stop', False)
    # Getting the type of 'num' (line 133)
    num_388769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 63), 'num', False)
    # Getting the type of 'endpoint' (line 133)
    endpoint_388770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 68), 'endpoint', False)
    # Processing the call keyword arguments (line 133)
    kwargs_388771 = {}
    # Getting the type of '_expm_multiply_interval' (line 133)
    _expm_multiply_interval_388764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 20), '_expm_multiply_interval', False)
    # Calling _expm_multiply_interval(args, kwargs) (line 133)
    _expm_multiply_interval_call_result_388772 = invoke(stypy.reporting.localization.Localization(__file__, 133, 20), _expm_multiply_interval_388764, *[A_388765, B_388766, start_388767, stop_388768, num_388769, endpoint_388770], **kwargs_388771)
    
    # Obtaining the member '__getitem__' of a type (line 133)
    getitem___388773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), _expm_multiply_interval_call_result_388772, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 133)
    subscript_call_result_388774 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), getitem___388773, int_388763)
    
    # Assigning a type to the variable 'tuple_var_assignment_388577' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_388577', subscript_call_result_388774)
    
    # Assigning a Name to a Name (line 133):
    # Getting the type of 'tuple_var_assignment_388576' (line 133)
    tuple_var_assignment_388576_388775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_388576')
    # Assigning a type to the variable 'X' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'X', tuple_var_assignment_388576_388775)
    
    # Assigning a Name to a Name (line 133):
    # Getting the type of 'tuple_var_assignment_388577' (line 133)
    tuple_var_assignment_388577_388776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_388577')
    # Assigning a type to the variable 'status' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'status', tuple_var_assignment_388577_388776)
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'X' (line 134)
    X_388777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'X')
    # Assigning a type to the variable 'stypy_return_type' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type', X_388777)
    
    # ################# End of 'expm_multiply(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expm_multiply' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_388778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_388778)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expm_multiply'
    return stypy_return_type_388778

# Assigning a type to the variable 'expm_multiply' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'expm_multiply', expm_multiply)

@norecursion
def _expm_multiply_simple(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_388779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 34), 'float')
    # Getting the type of 'False' (line 137)
    False_388780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 47), 'False')
    defaults = [float_388779, False_388780]
    # Create a new context for function '_expm_multiply_simple'
    module_type_store = module_type_store.open_function_context('_expm_multiply_simple', 137, 0, False)
    
    # Passed parameters checking function
    _expm_multiply_simple.stypy_localization = localization
    _expm_multiply_simple.stypy_type_of_self = None
    _expm_multiply_simple.stypy_type_store = module_type_store
    _expm_multiply_simple.stypy_function_name = '_expm_multiply_simple'
    _expm_multiply_simple.stypy_param_names_list = ['A', 'B', 't', 'balance']
    _expm_multiply_simple.stypy_varargs_param_name = None
    _expm_multiply_simple.stypy_kwargs_param_name = None
    _expm_multiply_simple.stypy_call_defaults = defaults
    _expm_multiply_simple.stypy_call_varargs = varargs
    _expm_multiply_simple.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_expm_multiply_simple', ['A', 'B', 't', 'balance'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_expm_multiply_simple', localization, ['A', 'B', 't', 'balance'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_expm_multiply_simple(...)' code ##################

    str_388781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, (-1)), 'str', '\n    Compute the action of the matrix exponential at a single time point.\n\n    Parameters\n    ----------\n    A : transposable linear operator\n        The operator whose exponential is of interest.\n    B : ndarray\n        The matrix to be multiplied by the matrix exponential of A.\n    t : float\n        A time point.\n    balance : bool\n        Indicates whether or not to apply balancing.\n\n    Returns\n    -------\n    F : ndarray\n        :math:`e^{t A} B`\n\n    Notes\n    -----\n    This is algorithm (3.2) in Al-Mohy and Higham (2011).\n\n    ')
    
    # Getting the type of 'balance' (line 162)
    balance_388782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 7), 'balance')
    # Testing the type of an if condition (line 162)
    if_condition_388783 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 4), balance_388782)
    # Assigning a type to the variable 'if_condition_388783' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'if_condition_388783', if_condition_388783)
    # SSA begins for if statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'NotImplementedError' (line 163)
    NotImplementedError_388784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 14), 'NotImplementedError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 163, 8), NotImplementedError_388784, 'raise parameter', BaseException)
    # SSA join for if statement (line 162)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'A' (line 164)
    A_388786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 164)
    shape_388787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 11), A_388786, 'shape')
    # Processing the call keyword arguments (line 164)
    kwargs_388788 = {}
    # Getting the type of 'len' (line 164)
    len_388785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 7), 'len', False)
    # Calling len(args, kwargs) (line 164)
    len_call_result_388789 = invoke(stypy.reporting.localization.Localization(__file__, 164, 7), len_388785, *[shape_388787], **kwargs_388788)
    
    int_388790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'int')
    # Applying the binary operator '!=' (line 164)
    result_ne_388791 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 7), '!=', len_call_result_388789, int_388790)
    
    
    
    # Obtaining the type of the subscript
    int_388792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 36), 'int')
    # Getting the type of 'A' (line 164)
    A_388793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'A')
    # Obtaining the member 'shape' of a type (line 164)
    shape_388794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 28), A_388793, 'shape')
    # Obtaining the member '__getitem__' of a type (line 164)
    getitem___388795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 28), shape_388794, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 164)
    subscript_call_result_388796 = invoke(stypy.reporting.localization.Localization(__file__, 164, 28), getitem___388795, int_388792)
    
    
    # Obtaining the type of the subscript
    int_388797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 50), 'int')
    # Getting the type of 'A' (line 164)
    A_388798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 42), 'A')
    # Obtaining the member 'shape' of a type (line 164)
    shape_388799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 42), A_388798, 'shape')
    # Obtaining the member '__getitem__' of a type (line 164)
    getitem___388800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 42), shape_388799, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 164)
    subscript_call_result_388801 = invoke(stypy.reporting.localization.Localization(__file__, 164, 42), getitem___388800, int_388797)
    
    # Applying the binary operator '!=' (line 164)
    result_ne_388802 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 28), '!=', subscript_call_result_388796, subscript_call_result_388801)
    
    # Applying the binary operator 'or' (line 164)
    result_or_keyword_388803 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 7), 'or', result_ne_388791, result_ne_388802)
    
    # Testing the type of an if condition (line 164)
    if_condition_388804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 4), result_or_keyword_388803)
    # Assigning a type to the variable 'if_condition_388804' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'if_condition_388804', if_condition_388804)
    # SSA begins for if statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 165)
    # Processing the call arguments (line 165)
    str_388806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 25), 'str', 'expected A to be like a square matrix')
    # Processing the call keyword arguments (line 165)
    kwargs_388807 = {}
    # Getting the type of 'ValueError' (line 165)
    ValueError_388805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 165)
    ValueError_call_result_388808 = invoke(stypy.reporting.localization.Localization(__file__, 165, 14), ValueError_388805, *[str_388806], **kwargs_388807)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 165, 8), ValueError_call_result_388808, 'raise parameter', BaseException)
    # SSA join for if statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_388809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 15), 'int')
    # Getting the type of 'A' (line 166)
    A_388810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 7), 'A')
    # Obtaining the member 'shape' of a type (line 166)
    shape_388811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 7), A_388810, 'shape')
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___388812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 7), shape_388811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_388813 = invoke(stypy.reporting.localization.Localization(__file__, 166, 7), getitem___388812, int_388809)
    
    
    # Obtaining the type of the subscript
    int_388814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 29), 'int')
    # Getting the type of 'B' (line 166)
    B_388815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'B')
    # Obtaining the member 'shape' of a type (line 166)
    shape_388816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 21), B_388815, 'shape')
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___388817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 21), shape_388816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_388818 = invoke(stypy.reporting.localization.Localization(__file__, 166, 21), getitem___388817, int_388814)
    
    # Applying the binary operator '!=' (line 166)
    result_ne_388819 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 7), '!=', subscript_call_result_388813, subscript_call_result_388818)
    
    # Testing the type of an if condition (line 166)
    if_condition_388820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 4), result_ne_388819)
    # Assigning a type to the variable 'if_condition_388820' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'if_condition_388820', if_condition_388820)
    # SSA begins for if statement (line 166)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 167)
    # Processing the call arguments (line 167)
    str_388822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 25), 'str', 'the matrices A and B have incompatible shapes')
    # Processing the call keyword arguments (line 167)
    kwargs_388823 = {}
    # Getting the type of 'ValueError' (line 167)
    ValueError_388821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 167)
    ValueError_call_result_388824 = invoke(stypy.reporting.localization.Localization(__file__, 167, 14), ValueError_388821, *[str_388822], **kwargs_388823)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 167, 8), ValueError_call_result_388824, 'raise parameter', BaseException)
    # SSA join for if statement (line 166)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to _ident_like(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'A' (line 168)
    A_388826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'A', False)
    # Processing the call keyword arguments (line 168)
    kwargs_388827 = {}
    # Getting the type of '_ident_like' (line 168)
    _ident_like_388825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), '_ident_like', False)
    # Calling _ident_like(args, kwargs) (line 168)
    _ident_like_call_result_388828 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), _ident_like_388825, *[A_388826], **kwargs_388827)
    
    # Assigning a type to the variable 'ident' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'ident', _ident_like_call_result_388828)
    
    # Assigning a Subscript to a Name (line 169):
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_388829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 16), 'int')
    # Getting the type of 'A' (line 169)
    A_388830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'A')
    # Obtaining the member 'shape' of a type (line 169)
    shape_388831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), A_388830, 'shape')
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___388832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), shape_388831, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_388833 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), getitem___388832, int_388829)
    
    # Assigning a type to the variable 'n' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'n', subscript_call_result_388833)
    
    
    
    # Call to len(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'B' (line 170)
    B_388835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'B', False)
    # Obtaining the member 'shape' of a type (line 170)
    shape_388836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 11), B_388835, 'shape')
    # Processing the call keyword arguments (line 170)
    kwargs_388837 = {}
    # Getting the type of 'len' (line 170)
    len_388834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 7), 'len', False)
    # Calling len(args, kwargs) (line 170)
    len_call_result_388838 = invoke(stypy.reporting.localization.Localization(__file__, 170, 7), len_388834, *[shape_388836], **kwargs_388837)
    
    int_388839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'int')
    # Applying the binary operator '==' (line 170)
    result_eq_388840 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 7), '==', len_call_result_388838, int_388839)
    
    # Testing the type of an if condition (line 170)
    if_condition_388841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 4), result_eq_388840)
    # Assigning a type to the variable 'if_condition_388841' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'if_condition_388841', if_condition_388841)
    # SSA begins for if statement (line 170)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 171):
    
    # Assigning a Num to a Name (line 171):
    int_388842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 13), 'int')
    # Assigning a type to the variable 'n0' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'n0', int_388842)
    # SSA branch for the else part of an if statement (line 170)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'B' (line 172)
    B_388844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 13), 'B', False)
    # Obtaining the member 'shape' of a type (line 172)
    shape_388845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 13), B_388844, 'shape')
    # Processing the call keyword arguments (line 172)
    kwargs_388846 = {}
    # Getting the type of 'len' (line 172)
    len_388843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 9), 'len', False)
    # Calling len(args, kwargs) (line 172)
    len_call_result_388847 = invoke(stypy.reporting.localization.Localization(__file__, 172, 9), len_388843, *[shape_388845], **kwargs_388846)
    
    int_388848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 25), 'int')
    # Applying the binary operator '==' (line 172)
    result_eq_388849 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 9), '==', len_call_result_388847, int_388848)
    
    # Testing the type of an if condition (line 172)
    if_condition_388850 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 9), result_eq_388849)
    # Assigning a type to the variable 'if_condition_388850' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 9), 'if_condition_388850', if_condition_388850)
    # SSA begins for if statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 173):
    
    # Assigning a Subscript to a Name (line 173):
    
    # Obtaining the type of the subscript
    int_388851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 21), 'int')
    # Getting the type of 'B' (line 173)
    B_388852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), 'B')
    # Obtaining the member 'shape' of a type (line 173)
    shape_388853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 13), B_388852, 'shape')
    # Obtaining the member '__getitem__' of a type (line 173)
    getitem___388854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 13), shape_388853, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 173)
    subscript_call_result_388855 = invoke(stypy.reporting.localization.Localization(__file__, 173, 13), getitem___388854, int_388851)
    
    # Assigning a type to the variable 'n0' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'n0', subscript_call_result_388855)
    # SSA branch for the else part of an if statement (line 172)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 175)
    # Processing the call arguments (line 175)
    str_388857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 25), 'str', 'expected B to be like a matrix or a vector')
    # Processing the call keyword arguments (line 175)
    kwargs_388858 = {}
    # Getting the type of 'ValueError' (line 175)
    ValueError_388856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 175)
    ValueError_call_result_388859 = invoke(stypy.reporting.localization.Localization(__file__, 175, 14), ValueError_388856, *[str_388857], **kwargs_388858)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 175, 8), ValueError_call_result_388859, 'raise parameter', BaseException)
    # SSA join for if statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 170)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 176):
    
    # Assigning a BinOp to a Name (line 176):
    int_388860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 10), 'int')
    int_388861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 13), 'int')
    # Applying the binary operator '**' (line 176)
    result_pow_388862 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 10), '**', int_388860, int_388861)
    
    # Assigning a type to the variable 'u_d' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'u_d', result_pow_388862)
    
    # Assigning a Name to a Name (line 177):
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'u_d' (line 177)
    u_d_388863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 10), 'u_d')
    # Assigning a type to the variable 'tol' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tol', u_d_388863)
    
    # Assigning a BinOp to a Name (line 178):
    
    # Assigning a BinOp to a Name (line 178):
    
    # Call to _trace(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'A' (line 178)
    A_388865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'A', False)
    # Processing the call keyword arguments (line 178)
    kwargs_388866 = {}
    # Getting the type of '_trace' (line 178)
    _trace_388864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 9), '_trace', False)
    # Calling _trace(args, kwargs) (line 178)
    _trace_call_result_388867 = invoke(stypy.reporting.localization.Localization(__file__, 178, 9), _trace_388864, *[A_388865], **kwargs_388866)
    
    
    # Call to float(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'n' (line 178)
    n_388869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'n', False)
    # Processing the call keyword arguments (line 178)
    kwargs_388870 = {}
    # Getting the type of 'float' (line 178)
    float_388868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'float', False)
    # Calling float(args, kwargs) (line 178)
    float_call_result_388871 = invoke(stypy.reporting.localization.Localization(__file__, 178, 21), float_388868, *[n_388869], **kwargs_388870)
    
    # Applying the binary operator 'div' (line 178)
    result_div_388872 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 9), 'div', _trace_call_result_388867, float_call_result_388871)
    
    # Assigning a type to the variable 'mu' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'mu', result_div_388872)
    
    # Assigning a BinOp to a Name (line 179):
    
    # Assigning a BinOp to a Name (line 179):
    # Getting the type of 'A' (line 179)
    A_388873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'A')
    # Getting the type of 'mu' (line 179)
    mu_388874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'mu')
    # Getting the type of 'ident' (line 179)
    ident_388875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 17), 'ident')
    # Applying the binary operator '*' (line 179)
    result_mul_388876 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 12), '*', mu_388874, ident_388875)
    
    # Applying the binary operator '-' (line 179)
    result_sub_388877 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 8), '-', A_388873, result_mul_388876)
    
    # Assigning a type to the variable 'A' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'A', result_sub_388877)
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to _exact_1_norm(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'A' (line 180)
    A_388879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 29), 'A', False)
    # Processing the call keyword arguments (line 180)
    kwargs_388880 = {}
    # Getting the type of '_exact_1_norm' (line 180)
    _exact_1_norm_388878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), '_exact_1_norm', False)
    # Calling _exact_1_norm(args, kwargs) (line 180)
    _exact_1_norm_call_result_388881 = invoke(stypy.reporting.localization.Localization(__file__, 180, 15), _exact_1_norm_388878, *[A_388879], **kwargs_388880)
    
    # Assigning a type to the variable 'A_1_norm' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'A_1_norm', _exact_1_norm_call_result_388881)
    
    
    # Getting the type of 't' (line 181)
    t_388882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 7), 't')
    # Getting the type of 'A_1_norm' (line 181)
    A_1_norm_388883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 9), 'A_1_norm')
    # Applying the binary operator '*' (line 181)
    result_mul_388884 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 7), '*', t_388882, A_1_norm_388883)
    
    int_388885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 21), 'int')
    # Applying the binary operator '==' (line 181)
    result_eq_388886 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 7), '==', result_mul_388884, int_388885)
    
    # Testing the type of an if condition (line 181)
    if_condition_388887 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 4), result_eq_388886)
    # Assigning a type to the variable 'if_condition_388887' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'if_condition_388887', if_condition_388887)
    # SSA begins for if statement (line 181)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 182):
    
    # Assigning a Num to a Name (line 182):
    int_388888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 20), 'int')
    # Assigning a type to the variable 'tuple_assignment_388578' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_assignment_388578', int_388888)
    
    # Assigning a Num to a Name (line 182):
    int_388889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 23), 'int')
    # Assigning a type to the variable 'tuple_assignment_388579' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_assignment_388579', int_388889)
    
    # Assigning a Name to a Name (line 182):
    # Getting the type of 'tuple_assignment_388578' (line 182)
    tuple_assignment_388578_388890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_assignment_388578')
    # Assigning a type to the variable 'm_star' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'm_star', tuple_assignment_388578_388890)
    
    # Assigning a Name to a Name (line 182):
    # Getting the type of 'tuple_assignment_388579' (line 182)
    tuple_assignment_388579_388891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_assignment_388579')
    # Assigning a type to the variable 's' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 's', tuple_assignment_388579_388891)
    # SSA branch for the else part of an if statement (line 181)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 184):
    
    # Assigning a Num to a Name (line 184):
    int_388892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 14), 'int')
    # Assigning a type to the variable 'ell' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'ell', int_388892)
    
    # Assigning a Call to a Name (line 185):
    
    # Assigning a Call to a Name (line 185):
    
    # Call to LazyOperatorNormInfo(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 't' (line 185)
    t_388894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 41), 't', False)
    # Getting the type of 'A' (line 185)
    A_388895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 43), 'A', False)
    # Applying the binary operator '*' (line 185)
    result_mul_388896 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 41), '*', t_388894, A_388895)
    
    # Processing the call keyword arguments (line 185)
    # Getting the type of 't' (line 185)
    t_388897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 55), 't', False)
    # Getting the type of 'A_1_norm' (line 185)
    A_1_norm_388898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 57), 'A_1_norm', False)
    # Applying the binary operator '*' (line 185)
    result_mul_388899 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 55), '*', t_388897, A_1_norm_388898)
    
    keyword_388900 = result_mul_388899
    # Getting the type of 'ell' (line 185)
    ell_388901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 71), 'ell', False)
    keyword_388902 = ell_388901
    kwargs_388903 = {'A_1_norm': keyword_388900, 'ell': keyword_388902}
    # Getting the type of 'LazyOperatorNormInfo' (line 185)
    LazyOperatorNormInfo_388893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'LazyOperatorNormInfo', False)
    # Calling LazyOperatorNormInfo(args, kwargs) (line 185)
    LazyOperatorNormInfo_call_result_388904 = invoke(stypy.reporting.localization.Localization(__file__, 185, 20), LazyOperatorNormInfo_388893, *[result_mul_388896], **kwargs_388903)
    
    # Assigning a type to the variable 'norm_info' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'norm_info', LazyOperatorNormInfo_call_result_388904)
    
    # Assigning a Call to a Tuple (line 186):
    
    # Assigning a Subscript to a Name (line 186):
    
    # Obtaining the type of the subscript
    int_388905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 8), 'int')
    
    # Call to _fragment_3_1(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'norm_info' (line 186)
    norm_info_388907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 34), 'norm_info', False)
    # Getting the type of 'n0' (line 186)
    n0_388908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 45), 'n0', False)
    # Getting the type of 'tol' (line 186)
    tol_388909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 49), 'tol', False)
    # Processing the call keyword arguments (line 186)
    # Getting the type of 'ell' (line 186)
    ell_388910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 58), 'ell', False)
    keyword_388911 = ell_388910
    kwargs_388912 = {'ell': keyword_388911}
    # Getting the type of '_fragment_3_1' (line 186)
    _fragment_3_1_388906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), '_fragment_3_1', False)
    # Calling _fragment_3_1(args, kwargs) (line 186)
    _fragment_3_1_call_result_388913 = invoke(stypy.reporting.localization.Localization(__file__, 186, 20), _fragment_3_1_388906, *[norm_info_388907, n0_388908, tol_388909], **kwargs_388912)
    
    # Obtaining the member '__getitem__' of a type (line 186)
    getitem___388914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), _fragment_3_1_call_result_388913, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
    subscript_call_result_388915 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), getitem___388914, int_388905)
    
    # Assigning a type to the variable 'tuple_var_assignment_388580' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'tuple_var_assignment_388580', subscript_call_result_388915)
    
    # Assigning a Subscript to a Name (line 186):
    
    # Obtaining the type of the subscript
    int_388916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 8), 'int')
    
    # Call to _fragment_3_1(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'norm_info' (line 186)
    norm_info_388918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 34), 'norm_info', False)
    # Getting the type of 'n0' (line 186)
    n0_388919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 45), 'n0', False)
    # Getting the type of 'tol' (line 186)
    tol_388920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 49), 'tol', False)
    # Processing the call keyword arguments (line 186)
    # Getting the type of 'ell' (line 186)
    ell_388921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 58), 'ell', False)
    keyword_388922 = ell_388921
    kwargs_388923 = {'ell': keyword_388922}
    # Getting the type of '_fragment_3_1' (line 186)
    _fragment_3_1_388917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), '_fragment_3_1', False)
    # Calling _fragment_3_1(args, kwargs) (line 186)
    _fragment_3_1_call_result_388924 = invoke(stypy.reporting.localization.Localization(__file__, 186, 20), _fragment_3_1_388917, *[norm_info_388918, n0_388919, tol_388920], **kwargs_388923)
    
    # Obtaining the member '__getitem__' of a type (line 186)
    getitem___388925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), _fragment_3_1_call_result_388924, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
    subscript_call_result_388926 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), getitem___388925, int_388916)
    
    # Assigning a type to the variable 'tuple_var_assignment_388581' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'tuple_var_assignment_388581', subscript_call_result_388926)
    
    # Assigning a Name to a Name (line 186):
    # Getting the type of 'tuple_var_assignment_388580' (line 186)
    tuple_var_assignment_388580_388927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'tuple_var_assignment_388580')
    # Assigning a type to the variable 'm_star' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'm_star', tuple_var_assignment_388580_388927)
    
    # Assigning a Name to a Name (line 186):
    # Getting the type of 'tuple_var_assignment_388581' (line 186)
    tuple_var_assignment_388581_388928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'tuple_var_assignment_388581')
    # Assigning a type to the variable 's' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 's', tuple_var_assignment_388581_388928)
    # SSA join for if statement (line 181)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _expm_multiply_simple_core(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'A' (line 187)
    A_388930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 38), 'A', False)
    # Getting the type of 'B' (line 187)
    B_388931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 41), 'B', False)
    # Getting the type of 't' (line 187)
    t_388932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 44), 't', False)
    # Getting the type of 'mu' (line 187)
    mu_388933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 47), 'mu', False)
    # Getting the type of 'm_star' (line 187)
    m_star_388934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 51), 'm_star', False)
    # Getting the type of 's' (line 187)
    s_388935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 59), 's', False)
    # Getting the type of 'tol' (line 187)
    tol_388936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 62), 'tol', False)
    # Getting the type of 'balance' (line 187)
    balance_388937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 67), 'balance', False)
    # Processing the call keyword arguments (line 187)
    kwargs_388938 = {}
    # Getting the type of '_expm_multiply_simple_core' (line 187)
    _expm_multiply_simple_core_388929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), '_expm_multiply_simple_core', False)
    # Calling _expm_multiply_simple_core(args, kwargs) (line 187)
    _expm_multiply_simple_core_call_result_388939 = invoke(stypy.reporting.localization.Localization(__file__, 187, 11), _expm_multiply_simple_core_388929, *[A_388930, B_388931, t_388932, mu_388933, m_star_388934, s_388935, tol_388936, balance_388937], **kwargs_388938)
    
    # Assigning a type to the variable 'stypy_return_type' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type', _expm_multiply_simple_core_call_result_388939)
    
    # ################# End of '_expm_multiply_simple(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_expm_multiply_simple' in the type store
    # Getting the type of 'stypy_return_type' (line 137)
    stypy_return_type_388940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_388940)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_expm_multiply_simple'
    return stypy_return_type_388940

# Assigning a type to the variable '_expm_multiply_simple' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), '_expm_multiply_simple', _expm_multiply_simple)

@norecursion
def _expm_multiply_simple_core(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 190)
    None_388941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 59), 'None')
    # Getting the type of 'False' (line 190)
    False_388942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 73), 'False')
    defaults = [None_388941, False_388942]
    # Create a new context for function '_expm_multiply_simple_core'
    module_type_store = module_type_store.open_function_context('_expm_multiply_simple_core', 190, 0, False)
    
    # Passed parameters checking function
    _expm_multiply_simple_core.stypy_localization = localization
    _expm_multiply_simple_core.stypy_type_of_self = None
    _expm_multiply_simple_core.stypy_type_store = module_type_store
    _expm_multiply_simple_core.stypy_function_name = '_expm_multiply_simple_core'
    _expm_multiply_simple_core.stypy_param_names_list = ['A', 'B', 't', 'mu', 'm_star', 's', 'tol', 'balance']
    _expm_multiply_simple_core.stypy_varargs_param_name = None
    _expm_multiply_simple_core.stypy_kwargs_param_name = None
    _expm_multiply_simple_core.stypy_call_defaults = defaults
    _expm_multiply_simple_core.stypy_call_varargs = varargs
    _expm_multiply_simple_core.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_expm_multiply_simple_core', ['A', 'B', 't', 'mu', 'm_star', 's', 'tol', 'balance'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_expm_multiply_simple_core', localization, ['A', 'B', 't', 'mu', 'm_star', 's', 'tol', 'balance'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_expm_multiply_simple_core(...)' code ##################

    str_388943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, (-1)), 'str', '\n    A helper function.\n    ')
    
    # Getting the type of 'balance' (line 194)
    balance_388944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 7), 'balance')
    # Testing the type of an if condition (line 194)
    if_condition_388945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 4), balance_388944)
    # Assigning a type to the variable 'if_condition_388945' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'if_condition_388945', if_condition_388945)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'NotImplementedError' (line 195)
    NotImplementedError_388946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 14), 'NotImplementedError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 195, 8), NotImplementedError_388946, 'raise parameter', BaseException)
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 196)
    # Getting the type of 'tol' (line 196)
    tol_388947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 7), 'tol')
    # Getting the type of 'None' (line 196)
    None_388948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 14), 'None')
    
    (may_be_388949, more_types_in_union_388950) = may_be_none(tol_388947, None_388948)

    if may_be_388949:

        if more_types_in_union_388950:
            # Runtime conditional SSA (line 196)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 197):
        
        # Assigning a BinOp to a Name (line 197):
        int_388951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 14), 'int')
        int_388952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 19), 'int')
        # Applying the binary operator '**' (line 197)
        result_pow_388953 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 14), '**', int_388951, int_388952)
        
        # Assigning a type to the variable 'u_d' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'u_d', result_pow_388953)
        
        # Assigning a Name to a Name (line 198):
        
        # Assigning a Name to a Name (line 198):
        # Getting the type of 'u_d' (line 198)
        u_d_388954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 14), 'u_d')
        # Assigning a type to the variable 'tol' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tol', u_d_388954)

        if more_types_in_union_388950:
            # SSA join for if statement (line 196)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Name (line 199):
    
    # Assigning a Name to a Name (line 199):
    # Getting the type of 'B' (line 199)
    B_388955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'B')
    # Assigning a type to the variable 'F' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'F', B_388955)
    
    # Assigning a Call to a Name (line 200):
    
    # Assigning a Call to a Name (line 200):
    
    # Call to exp(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 't' (line 200)
    t_388958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 17), 't', False)
    # Getting the type of 'mu' (line 200)
    mu_388959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'mu', False)
    # Applying the binary operator '*' (line 200)
    result_mul_388960 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 17), '*', t_388958, mu_388959)
    
    
    # Call to float(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 's' (line 200)
    s_388962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 30), 's', False)
    # Processing the call keyword arguments (line 200)
    kwargs_388963 = {}
    # Getting the type of 'float' (line 200)
    float_388961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 24), 'float', False)
    # Calling float(args, kwargs) (line 200)
    float_call_result_388964 = invoke(stypy.reporting.localization.Localization(__file__, 200, 24), float_388961, *[s_388962], **kwargs_388963)
    
    # Applying the binary operator 'div' (line 200)
    result_div_388965 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 22), 'div', result_mul_388960, float_call_result_388964)
    
    # Processing the call keyword arguments (line 200)
    kwargs_388966 = {}
    # Getting the type of 'np' (line 200)
    np_388956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 10), 'np', False)
    # Obtaining the member 'exp' of a type (line 200)
    exp_388957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 10), np_388956, 'exp')
    # Calling exp(args, kwargs) (line 200)
    exp_call_result_388967 = invoke(stypy.reporting.localization.Localization(__file__, 200, 10), exp_388957, *[result_div_388965], **kwargs_388966)
    
    # Assigning a type to the variable 'eta' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'eta', exp_call_result_388967)
    
    
    # Call to range(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 's' (line 201)
    s_388969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 's', False)
    # Processing the call keyword arguments (line 201)
    kwargs_388970 = {}
    # Getting the type of 'range' (line 201)
    range_388968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 13), 'range', False)
    # Calling range(args, kwargs) (line 201)
    range_call_result_388971 = invoke(stypy.reporting.localization.Localization(__file__, 201, 13), range_388968, *[s_388969], **kwargs_388970)
    
    # Testing the type of a for loop iterable (line 201)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 201, 4), range_call_result_388971)
    # Getting the type of the for loop variable (line 201)
    for_loop_var_388972 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 201, 4), range_call_result_388971)
    # Assigning a type to the variable 'i' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'i', for_loop_var_388972)
    # SSA begins for a for statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Call to a Name (line 202):
    
    # Call to _exact_inf_norm(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'B' (line 202)
    B_388974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), 'B', False)
    # Processing the call keyword arguments (line 202)
    kwargs_388975 = {}
    # Getting the type of '_exact_inf_norm' (line 202)
    _exact_inf_norm_388973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 13), '_exact_inf_norm', False)
    # Calling _exact_inf_norm(args, kwargs) (line 202)
    _exact_inf_norm_call_result_388976 = invoke(stypy.reporting.localization.Localization(__file__, 202, 13), _exact_inf_norm_388973, *[B_388974], **kwargs_388975)
    
    # Assigning a type to the variable 'c1' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'c1', _exact_inf_norm_call_result_388976)
    
    
    # Call to range(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'm_star' (line 203)
    m_star_388978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'm_star', False)
    # Processing the call keyword arguments (line 203)
    kwargs_388979 = {}
    # Getting the type of 'range' (line 203)
    range_388977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 17), 'range', False)
    # Calling range(args, kwargs) (line 203)
    range_call_result_388980 = invoke(stypy.reporting.localization.Localization(__file__, 203, 17), range_388977, *[m_star_388978], **kwargs_388979)
    
    # Testing the type of a for loop iterable (line 203)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 203, 8), range_call_result_388980)
    # Getting the type of the for loop variable (line 203)
    for_loop_var_388981 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 203, 8), range_call_result_388980)
    # Assigning a type to the variable 'j' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'j', for_loop_var_388981)
    # SSA begins for a for statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 204):
    
    # Assigning a BinOp to a Name (line 204):
    # Getting the type of 't' (line 204)
    t_388982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 't')
    
    # Call to float(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 's' (line 204)
    s_388984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 30), 's', False)
    # Getting the type of 'j' (line 204)
    j_388985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 33), 'j', False)
    int_388986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 35), 'int')
    # Applying the binary operator '+' (line 204)
    result_add_388987 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 33), '+', j_388985, int_388986)
    
    # Applying the binary operator '*' (line 204)
    result_mul_388988 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 30), '*', s_388984, result_add_388987)
    
    # Processing the call keyword arguments (line 204)
    kwargs_388989 = {}
    # Getting the type of 'float' (line 204)
    float_388983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 24), 'float', False)
    # Calling float(args, kwargs) (line 204)
    float_call_result_388990 = invoke(stypy.reporting.localization.Localization(__file__, 204, 24), float_388983, *[result_mul_388988], **kwargs_388989)
    
    # Applying the binary operator 'div' (line 204)
    result_div_388991 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 20), 'div', t_388982, float_call_result_388990)
    
    # Assigning a type to the variable 'coeff' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'coeff', result_div_388991)
    
    # Assigning a BinOp to a Name (line 205):
    
    # Assigning a BinOp to a Name (line 205):
    # Getting the type of 'coeff' (line 205)
    coeff_388992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'coeff')
    
    # Call to dot(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'B' (line 205)
    B_388995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), 'B', False)
    # Processing the call keyword arguments (line 205)
    kwargs_388996 = {}
    # Getting the type of 'A' (line 205)
    A_388993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 'A', False)
    # Obtaining the member 'dot' of a type (line 205)
    dot_388994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 24), A_388993, 'dot')
    # Calling dot(args, kwargs) (line 205)
    dot_call_result_388997 = invoke(stypy.reporting.localization.Localization(__file__, 205, 24), dot_388994, *[B_388995], **kwargs_388996)
    
    # Applying the binary operator '*' (line 205)
    result_mul_388998 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 16), '*', coeff_388992, dot_call_result_388997)
    
    # Assigning a type to the variable 'B' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'B', result_mul_388998)
    
    # Assigning a Call to a Name (line 206):
    
    # Assigning a Call to a Name (line 206):
    
    # Call to _exact_inf_norm(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'B' (line 206)
    B_389000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 33), 'B', False)
    # Processing the call keyword arguments (line 206)
    kwargs_389001 = {}
    # Getting the type of '_exact_inf_norm' (line 206)
    _exact_inf_norm_388999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), '_exact_inf_norm', False)
    # Calling _exact_inf_norm(args, kwargs) (line 206)
    _exact_inf_norm_call_result_389002 = invoke(stypy.reporting.localization.Localization(__file__, 206, 17), _exact_inf_norm_388999, *[B_389000], **kwargs_389001)
    
    # Assigning a type to the variable 'c2' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'c2', _exact_inf_norm_call_result_389002)
    
    # Assigning a BinOp to a Name (line 207):
    
    # Assigning a BinOp to a Name (line 207):
    # Getting the type of 'F' (line 207)
    F_389003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'F')
    # Getting the type of 'B' (line 207)
    B_389004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 20), 'B')
    # Applying the binary operator '+' (line 207)
    result_add_389005 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 16), '+', F_389003, B_389004)
    
    # Assigning a type to the variable 'F' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'F', result_add_389005)
    
    
    # Getting the type of 'c1' (line 208)
    c1_389006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'c1')
    # Getting the type of 'c2' (line 208)
    c2_389007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'c2')
    # Applying the binary operator '+' (line 208)
    result_add_389008 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 15), '+', c1_389006, c2_389007)
    
    # Getting the type of 'tol' (line 208)
    tol_389009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 26), 'tol')
    
    # Call to _exact_inf_norm(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'F' (line 208)
    F_389011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 48), 'F', False)
    # Processing the call keyword arguments (line 208)
    kwargs_389012 = {}
    # Getting the type of '_exact_inf_norm' (line 208)
    _exact_inf_norm_389010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 32), '_exact_inf_norm', False)
    # Calling _exact_inf_norm(args, kwargs) (line 208)
    _exact_inf_norm_call_result_389013 = invoke(stypy.reporting.localization.Localization(__file__, 208, 32), _exact_inf_norm_389010, *[F_389011], **kwargs_389012)
    
    # Applying the binary operator '*' (line 208)
    result_mul_389014 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 26), '*', tol_389009, _exact_inf_norm_call_result_389013)
    
    # Applying the binary operator '<=' (line 208)
    result_le_389015 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 15), '<=', result_add_389008, result_mul_389014)
    
    # Testing the type of an if condition (line 208)
    if_condition_389016 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 12), result_le_389015)
    # Assigning a type to the variable 'if_condition_389016' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'if_condition_389016', if_condition_389016)
    # SSA begins for if statement (line 208)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 208)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 210):
    
    # Assigning a Name to a Name (line 210):
    # Getting the type of 'c2' (line 210)
    c2_389017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 17), 'c2')
    # Assigning a type to the variable 'c1' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'c1', c2_389017)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 211):
    
    # Assigning a BinOp to a Name (line 211):
    # Getting the type of 'eta' (line 211)
    eta_389018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'eta')
    # Getting the type of 'F' (line 211)
    F_389019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'F')
    # Applying the binary operator '*' (line 211)
    result_mul_389020 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 12), '*', eta_389018, F_389019)
    
    # Assigning a type to the variable 'F' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'F', result_mul_389020)
    
    # Assigning a Name to a Name (line 212):
    
    # Assigning a Name to a Name (line 212):
    # Getting the type of 'F' (line 212)
    F_389021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'F')
    # Assigning a type to the variable 'B' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'B', F_389021)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'F' (line 213)
    F_389022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 'F')
    # Assigning a type to the variable 'stypy_return_type' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type', F_389022)
    
    # ################# End of '_expm_multiply_simple_core(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_expm_multiply_simple_core' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_389023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_389023)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_expm_multiply_simple_core'
    return stypy_return_type_389023

# Assigning a type to the variable '_expm_multiply_simple_core' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), '_expm_multiply_simple_core', _expm_multiply_simple_core)

# Assigning a Dict to a Name (line 218):

# Assigning a Dict to a Name (line 218):

# Obtaining an instance of the builtin type 'dict' (line 218)
dict_389024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 218)
# Adding element type (key, value) (line 218)
int_389025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 8), 'int')
float_389026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 11), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389025, float_389026))
# Adding element type (key, value) (line 218)
int_389027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 8), 'int')
float_389028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 11), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389027, float_389028))
# Adding element type (key, value) (line 218)
int_389029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 8), 'int')
float_389030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 11), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389029, float_389030))
# Adding element type (key, value) (line 218)
int_389031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 8), 'int')
float_389032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 11), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389031, float_389032))
# Adding element type (key, value) (line 218)
int_389033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 8), 'int')
float_389034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 11), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389033, float_389034))
# Adding element type (key, value) (line 218)
int_389035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 8), 'int')
float_389036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 11), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389035, float_389036))
# Adding element type (key, value) (line 218)
int_389037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
float_389038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 11), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389037, float_389038))
# Adding element type (key, value) (line 218)
int_389039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
float_389040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 11), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389039, float_389040))
# Adding element type (key, value) (line 218)
int_389041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 8), 'int')
float_389042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 11), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389041, float_389042))
# Adding element type (key, value) (line 218)
int_389043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 8), 'int')
float_389044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389043, float_389044))
# Adding element type (key, value) (line 218)
int_389045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 8), 'int')
float_389046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389045, float_389046))
# Adding element type (key, value) (line 218)
int_389047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 8), 'int')
float_389048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389047, float_389048))
# Adding element type (key, value) (line 218)
int_389049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 8), 'int')
float_389050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389049, float_389050))
# Adding element type (key, value) (line 218)
int_389051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 8), 'int')
float_389052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389051, float_389052))
# Adding element type (key, value) (line 218)
int_389053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 8), 'int')
float_389054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389053, float_389054))
# Adding element type (key, value) (line 218)
int_389055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 8), 'int')
float_389056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389055, float_389056))
# Adding element type (key, value) (line 218)
int_389057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 8), 'int')
float_389058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389057, float_389058))
# Adding element type (key, value) (line 218)
int_389059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 8), 'int')
float_389060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389059, float_389060))
# Adding element type (key, value) (line 218)
int_389061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 8), 'int')
float_389062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389061, float_389062))
# Adding element type (key, value) (line 218)
int_389063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 8), 'int')
float_389064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389063, float_389064))
# Adding element type (key, value) (line 218)
int_389065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 8), 'int')
float_389066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389065, float_389066))
# Adding element type (key, value) (line 218)
int_389067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 8), 'int')
float_389068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389067, float_389068))
# Adding element type (key, value) (line 218)
int_389069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 8), 'int')
float_389070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389069, float_389070))
# Adding element type (key, value) (line 218)
int_389071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 8), 'int')
float_389072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389071, float_389072))
# Adding element type (key, value) (line 218)
int_389073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 8), 'int')
float_389074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389073, float_389074))
# Adding element type (key, value) (line 218)
int_389075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 8), 'int')
float_389076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389075, float_389076))
# Adding element type (key, value) (line 218)
int_389077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 8), 'int')
float_389078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389077, float_389078))
# Adding element type (key, value) (line 218)
int_389079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 8), 'int')
float_389080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389079, float_389080))
# Adding element type (key, value) (line 218)
int_389081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 8), 'int')
float_389082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389081, float_389082))
# Adding element type (key, value) (line 218)
int_389083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 8), 'int')
float_389084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389083, float_389084))
# Adding element type (key, value) (line 218)
int_389085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 8), 'int')
float_389086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389085, float_389086))
# Adding element type (key, value) (line 218)
int_389087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 8), 'int')
float_389088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389087, float_389088))
# Adding element type (key, value) (line 218)
int_389089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 8), 'int')
float_389090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389089, float_389090))
# Adding element type (key, value) (line 218)
int_389091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 8), 'int')
float_389092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389091, float_389092))
# Adding element type (key, value) (line 218)
int_389093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 8), 'int')
float_389094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 12), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 9), dict_389024, (int_389093, float_389094))

# Assigning a type to the variable '_theta' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), '_theta', dict_389024)

@norecursion
def _onenormest_matrix_power(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_389095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 10), 'int')
    int_389096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 19), 'int')
    # Getting the type of 'False' (line 263)
    False_389097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 32), 'False')
    # Getting the type of 'False' (line 263)
    False_389098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 49), 'False')
    defaults = [int_389095, int_389096, False_389097, False_389098]
    # Create a new context for function '_onenormest_matrix_power'
    module_type_store = module_type_store.open_function_context('_onenormest_matrix_power', 262, 0, False)
    
    # Passed parameters checking function
    _onenormest_matrix_power.stypy_localization = localization
    _onenormest_matrix_power.stypy_type_of_self = None
    _onenormest_matrix_power.stypy_type_store = module_type_store
    _onenormest_matrix_power.stypy_function_name = '_onenormest_matrix_power'
    _onenormest_matrix_power.stypy_param_names_list = ['A', 'p', 't', 'itmax', 'compute_v', 'compute_w']
    _onenormest_matrix_power.stypy_varargs_param_name = None
    _onenormest_matrix_power.stypy_kwargs_param_name = None
    _onenormest_matrix_power.stypy_call_defaults = defaults
    _onenormest_matrix_power.stypy_call_varargs = varargs
    _onenormest_matrix_power.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_onenormest_matrix_power', ['A', 'p', 't', 'itmax', 'compute_v', 'compute_w'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_onenormest_matrix_power', localization, ['A', 'p', 't', 'itmax', 'compute_v', 'compute_w'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_onenormest_matrix_power(...)' code ##################

    str_389099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, (-1)), 'str', '\n    Efficiently estimate the 1-norm of A^p.\n\n    Parameters\n    ----------\n    A : ndarray\n        Matrix whose 1-norm of a power is to be computed.\n    p : int\n        Non-negative integer power.\n    t : int, optional\n        A positive parameter controlling the tradeoff between\n        accuracy versus time and memory usage.\n        Larger values take longer and use more memory\n        but give more accurate output.\n    itmax : int, optional\n        Use at most this many iterations.\n    compute_v : bool, optional\n        Request a norm-maximizing linear operator input vector if True.\n    compute_w : bool, optional\n        Request a norm-maximizing linear operator output vector if True.\n\n    Returns\n    -------\n    est : float\n        An underestimate of the 1-norm of the sparse matrix.\n    v : ndarray, optional\n        The vector such that ||Av||_1 == est*||v||_1.\n        It can be thought of as an input to the linear operator\n        that gives an output with particularly large norm.\n    w : ndarray, optional\n        The vector Av which has relatively large 1-norm.\n        It can be thought of as an output of the linear operator\n        that is relatively large in norm compared to the input.\n\n    ')
    
    # Call to onenormest(...): (line 302)
    # Processing the call arguments (line 302)
    
    # Call to aslinearoperator(...): (line 302)
    # Processing the call arguments (line 302)
    # Getting the type of 'A' (line 302)
    A_389105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 59), 'A', False)
    # Processing the call keyword arguments (line 302)
    kwargs_389106 = {}
    # Getting the type of 'aslinearoperator' (line 302)
    aslinearoperator_389104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 42), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 302)
    aslinearoperator_call_result_389107 = invoke(stypy.reporting.localization.Localization(__file__, 302, 42), aslinearoperator_389104, *[A_389105], **kwargs_389106)
    
    # Getting the type of 'p' (line 302)
    p_389108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 65), 'p', False)
    # Applying the binary operator '**' (line 302)
    result_pow_389109 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 42), '**', aslinearoperator_call_result_389107, p_389108)
    
    # Processing the call keyword arguments (line 302)
    kwargs_389110 = {}
    # Getting the type of 'scipy' (line 302)
    scipy_389100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 11), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 302)
    sparse_389101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 11), scipy_389100, 'sparse')
    # Obtaining the member 'linalg' of a type (line 302)
    linalg_389102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 11), sparse_389101, 'linalg')
    # Obtaining the member 'onenormest' of a type (line 302)
    onenormest_389103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 11), linalg_389102, 'onenormest')
    # Calling onenormest(args, kwargs) (line 302)
    onenormest_call_result_389111 = invoke(stypy.reporting.localization.Localization(__file__, 302, 11), onenormest_389103, *[result_pow_389109], **kwargs_389110)
    
    # Assigning a type to the variable 'stypy_return_type' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'stypy_return_type', onenormest_call_result_389111)
    
    # ################# End of '_onenormest_matrix_power(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_onenormest_matrix_power' in the type store
    # Getting the type of 'stypy_return_type' (line 262)
    stypy_return_type_389112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_389112)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_onenormest_matrix_power'
    return stypy_return_type_389112

# Assigning a type to the variable '_onenormest_matrix_power' (line 262)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), '_onenormest_matrix_power', _onenormest_matrix_power)
# Declaration of the 'LazyOperatorNormInfo' class

class LazyOperatorNormInfo:
    str_389113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, (-1)), 'str', '\n    Information about an operator is lazily computed.\n\n    The information includes the exact 1-norm of the operator,\n    in addition to estimates of 1-norms of powers of the operator.\n    This uses the notation of Computing the Action (2011).\n    This class is specialized enough to probably not be of general interest\n    outside of this module.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 315)
        None_389114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 35), 'None')
        int_389115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 45), 'int')
        int_389116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 54), 'int')
        defaults = [None_389114, int_389115, int_389116]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 315, 4, False)
        # Assigning a type to the variable 'self' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LazyOperatorNormInfo.__init__', ['A', 'A_1_norm', 'ell', 'scale'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['A', 'A_1_norm', 'ell', 'scale'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_389117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, (-1)), 'str', '\n        Provide the operator and some norm-related information.\n\n        Parameters\n        ----------\n        A : linear operator\n            The operator of interest.\n        A_1_norm : float, optional\n            The exact 1-norm of A.\n        ell : int, optional\n            A technical parameter controlling norm estimation quality.\n        scale : int, optional\n            If specified, return the norms of scale*A instead of A.\n\n        ')
        
        # Assigning a Name to a Attribute (line 331):
        
        # Assigning a Name to a Attribute (line 331):
        # Getting the type of 'A' (line 331)
        A_389118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 18), 'A')
        # Getting the type of 'self' (line 331)
        self_389119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'self')
        # Setting the type of the member '_A' of a type (line 331)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), self_389119, '_A', A_389118)
        
        # Assigning a Name to a Attribute (line 332):
        
        # Assigning a Name to a Attribute (line 332):
        # Getting the type of 'A_1_norm' (line 332)
        A_1_norm_389120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 25), 'A_1_norm')
        # Getting the type of 'self' (line 332)
        self_389121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'self')
        # Setting the type of the member '_A_1_norm' of a type (line 332)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), self_389121, '_A_1_norm', A_1_norm_389120)
        
        # Assigning a Name to a Attribute (line 333):
        
        # Assigning a Name to a Attribute (line 333):
        # Getting the type of 'ell' (line 333)
        ell_389122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 20), 'ell')
        # Getting the type of 'self' (line 333)
        self_389123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'self')
        # Setting the type of the member '_ell' of a type (line 333)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), self_389123, '_ell', ell_389122)
        
        # Assigning a Dict to a Attribute (line 334):
        
        # Assigning a Dict to a Attribute (line 334):
        
        # Obtaining an instance of the builtin type 'dict' (line 334)
        dict_389124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 334)
        
        # Getting the type of 'self' (line 334)
        self_389125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'self')
        # Setting the type of the member '_d' of a type (line 334)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), self_389125, '_d', dict_389124)
        
        # Assigning a Name to a Attribute (line 335):
        
        # Assigning a Name to a Attribute (line 335):
        # Getting the type of 'scale' (line 335)
        scale_389126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 22), 'scale')
        # Getting the type of 'self' (line 335)
        self_389127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'self')
        # Setting the type of the member '_scale' of a type (line 335)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), self_389127, '_scale', scale_389126)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_scale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_scale'
        module_type_store = module_type_store.open_function_context('set_scale', 337, 4, False)
        # Assigning a type to the variable 'self' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LazyOperatorNormInfo.set_scale.__dict__.__setitem__('stypy_localization', localization)
        LazyOperatorNormInfo.set_scale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LazyOperatorNormInfo.set_scale.__dict__.__setitem__('stypy_type_store', module_type_store)
        LazyOperatorNormInfo.set_scale.__dict__.__setitem__('stypy_function_name', 'LazyOperatorNormInfo.set_scale')
        LazyOperatorNormInfo.set_scale.__dict__.__setitem__('stypy_param_names_list', ['scale'])
        LazyOperatorNormInfo.set_scale.__dict__.__setitem__('stypy_varargs_param_name', None)
        LazyOperatorNormInfo.set_scale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LazyOperatorNormInfo.set_scale.__dict__.__setitem__('stypy_call_defaults', defaults)
        LazyOperatorNormInfo.set_scale.__dict__.__setitem__('stypy_call_varargs', varargs)
        LazyOperatorNormInfo.set_scale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LazyOperatorNormInfo.set_scale.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LazyOperatorNormInfo.set_scale', ['scale'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_scale', localization, ['scale'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_scale(...)' code ##################

        str_389128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, (-1)), 'str', '\n        Set the scale parameter.\n        ')
        
        # Assigning a Name to a Attribute (line 341):
        
        # Assigning a Name to a Attribute (line 341):
        # Getting the type of 'scale' (line 341)
        scale_389129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 22), 'scale')
        # Getting the type of 'self' (line 341)
        self_389130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'self')
        # Setting the type of the member '_scale' of a type (line 341)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), self_389130, '_scale', scale_389129)
        
        # ################# End of 'set_scale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_scale' in the type store
        # Getting the type of 'stypy_return_type' (line 337)
        stypy_return_type_389131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_389131)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_scale'
        return stypy_return_type_389131


    @norecursion
    def onenorm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'onenorm'
        module_type_store = module_type_store.open_function_context('onenorm', 343, 4, False)
        # Assigning a type to the variable 'self' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LazyOperatorNormInfo.onenorm.__dict__.__setitem__('stypy_localization', localization)
        LazyOperatorNormInfo.onenorm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LazyOperatorNormInfo.onenorm.__dict__.__setitem__('stypy_type_store', module_type_store)
        LazyOperatorNormInfo.onenorm.__dict__.__setitem__('stypy_function_name', 'LazyOperatorNormInfo.onenorm')
        LazyOperatorNormInfo.onenorm.__dict__.__setitem__('stypy_param_names_list', [])
        LazyOperatorNormInfo.onenorm.__dict__.__setitem__('stypy_varargs_param_name', None)
        LazyOperatorNormInfo.onenorm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LazyOperatorNormInfo.onenorm.__dict__.__setitem__('stypy_call_defaults', defaults)
        LazyOperatorNormInfo.onenorm.__dict__.__setitem__('stypy_call_varargs', varargs)
        LazyOperatorNormInfo.onenorm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LazyOperatorNormInfo.onenorm.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LazyOperatorNormInfo.onenorm', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'onenorm', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'onenorm(...)' code ##################

        str_389132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, (-1)), 'str', '\n        Compute the exact 1-norm.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 347)
        # Getting the type of 'self' (line 347)
        self_389133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 11), 'self')
        # Obtaining the member '_A_1_norm' of a type (line 347)
        _A_1_norm_389134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 11), self_389133, '_A_1_norm')
        # Getting the type of 'None' (line 347)
        None_389135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 29), 'None')
        
        (may_be_389136, more_types_in_union_389137) = may_be_none(_A_1_norm_389134, None_389135)

        if may_be_389136:

            if more_types_in_union_389137:
                # Runtime conditional SSA (line 347)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 348):
            
            # Assigning a Call to a Attribute (line 348):
            
            # Call to _exact_1_norm(...): (line 348)
            # Processing the call arguments (line 348)
            # Getting the type of 'self' (line 348)
            self_389139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 43), 'self', False)
            # Obtaining the member '_A' of a type (line 348)
            _A_389140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 43), self_389139, '_A')
            # Processing the call keyword arguments (line 348)
            kwargs_389141 = {}
            # Getting the type of '_exact_1_norm' (line 348)
            _exact_1_norm_389138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 29), '_exact_1_norm', False)
            # Calling _exact_1_norm(args, kwargs) (line 348)
            _exact_1_norm_call_result_389142 = invoke(stypy.reporting.localization.Localization(__file__, 348, 29), _exact_1_norm_389138, *[_A_389140], **kwargs_389141)
            
            # Getting the type of 'self' (line 348)
            self_389143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'self')
            # Setting the type of the member '_A_1_norm' of a type (line 348)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 12), self_389143, '_A_1_norm', _exact_1_norm_call_result_389142)

            if more_types_in_union_389137:
                # SSA join for if statement (line 347)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 349)
        self_389144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'self')
        # Obtaining the member '_scale' of a type (line 349)
        _scale_389145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 15), self_389144, '_scale')
        # Getting the type of 'self' (line 349)
        self_389146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 27), 'self')
        # Obtaining the member '_A_1_norm' of a type (line 349)
        _A_1_norm_389147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 27), self_389146, '_A_1_norm')
        # Applying the binary operator '*' (line 349)
        result_mul_389148 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 15), '*', _scale_389145, _A_1_norm_389147)
        
        # Assigning a type to the variable 'stypy_return_type' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'stypy_return_type', result_mul_389148)
        
        # ################# End of 'onenorm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'onenorm' in the type store
        # Getting the type of 'stypy_return_type' (line 343)
        stypy_return_type_389149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_389149)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'onenorm'
        return stypy_return_type_389149


    @norecursion
    def d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'd'
        module_type_store = module_type_store.open_function_context('d', 351, 4, False)
        # Assigning a type to the variable 'self' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LazyOperatorNormInfo.d.__dict__.__setitem__('stypy_localization', localization)
        LazyOperatorNormInfo.d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LazyOperatorNormInfo.d.__dict__.__setitem__('stypy_type_store', module_type_store)
        LazyOperatorNormInfo.d.__dict__.__setitem__('stypy_function_name', 'LazyOperatorNormInfo.d')
        LazyOperatorNormInfo.d.__dict__.__setitem__('stypy_param_names_list', ['p'])
        LazyOperatorNormInfo.d.__dict__.__setitem__('stypy_varargs_param_name', None)
        LazyOperatorNormInfo.d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LazyOperatorNormInfo.d.__dict__.__setitem__('stypy_call_defaults', defaults)
        LazyOperatorNormInfo.d.__dict__.__setitem__('stypy_call_varargs', varargs)
        LazyOperatorNormInfo.d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LazyOperatorNormInfo.d.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LazyOperatorNormInfo.d', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'd', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'd(...)' code ##################

        str_389150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, (-1)), 'str', '\n        Lazily estimate d_p(A) ~= || A^p ||^(1/p) where ||.|| is the 1-norm.\n        ')
        
        
        # Getting the type of 'p' (line 355)
        p_389151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 11), 'p')
        # Getting the type of 'self' (line 355)
        self_389152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 20), 'self')
        # Obtaining the member '_d' of a type (line 355)
        _d_389153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 20), self_389152, '_d')
        # Applying the binary operator 'notin' (line 355)
        result_contains_389154 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 11), 'notin', p_389151, _d_389153)
        
        # Testing the type of an if condition (line 355)
        if_condition_389155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 355, 8), result_contains_389154)
        # Assigning a type to the variable 'if_condition_389155' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'if_condition_389155', if_condition_389155)
        # SSA begins for if statement (line 355)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 356):
        
        # Assigning a Call to a Name (line 356):
        
        # Call to _onenormest_matrix_power(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'self' (line 356)
        self_389157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 43), 'self', False)
        # Obtaining the member '_A' of a type (line 356)
        _A_389158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 43), self_389157, '_A')
        # Getting the type of 'p' (line 356)
        p_389159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 52), 'p', False)
        # Getting the type of 'self' (line 356)
        self_389160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 55), 'self', False)
        # Obtaining the member '_ell' of a type (line 356)
        _ell_389161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 55), self_389160, '_ell')
        # Processing the call keyword arguments (line 356)
        kwargs_389162 = {}
        # Getting the type of '_onenormest_matrix_power' (line 356)
        _onenormest_matrix_power_389156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 18), '_onenormest_matrix_power', False)
        # Calling _onenormest_matrix_power(args, kwargs) (line 356)
        _onenormest_matrix_power_call_result_389163 = invoke(stypy.reporting.localization.Localization(__file__, 356, 18), _onenormest_matrix_power_389156, *[_A_389158, p_389159, _ell_389161], **kwargs_389162)
        
        # Assigning a type to the variable 'est' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'est', _onenormest_matrix_power_call_result_389163)
        
        # Assigning a BinOp to a Subscript (line 357):
        
        # Assigning a BinOp to a Subscript (line 357):
        # Getting the type of 'est' (line 357)
        est_389164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 25), 'est')
        float_389165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 33), 'float')
        # Getting the type of 'p' (line 357)
        p_389166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 39), 'p')
        # Applying the binary operator 'div' (line 357)
        result_div_389167 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 33), 'div', float_389165, p_389166)
        
        # Applying the binary operator '**' (line 357)
        result_pow_389168 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 25), '**', est_389164, result_div_389167)
        
        # Getting the type of 'self' (line 357)
        self_389169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'self')
        # Obtaining the member '_d' of a type (line 357)
        _d_389170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 12), self_389169, '_d')
        # Getting the type of 'p' (line 357)
        p_389171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'p')
        # Storing an element on a container (line 357)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 12), _d_389170, (p_389171, result_pow_389168))
        # SSA join for if statement (line 355)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 358)
        self_389172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 15), 'self')
        # Obtaining the member '_scale' of a type (line 358)
        _scale_389173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 15), self_389172, '_scale')
        
        # Obtaining the type of the subscript
        # Getting the type of 'p' (line 358)
        p_389174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 35), 'p')
        # Getting the type of 'self' (line 358)
        self_389175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 27), 'self')
        # Obtaining the member '_d' of a type (line 358)
        _d_389176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 27), self_389175, '_d')
        # Obtaining the member '__getitem__' of a type (line 358)
        getitem___389177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 27), _d_389176, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 358)
        subscript_call_result_389178 = invoke(stypy.reporting.localization.Localization(__file__, 358, 27), getitem___389177, p_389174)
        
        # Applying the binary operator '*' (line 358)
        result_mul_389179 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 15), '*', _scale_389173, subscript_call_result_389178)
        
        # Assigning a type to the variable 'stypy_return_type' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'stypy_return_type', result_mul_389179)
        
        # ################# End of 'd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'd' in the type store
        # Getting the type of 'stypy_return_type' (line 351)
        stypy_return_type_389180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_389180)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'd'
        return stypy_return_type_389180


    @norecursion
    def alpha(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'alpha'
        module_type_store = module_type_store.open_function_context('alpha', 360, 4, False)
        # Assigning a type to the variable 'self' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LazyOperatorNormInfo.alpha.__dict__.__setitem__('stypy_localization', localization)
        LazyOperatorNormInfo.alpha.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LazyOperatorNormInfo.alpha.__dict__.__setitem__('stypy_type_store', module_type_store)
        LazyOperatorNormInfo.alpha.__dict__.__setitem__('stypy_function_name', 'LazyOperatorNormInfo.alpha')
        LazyOperatorNormInfo.alpha.__dict__.__setitem__('stypy_param_names_list', ['p'])
        LazyOperatorNormInfo.alpha.__dict__.__setitem__('stypy_varargs_param_name', None)
        LazyOperatorNormInfo.alpha.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LazyOperatorNormInfo.alpha.__dict__.__setitem__('stypy_call_defaults', defaults)
        LazyOperatorNormInfo.alpha.__dict__.__setitem__('stypy_call_varargs', varargs)
        LazyOperatorNormInfo.alpha.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LazyOperatorNormInfo.alpha.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LazyOperatorNormInfo.alpha', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'alpha', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'alpha(...)' code ##################

        str_389181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, (-1)), 'str', '\n        Lazily compute max(d(p), d(p+1)).\n        ')
        
        # Call to max(...): (line 364)
        # Processing the call arguments (line 364)
        
        # Call to d(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'p' (line 364)
        p_389185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 26), 'p', False)
        # Processing the call keyword arguments (line 364)
        kwargs_389186 = {}
        # Getting the type of 'self' (line 364)
        self_389183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 19), 'self', False)
        # Obtaining the member 'd' of a type (line 364)
        d_389184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 19), self_389183, 'd')
        # Calling d(args, kwargs) (line 364)
        d_call_result_389187 = invoke(stypy.reporting.localization.Localization(__file__, 364, 19), d_389184, *[p_389185], **kwargs_389186)
        
        
        # Call to d(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'p' (line 364)
        p_389190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 37), 'p', False)
        int_389191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 39), 'int')
        # Applying the binary operator '+' (line 364)
        result_add_389192 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 37), '+', p_389190, int_389191)
        
        # Processing the call keyword arguments (line 364)
        kwargs_389193 = {}
        # Getting the type of 'self' (line 364)
        self_389188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 30), 'self', False)
        # Obtaining the member 'd' of a type (line 364)
        d_389189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 30), self_389188, 'd')
        # Calling d(args, kwargs) (line 364)
        d_call_result_389194 = invoke(stypy.reporting.localization.Localization(__file__, 364, 30), d_389189, *[result_add_389192], **kwargs_389193)
        
        # Processing the call keyword arguments (line 364)
        kwargs_389195 = {}
        # Getting the type of 'max' (line 364)
        max_389182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 15), 'max', False)
        # Calling max(args, kwargs) (line 364)
        max_call_result_389196 = invoke(stypy.reporting.localization.Localization(__file__, 364, 15), max_389182, *[d_call_result_389187, d_call_result_389194], **kwargs_389195)
        
        # Assigning a type to the variable 'stypy_return_type' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'stypy_return_type', max_call_result_389196)
        
        # ################# End of 'alpha(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'alpha' in the type store
        # Getting the type of 'stypy_return_type' (line 360)
        stypy_return_type_389197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_389197)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'alpha'
        return stypy_return_type_389197


# Assigning a type to the variable 'LazyOperatorNormInfo' (line 304)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'LazyOperatorNormInfo', LazyOperatorNormInfo)

@norecursion
def _compute_cost_div_m(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_compute_cost_div_m'
    module_type_store = module_type_store.open_function_context('_compute_cost_div_m', 366, 0, False)
    
    # Passed parameters checking function
    _compute_cost_div_m.stypy_localization = localization
    _compute_cost_div_m.stypy_type_of_self = None
    _compute_cost_div_m.stypy_type_store = module_type_store
    _compute_cost_div_m.stypy_function_name = '_compute_cost_div_m'
    _compute_cost_div_m.stypy_param_names_list = ['m', 'p', 'norm_info']
    _compute_cost_div_m.stypy_varargs_param_name = None
    _compute_cost_div_m.stypy_kwargs_param_name = None
    _compute_cost_div_m.stypy_call_defaults = defaults
    _compute_cost_div_m.stypy_call_varargs = varargs
    _compute_cost_div_m.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_compute_cost_div_m', ['m', 'p', 'norm_info'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_compute_cost_div_m', localization, ['m', 'p', 'norm_info'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_compute_cost_div_m(...)' code ##################

    str_389198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, (-1)), 'str', '\n    A helper function for computing bounds.\n\n    This is equation (3.10).\n    It measures cost in terms of the number of required matrix products.\n\n    Parameters\n    ----------\n    m : int\n        A valid key of _theta.\n    p : int\n        A matrix power.\n    norm_info : LazyOperatorNormInfo\n        Information about 1-norms of related operators.\n\n    Returns\n    -------\n    cost_div_m : int\n        Required number of matrix products divided by m.\n\n    ')
    
    # Call to int(...): (line 388)
    # Processing the call arguments (line 388)
    
    # Call to ceil(...): (line 388)
    # Processing the call arguments (line 388)
    
    # Call to alpha(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'p' (line 388)
    p_389204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 39), 'p', False)
    # Processing the call keyword arguments (line 388)
    kwargs_389205 = {}
    # Getting the type of 'norm_info' (line 388)
    norm_info_389202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 23), 'norm_info', False)
    # Obtaining the member 'alpha' of a type (line 388)
    alpha_389203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 23), norm_info_389202, 'alpha')
    # Calling alpha(args, kwargs) (line 388)
    alpha_call_result_389206 = invoke(stypy.reporting.localization.Localization(__file__, 388, 23), alpha_389203, *[p_389204], **kwargs_389205)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 388)
    m_389207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 51), 'm', False)
    # Getting the type of '_theta' (line 388)
    _theta_389208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 44), '_theta', False)
    # Obtaining the member '__getitem__' of a type (line 388)
    getitem___389209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 44), _theta_389208, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 388)
    subscript_call_result_389210 = invoke(stypy.reporting.localization.Localization(__file__, 388, 44), getitem___389209, m_389207)
    
    # Applying the binary operator 'div' (line 388)
    result_div_389211 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 23), 'div', alpha_call_result_389206, subscript_call_result_389210)
    
    # Processing the call keyword arguments (line 388)
    kwargs_389212 = {}
    # Getting the type of 'np' (line 388)
    np_389200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'np', False)
    # Obtaining the member 'ceil' of a type (line 388)
    ceil_389201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 15), np_389200, 'ceil')
    # Calling ceil(args, kwargs) (line 388)
    ceil_call_result_389213 = invoke(stypy.reporting.localization.Localization(__file__, 388, 15), ceil_389201, *[result_div_389211], **kwargs_389212)
    
    # Processing the call keyword arguments (line 388)
    kwargs_389214 = {}
    # Getting the type of 'int' (line 388)
    int_389199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 11), 'int', False)
    # Calling int(args, kwargs) (line 388)
    int_call_result_389215 = invoke(stypy.reporting.localization.Localization(__file__, 388, 11), int_389199, *[ceil_call_result_389213], **kwargs_389214)
    
    # Assigning a type to the variable 'stypy_return_type' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'stypy_return_type', int_call_result_389215)
    
    # ################# End of '_compute_cost_div_m(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_compute_cost_div_m' in the type store
    # Getting the type of 'stypy_return_type' (line 366)
    stypy_return_type_389216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_389216)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_compute_cost_div_m'
    return stypy_return_type_389216

# Assigning a type to the variable '_compute_cost_div_m' (line 366)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 0), '_compute_cost_div_m', _compute_cost_div_m)

@norecursion
def _compute_p_max(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_compute_p_max'
    module_type_store = module_type_store.open_function_context('_compute_p_max', 391, 0, False)
    
    # Passed parameters checking function
    _compute_p_max.stypy_localization = localization
    _compute_p_max.stypy_type_of_self = None
    _compute_p_max.stypy_type_store = module_type_store
    _compute_p_max.stypy_function_name = '_compute_p_max'
    _compute_p_max.stypy_param_names_list = ['m_max']
    _compute_p_max.stypy_varargs_param_name = None
    _compute_p_max.stypy_kwargs_param_name = None
    _compute_p_max.stypy_call_defaults = defaults
    _compute_p_max.stypy_call_varargs = varargs
    _compute_p_max.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_compute_p_max', ['m_max'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_compute_p_max', localization, ['m_max'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_compute_p_max(...)' code ##################

    str_389217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, (-1)), 'str', '\n    Compute the largest positive integer p such that p*(p-1) <= m_max + 1.\n\n    Do this in a slightly dumb way, but safe and not too slow.\n\n    Parameters\n    ----------\n    m_max : int\n        A count related to bounds.\n\n    ')
    
    # Assigning a Call to a Name (line 403):
    
    # Assigning a Call to a Name (line 403):
    
    # Call to sqrt(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 'm_max' (line 403)
    m_max_389220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 25), 'm_max', False)
    # Processing the call keyword arguments (line 403)
    kwargs_389221 = {}
    # Getting the type of 'np' (line 403)
    np_389218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 17), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 403)
    sqrt_389219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 17), np_389218, 'sqrt')
    # Calling sqrt(args, kwargs) (line 403)
    sqrt_call_result_389222 = invoke(stypy.reporting.localization.Localization(__file__, 403, 17), sqrt_389219, *[m_max_389220], **kwargs_389221)
    
    # Assigning a type to the variable 'sqrt_m_max' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'sqrt_m_max', sqrt_call_result_389222)
    
    # Assigning a Call to a Name (line 404):
    
    # Assigning a Call to a Name (line 404):
    
    # Call to int(...): (line 404)
    # Processing the call arguments (line 404)
    
    # Call to floor(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'sqrt_m_max' (line 404)
    sqrt_m_max_389226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 25), 'sqrt_m_max', False)
    # Processing the call keyword arguments (line 404)
    kwargs_389227 = {}
    # Getting the type of 'np' (line 404)
    np_389224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 16), 'np', False)
    # Obtaining the member 'floor' of a type (line 404)
    floor_389225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 16), np_389224, 'floor')
    # Calling floor(args, kwargs) (line 404)
    floor_call_result_389228 = invoke(stypy.reporting.localization.Localization(__file__, 404, 16), floor_389225, *[sqrt_m_max_389226], **kwargs_389227)
    
    # Processing the call keyword arguments (line 404)
    kwargs_389229 = {}
    # Getting the type of 'int' (line 404)
    int_389223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'int', False)
    # Calling int(args, kwargs) (line 404)
    int_call_result_389230 = invoke(stypy.reporting.localization.Localization(__file__, 404, 12), int_389223, *[floor_call_result_389228], **kwargs_389229)
    
    # Assigning a type to the variable 'p_low' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'p_low', int_call_result_389230)
    
    # Assigning a Call to a Name (line 405):
    
    # Assigning a Call to a Name (line 405):
    
    # Call to int(...): (line 405)
    # Processing the call arguments (line 405)
    
    # Call to ceil(...): (line 405)
    # Processing the call arguments (line 405)
    # Getting the type of 'sqrt_m_max' (line 405)
    sqrt_m_max_389234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 25), 'sqrt_m_max', False)
    int_389235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 38), 'int')
    # Applying the binary operator '+' (line 405)
    result_add_389236 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 25), '+', sqrt_m_max_389234, int_389235)
    
    # Processing the call keyword arguments (line 405)
    kwargs_389237 = {}
    # Getting the type of 'np' (line 405)
    np_389232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 17), 'np', False)
    # Obtaining the member 'ceil' of a type (line 405)
    ceil_389233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 17), np_389232, 'ceil')
    # Calling ceil(args, kwargs) (line 405)
    ceil_call_result_389238 = invoke(stypy.reporting.localization.Localization(__file__, 405, 17), ceil_389233, *[result_add_389236], **kwargs_389237)
    
    # Processing the call keyword arguments (line 405)
    kwargs_389239 = {}
    # Getting the type of 'int' (line 405)
    int_389231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 13), 'int', False)
    # Calling int(args, kwargs) (line 405)
    int_call_result_389240 = invoke(stypy.reporting.localization.Localization(__file__, 405, 13), int_389231, *[ceil_call_result_389238], **kwargs_389239)
    
    # Assigning a type to the variable 'p_high' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'p_high', int_call_result_389240)
    
    # Call to max(...): (line 406)
    # Processing the call arguments (line 406)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 406, 15, True)
    # Calculating comprehension expression
    
    # Call to range(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'p_low' (line 406)
    p_low_389253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 32), 'p_low', False)
    # Getting the type of 'p_high' (line 406)
    p_high_389254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 39), 'p_high', False)
    int_389255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 46), 'int')
    # Applying the binary operator '+' (line 406)
    result_add_389256 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 39), '+', p_high_389254, int_389255)
    
    # Processing the call keyword arguments (line 406)
    kwargs_389257 = {}
    # Getting the type of 'range' (line 406)
    range_389252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 26), 'range', False)
    # Calling range(args, kwargs) (line 406)
    range_call_result_389258 = invoke(stypy.reporting.localization.Localization(__file__, 406, 26), range_389252, *[p_low_389253, result_add_389256], **kwargs_389257)
    
    comprehension_389259 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 15), range_call_result_389258)
    # Assigning a type to the variable 'p' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 15), 'p', comprehension_389259)
    
    # Getting the type of 'p' (line 406)
    p_389243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 52), 'p', False)
    # Getting the type of 'p' (line 406)
    p_389244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 55), 'p', False)
    int_389245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 57), 'int')
    # Applying the binary operator '-' (line 406)
    result_sub_389246 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 55), '-', p_389244, int_389245)
    
    # Applying the binary operator '*' (line 406)
    result_mul_389247 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 52), '*', p_389243, result_sub_389246)
    
    # Getting the type of 'm_max' (line 406)
    m_max_389248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 63), 'm_max', False)
    int_389249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 71), 'int')
    # Applying the binary operator '+' (line 406)
    result_add_389250 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 63), '+', m_max_389248, int_389249)
    
    # Applying the binary operator '<=' (line 406)
    result_le_389251 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 52), '<=', result_mul_389247, result_add_389250)
    
    # Getting the type of 'p' (line 406)
    p_389242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 15), 'p', False)
    list_389260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 15), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 15), list_389260, p_389242)
    # Processing the call keyword arguments (line 406)
    kwargs_389261 = {}
    # Getting the type of 'max' (line 406)
    max_389241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 11), 'max', False)
    # Calling max(args, kwargs) (line 406)
    max_call_result_389262 = invoke(stypy.reporting.localization.Localization(__file__, 406, 11), max_389241, *[list_389260], **kwargs_389261)
    
    # Assigning a type to the variable 'stypy_return_type' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'stypy_return_type', max_call_result_389262)
    
    # ################# End of '_compute_p_max(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_compute_p_max' in the type store
    # Getting the type of 'stypy_return_type' (line 391)
    stypy_return_type_389263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_389263)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_compute_p_max'
    return stypy_return_type_389263

# Assigning a type to the variable '_compute_p_max' (line 391)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 0), '_compute_p_max', _compute_p_max)

@norecursion
def _fragment_3_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_389264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 44), 'int')
    int_389265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 52), 'int')
    defaults = [int_389264, int_389265]
    # Create a new context for function '_fragment_3_1'
    module_type_store = module_type_store.open_function_context('_fragment_3_1', 409, 0, False)
    
    # Passed parameters checking function
    _fragment_3_1.stypy_localization = localization
    _fragment_3_1.stypy_type_of_self = None
    _fragment_3_1.stypy_type_store = module_type_store
    _fragment_3_1.stypy_function_name = '_fragment_3_1'
    _fragment_3_1.stypy_param_names_list = ['norm_info', 'n0', 'tol', 'm_max', 'ell']
    _fragment_3_1.stypy_varargs_param_name = None
    _fragment_3_1.stypy_kwargs_param_name = None
    _fragment_3_1.stypy_call_defaults = defaults
    _fragment_3_1.stypy_call_varargs = varargs
    _fragment_3_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fragment_3_1', ['norm_info', 'n0', 'tol', 'm_max', 'ell'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fragment_3_1', localization, ['norm_info', 'n0', 'tol', 'm_max', 'ell'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fragment_3_1(...)' code ##################

    str_389266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, (-1)), 'str', '\n    A helper function for the _expm_multiply_* functions.\n\n    Parameters\n    ----------\n    norm_info : LazyOperatorNormInfo\n        Information about norms of certain linear operators of interest.\n    n0 : int\n        Number of columns in the _expm_multiply_* B matrix.\n    tol : float\n        Expected to be\n        :math:`2^{-24}` for single precision or\n        :math:`2^{-53}` for double precision.\n    m_max : int\n        A value related to a bound.\n    ell : int\n        The number of columns used in the 1-norm approximation.\n        This is usually taken to be small, maybe between 1 and 5.\n\n    Returns\n    -------\n    best_m : int\n        Related to bounds for error control.\n    best_s : int\n        Amount of scaling.\n\n    Notes\n    -----\n    This is code fragment (3.1) in Al-Mohy and Higham (2011).\n    The discussion of default values for m_max and ell\n    is given between the definitions of equation (3.11)\n    and the definition of equation (3.12).\n\n    ')
    
    
    # Getting the type of 'ell' (line 444)
    ell_389267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 7), 'ell')
    int_389268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 13), 'int')
    # Applying the binary operator '<' (line 444)
    result_lt_389269 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 7), '<', ell_389267, int_389268)
    
    # Testing the type of an if condition (line 444)
    if_condition_389270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 444, 4), result_lt_389269)
    # Assigning a type to the variable 'if_condition_389270' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'if_condition_389270', if_condition_389270)
    # SSA begins for if statement (line 444)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 445)
    # Processing the call arguments (line 445)
    str_389272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 25), 'str', 'expected ell to be a positive integer')
    # Processing the call keyword arguments (line 445)
    kwargs_389273 = {}
    # Getting the type of 'ValueError' (line 445)
    ValueError_389271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 445)
    ValueError_call_result_389274 = invoke(stypy.reporting.localization.Localization(__file__, 445, 14), ValueError_389271, *[str_389272], **kwargs_389273)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 445, 8), ValueError_call_result_389274, 'raise parameter', BaseException)
    # SSA join for if statement (line 444)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 446):
    
    # Assigning a Name to a Name (line 446):
    # Getting the type of 'None' (line 446)
    None_389275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 13), 'None')
    # Assigning a type to the variable 'best_m' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'best_m', None_389275)
    
    # Assigning a Name to a Name (line 447):
    
    # Assigning a Name to a Name (line 447):
    # Getting the type of 'None' (line 447)
    None_389276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 13), 'None')
    # Assigning a type to the variable 'best_s' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'best_s', None_389276)
    
    
    # Call to _condition_3_13(...): (line 448)
    # Processing the call arguments (line 448)
    
    # Call to onenorm(...): (line 448)
    # Processing the call keyword arguments (line 448)
    kwargs_389280 = {}
    # Getting the type of 'norm_info' (line 448)
    norm_info_389278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 23), 'norm_info', False)
    # Obtaining the member 'onenorm' of a type (line 448)
    onenorm_389279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 23), norm_info_389278, 'onenorm')
    # Calling onenorm(args, kwargs) (line 448)
    onenorm_call_result_389281 = invoke(stypy.reporting.localization.Localization(__file__, 448, 23), onenorm_389279, *[], **kwargs_389280)
    
    # Getting the type of 'n0' (line 448)
    n0_389282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 44), 'n0', False)
    # Getting the type of 'm_max' (line 448)
    m_max_389283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 48), 'm_max', False)
    # Getting the type of 'ell' (line 448)
    ell_389284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 55), 'ell', False)
    # Processing the call keyword arguments (line 448)
    kwargs_389285 = {}
    # Getting the type of '_condition_3_13' (line 448)
    _condition_3_13_389277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 7), '_condition_3_13', False)
    # Calling _condition_3_13(args, kwargs) (line 448)
    _condition_3_13_call_result_389286 = invoke(stypy.reporting.localization.Localization(__file__, 448, 7), _condition_3_13_389277, *[onenorm_call_result_389281, n0_389282, m_max_389283, ell_389284], **kwargs_389285)
    
    # Testing the type of an if condition (line 448)
    if_condition_389287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 448, 4), _condition_3_13_call_result_389286)
    # Assigning a type to the variable 'if_condition_389287' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'if_condition_389287', if_condition_389287)
    # SSA begins for if statement (line 448)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to items(...): (line 449)
    # Processing the call keyword arguments (line 449)
    kwargs_389290 = {}
    # Getting the type of '_theta' (line 449)
    _theta_389288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 24), '_theta', False)
    # Obtaining the member 'items' of a type (line 449)
    items_389289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 24), _theta_389288, 'items')
    # Calling items(args, kwargs) (line 449)
    items_call_result_389291 = invoke(stypy.reporting.localization.Localization(__file__, 449, 24), items_389289, *[], **kwargs_389290)
    
    # Testing the type of a for loop iterable (line 449)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 449, 8), items_call_result_389291)
    # Getting the type of the for loop variable (line 449)
    for_loop_var_389292 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 449, 8), items_call_result_389291)
    # Assigning a type to the variable 'm' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 8), for_loop_var_389292))
    # Assigning a type to the variable 'theta' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'theta', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 8), for_loop_var_389292))
    # SSA begins for a for statement (line 449)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 450):
    
    # Assigning a Call to a Name (line 450):
    
    # Call to int(...): (line 450)
    # Processing the call arguments (line 450)
    
    # Call to ceil(...): (line 450)
    # Processing the call arguments (line 450)
    
    # Call to onenorm(...): (line 450)
    # Processing the call keyword arguments (line 450)
    kwargs_389298 = {}
    # Getting the type of 'norm_info' (line 450)
    norm_info_389296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 28), 'norm_info', False)
    # Obtaining the member 'onenorm' of a type (line 450)
    onenorm_389297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 28), norm_info_389296, 'onenorm')
    # Calling onenorm(args, kwargs) (line 450)
    onenorm_call_result_389299 = invoke(stypy.reporting.localization.Localization(__file__, 450, 28), onenorm_389297, *[], **kwargs_389298)
    
    # Getting the type of 'theta' (line 450)
    theta_389300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 50), 'theta', False)
    # Applying the binary operator 'div' (line 450)
    result_div_389301 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 28), 'div', onenorm_call_result_389299, theta_389300)
    
    # Processing the call keyword arguments (line 450)
    kwargs_389302 = {}
    # Getting the type of 'np' (line 450)
    np_389294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 20), 'np', False)
    # Obtaining the member 'ceil' of a type (line 450)
    ceil_389295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 20), np_389294, 'ceil')
    # Calling ceil(args, kwargs) (line 450)
    ceil_call_result_389303 = invoke(stypy.reporting.localization.Localization(__file__, 450, 20), ceil_389295, *[result_div_389301], **kwargs_389302)
    
    # Processing the call keyword arguments (line 450)
    kwargs_389304 = {}
    # Getting the type of 'int' (line 450)
    int_389293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 16), 'int', False)
    # Calling int(args, kwargs) (line 450)
    int_call_result_389305 = invoke(stypy.reporting.localization.Localization(__file__, 450, 16), int_389293, *[ceil_call_result_389303], **kwargs_389304)
    
    # Assigning a type to the variable 's' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 's', int_call_result_389305)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'best_m' (line 451)
    best_m_389306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 15), 'best_m')
    # Getting the type of 'None' (line 451)
    None_389307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 25), 'None')
    # Applying the binary operator 'is' (line 451)
    result_is__389308 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 15), 'is', best_m_389306, None_389307)
    
    
    # Getting the type of 'm' (line 451)
    m_389309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 33), 'm')
    # Getting the type of 's' (line 451)
    s_389310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 37), 's')
    # Applying the binary operator '*' (line 451)
    result_mul_389311 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 33), '*', m_389309, s_389310)
    
    # Getting the type of 'best_m' (line 451)
    best_m_389312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 41), 'best_m')
    # Getting the type of 'best_s' (line 451)
    best_s_389313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 50), 'best_s')
    # Applying the binary operator '*' (line 451)
    result_mul_389314 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 41), '*', best_m_389312, best_s_389313)
    
    # Applying the binary operator '<' (line 451)
    result_lt_389315 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 33), '<', result_mul_389311, result_mul_389314)
    
    # Applying the binary operator 'or' (line 451)
    result_or_keyword_389316 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 15), 'or', result_is__389308, result_lt_389315)
    
    # Testing the type of an if condition (line 451)
    if_condition_389317 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 451, 12), result_or_keyword_389316)
    # Assigning a type to the variable 'if_condition_389317' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'if_condition_389317', if_condition_389317)
    # SSA begins for if statement (line 451)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 452):
    
    # Assigning a Name to a Name (line 452):
    # Getting the type of 'm' (line 452)
    m_389318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 25), 'm')
    # Assigning a type to the variable 'best_m' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 16), 'best_m', m_389318)
    
    # Assigning a Name to a Name (line 453):
    
    # Assigning a Name to a Name (line 453):
    # Getting the type of 's' (line 453)
    s_389319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 25), 's')
    # Assigning a type to the variable 'best_s' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 16), 'best_s', s_389319)
    # SSA join for if statement (line 451)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 448)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to range(...): (line 456)
    # Processing the call arguments (line 456)
    int_389321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 23), 'int')
    
    # Call to _compute_p_max(...): (line 456)
    # Processing the call arguments (line 456)
    # Getting the type of 'm_max' (line 456)
    m_max_389323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 41), 'm_max', False)
    # Processing the call keyword arguments (line 456)
    kwargs_389324 = {}
    # Getting the type of '_compute_p_max' (line 456)
    _compute_p_max_389322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 26), '_compute_p_max', False)
    # Calling _compute_p_max(args, kwargs) (line 456)
    _compute_p_max_call_result_389325 = invoke(stypy.reporting.localization.Localization(__file__, 456, 26), _compute_p_max_389322, *[m_max_389323], **kwargs_389324)
    
    int_389326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 50), 'int')
    # Applying the binary operator '+' (line 456)
    result_add_389327 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 26), '+', _compute_p_max_call_result_389325, int_389326)
    
    # Processing the call keyword arguments (line 456)
    kwargs_389328 = {}
    # Getting the type of 'range' (line 456)
    range_389320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'range', False)
    # Calling range(args, kwargs) (line 456)
    range_call_result_389329 = invoke(stypy.reporting.localization.Localization(__file__, 456, 17), range_389320, *[int_389321, result_add_389327], **kwargs_389328)
    
    # Testing the type of a for loop iterable (line 456)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 456, 8), range_call_result_389329)
    # Getting the type of the for loop variable (line 456)
    for_loop_var_389330 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 456, 8), range_call_result_389329)
    # Assigning a type to the variable 'p' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'p', for_loop_var_389330)
    # SSA begins for a for statement (line 456)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 457)
    # Processing the call arguments (line 457)
    # Getting the type of 'p' (line 457)
    p_389332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 27), 'p', False)
    # Getting the type of 'p' (line 457)
    p_389333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 30), 'p', False)
    int_389334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 32), 'int')
    # Applying the binary operator '-' (line 457)
    result_sub_389335 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 30), '-', p_389333, int_389334)
    
    # Applying the binary operator '*' (line 457)
    result_mul_389336 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 27), '*', p_389332, result_sub_389335)
    
    int_389337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 35), 'int')
    # Applying the binary operator '-' (line 457)
    result_sub_389338 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 27), '-', result_mul_389336, int_389337)
    
    # Getting the type of 'm_max' (line 457)
    m_max_389339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 38), 'm_max', False)
    int_389340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 44), 'int')
    # Applying the binary operator '+' (line 457)
    result_add_389341 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 38), '+', m_max_389339, int_389340)
    
    # Processing the call keyword arguments (line 457)
    kwargs_389342 = {}
    # Getting the type of 'range' (line 457)
    range_389331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 21), 'range', False)
    # Calling range(args, kwargs) (line 457)
    range_call_result_389343 = invoke(stypy.reporting.localization.Localization(__file__, 457, 21), range_389331, *[result_sub_389338, result_add_389341], **kwargs_389342)
    
    # Testing the type of a for loop iterable (line 457)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 457, 12), range_call_result_389343)
    # Getting the type of the for loop variable (line 457)
    for_loop_var_389344 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 457, 12), range_call_result_389343)
    # Assigning a type to the variable 'm' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'm', for_loop_var_389344)
    # SSA begins for a for statement (line 457)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'm' (line 458)
    m_389345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 19), 'm')
    # Getting the type of '_theta' (line 458)
    _theta_389346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 24), '_theta')
    # Applying the binary operator 'in' (line 458)
    result_contains_389347 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 19), 'in', m_389345, _theta_389346)
    
    # Testing the type of an if condition (line 458)
    if_condition_389348 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 458, 16), result_contains_389347)
    # Assigning a type to the variable 'if_condition_389348' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 16), 'if_condition_389348', if_condition_389348)
    # SSA begins for if statement (line 458)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Call to _compute_cost_div_m(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'm' (line 459)
    m_389350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 44), 'm', False)
    # Getting the type of 'p' (line 459)
    p_389351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 47), 'p', False)
    # Getting the type of 'norm_info' (line 459)
    norm_info_389352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 50), 'norm_info', False)
    # Processing the call keyword arguments (line 459)
    kwargs_389353 = {}
    # Getting the type of '_compute_cost_div_m' (line 459)
    _compute_cost_div_m_389349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 24), '_compute_cost_div_m', False)
    # Calling _compute_cost_div_m(args, kwargs) (line 459)
    _compute_cost_div_m_call_result_389354 = invoke(stypy.reporting.localization.Localization(__file__, 459, 24), _compute_cost_div_m_389349, *[m_389350, p_389351, norm_info_389352], **kwargs_389353)
    
    # Assigning a type to the variable 's' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 20), 's', _compute_cost_div_m_call_result_389354)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'best_m' (line 460)
    best_m_389355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'best_m')
    # Getting the type of 'None' (line 460)
    None_389356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 33), 'None')
    # Applying the binary operator 'is' (line 460)
    result_is__389357 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 23), 'is', best_m_389355, None_389356)
    
    
    # Getting the type of 'm' (line 460)
    m_389358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 41), 'm')
    # Getting the type of 's' (line 460)
    s_389359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 45), 's')
    # Applying the binary operator '*' (line 460)
    result_mul_389360 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 41), '*', m_389358, s_389359)
    
    # Getting the type of 'best_m' (line 460)
    best_m_389361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 49), 'best_m')
    # Getting the type of 'best_s' (line 460)
    best_s_389362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 58), 'best_s')
    # Applying the binary operator '*' (line 460)
    result_mul_389363 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 49), '*', best_m_389361, best_s_389362)
    
    # Applying the binary operator '<' (line 460)
    result_lt_389364 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 41), '<', result_mul_389360, result_mul_389363)
    
    # Applying the binary operator 'or' (line 460)
    result_or_keyword_389365 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 23), 'or', result_is__389357, result_lt_389364)
    
    # Testing the type of an if condition (line 460)
    if_condition_389366 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 460, 20), result_or_keyword_389365)
    # Assigning a type to the variable 'if_condition_389366' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 20), 'if_condition_389366', if_condition_389366)
    # SSA begins for if statement (line 460)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 461):
    
    # Assigning a Name to a Name (line 461):
    # Getting the type of 'm' (line 461)
    m_389367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 33), 'm')
    # Assigning a type to the variable 'best_m' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 24), 'best_m', m_389367)
    
    # Assigning a Name to a Name (line 462):
    
    # Assigning a Name to a Name (line 462):
    # Getting the type of 's' (line 462)
    s_389368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 33), 's')
    # Assigning a type to the variable 'best_s' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 24), 'best_s', s_389368)
    # SSA join for if statement (line 460)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 458)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 463):
    
    # Assigning a Call to a Name (line 463):
    
    # Call to max(...): (line 463)
    # Processing the call arguments (line 463)
    # Getting the type of 'best_s' (line 463)
    best_s_389370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 21), 'best_s', False)
    int_389371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 29), 'int')
    # Processing the call keyword arguments (line 463)
    kwargs_389372 = {}
    # Getting the type of 'max' (line 463)
    max_389369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 17), 'max', False)
    # Calling max(args, kwargs) (line 463)
    max_call_result_389373 = invoke(stypy.reporting.localization.Localization(__file__, 463, 17), max_389369, *[best_s_389370, int_389371], **kwargs_389372)
    
    # Assigning a type to the variable 'best_s' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'best_s', max_call_result_389373)
    # SSA join for if statement (line 448)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 464)
    tuple_389374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 464)
    # Adding element type (line 464)
    # Getting the type of 'best_m' (line 464)
    best_m_389375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 11), 'best_m')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 11), tuple_389374, best_m_389375)
    # Adding element type (line 464)
    # Getting the type of 'best_s' (line 464)
    best_s_389376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 19), 'best_s')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 11), tuple_389374, best_s_389376)
    
    # Assigning a type to the variable 'stypy_return_type' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'stypy_return_type', tuple_389374)
    
    # ################# End of '_fragment_3_1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fragment_3_1' in the type store
    # Getting the type of 'stypy_return_type' (line 409)
    stypy_return_type_389377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_389377)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fragment_3_1'
    return stypy_return_type_389377

# Assigning a type to the variable '_fragment_3_1' (line 409)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), '_fragment_3_1', _fragment_3_1)

@norecursion
def _condition_3_13(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_condition_3_13'
    module_type_store = module_type_store.open_function_context('_condition_3_13', 467, 0, False)
    
    # Passed parameters checking function
    _condition_3_13.stypy_localization = localization
    _condition_3_13.stypy_type_of_self = None
    _condition_3_13.stypy_type_store = module_type_store
    _condition_3_13.stypy_function_name = '_condition_3_13'
    _condition_3_13.stypy_param_names_list = ['A_1_norm', 'n0', 'm_max', 'ell']
    _condition_3_13.stypy_varargs_param_name = None
    _condition_3_13.stypy_kwargs_param_name = None
    _condition_3_13.stypy_call_defaults = defaults
    _condition_3_13.stypy_call_varargs = varargs
    _condition_3_13.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_condition_3_13', ['A_1_norm', 'n0', 'm_max', 'ell'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_condition_3_13', localization, ['A_1_norm', 'n0', 'm_max', 'ell'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_condition_3_13(...)' code ##################

    str_389378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, (-1)), 'str', '\n    A helper function for the _expm_multiply_* functions.\n\n    Parameters\n    ----------\n    A_1_norm : float\n        The precomputed 1-norm of A.\n    n0 : int\n        Number of columns in the _expm_multiply_* B matrix.\n    m_max : int\n        A value related to a bound.\n    ell : int\n        The number of columns used in the 1-norm approximation.\n        This is usually taken to be small, maybe between 1 and 5.\n\n    Returns\n    -------\n    value : bool\n        Indicates whether or not the condition has been met.\n\n    Notes\n    -----\n    This is condition (3.13) in Al-Mohy and Higham (2011).\n\n    ')
    
    # Assigning a Call to a Name (line 495):
    
    # Assigning a Call to a Name (line 495):
    
    # Call to _compute_p_max(...): (line 495)
    # Processing the call arguments (line 495)
    # Getting the type of 'm_max' (line 495)
    m_max_389380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 27), 'm_max', False)
    # Processing the call keyword arguments (line 495)
    kwargs_389381 = {}
    # Getting the type of '_compute_p_max' (line 495)
    _compute_p_max_389379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), '_compute_p_max', False)
    # Calling _compute_p_max(args, kwargs) (line 495)
    _compute_p_max_call_result_389382 = invoke(stypy.reporting.localization.Localization(__file__, 495, 12), _compute_p_max_389379, *[m_max_389380], **kwargs_389381)
    
    # Assigning a type to the variable 'p_max' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'p_max', _compute_p_max_call_result_389382)
    
    # Assigning a BinOp to a Name (line 496):
    
    # Assigning a BinOp to a Name (line 496):
    int_389383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 8), 'int')
    # Getting the type of 'ell' (line 496)
    ell_389384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'ell')
    # Applying the binary operator '*' (line 496)
    result_mul_389385 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 8), '*', int_389383, ell_389384)
    
    # Getting the type of 'p_max' (line 496)
    p_max_389386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 18), 'p_max')
    # Applying the binary operator '*' (line 496)
    result_mul_389387 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 16), '*', result_mul_389385, p_max_389386)
    
    # Getting the type of 'p_max' (line 496)
    p_max_389388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 27), 'p_max')
    int_389389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 35), 'int')
    # Applying the binary operator '+' (line 496)
    result_add_389390 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 27), '+', p_max_389388, int_389389)
    
    # Applying the binary operator '*' (line 496)
    result_mul_389391 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 24), '*', result_mul_389387, result_add_389390)
    
    # Assigning a type to the variable 'a' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'a', result_mul_389391)
    
    # Assigning a BinOp to a Name (line 499):
    
    # Assigning a BinOp to a Name (line 499):
    
    # Obtaining the type of the subscript
    # Getting the type of 'm_max' (line 499)
    m_max_389392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 15), 'm_max')
    # Getting the type of '_theta' (line 499)
    _theta_389393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), '_theta')
    # Obtaining the member '__getitem__' of a type (line 499)
    getitem___389394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 8), _theta_389393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 499)
    subscript_call_result_389395 = invoke(stypy.reporting.localization.Localization(__file__, 499, 8), getitem___389394, m_max_389392)
    
    
    # Call to float(...): (line 499)
    # Processing the call arguments (line 499)
    # Getting the type of 'n0' (line 499)
    n0_389397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 30), 'n0', False)
    # Getting the type of 'm_max' (line 499)
    m_max_389398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 35), 'm_max', False)
    # Applying the binary operator '*' (line 499)
    result_mul_389399 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 30), '*', n0_389397, m_max_389398)
    
    # Processing the call keyword arguments (line 499)
    kwargs_389400 = {}
    # Getting the type of 'float' (line 499)
    float_389396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 24), 'float', False)
    # Calling float(args, kwargs) (line 499)
    float_call_result_389401 = invoke(stypy.reporting.localization.Localization(__file__, 499, 24), float_389396, *[result_mul_389399], **kwargs_389400)
    
    # Applying the binary operator 'div' (line 499)
    result_div_389402 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 8), 'div', subscript_call_result_389395, float_call_result_389401)
    
    # Assigning a type to the variable 'b' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'b', result_div_389402)
    
    # Getting the type of 'A_1_norm' (line 500)
    A_1_norm_389403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 11), 'A_1_norm')
    # Getting the type of 'a' (line 500)
    a_389404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 23), 'a')
    # Getting the type of 'b' (line 500)
    b_389405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 27), 'b')
    # Applying the binary operator '*' (line 500)
    result_mul_389406 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 23), '*', a_389404, b_389405)
    
    # Applying the binary operator '<=' (line 500)
    result_le_389407 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 11), '<=', A_1_norm_389403, result_mul_389406)
    
    # Assigning a type to the variable 'stypy_return_type' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'stypy_return_type', result_le_389407)
    
    # ################# End of '_condition_3_13(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_condition_3_13' in the type store
    # Getting the type of 'stypy_return_type' (line 467)
    stypy_return_type_389408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_389408)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_condition_3_13'
    return stypy_return_type_389408

# Assigning a type to the variable '_condition_3_13' (line 467)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 0), '_condition_3_13', _condition_3_13)

@norecursion
def _expm_multiply_interval(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 503)
    None_389409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 40), 'None')
    # Getting the type of 'None' (line 503)
    None_389410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 51), 'None')
    # Getting the type of 'None' (line 504)
    None_389411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'None')
    # Getting the type of 'None' (line 504)
    None_389412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 27), 'None')
    # Getting the type of 'False' (line 504)
    False_389413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 41), 'False')
    # Getting the type of 'False' (line 504)
    False_389414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 60), 'False')
    defaults = [None_389409, None_389410, None_389411, None_389412, False_389413, False_389414]
    # Create a new context for function '_expm_multiply_interval'
    module_type_store = module_type_store.open_function_context('_expm_multiply_interval', 503, 0, False)
    
    # Passed parameters checking function
    _expm_multiply_interval.stypy_localization = localization
    _expm_multiply_interval.stypy_type_of_self = None
    _expm_multiply_interval.stypy_type_store = module_type_store
    _expm_multiply_interval.stypy_function_name = '_expm_multiply_interval'
    _expm_multiply_interval.stypy_param_names_list = ['A', 'B', 'start', 'stop', 'num', 'endpoint', 'balance', 'status_only']
    _expm_multiply_interval.stypy_varargs_param_name = None
    _expm_multiply_interval.stypy_kwargs_param_name = None
    _expm_multiply_interval.stypy_call_defaults = defaults
    _expm_multiply_interval.stypy_call_varargs = varargs
    _expm_multiply_interval.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_expm_multiply_interval', ['A', 'B', 'start', 'stop', 'num', 'endpoint', 'balance', 'status_only'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_expm_multiply_interval', localization, ['A', 'B', 'start', 'stop', 'num', 'endpoint', 'balance', 'status_only'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_expm_multiply_interval(...)' code ##################

    str_389415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, (-1)), 'str', '\n    Compute the action of the matrix exponential at multiple time points.\n\n    Parameters\n    ----------\n    A : transposable linear operator\n        The operator whose exponential is of interest.\n    B : ndarray\n        The matrix to be multiplied by the matrix exponential of A.\n    start : scalar, optional\n        The starting time point of the sequence.\n    stop : scalar, optional\n        The end time point of the sequence, unless `endpoint` is set to False.\n        In that case, the sequence consists of all but the last of ``num + 1``\n        evenly spaced time points, so that `stop` is excluded.\n        Note that the step size changes when `endpoint` is False.\n    num : int, optional\n        Number of time points to use.\n    endpoint : bool, optional\n        If True, `stop` is the last time point.  Otherwise, it is not included.\n    balance : bool\n        Indicates whether or not to apply balancing.\n    status_only : bool\n        A flag that is set to True for some debugging and testing operations.\n\n    Returns\n    -------\n    F : ndarray\n        :math:`e^{t_k A} B`\n    status : int\n        An integer status for testing and debugging.\n\n    Notes\n    -----\n    This is algorithm (5.2) in Al-Mohy and Higham (2011).\n\n    There seems to be a typo, where line 15 of the algorithm should be\n    moved to line 6.5 (between lines 6 and 7).\n\n    ')
    
    # Getting the type of 'balance' (line 545)
    balance_389416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 7), 'balance')
    # Testing the type of an if condition (line 545)
    if_condition_389417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 545, 4), balance_389416)
    # Assigning a type to the variable 'if_condition_389417' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'if_condition_389417', if_condition_389417)
    # SSA begins for if statement (line 545)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'NotImplementedError' (line 546)
    NotImplementedError_389418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 14), 'NotImplementedError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 546, 8), NotImplementedError_389418, 'raise parameter', BaseException)
    # SSA join for if statement (line 545)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 547)
    # Processing the call arguments (line 547)
    # Getting the type of 'A' (line 547)
    A_389420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 547)
    shape_389421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 11), A_389420, 'shape')
    # Processing the call keyword arguments (line 547)
    kwargs_389422 = {}
    # Getting the type of 'len' (line 547)
    len_389419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 7), 'len', False)
    # Calling len(args, kwargs) (line 547)
    len_call_result_389423 = invoke(stypy.reporting.localization.Localization(__file__, 547, 7), len_389419, *[shape_389421], **kwargs_389422)
    
    int_389424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 23), 'int')
    # Applying the binary operator '!=' (line 547)
    result_ne_389425 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 7), '!=', len_call_result_389423, int_389424)
    
    
    
    # Obtaining the type of the subscript
    int_389426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 36), 'int')
    # Getting the type of 'A' (line 547)
    A_389427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 28), 'A')
    # Obtaining the member 'shape' of a type (line 547)
    shape_389428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 28), A_389427, 'shape')
    # Obtaining the member '__getitem__' of a type (line 547)
    getitem___389429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 28), shape_389428, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 547)
    subscript_call_result_389430 = invoke(stypy.reporting.localization.Localization(__file__, 547, 28), getitem___389429, int_389426)
    
    
    # Obtaining the type of the subscript
    int_389431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 50), 'int')
    # Getting the type of 'A' (line 547)
    A_389432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 42), 'A')
    # Obtaining the member 'shape' of a type (line 547)
    shape_389433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 42), A_389432, 'shape')
    # Obtaining the member '__getitem__' of a type (line 547)
    getitem___389434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 42), shape_389433, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 547)
    subscript_call_result_389435 = invoke(stypy.reporting.localization.Localization(__file__, 547, 42), getitem___389434, int_389431)
    
    # Applying the binary operator '!=' (line 547)
    result_ne_389436 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 28), '!=', subscript_call_result_389430, subscript_call_result_389435)
    
    # Applying the binary operator 'or' (line 547)
    result_or_keyword_389437 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 7), 'or', result_ne_389425, result_ne_389436)
    
    # Testing the type of an if condition (line 547)
    if_condition_389438 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 547, 4), result_or_keyword_389437)
    # Assigning a type to the variable 'if_condition_389438' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'if_condition_389438', if_condition_389438)
    # SSA begins for if statement (line 547)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 548)
    # Processing the call arguments (line 548)
    str_389440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 25), 'str', 'expected A to be like a square matrix')
    # Processing the call keyword arguments (line 548)
    kwargs_389441 = {}
    # Getting the type of 'ValueError' (line 548)
    ValueError_389439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 548)
    ValueError_call_result_389442 = invoke(stypy.reporting.localization.Localization(__file__, 548, 14), ValueError_389439, *[str_389440], **kwargs_389441)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 548, 8), ValueError_call_result_389442, 'raise parameter', BaseException)
    # SSA join for if statement (line 547)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_389443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 15), 'int')
    # Getting the type of 'A' (line 549)
    A_389444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 7), 'A')
    # Obtaining the member 'shape' of a type (line 549)
    shape_389445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 7), A_389444, 'shape')
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___389446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 7), shape_389445, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_389447 = invoke(stypy.reporting.localization.Localization(__file__, 549, 7), getitem___389446, int_389443)
    
    
    # Obtaining the type of the subscript
    int_389448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 29), 'int')
    # Getting the type of 'B' (line 549)
    B_389449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 21), 'B')
    # Obtaining the member 'shape' of a type (line 549)
    shape_389450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 21), B_389449, 'shape')
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___389451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 21), shape_389450, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_389452 = invoke(stypy.reporting.localization.Localization(__file__, 549, 21), getitem___389451, int_389448)
    
    # Applying the binary operator '!=' (line 549)
    result_ne_389453 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 7), '!=', subscript_call_result_389447, subscript_call_result_389452)
    
    # Testing the type of an if condition (line 549)
    if_condition_389454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 549, 4), result_ne_389453)
    # Assigning a type to the variable 'if_condition_389454' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'if_condition_389454', if_condition_389454)
    # SSA begins for if statement (line 549)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 550)
    # Processing the call arguments (line 550)
    str_389456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 25), 'str', 'the matrices A and B have incompatible shapes')
    # Processing the call keyword arguments (line 550)
    kwargs_389457 = {}
    # Getting the type of 'ValueError' (line 550)
    ValueError_389455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 550)
    ValueError_call_result_389458 = invoke(stypy.reporting.localization.Localization(__file__, 550, 14), ValueError_389455, *[str_389456], **kwargs_389457)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 550, 8), ValueError_call_result_389458, 'raise parameter', BaseException)
    # SSA join for if statement (line 549)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 551):
    
    # Assigning a Call to a Name (line 551):
    
    # Call to _ident_like(...): (line 551)
    # Processing the call arguments (line 551)
    # Getting the type of 'A' (line 551)
    A_389460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 24), 'A', False)
    # Processing the call keyword arguments (line 551)
    kwargs_389461 = {}
    # Getting the type of '_ident_like' (line 551)
    _ident_like_389459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), '_ident_like', False)
    # Calling _ident_like(args, kwargs) (line 551)
    _ident_like_call_result_389462 = invoke(stypy.reporting.localization.Localization(__file__, 551, 12), _ident_like_389459, *[A_389460], **kwargs_389461)
    
    # Assigning a type to the variable 'ident' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'ident', _ident_like_call_result_389462)
    
    # Assigning a Subscript to a Name (line 552):
    
    # Assigning a Subscript to a Name (line 552):
    
    # Obtaining the type of the subscript
    int_389463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 16), 'int')
    # Getting the type of 'A' (line 552)
    A_389464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'A')
    # Obtaining the member 'shape' of a type (line 552)
    shape_389465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 8), A_389464, 'shape')
    # Obtaining the member '__getitem__' of a type (line 552)
    getitem___389466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 8), shape_389465, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 552)
    subscript_call_result_389467 = invoke(stypy.reporting.localization.Localization(__file__, 552, 8), getitem___389466, int_389463)
    
    # Assigning a type to the variable 'n' (line 552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'n', subscript_call_result_389467)
    
    
    
    # Call to len(...): (line 553)
    # Processing the call arguments (line 553)
    # Getting the type of 'B' (line 553)
    B_389469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 11), 'B', False)
    # Obtaining the member 'shape' of a type (line 553)
    shape_389470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 11), B_389469, 'shape')
    # Processing the call keyword arguments (line 553)
    kwargs_389471 = {}
    # Getting the type of 'len' (line 553)
    len_389468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 7), 'len', False)
    # Calling len(args, kwargs) (line 553)
    len_call_result_389472 = invoke(stypy.reporting.localization.Localization(__file__, 553, 7), len_389468, *[shape_389470], **kwargs_389471)
    
    int_389473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 23), 'int')
    # Applying the binary operator '==' (line 553)
    result_eq_389474 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 7), '==', len_call_result_389472, int_389473)
    
    # Testing the type of an if condition (line 553)
    if_condition_389475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 553, 4), result_eq_389474)
    # Assigning a type to the variable 'if_condition_389475' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'if_condition_389475', if_condition_389475)
    # SSA begins for if statement (line 553)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 554):
    
    # Assigning a Num to a Name (line 554):
    int_389476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 13), 'int')
    # Assigning a type to the variable 'n0' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'n0', int_389476)
    # SSA branch for the else part of an if statement (line 553)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 555)
    # Processing the call arguments (line 555)
    # Getting the type of 'B' (line 555)
    B_389478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 13), 'B', False)
    # Obtaining the member 'shape' of a type (line 555)
    shape_389479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 13), B_389478, 'shape')
    # Processing the call keyword arguments (line 555)
    kwargs_389480 = {}
    # Getting the type of 'len' (line 555)
    len_389477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 9), 'len', False)
    # Calling len(args, kwargs) (line 555)
    len_call_result_389481 = invoke(stypy.reporting.localization.Localization(__file__, 555, 9), len_389477, *[shape_389479], **kwargs_389480)
    
    int_389482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 25), 'int')
    # Applying the binary operator '==' (line 555)
    result_eq_389483 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 9), '==', len_call_result_389481, int_389482)
    
    # Testing the type of an if condition (line 555)
    if_condition_389484 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 555, 9), result_eq_389483)
    # Assigning a type to the variable 'if_condition_389484' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 9), 'if_condition_389484', if_condition_389484)
    # SSA begins for if statement (line 555)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 556):
    
    # Assigning a Subscript to a Name (line 556):
    
    # Obtaining the type of the subscript
    int_389485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 21), 'int')
    # Getting the type of 'B' (line 556)
    B_389486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 13), 'B')
    # Obtaining the member 'shape' of a type (line 556)
    shape_389487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 13), B_389486, 'shape')
    # Obtaining the member '__getitem__' of a type (line 556)
    getitem___389488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 13), shape_389487, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 556)
    subscript_call_result_389489 = invoke(stypy.reporting.localization.Localization(__file__, 556, 13), getitem___389488, int_389485)
    
    # Assigning a type to the variable 'n0' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'n0', subscript_call_result_389489)
    # SSA branch for the else part of an if statement (line 555)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 558)
    # Processing the call arguments (line 558)
    str_389491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 25), 'str', 'expected B to be like a matrix or a vector')
    # Processing the call keyword arguments (line 558)
    kwargs_389492 = {}
    # Getting the type of 'ValueError' (line 558)
    ValueError_389490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 558)
    ValueError_call_result_389493 = invoke(stypy.reporting.localization.Localization(__file__, 558, 14), ValueError_389490, *[str_389491], **kwargs_389492)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 558, 8), ValueError_call_result_389493, 'raise parameter', BaseException)
    # SSA join for if statement (line 555)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 553)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 559):
    
    # Assigning a BinOp to a Name (line 559):
    int_389494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 10), 'int')
    int_389495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 13), 'int')
    # Applying the binary operator '**' (line 559)
    result_pow_389496 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 10), '**', int_389494, int_389495)
    
    # Assigning a type to the variable 'u_d' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'u_d', result_pow_389496)
    
    # Assigning a Name to a Name (line 560):
    
    # Assigning a Name to a Name (line 560):
    # Getting the type of 'u_d' (line 560)
    u_d_389497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 10), 'u_d')
    # Assigning a type to the variable 'tol' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'tol', u_d_389497)
    
    # Assigning a BinOp to a Name (line 561):
    
    # Assigning a BinOp to a Name (line 561):
    
    # Call to _trace(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'A' (line 561)
    A_389499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), 'A', False)
    # Processing the call keyword arguments (line 561)
    kwargs_389500 = {}
    # Getting the type of '_trace' (line 561)
    _trace_389498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 9), '_trace', False)
    # Calling _trace(args, kwargs) (line 561)
    _trace_call_result_389501 = invoke(stypy.reporting.localization.Localization(__file__, 561, 9), _trace_389498, *[A_389499], **kwargs_389500)
    
    
    # Call to float(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'n' (line 561)
    n_389503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 27), 'n', False)
    # Processing the call keyword arguments (line 561)
    kwargs_389504 = {}
    # Getting the type of 'float' (line 561)
    float_389502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 21), 'float', False)
    # Calling float(args, kwargs) (line 561)
    float_call_result_389505 = invoke(stypy.reporting.localization.Localization(__file__, 561, 21), float_389502, *[n_389503], **kwargs_389504)
    
    # Applying the binary operator 'div' (line 561)
    result_div_389506 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 9), 'div', _trace_call_result_389501, float_call_result_389505)
    
    # Assigning a type to the variable 'mu' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'mu', result_div_389506)
    
    # Assigning a Dict to a Name (line 564):
    
    # Assigning a Dict to a Name (line 564):
    
    # Obtaining an instance of the builtin type 'dict' (line 564)
    dict_389507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 22), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 564)
    # Adding element type (key, value) (line 564)
    str_389508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 23), 'str', 'retstep')
    # Getting the type of 'True' (line 564)
    True_389509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 34), 'True')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 564, 22), dict_389507, (str_389508, True_389509))
    
    # Assigning a type to the variable 'linspace_kwargs' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'linspace_kwargs', dict_389507)
    
    # Type idiom detected: calculating its left and rigth part (line 565)
    # Getting the type of 'num' (line 565)
    num_389510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'num')
    # Getting the type of 'None' (line 565)
    None_389511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 18), 'None')
    
    (may_be_389512, more_types_in_union_389513) = may_not_be_none(num_389510, None_389511)

    if may_be_389512:

        if more_types_in_union_389513:
            # Runtime conditional SSA (line 565)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Subscript (line 566):
        
        # Assigning a Name to a Subscript (line 566):
        # Getting the type of 'num' (line 566)
        num_389514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 33), 'num')
        # Getting the type of 'linspace_kwargs' (line 566)
        linspace_kwargs_389515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'linspace_kwargs')
        str_389516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 24), 'str', 'num')
        # Storing an element on a container (line 566)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 8), linspace_kwargs_389515, (str_389516, num_389514))

        if more_types_in_union_389513:
            # SSA join for if statement (line 565)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 567)
    # Getting the type of 'endpoint' (line 567)
    endpoint_389517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'endpoint')
    # Getting the type of 'None' (line 567)
    None_389518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 23), 'None')
    
    (may_be_389519, more_types_in_union_389520) = may_not_be_none(endpoint_389517, None_389518)

    if may_be_389519:

        if more_types_in_union_389520:
            # Runtime conditional SSA (line 567)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Subscript (line 568):
        
        # Assigning a Name to a Subscript (line 568):
        # Getting the type of 'endpoint' (line 568)
        endpoint_389521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 38), 'endpoint')
        # Getting the type of 'linspace_kwargs' (line 568)
        linspace_kwargs_389522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'linspace_kwargs')
        str_389523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 24), 'str', 'endpoint')
        # Storing an element on a container (line 568)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 8), linspace_kwargs_389522, (str_389523, endpoint_389521))

        if more_types_in_union_389520:
            # SSA join for if statement (line 567)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 569):
    
    # Assigning a Subscript to a Name (line 569):
    
    # Obtaining the type of the subscript
    int_389524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 4), 'int')
    
    # Call to linspace(...): (line 569)
    # Processing the call arguments (line 569)
    # Getting the type of 'start' (line 569)
    start_389527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 32), 'start', False)
    # Getting the type of 'stop' (line 569)
    stop_389528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 39), 'stop', False)
    # Processing the call keyword arguments (line 569)
    # Getting the type of 'linspace_kwargs' (line 569)
    linspace_kwargs_389529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 47), 'linspace_kwargs', False)
    kwargs_389530 = {'linspace_kwargs_389529': linspace_kwargs_389529}
    # Getting the type of 'np' (line 569)
    np_389525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 20), 'np', False)
    # Obtaining the member 'linspace' of a type (line 569)
    linspace_389526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 20), np_389525, 'linspace')
    # Calling linspace(args, kwargs) (line 569)
    linspace_call_result_389531 = invoke(stypy.reporting.localization.Localization(__file__, 569, 20), linspace_389526, *[start_389527, stop_389528], **kwargs_389530)
    
    # Obtaining the member '__getitem__' of a type (line 569)
    getitem___389532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 4), linspace_call_result_389531, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 569)
    subscript_call_result_389533 = invoke(stypy.reporting.localization.Localization(__file__, 569, 4), getitem___389532, int_389524)
    
    # Assigning a type to the variable 'tuple_var_assignment_388582' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'tuple_var_assignment_388582', subscript_call_result_389533)
    
    # Assigning a Subscript to a Name (line 569):
    
    # Obtaining the type of the subscript
    int_389534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 4), 'int')
    
    # Call to linspace(...): (line 569)
    # Processing the call arguments (line 569)
    # Getting the type of 'start' (line 569)
    start_389537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 32), 'start', False)
    # Getting the type of 'stop' (line 569)
    stop_389538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 39), 'stop', False)
    # Processing the call keyword arguments (line 569)
    # Getting the type of 'linspace_kwargs' (line 569)
    linspace_kwargs_389539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 47), 'linspace_kwargs', False)
    kwargs_389540 = {'linspace_kwargs_389539': linspace_kwargs_389539}
    # Getting the type of 'np' (line 569)
    np_389535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 20), 'np', False)
    # Obtaining the member 'linspace' of a type (line 569)
    linspace_389536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 20), np_389535, 'linspace')
    # Calling linspace(args, kwargs) (line 569)
    linspace_call_result_389541 = invoke(stypy.reporting.localization.Localization(__file__, 569, 20), linspace_389536, *[start_389537, stop_389538], **kwargs_389540)
    
    # Obtaining the member '__getitem__' of a type (line 569)
    getitem___389542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 4), linspace_call_result_389541, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 569)
    subscript_call_result_389543 = invoke(stypy.reporting.localization.Localization(__file__, 569, 4), getitem___389542, int_389534)
    
    # Assigning a type to the variable 'tuple_var_assignment_388583' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'tuple_var_assignment_388583', subscript_call_result_389543)
    
    # Assigning a Name to a Name (line 569):
    # Getting the type of 'tuple_var_assignment_388582' (line 569)
    tuple_var_assignment_388582_389544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'tuple_var_assignment_388582')
    # Assigning a type to the variable 'samples' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'samples', tuple_var_assignment_388582_389544)
    
    # Assigning a Name to a Name (line 569):
    # Getting the type of 'tuple_var_assignment_388583' (line 569)
    tuple_var_assignment_388583_389545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'tuple_var_assignment_388583')
    # Assigning a type to the variable 'step' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 13), 'step', tuple_var_assignment_388583_389545)
    
    # Assigning a Call to a Name (line 572):
    
    # Assigning a Call to a Name (line 572):
    
    # Call to len(...): (line 572)
    # Processing the call arguments (line 572)
    # Getting the type of 'samples' (line 572)
    samples_389547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 19), 'samples', False)
    # Processing the call keyword arguments (line 572)
    kwargs_389548 = {}
    # Getting the type of 'len' (line 572)
    len_389546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 15), 'len', False)
    # Calling len(args, kwargs) (line 572)
    len_call_result_389549 = invoke(stypy.reporting.localization.Localization(__file__, 572, 15), len_389546, *[samples_389547], **kwargs_389548)
    
    # Assigning a type to the variable 'nsamples' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'nsamples', len_call_result_389549)
    
    
    # Getting the type of 'nsamples' (line 573)
    nsamples_389550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 7), 'nsamples')
    int_389551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 18), 'int')
    # Applying the binary operator '<' (line 573)
    result_lt_389552 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 7), '<', nsamples_389550, int_389551)
    
    # Testing the type of an if condition (line 573)
    if_condition_389553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 573, 4), result_lt_389552)
    # Assigning a type to the variable 'if_condition_389553' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'if_condition_389553', if_condition_389553)
    # SSA begins for if statement (line 573)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 574)
    # Processing the call arguments (line 574)
    str_389555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 25), 'str', 'at least two time points are required')
    # Processing the call keyword arguments (line 574)
    kwargs_389556 = {}
    # Getting the type of 'ValueError' (line 574)
    ValueError_389554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 574)
    ValueError_call_result_389557 = invoke(stypy.reporting.localization.Localization(__file__, 574, 14), ValueError_389554, *[str_389555], **kwargs_389556)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 574, 8), ValueError_call_result_389557, 'raise parameter', BaseException)
    # SSA join for if statement (line 573)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 575):
    
    # Assigning a BinOp to a Name (line 575):
    # Getting the type of 'nsamples' (line 575)
    nsamples_389558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'nsamples')
    int_389559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 19), 'int')
    # Applying the binary operator '-' (line 575)
    result_sub_389560 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 8), '-', nsamples_389558, int_389559)
    
    # Assigning a type to the variable 'q' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'q', result_sub_389560)
    
    # Assigning a Name to a Name (line 576):
    
    # Assigning a Name to a Name (line 576):
    # Getting the type of 'step' (line 576)
    step_389561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'step')
    # Assigning a type to the variable 'h' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'h', step_389561)
    
    # Assigning a Subscript to a Name (line 577):
    
    # Assigning a Subscript to a Name (line 577):
    
    # Obtaining the type of the subscript
    int_389562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 18), 'int')
    # Getting the type of 'samples' (line 577)
    samples_389563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 10), 'samples')
    # Obtaining the member '__getitem__' of a type (line 577)
    getitem___389564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 10), samples_389563, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 577)
    subscript_call_result_389565 = invoke(stypy.reporting.localization.Localization(__file__, 577, 10), getitem___389564, int_389562)
    
    # Assigning a type to the variable 't_0' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 't_0', subscript_call_result_389565)
    
    # Assigning a Subscript to a Name (line 578):
    
    # Assigning a Subscript to a Name (line 578):
    
    # Obtaining the type of the subscript
    # Getting the type of 'q' (line 578)
    q_389566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 18), 'q')
    # Getting the type of 'samples' (line 578)
    samples_389567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 10), 'samples')
    # Obtaining the member '__getitem__' of a type (line 578)
    getitem___389568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 10), samples_389567, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 578)
    subscript_call_result_389569 = invoke(stypy.reporting.localization.Localization(__file__, 578, 10), getitem___389568, q_389566)
    
    # Assigning a type to the variable 't_q' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 't_q', subscript_call_result_389569)
    
    # Assigning a BinOp to a Name (line 583):
    
    # Assigning a BinOp to a Name (line 583):
    
    # Obtaining an instance of the builtin type 'tuple' (line 583)
    tuple_389570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 583)
    # Adding element type (line 583)
    # Getting the type of 'nsamples' (line 583)
    nsamples_389571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 15), 'nsamples')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 15), tuple_389570, nsamples_389571)
    
    # Getting the type of 'B' (line 583)
    B_389572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 28), 'B')
    # Obtaining the member 'shape' of a type (line 583)
    shape_389573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 28), B_389572, 'shape')
    # Applying the binary operator '+' (line 583)
    result_add_389574 = python_operator(stypy.reporting.localization.Localization(__file__, 583, 14), '+', tuple_389570, shape_389573)
    
    # Assigning a type to the variable 'X_shape' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'X_shape', result_add_389574)
    
    # Assigning a Call to a Name (line 584):
    
    # Assigning a Call to a Name (line 584):
    
    # Call to empty(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'X_shape' (line 584)
    X_shape_389577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 17), 'X_shape', False)
    # Processing the call keyword arguments (line 584)
    
    # Call to result_type(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'A' (line 584)
    A_389580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 47), 'A', False)
    # Obtaining the member 'dtype' of a type (line 584)
    dtype_389581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 47), A_389580, 'dtype')
    # Getting the type of 'B' (line 584)
    B_389582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 56), 'B', False)
    # Obtaining the member 'dtype' of a type (line 584)
    dtype_389583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 56), B_389582, 'dtype')
    # Getting the type of 'float' (line 584)
    float_389584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 65), 'float', False)
    # Processing the call keyword arguments (line 584)
    kwargs_389585 = {}
    # Getting the type of 'np' (line 584)
    np_389578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 32), 'np', False)
    # Obtaining the member 'result_type' of a type (line 584)
    result_type_389579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 32), np_389578, 'result_type')
    # Calling result_type(args, kwargs) (line 584)
    result_type_call_result_389586 = invoke(stypy.reporting.localization.Localization(__file__, 584, 32), result_type_389579, *[dtype_389581, dtype_389583, float_389584], **kwargs_389585)
    
    keyword_389587 = result_type_call_result_389586
    kwargs_389588 = {'dtype': keyword_389587}
    # Getting the type of 'np' (line 584)
    np_389575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 584)
    empty_389576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 8), np_389575, 'empty')
    # Calling empty(args, kwargs) (line 584)
    empty_call_result_389589 = invoke(stypy.reporting.localization.Localization(__file__, 584, 8), empty_389576, *[X_shape_389577], **kwargs_389588)
    
    # Assigning a type to the variable 'X' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'X', empty_call_result_389589)
    
    # Assigning a BinOp to a Name (line 585):
    
    # Assigning a BinOp to a Name (line 585):
    # Getting the type of 't_q' (line 585)
    t_q_389590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 't_q')
    # Getting the type of 't_0' (line 585)
    t_0_389591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 14), 't_0')
    # Applying the binary operator '-' (line 585)
    result_sub_389592 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 8), '-', t_q_389590, t_0_389591)
    
    # Assigning a type to the variable 't' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 't', result_sub_389592)
    
    # Assigning a BinOp to a Name (line 586):
    
    # Assigning a BinOp to a Name (line 586):
    # Getting the type of 'A' (line 586)
    A_389593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'A')
    # Getting the type of 'mu' (line 586)
    mu_389594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 12), 'mu')
    # Getting the type of 'ident' (line 586)
    ident_389595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 17), 'ident')
    # Applying the binary operator '*' (line 586)
    result_mul_389596 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 12), '*', mu_389594, ident_389595)
    
    # Applying the binary operator '-' (line 586)
    result_sub_389597 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 8), '-', A_389593, result_mul_389596)
    
    # Assigning a type to the variable 'A' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'A', result_sub_389597)
    
    # Assigning a Call to a Name (line 587):
    
    # Assigning a Call to a Name (line 587):
    
    # Call to _exact_1_norm(...): (line 587)
    # Processing the call arguments (line 587)
    # Getting the type of 'A' (line 587)
    A_389599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 29), 'A', False)
    # Processing the call keyword arguments (line 587)
    kwargs_389600 = {}
    # Getting the type of '_exact_1_norm' (line 587)
    _exact_1_norm_389598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 15), '_exact_1_norm', False)
    # Calling _exact_1_norm(args, kwargs) (line 587)
    _exact_1_norm_call_result_389601 = invoke(stypy.reporting.localization.Localization(__file__, 587, 15), _exact_1_norm_389598, *[A_389599], **kwargs_389600)
    
    # Assigning a type to the variable 'A_1_norm' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'A_1_norm', _exact_1_norm_call_result_389601)
    
    # Assigning a Num to a Name (line 588):
    
    # Assigning a Num to a Name (line 588):
    int_389602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 10), 'int')
    # Assigning a type to the variable 'ell' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 4), 'ell', int_389602)
    
    # Assigning a Call to a Name (line 589):
    
    # Assigning a Call to a Name (line 589):
    
    # Call to LazyOperatorNormInfo(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 't' (line 589)
    t_389604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 37), 't', False)
    # Getting the type of 'A' (line 589)
    A_389605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 39), 'A', False)
    # Applying the binary operator '*' (line 589)
    result_mul_389606 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 37), '*', t_389604, A_389605)
    
    # Processing the call keyword arguments (line 589)
    # Getting the type of 't' (line 589)
    t_389607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 51), 't', False)
    # Getting the type of 'A_1_norm' (line 589)
    A_1_norm_389608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 53), 'A_1_norm', False)
    # Applying the binary operator '*' (line 589)
    result_mul_389609 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 51), '*', t_389607, A_1_norm_389608)
    
    keyword_389610 = result_mul_389609
    # Getting the type of 'ell' (line 589)
    ell_389611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 67), 'ell', False)
    keyword_389612 = ell_389611
    kwargs_389613 = {'A_1_norm': keyword_389610, 'ell': keyword_389612}
    # Getting the type of 'LazyOperatorNormInfo' (line 589)
    LazyOperatorNormInfo_389603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'LazyOperatorNormInfo', False)
    # Calling LazyOperatorNormInfo(args, kwargs) (line 589)
    LazyOperatorNormInfo_call_result_389614 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), LazyOperatorNormInfo_389603, *[result_mul_389606], **kwargs_389613)
    
    # Assigning a type to the variable 'norm_info' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'norm_info', LazyOperatorNormInfo_call_result_389614)
    
    
    # Getting the type of 't' (line 590)
    t_389615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 7), 't')
    # Getting the type of 'A_1_norm' (line 590)
    A_1_norm_389616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 9), 'A_1_norm')
    # Applying the binary operator '*' (line 590)
    result_mul_389617 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 7), '*', t_389615, A_1_norm_389616)
    
    int_389618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 21), 'int')
    # Applying the binary operator '==' (line 590)
    result_eq_389619 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 7), '==', result_mul_389617, int_389618)
    
    # Testing the type of an if condition (line 590)
    if_condition_389620 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 590, 4), result_eq_389619)
    # Assigning a type to the variable 'if_condition_389620' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 4), 'if_condition_389620', if_condition_389620)
    # SSA begins for if statement (line 590)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 591):
    
    # Assigning a Num to a Name (line 591):
    int_389621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 20), 'int')
    # Assigning a type to the variable 'tuple_assignment_388584' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'tuple_assignment_388584', int_389621)
    
    # Assigning a Num to a Name (line 591):
    int_389622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 23), 'int')
    # Assigning a type to the variable 'tuple_assignment_388585' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'tuple_assignment_388585', int_389622)
    
    # Assigning a Name to a Name (line 591):
    # Getting the type of 'tuple_assignment_388584' (line 591)
    tuple_assignment_388584_389623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'tuple_assignment_388584')
    # Assigning a type to the variable 'm_star' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'm_star', tuple_assignment_388584_389623)
    
    # Assigning a Name to a Name (line 591):
    # Getting the type of 'tuple_assignment_388585' (line 591)
    tuple_assignment_388585_389624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'tuple_assignment_388585')
    # Assigning a type to the variable 's' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 16), 's', tuple_assignment_388585_389624)
    # SSA branch for the else part of an if statement (line 590)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 593):
    
    # Assigning a Subscript to a Name (line 593):
    
    # Obtaining the type of the subscript
    int_389625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 8), 'int')
    
    # Call to _fragment_3_1(...): (line 593)
    # Processing the call arguments (line 593)
    # Getting the type of 'norm_info' (line 593)
    norm_info_389627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 34), 'norm_info', False)
    # Getting the type of 'n0' (line 593)
    n0_389628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 45), 'n0', False)
    # Getting the type of 'tol' (line 593)
    tol_389629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 49), 'tol', False)
    # Processing the call keyword arguments (line 593)
    # Getting the type of 'ell' (line 593)
    ell_389630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 58), 'ell', False)
    keyword_389631 = ell_389630
    kwargs_389632 = {'ell': keyword_389631}
    # Getting the type of '_fragment_3_1' (line 593)
    _fragment_3_1_389626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 20), '_fragment_3_1', False)
    # Calling _fragment_3_1(args, kwargs) (line 593)
    _fragment_3_1_call_result_389633 = invoke(stypy.reporting.localization.Localization(__file__, 593, 20), _fragment_3_1_389626, *[norm_info_389627, n0_389628, tol_389629], **kwargs_389632)
    
    # Obtaining the member '__getitem__' of a type (line 593)
    getitem___389634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 8), _fragment_3_1_call_result_389633, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 593)
    subscript_call_result_389635 = invoke(stypy.reporting.localization.Localization(__file__, 593, 8), getitem___389634, int_389625)
    
    # Assigning a type to the variable 'tuple_var_assignment_388586' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_388586', subscript_call_result_389635)
    
    # Assigning a Subscript to a Name (line 593):
    
    # Obtaining the type of the subscript
    int_389636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 8), 'int')
    
    # Call to _fragment_3_1(...): (line 593)
    # Processing the call arguments (line 593)
    # Getting the type of 'norm_info' (line 593)
    norm_info_389638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 34), 'norm_info', False)
    # Getting the type of 'n0' (line 593)
    n0_389639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 45), 'n0', False)
    # Getting the type of 'tol' (line 593)
    tol_389640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 49), 'tol', False)
    # Processing the call keyword arguments (line 593)
    # Getting the type of 'ell' (line 593)
    ell_389641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 58), 'ell', False)
    keyword_389642 = ell_389641
    kwargs_389643 = {'ell': keyword_389642}
    # Getting the type of '_fragment_3_1' (line 593)
    _fragment_3_1_389637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 20), '_fragment_3_1', False)
    # Calling _fragment_3_1(args, kwargs) (line 593)
    _fragment_3_1_call_result_389644 = invoke(stypy.reporting.localization.Localization(__file__, 593, 20), _fragment_3_1_389637, *[norm_info_389638, n0_389639, tol_389640], **kwargs_389643)
    
    # Obtaining the member '__getitem__' of a type (line 593)
    getitem___389645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 8), _fragment_3_1_call_result_389644, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 593)
    subscript_call_result_389646 = invoke(stypy.reporting.localization.Localization(__file__, 593, 8), getitem___389645, int_389636)
    
    # Assigning a type to the variable 'tuple_var_assignment_388587' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_388587', subscript_call_result_389646)
    
    # Assigning a Name to a Name (line 593):
    # Getting the type of 'tuple_var_assignment_388586' (line 593)
    tuple_var_assignment_388586_389647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_388586')
    # Assigning a type to the variable 'm_star' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'm_star', tuple_var_assignment_388586_389647)
    
    # Assigning a Name to a Name (line 593):
    # Getting the type of 'tuple_var_assignment_388587' (line 593)
    tuple_var_assignment_388587_389648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_388587')
    # Assigning a type to the variable 's' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 16), 's', tuple_var_assignment_388587_389648)
    # SSA join for if statement (line 590)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 596):
    
    # Assigning a Call to a Subscript (line 596):
    
    # Call to _expm_multiply_simple_core(...): (line 596)
    # Processing the call arguments (line 596)
    # Getting the type of 'A' (line 596)
    A_389650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 38), 'A', False)
    # Getting the type of 'B' (line 596)
    B_389651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 41), 'B', False)
    # Getting the type of 't_0' (line 596)
    t_0_389652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 44), 't_0', False)
    # Getting the type of 'mu' (line 596)
    mu_389653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 49), 'mu', False)
    # Getting the type of 'm_star' (line 596)
    m_star_389654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 53), 'm_star', False)
    # Getting the type of 's' (line 596)
    s_389655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 61), 's', False)
    # Processing the call keyword arguments (line 596)
    kwargs_389656 = {}
    # Getting the type of '_expm_multiply_simple_core' (line 596)
    _expm_multiply_simple_core_389649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 11), '_expm_multiply_simple_core', False)
    # Calling _expm_multiply_simple_core(args, kwargs) (line 596)
    _expm_multiply_simple_core_call_result_389657 = invoke(stypy.reporting.localization.Localization(__file__, 596, 11), _expm_multiply_simple_core_389649, *[A_389650, B_389651, t_0_389652, mu_389653, m_star_389654, s_389655], **kwargs_389656)
    
    # Getting the type of 'X' (line 596)
    X_389658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'X')
    int_389659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 6), 'int')
    # Storing an element on a container (line 596)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 4), X_389658, (int_389659, _expm_multiply_simple_core_call_result_389657))
    
    
    # Getting the type of 'q' (line 599)
    q_389660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 7), 'q')
    # Getting the type of 's' (line 599)
    s_389661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 12), 's')
    # Applying the binary operator '<=' (line 599)
    result_le_389662 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 7), '<=', q_389660, s_389661)
    
    # Testing the type of an if condition (line 599)
    if_condition_389663 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 599, 4), result_le_389662)
    # Assigning a type to the variable 'if_condition_389663' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'if_condition_389663', if_condition_389663)
    # SSA begins for if statement (line 599)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'status_only' (line 600)
    status_only_389664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 11), 'status_only')
    # Testing the type of an if condition (line 600)
    if_condition_389665 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 600, 8), status_only_389664)
    # Assigning a type to the variable 'if_condition_389665' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'if_condition_389665', if_condition_389665)
    # SSA begins for if statement (line 600)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_389666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 19), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'stypy_return_type', int_389666)
    # SSA branch for the else part of an if statement (line 600)
    module_type_store.open_ssa_branch('else')
    
    # Call to _expm_multiply_interval_core_0(...): (line 603)
    # Processing the call arguments (line 603)
    # Getting the type of 'A' (line 603)
    A_389668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 50), 'A', False)
    # Getting the type of 'X' (line 603)
    X_389669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 53), 'X', False)
    # Getting the type of 'h' (line 604)
    h_389670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 20), 'h', False)
    # Getting the type of 'mu' (line 604)
    mu_389671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 23), 'mu', False)
    # Getting the type of 'q' (line 604)
    q_389672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 27), 'q', False)
    # Getting the type of 'norm_info' (line 604)
    norm_info_389673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 30), 'norm_info', False)
    # Getting the type of 'tol' (line 604)
    tol_389674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 41), 'tol', False)
    # Getting the type of 'ell' (line 604)
    ell_389675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 46), 'ell', False)
    # Getting the type of 'n0' (line 604)
    n0_389676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 50), 'n0', False)
    # Processing the call keyword arguments (line 603)
    kwargs_389677 = {}
    # Getting the type of '_expm_multiply_interval_core_0' (line 603)
    _expm_multiply_interval_core_0_389667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 19), '_expm_multiply_interval_core_0', False)
    # Calling _expm_multiply_interval_core_0(args, kwargs) (line 603)
    _expm_multiply_interval_core_0_call_result_389678 = invoke(stypy.reporting.localization.Localization(__file__, 603, 19), _expm_multiply_interval_core_0_389667, *[A_389668, X_389669, h_389670, mu_389671, q_389672, norm_info_389673, tol_389674, ell_389675, n0_389676], **kwargs_389677)
    
    # Assigning a type to the variable 'stypy_return_type' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), 'stypy_return_type', _expm_multiply_interval_core_0_call_result_389678)
    # SSA join for if statement (line 600)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 599)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'q' (line 605)
    q_389679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 14), 'q')
    # Getting the type of 's' (line 605)
    s_389680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 18), 's')
    # Applying the binary operator '%' (line 605)
    result_mod_389681 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 14), '%', q_389679, s_389680)
    
    # Applying the 'not' unary operator (line 605)
    result_not__389682 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 9), 'not', result_mod_389681)
    
    # Testing the type of an if condition (line 605)
    if_condition_389683 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 605, 9), result_not__389682)
    # Assigning a type to the variable 'if_condition_389683' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 9), 'if_condition_389683', if_condition_389683)
    # SSA begins for if statement (line 605)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'status_only' (line 606)
    status_only_389684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 11), 'status_only')
    # Testing the type of an if condition (line 606)
    if_condition_389685 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 606, 8), status_only_389684)
    # Assigning a type to the variable 'if_condition_389685' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'if_condition_389685', if_condition_389685)
    # SSA begins for if statement (line 606)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_389686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 19), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'stypy_return_type', int_389686)
    # SSA branch for the else part of an if statement (line 606)
    module_type_store.open_ssa_branch('else')
    
    # Call to _expm_multiply_interval_core_1(...): (line 609)
    # Processing the call arguments (line 609)
    # Getting the type of 'A' (line 609)
    A_389688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 50), 'A', False)
    # Getting the type of 'X' (line 609)
    X_389689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 53), 'X', False)
    # Getting the type of 'h' (line 610)
    h_389690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 20), 'h', False)
    # Getting the type of 'mu' (line 610)
    mu_389691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 23), 'mu', False)
    # Getting the type of 'm_star' (line 610)
    m_star_389692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 27), 'm_star', False)
    # Getting the type of 's' (line 610)
    s_389693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 35), 's', False)
    # Getting the type of 'q' (line 610)
    q_389694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 38), 'q', False)
    # Getting the type of 'tol' (line 610)
    tol_389695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 41), 'tol', False)
    # Processing the call keyword arguments (line 609)
    kwargs_389696 = {}
    # Getting the type of '_expm_multiply_interval_core_1' (line 609)
    _expm_multiply_interval_core_1_389687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 19), '_expm_multiply_interval_core_1', False)
    # Calling _expm_multiply_interval_core_1(args, kwargs) (line 609)
    _expm_multiply_interval_core_1_call_result_389697 = invoke(stypy.reporting.localization.Localization(__file__, 609, 19), _expm_multiply_interval_core_1_389687, *[A_389688, X_389689, h_389690, mu_389691, m_star_389692, s_389693, q_389694, tol_389695], **kwargs_389696)
    
    # Assigning a type to the variable 'stypy_return_type' (line 609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'stypy_return_type', _expm_multiply_interval_core_1_call_result_389697)
    # SSA join for if statement (line 606)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 605)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'q' (line 611)
    q_389698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 10), 'q')
    # Getting the type of 's' (line 611)
    s_389699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 14), 's')
    # Applying the binary operator '%' (line 611)
    result_mod_389700 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 10), '%', q_389698, s_389699)
    
    # Testing the type of an if condition (line 611)
    if_condition_389701 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 611, 9), result_mod_389700)
    # Assigning a type to the variable 'if_condition_389701' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 9), 'if_condition_389701', if_condition_389701)
    # SSA begins for if statement (line 611)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'status_only' (line 612)
    status_only_389702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 11), 'status_only')
    # Testing the type of an if condition (line 612)
    if_condition_389703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 612, 8), status_only_389702)
    # Assigning a type to the variable 'if_condition_389703' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'if_condition_389703', if_condition_389703)
    # SSA begins for if statement (line 612)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_389704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 19), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'stypy_return_type', int_389704)
    # SSA branch for the else part of an if statement (line 612)
    module_type_store.open_ssa_branch('else')
    
    # Call to _expm_multiply_interval_core_2(...): (line 615)
    # Processing the call arguments (line 615)
    # Getting the type of 'A' (line 615)
    A_389706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 50), 'A', False)
    # Getting the type of 'X' (line 615)
    X_389707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 53), 'X', False)
    # Getting the type of 'h' (line 616)
    h_389708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 20), 'h', False)
    # Getting the type of 'mu' (line 616)
    mu_389709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 23), 'mu', False)
    # Getting the type of 'm_star' (line 616)
    m_star_389710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 27), 'm_star', False)
    # Getting the type of 's' (line 616)
    s_389711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 35), 's', False)
    # Getting the type of 'q' (line 616)
    q_389712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 38), 'q', False)
    # Getting the type of 'tol' (line 616)
    tol_389713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 41), 'tol', False)
    # Processing the call keyword arguments (line 615)
    kwargs_389714 = {}
    # Getting the type of '_expm_multiply_interval_core_2' (line 615)
    _expm_multiply_interval_core_2_389705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 19), '_expm_multiply_interval_core_2', False)
    # Calling _expm_multiply_interval_core_2(args, kwargs) (line 615)
    _expm_multiply_interval_core_2_call_result_389715 = invoke(stypy.reporting.localization.Localization(__file__, 615, 19), _expm_multiply_interval_core_2_389705, *[A_389706, X_389707, h_389708, mu_389709, m_star_389710, s_389711, q_389712, tol_389713], **kwargs_389714)
    
    # Assigning a type to the variable 'stypy_return_type' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'stypy_return_type', _expm_multiply_interval_core_2_call_result_389715)
    # SSA join for if statement (line 612)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 611)
    module_type_store.open_ssa_branch('else')
    
    # Call to Exception(...): (line 618)
    # Processing the call arguments (line 618)
    str_389717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 24), 'str', 'internal error')
    # Processing the call keyword arguments (line 618)
    kwargs_389718 = {}
    # Getting the type of 'Exception' (line 618)
    Exception_389716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 618)
    Exception_call_result_389719 = invoke(stypy.reporting.localization.Localization(__file__, 618, 14), Exception_389716, *[str_389717], **kwargs_389718)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 618, 8), Exception_call_result_389719, 'raise parameter', BaseException)
    # SSA join for if statement (line 611)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 605)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 599)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_expm_multiply_interval(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_expm_multiply_interval' in the type store
    # Getting the type of 'stypy_return_type' (line 503)
    stypy_return_type_389720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_389720)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_expm_multiply_interval'
    return stypy_return_type_389720

# Assigning a type to the variable '_expm_multiply_interval' (line 503)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 0), '_expm_multiply_interval', _expm_multiply_interval)

@norecursion
def _expm_multiply_interval_core_0(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_expm_multiply_interval_core_0'
    module_type_store = module_type_store.open_function_context('_expm_multiply_interval_core_0', 621, 0, False)
    
    # Passed parameters checking function
    _expm_multiply_interval_core_0.stypy_localization = localization
    _expm_multiply_interval_core_0.stypy_type_of_self = None
    _expm_multiply_interval_core_0.stypy_type_store = module_type_store
    _expm_multiply_interval_core_0.stypy_function_name = '_expm_multiply_interval_core_0'
    _expm_multiply_interval_core_0.stypy_param_names_list = ['A', 'X', 'h', 'mu', 'q', 'norm_info', 'tol', 'ell', 'n0']
    _expm_multiply_interval_core_0.stypy_varargs_param_name = None
    _expm_multiply_interval_core_0.stypy_kwargs_param_name = None
    _expm_multiply_interval_core_0.stypy_call_defaults = defaults
    _expm_multiply_interval_core_0.stypy_call_varargs = varargs
    _expm_multiply_interval_core_0.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_expm_multiply_interval_core_0', ['A', 'X', 'h', 'mu', 'q', 'norm_info', 'tol', 'ell', 'n0'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_expm_multiply_interval_core_0', localization, ['A', 'X', 'h', 'mu', 'q', 'norm_info', 'tol', 'ell', 'n0'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_expm_multiply_interval_core_0(...)' code ##################

    str_389721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, (-1)), 'str', '\n    A helper function, for the case q <= s.\n    ')
    
    
    
    # Call to onenorm(...): (line 628)
    # Processing the call keyword arguments (line 628)
    kwargs_389724 = {}
    # Getting the type of 'norm_info' (line 628)
    norm_info_389722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 7), 'norm_info', False)
    # Obtaining the member 'onenorm' of a type (line 628)
    onenorm_389723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 7), norm_info_389722, 'onenorm')
    # Calling onenorm(args, kwargs) (line 628)
    onenorm_call_result_389725 = invoke(stypy.reporting.localization.Localization(__file__, 628, 7), onenorm_389723, *[], **kwargs_389724)
    
    int_389726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 30), 'int')
    # Applying the binary operator '==' (line 628)
    result_eq_389727 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 7), '==', onenorm_call_result_389725, int_389726)
    
    # Testing the type of an if condition (line 628)
    if_condition_389728 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 628, 4), result_eq_389727)
    # Assigning a type to the variable 'if_condition_389728' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'if_condition_389728', if_condition_389728)
    # SSA begins for if statement (line 628)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 629):
    
    # Assigning a Num to a Name (line 629):
    int_389729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 20), 'int')
    # Assigning a type to the variable 'tuple_assignment_388588' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'tuple_assignment_388588', int_389729)
    
    # Assigning a Num to a Name (line 629):
    int_389730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 23), 'int')
    # Assigning a type to the variable 'tuple_assignment_388589' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'tuple_assignment_388589', int_389730)
    
    # Assigning a Name to a Name (line 629):
    # Getting the type of 'tuple_assignment_388588' (line 629)
    tuple_assignment_388588_389731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'tuple_assignment_388588')
    # Assigning a type to the variable 'm_star' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'm_star', tuple_assignment_388588_389731)
    
    # Assigning a Name to a Name (line 629):
    # Getting the type of 'tuple_assignment_388589' (line 629)
    tuple_assignment_388589_389732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'tuple_assignment_388589')
    # Assigning a type to the variable 's' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 16), 's', tuple_assignment_388589_389732)
    # SSA branch for the else part of an if statement (line 628)
    module_type_store.open_ssa_branch('else')
    
    # Call to set_scale(...): (line 631)
    # Processing the call arguments (line 631)
    float_389735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 28), 'float')
    # Getting the type of 'q' (line 631)
    q_389736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 31), 'q', False)
    # Applying the binary operator 'div' (line 631)
    result_div_389737 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 28), 'div', float_389735, q_389736)
    
    # Processing the call keyword arguments (line 631)
    kwargs_389738 = {}
    # Getting the type of 'norm_info' (line 631)
    norm_info_389733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'norm_info', False)
    # Obtaining the member 'set_scale' of a type (line 631)
    set_scale_389734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 8), norm_info_389733, 'set_scale')
    # Calling set_scale(args, kwargs) (line 631)
    set_scale_call_result_389739 = invoke(stypy.reporting.localization.Localization(__file__, 631, 8), set_scale_389734, *[result_div_389737], **kwargs_389738)
    
    
    # Assigning a Call to a Tuple (line 632):
    
    # Assigning a Subscript to a Name (line 632):
    
    # Obtaining the type of the subscript
    int_389740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 8), 'int')
    
    # Call to _fragment_3_1(...): (line 632)
    # Processing the call arguments (line 632)
    # Getting the type of 'norm_info' (line 632)
    norm_info_389742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 34), 'norm_info', False)
    # Getting the type of 'n0' (line 632)
    n0_389743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 45), 'n0', False)
    # Getting the type of 'tol' (line 632)
    tol_389744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 49), 'tol', False)
    # Processing the call keyword arguments (line 632)
    # Getting the type of 'ell' (line 632)
    ell_389745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 58), 'ell', False)
    keyword_389746 = ell_389745
    kwargs_389747 = {'ell': keyword_389746}
    # Getting the type of '_fragment_3_1' (line 632)
    _fragment_3_1_389741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 20), '_fragment_3_1', False)
    # Calling _fragment_3_1(args, kwargs) (line 632)
    _fragment_3_1_call_result_389748 = invoke(stypy.reporting.localization.Localization(__file__, 632, 20), _fragment_3_1_389741, *[norm_info_389742, n0_389743, tol_389744], **kwargs_389747)
    
    # Obtaining the member '__getitem__' of a type (line 632)
    getitem___389749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 8), _fragment_3_1_call_result_389748, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 632)
    subscript_call_result_389750 = invoke(stypy.reporting.localization.Localization(__file__, 632, 8), getitem___389749, int_389740)
    
    # Assigning a type to the variable 'tuple_var_assignment_388590' (line 632)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'tuple_var_assignment_388590', subscript_call_result_389750)
    
    # Assigning a Subscript to a Name (line 632):
    
    # Obtaining the type of the subscript
    int_389751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 8), 'int')
    
    # Call to _fragment_3_1(...): (line 632)
    # Processing the call arguments (line 632)
    # Getting the type of 'norm_info' (line 632)
    norm_info_389753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 34), 'norm_info', False)
    # Getting the type of 'n0' (line 632)
    n0_389754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 45), 'n0', False)
    # Getting the type of 'tol' (line 632)
    tol_389755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 49), 'tol', False)
    # Processing the call keyword arguments (line 632)
    # Getting the type of 'ell' (line 632)
    ell_389756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 58), 'ell', False)
    keyword_389757 = ell_389756
    kwargs_389758 = {'ell': keyword_389757}
    # Getting the type of '_fragment_3_1' (line 632)
    _fragment_3_1_389752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 20), '_fragment_3_1', False)
    # Calling _fragment_3_1(args, kwargs) (line 632)
    _fragment_3_1_call_result_389759 = invoke(stypy.reporting.localization.Localization(__file__, 632, 20), _fragment_3_1_389752, *[norm_info_389753, n0_389754, tol_389755], **kwargs_389758)
    
    # Obtaining the member '__getitem__' of a type (line 632)
    getitem___389760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 8), _fragment_3_1_call_result_389759, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 632)
    subscript_call_result_389761 = invoke(stypy.reporting.localization.Localization(__file__, 632, 8), getitem___389760, int_389751)
    
    # Assigning a type to the variable 'tuple_var_assignment_388591' (line 632)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'tuple_var_assignment_388591', subscript_call_result_389761)
    
    # Assigning a Name to a Name (line 632):
    # Getting the type of 'tuple_var_assignment_388590' (line 632)
    tuple_var_assignment_388590_389762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'tuple_var_assignment_388590')
    # Assigning a type to the variable 'm_star' (line 632)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'm_star', tuple_var_assignment_388590_389762)
    
    # Assigning a Name to a Name (line 632):
    # Getting the type of 'tuple_var_assignment_388591' (line 632)
    tuple_var_assignment_388591_389763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'tuple_var_assignment_388591')
    # Assigning a type to the variable 's' (line 632)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 16), 's', tuple_var_assignment_388591_389763)
    
    # Call to set_scale(...): (line 633)
    # Processing the call arguments (line 633)
    int_389766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 28), 'int')
    # Processing the call keyword arguments (line 633)
    kwargs_389767 = {}
    # Getting the type of 'norm_info' (line 633)
    norm_info_389764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'norm_info', False)
    # Obtaining the member 'set_scale' of a type (line 633)
    set_scale_389765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 8), norm_info_389764, 'set_scale')
    # Calling set_scale(args, kwargs) (line 633)
    set_scale_call_result_389768 = invoke(stypy.reporting.localization.Localization(__file__, 633, 8), set_scale_389765, *[int_389766], **kwargs_389767)
    
    # SSA join for if statement (line 628)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 635)
    # Processing the call arguments (line 635)
    # Getting the type of 'q' (line 635)
    q_389770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 19), 'q', False)
    # Processing the call keyword arguments (line 635)
    kwargs_389771 = {}
    # Getting the type of 'range' (line 635)
    range_389769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 13), 'range', False)
    # Calling range(args, kwargs) (line 635)
    range_call_result_389772 = invoke(stypy.reporting.localization.Localization(__file__, 635, 13), range_389769, *[q_389770], **kwargs_389771)
    
    # Testing the type of a for loop iterable (line 635)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 635, 4), range_call_result_389772)
    # Getting the type of the for loop variable (line 635)
    for_loop_var_389773 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 635, 4), range_call_result_389772)
    # Assigning a type to the variable 'k' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'k', for_loop_var_389773)
    # SSA begins for a for statement (line 635)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 636):
    
    # Assigning a Call to a Subscript (line 636):
    
    # Call to _expm_multiply_simple_core(...): (line 636)
    # Processing the call arguments (line 636)
    # Getting the type of 'A' (line 636)
    A_389775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 44), 'A', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 636)
    k_389776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 49), 'k', False)
    # Getting the type of 'X' (line 636)
    X_389777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 47), 'X', False)
    # Obtaining the member '__getitem__' of a type (line 636)
    getitem___389778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 47), X_389777, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 636)
    subscript_call_result_389779 = invoke(stypy.reporting.localization.Localization(__file__, 636, 47), getitem___389778, k_389776)
    
    # Getting the type of 'h' (line 636)
    h_389780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 53), 'h', False)
    # Getting the type of 'mu' (line 636)
    mu_389781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 56), 'mu', False)
    # Getting the type of 'm_star' (line 636)
    m_star_389782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 60), 'm_star', False)
    # Getting the type of 's' (line 636)
    s_389783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 68), 's', False)
    # Processing the call keyword arguments (line 636)
    kwargs_389784 = {}
    # Getting the type of '_expm_multiply_simple_core' (line 636)
    _expm_multiply_simple_core_389774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 17), '_expm_multiply_simple_core', False)
    # Calling _expm_multiply_simple_core(args, kwargs) (line 636)
    _expm_multiply_simple_core_call_result_389785 = invoke(stypy.reporting.localization.Localization(__file__, 636, 17), _expm_multiply_simple_core_389774, *[A_389775, subscript_call_result_389779, h_389780, mu_389781, m_star_389782, s_389783], **kwargs_389784)
    
    # Getting the type of 'X' (line 636)
    X_389786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 8), 'X')
    # Getting the type of 'k' (line 636)
    k_389787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 10), 'k')
    int_389788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 12), 'int')
    # Applying the binary operator '+' (line 636)
    result_add_389789 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 10), '+', k_389787, int_389788)
    
    # Storing an element on a container (line 636)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 8), X_389786, (result_add_389789, _expm_multiply_simple_core_call_result_389785))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 637)
    tuple_389790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 637)
    # Adding element type (line 637)
    # Getting the type of 'X' (line 637)
    X_389791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 11), 'X')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 637, 11), tuple_389790, X_389791)
    # Adding element type (line 637)
    int_389792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 637, 11), tuple_389790, int_389792)
    
    # Assigning a type to the variable 'stypy_return_type' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'stypy_return_type', tuple_389790)
    
    # ################# End of '_expm_multiply_interval_core_0(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_expm_multiply_interval_core_0' in the type store
    # Getting the type of 'stypy_return_type' (line 621)
    stypy_return_type_389793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_389793)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_expm_multiply_interval_core_0'
    return stypy_return_type_389793

# Assigning a type to the variable '_expm_multiply_interval_core_0' (line 621)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 0), '_expm_multiply_interval_core_0', _expm_multiply_interval_core_0)

@norecursion
def _expm_multiply_interval_core_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_expm_multiply_interval_core_1'
    module_type_store = module_type_store.open_function_context('_expm_multiply_interval_core_1', 640, 0, False)
    
    # Passed parameters checking function
    _expm_multiply_interval_core_1.stypy_localization = localization
    _expm_multiply_interval_core_1.stypy_type_of_self = None
    _expm_multiply_interval_core_1.stypy_type_store = module_type_store
    _expm_multiply_interval_core_1.stypy_function_name = '_expm_multiply_interval_core_1'
    _expm_multiply_interval_core_1.stypy_param_names_list = ['A', 'X', 'h', 'mu', 'm_star', 's', 'q', 'tol']
    _expm_multiply_interval_core_1.stypy_varargs_param_name = None
    _expm_multiply_interval_core_1.stypy_kwargs_param_name = None
    _expm_multiply_interval_core_1.stypy_call_defaults = defaults
    _expm_multiply_interval_core_1.stypy_call_varargs = varargs
    _expm_multiply_interval_core_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_expm_multiply_interval_core_1', ['A', 'X', 'h', 'mu', 'm_star', 's', 'q', 'tol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_expm_multiply_interval_core_1', localization, ['A', 'X', 'h', 'mu', 'm_star', 's', 'q', 'tol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_expm_multiply_interval_core_1(...)' code ##################

    str_389794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, (-1)), 'str', '\n    A helper function, for the case q > s and q % s == 0.\n    ')
    
    # Assigning a BinOp to a Name (line 644):
    
    # Assigning a BinOp to a Name (line 644):
    # Getting the type of 'q' (line 644)
    q_389795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'q')
    # Getting the type of 's' (line 644)
    s_389796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 13), 's')
    # Applying the binary operator '//' (line 644)
    result_floordiv_389797 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 8), '//', q_389795, s_389796)
    
    # Assigning a type to the variable 'd' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'd', result_floordiv_389797)
    
    # Assigning a Subscript to a Name (line 645):
    
    # Assigning a Subscript to a Name (line 645):
    
    # Obtaining the type of the subscript
    int_389798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 26), 'int')
    slice_389799 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 645, 18), int_389798, None, None)
    # Getting the type of 'X' (line 645)
    X_389800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 18), 'X')
    # Obtaining the member 'shape' of a type (line 645)
    shape_389801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 18), X_389800, 'shape')
    # Obtaining the member '__getitem__' of a type (line 645)
    getitem___389802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 18), shape_389801, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 645)
    subscript_call_result_389803 = invoke(stypy.reporting.localization.Localization(__file__, 645, 18), getitem___389802, slice_389799)
    
    # Assigning a type to the variable 'input_shape' (line 645)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 4), 'input_shape', subscript_call_result_389803)
    
    # Assigning a BinOp to a Name (line 646):
    
    # Assigning a BinOp to a Name (line 646):
    
    # Obtaining an instance of the builtin type 'tuple' (line 646)
    tuple_389804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 646)
    # Adding element type (line 646)
    # Getting the type of 'm_star' (line 646)
    m_star_389805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 15), 'm_star')
    int_389806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 24), 'int')
    # Applying the binary operator '+' (line 646)
    result_add_389807 = python_operator(stypy.reporting.localization.Localization(__file__, 646, 15), '+', m_star_389805, int_389806)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 646, 15), tuple_389804, result_add_389807)
    
    # Getting the type of 'input_shape' (line 646)
    input_shape_389808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 31), 'input_shape')
    # Applying the binary operator '+' (line 646)
    result_add_389809 = python_operator(stypy.reporting.localization.Localization(__file__, 646, 14), '+', tuple_389804, input_shape_389808)
    
    # Assigning a type to the variable 'K_shape' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 4), 'K_shape', result_add_389809)
    
    # Assigning a Call to a Name (line 647):
    
    # Assigning a Call to a Name (line 647):
    
    # Call to empty(...): (line 647)
    # Processing the call arguments (line 647)
    # Getting the type of 'K_shape' (line 647)
    K_shape_389812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 17), 'K_shape', False)
    # Processing the call keyword arguments (line 647)
    # Getting the type of 'X' (line 647)
    X_389813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 32), 'X', False)
    # Obtaining the member 'dtype' of a type (line 647)
    dtype_389814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 32), X_389813, 'dtype')
    keyword_389815 = dtype_389814
    kwargs_389816 = {'dtype': keyword_389815}
    # Getting the type of 'np' (line 647)
    np_389810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 647)
    empty_389811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 8), np_389810, 'empty')
    # Calling empty(args, kwargs) (line 647)
    empty_call_result_389817 = invoke(stypy.reporting.localization.Localization(__file__, 647, 8), empty_389811, *[K_shape_389812], **kwargs_389816)
    
    # Assigning a type to the variable 'K' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 4), 'K', empty_call_result_389817)
    
    
    # Call to range(...): (line 648)
    # Processing the call arguments (line 648)
    # Getting the type of 's' (line 648)
    s_389819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 19), 's', False)
    # Processing the call keyword arguments (line 648)
    kwargs_389820 = {}
    # Getting the type of 'range' (line 648)
    range_389818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 13), 'range', False)
    # Calling range(args, kwargs) (line 648)
    range_call_result_389821 = invoke(stypy.reporting.localization.Localization(__file__, 648, 13), range_389818, *[s_389819], **kwargs_389820)
    
    # Testing the type of a for loop iterable (line 648)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 648, 4), range_call_result_389821)
    # Getting the type of the for loop variable (line 648)
    for_loop_var_389822 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 648, 4), range_call_result_389821)
    # Assigning a type to the variable 'i' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'i', for_loop_var_389822)
    # SSA begins for a for statement (line 648)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 649):
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 649)
    i_389823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 14), 'i')
    # Getting the type of 'd' (line 649)
    d_389824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 16), 'd')
    # Applying the binary operator '*' (line 649)
    result_mul_389825 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 14), '*', i_389823, d_389824)
    
    # Getting the type of 'X' (line 649)
    X_389826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 12), 'X')
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___389827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 12), X_389826, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_389828 = invoke(stypy.reporting.localization.Localization(__file__, 649, 12), getitem___389827, result_mul_389825)
    
    # Assigning a type to the variable 'Z' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'Z', subscript_call_result_389828)
    
    # Assigning a Name to a Subscript (line 650):
    
    # Assigning a Name to a Subscript (line 650):
    # Getting the type of 'Z' (line 650)
    Z_389829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 15), 'Z')
    # Getting the type of 'K' (line 650)
    K_389830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'K')
    int_389831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 10), 'int')
    # Storing an element on a container (line 650)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 8), K_389830, (int_389831, Z_389829))
    
    # Assigning a Num to a Name (line 651):
    
    # Assigning a Num to a Name (line 651):
    int_389832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 17), 'int')
    # Assigning a type to the variable 'high_p' (line 651)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 8), 'high_p', int_389832)
    
    
    # Call to range(...): (line 652)
    # Processing the call arguments (line 652)
    int_389834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 23), 'int')
    # Getting the type of 'd' (line 652)
    d_389835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 26), 'd', False)
    int_389836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 28), 'int')
    # Applying the binary operator '+' (line 652)
    result_add_389837 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 26), '+', d_389835, int_389836)
    
    # Processing the call keyword arguments (line 652)
    kwargs_389838 = {}
    # Getting the type of 'range' (line 652)
    range_389833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 17), 'range', False)
    # Calling range(args, kwargs) (line 652)
    range_call_result_389839 = invoke(stypy.reporting.localization.Localization(__file__, 652, 17), range_389833, *[int_389834, result_add_389837], **kwargs_389838)
    
    # Testing the type of a for loop iterable (line 652)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 652, 8), range_call_result_389839)
    # Getting the type of the for loop variable (line 652)
    for_loop_var_389840 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 652, 8), range_call_result_389839)
    # Assigning a type to the variable 'k' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'k', for_loop_var_389840)
    # SSA begins for a for statement (line 652)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 653):
    
    # Assigning a Subscript to a Name (line 653):
    
    # Obtaining the type of the subscript
    int_389841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 18), 'int')
    # Getting the type of 'K' (line 653)
    K_389842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 16), 'K')
    # Obtaining the member '__getitem__' of a type (line 653)
    getitem___389843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 16), K_389842, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 653)
    subscript_call_result_389844 = invoke(stypy.reporting.localization.Localization(__file__, 653, 16), getitem___389843, int_389841)
    
    # Assigning a type to the variable 'F' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 12), 'F', subscript_call_result_389844)
    
    # Assigning a Call to a Name (line 654):
    
    # Assigning a Call to a Name (line 654):
    
    # Call to _exact_inf_norm(...): (line 654)
    # Processing the call arguments (line 654)
    # Getting the type of 'F' (line 654)
    F_389846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 33), 'F', False)
    # Processing the call keyword arguments (line 654)
    kwargs_389847 = {}
    # Getting the type of '_exact_inf_norm' (line 654)
    _exact_inf_norm_389845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 17), '_exact_inf_norm', False)
    # Calling _exact_inf_norm(args, kwargs) (line 654)
    _exact_inf_norm_call_result_389848 = invoke(stypy.reporting.localization.Localization(__file__, 654, 17), _exact_inf_norm_389845, *[F_389846], **kwargs_389847)
    
    # Assigning a type to the variable 'c1' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'c1', _exact_inf_norm_call_result_389848)
    
    
    # Call to range(...): (line 655)
    # Processing the call arguments (line 655)
    int_389850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 27), 'int')
    # Getting the type of 'm_star' (line 655)
    m_star_389851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 30), 'm_star', False)
    int_389852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 37), 'int')
    # Applying the binary operator '+' (line 655)
    result_add_389853 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 30), '+', m_star_389851, int_389852)
    
    # Processing the call keyword arguments (line 655)
    kwargs_389854 = {}
    # Getting the type of 'range' (line 655)
    range_389849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 21), 'range', False)
    # Calling range(args, kwargs) (line 655)
    range_call_result_389855 = invoke(stypy.reporting.localization.Localization(__file__, 655, 21), range_389849, *[int_389850, result_add_389853], **kwargs_389854)
    
    # Testing the type of a for loop iterable (line 655)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 655, 12), range_call_result_389855)
    # Getting the type of the for loop variable (line 655)
    for_loop_var_389856 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 655, 12), range_call_result_389855)
    # Assigning a type to the variable 'p' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 12), 'p', for_loop_var_389856)
    # SSA begins for a for statement (line 655)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'p' (line 656)
    p_389857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 19), 'p')
    # Getting the type of 'high_p' (line 656)
    high_p_389858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 23), 'high_p')
    # Applying the binary operator '>' (line 656)
    result_gt_389859 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 19), '>', p_389857, high_p_389858)
    
    # Testing the type of an if condition (line 656)
    if_condition_389860 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 656, 16), result_gt_389859)
    # Assigning a type to the variable 'if_condition_389860' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 16), 'if_condition_389860', if_condition_389860)
    # SSA begins for if statement (line 656)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 657):
    
    # Assigning a BinOp to a Subscript (line 657):
    # Getting the type of 'h' (line 657)
    h_389861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 27), 'h')
    
    # Call to dot(...): (line 657)
    # Processing the call arguments (line 657)
    
    # Obtaining the type of the subscript
    # Getting the type of 'p' (line 657)
    p_389864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 39), 'p', False)
    int_389865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 41), 'int')
    # Applying the binary operator '-' (line 657)
    result_sub_389866 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 39), '-', p_389864, int_389865)
    
    # Getting the type of 'K' (line 657)
    K_389867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 37), 'K', False)
    # Obtaining the member '__getitem__' of a type (line 657)
    getitem___389868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 37), K_389867, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 657)
    subscript_call_result_389869 = invoke(stypy.reporting.localization.Localization(__file__, 657, 37), getitem___389868, result_sub_389866)
    
    # Processing the call keyword arguments (line 657)
    kwargs_389870 = {}
    # Getting the type of 'A' (line 657)
    A_389862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 31), 'A', False)
    # Obtaining the member 'dot' of a type (line 657)
    dot_389863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 31), A_389862, 'dot')
    # Calling dot(args, kwargs) (line 657)
    dot_call_result_389871 = invoke(stypy.reporting.localization.Localization(__file__, 657, 31), dot_389863, *[subscript_call_result_389869], **kwargs_389870)
    
    # Applying the binary operator '*' (line 657)
    result_mul_389872 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 27), '*', h_389861, dot_call_result_389871)
    
    
    # Call to float(...): (line 657)
    # Processing the call arguments (line 657)
    # Getting the type of 'p' (line 657)
    p_389874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 53), 'p', False)
    # Processing the call keyword arguments (line 657)
    kwargs_389875 = {}
    # Getting the type of 'float' (line 657)
    float_389873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 47), 'float', False)
    # Calling float(args, kwargs) (line 657)
    float_call_result_389876 = invoke(stypy.reporting.localization.Localization(__file__, 657, 47), float_389873, *[p_389874], **kwargs_389875)
    
    # Applying the binary operator 'div' (line 657)
    result_div_389877 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 45), 'div', result_mul_389872, float_call_result_389876)
    
    # Getting the type of 'K' (line 657)
    K_389878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 20), 'K')
    # Getting the type of 'p' (line 657)
    p_389879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 22), 'p')
    # Storing an element on a container (line 657)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 657, 20), K_389878, (p_389879, result_div_389877))
    # SSA join for if statement (line 656)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 658):
    
    # Assigning a Call to a Name (line 658):
    
    # Call to float(...): (line 658)
    # Processing the call arguments (line 658)
    
    # Call to pow(...): (line 658)
    # Processing the call arguments (line 658)
    # Getting the type of 'k' (line 658)
    k_389882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 34), 'k', False)
    # Getting the type of 'p' (line 658)
    p_389883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 37), 'p', False)
    # Processing the call keyword arguments (line 658)
    kwargs_389884 = {}
    # Getting the type of 'pow' (line 658)
    pow_389881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 30), 'pow', False)
    # Calling pow(args, kwargs) (line 658)
    pow_call_result_389885 = invoke(stypy.reporting.localization.Localization(__file__, 658, 30), pow_389881, *[k_389882, p_389883], **kwargs_389884)
    
    # Processing the call keyword arguments (line 658)
    kwargs_389886 = {}
    # Getting the type of 'float' (line 658)
    float_389880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 24), 'float', False)
    # Calling float(args, kwargs) (line 658)
    float_call_result_389887 = invoke(stypy.reporting.localization.Localization(__file__, 658, 24), float_389880, *[pow_call_result_389885], **kwargs_389886)
    
    # Assigning a type to the variable 'coeff' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 16), 'coeff', float_call_result_389887)
    
    # Assigning a BinOp to a Name (line 659):
    
    # Assigning a BinOp to a Name (line 659):
    # Getting the type of 'F' (line 659)
    F_389888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 20), 'F')
    # Getting the type of 'coeff' (line 659)
    coeff_389889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 24), 'coeff')
    
    # Obtaining the type of the subscript
    # Getting the type of 'p' (line 659)
    p_389890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 34), 'p')
    # Getting the type of 'K' (line 659)
    K_389891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 32), 'K')
    # Obtaining the member '__getitem__' of a type (line 659)
    getitem___389892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 32), K_389891, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 659)
    subscript_call_result_389893 = invoke(stypy.reporting.localization.Localization(__file__, 659, 32), getitem___389892, p_389890)
    
    # Applying the binary operator '*' (line 659)
    result_mul_389894 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 24), '*', coeff_389889, subscript_call_result_389893)
    
    # Applying the binary operator '+' (line 659)
    result_add_389895 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 20), '+', F_389888, result_mul_389894)
    
    # Assigning a type to the variable 'F' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 16), 'F', result_add_389895)
    
    # Assigning a Call to a Name (line 660):
    
    # Assigning a Call to a Name (line 660):
    
    # Call to _exact_inf_norm(...): (line 660)
    # Processing the call arguments (line 660)
    
    # Obtaining the type of the subscript
    # Getting the type of 'p' (line 660)
    p_389897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 51), 'p', False)
    # Getting the type of 'K' (line 660)
    K_389898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 49), 'K', False)
    # Obtaining the member '__getitem__' of a type (line 660)
    getitem___389899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 49), K_389898, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 660)
    subscript_call_result_389900 = invoke(stypy.reporting.localization.Localization(__file__, 660, 49), getitem___389899, p_389897)
    
    # Processing the call keyword arguments (line 660)
    kwargs_389901 = {}
    # Getting the type of '_exact_inf_norm' (line 660)
    _exact_inf_norm_389896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 33), '_exact_inf_norm', False)
    # Calling _exact_inf_norm(args, kwargs) (line 660)
    _exact_inf_norm_call_result_389902 = invoke(stypy.reporting.localization.Localization(__file__, 660, 33), _exact_inf_norm_389896, *[subscript_call_result_389900], **kwargs_389901)
    
    # Assigning a type to the variable 'inf_norm_K_p_1' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 16), 'inf_norm_K_p_1', _exact_inf_norm_call_result_389902)
    
    # Assigning a BinOp to a Name (line 661):
    
    # Assigning a BinOp to a Name (line 661):
    # Getting the type of 'coeff' (line 661)
    coeff_389903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 21), 'coeff')
    # Getting the type of 'inf_norm_K_p_1' (line 661)
    inf_norm_K_p_1_389904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 29), 'inf_norm_K_p_1')
    # Applying the binary operator '*' (line 661)
    result_mul_389905 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 21), '*', coeff_389903, inf_norm_K_p_1_389904)
    
    # Assigning a type to the variable 'c2' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'c2', result_mul_389905)
    
    
    # Getting the type of 'c1' (line 662)
    c1_389906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 19), 'c1')
    # Getting the type of 'c2' (line 662)
    c2_389907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 24), 'c2')
    # Applying the binary operator '+' (line 662)
    result_add_389908 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 19), '+', c1_389906, c2_389907)
    
    # Getting the type of 'tol' (line 662)
    tol_389909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 30), 'tol')
    
    # Call to _exact_inf_norm(...): (line 662)
    # Processing the call arguments (line 662)
    # Getting the type of 'F' (line 662)
    F_389911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 52), 'F', False)
    # Processing the call keyword arguments (line 662)
    kwargs_389912 = {}
    # Getting the type of '_exact_inf_norm' (line 662)
    _exact_inf_norm_389910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 36), '_exact_inf_norm', False)
    # Calling _exact_inf_norm(args, kwargs) (line 662)
    _exact_inf_norm_call_result_389913 = invoke(stypy.reporting.localization.Localization(__file__, 662, 36), _exact_inf_norm_389910, *[F_389911], **kwargs_389912)
    
    # Applying the binary operator '*' (line 662)
    result_mul_389914 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 30), '*', tol_389909, _exact_inf_norm_call_result_389913)
    
    # Applying the binary operator '<=' (line 662)
    result_le_389915 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 19), '<=', result_add_389908, result_mul_389914)
    
    # Testing the type of an if condition (line 662)
    if_condition_389916 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 662, 16), result_le_389915)
    # Assigning a type to the variable 'if_condition_389916' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'if_condition_389916', if_condition_389916)
    # SSA begins for if statement (line 662)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 662)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 664):
    
    # Assigning a Name to a Name (line 664):
    # Getting the type of 'c2' (line 664)
    c2_389917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 21), 'c2')
    # Assigning a type to the variable 'c1' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 16), 'c1', c2_389917)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 665):
    
    # Assigning a BinOp to a Subscript (line 665):
    
    # Call to exp(...): (line 665)
    # Processing the call arguments (line 665)
    # Getting the type of 'k' (line 665)
    k_389920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 32), 'k', False)
    # Getting the type of 'h' (line 665)
    h_389921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 34), 'h', False)
    # Applying the binary operator '*' (line 665)
    result_mul_389922 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 32), '*', k_389920, h_389921)
    
    # Getting the type of 'mu' (line 665)
    mu_389923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 36), 'mu', False)
    # Applying the binary operator '*' (line 665)
    result_mul_389924 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 35), '*', result_mul_389922, mu_389923)
    
    # Processing the call keyword arguments (line 665)
    kwargs_389925 = {}
    # Getting the type of 'np' (line 665)
    np_389918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 25), 'np', False)
    # Obtaining the member 'exp' of a type (line 665)
    exp_389919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 25), np_389918, 'exp')
    # Calling exp(args, kwargs) (line 665)
    exp_call_result_389926 = invoke(stypy.reporting.localization.Localization(__file__, 665, 25), exp_389919, *[result_mul_389924], **kwargs_389925)
    
    # Getting the type of 'F' (line 665)
    F_389927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 42), 'F')
    # Applying the binary operator '*' (line 665)
    result_mul_389928 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 25), '*', exp_call_result_389926, F_389927)
    
    # Getting the type of 'X' (line 665)
    X_389929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'X')
    # Getting the type of 'k' (line 665)
    k_389930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 14), 'k')
    # Getting the type of 'i' (line 665)
    i_389931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 18), 'i')
    # Getting the type of 'd' (line 665)
    d_389932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 20), 'd')
    # Applying the binary operator '*' (line 665)
    result_mul_389933 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 18), '*', i_389931, d_389932)
    
    # Applying the binary operator '+' (line 665)
    result_add_389934 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 14), '+', k_389930, result_mul_389933)
    
    # Storing an element on a container (line 665)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 12), X_389929, (result_add_389934, result_mul_389928))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 666)
    tuple_389935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 666)
    # Adding element type (line 666)
    # Getting the type of 'X' (line 666)
    X_389936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 11), 'X')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 11), tuple_389935, X_389936)
    # Adding element type (line 666)
    int_389937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 11), tuple_389935, int_389937)
    
    # Assigning a type to the variable 'stypy_return_type' (line 666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 4), 'stypy_return_type', tuple_389935)
    
    # ################# End of '_expm_multiply_interval_core_1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_expm_multiply_interval_core_1' in the type store
    # Getting the type of 'stypy_return_type' (line 640)
    stypy_return_type_389938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_389938)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_expm_multiply_interval_core_1'
    return stypy_return_type_389938

# Assigning a type to the variable '_expm_multiply_interval_core_1' (line 640)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 0), '_expm_multiply_interval_core_1', _expm_multiply_interval_core_1)

@norecursion
def _expm_multiply_interval_core_2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_expm_multiply_interval_core_2'
    module_type_store = module_type_store.open_function_context('_expm_multiply_interval_core_2', 669, 0, False)
    
    # Passed parameters checking function
    _expm_multiply_interval_core_2.stypy_localization = localization
    _expm_multiply_interval_core_2.stypy_type_of_self = None
    _expm_multiply_interval_core_2.stypy_type_store = module_type_store
    _expm_multiply_interval_core_2.stypy_function_name = '_expm_multiply_interval_core_2'
    _expm_multiply_interval_core_2.stypy_param_names_list = ['A', 'X', 'h', 'mu', 'm_star', 's', 'q', 'tol']
    _expm_multiply_interval_core_2.stypy_varargs_param_name = None
    _expm_multiply_interval_core_2.stypy_kwargs_param_name = None
    _expm_multiply_interval_core_2.stypy_call_defaults = defaults
    _expm_multiply_interval_core_2.stypy_call_varargs = varargs
    _expm_multiply_interval_core_2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_expm_multiply_interval_core_2', ['A', 'X', 'h', 'mu', 'm_star', 's', 'q', 'tol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_expm_multiply_interval_core_2', localization, ['A', 'X', 'h', 'mu', 'm_star', 's', 'q', 'tol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_expm_multiply_interval_core_2(...)' code ##################

    str_389939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, (-1)), 'str', '\n    A helper function, for the case q > s and q % s > 0.\n    ')
    
    # Assigning a BinOp to a Name (line 673):
    
    # Assigning a BinOp to a Name (line 673):
    # Getting the type of 'q' (line 673)
    q_389940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'q')
    # Getting the type of 's' (line 673)
    s_389941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 13), 's')
    # Applying the binary operator '//' (line 673)
    result_floordiv_389942 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 8), '//', q_389940, s_389941)
    
    # Assigning a type to the variable 'd' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 4), 'd', result_floordiv_389942)
    
    # Assigning a BinOp to a Name (line 674):
    
    # Assigning a BinOp to a Name (line 674):
    # Getting the type of 'q' (line 674)
    q_389943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'q')
    # Getting the type of 'd' (line 674)
    d_389944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 13), 'd')
    # Applying the binary operator '//' (line 674)
    result_floordiv_389945 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 8), '//', q_389943, d_389944)
    
    # Assigning a type to the variable 'j' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 4), 'j', result_floordiv_389945)
    
    # Assigning a BinOp to a Name (line 675):
    
    # Assigning a BinOp to a Name (line 675):
    # Getting the type of 'q' (line 675)
    q_389946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 8), 'q')
    # Getting the type of 'd' (line 675)
    d_389947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 12), 'd')
    # Getting the type of 'j' (line 675)
    j_389948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 16), 'j')
    # Applying the binary operator '*' (line 675)
    result_mul_389949 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 12), '*', d_389947, j_389948)
    
    # Applying the binary operator '-' (line 675)
    result_sub_389950 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 8), '-', q_389946, result_mul_389949)
    
    # Assigning a type to the variable 'r' (line 675)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 4), 'r', result_sub_389950)
    
    # Assigning a Subscript to a Name (line 676):
    
    # Assigning a Subscript to a Name (line 676):
    
    # Obtaining the type of the subscript
    int_389951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 26), 'int')
    slice_389952 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 676, 18), int_389951, None, None)
    # Getting the type of 'X' (line 676)
    X_389953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 18), 'X')
    # Obtaining the member 'shape' of a type (line 676)
    shape_389954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 18), X_389953, 'shape')
    # Obtaining the member '__getitem__' of a type (line 676)
    getitem___389955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 18), shape_389954, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 676)
    subscript_call_result_389956 = invoke(stypy.reporting.localization.Localization(__file__, 676, 18), getitem___389955, slice_389952)
    
    # Assigning a type to the variable 'input_shape' (line 676)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 4), 'input_shape', subscript_call_result_389956)
    
    # Assigning a BinOp to a Name (line 677):
    
    # Assigning a BinOp to a Name (line 677):
    
    # Obtaining an instance of the builtin type 'tuple' (line 677)
    tuple_389957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 677)
    # Adding element type (line 677)
    # Getting the type of 'm_star' (line 677)
    m_star_389958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 15), 'm_star')
    int_389959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 24), 'int')
    # Applying the binary operator '+' (line 677)
    result_add_389960 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 15), '+', m_star_389958, int_389959)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 15), tuple_389957, result_add_389960)
    
    # Getting the type of 'input_shape' (line 677)
    input_shape_389961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 31), 'input_shape')
    # Applying the binary operator '+' (line 677)
    result_add_389962 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 14), '+', tuple_389957, input_shape_389961)
    
    # Assigning a type to the variable 'K_shape' (line 677)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 4), 'K_shape', result_add_389962)
    
    # Assigning a Call to a Name (line 678):
    
    # Assigning a Call to a Name (line 678):
    
    # Call to empty(...): (line 678)
    # Processing the call arguments (line 678)
    # Getting the type of 'K_shape' (line 678)
    K_shape_389965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 17), 'K_shape', False)
    # Processing the call keyword arguments (line 678)
    # Getting the type of 'X' (line 678)
    X_389966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 32), 'X', False)
    # Obtaining the member 'dtype' of a type (line 678)
    dtype_389967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 32), X_389966, 'dtype')
    keyword_389968 = dtype_389967
    kwargs_389969 = {'dtype': keyword_389968}
    # Getting the type of 'np' (line 678)
    np_389963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 678)
    empty_389964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 8), np_389963, 'empty')
    # Calling empty(args, kwargs) (line 678)
    empty_call_result_389970 = invoke(stypy.reporting.localization.Localization(__file__, 678, 8), empty_389964, *[K_shape_389965], **kwargs_389969)
    
    # Assigning a type to the variable 'K' (line 678)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 4), 'K', empty_call_result_389970)
    
    
    # Call to range(...): (line 679)
    # Processing the call arguments (line 679)
    # Getting the type of 'j' (line 679)
    j_389972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 19), 'j', False)
    int_389973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 23), 'int')
    # Applying the binary operator '+' (line 679)
    result_add_389974 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 19), '+', j_389972, int_389973)
    
    # Processing the call keyword arguments (line 679)
    kwargs_389975 = {}
    # Getting the type of 'range' (line 679)
    range_389971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 13), 'range', False)
    # Calling range(args, kwargs) (line 679)
    range_call_result_389976 = invoke(stypy.reporting.localization.Localization(__file__, 679, 13), range_389971, *[result_add_389974], **kwargs_389975)
    
    # Testing the type of a for loop iterable (line 679)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 679, 4), range_call_result_389976)
    # Getting the type of the for loop variable (line 679)
    for_loop_var_389977 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 679, 4), range_call_result_389976)
    # Assigning a type to the variable 'i' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 4), 'i', for_loop_var_389977)
    # SSA begins for a for statement (line 679)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 680):
    
    # Assigning a Subscript to a Name (line 680):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 680)
    i_389978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 14), 'i')
    # Getting the type of 'd' (line 680)
    d_389979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 16), 'd')
    # Applying the binary operator '*' (line 680)
    result_mul_389980 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 14), '*', i_389978, d_389979)
    
    # Getting the type of 'X' (line 680)
    X_389981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 12), 'X')
    # Obtaining the member '__getitem__' of a type (line 680)
    getitem___389982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 12), X_389981, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 680)
    subscript_call_result_389983 = invoke(stypy.reporting.localization.Localization(__file__, 680, 12), getitem___389982, result_mul_389980)
    
    # Assigning a type to the variable 'Z' (line 680)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 8), 'Z', subscript_call_result_389983)
    
    # Assigning a Name to a Subscript (line 681):
    
    # Assigning a Name to a Subscript (line 681):
    # Getting the type of 'Z' (line 681)
    Z_389984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 15), 'Z')
    # Getting the type of 'K' (line 681)
    K_389985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'K')
    int_389986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 10), 'int')
    # Storing an element on a container (line 681)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 8), K_389985, (int_389986, Z_389984))
    
    # Assigning a Num to a Name (line 682):
    
    # Assigning a Num to a Name (line 682):
    int_389987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 17), 'int')
    # Assigning a type to the variable 'high_p' (line 682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'high_p', int_389987)
    
    
    # Getting the type of 'i' (line 683)
    i_389988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 11), 'i')
    # Getting the type of 'j' (line 683)
    j_389989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 15), 'j')
    # Applying the binary operator '<' (line 683)
    result_lt_389990 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 11), '<', i_389988, j_389989)
    
    # Testing the type of an if condition (line 683)
    if_condition_389991 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 683, 8), result_lt_389990)
    # Assigning a type to the variable 'if_condition_389991' (line 683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 8), 'if_condition_389991', if_condition_389991)
    # SSA begins for if statement (line 683)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 684):
    
    # Assigning a Name to a Name (line 684):
    # Getting the type of 'd' (line 684)
    d_389992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 26), 'd')
    # Assigning a type to the variable 'effective_d' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 12), 'effective_d', d_389992)
    # SSA branch for the else part of an if statement (line 683)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 686):
    
    # Assigning a Name to a Name (line 686):
    # Getting the type of 'r' (line 686)
    r_389993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 26), 'r')
    # Assigning a type to the variable 'effective_d' (line 686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 12), 'effective_d', r_389993)
    # SSA join for if statement (line 683)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 687)
    # Processing the call arguments (line 687)
    int_389995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 23), 'int')
    # Getting the type of 'effective_d' (line 687)
    effective_d_389996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 26), 'effective_d', False)
    int_389997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 38), 'int')
    # Applying the binary operator '+' (line 687)
    result_add_389998 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 26), '+', effective_d_389996, int_389997)
    
    # Processing the call keyword arguments (line 687)
    kwargs_389999 = {}
    # Getting the type of 'range' (line 687)
    range_389994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 17), 'range', False)
    # Calling range(args, kwargs) (line 687)
    range_call_result_390000 = invoke(stypy.reporting.localization.Localization(__file__, 687, 17), range_389994, *[int_389995, result_add_389998], **kwargs_389999)
    
    # Testing the type of a for loop iterable (line 687)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 687, 8), range_call_result_390000)
    # Getting the type of the for loop variable (line 687)
    for_loop_var_390001 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 687, 8), range_call_result_390000)
    # Assigning a type to the variable 'k' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 8), 'k', for_loop_var_390001)
    # SSA begins for a for statement (line 687)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 688):
    
    # Assigning a Subscript to a Name (line 688):
    
    # Obtaining the type of the subscript
    int_390002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 18), 'int')
    # Getting the type of 'K' (line 688)
    K_390003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 16), 'K')
    # Obtaining the member '__getitem__' of a type (line 688)
    getitem___390004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 16), K_390003, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 688)
    subscript_call_result_390005 = invoke(stypy.reporting.localization.Localization(__file__, 688, 16), getitem___390004, int_390002)
    
    # Assigning a type to the variable 'F' (line 688)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 12), 'F', subscript_call_result_390005)
    
    # Assigning a Call to a Name (line 689):
    
    # Assigning a Call to a Name (line 689):
    
    # Call to _exact_inf_norm(...): (line 689)
    # Processing the call arguments (line 689)
    # Getting the type of 'F' (line 689)
    F_390007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 33), 'F', False)
    # Processing the call keyword arguments (line 689)
    kwargs_390008 = {}
    # Getting the type of '_exact_inf_norm' (line 689)
    _exact_inf_norm_390006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 17), '_exact_inf_norm', False)
    # Calling _exact_inf_norm(args, kwargs) (line 689)
    _exact_inf_norm_call_result_390009 = invoke(stypy.reporting.localization.Localization(__file__, 689, 17), _exact_inf_norm_390006, *[F_390007], **kwargs_390008)
    
    # Assigning a type to the variable 'c1' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 12), 'c1', _exact_inf_norm_call_result_390009)
    
    
    # Call to range(...): (line 690)
    # Processing the call arguments (line 690)
    int_390011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 27), 'int')
    # Getting the type of 'm_star' (line 690)
    m_star_390012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 30), 'm_star', False)
    int_390013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 37), 'int')
    # Applying the binary operator '+' (line 690)
    result_add_390014 = python_operator(stypy.reporting.localization.Localization(__file__, 690, 30), '+', m_star_390012, int_390013)
    
    # Processing the call keyword arguments (line 690)
    kwargs_390015 = {}
    # Getting the type of 'range' (line 690)
    range_390010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 21), 'range', False)
    # Calling range(args, kwargs) (line 690)
    range_call_result_390016 = invoke(stypy.reporting.localization.Localization(__file__, 690, 21), range_390010, *[int_390011, result_add_390014], **kwargs_390015)
    
    # Testing the type of a for loop iterable (line 690)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 690, 12), range_call_result_390016)
    # Getting the type of the for loop variable (line 690)
    for_loop_var_390017 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 690, 12), range_call_result_390016)
    # Assigning a type to the variable 'p' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 12), 'p', for_loop_var_390017)
    # SSA begins for a for statement (line 690)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'p' (line 691)
    p_390018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 19), 'p')
    # Getting the type of 'high_p' (line 691)
    high_p_390019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 24), 'high_p')
    int_390020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 33), 'int')
    # Applying the binary operator '+' (line 691)
    result_add_390021 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 24), '+', high_p_390019, int_390020)
    
    # Applying the binary operator '==' (line 691)
    result_eq_390022 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 19), '==', p_390018, result_add_390021)
    
    # Testing the type of an if condition (line 691)
    if_condition_390023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 691, 16), result_eq_390022)
    # Assigning a type to the variable 'if_condition_390023' (line 691)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 'if_condition_390023', if_condition_390023)
    # SSA begins for if statement (line 691)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 692):
    
    # Assigning a BinOp to a Subscript (line 692):
    # Getting the type of 'h' (line 692)
    h_390024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 27), 'h')
    
    # Call to dot(...): (line 692)
    # Processing the call arguments (line 692)
    
    # Obtaining the type of the subscript
    # Getting the type of 'p' (line 692)
    p_390027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 39), 'p', False)
    int_390028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 41), 'int')
    # Applying the binary operator '-' (line 692)
    result_sub_390029 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 39), '-', p_390027, int_390028)
    
    # Getting the type of 'K' (line 692)
    K_390030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 37), 'K', False)
    # Obtaining the member '__getitem__' of a type (line 692)
    getitem___390031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 37), K_390030, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 692)
    subscript_call_result_390032 = invoke(stypy.reporting.localization.Localization(__file__, 692, 37), getitem___390031, result_sub_390029)
    
    # Processing the call keyword arguments (line 692)
    kwargs_390033 = {}
    # Getting the type of 'A' (line 692)
    A_390025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 31), 'A', False)
    # Obtaining the member 'dot' of a type (line 692)
    dot_390026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 31), A_390025, 'dot')
    # Calling dot(args, kwargs) (line 692)
    dot_call_result_390034 = invoke(stypy.reporting.localization.Localization(__file__, 692, 31), dot_390026, *[subscript_call_result_390032], **kwargs_390033)
    
    # Applying the binary operator '*' (line 692)
    result_mul_390035 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 27), '*', h_390024, dot_call_result_390034)
    
    
    # Call to float(...): (line 692)
    # Processing the call arguments (line 692)
    # Getting the type of 'p' (line 692)
    p_390037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 53), 'p', False)
    # Processing the call keyword arguments (line 692)
    kwargs_390038 = {}
    # Getting the type of 'float' (line 692)
    float_390036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 47), 'float', False)
    # Calling float(args, kwargs) (line 692)
    float_call_result_390039 = invoke(stypy.reporting.localization.Localization(__file__, 692, 47), float_390036, *[p_390037], **kwargs_390038)
    
    # Applying the binary operator 'div' (line 692)
    result_div_390040 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 45), 'div', result_mul_390035, float_call_result_390039)
    
    # Getting the type of 'K' (line 692)
    K_390041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 20), 'K')
    # Getting the type of 'p' (line 692)
    p_390042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 22), 'p')
    # Storing an element on a container (line 692)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 20), K_390041, (p_390042, result_div_390040))
    
    # Assigning a Name to a Name (line 693):
    
    # Assigning a Name to a Name (line 693):
    # Getting the type of 'p' (line 693)
    p_390043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 29), 'p')
    # Assigning a type to the variable 'high_p' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 20), 'high_p', p_390043)
    # SSA join for if statement (line 691)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 694):
    
    # Assigning a Call to a Name (line 694):
    
    # Call to float(...): (line 694)
    # Processing the call arguments (line 694)
    
    # Call to pow(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'k' (line 694)
    k_390046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 34), 'k', False)
    # Getting the type of 'p' (line 694)
    p_390047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 37), 'p', False)
    # Processing the call keyword arguments (line 694)
    kwargs_390048 = {}
    # Getting the type of 'pow' (line 694)
    pow_390045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 30), 'pow', False)
    # Calling pow(args, kwargs) (line 694)
    pow_call_result_390049 = invoke(stypy.reporting.localization.Localization(__file__, 694, 30), pow_390045, *[k_390046, p_390047], **kwargs_390048)
    
    # Processing the call keyword arguments (line 694)
    kwargs_390050 = {}
    # Getting the type of 'float' (line 694)
    float_390044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 24), 'float', False)
    # Calling float(args, kwargs) (line 694)
    float_call_result_390051 = invoke(stypy.reporting.localization.Localization(__file__, 694, 24), float_390044, *[pow_call_result_390049], **kwargs_390050)
    
    # Assigning a type to the variable 'coeff' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), 'coeff', float_call_result_390051)
    
    # Assigning a BinOp to a Name (line 695):
    
    # Assigning a BinOp to a Name (line 695):
    # Getting the type of 'F' (line 695)
    F_390052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 20), 'F')
    # Getting the type of 'coeff' (line 695)
    coeff_390053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 24), 'coeff')
    
    # Obtaining the type of the subscript
    # Getting the type of 'p' (line 695)
    p_390054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 34), 'p')
    # Getting the type of 'K' (line 695)
    K_390055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 32), 'K')
    # Obtaining the member '__getitem__' of a type (line 695)
    getitem___390056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 32), K_390055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 695)
    subscript_call_result_390057 = invoke(stypy.reporting.localization.Localization(__file__, 695, 32), getitem___390056, p_390054)
    
    # Applying the binary operator '*' (line 695)
    result_mul_390058 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 24), '*', coeff_390053, subscript_call_result_390057)
    
    # Applying the binary operator '+' (line 695)
    result_add_390059 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 20), '+', F_390052, result_mul_390058)
    
    # Assigning a type to the variable 'F' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'F', result_add_390059)
    
    # Assigning a Call to a Name (line 696):
    
    # Assigning a Call to a Name (line 696):
    
    # Call to _exact_inf_norm(...): (line 696)
    # Processing the call arguments (line 696)
    
    # Obtaining the type of the subscript
    # Getting the type of 'p' (line 696)
    p_390061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 51), 'p', False)
    # Getting the type of 'K' (line 696)
    K_390062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 49), 'K', False)
    # Obtaining the member '__getitem__' of a type (line 696)
    getitem___390063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 49), K_390062, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 696)
    subscript_call_result_390064 = invoke(stypy.reporting.localization.Localization(__file__, 696, 49), getitem___390063, p_390061)
    
    # Processing the call keyword arguments (line 696)
    kwargs_390065 = {}
    # Getting the type of '_exact_inf_norm' (line 696)
    _exact_inf_norm_390060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 33), '_exact_inf_norm', False)
    # Calling _exact_inf_norm(args, kwargs) (line 696)
    _exact_inf_norm_call_result_390066 = invoke(stypy.reporting.localization.Localization(__file__, 696, 33), _exact_inf_norm_390060, *[subscript_call_result_390064], **kwargs_390065)
    
    # Assigning a type to the variable 'inf_norm_K_p_1' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 16), 'inf_norm_K_p_1', _exact_inf_norm_call_result_390066)
    
    # Assigning a BinOp to a Name (line 697):
    
    # Assigning a BinOp to a Name (line 697):
    # Getting the type of 'coeff' (line 697)
    coeff_390067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 21), 'coeff')
    # Getting the type of 'inf_norm_K_p_1' (line 697)
    inf_norm_K_p_1_390068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 29), 'inf_norm_K_p_1')
    # Applying the binary operator '*' (line 697)
    result_mul_390069 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 21), '*', coeff_390067, inf_norm_K_p_1_390068)
    
    # Assigning a type to the variable 'c2' (line 697)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'c2', result_mul_390069)
    
    
    # Getting the type of 'c1' (line 698)
    c1_390070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 19), 'c1')
    # Getting the type of 'c2' (line 698)
    c2_390071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 24), 'c2')
    # Applying the binary operator '+' (line 698)
    result_add_390072 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 19), '+', c1_390070, c2_390071)
    
    # Getting the type of 'tol' (line 698)
    tol_390073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 30), 'tol')
    
    # Call to _exact_inf_norm(...): (line 698)
    # Processing the call arguments (line 698)
    # Getting the type of 'F' (line 698)
    F_390075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 52), 'F', False)
    # Processing the call keyword arguments (line 698)
    kwargs_390076 = {}
    # Getting the type of '_exact_inf_norm' (line 698)
    _exact_inf_norm_390074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 36), '_exact_inf_norm', False)
    # Calling _exact_inf_norm(args, kwargs) (line 698)
    _exact_inf_norm_call_result_390077 = invoke(stypy.reporting.localization.Localization(__file__, 698, 36), _exact_inf_norm_390074, *[F_390075], **kwargs_390076)
    
    # Applying the binary operator '*' (line 698)
    result_mul_390078 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 30), '*', tol_390073, _exact_inf_norm_call_result_390077)
    
    # Applying the binary operator '<=' (line 698)
    result_le_390079 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 19), '<=', result_add_390072, result_mul_390078)
    
    # Testing the type of an if condition (line 698)
    if_condition_390080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 698, 16), result_le_390079)
    # Assigning a type to the variable 'if_condition_390080' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 16), 'if_condition_390080', if_condition_390080)
    # SSA begins for if statement (line 698)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 698)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 700):
    
    # Assigning a Name to a Name (line 700):
    # Getting the type of 'c2' (line 700)
    c2_390081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 21), 'c2')
    # Assigning a type to the variable 'c1' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 16), 'c1', c2_390081)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 701):
    
    # Assigning a BinOp to a Subscript (line 701):
    
    # Call to exp(...): (line 701)
    # Processing the call arguments (line 701)
    # Getting the type of 'k' (line 701)
    k_390084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 32), 'k', False)
    # Getting the type of 'h' (line 701)
    h_390085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 34), 'h', False)
    # Applying the binary operator '*' (line 701)
    result_mul_390086 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 32), '*', k_390084, h_390085)
    
    # Getting the type of 'mu' (line 701)
    mu_390087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 36), 'mu', False)
    # Applying the binary operator '*' (line 701)
    result_mul_390088 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 35), '*', result_mul_390086, mu_390087)
    
    # Processing the call keyword arguments (line 701)
    kwargs_390089 = {}
    # Getting the type of 'np' (line 701)
    np_390082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 25), 'np', False)
    # Obtaining the member 'exp' of a type (line 701)
    exp_390083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 25), np_390082, 'exp')
    # Calling exp(args, kwargs) (line 701)
    exp_call_result_390090 = invoke(stypy.reporting.localization.Localization(__file__, 701, 25), exp_390083, *[result_mul_390088], **kwargs_390089)
    
    # Getting the type of 'F' (line 701)
    F_390091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 42), 'F')
    # Applying the binary operator '*' (line 701)
    result_mul_390092 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 25), '*', exp_call_result_390090, F_390091)
    
    # Getting the type of 'X' (line 701)
    X_390093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 12), 'X')
    # Getting the type of 'k' (line 701)
    k_390094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 14), 'k')
    # Getting the type of 'i' (line 701)
    i_390095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 18), 'i')
    # Getting the type of 'd' (line 701)
    d_390096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 20), 'd')
    # Applying the binary operator '*' (line 701)
    result_mul_390097 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 18), '*', i_390095, d_390096)
    
    # Applying the binary operator '+' (line 701)
    result_add_390098 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 14), '+', k_390094, result_mul_390097)
    
    # Storing an element on a container (line 701)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 701, 12), X_390093, (result_add_390098, result_mul_390092))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 702)
    tuple_390099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 702)
    # Adding element type (line 702)
    # Getting the type of 'X' (line 702)
    X_390100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 11), 'X')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 11), tuple_390099, X_390100)
    # Adding element type (line 702)
    int_390101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 11), tuple_390099, int_390101)
    
    # Assigning a type to the variable 'stypy_return_type' (line 702)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 4), 'stypy_return_type', tuple_390099)
    
    # ################# End of '_expm_multiply_interval_core_2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_expm_multiply_interval_core_2' in the type store
    # Getting the type of 'stypy_return_type' (line 669)
    stypy_return_type_390102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_390102)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_expm_multiply_interval_core_2'
    return stypy_return_type_390102

# Assigning a type to the variable '_expm_multiply_interval_core_2' (line 669)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 0), '_expm_multiply_interval_core_2', _expm_multiply_interval_core_2)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
