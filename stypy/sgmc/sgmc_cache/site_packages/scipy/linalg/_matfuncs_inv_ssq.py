
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Matrix functions that use Pade approximation with inverse scaling and squaring.
3: 
4: '''
5: from __future__ import division, print_function, absolute_import
6: 
7: import warnings
8: 
9: import numpy as np
10: 
11: from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu
12: from scipy.linalg.decomp_schur import schur, rsf2csf
13: from scipy.linalg.matfuncs import funm
14: from scipy.linalg import svdvals, solve_triangular
15: from scipy.sparse.linalg.interface import LinearOperator
16: from scipy.sparse.linalg import onenormest
17: import scipy.special
18: 
19: 
20: class LogmRankWarning(UserWarning):
21:     pass
22: 
23: 
24: class LogmExactlySingularWarning(LogmRankWarning):
25:     pass
26: 
27: 
28: class LogmNearlySingularWarning(LogmRankWarning):
29:     pass
30: 
31: 
32: class LogmError(np.linalg.LinAlgError):
33:     pass
34: 
35: 
36: class FractionalMatrixPowerError(np.linalg.LinAlgError):
37:     pass
38: 
39: 
40: #TODO renovate or move this class when scipy operators are more mature
41: class _MatrixM1PowerOperator(LinearOperator):
42:     '''
43:     A representation of the linear operator (A - I)^p.
44:     '''
45: 
46:     def __init__(self, A, p):
47:         if A.ndim != 2 or A.shape[0] != A.shape[1]:
48:             raise ValueError('expected A to be like a square matrix')
49:         if p < 0 or p != int(p):
50:             raise ValueError('expected p to be a non-negative integer')
51:         self._A = A
52:         self._p = p
53:         self.ndim = A.ndim
54:         self.shape = A.shape
55: 
56:     def _matvec(self, x):
57:         for i in range(self._p):
58:             x = self._A.dot(x) - x
59:         return x
60: 
61:     def _rmatvec(self, x):
62:         for i in range(self._p):
63:             x = x.dot(self._A) - x
64:         return x
65: 
66:     def _matmat(self, X):
67:         for i in range(self._p):
68:             X = self._A.dot(X) - X
69:         return X
70: 
71:     def _adjoint(self):
72:         return _MatrixM1PowerOperator(self._A.T, self._p)
73: 
74: 
75: #TODO renovate or move this function when scipy operators are more mature
76: def _onenormest_m1_power(A, p,
77:         t=2, itmax=5, compute_v=False, compute_w=False):
78:     '''
79:     Efficiently estimate the 1-norm of (A - I)^p.
80: 
81:     Parameters
82:     ----------
83:     A : ndarray
84:         Matrix whose 1-norm of a power is to be computed.
85:     p : int
86:         Non-negative integer power.
87:     t : int, optional
88:         A positive parameter controlling the tradeoff between
89:         accuracy versus time and memory usage.
90:         Larger values take longer and use more memory
91:         but give more accurate output.
92:     itmax : int, optional
93:         Use at most this many iterations.
94:     compute_v : bool, optional
95:         Request a norm-maximizing linear operator input vector if True.
96:     compute_w : bool, optional
97:         Request a norm-maximizing linear operator output vector if True.
98: 
99:     Returns
100:     -------
101:     est : float
102:         An underestimate of the 1-norm of the sparse matrix.
103:     v : ndarray, optional
104:         The vector such that ||Av||_1 == est*||v||_1.
105:         It can be thought of as an input to the linear operator
106:         that gives an output with particularly large norm.
107:     w : ndarray, optional
108:         The vector Av which has relatively large 1-norm.
109:         It can be thought of as an output of the linear operator
110:         that is relatively large in norm compared to the input.
111: 
112:     '''
113:     return onenormest(_MatrixM1PowerOperator(A, p),
114:             t=t, itmax=itmax, compute_v=compute_v, compute_w=compute_w)
115: 
116: 
117: def _unwindk(z):
118:     '''
119:     Compute the scalar unwinding number.
120: 
121:     Uses Eq. (5.3) in [1]_, and should be equal to (z - log(exp(z)) / (2 pi i).
122:     Note that this definition differs in sign from the original definition
123:     in equations (5, 6) in [2]_.  The sign convention is justified in [3]_.
124: 
125:     Parameters
126:     ----------
127:     z : complex
128:         A complex number.
129: 
130:     Returns
131:     -------
132:     unwinding_number : integer
133:         The scalar unwinding number of z.
134: 
135:     References
136:     ----------
137:     .. [1] Nicholas J. Higham and Lijing lin (2011)
138:            "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
139:            SIAM Journal on Matrix Analysis and Applications,
140:            32 (3). pp. 1056-1078. ISSN 0895-4798
141: 
142:     .. [2] Robert M. Corless and David J. Jeffrey,
143:            "The unwinding number." Newsletter ACM SIGSAM Bulletin
144:            Volume 30, Issue 2, June 1996, Pages 28-35.
145: 
146:     .. [3] Russell Bradford and Robert M. Corless and James H. Davenport and
147:            David J. Jeffrey and Stephen M. Watt,
148:            "Reasoning about the elementary functions of complex analysis"
149:            Annals of Mathematics and Artificial Intelligence,
150:            36: 303-318, 2002.
151: 
152:     '''
153:     return int(np.ceil((z.imag - np.pi) / (2*np.pi)))
154: 
155: 
156: def _briggs_helper_function(a, k):
157:     '''
158:     Computes r = a^(1 / (2^k)) - 1.
159: 
160:     This is algorithm (2) of [1]_.
161:     The purpose is to avoid a danger of subtractive cancellation.
162:     For more computational efficiency it should probably be cythonized.
163: 
164:     Parameters
165:     ----------
166:     a : complex
167:         A complex number.
168:     k : integer
169:         A nonnegative integer.
170: 
171:     Returns
172:     -------
173:     r : complex
174:         The value r = a^(1 / (2^k)) - 1 computed with less cancellation.
175: 
176:     Notes
177:     -----
178:     The algorithm as formulated in the reference does not handle k=0 or k=1
179:     correctly, so these are special-cased in this implementation.
180:     This function is intended to not allow `a` to belong to the closed
181:     negative real axis, but this constraint is relaxed.
182: 
183:     References
184:     ----------
185:     .. [1] Awad H. Al-Mohy (2012)
186:            "A more accurate Briggs method for the logarithm",
187:            Numerical Algorithms, 59 : 393--402.
188: 
189:     '''
190:     if k < 0 or int(k) != k:
191:         raise ValueError('expected a nonnegative integer k')
192:     if k == 0:
193:         return a - 1
194:     elif k == 1:
195:         return np.sqrt(a) - 1
196:     else:
197:         k_hat = k
198:         if np.angle(a) >= np.pi / 2:
199:             a = np.sqrt(a)
200:             k_hat = k - 1
201:         z0 = a - 1
202:         a = np.sqrt(a)
203:         r = 1 + a
204:         for j in range(1, k_hat):
205:             a = np.sqrt(a)
206:             r = r * (1 + a)
207:         r = z0 / r
208:         return r
209: 
210: 
211: def _fractional_power_superdiag_entry(l1, l2, t12, p):
212:     '''
213:     Compute a superdiagonal entry of a fractional matrix power.
214: 
215:     This is Eq. (5.6) in [1]_.
216: 
217:     Parameters
218:     ----------
219:     l1 : complex
220:         A diagonal entry of the matrix.
221:     l2 : complex
222:         A diagonal entry of the matrix.
223:     t12 : complex
224:         A superdiagonal entry of the matrix.
225:     p : float
226:         A fractional power.
227: 
228:     Returns
229:     -------
230:     f12 : complex
231:         A superdiagonal entry of the fractional matrix power.
232: 
233:     Notes
234:     -----
235:     Care has been taken to return a real number if possible when
236:     all of the inputs are real numbers.
237: 
238:     References
239:     ----------
240:     .. [1] Nicholas J. Higham and Lijing lin (2011)
241:            "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
242:            SIAM Journal on Matrix Analysis and Applications,
243:            32 (3). pp. 1056-1078. ISSN 0895-4798
244: 
245:     '''
246:     if l1 == l2:
247:         f12 = t12 * p * l1**(p-1)
248:     elif abs(l2 - l1) > abs(l1 + l2) / 2:
249:         f12 = t12 * ((l2**p) - (l1**p)) / (l2 - l1)
250:     else:
251:         # This is Eq. (5.5) in [1].
252:         z = (l2 - l1) / (l2 + l1)
253:         log_l1 = np.log(l1)
254:         log_l2 = np.log(l2)
255:         arctanh_z = np.arctanh(z)
256:         tmp_a = t12 * np.exp((p/2)*(log_l2 + log_l1))
257:         tmp_u = _unwindk(log_l2 - log_l1)
258:         if tmp_u:
259:             tmp_b = p * (arctanh_z + np.pi * 1j * tmp_u)
260:         else:
261:             tmp_b = p * arctanh_z
262:         tmp_c = 2 * np.sinh(tmp_b) / (l2 - l1)
263:         f12 = tmp_a * tmp_c
264:     return f12
265: 
266: 
267: def _logm_superdiag_entry(l1, l2, t12):
268:     '''
269:     Compute a superdiagonal entry of a matrix logarithm.
270: 
271:     This is like Eq. (11.28) in [1]_, except the determination of whether
272:     l1 and l2 are sufficiently far apart has been modified.
273: 
274:     Parameters
275:     ----------
276:     l1 : complex
277:         A diagonal entry of the matrix.
278:     l2 : complex
279:         A diagonal entry of the matrix.
280:     t12 : complex
281:         A superdiagonal entry of the matrix.
282: 
283:     Returns
284:     -------
285:     f12 : complex
286:         A superdiagonal entry of the matrix logarithm.
287: 
288:     Notes
289:     -----
290:     Care has been taken to return a real number if possible when
291:     all of the inputs are real numbers.
292: 
293:     References
294:     ----------
295:     .. [1] Nicholas J. Higham (2008)
296:            "Functions of Matrices: Theory and Computation"
297:            ISBN 978-0-898716-46-7
298: 
299:     '''
300:     if l1 == l2:
301:         f12 = t12 / l1
302:     elif abs(l2 - l1) > abs(l1 + l2) / 2:
303:         f12 = t12 * (np.log(l2) - np.log(l1)) / (l2 - l1)
304:     else:
305:         z = (l2 - l1) / (l2 + l1)
306:         u = _unwindk(np.log(l2) - np.log(l1))
307:         if u:
308:             f12 = t12 * 2 * (np.arctanh(z) + np.pi*1j*u) / (l2 - l1)
309:         else:
310:             f12 = t12 * 2 * np.arctanh(z) / (l2 - l1)
311:     return f12
312: 
313: 
314: def _inverse_squaring_helper(T0, theta):
315:     '''
316:     A helper function for inverse scaling and squaring for Pade approximation.
317: 
318:     Parameters
319:     ----------
320:     T0 : (N, N) array_like upper triangular
321:         Matrix involved in inverse scaling and squaring.
322:     theta : indexable
323:         The values theta[1] .. theta[7] must be available.
324:         They represent bounds related to Pade approximation, and they depend
325:         on the matrix function which is being computed.
326:         For example, different values of theta are required for
327:         matrix logarithm than for fractional matrix power.
328: 
329:     Returns
330:     -------
331:     R : (N, N) array_like upper triangular
332:         Composition of zero or more matrix square roots of T0, minus I.
333:     s : non-negative integer
334:         Number of square roots taken.
335:     m : positive integer
336:         The degree of the Pade approximation.
337: 
338:     Notes
339:     -----
340:     This subroutine appears as a chunk of lines within
341:     a couple of published algorithms; for example it appears
342:     as lines 4--35 in algorithm (3.1) of [1]_, and
343:     as lines 3--34 in algorithm (4.1) of [2]_.
344:     The instances of 'goto line 38' in algorithm (3.1) of [1]_
345:     probably mean 'goto line 36' and have been intepreted accordingly.
346: 
347:     References
348:     ----------
349:     .. [1] Nicholas J. Higham and Lijing Lin (2013)
350:            "An Improved Schur-Pade Algorithm for Fractional Powers
351:            of a Matrix and their Frechet Derivatives."
352: 
353:     .. [2] Awad H. Al-Mohy and Nicholas J. Higham (2012)
354:            "Improved Inverse Scaling and Squaring Algorithms
355:            for the Matrix Logarithm."
356:            SIAM Journal on Scientific Computing, 34 (4). C152-C169.
357:            ISSN 1095-7197
358: 
359:     '''
360:     if len(T0.shape) != 2 or T0.shape[0] != T0.shape[1]:
361:         raise ValueError('expected an upper triangular square matrix')
362:     n, n = T0.shape
363:     T = T0
364: 
365:     # Find s0, the smallest s such that the spectral radius
366:     # of a certain diagonal matrix is at most theta[7].
367:     # Note that because theta[7] < 1,
368:     # this search will not terminate if any diagonal entry of T is zero.
369:     s0 = 0
370:     tmp_diag = np.diag(T)
371:     if np.count_nonzero(tmp_diag) != n:
372:         raise Exception('internal inconsistency')
373:     while np.max(np.absolute(tmp_diag - 1)) > theta[7]:
374:         tmp_diag = np.sqrt(tmp_diag)
375:         s0 += 1
376: 
377:     # Take matrix square roots of T.
378:     for i in range(s0):
379:         T = _sqrtm_triu(T)
380: 
381:     # Flow control in this section is a little odd.
382:     # This is because I am translating algorithm descriptions
383:     # which have GOTOs in the publication.
384:     s = s0
385:     k = 0
386:     d2 = _onenormest_m1_power(T, 2) ** (1/2)
387:     d3 = _onenormest_m1_power(T, 3) ** (1/3)
388:     a2 = max(d2, d3)
389:     m = None
390:     for i in (1, 2):
391:         if a2 <= theta[i]:
392:             m = i
393:             break
394:     while m is None:
395:         if s > s0:
396:             d3 = _onenormest_m1_power(T, 3) ** (1/3)
397:         d4 = _onenormest_m1_power(T, 4) ** (1/4)
398:         a3 = max(d3, d4)
399:         if a3 <= theta[7]:
400:             j1 = min(i for i in (3, 4, 5, 6, 7) if a3 <= theta[i])
401:             if j1 <= 6:
402:                 m = j1
403:                 break
404:             elif a3 / 2 <= theta[5] and k < 2:
405:                 k += 1
406:                 T = _sqrtm_triu(T)
407:                 s += 1
408:                 continue
409:         d5 = _onenormest_m1_power(T, 5) ** (1/5)
410:         a4 = max(d4, d5)
411:         eta = min(a3, a4)
412:         for i in (6, 7):
413:             if eta <= theta[i]:
414:                 m = i
415:                 break
416:         if m is not None:
417:             break
418:         T = _sqrtm_triu(T)
419:         s += 1
420: 
421:     # The subtraction of the identity is redundant here,
422:     # because the diagonal will be replaced for improved numerical accuracy,
423:     # but this formulation should help clarify the meaning of R.
424:     R = T - np.identity(n)
425: 
426:     # Replace the diagonal and first superdiagonal of T0^(1/(2^s)) - I
427:     # using formulas that have less subtractive cancellation.
428:     # Skip this step if the principal branch
429:     # does not exist at T0; this happens when a diagonal entry of T0
430:     # is negative with imaginary part 0.
431:     has_principal_branch = all(x.real > 0 or x.imag != 0 for x in np.diag(T0))
432:     if has_principal_branch:
433:         for j in range(n):
434:             a = T0[j, j]
435:             r = _briggs_helper_function(a, s)
436:             R[j, j] = r
437:         p = np.exp2(-s)
438:         for j in range(n-1):
439:             l1 = T0[j, j]
440:             l2 = T0[j+1, j+1]
441:             t12 = T0[j, j+1]
442:             f12 = _fractional_power_superdiag_entry(l1, l2, t12, p)
443:             R[j, j+1] = f12
444: 
445:     # Return the T-I matrix, the number of square roots, and the Pade degree.
446:     if not np.array_equal(R, np.triu(R)):
447:         raise Exception('internal inconsistency')
448:     return R, s, m
449: 
450: 
451: def _fractional_power_pade_constant(i, t):
452:     # A helper function for matrix fractional power.
453:     if i < 1:
454:         raise ValueError('expected a positive integer i')
455:     if not (-1 < t < 1):
456:         raise ValueError('expected -1 < t < 1')
457:     if i == 1:
458:         return -t
459:     elif i % 2 == 0:
460:         j = i // 2
461:         return (-j + t) / (2 * (2*j - 1))
462:     elif i % 2 == 1:
463:         j = (i - 1) // 2
464:         return (-j - t) / (2 * (2*j + 1))
465:     else:
466:         raise Exception('internal error')
467: 
468: 
469: def _fractional_power_pade(R, t, m):
470:     '''
471:     Evaluate the Pade approximation of a fractional matrix power.
472: 
473:     Evaluate the degree-m Pade approximation of R
474:     to the fractional matrix power t using the continued fraction
475:     in bottom-up fashion using algorithm (4.1) in [1]_.
476: 
477:     Parameters
478:     ----------
479:     R : (N, N) array_like
480:         Upper triangular matrix whose fractional power to evaluate.
481:     t : float
482:         Fractional power between -1 and 1 exclusive.
483:     m : positive integer
484:         Degree of Pade approximation.
485: 
486:     Returns
487:     -------
488:     U : (N, N) array_like
489:         The degree-m Pade approximation of R to the fractional power t.
490:         This matrix will be upper triangular.
491: 
492:     References
493:     ----------
494:     .. [1] Nicholas J. Higham and Lijing lin (2011)
495:            "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
496:            SIAM Journal on Matrix Analysis and Applications,
497:            32 (3). pp. 1056-1078. ISSN 0895-4798
498: 
499:     '''
500:     if m < 1 or int(m) != m:
501:         raise ValueError('expected a positive integer m')
502:     if not (-1 < t < 1):
503:         raise ValueError('expected -1 < t < 1')
504:     R = np.asarray(R)
505:     if len(R.shape) != 2 or R.shape[0] != R.shape[1]:
506:         raise ValueError('expected an upper triangular square matrix')
507:     n, n = R.shape
508:     ident = np.identity(n)
509:     Y = R * _fractional_power_pade_constant(2*m, t)
510:     for j in range(2*m - 1, 0, -1):
511:         rhs = R * _fractional_power_pade_constant(j, t)
512:         Y = solve_triangular(ident + Y, rhs)
513:     U = ident + Y
514:     if not np.array_equal(U, np.triu(U)):
515:         raise Exception('internal inconsistency')
516:     return U
517: 
518: 
519: def _remainder_matrix_power_triu(T, t):
520:     '''
521:     Compute a fractional power of an upper triangular matrix.
522: 
523:     The fractional power is restricted to fractions -1 < t < 1.
524:     This uses algorithm (3.1) of [1]_.
525:     The Pade approximation itself uses algorithm (4.1) of [2]_.
526: 
527:     Parameters
528:     ----------
529:     T : (N, N) array_like
530:         Upper triangular matrix whose fractional power to evaluate.
531:     t : float
532:         Fractional power between -1 and 1 exclusive.
533: 
534:     Returns
535:     -------
536:     X : (N, N) array_like
537:         The fractional power of the matrix.
538: 
539:     References
540:     ----------
541:     .. [1] Nicholas J. Higham and Lijing Lin (2013)
542:            "An Improved Schur-Pade Algorithm for Fractional Powers
543:            of a Matrix and their Frechet Derivatives."
544: 
545:     .. [2] Nicholas J. Higham and Lijing lin (2011)
546:            "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
547:            SIAM Journal on Matrix Analysis and Applications,
548:            32 (3). pp. 1056-1078. ISSN 0895-4798
549: 
550:     '''
551:     m_to_theta = {
552:             1: 1.51e-5,
553:             2: 2.24e-3,
554:             3: 1.88e-2,
555:             4: 6.04e-2,
556:             5: 1.24e-1,
557:             6: 2.00e-1,
558:             7: 2.79e-1,
559:             }
560:     n, n = T.shape
561:     T0 = T
562:     T0_diag = np.diag(T0)
563:     if np.array_equal(T0, np.diag(T0_diag)):
564:         U = np.diag(T0_diag ** t)
565:     else:
566:         R, s, m = _inverse_squaring_helper(T0, m_to_theta)
567: 
568:         # Evaluate the Pade approximation.
569:         # Note that this function expects the negative of the matrix
570:         # returned by the inverse squaring helper.
571:         U = _fractional_power_pade(-R, t, m)
572: 
573:         # Undo the inverse scaling and squaring.
574:         # Be less clever about this
575:         # if the principal branch does not exist at T0;
576:         # this happens when a diagonal entry of T0
577:         # is negative with imaginary part 0.
578:         eivals = np.diag(T0)
579:         has_principal_branch = all(x.real > 0 or x.imag != 0 for x in eivals)
580:         for i in range(s, -1, -1):
581:             if i < s:
582:                 U = U.dot(U)
583:             else:
584:                 if has_principal_branch:
585:                     p = t * np.exp2(-i)
586:                     U[np.diag_indices(n)] = T0_diag ** p
587:                     for j in range(n-1):
588:                         l1 = T0[j, j]
589:                         l2 = T0[j+1, j+1]
590:                         t12 = T0[j, j+1]
591:                         f12 = _fractional_power_superdiag_entry(l1, l2, t12, p)
592:                         U[j, j+1] = f12
593:     if not np.array_equal(U, np.triu(U)):
594:         raise Exception('internal inconsistency')
595:     return U
596: 
597: 
598: def _remainder_matrix_power(A, t):
599:     '''
600:     Compute the fractional power of a matrix, for fractions -1 < t < 1.
601: 
602:     This uses algorithm (3.1) of [1]_.
603:     The Pade approximation itself uses algorithm (4.1) of [2]_.
604: 
605:     Parameters
606:     ----------
607:     A : (N, N) array_like
608:         Matrix whose fractional power to evaluate.
609:     t : float
610:         Fractional power between -1 and 1 exclusive.
611: 
612:     Returns
613:     -------
614:     X : (N, N) array_like
615:         The fractional power of the matrix.
616: 
617:     References
618:     ----------
619:     .. [1] Nicholas J. Higham and Lijing Lin (2013)
620:            "An Improved Schur-Pade Algorithm for Fractional Powers
621:            of a Matrix and their Frechet Derivatives."
622: 
623:     .. [2] Nicholas J. Higham and Lijing lin (2011)
624:            "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
625:            SIAM Journal on Matrix Analysis and Applications,
626:            32 (3). pp. 1056-1078. ISSN 0895-4798
627: 
628:     '''
629:     # This code block is copied from numpy.matrix_power().
630:     A = np.asarray(A)
631:     if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
632:         raise ValueError('input must be a square array')
633: 
634:     # Get the number of rows and columns.
635:     n, n = A.shape
636: 
637:     # Triangularize the matrix if necessary,
638:     # attempting to preserve dtype if possible.
639:     if np.array_equal(A, np.triu(A)):
640:         Z = None
641:         T = A
642:     else:
643:         if np.isrealobj(A):
644:             T, Z = schur(A)
645:             if not np.array_equal(T, np.triu(T)):
646:                 T, Z = rsf2csf(T, Z)
647:         else:
648:             T, Z = schur(A, output='complex')
649: 
650:     # Zeros on the diagonal of the triangular matrix are forbidden,
651:     # because the inverse scaling and squaring cannot deal with it.
652:     T_diag = np.diag(T)
653:     if np.count_nonzero(T_diag) != n:
654:         raise FractionalMatrixPowerError(
655:                 'cannot use inverse scaling and squaring to find '
656:                 'the fractional matrix power of a singular matrix')
657: 
658:     # If the triangular matrix is real and has a negative
659:     # entry on the diagonal, then force the matrix to be complex.
660:     if np.isrealobj(T) and np.min(T_diag) < 0:
661:         T = T.astype(complex)
662: 
663:     # Get the fractional power of the triangular matrix,
664:     # and de-triangularize it if necessary.
665:     U = _remainder_matrix_power_triu(T, t)
666:     if Z is not None:
667:         ZH = np.conjugate(Z).T
668:         return Z.dot(U).dot(ZH)
669:     else:
670:         return U
671: 
672: 
673: def _fractional_matrix_power(A, p):
674:     '''
675:     Compute the fractional power of a matrix.
676: 
677:     See the fractional_matrix_power docstring in matfuncs.py for more info.
678: 
679:     '''
680:     A = np.asarray(A)
681:     if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
682:         raise ValueError('expected a square matrix')
683:     if p == int(p):
684:         return np.linalg.matrix_power(A, int(p))
685:     # Compute singular values.
686:     s = svdvals(A)
687:     # Inverse scaling and squaring cannot deal with a singular matrix,
688:     # because the process of repeatedly taking square roots
689:     # would not converge to the identity matrix.
690:     if s[-1]:
691:         # Compute the condition number relative to matrix inversion,
692:         # and use this to decide between floor(p) and ceil(p).
693:         k2 = s[0] / s[-1]
694:         p1 = p - np.floor(p)
695:         p2 = p - np.ceil(p)
696:         if p1 * k2 ** (1 - p1) <= -p2 * k2:
697:             a = int(np.floor(p))
698:             b = p1
699:         else:
700:             a = int(np.ceil(p))
701:             b = p2
702:         try:
703:             R = _remainder_matrix_power(A, b)
704:             Q = np.linalg.matrix_power(A, a)
705:             return Q.dot(R)
706:         except np.linalg.LinAlgError:
707:             pass
708:     # If p is negative then we are going to give up.
709:     # If p is non-negative then we can fall back to generic funm.
710:     if p < 0:
711:         X = np.empty_like(A)
712:         X.fill(np.nan)
713:         return X
714:     else:
715:         p1 = p - np.floor(p)
716:         a = int(np.floor(p))
717:         b = p1
718:         R, info = funm(A, lambda x: pow(x, b), disp=False)
719:         Q = np.linalg.matrix_power(A, a)
720:         return Q.dot(R)
721: 
722: 
723: def _logm_triu(T):
724:     '''
725:     Compute matrix logarithm of an upper triangular matrix.
726: 
727:     The matrix logarithm is the inverse of
728:     expm: expm(logm(`T`)) == `T`
729: 
730:     Parameters
731:     ----------
732:     T : (N, N) array_like
733:         Upper triangular matrix whose logarithm to evaluate
734: 
735:     Returns
736:     -------
737:     logm : (N, N) ndarray
738:         Matrix logarithm of `T`
739: 
740:     References
741:     ----------
742:     .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
743:            "Improved Inverse Scaling and Squaring Algorithms
744:            for the Matrix Logarithm."
745:            SIAM Journal on Scientific Computing, 34 (4). C152-C169.
746:            ISSN 1095-7197
747: 
748:     .. [2] Nicholas J. Higham (2008)
749:            "Functions of Matrices: Theory and Computation"
750:            ISBN 978-0-898716-46-7
751: 
752:     .. [3] Nicholas J. Higham and Lijing lin (2011)
753:            "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
754:            SIAM Journal on Matrix Analysis and Applications,
755:            32 (3). pp. 1056-1078. ISSN 0895-4798
756: 
757:     '''
758:     T = np.asarray(T)
759:     if len(T.shape) != 2 or T.shape[0] != T.shape[1]:
760:         raise ValueError('expected an upper triangular square matrix')
761:     n, n = T.shape
762: 
763:     # Construct T0 with the appropriate type,
764:     # depending on the dtype and the spectrum of T.
765:     T_diag = np.diag(T)
766:     keep_it_real = np.isrealobj(T) and np.min(T_diag) >= 0
767:     if keep_it_real:
768:         T0 = T
769:     else:
770:         T0 = T.astype(complex)
771: 
772:     # Define bounds given in Table (2.1).
773:     theta = (None,
774:             1.59e-5, 2.31e-3, 1.94e-2, 6.21e-2,
775:             1.28e-1, 2.06e-1, 2.88e-1, 3.67e-1,
776:             4.39e-1, 5.03e-1, 5.60e-1, 6.09e-1,
777:             6.52e-1, 6.89e-1, 7.21e-1, 7.49e-1)
778: 
779:     R, s, m = _inverse_squaring_helper(T0, theta)
780: 
781:     # Evaluate U = 2**s r_m(T - I) using the partial fraction expansion (1.1).
782:     # This requires the nodes and weights
783:     # corresponding to degree-m Gauss-Legendre quadrature.
784:     # These quadrature arrays need to be transformed from the [-1, 1] interval
785:     # to the [0, 1] interval.
786:     nodes, weights = scipy.special.p_roots(m)
787:     nodes = nodes.real
788:     if nodes.shape != (m,) or weights.shape != (m,):
789:         raise Exception('internal error')
790:     nodes = 0.5 + 0.5 * nodes
791:     weights = 0.5 * weights
792:     ident = np.identity(n)
793:     U = np.zeros_like(R)
794:     for alpha, beta in zip(weights, nodes):
795:         U += solve_triangular(ident + beta*R, alpha*R)
796:     U *= np.exp2(s)
797: 
798:     # Skip this step if the principal branch
799:     # does not exist at T0; this happens when a diagonal entry of T0
800:     # is negative with imaginary part 0.
801:     has_principal_branch = all(x.real > 0 or x.imag != 0 for x in np.diag(T0))
802:     if has_principal_branch:
803: 
804:         # Recompute diagonal entries of U.
805:         U[np.diag_indices(n)] = np.log(np.diag(T0))
806: 
807:         # Recompute superdiagonal entries of U.
808:         # This indexing of this code should be renovated
809:         # when newer np.diagonal() becomes available.
810:         for i in range(n-1):
811:             l1 = T0[i, i]
812:             l2 = T0[i+1, i+1]
813:             t12 = T0[i, i+1]
814:             U[i, i+1] = _logm_superdiag_entry(l1, l2, t12)
815: 
816:     # Return the logm of the upper triangular matrix.
817:     if not np.array_equal(U, np.triu(U)):
818:         raise Exception('internal inconsistency')
819:     return U
820: 
821: 
822: def _logm_force_nonsingular_triangular_matrix(T, inplace=False):
823:     # The input matrix should be upper triangular.
824:     # The eps is ad hoc and is not meant to be machine precision.
825:     tri_eps = 1e-20
826:     abs_diag = np.absolute(np.diag(T))
827:     if np.any(abs_diag == 0):
828:         exact_singularity_msg = 'The logm input matrix is exactly singular.'
829:         warnings.warn(exact_singularity_msg, LogmExactlySingularWarning)
830:         if not inplace:
831:             T = T.copy()
832:         n = T.shape[0]
833:         for i in range(n):
834:             if not T[i, i]:
835:                 T[i, i] = tri_eps
836:     elif np.any(abs_diag < tri_eps):
837:         near_singularity_msg = 'The logm input matrix may be nearly singular.'
838:         warnings.warn(near_singularity_msg, LogmNearlySingularWarning)
839:     return T
840: 
841: 
842: def _logm(A):
843:     '''
844:     Compute the matrix logarithm.
845: 
846:     See the logm docstring in matfuncs.py for more info.
847: 
848:     Notes
849:     -----
850:     In this function we look at triangular matrices that are similar
851:     to the input matrix.  If any diagonal entry of such a triangular matrix
852:     is exactly zero then the original matrix is singular.
853:     The matrix logarithm does not exist for such matrices,
854:     but in such cases we will pretend that the diagonal entries that are zero
855:     are actually slightly positive by an ad-hoc amount, in the interest
856:     of returning something more useful than NaN.  This will cause a warning.
857: 
858:     '''
859:     A = np.asarray(A)
860:     if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
861:         raise ValueError('expected a square matrix')
862: 
863:     # If the input matrix dtype is integer then copy to a float dtype matrix.
864:     if issubclass(A.dtype.type, np.integer):
865:         A = np.asarray(A, dtype=float)
866: 
867:     keep_it_real = np.isrealobj(A)
868:     try:
869:         if np.array_equal(A, np.triu(A)):
870:             A = _logm_force_nonsingular_triangular_matrix(A)
871:             if np.min(np.diag(A)) < 0:
872:                 A = A.astype(complex)
873:             return _logm_triu(A)
874:         else:
875:             if keep_it_real:
876:                 T, Z = schur(A)
877:                 if not np.array_equal(T, np.triu(T)):
878:                     T, Z = rsf2csf(T, Z)
879:             else:
880:                 T, Z = schur(A, output='complex')
881:             T = _logm_force_nonsingular_triangular_matrix(T, inplace=True)
882:             U = _logm_triu(T)
883:             ZH = np.conjugate(Z).T
884:             return Z.dot(U).dot(ZH)
885:     except (SqrtmError, LogmError):
886:         X = np.empty_like(A)
887:         X.fill(np.nan)
888:         return X
889: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_32870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nMatrix functions that use Pade approximation with inverse scaling and squaring.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import warnings' statement (line 7)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_32871 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_32871) is not StypyTypeError):

    if (import_32871 != 'pyd_module'):
        __import__(import_32871)
        sys_modules_32872 = sys.modules[import_32871]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_32872.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_32871)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_32873 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg._matfuncs_sqrtm')

if (type(import_32873) is not StypyTypeError):

    if (import_32873 != 'pyd_module'):
        __import__(import_32873)
        sys_modules_32874 = sys.modules[import_32873]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg._matfuncs_sqrtm', sys_modules_32874.module_type_store, module_type_store, ['SqrtmError', '_sqrtm_triu'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_32874, sys_modules_32874.module_type_store, module_type_store)
    else:
        from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg._matfuncs_sqrtm', None, module_type_store, ['SqrtmError', '_sqrtm_triu'], [SqrtmError, _sqrtm_triu])

else:
    # Assigning a type to the variable 'scipy.linalg._matfuncs_sqrtm' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg._matfuncs_sqrtm', import_32873)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.linalg.decomp_schur import schur, rsf2csf' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_32875 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.decomp_schur')

if (type(import_32875) is not StypyTypeError):

    if (import_32875 != 'pyd_module'):
        __import__(import_32875)
        sys_modules_32876 = sys.modules[import_32875]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.decomp_schur', sys_modules_32876.module_type_store, module_type_store, ['schur', 'rsf2csf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_32876, sys_modules_32876.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp_schur import schur, rsf2csf

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.decomp_schur', None, module_type_store, ['schur', 'rsf2csf'], [schur, rsf2csf])

else:
    # Assigning a type to the variable 'scipy.linalg.decomp_schur' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.decomp_schur', import_32875)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.linalg.matfuncs import funm' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_32877 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.linalg.matfuncs')

if (type(import_32877) is not StypyTypeError):

    if (import_32877 != 'pyd_module'):
        __import__(import_32877)
        sys_modules_32878 = sys.modules[import_32877]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.linalg.matfuncs', sys_modules_32878.module_type_store, module_type_store, ['funm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_32878, sys_modules_32878.module_type_store, module_type_store)
    else:
        from scipy.linalg.matfuncs import funm

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.linalg.matfuncs', None, module_type_store, ['funm'], [funm])

else:
    # Assigning a type to the variable 'scipy.linalg.matfuncs' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.linalg.matfuncs', import_32877)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.linalg import svdvals, solve_triangular' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_32879 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg')

if (type(import_32879) is not StypyTypeError):

    if (import_32879 != 'pyd_module'):
        __import__(import_32879)
        sys_modules_32880 = sys.modules[import_32879]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg', sys_modules_32880.module_type_store, module_type_store, ['svdvals', 'solve_triangular'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_32880, sys_modules_32880.module_type_store, module_type_store)
    else:
        from scipy.linalg import svdvals, solve_triangular

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg', None, module_type_store, ['svdvals', 'solve_triangular'], [svdvals, solve_triangular])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg', import_32879)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.sparse.linalg.interface import LinearOperator' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_32881 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.linalg.interface')

if (type(import_32881) is not StypyTypeError):

    if (import_32881 != 'pyd_module'):
        __import__(import_32881)
        sys_modules_32882 = sys.modules[import_32881]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.linalg.interface', sys_modules_32882.module_type_store, module_type_store, ['LinearOperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_32882, sys_modules_32882.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.interface import LinearOperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.linalg.interface', None, module_type_store, ['LinearOperator'], [LinearOperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.interface' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.linalg.interface', import_32881)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.sparse.linalg import onenormest' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_32883 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.linalg')

if (type(import_32883) is not StypyTypeError):

    if (import_32883 != 'pyd_module'):
        __import__(import_32883)
        sys_modules_32884 = sys.modules[import_32883]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.linalg', sys_modules_32884.module_type_store, module_type_store, ['onenormest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_32884, sys_modules_32884.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import onenormest

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.linalg', None, module_type_store, ['onenormest'], [onenormest])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.linalg', import_32883)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import scipy.special' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_32885 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.special')

if (type(import_32885) is not StypyTypeError):

    if (import_32885 != 'pyd_module'):
        __import__(import_32885)
        sys_modules_32886 = sys.modules[import_32885]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.special', sys_modules_32886.module_type_store, module_type_store)
    else:
        import scipy.special

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.special', scipy.special, module_type_store)

else:
    # Assigning a type to the variable 'scipy.special' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.special', import_32885)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

# Declaration of the 'LogmRankWarning' class
# Getting the type of 'UserWarning' (line 20)
UserWarning_32887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'UserWarning')

class LogmRankWarning(UserWarning_32887, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 0, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogmRankWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LogmRankWarning' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'LogmRankWarning', LogmRankWarning)
# Declaration of the 'LogmExactlySingularWarning' class
# Getting the type of 'LogmRankWarning' (line 24)
LogmRankWarning_32888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 33), 'LogmRankWarning')

class LogmExactlySingularWarning(LogmRankWarning_32888, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 24, 0, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogmExactlySingularWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LogmExactlySingularWarning' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'LogmExactlySingularWarning', LogmExactlySingularWarning)
# Declaration of the 'LogmNearlySingularWarning' class
# Getting the type of 'LogmRankWarning' (line 28)
LogmRankWarning_32889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 32), 'LogmRankWarning')

class LogmNearlySingularWarning(LogmRankWarning_32889, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 28, 0, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogmNearlySingularWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LogmNearlySingularWarning' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'LogmNearlySingularWarning', LogmNearlySingularWarning)
# Declaration of the 'LogmError' class
# Getting the type of 'np' (line 32)
np_32890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'np')
# Obtaining the member 'linalg' of a type (line 32)
linalg_32891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 16), np_32890, 'linalg')
# Obtaining the member 'LinAlgError' of a type (line 32)
LinAlgError_32892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 16), linalg_32891, 'LinAlgError')

class LogmError(LinAlgError_32892, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 32, 0, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogmError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LogmError' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'LogmError', LogmError)
# Declaration of the 'FractionalMatrixPowerError' class
# Getting the type of 'np' (line 36)
np_32893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 33), 'np')
# Obtaining the member 'linalg' of a type (line 36)
linalg_32894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 33), np_32893, 'linalg')
# Obtaining the member 'LinAlgError' of a type (line 36)
LinAlgError_32895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 33), linalg_32894, 'LinAlgError')

class FractionalMatrixPowerError(LinAlgError_32895, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 36, 0, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FractionalMatrixPowerError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FractionalMatrixPowerError' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'FractionalMatrixPowerError', FractionalMatrixPowerError)
# Declaration of the '_MatrixM1PowerOperator' class
# Getting the type of 'LinearOperator' (line 41)
LinearOperator_32896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 29), 'LinearOperator')

class _MatrixM1PowerOperator(LinearOperator_32896, ):
    str_32897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, (-1)), 'str', '\n    A representation of the linear operator (A - I)^p.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_MatrixM1PowerOperator.__init__', ['A', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['A', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'A' (line 47)
        A_32898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'A')
        # Obtaining the member 'ndim' of a type (line 47)
        ndim_32899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), A_32898, 'ndim')
        int_32900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'int')
        # Applying the binary operator '!=' (line 47)
        result_ne_32901 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '!=', ndim_32899, int_32900)
        
        
        
        # Obtaining the type of the subscript
        int_32902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 34), 'int')
        # Getting the type of 'A' (line 47)
        A_32903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'A')
        # Obtaining the member 'shape' of a type (line 47)
        shape_32904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 26), A_32903, 'shape')
        # Obtaining the member '__getitem__' of a type (line 47)
        getitem___32905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 26), shape_32904, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 47)
        subscript_call_result_32906 = invoke(stypy.reporting.localization.Localization(__file__, 47, 26), getitem___32905, int_32902)
        
        
        # Obtaining the type of the subscript
        int_32907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 48), 'int')
        # Getting the type of 'A' (line 47)
        A_32908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 40), 'A')
        # Obtaining the member 'shape' of a type (line 47)
        shape_32909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 40), A_32908, 'shape')
        # Obtaining the member '__getitem__' of a type (line 47)
        getitem___32910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 40), shape_32909, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 47)
        subscript_call_result_32911 = invoke(stypy.reporting.localization.Localization(__file__, 47, 40), getitem___32910, int_32907)
        
        # Applying the binary operator '!=' (line 47)
        result_ne_32912 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 26), '!=', subscript_call_result_32906, subscript_call_result_32911)
        
        # Applying the binary operator 'or' (line 47)
        result_or_keyword_32913 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), 'or', result_ne_32901, result_ne_32912)
        
        # Testing the type of an if condition (line 47)
        if_condition_32914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 8), result_or_keyword_32913)
        # Assigning a type to the variable 'if_condition_32914' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'if_condition_32914', if_condition_32914)
        # SSA begins for if statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 48)
        # Processing the call arguments (line 48)
        str_32916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'str', 'expected A to be like a square matrix')
        # Processing the call keyword arguments (line 48)
        kwargs_32917 = {}
        # Getting the type of 'ValueError' (line 48)
        ValueError_32915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 48)
        ValueError_call_result_32918 = invoke(stypy.reporting.localization.Localization(__file__, 48, 18), ValueError_32915, *[str_32916], **kwargs_32917)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 48, 12), ValueError_call_result_32918, 'raise parameter', BaseException)
        # SSA join for if statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'p' (line 49)
        p_32919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'p')
        int_32920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 15), 'int')
        # Applying the binary operator '<' (line 49)
        result_lt_32921 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 11), '<', p_32919, int_32920)
        
        
        # Getting the type of 'p' (line 49)
        p_32922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 20), 'p')
        
        # Call to int(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'p' (line 49)
        p_32924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 29), 'p', False)
        # Processing the call keyword arguments (line 49)
        kwargs_32925 = {}
        # Getting the type of 'int' (line 49)
        int_32923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'int', False)
        # Calling int(args, kwargs) (line 49)
        int_call_result_32926 = invoke(stypy.reporting.localization.Localization(__file__, 49, 25), int_32923, *[p_32924], **kwargs_32925)
        
        # Applying the binary operator '!=' (line 49)
        result_ne_32927 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 20), '!=', p_32922, int_call_result_32926)
        
        # Applying the binary operator 'or' (line 49)
        result_or_keyword_32928 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 11), 'or', result_lt_32921, result_ne_32927)
        
        # Testing the type of an if condition (line 49)
        if_condition_32929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 8), result_or_keyword_32928)
        # Assigning a type to the variable 'if_condition_32929' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'if_condition_32929', if_condition_32929)
        # SSA begins for if statement (line 49)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 50)
        # Processing the call arguments (line 50)
        str_32931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'str', 'expected p to be a non-negative integer')
        # Processing the call keyword arguments (line 50)
        kwargs_32932 = {}
        # Getting the type of 'ValueError' (line 50)
        ValueError_32930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 50)
        ValueError_call_result_32933 = invoke(stypy.reporting.localization.Localization(__file__, 50, 18), ValueError_32930, *[str_32931], **kwargs_32932)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 50, 12), ValueError_call_result_32933, 'raise parameter', BaseException)
        # SSA join for if statement (line 49)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 51):
        
        # Assigning a Name to a Attribute (line 51):
        # Getting the type of 'A' (line 51)
        A_32934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'A')
        # Getting the type of 'self' (line 51)
        self_32935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self')
        # Setting the type of the member '_A' of a type (line 51)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_32935, '_A', A_32934)
        
        # Assigning a Name to a Attribute (line 52):
        
        # Assigning a Name to a Attribute (line 52):
        # Getting the type of 'p' (line 52)
        p_32936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'p')
        # Getting the type of 'self' (line 52)
        self_32937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self')
        # Setting the type of the member '_p' of a type (line 52)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_32937, '_p', p_32936)
        
        # Assigning a Attribute to a Attribute (line 53):
        
        # Assigning a Attribute to a Attribute (line 53):
        # Getting the type of 'A' (line 53)
        A_32938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'A')
        # Obtaining the member 'ndim' of a type (line 53)
        ndim_32939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 20), A_32938, 'ndim')
        # Getting the type of 'self' (line 53)
        self_32940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self')
        # Setting the type of the member 'ndim' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_32940, 'ndim', ndim_32939)
        
        # Assigning a Attribute to a Attribute (line 54):
        
        # Assigning a Attribute to a Attribute (line 54):
        # Getting the type of 'A' (line 54)
        A_32941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'A')
        # Obtaining the member 'shape' of a type (line 54)
        shape_32942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 21), A_32941, 'shape')
        # Getting the type of 'self' (line 54)
        self_32943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self')
        # Setting the type of the member 'shape' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_32943, 'shape', shape_32942)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matvec'
        module_type_store = module_type_store.open_function_context('_matvec', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _MatrixM1PowerOperator._matvec.__dict__.__setitem__('stypy_localization', localization)
        _MatrixM1PowerOperator._matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _MatrixM1PowerOperator._matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        _MatrixM1PowerOperator._matvec.__dict__.__setitem__('stypy_function_name', '_MatrixM1PowerOperator._matvec')
        _MatrixM1PowerOperator._matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _MatrixM1PowerOperator._matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        _MatrixM1PowerOperator._matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _MatrixM1PowerOperator._matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        _MatrixM1PowerOperator._matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        _MatrixM1PowerOperator._matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _MatrixM1PowerOperator._matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_MatrixM1PowerOperator._matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matvec(...)' code ##################

        
        
        # Call to range(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'self' (line 57)
        self_32945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'self', False)
        # Obtaining the member '_p' of a type (line 57)
        _p_32946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 23), self_32945, '_p')
        # Processing the call keyword arguments (line 57)
        kwargs_32947 = {}
        # Getting the type of 'range' (line 57)
        range_32944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'range', False)
        # Calling range(args, kwargs) (line 57)
        range_call_result_32948 = invoke(stypy.reporting.localization.Localization(__file__, 57, 17), range_32944, *[_p_32946], **kwargs_32947)
        
        # Testing the type of a for loop iterable (line 57)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 57, 8), range_call_result_32948)
        # Getting the type of the for loop variable (line 57)
        for_loop_var_32949 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 57, 8), range_call_result_32948)
        # Assigning a type to the variable 'i' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'i', for_loop_var_32949)
        # SSA begins for a for statement (line 57)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 58):
        
        # Assigning a BinOp to a Name (line 58):
        
        # Call to dot(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'x' (line 58)
        x_32953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'x', False)
        # Processing the call keyword arguments (line 58)
        kwargs_32954 = {}
        # Getting the type of 'self' (line 58)
        self_32950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'self', False)
        # Obtaining the member '_A' of a type (line 58)
        _A_32951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), self_32950, '_A')
        # Obtaining the member 'dot' of a type (line 58)
        dot_32952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), _A_32951, 'dot')
        # Calling dot(args, kwargs) (line 58)
        dot_call_result_32955 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), dot_32952, *[x_32953], **kwargs_32954)
        
        # Getting the type of 'x' (line 58)
        x_32956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 33), 'x')
        # Applying the binary operator '-' (line 58)
        result_sub_32957 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 16), '-', dot_call_result_32955, x_32956)
        
        # Assigning a type to the variable 'x' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'x', result_sub_32957)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'x' (line 59)
        x_32958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', x_32958)
        
        # ################# End of '_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_32959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32959)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matvec'
        return stypy_return_type_32959


    @norecursion
    def _rmatvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rmatvec'
        module_type_store = module_type_store.open_function_context('_rmatvec', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _MatrixM1PowerOperator._rmatvec.__dict__.__setitem__('stypy_localization', localization)
        _MatrixM1PowerOperator._rmatvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _MatrixM1PowerOperator._rmatvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        _MatrixM1PowerOperator._rmatvec.__dict__.__setitem__('stypy_function_name', '_MatrixM1PowerOperator._rmatvec')
        _MatrixM1PowerOperator._rmatvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _MatrixM1PowerOperator._rmatvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        _MatrixM1PowerOperator._rmatvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _MatrixM1PowerOperator._rmatvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        _MatrixM1PowerOperator._rmatvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        _MatrixM1PowerOperator._rmatvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _MatrixM1PowerOperator._rmatvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_MatrixM1PowerOperator._rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rmatvec(...)' code ##################

        
        
        # Call to range(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_32961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'self', False)
        # Obtaining the member '_p' of a type (line 62)
        _p_32962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 23), self_32961, '_p')
        # Processing the call keyword arguments (line 62)
        kwargs_32963 = {}
        # Getting the type of 'range' (line 62)
        range_32960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'range', False)
        # Calling range(args, kwargs) (line 62)
        range_call_result_32964 = invoke(stypy.reporting.localization.Localization(__file__, 62, 17), range_32960, *[_p_32962], **kwargs_32963)
        
        # Testing the type of a for loop iterable (line 62)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 8), range_call_result_32964)
        # Getting the type of the for loop variable (line 62)
        for_loop_var_32965 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 8), range_call_result_32964)
        # Assigning a type to the variable 'i' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'i', for_loop_var_32965)
        # SSA begins for a for statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 63):
        
        # Assigning a BinOp to a Name (line 63):
        
        # Call to dot(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'self' (line 63)
        self_32968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'self', False)
        # Obtaining the member '_A' of a type (line 63)
        _A_32969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 22), self_32968, '_A')
        # Processing the call keyword arguments (line 63)
        kwargs_32970 = {}
        # Getting the type of 'x' (line 63)
        x_32966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'x', False)
        # Obtaining the member 'dot' of a type (line 63)
        dot_32967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 16), x_32966, 'dot')
        # Calling dot(args, kwargs) (line 63)
        dot_call_result_32971 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), dot_32967, *[_A_32969], **kwargs_32970)
        
        # Getting the type of 'x' (line 63)
        x_32972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 33), 'x')
        # Applying the binary operator '-' (line 63)
        result_sub_32973 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 16), '-', dot_call_result_32971, x_32972)
        
        # Assigning a type to the variable 'x' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'x', result_sub_32973)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'x' (line 64)
        x_32974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', x_32974)
        
        # ################# End of '_rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_32975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32975)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rmatvec'
        return stypy_return_type_32975


    @norecursion
    def _matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matmat'
        module_type_store = module_type_store.open_function_context('_matmat', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _MatrixM1PowerOperator._matmat.__dict__.__setitem__('stypy_localization', localization)
        _MatrixM1PowerOperator._matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _MatrixM1PowerOperator._matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        _MatrixM1PowerOperator._matmat.__dict__.__setitem__('stypy_function_name', '_MatrixM1PowerOperator._matmat')
        _MatrixM1PowerOperator._matmat.__dict__.__setitem__('stypy_param_names_list', ['X'])
        _MatrixM1PowerOperator._matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        _MatrixM1PowerOperator._matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _MatrixM1PowerOperator._matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        _MatrixM1PowerOperator._matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        _MatrixM1PowerOperator._matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _MatrixM1PowerOperator._matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_MatrixM1PowerOperator._matmat', ['X'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matmat', localization, ['X'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matmat(...)' code ##################

        
        
        # Call to range(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'self' (line 67)
        self_32977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'self', False)
        # Obtaining the member '_p' of a type (line 67)
        _p_32978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 23), self_32977, '_p')
        # Processing the call keyword arguments (line 67)
        kwargs_32979 = {}
        # Getting the type of 'range' (line 67)
        range_32976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'range', False)
        # Calling range(args, kwargs) (line 67)
        range_call_result_32980 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), range_32976, *[_p_32978], **kwargs_32979)
        
        # Testing the type of a for loop iterable (line 67)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 67, 8), range_call_result_32980)
        # Getting the type of the for loop variable (line 67)
        for_loop_var_32981 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 67, 8), range_call_result_32980)
        # Assigning a type to the variable 'i' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'i', for_loop_var_32981)
        # SSA begins for a for statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 68):
        
        # Assigning a BinOp to a Name (line 68):
        
        # Call to dot(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'X' (line 68)
        X_32985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 28), 'X', False)
        # Processing the call keyword arguments (line 68)
        kwargs_32986 = {}
        # Getting the type of 'self' (line 68)
        self_32982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'self', False)
        # Obtaining the member '_A' of a type (line 68)
        _A_32983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), self_32982, '_A')
        # Obtaining the member 'dot' of a type (line 68)
        dot_32984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), _A_32983, 'dot')
        # Calling dot(args, kwargs) (line 68)
        dot_call_result_32987 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), dot_32984, *[X_32985], **kwargs_32986)
        
        # Getting the type of 'X' (line 68)
        X_32988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 33), 'X')
        # Applying the binary operator '-' (line 68)
        result_sub_32989 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 16), '-', dot_call_result_32987, X_32988)
        
        # Assigning a type to the variable 'X' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'X', result_sub_32989)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'X' (line 69)
        X_32990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'X')
        # Assigning a type to the variable 'stypy_return_type' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stypy_return_type', X_32990)
        
        # ################# End of '_matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_32991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32991)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matmat'
        return stypy_return_type_32991


    @norecursion
    def _adjoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_adjoint'
        module_type_store = module_type_store.open_function_context('_adjoint', 71, 4, False)
        # Assigning a type to the variable 'self' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _MatrixM1PowerOperator._adjoint.__dict__.__setitem__('stypy_localization', localization)
        _MatrixM1PowerOperator._adjoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _MatrixM1PowerOperator._adjoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        _MatrixM1PowerOperator._adjoint.__dict__.__setitem__('stypy_function_name', '_MatrixM1PowerOperator._adjoint')
        _MatrixM1PowerOperator._adjoint.__dict__.__setitem__('stypy_param_names_list', [])
        _MatrixM1PowerOperator._adjoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        _MatrixM1PowerOperator._adjoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _MatrixM1PowerOperator._adjoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        _MatrixM1PowerOperator._adjoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        _MatrixM1PowerOperator._adjoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _MatrixM1PowerOperator._adjoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_MatrixM1PowerOperator._adjoint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_adjoint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_adjoint(...)' code ##################

        
        # Call to _MatrixM1PowerOperator(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'self' (line 72)
        self_32993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 38), 'self', False)
        # Obtaining the member '_A' of a type (line 72)
        _A_32994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 38), self_32993, '_A')
        # Obtaining the member 'T' of a type (line 72)
        T_32995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 38), _A_32994, 'T')
        # Getting the type of 'self' (line 72)
        self_32996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 49), 'self', False)
        # Obtaining the member '_p' of a type (line 72)
        _p_32997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 49), self_32996, '_p')
        # Processing the call keyword arguments (line 72)
        kwargs_32998 = {}
        # Getting the type of '_MatrixM1PowerOperator' (line 72)
        _MatrixM1PowerOperator_32992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), '_MatrixM1PowerOperator', False)
        # Calling _MatrixM1PowerOperator(args, kwargs) (line 72)
        _MatrixM1PowerOperator_call_result_32999 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), _MatrixM1PowerOperator_32992, *[T_32995, _p_32997], **kwargs_32998)
        
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', _MatrixM1PowerOperator_call_result_32999)
        
        # ################# End of '_adjoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_adjoint' in the type store
        # Getting the type of 'stypy_return_type' (line 71)
        stypy_return_type_33000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33000)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_adjoint'
        return stypy_return_type_33000


# Assigning a type to the variable '_MatrixM1PowerOperator' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '_MatrixM1PowerOperator', _MatrixM1PowerOperator)

@norecursion
def _onenormest_m1_power(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_33001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 10), 'int')
    int_33002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 19), 'int')
    # Getting the type of 'False' (line 77)
    False_33003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 32), 'False')
    # Getting the type of 'False' (line 77)
    False_33004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 49), 'False')
    defaults = [int_33001, int_33002, False_33003, False_33004]
    # Create a new context for function '_onenormest_m1_power'
    module_type_store = module_type_store.open_function_context('_onenormest_m1_power', 76, 0, False)
    
    # Passed parameters checking function
    _onenormest_m1_power.stypy_localization = localization
    _onenormest_m1_power.stypy_type_of_self = None
    _onenormest_m1_power.stypy_type_store = module_type_store
    _onenormest_m1_power.stypy_function_name = '_onenormest_m1_power'
    _onenormest_m1_power.stypy_param_names_list = ['A', 'p', 't', 'itmax', 'compute_v', 'compute_w']
    _onenormest_m1_power.stypy_varargs_param_name = None
    _onenormest_m1_power.stypy_kwargs_param_name = None
    _onenormest_m1_power.stypy_call_defaults = defaults
    _onenormest_m1_power.stypy_call_varargs = varargs
    _onenormest_m1_power.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_onenormest_m1_power', ['A', 'p', 't', 'itmax', 'compute_v', 'compute_w'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_onenormest_m1_power', localization, ['A', 'p', 't', 'itmax', 'compute_v', 'compute_w'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_onenormest_m1_power(...)' code ##################

    str_33005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, (-1)), 'str', '\n    Efficiently estimate the 1-norm of (A - I)^p.\n\n    Parameters\n    ----------\n    A : ndarray\n        Matrix whose 1-norm of a power is to be computed.\n    p : int\n        Non-negative integer power.\n    t : int, optional\n        A positive parameter controlling the tradeoff between\n        accuracy versus time and memory usage.\n        Larger values take longer and use more memory\n        but give more accurate output.\n    itmax : int, optional\n        Use at most this many iterations.\n    compute_v : bool, optional\n        Request a norm-maximizing linear operator input vector if True.\n    compute_w : bool, optional\n        Request a norm-maximizing linear operator output vector if True.\n\n    Returns\n    -------\n    est : float\n        An underestimate of the 1-norm of the sparse matrix.\n    v : ndarray, optional\n        The vector such that ||Av||_1 == est*||v||_1.\n        It can be thought of as an input to the linear operator\n        that gives an output with particularly large norm.\n    w : ndarray, optional\n        The vector Av which has relatively large 1-norm.\n        It can be thought of as an output of the linear operator\n        that is relatively large in norm compared to the input.\n\n    ')
    
    # Call to onenormest(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Call to _MatrixM1PowerOperator(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'A' (line 113)
    A_33008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 45), 'A', False)
    # Getting the type of 'p' (line 113)
    p_33009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 48), 'p', False)
    # Processing the call keyword arguments (line 113)
    kwargs_33010 = {}
    # Getting the type of '_MatrixM1PowerOperator' (line 113)
    _MatrixM1PowerOperator_33007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 22), '_MatrixM1PowerOperator', False)
    # Calling _MatrixM1PowerOperator(args, kwargs) (line 113)
    _MatrixM1PowerOperator_call_result_33011 = invoke(stypy.reporting.localization.Localization(__file__, 113, 22), _MatrixM1PowerOperator_33007, *[A_33008, p_33009], **kwargs_33010)
    
    # Processing the call keyword arguments (line 113)
    # Getting the type of 't' (line 114)
    t_33012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 't', False)
    keyword_33013 = t_33012
    # Getting the type of 'itmax' (line 114)
    itmax_33014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'itmax', False)
    keyword_33015 = itmax_33014
    # Getting the type of 'compute_v' (line 114)
    compute_v_33016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 40), 'compute_v', False)
    keyword_33017 = compute_v_33016
    # Getting the type of 'compute_w' (line 114)
    compute_w_33018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 61), 'compute_w', False)
    keyword_33019 = compute_w_33018
    kwargs_33020 = {'compute_w': keyword_33019, 't': keyword_33013, 'itmax': keyword_33015, 'compute_v': keyword_33017}
    # Getting the type of 'onenormest' (line 113)
    onenormest_33006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'onenormest', False)
    # Calling onenormest(args, kwargs) (line 113)
    onenormest_call_result_33021 = invoke(stypy.reporting.localization.Localization(__file__, 113, 11), onenormest_33006, *[_MatrixM1PowerOperator_call_result_33011], **kwargs_33020)
    
    # Assigning a type to the variable 'stypy_return_type' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type', onenormest_call_result_33021)
    
    # ################# End of '_onenormest_m1_power(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_onenormest_m1_power' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_33022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33022)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_onenormest_m1_power'
    return stypy_return_type_33022

# Assigning a type to the variable '_onenormest_m1_power' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), '_onenormest_m1_power', _onenormest_m1_power)

@norecursion
def _unwindk(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_unwindk'
    module_type_store = module_type_store.open_function_context('_unwindk', 117, 0, False)
    
    # Passed parameters checking function
    _unwindk.stypy_localization = localization
    _unwindk.stypy_type_of_self = None
    _unwindk.stypy_type_store = module_type_store
    _unwindk.stypy_function_name = '_unwindk'
    _unwindk.stypy_param_names_list = ['z']
    _unwindk.stypy_varargs_param_name = None
    _unwindk.stypy_kwargs_param_name = None
    _unwindk.stypy_call_defaults = defaults
    _unwindk.stypy_call_varargs = varargs
    _unwindk.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_unwindk', ['z'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_unwindk', localization, ['z'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_unwindk(...)' code ##################

    str_33023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, (-1)), 'str', '\n    Compute the scalar unwinding number.\n\n    Uses Eq. (5.3) in [1]_, and should be equal to (z - log(exp(z)) / (2 pi i).\n    Note that this definition differs in sign from the original definition\n    in equations (5, 6) in [2]_.  The sign convention is justified in [3]_.\n\n    Parameters\n    ----------\n    z : complex\n        A complex number.\n\n    Returns\n    -------\n    unwinding_number : integer\n        The scalar unwinding number of z.\n\n    References\n    ----------\n    .. [1] Nicholas J. Higham and Lijing lin (2011)\n           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."\n           SIAM Journal on Matrix Analysis and Applications,\n           32 (3). pp. 1056-1078. ISSN 0895-4798\n\n    .. [2] Robert M. Corless and David J. Jeffrey,\n           "The unwinding number." Newsletter ACM SIGSAM Bulletin\n           Volume 30, Issue 2, June 1996, Pages 28-35.\n\n    .. [3] Russell Bradford and Robert M. Corless and James H. Davenport and\n           David J. Jeffrey and Stephen M. Watt,\n           "Reasoning about the elementary functions of complex analysis"\n           Annals of Mathematics and Artificial Intelligence,\n           36: 303-318, 2002.\n\n    ')
    
    # Call to int(...): (line 153)
    # Processing the call arguments (line 153)
    
    # Call to ceil(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'z' (line 153)
    z_33027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'z', False)
    # Obtaining the member 'imag' of a type (line 153)
    imag_33028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 24), z_33027, 'imag')
    # Getting the type of 'np' (line 153)
    np_33029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'np', False)
    # Obtaining the member 'pi' of a type (line 153)
    pi_33030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 33), np_33029, 'pi')
    # Applying the binary operator '-' (line 153)
    result_sub_33031 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 24), '-', imag_33028, pi_33030)
    
    int_33032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 43), 'int')
    # Getting the type of 'np' (line 153)
    np_33033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 45), 'np', False)
    # Obtaining the member 'pi' of a type (line 153)
    pi_33034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 45), np_33033, 'pi')
    # Applying the binary operator '*' (line 153)
    result_mul_33035 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 43), '*', int_33032, pi_33034)
    
    # Applying the binary operator 'div' (line 153)
    result_div_33036 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 23), 'div', result_sub_33031, result_mul_33035)
    
    # Processing the call keyword arguments (line 153)
    kwargs_33037 = {}
    # Getting the type of 'np' (line 153)
    np_33025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'np', False)
    # Obtaining the member 'ceil' of a type (line 153)
    ceil_33026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 15), np_33025, 'ceil')
    # Calling ceil(args, kwargs) (line 153)
    ceil_call_result_33038 = invoke(stypy.reporting.localization.Localization(__file__, 153, 15), ceil_33026, *[result_div_33036], **kwargs_33037)
    
    # Processing the call keyword arguments (line 153)
    kwargs_33039 = {}
    # Getting the type of 'int' (line 153)
    int_33024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'int', False)
    # Calling int(args, kwargs) (line 153)
    int_call_result_33040 = invoke(stypy.reporting.localization.Localization(__file__, 153, 11), int_33024, *[ceil_call_result_33038], **kwargs_33039)
    
    # Assigning a type to the variable 'stypy_return_type' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type', int_call_result_33040)
    
    # ################# End of '_unwindk(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_unwindk' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_33041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33041)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_unwindk'
    return stypy_return_type_33041

# Assigning a type to the variable '_unwindk' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), '_unwindk', _unwindk)

@norecursion
def _briggs_helper_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_briggs_helper_function'
    module_type_store = module_type_store.open_function_context('_briggs_helper_function', 156, 0, False)
    
    # Passed parameters checking function
    _briggs_helper_function.stypy_localization = localization
    _briggs_helper_function.stypy_type_of_self = None
    _briggs_helper_function.stypy_type_store = module_type_store
    _briggs_helper_function.stypy_function_name = '_briggs_helper_function'
    _briggs_helper_function.stypy_param_names_list = ['a', 'k']
    _briggs_helper_function.stypy_varargs_param_name = None
    _briggs_helper_function.stypy_kwargs_param_name = None
    _briggs_helper_function.stypy_call_defaults = defaults
    _briggs_helper_function.stypy_call_varargs = varargs
    _briggs_helper_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_briggs_helper_function', ['a', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_briggs_helper_function', localization, ['a', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_briggs_helper_function(...)' code ##################

    str_33042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, (-1)), 'str', '\n    Computes r = a^(1 / (2^k)) - 1.\n\n    This is algorithm (2) of [1]_.\n    The purpose is to avoid a danger of subtractive cancellation.\n    For more computational efficiency it should probably be cythonized.\n\n    Parameters\n    ----------\n    a : complex\n        A complex number.\n    k : integer\n        A nonnegative integer.\n\n    Returns\n    -------\n    r : complex\n        The value r = a^(1 / (2^k)) - 1 computed with less cancellation.\n\n    Notes\n    -----\n    The algorithm as formulated in the reference does not handle k=0 or k=1\n    correctly, so these are special-cased in this implementation.\n    This function is intended to not allow `a` to belong to the closed\n    negative real axis, but this constraint is relaxed.\n\n    References\n    ----------\n    .. [1] Awad H. Al-Mohy (2012)\n           "A more accurate Briggs method for the logarithm",\n           Numerical Algorithms, 59 : 393--402.\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'k' (line 190)
    k_33043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 7), 'k')
    int_33044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 11), 'int')
    # Applying the binary operator '<' (line 190)
    result_lt_33045 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 7), '<', k_33043, int_33044)
    
    
    
    # Call to int(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'k' (line 190)
    k_33047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'k', False)
    # Processing the call keyword arguments (line 190)
    kwargs_33048 = {}
    # Getting the type of 'int' (line 190)
    int_33046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'int', False)
    # Calling int(args, kwargs) (line 190)
    int_call_result_33049 = invoke(stypy.reporting.localization.Localization(__file__, 190, 16), int_33046, *[k_33047], **kwargs_33048)
    
    # Getting the type of 'k' (line 190)
    k_33050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 26), 'k')
    # Applying the binary operator '!=' (line 190)
    result_ne_33051 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 16), '!=', int_call_result_33049, k_33050)
    
    # Applying the binary operator 'or' (line 190)
    result_or_keyword_33052 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 7), 'or', result_lt_33045, result_ne_33051)
    
    # Testing the type of an if condition (line 190)
    if_condition_33053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 4), result_or_keyword_33052)
    # Assigning a type to the variable 'if_condition_33053' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'if_condition_33053', if_condition_33053)
    # SSA begins for if statement (line 190)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 191)
    # Processing the call arguments (line 191)
    str_33055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 25), 'str', 'expected a nonnegative integer k')
    # Processing the call keyword arguments (line 191)
    kwargs_33056 = {}
    # Getting the type of 'ValueError' (line 191)
    ValueError_33054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 191)
    ValueError_call_result_33057 = invoke(stypy.reporting.localization.Localization(__file__, 191, 14), ValueError_33054, *[str_33055], **kwargs_33056)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 191, 8), ValueError_call_result_33057, 'raise parameter', BaseException)
    # SSA join for if statement (line 190)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'k' (line 192)
    k_33058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 7), 'k')
    int_33059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 12), 'int')
    # Applying the binary operator '==' (line 192)
    result_eq_33060 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 7), '==', k_33058, int_33059)
    
    # Testing the type of an if condition (line 192)
    if_condition_33061 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 4), result_eq_33060)
    # Assigning a type to the variable 'if_condition_33061' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'if_condition_33061', if_condition_33061)
    # SSA begins for if statement (line 192)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'a' (line 193)
    a_33062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'a')
    int_33063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 19), 'int')
    # Applying the binary operator '-' (line 193)
    result_sub_33064 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 15), '-', a_33062, int_33063)
    
    # Assigning a type to the variable 'stypy_return_type' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'stypy_return_type', result_sub_33064)
    # SSA branch for the else part of an if statement (line 192)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'k' (line 194)
    k_33065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 9), 'k')
    int_33066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 14), 'int')
    # Applying the binary operator '==' (line 194)
    result_eq_33067 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 9), '==', k_33065, int_33066)
    
    # Testing the type of an if condition (line 194)
    if_condition_33068 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 9), result_eq_33067)
    # Assigning a type to the variable 'if_condition_33068' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 9), 'if_condition_33068', if_condition_33068)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to sqrt(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'a' (line 195)
    a_33071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 23), 'a', False)
    # Processing the call keyword arguments (line 195)
    kwargs_33072 = {}
    # Getting the type of 'np' (line 195)
    np_33069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 195)
    sqrt_33070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 15), np_33069, 'sqrt')
    # Calling sqrt(args, kwargs) (line 195)
    sqrt_call_result_33073 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), sqrt_33070, *[a_33071], **kwargs_33072)
    
    int_33074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 28), 'int')
    # Applying the binary operator '-' (line 195)
    result_sub_33075 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 15), '-', sqrt_call_result_33073, int_33074)
    
    # Assigning a type to the variable 'stypy_return_type' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'stypy_return_type', result_sub_33075)
    # SSA branch for the else part of an if statement (line 194)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 197):
    
    # Assigning a Name to a Name (line 197):
    # Getting the type of 'k' (line 197)
    k_33076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'k')
    # Assigning a type to the variable 'k_hat' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'k_hat', k_33076)
    
    
    
    # Call to angle(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'a' (line 198)
    a_33079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'a', False)
    # Processing the call keyword arguments (line 198)
    kwargs_33080 = {}
    # Getting the type of 'np' (line 198)
    np_33077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'np', False)
    # Obtaining the member 'angle' of a type (line 198)
    angle_33078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 11), np_33077, 'angle')
    # Calling angle(args, kwargs) (line 198)
    angle_call_result_33081 = invoke(stypy.reporting.localization.Localization(__file__, 198, 11), angle_33078, *[a_33079], **kwargs_33080)
    
    # Getting the type of 'np' (line 198)
    np_33082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 26), 'np')
    # Obtaining the member 'pi' of a type (line 198)
    pi_33083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 26), np_33082, 'pi')
    int_33084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 34), 'int')
    # Applying the binary operator 'div' (line 198)
    result_div_33085 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 26), 'div', pi_33083, int_33084)
    
    # Applying the binary operator '>=' (line 198)
    result_ge_33086 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), '>=', angle_call_result_33081, result_div_33085)
    
    # Testing the type of an if condition (line 198)
    if_condition_33087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), result_ge_33086)
    # Assigning a type to the variable 'if_condition_33087' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_33087', if_condition_33087)
    # SSA begins for if statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 199):
    
    # Assigning a Call to a Name (line 199):
    
    # Call to sqrt(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'a' (line 199)
    a_33090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 24), 'a', False)
    # Processing the call keyword arguments (line 199)
    kwargs_33091 = {}
    # Getting the type of 'np' (line 199)
    np_33088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 199)
    sqrt_33089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), np_33088, 'sqrt')
    # Calling sqrt(args, kwargs) (line 199)
    sqrt_call_result_33092 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), sqrt_33089, *[a_33090], **kwargs_33091)
    
    # Assigning a type to the variable 'a' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'a', sqrt_call_result_33092)
    
    # Assigning a BinOp to a Name (line 200):
    
    # Assigning a BinOp to a Name (line 200):
    # Getting the type of 'k' (line 200)
    k_33093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'k')
    int_33094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 24), 'int')
    # Applying the binary operator '-' (line 200)
    result_sub_33095 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 20), '-', k_33093, int_33094)
    
    # Assigning a type to the variable 'k_hat' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'k_hat', result_sub_33095)
    # SSA join for if statement (line 198)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 201):
    
    # Assigning a BinOp to a Name (line 201):
    # Getting the type of 'a' (line 201)
    a_33096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 13), 'a')
    int_33097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 17), 'int')
    # Applying the binary operator '-' (line 201)
    result_sub_33098 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 13), '-', a_33096, int_33097)
    
    # Assigning a type to the variable 'z0' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'z0', result_sub_33098)
    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Call to a Name (line 202):
    
    # Call to sqrt(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'a' (line 202)
    a_33101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'a', False)
    # Processing the call keyword arguments (line 202)
    kwargs_33102 = {}
    # Getting the type of 'np' (line 202)
    np_33099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 202)
    sqrt_33100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 12), np_33099, 'sqrt')
    # Calling sqrt(args, kwargs) (line 202)
    sqrt_call_result_33103 = invoke(stypy.reporting.localization.Localization(__file__, 202, 12), sqrt_33100, *[a_33101], **kwargs_33102)
    
    # Assigning a type to the variable 'a' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'a', sqrt_call_result_33103)
    
    # Assigning a BinOp to a Name (line 203):
    
    # Assigning a BinOp to a Name (line 203):
    int_33104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 12), 'int')
    # Getting the type of 'a' (line 203)
    a_33105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'a')
    # Applying the binary operator '+' (line 203)
    result_add_33106 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 12), '+', int_33104, a_33105)
    
    # Assigning a type to the variable 'r' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'r', result_add_33106)
    
    
    # Call to range(...): (line 204)
    # Processing the call arguments (line 204)
    int_33108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 23), 'int')
    # Getting the type of 'k_hat' (line 204)
    k_hat_33109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 26), 'k_hat', False)
    # Processing the call keyword arguments (line 204)
    kwargs_33110 = {}
    # Getting the type of 'range' (line 204)
    range_33107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 17), 'range', False)
    # Calling range(args, kwargs) (line 204)
    range_call_result_33111 = invoke(stypy.reporting.localization.Localization(__file__, 204, 17), range_33107, *[int_33108, k_hat_33109], **kwargs_33110)
    
    # Testing the type of a for loop iterable (line 204)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 204, 8), range_call_result_33111)
    # Getting the type of the for loop variable (line 204)
    for_loop_var_33112 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 204, 8), range_call_result_33111)
    # Assigning a type to the variable 'j' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'j', for_loop_var_33112)
    # SSA begins for a for statement (line 204)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 205):
    
    # Assigning a Call to a Name (line 205):
    
    # Call to sqrt(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'a' (line 205)
    a_33115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 'a', False)
    # Processing the call keyword arguments (line 205)
    kwargs_33116 = {}
    # Getting the type of 'np' (line 205)
    np_33113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 205)
    sqrt_33114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 16), np_33113, 'sqrt')
    # Calling sqrt(args, kwargs) (line 205)
    sqrt_call_result_33117 = invoke(stypy.reporting.localization.Localization(__file__, 205, 16), sqrt_33114, *[a_33115], **kwargs_33116)
    
    # Assigning a type to the variable 'a' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'a', sqrt_call_result_33117)
    
    # Assigning a BinOp to a Name (line 206):
    
    # Assigning a BinOp to a Name (line 206):
    # Getting the type of 'r' (line 206)
    r_33118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'r')
    int_33119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 21), 'int')
    # Getting the type of 'a' (line 206)
    a_33120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 25), 'a')
    # Applying the binary operator '+' (line 206)
    result_add_33121 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 21), '+', int_33119, a_33120)
    
    # Applying the binary operator '*' (line 206)
    result_mul_33122 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 16), '*', r_33118, result_add_33121)
    
    # Assigning a type to the variable 'r' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'r', result_mul_33122)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 207):
    
    # Assigning a BinOp to a Name (line 207):
    # Getting the type of 'z0' (line 207)
    z0_33123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'z0')
    # Getting the type of 'r' (line 207)
    r_33124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 17), 'r')
    # Applying the binary operator 'div' (line 207)
    result_div_33125 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 12), 'div', z0_33123, r_33124)
    
    # Assigning a type to the variable 'r' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'r', result_div_33125)
    # Getting the type of 'r' (line 208)
    r_33126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', r_33126)
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 192)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_briggs_helper_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_briggs_helper_function' in the type store
    # Getting the type of 'stypy_return_type' (line 156)
    stypy_return_type_33127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33127)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_briggs_helper_function'
    return stypy_return_type_33127

# Assigning a type to the variable '_briggs_helper_function' (line 156)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 0), '_briggs_helper_function', _briggs_helper_function)

@norecursion
def _fractional_power_superdiag_entry(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fractional_power_superdiag_entry'
    module_type_store = module_type_store.open_function_context('_fractional_power_superdiag_entry', 211, 0, False)
    
    # Passed parameters checking function
    _fractional_power_superdiag_entry.stypy_localization = localization
    _fractional_power_superdiag_entry.stypy_type_of_self = None
    _fractional_power_superdiag_entry.stypy_type_store = module_type_store
    _fractional_power_superdiag_entry.stypy_function_name = '_fractional_power_superdiag_entry'
    _fractional_power_superdiag_entry.stypy_param_names_list = ['l1', 'l2', 't12', 'p']
    _fractional_power_superdiag_entry.stypy_varargs_param_name = None
    _fractional_power_superdiag_entry.stypy_kwargs_param_name = None
    _fractional_power_superdiag_entry.stypy_call_defaults = defaults
    _fractional_power_superdiag_entry.stypy_call_varargs = varargs
    _fractional_power_superdiag_entry.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fractional_power_superdiag_entry', ['l1', 'l2', 't12', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fractional_power_superdiag_entry', localization, ['l1', 'l2', 't12', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fractional_power_superdiag_entry(...)' code ##################

    str_33128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, (-1)), 'str', '\n    Compute a superdiagonal entry of a fractional matrix power.\n\n    This is Eq. (5.6) in [1]_.\n\n    Parameters\n    ----------\n    l1 : complex\n        A diagonal entry of the matrix.\n    l2 : complex\n        A diagonal entry of the matrix.\n    t12 : complex\n        A superdiagonal entry of the matrix.\n    p : float\n        A fractional power.\n\n    Returns\n    -------\n    f12 : complex\n        A superdiagonal entry of the fractional matrix power.\n\n    Notes\n    -----\n    Care has been taken to return a real number if possible when\n    all of the inputs are real numbers.\n\n    References\n    ----------\n    .. [1] Nicholas J. Higham and Lijing lin (2011)\n           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."\n           SIAM Journal on Matrix Analysis and Applications,\n           32 (3). pp. 1056-1078. ISSN 0895-4798\n\n    ')
    
    
    # Getting the type of 'l1' (line 246)
    l1_33129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 7), 'l1')
    # Getting the type of 'l2' (line 246)
    l2_33130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 13), 'l2')
    # Applying the binary operator '==' (line 246)
    result_eq_33131 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 7), '==', l1_33129, l2_33130)
    
    # Testing the type of an if condition (line 246)
    if_condition_33132 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 4), result_eq_33131)
    # Assigning a type to the variable 'if_condition_33132' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'if_condition_33132', if_condition_33132)
    # SSA begins for if statement (line 246)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 247):
    
    # Assigning a BinOp to a Name (line 247):
    # Getting the type of 't12' (line 247)
    t12_33133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 14), 't12')
    # Getting the type of 'p' (line 247)
    p_33134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 20), 'p')
    # Applying the binary operator '*' (line 247)
    result_mul_33135 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 14), '*', t12_33133, p_33134)
    
    # Getting the type of 'l1' (line 247)
    l1_33136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 24), 'l1')
    # Getting the type of 'p' (line 247)
    p_33137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 29), 'p')
    int_33138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 31), 'int')
    # Applying the binary operator '-' (line 247)
    result_sub_33139 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 29), '-', p_33137, int_33138)
    
    # Applying the binary operator '**' (line 247)
    result_pow_33140 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 24), '**', l1_33136, result_sub_33139)
    
    # Applying the binary operator '*' (line 247)
    result_mul_33141 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 22), '*', result_mul_33135, result_pow_33140)
    
    # Assigning a type to the variable 'f12' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'f12', result_mul_33141)
    # SSA branch for the else part of an if statement (line 246)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to abs(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'l2' (line 248)
    l2_33143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 13), 'l2', False)
    # Getting the type of 'l1' (line 248)
    l1_33144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 18), 'l1', False)
    # Applying the binary operator '-' (line 248)
    result_sub_33145 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 13), '-', l2_33143, l1_33144)
    
    # Processing the call keyword arguments (line 248)
    kwargs_33146 = {}
    # Getting the type of 'abs' (line 248)
    abs_33142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 9), 'abs', False)
    # Calling abs(args, kwargs) (line 248)
    abs_call_result_33147 = invoke(stypy.reporting.localization.Localization(__file__, 248, 9), abs_33142, *[result_sub_33145], **kwargs_33146)
    
    
    # Call to abs(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'l1' (line 248)
    l1_33149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 28), 'l1', False)
    # Getting the type of 'l2' (line 248)
    l2_33150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 33), 'l2', False)
    # Applying the binary operator '+' (line 248)
    result_add_33151 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 28), '+', l1_33149, l2_33150)
    
    # Processing the call keyword arguments (line 248)
    kwargs_33152 = {}
    # Getting the type of 'abs' (line 248)
    abs_33148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'abs', False)
    # Calling abs(args, kwargs) (line 248)
    abs_call_result_33153 = invoke(stypy.reporting.localization.Localization(__file__, 248, 24), abs_33148, *[result_add_33151], **kwargs_33152)
    
    int_33154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 39), 'int')
    # Applying the binary operator 'div' (line 248)
    result_div_33155 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 24), 'div', abs_call_result_33153, int_33154)
    
    # Applying the binary operator '>' (line 248)
    result_gt_33156 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 9), '>', abs_call_result_33147, result_div_33155)
    
    # Testing the type of an if condition (line 248)
    if_condition_33157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 248, 9), result_gt_33156)
    # Assigning a type to the variable 'if_condition_33157' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 9), 'if_condition_33157', if_condition_33157)
    # SSA begins for if statement (line 248)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 249):
    
    # Assigning a BinOp to a Name (line 249):
    # Getting the type of 't12' (line 249)
    t12_33158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 14), 't12')
    # Getting the type of 'l2' (line 249)
    l2_33159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 22), 'l2')
    # Getting the type of 'p' (line 249)
    p_33160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 26), 'p')
    # Applying the binary operator '**' (line 249)
    result_pow_33161 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 22), '**', l2_33159, p_33160)
    
    # Getting the type of 'l1' (line 249)
    l1_33162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 32), 'l1')
    # Getting the type of 'p' (line 249)
    p_33163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 36), 'p')
    # Applying the binary operator '**' (line 249)
    result_pow_33164 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 32), '**', l1_33162, p_33163)
    
    # Applying the binary operator '-' (line 249)
    result_sub_33165 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 21), '-', result_pow_33161, result_pow_33164)
    
    # Applying the binary operator '*' (line 249)
    result_mul_33166 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 14), '*', t12_33158, result_sub_33165)
    
    # Getting the type of 'l2' (line 249)
    l2_33167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 43), 'l2')
    # Getting the type of 'l1' (line 249)
    l1_33168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 48), 'l1')
    # Applying the binary operator '-' (line 249)
    result_sub_33169 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 43), '-', l2_33167, l1_33168)
    
    # Applying the binary operator 'div' (line 249)
    result_div_33170 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 40), 'div', result_mul_33166, result_sub_33169)
    
    # Assigning a type to the variable 'f12' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'f12', result_div_33170)
    # SSA branch for the else part of an if statement (line 248)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 252):
    
    # Assigning a BinOp to a Name (line 252):
    # Getting the type of 'l2' (line 252)
    l2_33171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 13), 'l2')
    # Getting the type of 'l1' (line 252)
    l1_33172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 18), 'l1')
    # Applying the binary operator '-' (line 252)
    result_sub_33173 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 13), '-', l2_33171, l1_33172)
    
    # Getting the type of 'l2' (line 252)
    l2_33174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 25), 'l2')
    # Getting the type of 'l1' (line 252)
    l1_33175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 30), 'l1')
    # Applying the binary operator '+' (line 252)
    result_add_33176 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 25), '+', l2_33174, l1_33175)
    
    # Applying the binary operator 'div' (line 252)
    result_div_33177 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 12), 'div', result_sub_33173, result_add_33176)
    
    # Assigning a type to the variable 'z' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'z', result_div_33177)
    
    # Assigning a Call to a Name (line 253):
    
    # Assigning a Call to a Name (line 253):
    
    # Call to log(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 'l1' (line 253)
    l1_33180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 24), 'l1', False)
    # Processing the call keyword arguments (line 253)
    kwargs_33181 = {}
    # Getting the type of 'np' (line 253)
    np_33178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 17), 'np', False)
    # Obtaining the member 'log' of a type (line 253)
    log_33179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 17), np_33178, 'log')
    # Calling log(args, kwargs) (line 253)
    log_call_result_33182 = invoke(stypy.reporting.localization.Localization(__file__, 253, 17), log_33179, *[l1_33180], **kwargs_33181)
    
    # Assigning a type to the variable 'log_l1' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'log_l1', log_call_result_33182)
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to log(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'l2' (line 254)
    l2_33185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 24), 'l2', False)
    # Processing the call keyword arguments (line 254)
    kwargs_33186 = {}
    # Getting the type of 'np' (line 254)
    np_33183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 17), 'np', False)
    # Obtaining the member 'log' of a type (line 254)
    log_33184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 17), np_33183, 'log')
    # Calling log(args, kwargs) (line 254)
    log_call_result_33187 = invoke(stypy.reporting.localization.Localization(__file__, 254, 17), log_33184, *[l2_33185], **kwargs_33186)
    
    # Assigning a type to the variable 'log_l2' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'log_l2', log_call_result_33187)
    
    # Assigning a Call to a Name (line 255):
    
    # Assigning a Call to a Name (line 255):
    
    # Call to arctanh(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'z' (line 255)
    z_33190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 31), 'z', False)
    # Processing the call keyword arguments (line 255)
    kwargs_33191 = {}
    # Getting the type of 'np' (line 255)
    np_33188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'np', False)
    # Obtaining the member 'arctanh' of a type (line 255)
    arctanh_33189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 20), np_33188, 'arctanh')
    # Calling arctanh(args, kwargs) (line 255)
    arctanh_call_result_33192 = invoke(stypy.reporting.localization.Localization(__file__, 255, 20), arctanh_33189, *[z_33190], **kwargs_33191)
    
    # Assigning a type to the variable 'arctanh_z' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'arctanh_z', arctanh_call_result_33192)
    
    # Assigning a BinOp to a Name (line 256):
    
    # Assigning a BinOp to a Name (line 256):
    # Getting the type of 't12' (line 256)
    t12_33193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 't12')
    
    # Call to exp(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'p' (line 256)
    p_33196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 30), 'p', False)
    int_33197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 32), 'int')
    # Applying the binary operator 'div' (line 256)
    result_div_33198 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 30), 'div', p_33196, int_33197)
    
    # Getting the type of 'log_l2' (line 256)
    log_l2_33199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 36), 'log_l2', False)
    # Getting the type of 'log_l1' (line 256)
    log_l1_33200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 45), 'log_l1', False)
    # Applying the binary operator '+' (line 256)
    result_add_33201 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 36), '+', log_l2_33199, log_l1_33200)
    
    # Applying the binary operator '*' (line 256)
    result_mul_33202 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 29), '*', result_div_33198, result_add_33201)
    
    # Processing the call keyword arguments (line 256)
    kwargs_33203 = {}
    # Getting the type of 'np' (line 256)
    np_33194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 22), 'np', False)
    # Obtaining the member 'exp' of a type (line 256)
    exp_33195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 22), np_33194, 'exp')
    # Calling exp(args, kwargs) (line 256)
    exp_call_result_33204 = invoke(stypy.reporting.localization.Localization(__file__, 256, 22), exp_33195, *[result_mul_33202], **kwargs_33203)
    
    # Applying the binary operator '*' (line 256)
    result_mul_33205 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 16), '*', t12_33193, exp_call_result_33204)
    
    # Assigning a type to the variable 'tmp_a' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'tmp_a', result_mul_33205)
    
    # Assigning a Call to a Name (line 257):
    
    # Assigning a Call to a Name (line 257):
    
    # Call to _unwindk(...): (line 257)
    # Processing the call arguments (line 257)
    # Getting the type of 'log_l2' (line 257)
    log_l2_33207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 25), 'log_l2', False)
    # Getting the type of 'log_l1' (line 257)
    log_l1_33208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 34), 'log_l1', False)
    # Applying the binary operator '-' (line 257)
    result_sub_33209 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 25), '-', log_l2_33207, log_l1_33208)
    
    # Processing the call keyword arguments (line 257)
    kwargs_33210 = {}
    # Getting the type of '_unwindk' (line 257)
    _unwindk_33206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), '_unwindk', False)
    # Calling _unwindk(args, kwargs) (line 257)
    _unwindk_call_result_33211 = invoke(stypy.reporting.localization.Localization(__file__, 257, 16), _unwindk_33206, *[result_sub_33209], **kwargs_33210)
    
    # Assigning a type to the variable 'tmp_u' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'tmp_u', _unwindk_call_result_33211)
    
    # Getting the type of 'tmp_u' (line 258)
    tmp_u_33212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 11), 'tmp_u')
    # Testing the type of an if condition (line 258)
    if_condition_33213 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 8), tmp_u_33212)
    # Assigning a type to the variable 'if_condition_33213' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'if_condition_33213', if_condition_33213)
    # SSA begins for if statement (line 258)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 259):
    
    # Assigning a BinOp to a Name (line 259):
    # Getting the type of 'p' (line 259)
    p_33214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'p')
    # Getting the type of 'arctanh_z' (line 259)
    arctanh_z_33215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 25), 'arctanh_z')
    # Getting the type of 'np' (line 259)
    np_33216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 37), 'np')
    # Obtaining the member 'pi' of a type (line 259)
    pi_33217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 37), np_33216, 'pi')
    complex_33218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 45), 'complex')
    # Applying the binary operator '*' (line 259)
    result_mul_33219 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 37), '*', pi_33217, complex_33218)
    
    # Getting the type of 'tmp_u' (line 259)
    tmp_u_33220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 50), 'tmp_u')
    # Applying the binary operator '*' (line 259)
    result_mul_33221 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 48), '*', result_mul_33219, tmp_u_33220)
    
    # Applying the binary operator '+' (line 259)
    result_add_33222 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 25), '+', arctanh_z_33215, result_mul_33221)
    
    # Applying the binary operator '*' (line 259)
    result_mul_33223 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 20), '*', p_33214, result_add_33222)
    
    # Assigning a type to the variable 'tmp_b' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'tmp_b', result_mul_33223)
    # SSA branch for the else part of an if statement (line 258)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 261):
    
    # Assigning a BinOp to a Name (line 261):
    # Getting the type of 'p' (line 261)
    p_33224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'p')
    # Getting the type of 'arctanh_z' (line 261)
    arctanh_z_33225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 24), 'arctanh_z')
    # Applying the binary operator '*' (line 261)
    result_mul_33226 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 20), '*', p_33224, arctanh_z_33225)
    
    # Assigning a type to the variable 'tmp_b' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'tmp_b', result_mul_33226)
    # SSA join for if statement (line 258)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 262):
    
    # Assigning a BinOp to a Name (line 262):
    int_33227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 16), 'int')
    
    # Call to sinh(...): (line 262)
    # Processing the call arguments (line 262)
    # Getting the type of 'tmp_b' (line 262)
    tmp_b_33230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 28), 'tmp_b', False)
    # Processing the call keyword arguments (line 262)
    kwargs_33231 = {}
    # Getting the type of 'np' (line 262)
    np_33228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'np', False)
    # Obtaining the member 'sinh' of a type (line 262)
    sinh_33229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 20), np_33228, 'sinh')
    # Calling sinh(args, kwargs) (line 262)
    sinh_call_result_33232 = invoke(stypy.reporting.localization.Localization(__file__, 262, 20), sinh_33229, *[tmp_b_33230], **kwargs_33231)
    
    # Applying the binary operator '*' (line 262)
    result_mul_33233 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 16), '*', int_33227, sinh_call_result_33232)
    
    # Getting the type of 'l2' (line 262)
    l2_33234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 38), 'l2')
    # Getting the type of 'l1' (line 262)
    l1_33235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 43), 'l1')
    # Applying the binary operator '-' (line 262)
    result_sub_33236 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 38), '-', l2_33234, l1_33235)
    
    # Applying the binary operator 'div' (line 262)
    result_div_33237 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 35), 'div', result_mul_33233, result_sub_33236)
    
    # Assigning a type to the variable 'tmp_c' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'tmp_c', result_div_33237)
    
    # Assigning a BinOp to a Name (line 263):
    
    # Assigning a BinOp to a Name (line 263):
    # Getting the type of 'tmp_a' (line 263)
    tmp_a_33238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 14), 'tmp_a')
    # Getting the type of 'tmp_c' (line 263)
    tmp_c_33239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 22), 'tmp_c')
    # Applying the binary operator '*' (line 263)
    result_mul_33240 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 14), '*', tmp_a_33238, tmp_c_33239)
    
    # Assigning a type to the variable 'f12' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'f12', result_mul_33240)
    # SSA join for if statement (line 248)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 246)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'f12' (line 264)
    f12_33241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 11), 'f12')
    # Assigning a type to the variable 'stypy_return_type' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type', f12_33241)
    
    # ################# End of '_fractional_power_superdiag_entry(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fractional_power_superdiag_entry' in the type store
    # Getting the type of 'stypy_return_type' (line 211)
    stypy_return_type_33242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33242)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fractional_power_superdiag_entry'
    return stypy_return_type_33242

# Assigning a type to the variable '_fractional_power_superdiag_entry' (line 211)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), '_fractional_power_superdiag_entry', _fractional_power_superdiag_entry)

@norecursion
def _logm_superdiag_entry(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_logm_superdiag_entry'
    module_type_store = module_type_store.open_function_context('_logm_superdiag_entry', 267, 0, False)
    
    # Passed parameters checking function
    _logm_superdiag_entry.stypy_localization = localization
    _logm_superdiag_entry.stypy_type_of_self = None
    _logm_superdiag_entry.stypy_type_store = module_type_store
    _logm_superdiag_entry.stypy_function_name = '_logm_superdiag_entry'
    _logm_superdiag_entry.stypy_param_names_list = ['l1', 'l2', 't12']
    _logm_superdiag_entry.stypy_varargs_param_name = None
    _logm_superdiag_entry.stypy_kwargs_param_name = None
    _logm_superdiag_entry.stypy_call_defaults = defaults
    _logm_superdiag_entry.stypy_call_varargs = varargs
    _logm_superdiag_entry.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_logm_superdiag_entry', ['l1', 'l2', 't12'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_logm_superdiag_entry', localization, ['l1', 'l2', 't12'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_logm_superdiag_entry(...)' code ##################

    str_33243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, (-1)), 'str', '\n    Compute a superdiagonal entry of a matrix logarithm.\n\n    This is like Eq. (11.28) in [1]_, except the determination of whether\n    l1 and l2 are sufficiently far apart has been modified.\n\n    Parameters\n    ----------\n    l1 : complex\n        A diagonal entry of the matrix.\n    l2 : complex\n        A diagonal entry of the matrix.\n    t12 : complex\n        A superdiagonal entry of the matrix.\n\n    Returns\n    -------\n    f12 : complex\n        A superdiagonal entry of the matrix logarithm.\n\n    Notes\n    -----\n    Care has been taken to return a real number if possible when\n    all of the inputs are real numbers.\n\n    References\n    ----------\n    .. [1] Nicholas J. Higham (2008)\n           "Functions of Matrices: Theory and Computation"\n           ISBN 978-0-898716-46-7\n\n    ')
    
    
    # Getting the type of 'l1' (line 300)
    l1_33244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 7), 'l1')
    # Getting the type of 'l2' (line 300)
    l2_33245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 13), 'l2')
    # Applying the binary operator '==' (line 300)
    result_eq_33246 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 7), '==', l1_33244, l2_33245)
    
    # Testing the type of an if condition (line 300)
    if_condition_33247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 4), result_eq_33246)
    # Assigning a type to the variable 'if_condition_33247' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'if_condition_33247', if_condition_33247)
    # SSA begins for if statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 301):
    
    # Assigning a BinOp to a Name (line 301):
    # Getting the type of 't12' (line 301)
    t12_33248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 14), 't12')
    # Getting the type of 'l1' (line 301)
    l1_33249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 20), 'l1')
    # Applying the binary operator 'div' (line 301)
    result_div_33250 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 14), 'div', t12_33248, l1_33249)
    
    # Assigning a type to the variable 'f12' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'f12', result_div_33250)
    # SSA branch for the else part of an if statement (line 300)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to abs(...): (line 302)
    # Processing the call arguments (line 302)
    # Getting the type of 'l2' (line 302)
    l2_33252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 13), 'l2', False)
    # Getting the type of 'l1' (line 302)
    l1_33253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 18), 'l1', False)
    # Applying the binary operator '-' (line 302)
    result_sub_33254 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 13), '-', l2_33252, l1_33253)
    
    # Processing the call keyword arguments (line 302)
    kwargs_33255 = {}
    # Getting the type of 'abs' (line 302)
    abs_33251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 9), 'abs', False)
    # Calling abs(args, kwargs) (line 302)
    abs_call_result_33256 = invoke(stypy.reporting.localization.Localization(__file__, 302, 9), abs_33251, *[result_sub_33254], **kwargs_33255)
    
    
    # Call to abs(...): (line 302)
    # Processing the call arguments (line 302)
    # Getting the type of 'l1' (line 302)
    l1_33258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 28), 'l1', False)
    # Getting the type of 'l2' (line 302)
    l2_33259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 33), 'l2', False)
    # Applying the binary operator '+' (line 302)
    result_add_33260 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 28), '+', l1_33258, l2_33259)
    
    # Processing the call keyword arguments (line 302)
    kwargs_33261 = {}
    # Getting the type of 'abs' (line 302)
    abs_33257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 24), 'abs', False)
    # Calling abs(args, kwargs) (line 302)
    abs_call_result_33262 = invoke(stypy.reporting.localization.Localization(__file__, 302, 24), abs_33257, *[result_add_33260], **kwargs_33261)
    
    int_33263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 39), 'int')
    # Applying the binary operator 'div' (line 302)
    result_div_33264 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 24), 'div', abs_call_result_33262, int_33263)
    
    # Applying the binary operator '>' (line 302)
    result_gt_33265 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 9), '>', abs_call_result_33256, result_div_33264)
    
    # Testing the type of an if condition (line 302)
    if_condition_33266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 9), result_gt_33265)
    # Assigning a type to the variable 'if_condition_33266' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 9), 'if_condition_33266', if_condition_33266)
    # SSA begins for if statement (line 302)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 303):
    
    # Assigning a BinOp to a Name (line 303):
    # Getting the type of 't12' (line 303)
    t12_33267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 14), 't12')
    
    # Call to log(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'l2' (line 303)
    l2_33270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 28), 'l2', False)
    # Processing the call keyword arguments (line 303)
    kwargs_33271 = {}
    # Getting the type of 'np' (line 303)
    np_33268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 21), 'np', False)
    # Obtaining the member 'log' of a type (line 303)
    log_33269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 21), np_33268, 'log')
    # Calling log(args, kwargs) (line 303)
    log_call_result_33272 = invoke(stypy.reporting.localization.Localization(__file__, 303, 21), log_33269, *[l2_33270], **kwargs_33271)
    
    
    # Call to log(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'l1' (line 303)
    l1_33275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 41), 'l1', False)
    # Processing the call keyword arguments (line 303)
    kwargs_33276 = {}
    # Getting the type of 'np' (line 303)
    np_33273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 34), 'np', False)
    # Obtaining the member 'log' of a type (line 303)
    log_33274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 34), np_33273, 'log')
    # Calling log(args, kwargs) (line 303)
    log_call_result_33277 = invoke(stypy.reporting.localization.Localization(__file__, 303, 34), log_33274, *[l1_33275], **kwargs_33276)
    
    # Applying the binary operator '-' (line 303)
    result_sub_33278 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 21), '-', log_call_result_33272, log_call_result_33277)
    
    # Applying the binary operator '*' (line 303)
    result_mul_33279 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 14), '*', t12_33267, result_sub_33278)
    
    # Getting the type of 'l2' (line 303)
    l2_33280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 49), 'l2')
    # Getting the type of 'l1' (line 303)
    l1_33281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 54), 'l1')
    # Applying the binary operator '-' (line 303)
    result_sub_33282 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 49), '-', l2_33280, l1_33281)
    
    # Applying the binary operator 'div' (line 303)
    result_div_33283 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 46), 'div', result_mul_33279, result_sub_33282)
    
    # Assigning a type to the variable 'f12' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'f12', result_div_33283)
    # SSA branch for the else part of an if statement (line 302)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 305):
    
    # Assigning a BinOp to a Name (line 305):
    # Getting the type of 'l2' (line 305)
    l2_33284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 13), 'l2')
    # Getting the type of 'l1' (line 305)
    l1_33285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 18), 'l1')
    # Applying the binary operator '-' (line 305)
    result_sub_33286 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 13), '-', l2_33284, l1_33285)
    
    # Getting the type of 'l2' (line 305)
    l2_33287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 25), 'l2')
    # Getting the type of 'l1' (line 305)
    l1_33288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 30), 'l1')
    # Applying the binary operator '+' (line 305)
    result_add_33289 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 25), '+', l2_33287, l1_33288)
    
    # Applying the binary operator 'div' (line 305)
    result_div_33290 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 12), 'div', result_sub_33286, result_add_33289)
    
    # Assigning a type to the variable 'z' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'z', result_div_33290)
    
    # Assigning a Call to a Name (line 306):
    
    # Assigning a Call to a Name (line 306):
    
    # Call to _unwindk(...): (line 306)
    # Processing the call arguments (line 306)
    
    # Call to log(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'l2' (line 306)
    l2_33294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 28), 'l2', False)
    # Processing the call keyword arguments (line 306)
    kwargs_33295 = {}
    # Getting the type of 'np' (line 306)
    np_33292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 21), 'np', False)
    # Obtaining the member 'log' of a type (line 306)
    log_33293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 21), np_33292, 'log')
    # Calling log(args, kwargs) (line 306)
    log_call_result_33296 = invoke(stypy.reporting.localization.Localization(__file__, 306, 21), log_33293, *[l2_33294], **kwargs_33295)
    
    
    # Call to log(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'l1' (line 306)
    l1_33299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 41), 'l1', False)
    # Processing the call keyword arguments (line 306)
    kwargs_33300 = {}
    # Getting the type of 'np' (line 306)
    np_33297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 34), 'np', False)
    # Obtaining the member 'log' of a type (line 306)
    log_33298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 34), np_33297, 'log')
    # Calling log(args, kwargs) (line 306)
    log_call_result_33301 = invoke(stypy.reporting.localization.Localization(__file__, 306, 34), log_33298, *[l1_33299], **kwargs_33300)
    
    # Applying the binary operator '-' (line 306)
    result_sub_33302 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 21), '-', log_call_result_33296, log_call_result_33301)
    
    # Processing the call keyword arguments (line 306)
    kwargs_33303 = {}
    # Getting the type of '_unwindk' (line 306)
    _unwindk_33291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), '_unwindk', False)
    # Calling _unwindk(args, kwargs) (line 306)
    _unwindk_call_result_33304 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), _unwindk_33291, *[result_sub_33302], **kwargs_33303)
    
    # Assigning a type to the variable 'u' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'u', _unwindk_call_result_33304)
    
    # Getting the type of 'u' (line 307)
    u_33305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 11), 'u')
    # Testing the type of an if condition (line 307)
    if_condition_33306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 8), u_33305)
    # Assigning a type to the variable 'if_condition_33306' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'if_condition_33306', if_condition_33306)
    # SSA begins for if statement (line 307)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 308):
    
    # Assigning a BinOp to a Name (line 308):
    # Getting the type of 't12' (line 308)
    t12_33307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 18), 't12')
    int_33308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 24), 'int')
    # Applying the binary operator '*' (line 308)
    result_mul_33309 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 18), '*', t12_33307, int_33308)
    
    
    # Call to arctanh(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'z' (line 308)
    z_33312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 40), 'z', False)
    # Processing the call keyword arguments (line 308)
    kwargs_33313 = {}
    # Getting the type of 'np' (line 308)
    np_33310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 29), 'np', False)
    # Obtaining the member 'arctanh' of a type (line 308)
    arctanh_33311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 29), np_33310, 'arctanh')
    # Calling arctanh(args, kwargs) (line 308)
    arctanh_call_result_33314 = invoke(stypy.reporting.localization.Localization(__file__, 308, 29), arctanh_33311, *[z_33312], **kwargs_33313)
    
    # Getting the type of 'np' (line 308)
    np_33315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 45), 'np')
    # Obtaining the member 'pi' of a type (line 308)
    pi_33316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 45), np_33315, 'pi')
    complex_33317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 51), 'complex')
    # Applying the binary operator '*' (line 308)
    result_mul_33318 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 45), '*', pi_33316, complex_33317)
    
    # Getting the type of 'u' (line 308)
    u_33319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 54), 'u')
    # Applying the binary operator '*' (line 308)
    result_mul_33320 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 53), '*', result_mul_33318, u_33319)
    
    # Applying the binary operator '+' (line 308)
    result_add_33321 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 29), '+', arctanh_call_result_33314, result_mul_33320)
    
    # Applying the binary operator '*' (line 308)
    result_mul_33322 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 26), '*', result_mul_33309, result_add_33321)
    
    # Getting the type of 'l2' (line 308)
    l2_33323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 60), 'l2')
    # Getting the type of 'l1' (line 308)
    l1_33324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 65), 'l1')
    # Applying the binary operator '-' (line 308)
    result_sub_33325 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 60), '-', l2_33323, l1_33324)
    
    # Applying the binary operator 'div' (line 308)
    result_div_33326 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 57), 'div', result_mul_33322, result_sub_33325)
    
    # Assigning a type to the variable 'f12' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'f12', result_div_33326)
    # SSA branch for the else part of an if statement (line 307)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 310):
    
    # Assigning a BinOp to a Name (line 310):
    # Getting the type of 't12' (line 310)
    t12_33327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 18), 't12')
    int_33328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 24), 'int')
    # Applying the binary operator '*' (line 310)
    result_mul_33329 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 18), '*', t12_33327, int_33328)
    
    
    # Call to arctanh(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'z' (line 310)
    z_33332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 39), 'z', False)
    # Processing the call keyword arguments (line 310)
    kwargs_33333 = {}
    # Getting the type of 'np' (line 310)
    np_33330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 28), 'np', False)
    # Obtaining the member 'arctanh' of a type (line 310)
    arctanh_33331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 28), np_33330, 'arctanh')
    # Calling arctanh(args, kwargs) (line 310)
    arctanh_call_result_33334 = invoke(stypy.reporting.localization.Localization(__file__, 310, 28), arctanh_33331, *[z_33332], **kwargs_33333)
    
    # Applying the binary operator '*' (line 310)
    result_mul_33335 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 26), '*', result_mul_33329, arctanh_call_result_33334)
    
    # Getting the type of 'l2' (line 310)
    l2_33336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 45), 'l2')
    # Getting the type of 'l1' (line 310)
    l1_33337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 50), 'l1')
    # Applying the binary operator '-' (line 310)
    result_sub_33338 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 45), '-', l2_33336, l1_33337)
    
    # Applying the binary operator 'div' (line 310)
    result_div_33339 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 42), 'div', result_mul_33335, result_sub_33338)
    
    # Assigning a type to the variable 'f12' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'f12', result_div_33339)
    # SSA join for if statement (line 307)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 302)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 300)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'f12' (line 311)
    f12_33340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 11), 'f12')
    # Assigning a type to the variable 'stypy_return_type' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'stypy_return_type', f12_33340)
    
    # ################# End of '_logm_superdiag_entry(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_logm_superdiag_entry' in the type store
    # Getting the type of 'stypy_return_type' (line 267)
    stypy_return_type_33341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33341)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_logm_superdiag_entry'
    return stypy_return_type_33341

# Assigning a type to the variable '_logm_superdiag_entry' (line 267)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), '_logm_superdiag_entry', _logm_superdiag_entry)

@norecursion
def _inverse_squaring_helper(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_inverse_squaring_helper'
    module_type_store = module_type_store.open_function_context('_inverse_squaring_helper', 314, 0, False)
    
    # Passed parameters checking function
    _inverse_squaring_helper.stypy_localization = localization
    _inverse_squaring_helper.stypy_type_of_self = None
    _inverse_squaring_helper.stypy_type_store = module_type_store
    _inverse_squaring_helper.stypy_function_name = '_inverse_squaring_helper'
    _inverse_squaring_helper.stypy_param_names_list = ['T0', 'theta']
    _inverse_squaring_helper.stypy_varargs_param_name = None
    _inverse_squaring_helper.stypy_kwargs_param_name = None
    _inverse_squaring_helper.stypy_call_defaults = defaults
    _inverse_squaring_helper.stypy_call_varargs = varargs
    _inverse_squaring_helper.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_inverse_squaring_helper', ['T0', 'theta'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_inverse_squaring_helper', localization, ['T0', 'theta'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_inverse_squaring_helper(...)' code ##################

    str_33342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, (-1)), 'str', '\n    A helper function for inverse scaling and squaring for Pade approximation.\n\n    Parameters\n    ----------\n    T0 : (N, N) array_like upper triangular\n        Matrix involved in inverse scaling and squaring.\n    theta : indexable\n        The values theta[1] .. theta[7] must be available.\n        They represent bounds related to Pade approximation, and they depend\n        on the matrix function which is being computed.\n        For example, different values of theta are required for\n        matrix logarithm than for fractional matrix power.\n\n    Returns\n    -------\n    R : (N, N) array_like upper triangular\n        Composition of zero or more matrix square roots of T0, minus I.\n    s : non-negative integer\n        Number of square roots taken.\n    m : positive integer\n        The degree of the Pade approximation.\n\n    Notes\n    -----\n    This subroutine appears as a chunk of lines within\n    a couple of published algorithms; for example it appears\n    as lines 4--35 in algorithm (3.1) of [1]_, and\n    as lines 3--34 in algorithm (4.1) of [2]_.\n    The instances of \'goto line 38\' in algorithm (3.1) of [1]_\n    probably mean \'goto line 36\' and have been intepreted accordingly.\n\n    References\n    ----------\n    .. [1] Nicholas J. Higham and Lijing Lin (2013)\n           "An Improved Schur-Pade Algorithm for Fractional Powers\n           of a Matrix and their Frechet Derivatives."\n\n    .. [2] Awad H. Al-Mohy and Nicholas J. Higham (2012)\n           "Improved Inverse Scaling and Squaring Algorithms\n           for the Matrix Logarithm."\n           SIAM Journal on Scientific Computing, 34 (4). C152-C169.\n           ISSN 1095-7197\n\n    ')
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'T0' (line 360)
    T0_33344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 11), 'T0', False)
    # Obtaining the member 'shape' of a type (line 360)
    shape_33345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 11), T0_33344, 'shape')
    # Processing the call keyword arguments (line 360)
    kwargs_33346 = {}
    # Getting the type of 'len' (line 360)
    len_33343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 7), 'len', False)
    # Calling len(args, kwargs) (line 360)
    len_call_result_33347 = invoke(stypy.reporting.localization.Localization(__file__, 360, 7), len_33343, *[shape_33345], **kwargs_33346)
    
    int_33348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 24), 'int')
    # Applying the binary operator '!=' (line 360)
    result_ne_33349 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 7), '!=', len_call_result_33347, int_33348)
    
    
    
    # Obtaining the type of the subscript
    int_33350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 38), 'int')
    # Getting the type of 'T0' (line 360)
    T0_33351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 29), 'T0')
    # Obtaining the member 'shape' of a type (line 360)
    shape_33352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 29), T0_33351, 'shape')
    # Obtaining the member '__getitem__' of a type (line 360)
    getitem___33353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 29), shape_33352, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 360)
    subscript_call_result_33354 = invoke(stypy.reporting.localization.Localization(__file__, 360, 29), getitem___33353, int_33350)
    
    
    # Obtaining the type of the subscript
    int_33355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 53), 'int')
    # Getting the type of 'T0' (line 360)
    T0_33356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 44), 'T0')
    # Obtaining the member 'shape' of a type (line 360)
    shape_33357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 44), T0_33356, 'shape')
    # Obtaining the member '__getitem__' of a type (line 360)
    getitem___33358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 44), shape_33357, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 360)
    subscript_call_result_33359 = invoke(stypy.reporting.localization.Localization(__file__, 360, 44), getitem___33358, int_33355)
    
    # Applying the binary operator '!=' (line 360)
    result_ne_33360 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 29), '!=', subscript_call_result_33354, subscript_call_result_33359)
    
    # Applying the binary operator 'or' (line 360)
    result_or_keyword_33361 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 7), 'or', result_ne_33349, result_ne_33360)
    
    # Testing the type of an if condition (line 360)
    if_condition_33362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 4), result_or_keyword_33361)
    # Assigning a type to the variable 'if_condition_33362' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'if_condition_33362', if_condition_33362)
    # SSA begins for if statement (line 360)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 361)
    # Processing the call arguments (line 361)
    str_33364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 25), 'str', 'expected an upper triangular square matrix')
    # Processing the call keyword arguments (line 361)
    kwargs_33365 = {}
    # Getting the type of 'ValueError' (line 361)
    ValueError_33363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 361)
    ValueError_call_result_33366 = invoke(stypy.reporting.localization.Localization(__file__, 361, 14), ValueError_33363, *[str_33364], **kwargs_33365)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 361, 8), ValueError_call_result_33366, 'raise parameter', BaseException)
    # SSA join for if statement (line 360)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 362):
    
    # Assigning a Subscript to a Name (line 362):
    
    # Obtaining the type of the subscript
    int_33367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 4), 'int')
    # Getting the type of 'T0' (line 362)
    T0_33368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 11), 'T0')
    # Obtaining the member 'shape' of a type (line 362)
    shape_33369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 11), T0_33368, 'shape')
    # Obtaining the member '__getitem__' of a type (line 362)
    getitem___33370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 4), shape_33369, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 362)
    subscript_call_result_33371 = invoke(stypy.reporting.localization.Localization(__file__, 362, 4), getitem___33370, int_33367)
    
    # Assigning a type to the variable 'tuple_var_assignment_32838' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'tuple_var_assignment_32838', subscript_call_result_33371)
    
    # Assigning a Subscript to a Name (line 362):
    
    # Obtaining the type of the subscript
    int_33372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 4), 'int')
    # Getting the type of 'T0' (line 362)
    T0_33373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 11), 'T0')
    # Obtaining the member 'shape' of a type (line 362)
    shape_33374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 11), T0_33373, 'shape')
    # Obtaining the member '__getitem__' of a type (line 362)
    getitem___33375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 4), shape_33374, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 362)
    subscript_call_result_33376 = invoke(stypy.reporting.localization.Localization(__file__, 362, 4), getitem___33375, int_33372)
    
    # Assigning a type to the variable 'tuple_var_assignment_32839' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'tuple_var_assignment_32839', subscript_call_result_33376)
    
    # Assigning a Name to a Name (line 362):
    # Getting the type of 'tuple_var_assignment_32838' (line 362)
    tuple_var_assignment_32838_33377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'tuple_var_assignment_32838')
    # Assigning a type to the variable 'n' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'n', tuple_var_assignment_32838_33377)
    
    # Assigning a Name to a Name (line 362):
    # Getting the type of 'tuple_var_assignment_32839' (line 362)
    tuple_var_assignment_32839_33378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'tuple_var_assignment_32839')
    # Assigning a type to the variable 'n' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 7), 'n', tuple_var_assignment_32839_33378)
    
    # Assigning a Name to a Name (line 363):
    
    # Assigning a Name to a Name (line 363):
    # Getting the type of 'T0' (line 363)
    T0_33379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'T0')
    # Assigning a type to the variable 'T' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'T', T0_33379)
    
    # Assigning a Num to a Name (line 369):
    
    # Assigning a Num to a Name (line 369):
    int_33380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 9), 'int')
    # Assigning a type to the variable 's0' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 's0', int_33380)
    
    # Assigning a Call to a Name (line 370):
    
    # Assigning a Call to a Name (line 370):
    
    # Call to diag(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'T' (line 370)
    T_33383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 23), 'T', False)
    # Processing the call keyword arguments (line 370)
    kwargs_33384 = {}
    # Getting the type of 'np' (line 370)
    np_33381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 15), 'np', False)
    # Obtaining the member 'diag' of a type (line 370)
    diag_33382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 15), np_33381, 'diag')
    # Calling diag(args, kwargs) (line 370)
    diag_call_result_33385 = invoke(stypy.reporting.localization.Localization(__file__, 370, 15), diag_33382, *[T_33383], **kwargs_33384)
    
    # Assigning a type to the variable 'tmp_diag' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'tmp_diag', diag_call_result_33385)
    
    
    
    # Call to count_nonzero(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of 'tmp_diag' (line 371)
    tmp_diag_33388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'tmp_diag', False)
    # Processing the call keyword arguments (line 371)
    kwargs_33389 = {}
    # Getting the type of 'np' (line 371)
    np_33386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 7), 'np', False)
    # Obtaining the member 'count_nonzero' of a type (line 371)
    count_nonzero_33387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 7), np_33386, 'count_nonzero')
    # Calling count_nonzero(args, kwargs) (line 371)
    count_nonzero_call_result_33390 = invoke(stypy.reporting.localization.Localization(__file__, 371, 7), count_nonzero_33387, *[tmp_diag_33388], **kwargs_33389)
    
    # Getting the type of 'n' (line 371)
    n_33391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 37), 'n')
    # Applying the binary operator '!=' (line 371)
    result_ne_33392 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 7), '!=', count_nonzero_call_result_33390, n_33391)
    
    # Testing the type of an if condition (line 371)
    if_condition_33393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 4), result_ne_33392)
    # Assigning a type to the variable 'if_condition_33393' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'if_condition_33393', if_condition_33393)
    # SSA begins for if statement (line 371)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 372)
    # Processing the call arguments (line 372)
    str_33395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 24), 'str', 'internal inconsistency')
    # Processing the call keyword arguments (line 372)
    kwargs_33396 = {}
    # Getting the type of 'Exception' (line 372)
    Exception_33394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 372)
    Exception_call_result_33397 = invoke(stypy.reporting.localization.Localization(__file__, 372, 14), Exception_33394, *[str_33395], **kwargs_33396)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 372, 8), Exception_call_result_33397, 'raise parameter', BaseException)
    # SSA join for if statement (line 371)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to max(...): (line 373)
    # Processing the call arguments (line 373)
    
    # Call to absolute(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of 'tmp_diag' (line 373)
    tmp_diag_33402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 29), 'tmp_diag', False)
    int_33403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 40), 'int')
    # Applying the binary operator '-' (line 373)
    result_sub_33404 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 29), '-', tmp_diag_33402, int_33403)
    
    # Processing the call keyword arguments (line 373)
    kwargs_33405 = {}
    # Getting the type of 'np' (line 373)
    np_33400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 17), 'np', False)
    # Obtaining the member 'absolute' of a type (line 373)
    absolute_33401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 17), np_33400, 'absolute')
    # Calling absolute(args, kwargs) (line 373)
    absolute_call_result_33406 = invoke(stypy.reporting.localization.Localization(__file__, 373, 17), absolute_33401, *[result_sub_33404], **kwargs_33405)
    
    # Processing the call keyword arguments (line 373)
    kwargs_33407 = {}
    # Getting the type of 'np' (line 373)
    np_33398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 10), 'np', False)
    # Obtaining the member 'max' of a type (line 373)
    max_33399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 10), np_33398, 'max')
    # Calling max(args, kwargs) (line 373)
    max_call_result_33408 = invoke(stypy.reporting.localization.Localization(__file__, 373, 10), max_33399, *[absolute_call_result_33406], **kwargs_33407)
    
    
    # Obtaining the type of the subscript
    int_33409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 52), 'int')
    # Getting the type of 'theta' (line 373)
    theta_33410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 46), 'theta')
    # Obtaining the member '__getitem__' of a type (line 373)
    getitem___33411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 46), theta_33410, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 373)
    subscript_call_result_33412 = invoke(stypy.reporting.localization.Localization(__file__, 373, 46), getitem___33411, int_33409)
    
    # Applying the binary operator '>' (line 373)
    result_gt_33413 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 10), '>', max_call_result_33408, subscript_call_result_33412)
    
    # Testing the type of an if condition (line 373)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 4), result_gt_33413)
    # SSA begins for while statement (line 373)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 374):
    
    # Assigning a Call to a Name (line 374):
    
    # Call to sqrt(...): (line 374)
    # Processing the call arguments (line 374)
    # Getting the type of 'tmp_diag' (line 374)
    tmp_diag_33416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 27), 'tmp_diag', False)
    # Processing the call keyword arguments (line 374)
    kwargs_33417 = {}
    # Getting the type of 'np' (line 374)
    np_33414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 19), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 374)
    sqrt_33415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 19), np_33414, 'sqrt')
    # Calling sqrt(args, kwargs) (line 374)
    sqrt_call_result_33418 = invoke(stypy.reporting.localization.Localization(__file__, 374, 19), sqrt_33415, *[tmp_diag_33416], **kwargs_33417)
    
    # Assigning a type to the variable 'tmp_diag' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'tmp_diag', sqrt_call_result_33418)
    
    # Getting the type of 's0' (line 375)
    s0_33419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 's0')
    int_33420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 14), 'int')
    # Applying the binary operator '+=' (line 375)
    result_iadd_33421 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 8), '+=', s0_33419, int_33420)
    # Assigning a type to the variable 's0' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 's0', result_iadd_33421)
    
    # SSA join for while statement (line 373)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 378)
    # Processing the call arguments (line 378)
    # Getting the type of 's0' (line 378)
    s0_33423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 19), 's0', False)
    # Processing the call keyword arguments (line 378)
    kwargs_33424 = {}
    # Getting the type of 'range' (line 378)
    range_33422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 13), 'range', False)
    # Calling range(args, kwargs) (line 378)
    range_call_result_33425 = invoke(stypy.reporting.localization.Localization(__file__, 378, 13), range_33422, *[s0_33423], **kwargs_33424)
    
    # Testing the type of a for loop iterable (line 378)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 378, 4), range_call_result_33425)
    # Getting the type of the for loop variable (line 378)
    for_loop_var_33426 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 378, 4), range_call_result_33425)
    # Assigning a type to the variable 'i' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'i', for_loop_var_33426)
    # SSA begins for a for statement (line 378)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 379):
    
    # Assigning a Call to a Name (line 379):
    
    # Call to _sqrtm_triu(...): (line 379)
    # Processing the call arguments (line 379)
    # Getting the type of 'T' (line 379)
    T_33428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 24), 'T', False)
    # Processing the call keyword arguments (line 379)
    kwargs_33429 = {}
    # Getting the type of '_sqrtm_triu' (line 379)
    _sqrtm_triu_33427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), '_sqrtm_triu', False)
    # Calling _sqrtm_triu(args, kwargs) (line 379)
    _sqrtm_triu_call_result_33430 = invoke(stypy.reporting.localization.Localization(__file__, 379, 12), _sqrtm_triu_33427, *[T_33428], **kwargs_33429)
    
    # Assigning a type to the variable 'T' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'T', _sqrtm_triu_call_result_33430)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 384):
    
    # Assigning a Name to a Name (line 384):
    # Getting the type of 's0' (line 384)
    s0_33431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 's0')
    # Assigning a type to the variable 's' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 's', s0_33431)
    
    # Assigning a Num to a Name (line 385):
    
    # Assigning a Num to a Name (line 385):
    int_33432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 8), 'int')
    # Assigning a type to the variable 'k' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'k', int_33432)
    
    # Assigning a BinOp to a Name (line 386):
    
    # Assigning a BinOp to a Name (line 386):
    
    # Call to _onenormest_m1_power(...): (line 386)
    # Processing the call arguments (line 386)
    # Getting the type of 'T' (line 386)
    T_33434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 30), 'T', False)
    int_33435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 33), 'int')
    # Processing the call keyword arguments (line 386)
    kwargs_33436 = {}
    # Getting the type of '_onenormest_m1_power' (line 386)
    _onenormest_m1_power_33433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 9), '_onenormest_m1_power', False)
    # Calling _onenormest_m1_power(args, kwargs) (line 386)
    _onenormest_m1_power_call_result_33437 = invoke(stypy.reporting.localization.Localization(__file__, 386, 9), _onenormest_m1_power_33433, *[T_33434, int_33435], **kwargs_33436)
    
    int_33438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 40), 'int')
    int_33439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 42), 'int')
    # Applying the binary operator 'div' (line 386)
    result_div_33440 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 40), 'div', int_33438, int_33439)
    
    # Applying the binary operator '**' (line 386)
    result_pow_33441 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 9), '**', _onenormest_m1_power_call_result_33437, result_div_33440)
    
    # Assigning a type to the variable 'd2' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'd2', result_pow_33441)
    
    # Assigning a BinOp to a Name (line 387):
    
    # Assigning a BinOp to a Name (line 387):
    
    # Call to _onenormest_m1_power(...): (line 387)
    # Processing the call arguments (line 387)
    # Getting the type of 'T' (line 387)
    T_33443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 30), 'T', False)
    int_33444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 33), 'int')
    # Processing the call keyword arguments (line 387)
    kwargs_33445 = {}
    # Getting the type of '_onenormest_m1_power' (line 387)
    _onenormest_m1_power_33442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 9), '_onenormest_m1_power', False)
    # Calling _onenormest_m1_power(args, kwargs) (line 387)
    _onenormest_m1_power_call_result_33446 = invoke(stypy.reporting.localization.Localization(__file__, 387, 9), _onenormest_m1_power_33442, *[T_33443, int_33444], **kwargs_33445)
    
    int_33447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 40), 'int')
    int_33448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 42), 'int')
    # Applying the binary operator 'div' (line 387)
    result_div_33449 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 40), 'div', int_33447, int_33448)
    
    # Applying the binary operator '**' (line 387)
    result_pow_33450 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 9), '**', _onenormest_m1_power_call_result_33446, result_div_33449)
    
    # Assigning a type to the variable 'd3' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'd3', result_pow_33450)
    
    # Assigning a Call to a Name (line 388):
    
    # Assigning a Call to a Name (line 388):
    
    # Call to max(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'd2' (line 388)
    d2_33452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 13), 'd2', False)
    # Getting the type of 'd3' (line 388)
    d3_33453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 17), 'd3', False)
    # Processing the call keyword arguments (line 388)
    kwargs_33454 = {}
    # Getting the type of 'max' (line 388)
    max_33451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 9), 'max', False)
    # Calling max(args, kwargs) (line 388)
    max_call_result_33455 = invoke(stypy.reporting.localization.Localization(__file__, 388, 9), max_33451, *[d2_33452, d3_33453], **kwargs_33454)
    
    # Assigning a type to the variable 'a2' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'a2', max_call_result_33455)
    
    # Assigning a Name to a Name (line 389):
    
    # Assigning a Name to a Name (line 389):
    # Getting the type of 'None' (line 389)
    None_33456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'None')
    # Assigning a type to the variable 'm' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'm', None_33456)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 390)
    tuple_33457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 390)
    # Adding element type (line 390)
    int_33458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 14), tuple_33457, int_33458)
    # Adding element type (line 390)
    int_33459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 14), tuple_33457, int_33459)
    
    # Testing the type of a for loop iterable (line 390)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 390, 4), tuple_33457)
    # Getting the type of the for loop variable (line 390)
    for_loop_var_33460 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 390, 4), tuple_33457)
    # Assigning a type to the variable 'i' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'i', for_loop_var_33460)
    # SSA begins for a for statement (line 390)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'a2' (line 391)
    a2_33461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 11), 'a2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 391)
    i_33462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 23), 'i')
    # Getting the type of 'theta' (line 391)
    theta_33463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 17), 'theta')
    # Obtaining the member '__getitem__' of a type (line 391)
    getitem___33464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 17), theta_33463, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 391)
    subscript_call_result_33465 = invoke(stypy.reporting.localization.Localization(__file__, 391, 17), getitem___33464, i_33462)
    
    # Applying the binary operator '<=' (line 391)
    result_le_33466 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 11), '<=', a2_33461, subscript_call_result_33465)
    
    # Testing the type of an if condition (line 391)
    if_condition_33467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 8), result_le_33466)
    # Assigning a type to the variable 'if_condition_33467' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'if_condition_33467', if_condition_33467)
    # SSA begins for if statement (line 391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 392):
    
    # Assigning a Name to a Name (line 392):
    # Getting the type of 'i' (line 392)
    i_33468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'i')
    # Assigning a type to the variable 'm' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'm', i_33468)
    # SSA join for if statement (line 391)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'm' (line 394)
    m_33469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 10), 'm')
    # Getting the type of 'None' (line 394)
    None_33470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 15), 'None')
    # Applying the binary operator 'is' (line 394)
    result_is__33471 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 10), 'is', m_33469, None_33470)
    
    # Testing the type of an if condition (line 394)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 4), result_is__33471)
    # SSA begins for while statement (line 394)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # Getting the type of 's' (line 395)
    s_33472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 11), 's')
    # Getting the type of 's0' (line 395)
    s0_33473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 15), 's0')
    # Applying the binary operator '>' (line 395)
    result_gt_33474 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 11), '>', s_33472, s0_33473)
    
    # Testing the type of an if condition (line 395)
    if_condition_33475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 8), result_gt_33474)
    # Assigning a type to the variable 'if_condition_33475' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'if_condition_33475', if_condition_33475)
    # SSA begins for if statement (line 395)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 396):
    
    # Assigning a BinOp to a Name (line 396):
    
    # Call to _onenormest_m1_power(...): (line 396)
    # Processing the call arguments (line 396)
    # Getting the type of 'T' (line 396)
    T_33477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 38), 'T', False)
    int_33478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 41), 'int')
    # Processing the call keyword arguments (line 396)
    kwargs_33479 = {}
    # Getting the type of '_onenormest_m1_power' (line 396)
    _onenormest_m1_power_33476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 17), '_onenormest_m1_power', False)
    # Calling _onenormest_m1_power(args, kwargs) (line 396)
    _onenormest_m1_power_call_result_33480 = invoke(stypy.reporting.localization.Localization(__file__, 396, 17), _onenormest_m1_power_33476, *[T_33477, int_33478], **kwargs_33479)
    
    int_33481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 48), 'int')
    int_33482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 50), 'int')
    # Applying the binary operator 'div' (line 396)
    result_div_33483 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 48), 'div', int_33481, int_33482)
    
    # Applying the binary operator '**' (line 396)
    result_pow_33484 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 17), '**', _onenormest_m1_power_call_result_33480, result_div_33483)
    
    # Assigning a type to the variable 'd3' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'd3', result_pow_33484)
    # SSA join for if statement (line 395)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 397):
    
    # Assigning a BinOp to a Name (line 397):
    
    # Call to _onenormest_m1_power(...): (line 397)
    # Processing the call arguments (line 397)
    # Getting the type of 'T' (line 397)
    T_33486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 34), 'T', False)
    int_33487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 37), 'int')
    # Processing the call keyword arguments (line 397)
    kwargs_33488 = {}
    # Getting the type of '_onenormest_m1_power' (line 397)
    _onenormest_m1_power_33485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 13), '_onenormest_m1_power', False)
    # Calling _onenormest_m1_power(args, kwargs) (line 397)
    _onenormest_m1_power_call_result_33489 = invoke(stypy.reporting.localization.Localization(__file__, 397, 13), _onenormest_m1_power_33485, *[T_33486, int_33487], **kwargs_33488)
    
    int_33490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 44), 'int')
    int_33491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 46), 'int')
    # Applying the binary operator 'div' (line 397)
    result_div_33492 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 44), 'div', int_33490, int_33491)
    
    # Applying the binary operator '**' (line 397)
    result_pow_33493 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 13), '**', _onenormest_m1_power_call_result_33489, result_div_33492)
    
    # Assigning a type to the variable 'd4' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'd4', result_pow_33493)
    
    # Assigning a Call to a Name (line 398):
    
    # Assigning a Call to a Name (line 398):
    
    # Call to max(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'd3' (line 398)
    d3_33495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 17), 'd3', False)
    # Getting the type of 'd4' (line 398)
    d4_33496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 21), 'd4', False)
    # Processing the call keyword arguments (line 398)
    kwargs_33497 = {}
    # Getting the type of 'max' (line 398)
    max_33494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 13), 'max', False)
    # Calling max(args, kwargs) (line 398)
    max_call_result_33498 = invoke(stypy.reporting.localization.Localization(__file__, 398, 13), max_33494, *[d3_33495, d4_33496], **kwargs_33497)
    
    # Assigning a type to the variable 'a3' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'a3', max_call_result_33498)
    
    
    # Getting the type of 'a3' (line 399)
    a3_33499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 11), 'a3')
    
    # Obtaining the type of the subscript
    int_33500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 23), 'int')
    # Getting the type of 'theta' (line 399)
    theta_33501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 17), 'theta')
    # Obtaining the member '__getitem__' of a type (line 399)
    getitem___33502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 17), theta_33501, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 399)
    subscript_call_result_33503 = invoke(stypy.reporting.localization.Localization(__file__, 399, 17), getitem___33502, int_33500)
    
    # Applying the binary operator '<=' (line 399)
    result_le_33504 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 11), '<=', a3_33499, subscript_call_result_33503)
    
    # Testing the type of an if condition (line 399)
    if_condition_33505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 399, 8), result_le_33504)
    # Assigning a type to the variable 'if_condition_33505' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'if_condition_33505', if_condition_33505)
    # SSA begins for if statement (line 399)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 400):
    
    # Assigning a Call to a Name (line 400):
    
    # Call to min(...): (line 400)
    # Processing the call arguments (line 400)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 400, 21, True)
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'tuple' (line 400)
    tuple_33514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 400)
    # Adding element type (line 400)
    int_33515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 33), tuple_33514, int_33515)
    # Adding element type (line 400)
    int_33516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 33), tuple_33514, int_33516)
    # Adding element type (line 400)
    int_33517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 33), tuple_33514, int_33517)
    # Adding element type (line 400)
    int_33518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 33), tuple_33514, int_33518)
    # Adding element type (line 400)
    int_33519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 33), tuple_33514, int_33519)
    
    comprehension_33520 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 21), tuple_33514)
    # Assigning a type to the variable 'i' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 21), 'i', comprehension_33520)
    
    # Getting the type of 'a3' (line 400)
    a3_33508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 51), 'a3', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 400)
    i_33509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 63), 'i', False)
    # Getting the type of 'theta' (line 400)
    theta_33510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 57), 'theta', False)
    # Obtaining the member '__getitem__' of a type (line 400)
    getitem___33511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 57), theta_33510, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 400)
    subscript_call_result_33512 = invoke(stypy.reporting.localization.Localization(__file__, 400, 57), getitem___33511, i_33509)
    
    # Applying the binary operator '<=' (line 400)
    result_le_33513 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 51), '<=', a3_33508, subscript_call_result_33512)
    
    # Getting the type of 'i' (line 400)
    i_33507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 21), 'i', False)
    list_33521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 21), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 21), list_33521, i_33507)
    # Processing the call keyword arguments (line 400)
    kwargs_33522 = {}
    # Getting the type of 'min' (line 400)
    min_33506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 17), 'min', False)
    # Calling min(args, kwargs) (line 400)
    min_call_result_33523 = invoke(stypy.reporting.localization.Localization(__file__, 400, 17), min_33506, *[list_33521], **kwargs_33522)
    
    # Assigning a type to the variable 'j1' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'j1', min_call_result_33523)
    
    
    # Getting the type of 'j1' (line 401)
    j1_33524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 15), 'j1')
    int_33525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 21), 'int')
    # Applying the binary operator '<=' (line 401)
    result_le_33526 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 15), '<=', j1_33524, int_33525)
    
    # Testing the type of an if condition (line 401)
    if_condition_33527 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 401, 12), result_le_33526)
    # Assigning a type to the variable 'if_condition_33527' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'if_condition_33527', if_condition_33527)
    # SSA begins for if statement (line 401)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 402):
    
    # Assigning a Name to a Name (line 402):
    # Getting the type of 'j1' (line 402)
    j1_33528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'j1')
    # Assigning a type to the variable 'm' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 16), 'm', j1_33528)
    # SSA branch for the else part of an if statement (line 401)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'a3' (line 404)
    a3_33529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 17), 'a3')
    int_33530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 22), 'int')
    # Applying the binary operator 'div' (line 404)
    result_div_33531 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 17), 'div', a3_33529, int_33530)
    
    
    # Obtaining the type of the subscript
    int_33532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 33), 'int')
    # Getting the type of 'theta' (line 404)
    theta_33533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 27), 'theta')
    # Obtaining the member '__getitem__' of a type (line 404)
    getitem___33534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 27), theta_33533, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 404)
    subscript_call_result_33535 = invoke(stypy.reporting.localization.Localization(__file__, 404, 27), getitem___33534, int_33532)
    
    # Applying the binary operator '<=' (line 404)
    result_le_33536 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 17), '<=', result_div_33531, subscript_call_result_33535)
    
    
    # Getting the type of 'k' (line 404)
    k_33537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 40), 'k')
    int_33538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 44), 'int')
    # Applying the binary operator '<' (line 404)
    result_lt_33539 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 40), '<', k_33537, int_33538)
    
    # Applying the binary operator 'and' (line 404)
    result_and_keyword_33540 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 17), 'and', result_le_33536, result_lt_33539)
    
    # Testing the type of an if condition (line 404)
    if_condition_33541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 404, 17), result_and_keyword_33540)
    # Assigning a type to the variable 'if_condition_33541' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 17), 'if_condition_33541', if_condition_33541)
    # SSA begins for if statement (line 404)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'k' (line 405)
    k_33542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'k')
    int_33543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 21), 'int')
    # Applying the binary operator '+=' (line 405)
    result_iadd_33544 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 16), '+=', k_33542, int_33543)
    # Assigning a type to the variable 'k' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'k', result_iadd_33544)
    
    
    # Assigning a Call to a Name (line 406):
    
    # Assigning a Call to a Name (line 406):
    
    # Call to _sqrtm_triu(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'T' (line 406)
    T_33546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 32), 'T', False)
    # Processing the call keyword arguments (line 406)
    kwargs_33547 = {}
    # Getting the type of '_sqrtm_triu' (line 406)
    _sqrtm_triu_33545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), '_sqrtm_triu', False)
    # Calling _sqrtm_triu(args, kwargs) (line 406)
    _sqrtm_triu_call_result_33548 = invoke(stypy.reporting.localization.Localization(__file__, 406, 20), _sqrtm_triu_33545, *[T_33546], **kwargs_33547)
    
    # Assigning a type to the variable 'T' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'T', _sqrtm_triu_call_result_33548)
    
    # Getting the type of 's' (line 407)
    s_33549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 16), 's')
    int_33550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 21), 'int')
    # Applying the binary operator '+=' (line 407)
    result_iadd_33551 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 16), '+=', s_33549, int_33550)
    # Assigning a type to the variable 's' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 16), 's', result_iadd_33551)
    
    # SSA join for if statement (line 404)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 401)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 399)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 409):
    
    # Assigning a BinOp to a Name (line 409):
    
    # Call to _onenormest_m1_power(...): (line 409)
    # Processing the call arguments (line 409)
    # Getting the type of 'T' (line 409)
    T_33553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 34), 'T', False)
    int_33554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 37), 'int')
    # Processing the call keyword arguments (line 409)
    kwargs_33555 = {}
    # Getting the type of '_onenormest_m1_power' (line 409)
    _onenormest_m1_power_33552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 13), '_onenormest_m1_power', False)
    # Calling _onenormest_m1_power(args, kwargs) (line 409)
    _onenormest_m1_power_call_result_33556 = invoke(stypy.reporting.localization.Localization(__file__, 409, 13), _onenormest_m1_power_33552, *[T_33553, int_33554], **kwargs_33555)
    
    int_33557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 44), 'int')
    int_33558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 46), 'int')
    # Applying the binary operator 'div' (line 409)
    result_div_33559 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 44), 'div', int_33557, int_33558)
    
    # Applying the binary operator '**' (line 409)
    result_pow_33560 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 13), '**', _onenormest_m1_power_call_result_33556, result_div_33559)
    
    # Assigning a type to the variable 'd5' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'd5', result_pow_33560)
    
    # Assigning a Call to a Name (line 410):
    
    # Assigning a Call to a Name (line 410):
    
    # Call to max(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 'd4' (line 410)
    d4_33562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 17), 'd4', False)
    # Getting the type of 'd5' (line 410)
    d5_33563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 21), 'd5', False)
    # Processing the call keyword arguments (line 410)
    kwargs_33564 = {}
    # Getting the type of 'max' (line 410)
    max_33561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 13), 'max', False)
    # Calling max(args, kwargs) (line 410)
    max_call_result_33565 = invoke(stypy.reporting.localization.Localization(__file__, 410, 13), max_33561, *[d4_33562, d5_33563], **kwargs_33564)
    
    # Assigning a type to the variable 'a4' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'a4', max_call_result_33565)
    
    # Assigning a Call to a Name (line 411):
    
    # Assigning a Call to a Name (line 411):
    
    # Call to min(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'a3' (line 411)
    a3_33567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 18), 'a3', False)
    # Getting the type of 'a4' (line 411)
    a4_33568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 22), 'a4', False)
    # Processing the call keyword arguments (line 411)
    kwargs_33569 = {}
    # Getting the type of 'min' (line 411)
    min_33566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 14), 'min', False)
    # Calling min(args, kwargs) (line 411)
    min_call_result_33570 = invoke(stypy.reporting.localization.Localization(__file__, 411, 14), min_33566, *[a3_33567, a4_33568], **kwargs_33569)
    
    # Assigning a type to the variable 'eta' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'eta', min_call_result_33570)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 412)
    tuple_33571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 412)
    # Adding element type (line 412)
    int_33572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 18), tuple_33571, int_33572)
    # Adding element type (line 412)
    int_33573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 18), tuple_33571, int_33573)
    
    # Testing the type of a for loop iterable (line 412)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 412, 8), tuple_33571)
    # Getting the type of the for loop variable (line 412)
    for_loop_var_33574 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 412, 8), tuple_33571)
    # Assigning a type to the variable 'i' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'i', for_loop_var_33574)
    # SSA begins for a for statement (line 412)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'eta' (line 413)
    eta_33575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 15), 'eta')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 413)
    i_33576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 28), 'i')
    # Getting the type of 'theta' (line 413)
    theta_33577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 22), 'theta')
    # Obtaining the member '__getitem__' of a type (line 413)
    getitem___33578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 22), theta_33577, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 413)
    subscript_call_result_33579 = invoke(stypy.reporting.localization.Localization(__file__, 413, 22), getitem___33578, i_33576)
    
    # Applying the binary operator '<=' (line 413)
    result_le_33580 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 15), '<=', eta_33575, subscript_call_result_33579)
    
    # Testing the type of an if condition (line 413)
    if_condition_33581 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 12), result_le_33580)
    # Assigning a type to the variable 'if_condition_33581' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'if_condition_33581', if_condition_33581)
    # SSA begins for if statement (line 413)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 414):
    
    # Assigning a Name to a Name (line 414):
    # Getting the type of 'i' (line 414)
    i_33582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 20), 'i')
    # Assigning a type to the variable 'm' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 16), 'm', i_33582)
    # SSA join for if statement (line 413)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 416)
    # Getting the type of 'm' (line 416)
    m_33583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'm')
    # Getting the type of 'None' (line 416)
    None_33584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 20), 'None')
    
    (may_be_33585, more_types_in_union_33586) = may_not_be_none(m_33583, None_33584)

    if may_be_33585:

        if more_types_in_union_33586:
            # Runtime conditional SSA (line 416)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_33586:
            # SSA join for if statement (line 416)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 418):
    
    # Assigning a Call to a Name (line 418):
    
    # Call to _sqrtm_triu(...): (line 418)
    # Processing the call arguments (line 418)
    # Getting the type of 'T' (line 418)
    T_33588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 24), 'T', False)
    # Processing the call keyword arguments (line 418)
    kwargs_33589 = {}
    # Getting the type of '_sqrtm_triu' (line 418)
    _sqrtm_triu_33587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), '_sqrtm_triu', False)
    # Calling _sqrtm_triu(args, kwargs) (line 418)
    _sqrtm_triu_call_result_33590 = invoke(stypy.reporting.localization.Localization(__file__, 418, 12), _sqrtm_triu_33587, *[T_33588], **kwargs_33589)
    
    # Assigning a type to the variable 'T' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'T', _sqrtm_triu_call_result_33590)
    
    # Getting the type of 's' (line 419)
    s_33591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 's')
    int_33592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 13), 'int')
    # Applying the binary operator '+=' (line 419)
    result_iadd_33593 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 8), '+=', s_33591, int_33592)
    # Assigning a type to the variable 's' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 's', result_iadd_33593)
    
    # SSA join for while statement (line 394)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 424):
    
    # Assigning a BinOp to a Name (line 424):
    # Getting the type of 'T' (line 424)
    T_33594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'T')
    
    # Call to identity(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'n' (line 424)
    n_33597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 24), 'n', False)
    # Processing the call keyword arguments (line 424)
    kwargs_33598 = {}
    # Getting the type of 'np' (line 424)
    np_33595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'np', False)
    # Obtaining the member 'identity' of a type (line 424)
    identity_33596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), np_33595, 'identity')
    # Calling identity(args, kwargs) (line 424)
    identity_call_result_33599 = invoke(stypy.reporting.localization.Localization(__file__, 424, 12), identity_33596, *[n_33597], **kwargs_33598)
    
    # Applying the binary operator '-' (line 424)
    result_sub_33600 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 8), '-', T_33594, identity_call_result_33599)
    
    # Assigning a type to the variable 'R' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'R', result_sub_33600)
    
    # Assigning a Call to a Name (line 431):
    
    # Assigning a Call to a Name (line 431):
    
    # Call to all(...): (line 431)
    # Processing the call arguments (line 431)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 431, 31, True)
    # Calculating comprehension expression
    
    # Call to diag(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'T0' (line 431)
    T0_33613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 74), 'T0', False)
    # Processing the call keyword arguments (line 431)
    kwargs_33614 = {}
    # Getting the type of 'np' (line 431)
    np_33611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 66), 'np', False)
    # Obtaining the member 'diag' of a type (line 431)
    diag_33612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 66), np_33611, 'diag')
    # Calling diag(args, kwargs) (line 431)
    diag_call_result_33615 = invoke(stypy.reporting.localization.Localization(__file__, 431, 66), diag_33612, *[T0_33613], **kwargs_33614)
    
    comprehension_33616 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 31), diag_call_result_33615)
    # Assigning a type to the variable 'x' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 31), 'x', comprehension_33616)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'x' (line 431)
    x_33602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 31), 'x', False)
    # Obtaining the member 'real' of a type (line 431)
    real_33603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 31), x_33602, 'real')
    int_33604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 40), 'int')
    # Applying the binary operator '>' (line 431)
    result_gt_33605 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 31), '>', real_33603, int_33604)
    
    
    # Getting the type of 'x' (line 431)
    x_33606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 45), 'x', False)
    # Obtaining the member 'imag' of a type (line 431)
    imag_33607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 45), x_33606, 'imag')
    int_33608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 55), 'int')
    # Applying the binary operator '!=' (line 431)
    result_ne_33609 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 45), '!=', imag_33607, int_33608)
    
    # Applying the binary operator 'or' (line 431)
    result_or_keyword_33610 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 31), 'or', result_gt_33605, result_ne_33609)
    
    list_33617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 31), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 31), list_33617, result_or_keyword_33610)
    # Processing the call keyword arguments (line 431)
    kwargs_33618 = {}
    # Getting the type of 'all' (line 431)
    all_33601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 27), 'all', False)
    # Calling all(args, kwargs) (line 431)
    all_call_result_33619 = invoke(stypy.reporting.localization.Localization(__file__, 431, 27), all_33601, *[list_33617], **kwargs_33618)
    
    # Assigning a type to the variable 'has_principal_branch' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'has_principal_branch', all_call_result_33619)
    
    # Getting the type of 'has_principal_branch' (line 432)
    has_principal_branch_33620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 7), 'has_principal_branch')
    # Testing the type of an if condition (line 432)
    if_condition_33621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 4), has_principal_branch_33620)
    # Assigning a type to the variable 'if_condition_33621' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'if_condition_33621', if_condition_33621)
    # SSA begins for if statement (line 432)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to range(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'n' (line 433)
    n_33623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 23), 'n', False)
    # Processing the call keyword arguments (line 433)
    kwargs_33624 = {}
    # Getting the type of 'range' (line 433)
    range_33622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 17), 'range', False)
    # Calling range(args, kwargs) (line 433)
    range_call_result_33625 = invoke(stypy.reporting.localization.Localization(__file__, 433, 17), range_33622, *[n_33623], **kwargs_33624)
    
    # Testing the type of a for loop iterable (line 433)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 433, 8), range_call_result_33625)
    # Getting the type of the for loop variable (line 433)
    for_loop_var_33626 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 433, 8), range_call_result_33625)
    # Assigning a type to the variable 'j' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'j', for_loop_var_33626)
    # SSA begins for a for statement (line 433)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 434):
    
    # Assigning a Subscript to a Name (line 434):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 434)
    tuple_33627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 434)
    # Adding element type (line 434)
    # Getting the type of 'j' (line 434)
    j_33628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 19), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 19), tuple_33627, j_33628)
    # Adding element type (line 434)
    # Getting the type of 'j' (line 434)
    j_33629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 22), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 19), tuple_33627, j_33629)
    
    # Getting the type of 'T0' (line 434)
    T0_33630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'T0')
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___33631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 16), T0_33630, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 434)
    subscript_call_result_33632 = invoke(stypy.reporting.localization.Localization(__file__, 434, 16), getitem___33631, tuple_33627)
    
    # Assigning a type to the variable 'a' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'a', subscript_call_result_33632)
    
    # Assigning a Call to a Name (line 435):
    
    # Assigning a Call to a Name (line 435):
    
    # Call to _briggs_helper_function(...): (line 435)
    # Processing the call arguments (line 435)
    # Getting the type of 'a' (line 435)
    a_33634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 40), 'a', False)
    # Getting the type of 's' (line 435)
    s_33635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 43), 's', False)
    # Processing the call keyword arguments (line 435)
    kwargs_33636 = {}
    # Getting the type of '_briggs_helper_function' (line 435)
    _briggs_helper_function_33633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 16), '_briggs_helper_function', False)
    # Calling _briggs_helper_function(args, kwargs) (line 435)
    _briggs_helper_function_call_result_33637 = invoke(stypy.reporting.localization.Localization(__file__, 435, 16), _briggs_helper_function_33633, *[a_33634, s_33635], **kwargs_33636)
    
    # Assigning a type to the variable 'r' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'r', _briggs_helper_function_call_result_33637)
    
    # Assigning a Name to a Subscript (line 436):
    
    # Assigning a Name to a Subscript (line 436):
    # Getting the type of 'r' (line 436)
    r_33638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 22), 'r')
    # Getting the type of 'R' (line 436)
    R_33639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'R')
    
    # Obtaining an instance of the builtin type 'tuple' (line 436)
    tuple_33640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 436)
    # Adding element type (line 436)
    # Getting the type of 'j' (line 436)
    j_33641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 14), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 14), tuple_33640, j_33641)
    # Adding element type (line 436)
    # Getting the type of 'j' (line 436)
    j_33642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 17), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 14), tuple_33640, j_33642)
    
    # Storing an element on a container (line 436)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 12), R_33639, (tuple_33640, r_33638))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 437):
    
    # Assigning a Call to a Name (line 437):
    
    # Call to exp2(...): (line 437)
    # Processing the call arguments (line 437)
    
    # Getting the type of 's' (line 437)
    s_33645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 21), 's', False)
    # Applying the 'usub' unary operator (line 437)
    result___neg___33646 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 20), 'usub', s_33645)
    
    # Processing the call keyword arguments (line 437)
    kwargs_33647 = {}
    # Getting the type of 'np' (line 437)
    np_33643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'np', False)
    # Obtaining the member 'exp2' of a type (line 437)
    exp2_33644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 12), np_33643, 'exp2')
    # Calling exp2(args, kwargs) (line 437)
    exp2_call_result_33648 = invoke(stypy.reporting.localization.Localization(__file__, 437, 12), exp2_33644, *[result___neg___33646], **kwargs_33647)
    
    # Assigning a type to the variable 'p' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'p', exp2_call_result_33648)
    
    
    # Call to range(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'n' (line 438)
    n_33650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 23), 'n', False)
    int_33651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 25), 'int')
    # Applying the binary operator '-' (line 438)
    result_sub_33652 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 23), '-', n_33650, int_33651)
    
    # Processing the call keyword arguments (line 438)
    kwargs_33653 = {}
    # Getting the type of 'range' (line 438)
    range_33649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 17), 'range', False)
    # Calling range(args, kwargs) (line 438)
    range_call_result_33654 = invoke(stypy.reporting.localization.Localization(__file__, 438, 17), range_33649, *[result_sub_33652], **kwargs_33653)
    
    # Testing the type of a for loop iterable (line 438)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 438, 8), range_call_result_33654)
    # Getting the type of the for loop variable (line 438)
    for_loop_var_33655 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 438, 8), range_call_result_33654)
    # Assigning a type to the variable 'j' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'j', for_loop_var_33655)
    # SSA begins for a for statement (line 438)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 439):
    
    # Assigning a Subscript to a Name (line 439):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 439)
    tuple_33656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 439)
    # Adding element type (line 439)
    # Getting the type of 'j' (line 439)
    j_33657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 20), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 20), tuple_33656, j_33657)
    # Adding element type (line 439)
    # Getting the type of 'j' (line 439)
    j_33658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 23), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 20), tuple_33656, j_33658)
    
    # Getting the type of 'T0' (line 439)
    T0_33659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 17), 'T0')
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___33660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 17), T0_33659, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_33661 = invoke(stypy.reporting.localization.Localization(__file__, 439, 17), getitem___33660, tuple_33656)
    
    # Assigning a type to the variable 'l1' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'l1', subscript_call_result_33661)
    
    # Assigning a Subscript to a Name (line 440):
    
    # Assigning a Subscript to a Name (line 440):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 440)
    tuple_33662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 440)
    # Adding element type (line 440)
    # Getting the type of 'j' (line 440)
    j_33663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 20), 'j')
    int_33664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 22), 'int')
    # Applying the binary operator '+' (line 440)
    result_add_33665 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 20), '+', j_33663, int_33664)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 20), tuple_33662, result_add_33665)
    # Adding element type (line 440)
    # Getting the type of 'j' (line 440)
    j_33666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'j')
    int_33667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 27), 'int')
    # Applying the binary operator '+' (line 440)
    result_add_33668 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 25), '+', j_33666, int_33667)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 20), tuple_33662, result_add_33668)
    
    # Getting the type of 'T0' (line 440)
    T0_33669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 17), 'T0')
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___33670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 17), T0_33669, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_33671 = invoke(stypy.reporting.localization.Localization(__file__, 440, 17), getitem___33670, tuple_33662)
    
    # Assigning a type to the variable 'l2' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'l2', subscript_call_result_33671)
    
    # Assigning a Subscript to a Name (line 441):
    
    # Assigning a Subscript to a Name (line 441):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 441)
    tuple_33672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 441)
    # Adding element type (line 441)
    # Getting the type of 'j' (line 441)
    j_33673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 21), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 21), tuple_33672, j_33673)
    # Adding element type (line 441)
    # Getting the type of 'j' (line 441)
    j_33674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 24), 'j')
    int_33675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 26), 'int')
    # Applying the binary operator '+' (line 441)
    result_add_33676 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 24), '+', j_33674, int_33675)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 21), tuple_33672, result_add_33676)
    
    # Getting the type of 'T0' (line 441)
    T0_33677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 18), 'T0')
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___33678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 18), T0_33677, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_33679 = invoke(stypy.reporting.localization.Localization(__file__, 441, 18), getitem___33678, tuple_33672)
    
    # Assigning a type to the variable 't12' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 't12', subscript_call_result_33679)
    
    # Assigning a Call to a Name (line 442):
    
    # Assigning a Call to a Name (line 442):
    
    # Call to _fractional_power_superdiag_entry(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'l1' (line 442)
    l1_33681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 52), 'l1', False)
    # Getting the type of 'l2' (line 442)
    l2_33682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 56), 'l2', False)
    # Getting the type of 't12' (line 442)
    t12_33683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 60), 't12', False)
    # Getting the type of 'p' (line 442)
    p_33684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 65), 'p', False)
    # Processing the call keyword arguments (line 442)
    kwargs_33685 = {}
    # Getting the type of '_fractional_power_superdiag_entry' (line 442)
    _fractional_power_superdiag_entry_33680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 18), '_fractional_power_superdiag_entry', False)
    # Calling _fractional_power_superdiag_entry(args, kwargs) (line 442)
    _fractional_power_superdiag_entry_call_result_33686 = invoke(stypy.reporting.localization.Localization(__file__, 442, 18), _fractional_power_superdiag_entry_33680, *[l1_33681, l2_33682, t12_33683, p_33684], **kwargs_33685)
    
    # Assigning a type to the variable 'f12' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'f12', _fractional_power_superdiag_entry_call_result_33686)
    
    # Assigning a Name to a Subscript (line 443):
    
    # Assigning a Name to a Subscript (line 443):
    # Getting the type of 'f12' (line 443)
    f12_33687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 24), 'f12')
    # Getting the type of 'R' (line 443)
    R_33688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 12), 'R')
    
    # Obtaining an instance of the builtin type 'tuple' (line 443)
    tuple_33689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 443)
    # Adding element type (line 443)
    # Getting the type of 'j' (line 443)
    j_33690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 14), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 14), tuple_33689, j_33690)
    # Adding element type (line 443)
    # Getting the type of 'j' (line 443)
    j_33691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 17), 'j')
    int_33692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 19), 'int')
    # Applying the binary operator '+' (line 443)
    result_add_33693 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 17), '+', j_33691, int_33692)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 14), tuple_33689, result_add_33693)
    
    # Storing an element on a container (line 443)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 12), R_33688, (tuple_33689, f12_33687))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 432)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to array_equal(...): (line 446)
    # Processing the call arguments (line 446)
    # Getting the type of 'R' (line 446)
    R_33696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 26), 'R', False)
    
    # Call to triu(...): (line 446)
    # Processing the call arguments (line 446)
    # Getting the type of 'R' (line 446)
    R_33699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 37), 'R', False)
    # Processing the call keyword arguments (line 446)
    kwargs_33700 = {}
    # Getting the type of 'np' (line 446)
    np_33697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 29), 'np', False)
    # Obtaining the member 'triu' of a type (line 446)
    triu_33698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 29), np_33697, 'triu')
    # Calling triu(args, kwargs) (line 446)
    triu_call_result_33701 = invoke(stypy.reporting.localization.Localization(__file__, 446, 29), triu_33698, *[R_33699], **kwargs_33700)
    
    # Processing the call keyword arguments (line 446)
    kwargs_33702 = {}
    # Getting the type of 'np' (line 446)
    np_33694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 11), 'np', False)
    # Obtaining the member 'array_equal' of a type (line 446)
    array_equal_33695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 11), np_33694, 'array_equal')
    # Calling array_equal(args, kwargs) (line 446)
    array_equal_call_result_33703 = invoke(stypy.reporting.localization.Localization(__file__, 446, 11), array_equal_33695, *[R_33696, triu_call_result_33701], **kwargs_33702)
    
    # Applying the 'not' unary operator (line 446)
    result_not__33704 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 7), 'not', array_equal_call_result_33703)
    
    # Testing the type of an if condition (line 446)
    if_condition_33705 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 446, 4), result_not__33704)
    # Assigning a type to the variable 'if_condition_33705' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'if_condition_33705', if_condition_33705)
    # SSA begins for if statement (line 446)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 447)
    # Processing the call arguments (line 447)
    str_33707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 24), 'str', 'internal inconsistency')
    # Processing the call keyword arguments (line 447)
    kwargs_33708 = {}
    # Getting the type of 'Exception' (line 447)
    Exception_33706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 447)
    Exception_call_result_33709 = invoke(stypy.reporting.localization.Localization(__file__, 447, 14), Exception_33706, *[str_33707], **kwargs_33708)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 447, 8), Exception_call_result_33709, 'raise parameter', BaseException)
    # SSA join for if statement (line 446)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 448)
    tuple_33710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 448)
    # Adding element type (line 448)
    # Getting the type of 'R' (line 448)
    R_33711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 11), 'R')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 11), tuple_33710, R_33711)
    # Adding element type (line 448)
    # Getting the type of 's' (line 448)
    s_33712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 14), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 11), tuple_33710, s_33712)
    # Adding element type (line 448)
    # Getting the type of 'm' (line 448)
    m_33713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 17), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 11), tuple_33710, m_33713)
    
    # Assigning a type to the variable 'stypy_return_type' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'stypy_return_type', tuple_33710)
    
    # ################# End of '_inverse_squaring_helper(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_inverse_squaring_helper' in the type store
    # Getting the type of 'stypy_return_type' (line 314)
    stypy_return_type_33714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33714)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_inverse_squaring_helper'
    return stypy_return_type_33714

# Assigning a type to the variable '_inverse_squaring_helper' (line 314)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 0), '_inverse_squaring_helper', _inverse_squaring_helper)

@norecursion
def _fractional_power_pade_constant(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fractional_power_pade_constant'
    module_type_store = module_type_store.open_function_context('_fractional_power_pade_constant', 451, 0, False)
    
    # Passed parameters checking function
    _fractional_power_pade_constant.stypy_localization = localization
    _fractional_power_pade_constant.stypy_type_of_self = None
    _fractional_power_pade_constant.stypy_type_store = module_type_store
    _fractional_power_pade_constant.stypy_function_name = '_fractional_power_pade_constant'
    _fractional_power_pade_constant.stypy_param_names_list = ['i', 't']
    _fractional_power_pade_constant.stypy_varargs_param_name = None
    _fractional_power_pade_constant.stypy_kwargs_param_name = None
    _fractional_power_pade_constant.stypy_call_defaults = defaults
    _fractional_power_pade_constant.stypy_call_varargs = varargs
    _fractional_power_pade_constant.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fractional_power_pade_constant', ['i', 't'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fractional_power_pade_constant', localization, ['i', 't'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fractional_power_pade_constant(...)' code ##################

    
    
    # Getting the type of 'i' (line 453)
    i_33715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 7), 'i')
    int_33716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 11), 'int')
    # Applying the binary operator '<' (line 453)
    result_lt_33717 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 7), '<', i_33715, int_33716)
    
    # Testing the type of an if condition (line 453)
    if_condition_33718 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 4), result_lt_33717)
    # Assigning a type to the variable 'if_condition_33718' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'if_condition_33718', if_condition_33718)
    # SSA begins for if statement (line 453)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 454)
    # Processing the call arguments (line 454)
    str_33720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 25), 'str', 'expected a positive integer i')
    # Processing the call keyword arguments (line 454)
    kwargs_33721 = {}
    # Getting the type of 'ValueError' (line 454)
    ValueError_33719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 454)
    ValueError_call_result_33722 = invoke(stypy.reporting.localization.Localization(__file__, 454, 14), ValueError_33719, *[str_33720], **kwargs_33721)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 454, 8), ValueError_call_result_33722, 'raise parameter', BaseException)
    # SSA join for if statement (line 453)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    int_33723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 12), 'int')
    # Getting the type of 't' (line 455)
    t_33724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 17), 't')
    # Applying the binary operator '<' (line 455)
    result_lt_33725 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 12), '<', int_33723, t_33724)
    int_33726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 21), 'int')
    # Applying the binary operator '<' (line 455)
    result_lt_33727 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 12), '<', t_33724, int_33726)
    # Applying the binary operator '&' (line 455)
    result_and__33728 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 12), '&', result_lt_33725, result_lt_33727)
    
    # Applying the 'not' unary operator (line 455)
    result_not__33729 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 7), 'not', result_and__33728)
    
    # Testing the type of an if condition (line 455)
    if_condition_33730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 4), result_not__33729)
    # Assigning a type to the variable 'if_condition_33730' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'if_condition_33730', if_condition_33730)
    # SSA begins for if statement (line 455)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 456)
    # Processing the call arguments (line 456)
    str_33732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 25), 'str', 'expected -1 < t < 1')
    # Processing the call keyword arguments (line 456)
    kwargs_33733 = {}
    # Getting the type of 'ValueError' (line 456)
    ValueError_33731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 456)
    ValueError_call_result_33734 = invoke(stypy.reporting.localization.Localization(__file__, 456, 14), ValueError_33731, *[str_33732], **kwargs_33733)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 456, 8), ValueError_call_result_33734, 'raise parameter', BaseException)
    # SSA join for if statement (line 455)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'i' (line 457)
    i_33735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 7), 'i')
    int_33736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 12), 'int')
    # Applying the binary operator '==' (line 457)
    result_eq_33737 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 7), '==', i_33735, int_33736)
    
    # Testing the type of an if condition (line 457)
    if_condition_33738 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 457, 4), result_eq_33737)
    # Assigning a type to the variable 'if_condition_33738' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'if_condition_33738', if_condition_33738)
    # SSA begins for if statement (line 457)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 't' (line 458)
    t_33739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 16), 't')
    # Applying the 'usub' unary operator (line 458)
    result___neg___33740 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 15), 'usub', t_33739)
    
    # Assigning a type to the variable 'stypy_return_type' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'stypy_return_type', result___neg___33740)
    # SSA branch for the else part of an if statement (line 457)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'i' (line 459)
    i_33741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 9), 'i')
    int_33742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 13), 'int')
    # Applying the binary operator '%' (line 459)
    result_mod_33743 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 9), '%', i_33741, int_33742)
    
    int_33744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 18), 'int')
    # Applying the binary operator '==' (line 459)
    result_eq_33745 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 9), '==', result_mod_33743, int_33744)
    
    # Testing the type of an if condition (line 459)
    if_condition_33746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 9), result_eq_33745)
    # Assigning a type to the variable 'if_condition_33746' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 9), 'if_condition_33746', if_condition_33746)
    # SSA begins for if statement (line 459)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 460):
    
    # Assigning a BinOp to a Name (line 460):
    # Getting the type of 'i' (line 460)
    i_33747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'i')
    int_33748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 17), 'int')
    # Applying the binary operator '//' (line 460)
    result_floordiv_33749 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 12), '//', i_33747, int_33748)
    
    # Assigning a type to the variable 'j' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'j', result_floordiv_33749)
    
    # Getting the type of 'j' (line 461)
    j_33750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 17), 'j')
    # Applying the 'usub' unary operator (line 461)
    result___neg___33751 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 16), 'usub', j_33750)
    
    # Getting the type of 't' (line 461)
    t_33752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 21), 't')
    # Applying the binary operator '+' (line 461)
    result_add_33753 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 16), '+', result___neg___33751, t_33752)
    
    int_33754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 27), 'int')
    int_33755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 32), 'int')
    # Getting the type of 'j' (line 461)
    j_33756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 34), 'j')
    # Applying the binary operator '*' (line 461)
    result_mul_33757 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 32), '*', int_33755, j_33756)
    
    int_33758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 38), 'int')
    # Applying the binary operator '-' (line 461)
    result_sub_33759 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 32), '-', result_mul_33757, int_33758)
    
    # Applying the binary operator '*' (line 461)
    result_mul_33760 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 27), '*', int_33754, result_sub_33759)
    
    # Applying the binary operator 'div' (line 461)
    result_div_33761 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 15), 'div', result_add_33753, result_mul_33760)
    
    # Assigning a type to the variable 'stypy_return_type' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'stypy_return_type', result_div_33761)
    # SSA branch for the else part of an if statement (line 459)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'i' (line 462)
    i_33762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 9), 'i')
    int_33763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 13), 'int')
    # Applying the binary operator '%' (line 462)
    result_mod_33764 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 9), '%', i_33762, int_33763)
    
    int_33765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 18), 'int')
    # Applying the binary operator '==' (line 462)
    result_eq_33766 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 9), '==', result_mod_33764, int_33765)
    
    # Testing the type of an if condition (line 462)
    if_condition_33767 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 9), result_eq_33766)
    # Assigning a type to the variable 'if_condition_33767' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 9), 'if_condition_33767', if_condition_33767)
    # SSA begins for if statement (line 462)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 463):
    
    # Assigning a BinOp to a Name (line 463):
    # Getting the type of 'i' (line 463)
    i_33768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 13), 'i')
    int_33769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 17), 'int')
    # Applying the binary operator '-' (line 463)
    result_sub_33770 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 13), '-', i_33768, int_33769)
    
    int_33771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 23), 'int')
    # Applying the binary operator '//' (line 463)
    result_floordiv_33772 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 12), '//', result_sub_33770, int_33771)
    
    # Assigning a type to the variable 'j' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'j', result_floordiv_33772)
    
    # Getting the type of 'j' (line 464)
    j_33773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 17), 'j')
    # Applying the 'usub' unary operator (line 464)
    result___neg___33774 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 16), 'usub', j_33773)
    
    # Getting the type of 't' (line 464)
    t_33775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 21), 't')
    # Applying the binary operator '-' (line 464)
    result_sub_33776 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 16), '-', result___neg___33774, t_33775)
    
    int_33777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 27), 'int')
    int_33778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 32), 'int')
    # Getting the type of 'j' (line 464)
    j_33779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 34), 'j')
    # Applying the binary operator '*' (line 464)
    result_mul_33780 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 32), '*', int_33778, j_33779)
    
    int_33781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 38), 'int')
    # Applying the binary operator '+' (line 464)
    result_add_33782 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 32), '+', result_mul_33780, int_33781)
    
    # Applying the binary operator '*' (line 464)
    result_mul_33783 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 27), '*', int_33777, result_add_33782)
    
    # Applying the binary operator 'div' (line 464)
    result_div_33784 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 15), 'div', result_sub_33776, result_mul_33783)
    
    # Assigning a type to the variable 'stypy_return_type' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'stypy_return_type', result_div_33784)
    # SSA branch for the else part of an if statement (line 462)
    module_type_store.open_ssa_branch('else')
    
    # Call to Exception(...): (line 466)
    # Processing the call arguments (line 466)
    str_33786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 24), 'str', 'internal error')
    # Processing the call keyword arguments (line 466)
    kwargs_33787 = {}
    # Getting the type of 'Exception' (line 466)
    Exception_33785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 466)
    Exception_call_result_33788 = invoke(stypy.reporting.localization.Localization(__file__, 466, 14), Exception_33785, *[str_33786], **kwargs_33787)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 466, 8), Exception_call_result_33788, 'raise parameter', BaseException)
    # SSA join for if statement (line 462)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 459)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 457)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_fractional_power_pade_constant(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fractional_power_pade_constant' in the type store
    # Getting the type of 'stypy_return_type' (line 451)
    stypy_return_type_33789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33789)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fractional_power_pade_constant'
    return stypy_return_type_33789

# Assigning a type to the variable '_fractional_power_pade_constant' (line 451)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 0), '_fractional_power_pade_constant', _fractional_power_pade_constant)

@norecursion
def _fractional_power_pade(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fractional_power_pade'
    module_type_store = module_type_store.open_function_context('_fractional_power_pade', 469, 0, False)
    
    # Passed parameters checking function
    _fractional_power_pade.stypy_localization = localization
    _fractional_power_pade.stypy_type_of_self = None
    _fractional_power_pade.stypy_type_store = module_type_store
    _fractional_power_pade.stypy_function_name = '_fractional_power_pade'
    _fractional_power_pade.stypy_param_names_list = ['R', 't', 'm']
    _fractional_power_pade.stypy_varargs_param_name = None
    _fractional_power_pade.stypy_kwargs_param_name = None
    _fractional_power_pade.stypy_call_defaults = defaults
    _fractional_power_pade.stypy_call_varargs = varargs
    _fractional_power_pade.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fractional_power_pade', ['R', 't', 'm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fractional_power_pade', localization, ['R', 't', 'm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fractional_power_pade(...)' code ##################

    str_33790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, (-1)), 'str', '\n    Evaluate the Pade approximation of a fractional matrix power.\n\n    Evaluate the degree-m Pade approximation of R\n    to the fractional matrix power t using the continued fraction\n    in bottom-up fashion using algorithm (4.1) in [1]_.\n\n    Parameters\n    ----------\n    R : (N, N) array_like\n        Upper triangular matrix whose fractional power to evaluate.\n    t : float\n        Fractional power between -1 and 1 exclusive.\n    m : positive integer\n        Degree of Pade approximation.\n\n    Returns\n    -------\n    U : (N, N) array_like\n        The degree-m Pade approximation of R to the fractional power t.\n        This matrix will be upper triangular.\n\n    References\n    ----------\n    .. [1] Nicholas J. Higham and Lijing lin (2011)\n           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."\n           SIAM Journal on Matrix Analysis and Applications,\n           32 (3). pp. 1056-1078. ISSN 0895-4798\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'm' (line 500)
    m_33791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 7), 'm')
    int_33792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 11), 'int')
    # Applying the binary operator '<' (line 500)
    result_lt_33793 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 7), '<', m_33791, int_33792)
    
    
    
    # Call to int(...): (line 500)
    # Processing the call arguments (line 500)
    # Getting the type of 'm' (line 500)
    m_33795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 20), 'm', False)
    # Processing the call keyword arguments (line 500)
    kwargs_33796 = {}
    # Getting the type of 'int' (line 500)
    int_33794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 16), 'int', False)
    # Calling int(args, kwargs) (line 500)
    int_call_result_33797 = invoke(stypy.reporting.localization.Localization(__file__, 500, 16), int_33794, *[m_33795], **kwargs_33796)
    
    # Getting the type of 'm' (line 500)
    m_33798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 26), 'm')
    # Applying the binary operator '!=' (line 500)
    result_ne_33799 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 16), '!=', int_call_result_33797, m_33798)
    
    # Applying the binary operator 'or' (line 500)
    result_or_keyword_33800 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 7), 'or', result_lt_33793, result_ne_33799)
    
    # Testing the type of an if condition (line 500)
    if_condition_33801 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 500, 4), result_or_keyword_33800)
    # Assigning a type to the variable 'if_condition_33801' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'if_condition_33801', if_condition_33801)
    # SSA begins for if statement (line 500)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 501)
    # Processing the call arguments (line 501)
    str_33803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 25), 'str', 'expected a positive integer m')
    # Processing the call keyword arguments (line 501)
    kwargs_33804 = {}
    # Getting the type of 'ValueError' (line 501)
    ValueError_33802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 501)
    ValueError_call_result_33805 = invoke(stypy.reporting.localization.Localization(__file__, 501, 14), ValueError_33802, *[str_33803], **kwargs_33804)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 501, 8), ValueError_call_result_33805, 'raise parameter', BaseException)
    # SSA join for if statement (line 500)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    int_33806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 12), 'int')
    # Getting the type of 't' (line 502)
    t_33807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 17), 't')
    # Applying the binary operator '<' (line 502)
    result_lt_33808 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 12), '<', int_33806, t_33807)
    int_33809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 21), 'int')
    # Applying the binary operator '<' (line 502)
    result_lt_33810 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 12), '<', t_33807, int_33809)
    # Applying the binary operator '&' (line 502)
    result_and__33811 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 12), '&', result_lt_33808, result_lt_33810)
    
    # Applying the 'not' unary operator (line 502)
    result_not__33812 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 7), 'not', result_and__33811)
    
    # Testing the type of an if condition (line 502)
    if_condition_33813 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 4), result_not__33812)
    # Assigning a type to the variable 'if_condition_33813' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'if_condition_33813', if_condition_33813)
    # SSA begins for if statement (line 502)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 503)
    # Processing the call arguments (line 503)
    str_33815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 25), 'str', 'expected -1 < t < 1')
    # Processing the call keyword arguments (line 503)
    kwargs_33816 = {}
    # Getting the type of 'ValueError' (line 503)
    ValueError_33814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 503)
    ValueError_call_result_33817 = invoke(stypy.reporting.localization.Localization(__file__, 503, 14), ValueError_33814, *[str_33815], **kwargs_33816)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 503, 8), ValueError_call_result_33817, 'raise parameter', BaseException)
    # SSA join for if statement (line 502)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 504):
    
    # Assigning a Call to a Name (line 504):
    
    # Call to asarray(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'R' (line 504)
    R_33820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 19), 'R', False)
    # Processing the call keyword arguments (line 504)
    kwargs_33821 = {}
    # Getting the type of 'np' (line 504)
    np_33818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 504)
    asarray_33819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 8), np_33818, 'asarray')
    # Calling asarray(args, kwargs) (line 504)
    asarray_call_result_33822 = invoke(stypy.reporting.localization.Localization(__file__, 504, 8), asarray_33819, *[R_33820], **kwargs_33821)
    
    # Assigning a type to the variable 'R' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'R', asarray_call_result_33822)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 505)
    # Processing the call arguments (line 505)
    # Getting the type of 'R' (line 505)
    R_33824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 11), 'R', False)
    # Obtaining the member 'shape' of a type (line 505)
    shape_33825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 11), R_33824, 'shape')
    # Processing the call keyword arguments (line 505)
    kwargs_33826 = {}
    # Getting the type of 'len' (line 505)
    len_33823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 7), 'len', False)
    # Calling len(args, kwargs) (line 505)
    len_call_result_33827 = invoke(stypy.reporting.localization.Localization(__file__, 505, 7), len_33823, *[shape_33825], **kwargs_33826)
    
    int_33828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 23), 'int')
    # Applying the binary operator '!=' (line 505)
    result_ne_33829 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 7), '!=', len_call_result_33827, int_33828)
    
    
    
    # Obtaining the type of the subscript
    int_33830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 36), 'int')
    # Getting the type of 'R' (line 505)
    R_33831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 28), 'R')
    # Obtaining the member 'shape' of a type (line 505)
    shape_33832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 28), R_33831, 'shape')
    # Obtaining the member '__getitem__' of a type (line 505)
    getitem___33833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 28), shape_33832, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 505)
    subscript_call_result_33834 = invoke(stypy.reporting.localization.Localization(__file__, 505, 28), getitem___33833, int_33830)
    
    
    # Obtaining the type of the subscript
    int_33835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 50), 'int')
    # Getting the type of 'R' (line 505)
    R_33836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 42), 'R')
    # Obtaining the member 'shape' of a type (line 505)
    shape_33837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 42), R_33836, 'shape')
    # Obtaining the member '__getitem__' of a type (line 505)
    getitem___33838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 42), shape_33837, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 505)
    subscript_call_result_33839 = invoke(stypy.reporting.localization.Localization(__file__, 505, 42), getitem___33838, int_33835)
    
    # Applying the binary operator '!=' (line 505)
    result_ne_33840 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 28), '!=', subscript_call_result_33834, subscript_call_result_33839)
    
    # Applying the binary operator 'or' (line 505)
    result_or_keyword_33841 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 7), 'or', result_ne_33829, result_ne_33840)
    
    # Testing the type of an if condition (line 505)
    if_condition_33842 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 505, 4), result_or_keyword_33841)
    # Assigning a type to the variable 'if_condition_33842' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'if_condition_33842', if_condition_33842)
    # SSA begins for if statement (line 505)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 506)
    # Processing the call arguments (line 506)
    str_33844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 25), 'str', 'expected an upper triangular square matrix')
    # Processing the call keyword arguments (line 506)
    kwargs_33845 = {}
    # Getting the type of 'ValueError' (line 506)
    ValueError_33843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 506)
    ValueError_call_result_33846 = invoke(stypy.reporting.localization.Localization(__file__, 506, 14), ValueError_33843, *[str_33844], **kwargs_33845)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 506, 8), ValueError_call_result_33846, 'raise parameter', BaseException)
    # SSA join for if statement (line 505)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 507):
    
    # Assigning a Subscript to a Name (line 507):
    
    # Obtaining the type of the subscript
    int_33847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 4), 'int')
    # Getting the type of 'R' (line 507)
    R_33848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 11), 'R')
    # Obtaining the member 'shape' of a type (line 507)
    shape_33849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 11), R_33848, 'shape')
    # Obtaining the member '__getitem__' of a type (line 507)
    getitem___33850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 4), shape_33849, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 507)
    subscript_call_result_33851 = invoke(stypy.reporting.localization.Localization(__file__, 507, 4), getitem___33850, int_33847)
    
    # Assigning a type to the variable 'tuple_var_assignment_32840' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'tuple_var_assignment_32840', subscript_call_result_33851)
    
    # Assigning a Subscript to a Name (line 507):
    
    # Obtaining the type of the subscript
    int_33852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 4), 'int')
    # Getting the type of 'R' (line 507)
    R_33853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 11), 'R')
    # Obtaining the member 'shape' of a type (line 507)
    shape_33854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 11), R_33853, 'shape')
    # Obtaining the member '__getitem__' of a type (line 507)
    getitem___33855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 4), shape_33854, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 507)
    subscript_call_result_33856 = invoke(stypy.reporting.localization.Localization(__file__, 507, 4), getitem___33855, int_33852)
    
    # Assigning a type to the variable 'tuple_var_assignment_32841' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'tuple_var_assignment_32841', subscript_call_result_33856)
    
    # Assigning a Name to a Name (line 507):
    # Getting the type of 'tuple_var_assignment_32840' (line 507)
    tuple_var_assignment_32840_33857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'tuple_var_assignment_32840')
    # Assigning a type to the variable 'n' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'n', tuple_var_assignment_32840_33857)
    
    # Assigning a Name to a Name (line 507):
    # Getting the type of 'tuple_var_assignment_32841' (line 507)
    tuple_var_assignment_32841_33858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'tuple_var_assignment_32841')
    # Assigning a type to the variable 'n' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 7), 'n', tuple_var_assignment_32841_33858)
    
    # Assigning a Call to a Name (line 508):
    
    # Assigning a Call to a Name (line 508):
    
    # Call to identity(...): (line 508)
    # Processing the call arguments (line 508)
    # Getting the type of 'n' (line 508)
    n_33861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 24), 'n', False)
    # Processing the call keyword arguments (line 508)
    kwargs_33862 = {}
    # Getting the type of 'np' (line 508)
    np_33859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'np', False)
    # Obtaining the member 'identity' of a type (line 508)
    identity_33860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 12), np_33859, 'identity')
    # Calling identity(args, kwargs) (line 508)
    identity_call_result_33863 = invoke(stypy.reporting.localization.Localization(__file__, 508, 12), identity_33860, *[n_33861], **kwargs_33862)
    
    # Assigning a type to the variable 'ident' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'ident', identity_call_result_33863)
    
    # Assigning a BinOp to a Name (line 509):
    
    # Assigning a BinOp to a Name (line 509):
    # Getting the type of 'R' (line 509)
    R_33864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'R')
    
    # Call to _fractional_power_pade_constant(...): (line 509)
    # Processing the call arguments (line 509)
    int_33866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 44), 'int')
    # Getting the type of 'm' (line 509)
    m_33867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 46), 'm', False)
    # Applying the binary operator '*' (line 509)
    result_mul_33868 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 44), '*', int_33866, m_33867)
    
    # Getting the type of 't' (line 509)
    t_33869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 49), 't', False)
    # Processing the call keyword arguments (line 509)
    kwargs_33870 = {}
    # Getting the type of '_fractional_power_pade_constant' (line 509)
    _fractional_power_pade_constant_33865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), '_fractional_power_pade_constant', False)
    # Calling _fractional_power_pade_constant(args, kwargs) (line 509)
    _fractional_power_pade_constant_call_result_33871 = invoke(stypy.reporting.localization.Localization(__file__, 509, 12), _fractional_power_pade_constant_33865, *[result_mul_33868, t_33869], **kwargs_33870)
    
    # Applying the binary operator '*' (line 509)
    result_mul_33872 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 8), '*', R_33864, _fractional_power_pade_constant_call_result_33871)
    
    # Assigning a type to the variable 'Y' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'Y', result_mul_33872)
    
    
    # Call to range(...): (line 510)
    # Processing the call arguments (line 510)
    int_33874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 19), 'int')
    # Getting the type of 'm' (line 510)
    m_33875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 21), 'm', False)
    # Applying the binary operator '*' (line 510)
    result_mul_33876 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 19), '*', int_33874, m_33875)
    
    int_33877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 25), 'int')
    # Applying the binary operator '-' (line 510)
    result_sub_33878 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 19), '-', result_mul_33876, int_33877)
    
    int_33879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 28), 'int')
    int_33880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 31), 'int')
    # Processing the call keyword arguments (line 510)
    kwargs_33881 = {}
    # Getting the type of 'range' (line 510)
    range_33873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 13), 'range', False)
    # Calling range(args, kwargs) (line 510)
    range_call_result_33882 = invoke(stypy.reporting.localization.Localization(__file__, 510, 13), range_33873, *[result_sub_33878, int_33879, int_33880], **kwargs_33881)
    
    # Testing the type of a for loop iterable (line 510)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 510, 4), range_call_result_33882)
    # Getting the type of the for loop variable (line 510)
    for_loop_var_33883 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 510, 4), range_call_result_33882)
    # Assigning a type to the variable 'j' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'j', for_loop_var_33883)
    # SSA begins for a for statement (line 510)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 511):
    
    # Assigning a BinOp to a Name (line 511):
    # Getting the type of 'R' (line 511)
    R_33884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 14), 'R')
    
    # Call to _fractional_power_pade_constant(...): (line 511)
    # Processing the call arguments (line 511)
    # Getting the type of 'j' (line 511)
    j_33886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 50), 'j', False)
    # Getting the type of 't' (line 511)
    t_33887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 53), 't', False)
    # Processing the call keyword arguments (line 511)
    kwargs_33888 = {}
    # Getting the type of '_fractional_power_pade_constant' (line 511)
    _fractional_power_pade_constant_33885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 18), '_fractional_power_pade_constant', False)
    # Calling _fractional_power_pade_constant(args, kwargs) (line 511)
    _fractional_power_pade_constant_call_result_33889 = invoke(stypy.reporting.localization.Localization(__file__, 511, 18), _fractional_power_pade_constant_33885, *[j_33886, t_33887], **kwargs_33888)
    
    # Applying the binary operator '*' (line 511)
    result_mul_33890 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 14), '*', R_33884, _fractional_power_pade_constant_call_result_33889)
    
    # Assigning a type to the variable 'rhs' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'rhs', result_mul_33890)
    
    # Assigning a Call to a Name (line 512):
    
    # Assigning a Call to a Name (line 512):
    
    # Call to solve_triangular(...): (line 512)
    # Processing the call arguments (line 512)
    # Getting the type of 'ident' (line 512)
    ident_33892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 29), 'ident', False)
    # Getting the type of 'Y' (line 512)
    Y_33893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 37), 'Y', False)
    # Applying the binary operator '+' (line 512)
    result_add_33894 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 29), '+', ident_33892, Y_33893)
    
    # Getting the type of 'rhs' (line 512)
    rhs_33895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 40), 'rhs', False)
    # Processing the call keyword arguments (line 512)
    kwargs_33896 = {}
    # Getting the type of 'solve_triangular' (line 512)
    solve_triangular_33891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'solve_triangular', False)
    # Calling solve_triangular(args, kwargs) (line 512)
    solve_triangular_call_result_33897 = invoke(stypy.reporting.localization.Localization(__file__, 512, 12), solve_triangular_33891, *[result_add_33894, rhs_33895], **kwargs_33896)
    
    # Assigning a type to the variable 'Y' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'Y', solve_triangular_call_result_33897)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 513):
    
    # Assigning a BinOp to a Name (line 513):
    # Getting the type of 'ident' (line 513)
    ident_33898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'ident')
    # Getting the type of 'Y' (line 513)
    Y_33899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 16), 'Y')
    # Applying the binary operator '+' (line 513)
    result_add_33900 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 8), '+', ident_33898, Y_33899)
    
    # Assigning a type to the variable 'U' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'U', result_add_33900)
    
    
    
    # Call to array_equal(...): (line 514)
    # Processing the call arguments (line 514)
    # Getting the type of 'U' (line 514)
    U_33903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 26), 'U', False)
    
    # Call to triu(...): (line 514)
    # Processing the call arguments (line 514)
    # Getting the type of 'U' (line 514)
    U_33906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 37), 'U', False)
    # Processing the call keyword arguments (line 514)
    kwargs_33907 = {}
    # Getting the type of 'np' (line 514)
    np_33904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 29), 'np', False)
    # Obtaining the member 'triu' of a type (line 514)
    triu_33905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 29), np_33904, 'triu')
    # Calling triu(args, kwargs) (line 514)
    triu_call_result_33908 = invoke(stypy.reporting.localization.Localization(__file__, 514, 29), triu_33905, *[U_33906], **kwargs_33907)
    
    # Processing the call keyword arguments (line 514)
    kwargs_33909 = {}
    # Getting the type of 'np' (line 514)
    np_33901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 11), 'np', False)
    # Obtaining the member 'array_equal' of a type (line 514)
    array_equal_33902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 11), np_33901, 'array_equal')
    # Calling array_equal(args, kwargs) (line 514)
    array_equal_call_result_33910 = invoke(stypy.reporting.localization.Localization(__file__, 514, 11), array_equal_33902, *[U_33903, triu_call_result_33908], **kwargs_33909)
    
    # Applying the 'not' unary operator (line 514)
    result_not__33911 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 7), 'not', array_equal_call_result_33910)
    
    # Testing the type of an if condition (line 514)
    if_condition_33912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 514, 4), result_not__33911)
    # Assigning a type to the variable 'if_condition_33912' (line 514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'if_condition_33912', if_condition_33912)
    # SSA begins for if statement (line 514)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 515)
    # Processing the call arguments (line 515)
    str_33914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 24), 'str', 'internal inconsistency')
    # Processing the call keyword arguments (line 515)
    kwargs_33915 = {}
    # Getting the type of 'Exception' (line 515)
    Exception_33913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 515)
    Exception_call_result_33916 = invoke(stypy.reporting.localization.Localization(__file__, 515, 14), Exception_33913, *[str_33914], **kwargs_33915)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 515, 8), Exception_call_result_33916, 'raise parameter', BaseException)
    # SSA join for if statement (line 514)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'U' (line 516)
    U_33917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 11), 'U')
    # Assigning a type to the variable 'stypy_return_type' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'stypy_return_type', U_33917)
    
    # ################# End of '_fractional_power_pade(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fractional_power_pade' in the type store
    # Getting the type of 'stypy_return_type' (line 469)
    stypy_return_type_33918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33918)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fractional_power_pade'
    return stypy_return_type_33918

# Assigning a type to the variable '_fractional_power_pade' (line 469)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 0), '_fractional_power_pade', _fractional_power_pade)

@norecursion
def _remainder_matrix_power_triu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_remainder_matrix_power_triu'
    module_type_store = module_type_store.open_function_context('_remainder_matrix_power_triu', 519, 0, False)
    
    # Passed parameters checking function
    _remainder_matrix_power_triu.stypy_localization = localization
    _remainder_matrix_power_triu.stypy_type_of_self = None
    _remainder_matrix_power_triu.stypy_type_store = module_type_store
    _remainder_matrix_power_triu.stypy_function_name = '_remainder_matrix_power_triu'
    _remainder_matrix_power_triu.stypy_param_names_list = ['T', 't']
    _remainder_matrix_power_triu.stypy_varargs_param_name = None
    _remainder_matrix_power_triu.stypy_kwargs_param_name = None
    _remainder_matrix_power_triu.stypy_call_defaults = defaults
    _remainder_matrix_power_triu.stypy_call_varargs = varargs
    _remainder_matrix_power_triu.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_remainder_matrix_power_triu', ['T', 't'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_remainder_matrix_power_triu', localization, ['T', 't'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_remainder_matrix_power_triu(...)' code ##################

    str_33919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, (-1)), 'str', '\n    Compute a fractional power of an upper triangular matrix.\n\n    The fractional power is restricted to fractions -1 < t < 1.\n    This uses algorithm (3.1) of [1]_.\n    The Pade approximation itself uses algorithm (4.1) of [2]_.\n\n    Parameters\n    ----------\n    T : (N, N) array_like\n        Upper triangular matrix whose fractional power to evaluate.\n    t : float\n        Fractional power between -1 and 1 exclusive.\n\n    Returns\n    -------\n    X : (N, N) array_like\n        The fractional power of the matrix.\n\n    References\n    ----------\n    .. [1] Nicholas J. Higham and Lijing Lin (2013)\n           "An Improved Schur-Pade Algorithm for Fractional Powers\n           of a Matrix and their Frechet Derivatives."\n\n    .. [2] Nicholas J. Higham and Lijing lin (2011)\n           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."\n           SIAM Journal on Matrix Analysis and Applications,\n           32 (3). pp. 1056-1078. ISSN 0895-4798\n\n    ')
    
    # Assigning a Dict to a Name (line 551):
    
    # Assigning a Dict to a Name (line 551):
    
    # Obtaining an instance of the builtin type 'dict' (line 551)
    dict_33920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 551)
    # Adding element type (key, value) (line 551)
    int_33921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 12), 'int')
    float_33922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 15), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 17), dict_33920, (int_33921, float_33922))
    # Adding element type (key, value) (line 551)
    int_33923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 12), 'int')
    float_33924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 15), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 17), dict_33920, (int_33923, float_33924))
    # Adding element type (key, value) (line 551)
    int_33925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 12), 'int')
    float_33926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 15), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 17), dict_33920, (int_33925, float_33926))
    # Adding element type (key, value) (line 551)
    int_33927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 12), 'int')
    float_33928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 15), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 17), dict_33920, (int_33927, float_33928))
    # Adding element type (key, value) (line 551)
    int_33929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 12), 'int')
    float_33930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 15), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 17), dict_33920, (int_33929, float_33930))
    # Adding element type (key, value) (line 551)
    int_33931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 12), 'int')
    float_33932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 15), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 17), dict_33920, (int_33931, float_33932))
    # Adding element type (key, value) (line 551)
    int_33933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 12), 'int')
    float_33934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 15), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 17), dict_33920, (int_33933, float_33934))
    
    # Assigning a type to the variable 'm_to_theta' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'm_to_theta', dict_33920)
    
    # Assigning a Attribute to a Tuple (line 560):
    
    # Assigning a Subscript to a Name (line 560):
    
    # Obtaining the type of the subscript
    int_33935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 4), 'int')
    # Getting the type of 'T' (line 560)
    T_33936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 11), 'T')
    # Obtaining the member 'shape' of a type (line 560)
    shape_33937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 11), T_33936, 'shape')
    # Obtaining the member '__getitem__' of a type (line 560)
    getitem___33938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 4), shape_33937, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 560)
    subscript_call_result_33939 = invoke(stypy.reporting.localization.Localization(__file__, 560, 4), getitem___33938, int_33935)
    
    # Assigning a type to the variable 'tuple_var_assignment_32842' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'tuple_var_assignment_32842', subscript_call_result_33939)
    
    # Assigning a Subscript to a Name (line 560):
    
    # Obtaining the type of the subscript
    int_33940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 4), 'int')
    # Getting the type of 'T' (line 560)
    T_33941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 11), 'T')
    # Obtaining the member 'shape' of a type (line 560)
    shape_33942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 11), T_33941, 'shape')
    # Obtaining the member '__getitem__' of a type (line 560)
    getitem___33943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 4), shape_33942, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 560)
    subscript_call_result_33944 = invoke(stypy.reporting.localization.Localization(__file__, 560, 4), getitem___33943, int_33940)
    
    # Assigning a type to the variable 'tuple_var_assignment_32843' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'tuple_var_assignment_32843', subscript_call_result_33944)
    
    # Assigning a Name to a Name (line 560):
    # Getting the type of 'tuple_var_assignment_32842' (line 560)
    tuple_var_assignment_32842_33945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'tuple_var_assignment_32842')
    # Assigning a type to the variable 'n' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'n', tuple_var_assignment_32842_33945)
    
    # Assigning a Name to a Name (line 560):
    # Getting the type of 'tuple_var_assignment_32843' (line 560)
    tuple_var_assignment_32843_33946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'tuple_var_assignment_32843')
    # Assigning a type to the variable 'n' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 7), 'n', tuple_var_assignment_32843_33946)
    
    # Assigning a Name to a Name (line 561):
    
    # Assigning a Name to a Name (line 561):
    # Getting the type of 'T' (line 561)
    T_33947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 9), 'T')
    # Assigning a type to the variable 'T0' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'T0', T_33947)
    
    # Assigning a Call to a Name (line 562):
    
    # Assigning a Call to a Name (line 562):
    
    # Call to diag(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'T0' (line 562)
    T0_33950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 22), 'T0', False)
    # Processing the call keyword arguments (line 562)
    kwargs_33951 = {}
    # Getting the type of 'np' (line 562)
    np_33948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 14), 'np', False)
    # Obtaining the member 'diag' of a type (line 562)
    diag_33949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 14), np_33948, 'diag')
    # Calling diag(args, kwargs) (line 562)
    diag_call_result_33952 = invoke(stypy.reporting.localization.Localization(__file__, 562, 14), diag_33949, *[T0_33950], **kwargs_33951)
    
    # Assigning a type to the variable 'T0_diag' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'T0_diag', diag_call_result_33952)
    
    
    # Call to array_equal(...): (line 563)
    # Processing the call arguments (line 563)
    # Getting the type of 'T0' (line 563)
    T0_33955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 22), 'T0', False)
    
    # Call to diag(...): (line 563)
    # Processing the call arguments (line 563)
    # Getting the type of 'T0_diag' (line 563)
    T0_diag_33958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 34), 'T0_diag', False)
    # Processing the call keyword arguments (line 563)
    kwargs_33959 = {}
    # Getting the type of 'np' (line 563)
    np_33956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 26), 'np', False)
    # Obtaining the member 'diag' of a type (line 563)
    diag_33957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 26), np_33956, 'diag')
    # Calling diag(args, kwargs) (line 563)
    diag_call_result_33960 = invoke(stypy.reporting.localization.Localization(__file__, 563, 26), diag_33957, *[T0_diag_33958], **kwargs_33959)
    
    # Processing the call keyword arguments (line 563)
    kwargs_33961 = {}
    # Getting the type of 'np' (line 563)
    np_33953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 7), 'np', False)
    # Obtaining the member 'array_equal' of a type (line 563)
    array_equal_33954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 7), np_33953, 'array_equal')
    # Calling array_equal(args, kwargs) (line 563)
    array_equal_call_result_33962 = invoke(stypy.reporting.localization.Localization(__file__, 563, 7), array_equal_33954, *[T0_33955, diag_call_result_33960], **kwargs_33961)
    
    # Testing the type of an if condition (line 563)
    if_condition_33963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 563, 4), array_equal_call_result_33962)
    # Assigning a type to the variable 'if_condition_33963' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'if_condition_33963', if_condition_33963)
    # SSA begins for if statement (line 563)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 564):
    
    # Assigning a Call to a Name (line 564):
    
    # Call to diag(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'T0_diag' (line 564)
    T0_diag_33966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 20), 'T0_diag', False)
    # Getting the type of 't' (line 564)
    t_33967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 31), 't', False)
    # Applying the binary operator '**' (line 564)
    result_pow_33968 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 20), '**', T0_diag_33966, t_33967)
    
    # Processing the call keyword arguments (line 564)
    kwargs_33969 = {}
    # Getting the type of 'np' (line 564)
    np_33964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 12), 'np', False)
    # Obtaining the member 'diag' of a type (line 564)
    diag_33965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 12), np_33964, 'diag')
    # Calling diag(args, kwargs) (line 564)
    diag_call_result_33970 = invoke(stypy.reporting.localization.Localization(__file__, 564, 12), diag_33965, *[result_pow_33968], **kwargs_33969)
    
    # Assigning a type to the variable 'U' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'U', diag_call_result_33970)
    # SSA branch for the else part of an if statement (line 563)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 566):
    
    # Assigning a Subscript to a Name (line 566):
    
    # Obtaining the type of the subscript
    int_33971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 8), 'int')
    
    # Call to _inverse_squaring_helper(...): (line 566)
    # Processing the call arguments (line 566)
    # Getting the type of 'T0' (line 566)
    T0_33973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 43), 'T0', False)
    # Getting the type of 'm_to_theta' (line 566)
    m_to_theta_33974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 47), 'm_to_theta', False)
    # Processing the call keyword arguments (line 566)
    kwargs_33975 = {}
    # Getting the type of '_inverse_squaring_helper' (line 566)
    _inverse_squaring_helper_33972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 18), '_inverse_squaring_helper', False)
    # Calling _inverse_squaring_helper(args, kwargs) (line 566)
    _inverse_squaring_helper_call_result_33976 = invoke(stypy.reporting.localization.Localization(__file__, 566, 18), _inverse_squaring_helper_33972, *[T0_33973, m_to_theta_33974], **kwargs_33975)
    
    # Obtaining the member '__getitem__' of a type (line 566)
    getitem___33977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 8), _inverse_squaring_helper_call_result_33976, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 566)
    subscript_call_result_33978 = invoke(stypy.reporting.localization.Localization(__file__, 566, 8), getitem___33977, int_33971)
    
    # Assigning a type to the variable 'tuple_var_assignment_32844' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'tuple_var_assignment_32844', subscript_call_result_33978)
    
    # Assigning a Subscript to a Name (line 566):
    
    # Obtaining the type of the subscript
    int_33979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 8), 'int')
    
    # Call to _inverse_squaring_helper(...): (line 566)
    # Processing the call arguments (line 566)
    # Getting the type of 'T0' (line 566)
    T0_33981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 43), 'T0', False)
    # Getting the type of 'm_to_theta' (line 566)
    m_to_theta_33982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 47), 'm_to_theta', False)
    # Processing the call keyword arguments (line 566)
    kwargs_33983 = {}
    # Getting the type of '_inverse_squaring_helper' (line 566)
    _inverse_squaring_helper_33980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 18), '_inverse_squaring_helper', False)
    # Calling _inverse_squaring_helper(args, kwargs) (line 566)
    _inverse_squaring_helper_call_result_33984 = invoke(stypy.reporting.localization.Localization(__file__, 566, 18), _inverse_squaring_helper_33980, *[T0_33981, m_to_theta_33982], **kwargs_33983)
    
    # Obtaining the member '__getitem__' of a type (line 566)
    getitem___33985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 8), _inverse_squaring_helper_call_result_33984, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 566)
    subscript_call_result_33986 = invoke(stypy.reporting.localization.Localization(__file__, 566, 8), getitem___33985, int_33979)
    
    # Assigning a type to the variable 'tuple_var_assignment_32845' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'tuple_var_assignment_32845', subscript_call_result_33986)
    
    # Assigning a Subscript to a Name (line 566):
    
    # Obtaining the type of the subscript
    int_33987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 8), 'int')
    
    # Call to _inverse_squaring_helper(...): (line 566)
    # Processing the call arguments (line 566)
    # Getting the type of 'T0' (line 566)
    T0_33989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 43), 'T0', False)
    # Getting the type of 'm_to_theta' (line 566)
    m_to_theta_33990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 47), 'm_to_theta', False)
    # Processing the call keyword arguments (line 566)
    kwargs_33991 = {}
    # Getting the type of '_inverse_squaring_helper' (line 566)
    _inverse_squaring_helper_33988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 18), '_inverse_squaring_helper', False)
    # Calling _inverse_squaring_helper(args, kwargs) (line 566)
    _inverse_squaring_helper_call_result_33992 = invoke(stypy.reporting.localization.Localization(__file__, 566, 18), _inverse_squaring_helper_33988, *[T0_33989, m_to_theta_33990], **kwargs_33991)
    
    # Obtaining the member '__getitem__' of a type (line 566)
    getitem___33993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 8), _inverse_squaring_helper_call_result_33992, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 566)
    subscript_call_result_33994 = invoke(stypy.reporting.localization.Localization(__file__, 566, 8), getitem___33993, int_33987)
    
    # Assigning a type to the variable 'tuple_var_assignment_32846' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'tuple_var_assignment_32846', subscript_call_result_33994)
    
    # Assigning a Name to a Name (line 566):
    # Getting the type of 'tuple_var_assignment_32844' (line 566)
    tuple_var_assignment_32844_33995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'tuple_var_assignment_32844')
    # Assigning a type to the variable 'R' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'R', tuple_var_assignment_32844_33995)
    
    # Assigning a Name to a Name (line 566):
    # Getting the type of 'tuple_var_assignment_32845' (line 566)
    tuple_var_assignment_32845_33996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'tuple_var_assignment_32845')
    # Assigning a type to the variable 's' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 11), 's', tuple_var_assignment_32845_33996)
    
    # Assigning a Name to a Name (line 566):
    # Getting the type of 'tuple_var_assignment_32846' (line 566)
    tuple_var_assignment_32846_33997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'tuple_var_assignment_32846')
    # Assigning a type to the variable 'm' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 14), 'm', tuple_var_assignment_32846_33997)
    
    # Assigning a Call to a Name (line 571):
    
    # Assigning a Call to a Name (line 571):
    
    # Call to _fractional_power_pade(...): (line 571)
    # Processing the call arguments (line 571)
    
    # Getting the type of 'R' (line 571)
    R_33999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 36), 'R', False)
    # Applying the 'usub' unary operator (line 571)
    result___neg___34000 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 35), 'usub', R_33999)
    
    # Getting the type of 't' (line 571)
    t_34001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 39), 't', False)
    # Getting the type of 'm' (line 571)
    m_34002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 42), 'm', False)
    # Processing the call keyword arguments (line 571)
    kwargs_34003 = {}
    # Getting the type of '_fractional_power_pade' (line 571)
    _fractional_power_pade_33998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 12), '_fractional_power_pade', False)
    # Calling _fractional_power_pade(args, kwargs) (line 571)
    _fractional_power_pade_call_result_34004 = invoke(stypy.reporting.localization.Localization(__file__, 571, 12), _fractional_power_pade_33998, *[result___neg___34000, t_34001, m_34002], **kwargs_34003)
    
    # Assigning a type to the variable 'U' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'U', _fractional_power_pade_call_result_34004)
    
    # Assigning a Call to a Name (line 578):
    
    # Assigning a Call to a Name (line 578):
    
    # Call to diag(...): (line 578)
    # Processing the call arguments (line 578)
    # Getting the type of 'T0' (line 578)
    T0_34007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 25), 'T0', False)
    # Processing the call keyword arguments (line 578)
    kwargs_34008 = {}
    # Getting the type of 'np' (line 578)
    np_34005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 17), 'np', False)
    # Obtaining the member 'diag' of a type (line 578)
    diag_34006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 17), np_34005, 'diag')
    # Calling diag(args, kwargs) (line 578)
    diag_call_result_34009 = invoke(stypy.reporting.localization.Localization(__file__, 578, 17), diag_34006, *[T0_34007], **kwargs_34008)
    
    # Assigning a type to the variable 'eivals' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'eivals', diag_call_result_34009)
    
    # Assigning a Call to a Name (line 579):
    
    # Assigning a Call to a Name (line 579):
    
    # Call to all(...): (line 579)
    # Processing the call arguments (line 579)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 579, 35, True)
    # Calculating comprehension expression
    # Getting the type of 'eivals' (line 579)
    eivals_34020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 70), 'eivals', False)
    comprehension_34021 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 35), eivals_34020)
    # Assigning a type to the variable 'x' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 35), 'x', comprehension_34021)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'x' (line 579)
    x_34011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 35), 'x', False)
    # Obtaining the member 'real' of a type (line 579)
    real_34012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 35), x_34011, 'real')
    int_34013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 44), 'int')
    # Applying the binary operator '>' (line 579)
    result_gt_34014 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 35), '>', real_34012, int_34013)
    
    
    # Getting the type of 'x' (line 579)
    x_34015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 49), 'x', False)
    # Obtaining the member 'imag' of a type (line 579)
    imag_34016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 49), x_34015, 'imag')
    int_34017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 59), 'int')
    # Applying the binary operator '!=' (line 579)
    result_ne_34018 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 49), '!=', imag_34016, int_34017)
    
    # Applying the binary operator 'or' (line 579)
    result_or_keyword_34019 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 35), 'or', result_gt_34014, result_ne_34018)
    
    list_34022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 35), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 35), list_34022, result_or_keyword_34019)
    # Processing the call keyword arguments (line 579)
    kwargs_34023 = {}
    # Getting the type of 'all' (line 579)
    all_34010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 31), 'all', False)
    # Calling all(args, kwargs) (line 579)
    all_call_result_34024 = invoke(stypy.reporting.localization.Localization(__file__, 579, 31), all_34010, *[list_34022], **kwargs_34023)
    
    # Assigning a type to the variable 'has_principal_branch' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'has_principal_branch', all_call_result_34024)
    
    
    # Call to range(...): (line 580)
    # Processing the call arguments (line 580)
    # Getting the type of 's' (line 580)
    s_34026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 23), 's', False)
    int_34027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 26), 'int')
    int_34028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 30), 'int')
    # Processing the call keyword arguments (line 580)
    kwargs_34029 = {}
    # Getting the type of 'range' (line 580)
    range_34025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 17), 'range', False)
    # Calling range(args, kwargs) (line 580)
    range_call_result_34030 = invoke(stypy.reporting.localization.Localization(__file__, 580, 17), range_34025, *[s_34026, int_34027, int_34028], **kwargs_34029)
    
    # Testing the type of a for loop iterable (line 580)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 580, 8), range_call_result_34030)
    # Getting the type of the for loop variable (line 580)
    for_loop_var_34031 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 580, 8), range_call_result_34030)
    # Assigning a type to the variable 'i' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'i', for_loop_var_34031)
    # SSA begins for a for statement (line 580)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'i' (line 581)
    i_34032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 15), 'i')
    # Getting the type of 's' (line 581)
    s_34033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 19), 's')
    # Applying the binary operator '<' (line 581)
    result_lt_34034 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 15), '<', i_34032, s_34033)
    
    # Testing the type of an if condition (line 581)
    if_condition_34035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 581, 12), result_lt_34034)
    # Assigning a type to the variable 'if_condition_34035' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 12), 'if_condition_34035', if_condition_34035)
    # SSA begins for if statement (line 581)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 582):
    
    # Assigning a Call to a Name (line 582):
    
    # Call to dot(...): (line 582)
    # Processing the call arguments (line 582)
    # Getting the type of 'U' (line 582)
    U_34038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 26), 'U', False)
    # Processing the call keyword arguments (line 582)
    kwargs_34039 = {}
    # Getting the type of 'U' (line 582)
    U_34036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 20), 'U', False)
    # Obtaining the member 'dot' of a type (line 582)
    dot_34037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 20), U_34036, 'dot')
    # Calling dot(args, kwargs) (line 582)
    dot_call_result_34040 = invoke(stypy.reporting.localization.Localization(__file__, 582, 20), dot_34037, *[U_34038], **kwargs_34039)
    
    # Assigning a type to the variable 'U' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'U', dot_call_result_34040)
    # SSA branch for the else part of an if statement (line 581)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'has_principal_branch' (line 584)
    has_principal_branch_34041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 19), 'has_principal_branch')
    # Testing the type of an if condition (line 584)
    if_condition_34042 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 584, 16), has_principal_branch_34041)
    # Assigning a type to the variable 'if_condition_34042' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'if_condition_34042', if_condition_34042)
    # SSA begins for if statement (line 584)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 585):
    
    # Assigning a BinOp to a Name (line 585):
    # Getting the type of 't' (line 585)
    t_34043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 24), 't')
    
    # Call to exp2(...): (line 585)
    # Processing the call arguments (line 585)
    
    # Getting the type of 'i' (line 585)
    i_34046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 37), 'i', False)
    # Applying the 'usub' unary operator (line 585)
    result___neg___34047 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 36), 'usub', i_34046)
    
    # Processing the call keyword arguments (line 585)
    kwargs_34048 = {}
    # Getting the type of 'np' (line 585)
    np_34044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 28), 'np', False)
    # Obtaining the member 'exp2' of a type (line 585)
    exp2_34045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 28), np_34044, 'exp2')
    # Calling exp2(args, kwargs) (line 585)
    exp2_call_result_34049 = invoke(stypy.reporting.localization.Localization(__file__, 585, 28), exp2_34045, *[result___neg___34047], **kwargs_34048)
    
    # Applying the binary operator '*' (line 585)
    result_mul_34050 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 24), '*', t_34043, exp2_call_result_34049)
    
    # Assigning a type to the variable 'p' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 20), 'p', result_mul_34050)
    
    # Assigning a BinOp to a Subscript (line 586):
    
    # Assigning a BinOp to a Subscript (line 586):
    # Getting the type of 'T0_diag' (line 586)
    T0_diag_34051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 44), 'T0_diag')
    # Getting the type of 'p' (line 586)
    p_34052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 55), 'p')
    # Applying the binary operator '**' (line 586)
    result_pow_34053 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 44), '**', T0_diag_34051, p_34052)
    
    # Getting the type of 'U' (line 586)
    U_34054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'U')
    
    # Call to diag_indices(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'n' (line 586)
    n_34057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 38), 'n', False)
    # Processing the call keyword arguments (line 586)
    kwargs_34058 = {}
    # Getting the type of 'np' (line 586)
    np_34055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 22), 'np', False)
    # Obtaining the member 'diag_indices' of a type (line 586)
    diag_indices_34056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 22), np_34055, 'diag_indices')
    # Calling diag_indices(args, kwargs) (line 586)
    diag_indices_call_result_34059 = invoke(stypy.reporting.localization.Localization(__file__, 586, 22), diag_indices_34056, *[n_34057], **kwargs_34058)
    
    # Storing an element on a container (line 586)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 20), U_34054, (diag_indices_call_result_34059, result_pow_34053))
    
    
    # Call to range(...): (line 587)
    # Processing the call arguments (line 587)
    # Getting the type of 'n' (line 587)
    n_34061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 35), 'n', False)
    int_34062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 37), 'int')
    # Applying the binary operator '-' (line 587)
    result_sub_34063 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 35), '-', n_34061, int_34062)
    
    # Processing the call keyword arguments (line 587)
    kwargs_34064 = {}
    # Getting the type of 'range' (line 587)
    range_34060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 29), 'range', False)
    # Calling range(args, kwargs) (line 587)
    range_call_result_34065 = invoke(stypy.reporting.localization.Localization(__file__, 587, 29), range_34060, *[result_sub_34063], **kwargs_34064)
    
    # Testing the type of a for loop iterable (line 587)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 587, 20), range_call_result_34065)
    # Getting the type of the for loop variable (line 587)
    for_loop_var_34066 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 587, 20), range_call_result_34065)
    # Assigning a type to the variable 'j' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 20), 'j', for_loop_var_34066)
    # SSA begins for a for statement (line 587)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 588):
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 588)
    tuple_34067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 588)
    # Adding element type (line 588)
    # Getting the type of 'j' (line 588)
    j_34068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 32), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 32), tuple_34067, j_34068)
    # Adding element type (line 588)
    # Getting the type of 'j' (line 588)
    j_34069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 35), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 32), tuple_34067, j_34069)
    
    # Getting the type of 'T0' (line 588)
    T0_34070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 29), 'T0')
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___34071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 29), T0_34070, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_34072 = invoke(stypy.reporting.localization.Localization(__file__, 588, 29), getitem___34071, tuple_34067)
    
    # Assigning a type to the variable 'l1' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 24), 'l1', subscript_call_result_34072)
    
    # Assigning a Subscript to a Name (line 589):
    
    # Assigning a Subscript to a Name (line 589):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 589)
    tuple_34073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 589)
    # Adding element type (line 589)
    # Getting the type of 'j' (line 589)
    j_34074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 32), 'j')
    int_34075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 34), 'int')
    # Applying the binary operator '+' (line 589)
    result_add_34076 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 32), '+', j_34074, int_34075)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 32), tuple_34073, result_add_34076)
    # Adding element type (line 589)
    # Getting the type of 'j' (line 589)
    j_34077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 37), 'j')
    int_34078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 39), 'int')
    # Applying the binary operator '+' (line 589)
    result_add_34079 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 37), '+', j_34077, int_34078)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 32), tuple_34073, result_add_34079)
    
    # Getting the type of 'T0' (line 589)
    T0_34080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 29), 'T0')
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___34081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 29), T0_34080, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 589)
    subscript_call_result_34082 = invoke(stypy.reporting.localization.Localization(__file__, 589, 29), getitem___34081, tuple_34073)
    
    # Assigning a type to the variable 'l2' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 24), 'l2', subscript_call_result_34082)
    
    # Assigning a Subscript to a Name (line 590):
    
    # Assigning a Subscript to a Name (line 590):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 590)
    tuple_34083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 590)
    # Adding element type (line 590)
    # Getting the type of 'j' (line 590)
    j_34084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 33), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 33), tuple_34083, j_34084)
    # Adding element type (line 590)
    # Getting the type of 'j' (line 590)
    j_34085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 36), 'j')
    int_34086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 38), 'int')
    # Applying the binary operator '+' (line 590)
    result_add_34087 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 36), '+', j_34085, int_34086)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 33), tuple_34083, result_add_34087)
    
    # Getting the type of 'T0' (line 590)
    T0_34088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 30), 'T0')
    # Obtaining the member '__getitem__' of a type (line 590)
    getitem___34089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 30), T0_34088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 590)
    subscript_call_result_34090 = invoke(stypy.reporting.localization.Localization(__file__, 590, 30), getitem___34089, tuple_34083)
    
    # Assigning a type to the variable 't12' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 24), 't12', subscript_call_result_34090)
    
    # Assigning a Call to a Name (line 591):
    
    # Assigning a Call to a Name (line 591):
    
    # Call to _fractional_power_superdiag_entry(...): (line 591)
    # Processing the call arguments (line 591)
    # Getting the type of 'l1' (line 591)
    l1_34092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 64), 'l1', False)
    # Getting the type of 'l2' (line 591)
    l2_34093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 68), 'l2', False)
    # Getting the type of 't12' (line 591)
    t12_34094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 72), 't12', False)
    # Getting the type of 'p' (line 591)
    p_34095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 77), 'p', False)
    # Processing the call keyword arguments (line 591)
    kwargs_34096 = {}
    # Getting the type of '_fractional_power_superdiag_entry' (line 591)
    _fractional_power_superdiag_entry_34091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 30), '_fractional_power_superdiag_entry', False)
    # Calling _fractional_power_superdiag_entry(args, kwargs) (line 591)
    _fractional_power_superdiag_entry_call_result_34097 = invoke(stypy.reporting.localization.Localization(__file__, 591, 30), _fractional_power_superdiag_entry_34091, *[l1_34092, l2_34093, t12_34094, p_34095], **kwargs_34096)
    
    # Assigning a type to the variable 'f12' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 24), 'f12', _fractional_power_superdiag_entry_call_result_34097)
    
    # Assigning a Name to a Subscript (line 592):
    
    # Assigning a Name to a Subscript (line 592):
    # Getting the type of 'f12' (line 592)
    f12_34098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 36), 'f12')
    # Getting the type of 'U' (line 592)
    U_34099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 24), 'U')
    
    # Obtaining an instance of the builtin type 'tuple' (line 592)
    tuple_34100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 592)
    # Adding element type (line 592)
    # Getting the type of 'j' (line 592)
    j_34101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 26), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 592, 26), tuple_34100, j_34101)
    # Adding element type (line 592)
    # Getting the type of 'j' (line 592)
    j_34102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 29), 'j')
    int_34103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 31), 'int')
    # Applying the binary operator '+' (line 592)
    result_add_34104 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 29), '+', j_34102, int_34103)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 592, 26), tuple_34100, result_add_34104)
    
    # Storing an element on a container (line 592)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 592, 24), U_34099, (tuple_34100, f12_34098))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 584)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 581)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 563)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to array_equal(...): (line 593)
    # Processing the call arguments (line 593)
    # Getting the type of 'U' (line 593)
    U_34107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 26), 'U', False)
    
    # Call to triu(...): (line 593)
    # Processing the call arguments (line 593)
    # Getting the type of 'U' (line 593)
    U_34110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 37), 'U', False)
    # Processing the call keyword arguments (line 593)
    kwargs_34111 = {}
    # Getting the type of 'np' (line 593)
    np_34108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 29), 'np', False)
    # Obtaining the member 'triu' of a type (line 593)
    triu_34109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 29), np_34108, 'triu')
    # Calling triu(args, kwargs) (line 593)
    triu_call_result_34112 = invoke(stypy.reporting.localization.Localization(__file__, 593, 29), triu_34109, *[U_34110], **kwargs_34111)
    
    # Processing the call keyword arguments (line 593)
    kwargs_34113 = {}
    # Getting the type of 'np' (line 593)
    np_34105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 11), 'np', False)
    # Obtaining the member 'array_equal' of a type (line 593)
    array_equal_34106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 11), np_34105, 'array_equal')
    # Calling array_equal(args, kwargs) (line 593)
    array_equal_call_result_34114 = invoke(stypy.reporting.localization.Localization(__file__, 593, 11), array_equal_34106, *[U_34107, triu_call_result_34112], **kwargs_34113)
    
    # Applying the 'not' unary operator (line 593)
    result_not__34115 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 7), 'not', array_equal_call_result_34114)
    
    # Testing the type of an if condition (line 593)
    if_condition_34116 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 593, 4), result_not__34115)
    # Assigning a type to the variable 'if_condition_34116' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'if_condition_34116', if_condition_34116)
    # SSA begins for if statement (line 593)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 594)
    # Processing the call arguments (line 594)
    str_34118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 24), 'str', 'internal inconsistency')
    # Processing the call keyword arguments (line 594)
    kwargs_34119 = {}
    # Getting the type of 'Exception' (line 594)
    Exception_34117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 594)
    Exception_call_result_34120 = invoke(stypy.reporting.localization.Localization(__file__, 594, 14), Exception_34117, *[str_34118], **kwargs_34119)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 594, 8), Exception_call_result_34120, 'raise parameter', BaseException)
    # SSA join for if statement (line 593)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'U' (line 595)
    U_34121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 11), 'U')
    # Assigning a type to the variable 'stypy_return_type' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'stypy_return_type', U_34121)
    
    # ################# End of '_remainder_matrix_power_triu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_remainder_matrix_power_triu' in the type store
    # Getting the type of 'stypy_return_type' (line 519)
    stypy_return_type_34122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34122)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_remainder_matrix_power_triu'
    return stypy_return_type_34122

# Assigning a type to the variable '_remainder_matrix_power_triu' (line 519)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 0), '_remainder_matrix_power_triu', _remainder_matrix_power_triu)

@norecursion
def _remainder_matrix_power(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_remainder_matrix_power'
    module_type_store = module_type_store.open_function_context('_remainder_matrix_power', 598, 0, False)
    
    # Passed parameters checking function
    _remainder_matrix_power.stypy_localization = localization
    _remainder_matrix_power.stypy_type_of_self = None
    _remainder_matrix_power.stypy_type_store = module_type_store
    _remainder_matrix_power.stypy_function_name = '_remainder_matrix_power'
    _remainder_matrix_power.stypy_param_names_list = ['A', 't']
    _remainder_matrix_power.stypy_varargs_param_name = None
    _remainder_matrix_power.stypy_kwargs_param_name = None
    _remainder_matrix_power.stypy_call_defaults = defaults
    _remainder_matrix_power.stypy_call_varargs = varargs
    _remainder_matrix_power.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_remainder_matrix_power', ['A', 't'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_remainder_matrix_power', localization, ['A', 't'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_remainder_matrix_power(...)' code ##################

    str_34123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, (-1)), 'str', '\n    Compute the fractional power of a matrix, for fractions -1 < t < 1.\n\n    This uses algorithm (3.1) of [1]_.\n    The Pade approximation itself uses algorithm (4.1) of [2]_.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Matrix whose fractional power to evaluate.\n    t : float\n        Fractional power between -1 and 1 exclusive.\n\n    Returns\n    -------\n    X : (N, N) array_like\n        The fractional power of the matrix.\n\n    References\n    ----------\n    .. [1] Nicholas J. Higham and Lijing Lin (2013)\n           "An Improved Schur-Pade Algorithm for Fractional Powers\n           of a Matrix and their Frechet Derivatives."\n\n    .. [2] Nicholas J. Higham and Lijing lin (2011)\n           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."\n           SIAM Journal on Matrix Analysis and Applications,\n           32 (3). pp. 1056-1078. ISSN 0895-4798\n\n    ')
    
    # Assigning a Call to a Name (line 630):
    
    # Assigning a Call to a Name (line 630):
    
    # Call to asarray(...): (line 630)
    # Processing the call arguments (line 630)
    # Getting the type of 'A' (line 630)
    A_34126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 19), 'A', False)
    # Processing the call keyword arguments (line 630)
    kwargs_34127 = {}
    # Getting the type of 'np' (line 630)
    np_34124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 630)
    asarray_34125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 8), np_34124, 'asarray')
    # Calling asarray(args, kwargs) (line 630)
    asarray_call_result_34128 = invoke(stypy.reporting.localization.Localization(__file__, 630, 8), asarray_34125, *[A_34126], **kwargs_34127)
    
    # Assigning a type to the variable 'A' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'A', asarray_call_result_34128)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 631)
    # Processing the call arguments (line 631)
    # Getting the type of 'A' (line 631)
    A_34130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 631)
    shape_34131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 11), A_34130, 'shape')
    # Processing the call keyword arguments (line 631)
    kwargs_34132 = {}
    # Getting the type of 'len' (line 631)
    len_34129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 7), 'len', False)
    # Calling len(args, kwargs) (line 631)
    len_call_result_34133 = invoke(stypy.reporting.localization.Localization(__file__, 631, 7), len_34129, *[shape_34131], **kwargs_34132)
    
    int_34134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 23), 'int')
    # Applying the binary operator '!=' (line 631)
    result_ne_34135 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 7), '!=', len_call_result_34133, int_34134)
    
    
    
    # Obtaining the type of the subscript
    int_34136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 36), 'int')
    # Getting the type of 'A' (line 631)
    A_34137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 28), 'A')
    # Obtaining the member 'shape' of a type (line 631)
    shape_34138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 28), A_34137, 'shape')
    # Obtaining the member '__getitem__' of a type (line 631)
    getitem___34139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 28), shape_34138, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 631)
    subscript_call_result_34140 = invoke(stypy.reporting.localization.Localization(__file__, 631, 28), getitem___34139, int_34136)
    
    
    # Obtaining the type of the subscript
    int_34141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 50), 'int')
    # Getting the type of 'A' (line 631)
    A_34142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 42), 'A')
    # Obtaining the member 'shape' of a type (line 631)
    shape_34143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 42), A_34142, 'shape')
    # Obtaining the member '__getitem__' of a type (line 631)
    getitem___34144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 42), shape_34143, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 631)
    subscript_call_result_34145 = invoke(stypy.reporting.localization.Localization(__file__, 631, 42), getitem___34144, int_34141)
    
    # Applying the binary operator '!=' (line 631)
    result_ne_34146 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 28), '!=', subscript_call_result_34140, subscript_call_result_34145)
    
    # Applying the binary operator 'or' (line 631)
    result_or_keyword_34147 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 7), 'or', result_ne_34135, result_ne_34146)
    
    # Testing the type of an if condition (line 631)
    if_condition_34148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 631, 4), result_or_keyword_34147)
    # Assigning a type to the variable 'if_condition_34148' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 4), 'if_condition_34148', if_condition_34148)
    # SSA begins for if statement (line 631)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 632)
    # Processing the call arguments (line 632)
    str_34150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 25), 'str', 'input must be a square array')
    # Processing the call keyword arguments (line 632)
    kwargs_34151 = {}
    # Getting the type of 'ValueError' (line 632)
    ValueError_34149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 632)
    ValueError_call_result_34152 = invoke(stypy.reporting.localization.Localization(__file__, 632, 14), ValueError_34149, *[str_34150], **kwargs_34151)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 632, 8), ValueError_call_result_34152, 'raise parameter', BaseException)
    # SSA join for if statement (line 631)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 635):
    
    # Assigning a Subscript to a Name (line 635):
    
    # Obtaining the type of the subscript
    int_34153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 4), 'int')
    # Getting the type of 'A' (line 635)
    A_34154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 11), 'A')
    # Obtaining the member 'shape' of a type (line 635)
    shape_34155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 11), A_34154, 'shape')
    # Obtaining the member '__getitem__' of a type (line 635)
    getitem___34156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 4), shape_34155, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 635)
    subscript_call_result_34157 = invoke(stypy.reporting.localization.Localization(__file__, 635, 4), getitem___34156, int_34153)
    
    # Assigning a type to the variable 'tuple_var_assignment_32847' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_var_assignment_32847', subscript_call_result_34157)
    
    # Assigning a Subscript to a Name (line 635):
    
    # Obtaining the type of the subscript
    int_34158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 4), 'int')
    # Getting the type of 'A' (line 635)
    A_34159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 11), 'A')
    # Obtaining the member 'shape' of a type (line 635)
    shape_34160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 11), A_34159, 'shape')
    # Obtaining the member '__getitem__' of a type (line 635)
    getitem___34161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 4), shape_34160, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 635)
    subscript_call_result_34162 = invoke(stypy.reporting.localization.Localization(__file__, 635, 4), getitem___34161, int_34158)
    
    # Assigning a type to the variable 'tuple_var_assignment_32848' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_var_assignment_32848', subscript_call_result_34162)
    
    # Assigning a Name to a Name (line 635):
    # Getting the type of 'tuple_var_assignment_32847' (line 635)
    tuple_var_assignment_32847_34163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_var_assignment_32847')
    # Assigning a type to the variable 'n' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'n', tuple_var_assignment_32847_34163)
    
    # Assigning a Name to a Name (line 635):
    # Getting the type of 'tuple_var_assignment_32848' (line 635)
    tuple_var_assignment_32848_34164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_var_assignment_32848')
    # Assigning a type to the variable 'n' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 7), 'n', tuple_var_assignment_32848_34164)
    
    
    # Call to array_equal(...): (line 639)
    # Processing the call arguments (line 639)
    # Getting the type of 'A' (line 639)
    A_34167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 22), 'A', False)
    
    # Call to triu(...): (line 639)
    # Processing the call arguments (line 639)
    # Getting the type of 'A' (line 639)
    A_34170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 33), 'A', False)
    # Processing the call keyword arguments (line 639)
    kwargs_34171 = {}
    # Getting the type of 'np' (line 639)
    np_34168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 25), 'np', False)
    # Obtaining the member 'triu' of a type (line 639)
    triu_34169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 25), np_34168, 'triu')
    # Calling triu(args, kwargs) (line 639)
    triu_call_result_34172 = invoke(stypy.reporting.localization.Localization(__file__, 639, 25), triu_34169, *[A_34170], **kwargs_34171)
    
    # Processing the call keyword arguments (line 639)
    kwargs_34173 = {}
    # Getting the type of 'np' (line 639)
    np_34165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 7), 'np', False)
    # Obtaining the member 'array_equal' of a type (line 639)
    array_equal_34166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 7), np_34165, 'array_equal')
    # Calling array_equal(args, kwargs) (line 639)
    array_equal_call_result_34174 = invoke(stypy.reporting.localization.Localization(__file__, 639, 7), array_equal_34166, *[A_34167, triu_call_result_34172], **kwargs_34173)
    
    # Testing the type of an if condition (line 639)
    if_condition_34175 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 639, 4), array_equal_call_result_34174)
    # Assigning a type to the variable 'if_condition_34175' (line 639)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'if_condition_34175', if_condition_34175)
    # SSA begins for if statement (line 639)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 640):
    
    # Assigning a Name to a Name (line 640):
    # Getting the type of 'None' (line 640)
    None_34176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'None')
    # Assigning a type to the variable 'Z' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'Z', None_34176)
    
    # Assigning a Name to a Name (line 641):
    
    # Assigning a Name to a Name (line 641):
    # Getting the type of 'A' (line 641)
    A_34177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 12), 'A')
    # Assigning a type to the variable 'T' (line 641)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'T', A_34177)
    # SSA branch for the else part of an if statement (line 639)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isrealobj(...): (line 643)
    # Processing the call arguments (line 643)
    # Getting the type of 'A' (line 643)
    A_34180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 24), 'A', False)
    # Processing the call keyword arguments (line 643)
    kwargs_34181 = {}
    # Getting the type of 'np' (line 643)
    np_34178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 11), 'np', False)
    # Obtaining the member 'isrealobj' of a type (line 643)
    isrealobj_34179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 11), np_34178, 'isrealobj')
    # Calling isrealobj(args, kwargs) (line 643)
    isrealobj_call_result_34182 = invoke(stypy.reporting.localization.Localization(__file__, 643, 11), isrealobj_34179, *[A_34180], **kwargs_34181)
    
    # Testing the type of an if condition (line 643)
    if_condition_34183 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 643, 8), isrealobj_call_result_34182)
    # Assigning a type to the variable 'if_condition_34183' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'if_condition_34183', if_condition_34183)
    # SSA begins for if statement (line 643)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 644):
    
    # Assigning a Subscript to a Name (line 644):
    
    # Obtaining the type of the subscript
    int_34184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 12), 'int')
    
    # Call to schur(...): (line 644)
    # Processing the call arguments (line 644)
    # Getting the type of 'A' (line 644)
    A_34186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 25), 'A', False)
    # Processing the call keyword arguments (line 644)
    kwargs_34187 = {}
    # Getting the type of 'schur' (line 644)
    schur_34185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 19), 'schur', False)
    # Calling schur(args, kwargs) (line 644)
    schur_call_result_34188 = invoke(stypy.reporting.localization.Localization(__file__, 644, 19), schur_34185, *[A_34186], **kwargs_34187)
    
    # Obtaining the member '__getitem__' of a type (line 644)
    getitem___34189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 12), schur_call_result_34188, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 644)
    subscript_call_result_34190 = invoke(stypy.reporting.localization.Localization(__file__, 644, 12), getitem___34189, int_34184)
    
    # Assigning a type to the variable 'tuple_var_assignment_32849' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 12), 'tuple_var_assignment_32849', subscript_call_result_34190)
    
    # Assigning a Subscript to a Name (line 644):
    
    # Obtaining the type of the subscript
    int_34191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 12), 'int')
    
    # Call to schur(...): (line 644)
    # Processing the call arguments (line 644)
    # Getting the type of 'A' (line 644)
    A_34193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 25), 'A', False)
    # Processing the call keyword arguments (line 644)
    kwargs_34194 = {}
    # Getting the type of 'schur' (line 644)
    schur_34192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 19), 'schur', False)
    # Calling schur(args, kwargs) (line 644)
    schur_call_result_34195 = invoke(stypy.reporting.localization.Localization(__file__, 644, 19), schur_34192, *[A_34193], **kwargs_34194)
    
    # Obtaining the member '__getitem__' of a type (line 644)
    getitem___34196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 12), schur_call_result_34195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 644)
    subscript_call_result_34197 = invoke(stypy.reporting.localization.Localization(__file__, 644, 12), getitem___34196, int_34191)
    
    # Assigning a type to the variable 'tuple_var_assignment_32850' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 12), 'tuple_var_assignment_32850', subscript_call_result_34197)
    
    # Assigning a Name to a Name (line 644):
    # Getting the type of 'tuple_var_assignment_32849' (line 644)
    tuple_var_assignment_32849_34198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 12), 'tuple_var_assignment_32849')
    # Assigning a type to the variable 'T' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 12), 'T', tuple_var_assignment_32849_34198)
    
    # Assigning a Name to a Name (line 644):
    # Getting the type of 'tuple_var_assignment_32850' (line 644)
    tuple_var_assignment_32850_34199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 12), 'tuple_var_assignment_32850')
    # Assigning a type to the variable 'Z' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 15), 'Z', tuple_var_assignment_32850_34199)
    
    
    
    # Call to array_equal(...): (line 645)
    # Processing the call arguments (line 645)
    # Getting the type of 'T' (line 645)
    T_34202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 34), 'T', False)
    
    # Call to triu(...): (line 645)
    # Processing the call arguments (line 645)
    # Getting the type of 'T' (line 645)
    T_34205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 45), 'T', False)
    # Processing the call keyword arguments (line 645)
    kwargs_34206 = {}
    # Getting the type of 'np' (line 645)
    np_34203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 37), 'np', False)
    # Obtaining the member 'triu' of a type (line 645)
    triu_34204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 37), np_34203, 'triu')
    # Calling triu(args, kwargs) (line 645)
    triu_call_result_34207 = invoke(stypy.reporting.localization.Localization(__file__, 645, 37), triu_34204, *[T_34205], **kwargs_34206)
    
    # Processing the call keyword arguments (line 645)
    kwargs_34208 = {}
    # Getting the type of 'np' (line 645)
    np_34200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 19), 'np', False)
    # Obtaining the member 'array_equal' of a type (line 645)
    array_equal_34201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 19), np_34200, 'array_equal')
    # Calling array_equal(args, kwargs) (line 645)
    array_equal_call_result_34209 = invoke(stypy.reporting.localization.Localization(__file__, 645, 19), array_equal_34201, *[T_34202, triu_call_result_34207], **kwargs_34208)
    
    # Applying the 'not' unary operator (line 645)
    result_not__34210 = python_operator(stypy.reporting.localization.Localization(__file__, 645, 15), 'not', array_equal_call_result_34209)
    
    # Testing the type of an if condition (line 645)
    if_condition_34211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 645, 12), result_not__34210)
    # Assigning a type to the variable 'if_condition_34211' (line 645)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 12), 'if_condition_34211', if_condition_34211)
    # SSA begins for if statement (line 645)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 646):
    
    # Assigning a Subscript to a Name (line 646):
    
    # Obtaining the type of the subscript
    int_34212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 16), 'int')
    
    # Call to rsf2csf(...): (line 646)
    # Processing the call arguments (line 646)
    # Getting the type of 'T' (line 646)
    T_34214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 31), 'T', False)
    # Getting the type of 'Z' (line 646)
    Z_34215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 34), 'Z', False)
    # Processing the call keyword arguments (line 646)
    kwargs_34216 = {}
    # Getting the type of 'rsf2csf' (line 646)
    rsf2csf_34213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 23), 'rsf2csf', False)
    # Calling rsf2csf(args, kwargs) (line 646)
    rsf2csf_call_result_34217 = invoke(stypy.reporting.localization.Localization(__file__, 646, 23), rsf2csf_34213, *[T_34214, Z_34215], **kwargs_34216)
    
    # Obtaining the member '__getitem__' of a type (line 646)
    getitem___34218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 16), rsf2csf_call_result_34217, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 646)
    subscript_call_result_34219 = invoke(stypy.reporting.localization.Localization(__file__, 646, 16), getitem___34218, int_34212)
    
    # Assigning a type to the variable 'tuple_var_assignment_32851' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 16), 'tuple_var_assignment_32851', subscript_call_result_34219)
    
    # Assigning a Subscript to a Name (line 646):
    
    # Obtaining the type of the subscript
    int_34220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 16), 'int')
    
    # Call to rsf2csf(...): (line 646)
    # Processing the call arguments (line 646)
    # Getting the type of 'T' (line 646)
    T_34222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 31), 'T', False)
    # Getting the type of 'Z' (line 646)
    Z_34223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 34), 'Z', False)
    # Processing the call keyword arguments (line 646)
    kwargs_34224 = {}
    # Getting the type of 'rsf2csf' (line 646)
    rsf2csf_34221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 23), 'rsf2csf', False)
    # Calling rsf2csf(args, kwargs) (line 646)
    rsf2csf_call_result_34225 = invoke(stypy.reporting.localization.Localization(__file__, 646, 23), rsf2csf_34221, *[T_34222, Z_34223], **kwargs_34224)
    
    # Obtaining the member '__getitem__' of a type (line 646)
    getitem___34226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 16), rsf2csf_call_result_34225, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 646)
    subscript_call_result_34227 = invoke(stypy.reporting.localization.Localization(__file__, 646, 16), getitem___34226, int_34220)
    
    # Assigning a type to the variable 'tuple_var_assignment_32852' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 16), 'tuple_var_assignment_32852', subscript_call_result_34227)
    
    # Assigning a Name to a Name (line 646):
    # Getting the type of 'tuple_var_assignment_32851' (line 646)
    tuple_var_assignment_32851_34228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 16), 'tuple_var_assignment_32851')
    # Assigning a type to the variable 'T' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 16), 'T', tuple_var_assignment_32851_34228)
    
    # Assigning a Name to a Name (line 646):
    # Getting the type of 'tuple_var_assignment_32852' (line 646)
    tuple_var_assignment_32852_34229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 16), 'tuple_var_assignment_32852')
    # Assigning a type to the variable 'Z' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 19), 'Z', tuple_var_assignment_32852_34229)
    # SSA join for if statement (line 645)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 643)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 648):
    
    # Assigning a Subscript to a Name (line 648):
    
    # Obtaining the type of the subscript
    int_34230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 12), 'int')
    
    # Call to schur(...): (line 648)
    # Processing the call arguments (line 648)
    # Getting the type of 'A' (line 648)
    A_34232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 25), 'A', False)
    # Processing the call keyword arguments (line 648)
    str_34233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 35), 'str', 'complex')
    keyword_34234 = str_34233
    kwargs_34235 = {'output': keyword_34234}
    # Getting the type of 'schur' (line 648)
    schur_34231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 19), 'schur', False)
    # Calling schur(args, kwargs) (line 648)
    schur_call_result_34236 = invoke(stypy.reporting.localization.Localization(__file__, 648, 19), schur_34231, *[A_34232], **kwargs_34235)
    
    # Obtaining the member '__getitem__' of a type (line 648)
    getitem___34237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 12), schur_call_result_34236, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 648)
    subscript_call_result_34238 = invoke(stypy.reporting.localization.Localization(__file__, 648, 12), getitem___34237, int_34230)
    
    # Assigning a type to the variable 'tuple_var_assignment_32853' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 12), 'tuple_var_assignment_32853', subscript_call_result_34238)
    
    # Assigning a Subscript to a Name (line 648):
    
    # Obtaining the type of the subscript
    int_34239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 12), 'int')
    
    # Call to schur(...): (line 648)
    # Processing the call arguments (line 648)
    # Getting the type of 'A' (line 648)
    A_34241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 25), 'A', False)
    # Processing the call keyword arguments (line 648)
    str_34242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 35), 'str', 'complex')
    keyword_34243 = str_34242
    kwargs_34244 = {'output': keyword_34243}
    # Getting the type of 'schur' (line 648)
    schur_34240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 19), 'schur', False)
    # Calling schur(args, kwargs) (line 648)
    schur_call_result_34245 = invoke(stypy.reporting.localization.Localization(__file__, 648, 19), schur_34240, *[A_34241], **kwargs_34244)
    
    # Obtaining the member '__getitem__' of a type (line 648)
    getitem___34246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 12), schur_call_result_34245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 648)
    subscript_call_result_34247 = invoke(stypy.reporting.localization.Localization(__file__, 648, 12), getitem___34246, int_34239)
    
    # Assigning a type to the variable 'tuple_var_assignment_32854' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 12), 'tuple_var_assignment_32854', subscript_call_result_34247)
    
    # Assigning a Name to a Name (line 648):
    # Getting the type of 'tuple_var_assignment_32853' (line 648)
    tuple_var_assignment_32853_34248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 12), 'tuple_var_assignment_32853')
    # Assigning a type to the variable 'T' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 12), 'T', tuple_var_assignment_32853_34248)
    
    # Assigning a Name to a Name (line 648):
    # Getting the type of 'tuple_var_assignment_32854' (line 648)
    tuple_var_assignment_32854_34249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 12), 'tuple_var_assignment_32854')
    # Assigning a type to the variable 'Z' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 15), 'Z', tuple_var_assignment_32854_34249)
    # SSA join for if statement (line 643)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 639)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 652):
    
    # Assigning a Call to a Name (line 652):
    
    # Call to diag(...): (line 652)
    # Processing the call arguments (line 652)
    # Getting the type of 'T' (line 652)
    T_34252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 21), 'T', False)
    # Processing the call keyword arguments (line 652)
    kwargs_34253 = {}
    # Getting the type of 'np' (line 652)
    np_34250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 13), 'np', False)
    # Obtaining the member 'diag' of a type (line 652)
    diag_34251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 13), np_34250, 'diag')
    # Calling diag(args, kwargs) (line 652)
    diag_call_result_34254 = invoke(stypy.reporting.localization.Localization(__file__, 652, 13), diag_34251, *[T_34252], **kwargs_34253)
    
    # Assigning a type to the variable 'T_diag' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 4), 'T_diag', diag_call_result_34254)
    
    
    
    # Call to count_nonzero(...): (line 653)
    # Processing the call arguments (line 653)
    # Getting the type of 'T_diag' (line 653)
    T_diag_34257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 24), 'T_diag', False)
    # Processing the call keyword arguments (line 653)
    kwargs_34258 = {}
    # Getting the type of 'np' (line 653)
    np_34255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 7), 'np', False)
    # Obtaining the member 'count_nonzero' of a type (line 653)
    count_nonzero_34256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 7), np_34255, 'count_nonzero')
    # Calling count_nonzero(args, kwargs) (line 653)
    count_nonzero_call_result_34259 = invoke(stypy.reporting.localization.Localization(__file__, 653, 7), count_nonzero_34256, *[T_diag_34257], **kwargs_34258)
    
    # Getting the type of 'n' (line 653)
    n_34260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 35), 'n')
    # Applying the binary operator '!=' (line 653)
    result_ne_34261 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 7), '!=', count_nonzero_call_result_34259, n_34260)
    
    # Testing the type of an if condition (line 653)
    if_condition_34262 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 653, 4), result_ne_34261)
    # Assigning a type to the variable 'if_condition_34262' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'if_condition_34262', if_condition_34262)
    # SSA begins for if statement (line 653)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to FractionalMatrixPowerError(...): (line 654)
    # Processing the call arguments (line 654)
    str_34264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 16), 'str', 'cannot use inverse scaling and squaring to find the fractional matrix power of a singular matrix')
    # Processing the call keyword arguments (line 654)
    kwargs_34265 = {}
    # Getting the type of 'FractionalMatrixPowerError' (line 654)
    FractionalMatrixPowerError_34263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 14), 'FractionalMatrixPowerError', False)
    # Calling FractionalMatrixPowerError(args, kwargs) (line 654)
    FractionalMatrixPowerError_call_result_34266 = invoke(stypy.reporting.localization.Localization(__file__, 654, 14), FractionalMatrixPowerError_34263, *[str_34264], **kwargs_34265)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 654, 8), FractionalMatrixPowerError_call_result_34266, 'raise parameter', BaseException)
    # SSA join for if statement (line 653)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isrealobj(...): (line 660)
    # Processing the call arguments (line 660)
    # Getting the type of 'T' (line 660)
    T_34269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 20), 'T', False)
    # Processing the call keyword arguments (line 660)
    kwargs_34270 = {}
    # Getting the type of 'np' (line 660)
    np_34267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 7), 'np', False)
    # Obtaining the member 'isrealobj' of a type (line 660)
    isrealobj_34268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 7), np_34267, 'isrealobj')
    # Calling isrealobj(args, kwargs) (line 660)
    isrealobj_call_result_34271 = invoke(stypy.reporting.localization.Localization(__file__, 660, 7), isrealobj_34268, *[T_34269], **kwargs_34270)
    
    
    
    # Call to min(...): (line 660)
    # Processing the call arguments (line 660)
    # Getting the type of 'T_diag' (line 660)
    T_diag_34274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 34), 'T_diag', False)
    # Processing the call keyword arguments (line 660)
    kwargs_34275 = {}
    # Getting the type of 'np' (line 660)
    np_34272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 27), 'np', False)
    # Obtaining the member 'min' of a type (line 660)
    min_34273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 27), np_34272, 'min')
    # Calling min(args, kwargs) (line 660)
    min_call_result_34276 = invoke(stypy.reporting.localization.Localization(__file__, 660, 27), min_34273, *[T_diag_34274], **kwargs_34275)
    
    int_34277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 44), 'int')
    # Applying the binary operator '<' (line 660)
    result_lt_34278 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 27), '<', min_call_result_34276, int_34277)
    
    # Applying the binary operator 'and' (line 660)
    result_and_keyword_34279 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 7), 'and', isrealobj_call_result_34271, result_lt_34278)
    
    # Testing the type of an if condition (line 660)
    if_condition_34280 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 660, 4), result_and_keyword_34279)
    # Assigning a type to the variable 'if_condition_34280' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'if_condition_34280', if_condition_34280)
    # SSA begins for if statement (line 660)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 661):
    
    # Assigning a Call to a Name (line 661):
    
    # Call to astype(...): (line 661)
    # Processing the call arguments (line 661)
    # Getting the type of 'complex' (line 661)
    complex_34283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 21), 'complex', False)
    # Processing the call keyword arguments (line 661)
    kwargs_34284 = {}
    # Getting the type of 'T' (line 661)
    T_34281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'T', False)
    # Obtaining the member 'astype' of a type (line 661)
    astype_34282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 12), T_34281, 'astype')
    # Calling astype(args, kwargs) (line 661)
    astype_call_result_34285 = invoke(stypy.reporting.localization.Localization(__file__, 661, 12), astype_34282, *[complex_34283], **kwargs_34284)
    
    # Assigning a type to the variable 'T' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 8), 'T', astype_call_result_34285)
    # SSA join for if statement (line 660)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 665):
    
    # Assigning a Call to a Name (line 665):
    
    # Call to _remainder_matrix_power_triu(...): (line 665)
    # Processing the call arguments (line 665)
    # Getting the type of 'T' (line 665)
    T_34287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 37), 'T', False)
    # Getting the type of 't' (line 665)
    t_34288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 40), 't', False)
    # Processing the call keyword arguments (line 665)
    kwargs_34289 = {}
    # Getting the type of '_remainder_matrix_power_triu' (line 665)
    _remainder_matrix_power_triu_34286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 8), '_remainder_matrix_power_triu', False)
    # Calling _remainder_matrix_power_triu(args, kwargs) (line 665)
    _remainder_matrix_power_triu_call_result_34290 = invoke(stypy.reporting.localization.Localization(__file__, 665, 8), _remainder_matrix_power_triu_34286, *[T_34287, t_34288], **kwargs_34289)
    
    # Assigning a type to the variable 'U' (line 665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'U', _remainder_matrix_power_triu_call_result_34290)
    
    # Type idiom detected: calculating its left and rigth part (line 666)
    # Getting the type of 'Z' (line 666)
    Z_34291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 4), 'Z')
    # Getting the type of 'None' (line 666)
    None_34292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 16), 'None')
    
    (may_be_34293, more_types_in_union_34294) = may_not_be_none(Z_34291, None_34292)

    if may_be_34293:

        if more_types_in_union_34294:
            # Runtime conditional SSA (line 666)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 667):
        
        # Assigning a Attribute to a Name (line 667):
        
        # Call to conjugate(...): (line 667)
        # Processing the call arguments (line 667)
        # Getting the type of 'Z' (line 667)
        Z_34297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 26), 'Z', False)
        # Processing the call keyword arguments (line 667)
        kwargs_34298 = {}
        # Getting the type of 'np' (line 667)
        np_34295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 13), 'np', False)
        # Obtaining the member 'conjugate' of a type (line 667)
        conjugate_34296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 13), np_34295, 'conjugate')
        # Calling conjugate(args, kwargs) (line 667)
        conjugate_call_result_34299 = invoke(stypy.reporting.localization.Localization(__file__, 667, 13), conjugate_34296, *[Z_34297], **kwargs_34298)
        
        # Obtaining the member 'T' of a type (line 667)
        T_34300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 13), conjugate_call_result_34299, 'T')
        # Assigning a type to the variable 'ZH' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'ZH', T_34300)
        
        # Call to dot(...): (line 668)
        # Processing the call arguments (line 668)
        # Getting the type of 'ZH' (line 668)
        ZH_34307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 28), 'ZH', False)
        # Processing the call keyword arguments (line 668)
        kwargs_34308 = {}
        
        # Call to dot(...): (line 668)
        # Processing the call arguments (line 668)
        # Getting the type of 'U' (line 668)
        U_34303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 21), 'U', False)
        # Processing the call keyword arguments (line 668)
        kwargs_34304 = {}
        # Getting the type of 'Z' (line 668)
        Z_34301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 15), 'Z', False)
        # Obtaining the member 'dot' of a type (line 668)
        dot_34302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 15), Z_34301, 'dot')
        # Calling dot(args, kwargs) (line 668)
        dot_call_result_34305 = invoke(stypy.reporting.localization.Localization(__file__, 668, 15), dot_34302, *[U_34303], **kwargs_34304)
        
        # Obtaining the member 'dot' of a type (line 668)
        dot_34306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 15), dot_call_result_34305, 'dot')
        # Calling dot(args, kwargs) (line 668)
        dot_call_result_34309 = invoke(stypy.reporting.localization.Localization(__file__, 668, 15), dot_34306, *[ZH_34307], **kwargs_34308)
        
        # Assigning a type to the variable 'stypy_return_type' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'stypy_return_type', dot_call_result_34309)

        if more_types_in_union_34294:
            # Runtime conditional SSA for else branch (line 666)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_34293) or more_types_in_union_34294):
        # Getting the type of 'U' (line 670)
        U_34310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 15), 'U')
        # Assigning a type to the variable 'stypy_return_type' (line 670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 8), 'stypy_return_type', U_34310)

        if (may_be_34293 and more_types_in_union_34294):
            # SSA join for if statement (line 666)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_remainder_matrix_power(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_remainder_matrix_power' in the type store
    # Getting the type of 'stypy_return_type' (line 598)
    stypy_return_type_34311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34311)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_remainder_matrix_power'
    return stypy_return_type_34311

# Assigning a type to the variable '_remainder_matrix_power' (line 598)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 0), '_remainder_matrix_power', _remainder_matrix_power)

@norecursion
def _fractional_matrix_power(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fractional_matrix_power'
    module_type_store = module_type_store.open_function_context('_fractional_matrix_power', 673, 0, False)
    
    # Passed parameters checking function
    _fractional_matrix_power.stypy_localization = localization
    _fractional_matrix_power.stypy_type_of_self = None
    _fractional_matrix_power.stypy_type_store = module_type_store
    _fractional_matrix_power.stypy_function_name = '_fractional_matrix_power'
    _fractional_matrix_power.stypy_param_names_list = ['A', 'p']
    _fractional_matrix_power.stypy_varargs_param_name = None
    _fractional_matrix_power.stypy_kwargs_param_name = None
    _fractional_matrix_power.stypy_call_defaults = defaults
    _fractional_matrix_power.stypy_call_varargs = varargs
    _fractional_matrix_power.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fractional_matrix_power', ['A', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fractional_matrix_power', localization, ['A', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fractional_matrix_power(...)' code ##################

    str_34312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, (-1)), 'str', '\n    Compute the fractional power of a matrix.\n\n    See the fractional_matrix_power docstring in matfuncs.py for more info.\n\n    ')
    
    # Assigning a Call to a Name (line 680):
    
    # Assigning a Call to a Name (line 680):
    
    # Call to asarray(...): (line 680)
    # Processing the call arguments (line 680)
    # Getting the type of 'A' (line 680)
    A_34315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 19), 'A', False)
    # Processing the call keyword arguments (line 680)
    kwargs_34316 = {}
    # Getting the type of 'np' (line 680)
    np_34313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 680)
    asarray_34314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 8), np_34313, 'asarray')
    # Calling asarray(args, kwargs) (line 680)
    asarray_call_result_34317 = invoke(stypy.reporting.localization.Localization(__file__, 680, 8), asarray_34314, *[A_34315], **kwargs_34316)
    
    # Assigning a type to the variable 'A' (line 680)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 4), 'A', asarray_call_result_34317)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 681)
    # Processing the call arguments (line 681)
    # Getting the type of 'A' (line 681)
    A_34319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 681)
    shape_34320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 11), A_34319, 'shape')
    # Processing the call keyword arguments (line 681)
    kwargs_34321 = {}
    # Getting the type of 'len' (line 681)
    len_34318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 7), 'len', False)
    # Calling len(args, kwargs) (line 681)
    len_call_result_34322 = invoke(stypy.reporting.localization.Localization(__file__, 681, 7), len_34318, *[shape_34320], **kwargs_34321)
    
    int_34323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 23), 'int')
    # Applying the binary operator '!=' (line 681)
    result_ne_34324 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 7), '!=', len_call_result_34322, int_34323)
    
    
    
    # Obtaining the type of the subscript
    int_34325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 36), 'int')
    # Getting the type of 'A' (line 681)
    A_34326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 28), 'A')
    # Obtaining the member 'shape' of a type (line 681)
    shape_34327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 28), A_34326, 'shape')
    # Obtaining the member '__getitem__' of a type (line 681)
    getitem___34328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 28), shape_34327, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 681)
    subscript_call_result_34329 = invoke(stypy.reporting.localization.Localization(__file__, 681, 28), getitem___34328, int_34325)
    
    
    # Obtaining the type of the subscript
    int_34330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 50), 'int')
    # Getting the type of 'A' (line 681)
    A_34331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 42), 'A')
    # Obtaining the member 'shape' of a type (line 681)
    shape_34332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 42), A_34331, 'shape')
    # Obtaining the member '__getitem__' of a type (line 681)
    getitem___34333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 42), shape_34332, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 681)
    subscript_call_result_34334 = invoke(stypy.reporting.localization.Localization(__file__, 681, 42), getitem___34333, int_34330)
    
    # Applying the binary operator '!=' (line 681)
    result_ne_34335 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 28), '!=', subscript_call_result_34329, subscript_call_result_34334)
    
    # Applying the binary operator 'or' (line 681)
    result_or_keyword_34336 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 7), 'or', result_ne_34324, result_ne_34335)
    
    # Testing the type of an if condition (line 681)
    if_condition_34337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 681, 4), result_or_keyword_34336)
    # Assigning a type to the variable 'if_condition_34337' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 4), 'if_condition_34337', if_condition_34337)
    # SSA begins for if statement (line 681)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 682)
    # Processing the call arguments (line 682)
    str_34339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 25), 'str', 'expected a square matrix')
    # Processing the call keyword arguments (line 682)
    kwargs_34340 = {}
    # Getting the type of 'ValueError' (line 682)
    ValueError_34338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 682)
    ValueError_call_result_34341 = invoke(stypy.reporting.localization.Localization(__file__, 682, 14), ValueError_34338, *[str_34339], **kwargs_34340)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 682, 8), ValueError_call_result_34341, 'raise parameter', BaseException)
    # SSA join for if statement (line 681)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'p' (line 683)
    p_34342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 7), 'p')
    
    # Call to int(...): (line 683)
    # Processing the call arguments (line 683)
    # Getting the type of 'p' (line 683)
    p_34344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 16), 'p', False)
    # Processing the call keyword arguments (line 683)
    kwargs_34345 = {}
    # Getting the type of 'int' (line 683)
    int_34343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 12), 'int', False)
    # Calling int(args, kwargs) (line 683)
    int_call_result_34346 = invoke(stypy.reporting.localization.Localization(__file__, 683, 12), int_34343, *[p_34344], **kwargs_34345)
    
    # Applying the binary operator '==' (line 683)
    result_eq_34347 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 7), '==', p_34342, int_call_result_34346)
    
    # Testing the type of an if condition (line 683)
    if_condition_34348 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 683, 4), result_eq_34347)
    # Assigning a type to the variable 'if_condition_34348' (line 683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'if_condition_34348', if_condition_34348)
    # SSA begins for if statement (line 683)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to matrix_power(...): (line 684)
    # Processing the call arguments (line 684)
    # Getting the type of 'A' (line 684)
    A_34352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 38), 'A', False)
    
    # Call to int(...): (line 684)
    # Processing the call arguments (line 684)
    # Getting the type of 'p' (line 684)
    p_34354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 45), 'p', False)
    # Processing the call keyword arguments (line 684)
    kwargs_34355 = {}
    # Getting the type of 'int' (line 684)
    int_34353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 41), 'int', False)
    # Calling int(args, kwargs) (line 684)
    int_call_result_34356 = invoke(stypy.reporting.localization.Localization(__file__, 684, 41), int_34353, *[p_34354], **kwargs_34355)
    
    # Processing the call keyword arguments (line 684)
    kwargs_34357 = {}
    # Getting the type of 'np' (line 684)
    np_34349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 15), 'np', False)
    # Obtaining the member 'linalg' of a type (line 684)
    linalg_34350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 15), np_34349, 'linalg')
    # Obtaining the member 'matrix_power' of a type (line 684)
    matrix_power_34351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 15), linalg_34350, 'matrix_power')
    # Calling matrix_power(args, kwargs) (line 684)
    matrix_power_call_result_34358 = invoke(stypy.reporting.localization.Localization(__file__, 684, 15), matrix_power_34351, *[A_34352, int_call_result_34356], **kwargs_34357)
    
    # Assigning a type to the variable 'stypy_return_type' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'stypy_return_type', matrix_power_call_result_34358)
    # SSA join for if statement (line 683)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 686):
    
    # Assigning a Call to a Name (line 686):
    
    # Call to svdvals(...): (line 686)
    # Processing the call arguments (line 686)
    # Getting the type of 'A' (line 686)
    A_34360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 16), 'A', False)
    # Processing the call keyword arguments (line 686)
    kwargs_34361 = {}
    # Getting the type of 'svdvals' (line 686)
    svdvals_34359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'svdvals', False)
    # Calling svdvals(args, kwargs) (line 686)
    svdvals_call_result_34362 = invoke(stypy.reporting.localization.Localization(__file__, 686, 8), svdvals_34359, *[A_34360], **kwargs_34361)
    
    # Assigning a type to the variable 's' (line 686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 's', svdvals_call_result_34362)
    
    
    # Obtaining the type of the subscript
    int_34363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 9), 'int')
    # Getting the type of 's' (line 690)
    s_34364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 7), 's')
    # Obtaining the member '__getitem__' of a type (line 690)
    getitem___34365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 7), s_34364, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 690)
    subscript_call_result_34366 = invoke(stypy.reporting.localization.Localization(__file__, 690, 7), getitem___34365, int_34363)
    
    # Testing the type of an if condition (line 690)
    if_condition_34367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 690, 4), subscript_call_result_34366)
    # Assigning a type to the variable 'if_condition_34367' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'if_condition_34367', if_condition_34367)
    # SSA begins for if statement (line 690)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 693):
    
    # Assigning a BinOp to a Name (line 693):
    
    # Obtaining the type of the subscript
    int_34368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 15), 'int')
    # Getting the type of 's' (line 693)
    s_34369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 13), 's')
    # Obtaining the member '__getitem__' of a type (line 693)
    getitem___34370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 13), s_34369, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 693)
    subscript_call_result_34371 = invoke(stypy.reporting.localization.Localization(__file__, 693, 13), getitem___34370, int_34368)
    
    
    # Obtaining the type of the subscript
    int_34372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 22), 'int')
    # Getting the type of 's' (line 693)
    s_34373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 20), 's')
    # Obtaining the member '__getitem__' of a type (line 693)
    getitem___34374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 20), s_34373, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 693)
    subscript_call_result_34375 = invoke(stypy.reporting.localization.Localization(__file__, 693, 20), getitem___34374, int_34372)
    
    # Applying the binary operator 'div' (line 693)
    result_div_34376 = python_operator(stypy.reporting.localization.Localization(__file__, 693, 13), 'div', subscript_call_result_34371, subscript_call_result_34375)
    
    # Assigning a type to the variable 'k2' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'k2', result_div_34376)
    
    # Assigning a BinOp to a Name (line 694):
    
    # Assigning a BinOp to a Name (line 694):
    # Getting the type of 'p' (line 694)
    p_34377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 13), 'p')
    
    # Call to floor(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'p' (line 694)
    p_34380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 26), 'p', False)
    # Processing the call keyword arguments (line 694)
    kwargs_34381 = {}
    # Getting the type of 'np' (line 694)
    np_34378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'np', False)
    # Obtaining the member 'floor' of a type (line 694)
    floor_34379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 17), np_34378, 'floor')
    # Calling floor(args, kwargs) (line 694)
    floor_call_result_34382 = invoke(stypy.reporting.localization.Localization(__file__, 694, 17), floor_34379, *[p_34380], **kwargs_34381)
    
    # Applying the binary operator '-' (line 694)
    result_sub_34383 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 13), '-', p_34377, floor_call_result_34382)
    
    # Assigning a type to the variable 'p1' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'p1', result_sub_34383)
    
    # Assigning a BinOp to a Name (line 695):
    
    # Assigning a BinOp to a Name (line 695):
    # Getting the type of 'p' (line 695)
    p_34384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 13), 'p')
    
    # Call to ceil(...): (line 695)
    # Processing the call arguments (line 695)
    # Getting the type of 'p' (line 695)
    p_34387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 25), 'p', False)
    # Processing the call keyword arguments (line 695)
    kwargs_34388 = {}
    # Getting the type of 'np' (line 695)
    np_34385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 17), 'np', False)
    # Obtaining the member 'ceil' of a type (line 695)
    ceil_34386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 17), np_34385, 'ceil')
    # Calling ceil(args, kwargs) (line 695)
    ceil_call_result_34389 = invoke(stypy.reporting.localization.Localization(__file__, 695, 17), ceil_34386, *[p_34387], **kwargs_34388)
    
    # Applying the binary operator '-' (line 695)
    result_sub_34390 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 13), '-', p_34384, ceil_call_result_34389)
    
    # Assigning a type to the variable 'p2' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'p2', result_sub_34390)
    
    
    # Getting the type of 'p1' (line 696)
    p1_34391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 11), 'p1')
    # Getting the type of 'k2' (line 696)
    k2_34392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 16), 'k2')
    int_34393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 23), 'int')
    # Getting the type of 'p1' (line 696)
    p1_34394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 27), 'p1')
    # Applying the binary operator '-' (line 696)
    result_sub_34395 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 23), '-', int_34393, p1_34394)
    
    # Applying the binary operator '**' (line 696)
    result_pow_34396 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 16), '**', k2_34392, result_sub_34395)
    
    # Applying the binary operator '*' (line 696)
    result_mul_34397 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 11), '*', p1_34391, result_pow_34396)
    
    
    # Getting the type of 'p2' (line 696)
    p2_34398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 35), 'p2')
    # Applying the 'usub' unary operator (line 696)
    result___neg___34399 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 34), 'usub', p2_34398)
    
    # Getting the type of 'k2' (line 696)
    k2_34400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 40), 'k2')
    # Applying the binary operator '*' (line 696)
    result_mul_34401 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 34), '*', result___neg___34399, k2_34400)
    
    # Applying the binary operator '<=' (line 696)
    result_le_34402 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 11), '<=', result_mul_34397, result_mul_34401)
    
    # Testing the type of an if condition (line 696)
    if_condition_34403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 8), result_le_34402)
    # Assigning a type to the variable 'if_condition_34403' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'if_condition_34403', if_condition_34403)
    # SSA begins for if statement (line 696)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 697):
    
    # Assigning a Call to a Name (line 697):
    
    # Call to int(...): (line 697)
    # Processing the call arguments (line 697)
    
    # Call to floor(...): (line 697)
    # Processing the call arguments (line 697)
    # Getting the type of 'p' (line 697)
    p_34407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 29), 'p', False)
    # Processing the call keyword arguments (line 697)
    kwargs_34408 = {}
    # Getting the type of 'np' (line 697)
    np_34405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 20), 'np', False)
    # Obtaining the member 'floor' of a type (line 697)
    floor_34406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 20), np_34405, 'floor')
    # Calling floor(args, kwargs) (line 697)
    floor_call_result_34409 = invoke(stypy.reporting.localization.Localization(__file__, 697, 20), floor_34406, *[p_34407], **kwargs_34408)
    
    # Processing the call keyword arguments (line 697)
    kwargs_34410 = {}
    # Getting the type of 'int' (line 697)
    int_34404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'int', False)
    # Calling int(args, kwargs) (line 697)
    int_call_result_34411 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), int_34404, *[floor_call_result_34409], **kwargs_34410)
    
    # Assigning a type to the variable 'a' (line 697)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 12), 'a', int_call_result_34411)
    
    # Assigning a Name to a Name (line 698):
    
    # Assigning a Name to a Name (line 698):
    # Getting the type of 'p1' (line 698)
    p1_34412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 16), 'p1')
    # Assigning a type to the variable 'b' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 12), 'b', p1_34412)
    # SSA branch for the else part of an if statement (line 696)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 700):
    
    # Assigning a Call to a Name (line 700):
    
    # Call to int(...): (line 700)
    # Processing the call arguments (line 700)
    
    # Call to ceil(...): (line 700)
    # Processing the call arguments (line 700)
    # Getting the type of 'p' (line 700)
    p_34416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 28), 'p', False)
    # Processing the call keyword arguments (line 700)
    kwargs_34417 = {}
    # Getting the type of 'np' (line 700)
    np_34414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 20), 'np', False)
    # Obtaining the member 'ceil' of a type (line 700)
    ceil_34415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 20), np_34414, 'ceil')
    # Calling ceil(args, kwargs) (line 700)
    ceil_call_result_34418 = invoke(stypy.reporting.localization.Localization(__file__, 700, 20), ceil_34415, *[p_34416], **kwargs_34417)
    
    # Processing the call keyword arguments (line 700)
    kwargs_34419 = {}
    # Getting the type of 'int' (line 700)
    int_34413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 16), 'int', False)
    # Calling int(args, kwargs) (line 700)
    int_call_result_34420 = invoke(stypy.reporting.localization.Localization(__file__, 700, 16), int_34413, *[ceil_call_result_34418], **kwargs_34419)
    
    # Assigning a type to the variable 'a' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 12), 'a', int_call_result_34420)
    
    # Assigning a Name to a Name (line 701):
    
    # Assigning a Name to a Name (line 701):
    # Getting the type of 'p2' (line 701)
    p2_34421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 16), 'p2')
    # Assigning a type to the variable 'b' (line 701)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 12), 'b', p2_34421)
    # SSA join for if statement (line 696)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 702)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 703):
    
    # Assigning a Call to a Name (line 703):
    
    # Call to _remainder_matrix_power(...): (line 703)
    # Processing the call arguments (line 703)
    # Getting the type of 'A' (line 703)
    A_34423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 40), 'A', False)
    # Getting the type of 'b' (line 703)
    b_34424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 43), 'b', False)
    # Processing the call keyword arguments (line 703)
    kwargs_34425 = {}
    # Getting the type of '_remainder_matrix_power' (line 703)
    _remainder_matrix_power_34422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 16), '_remainder_matrix_power', False)
    # Calling _remainder_matrix_power(args, kwargs) (line 703)
    _remainder_matrix_power_call_result_34426 = invoke(stypy.reporting.localization.Localization(__file__, 703, 16), _remainder_matrix_power_34422, *[A_34423, b_34424], **kwargs_34425)
    
    # Assigning a type to the variable 'R' (line 703)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 12), 'R', _remainder_matrix_power_call_result_34426)
    
    # Assigning a Call to a Name (line 704):
    
    # Assigning a Call to a Name (line 704):
    
    # Call to matrix_power(...): (line 704)
    # Processing the call arguments (line 704)
    # Getting the type of 'A' (line 704)
    A_34430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 39), 'A', False)
    # Getting the type of 'a' (line 704)
    a_34431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 42), 'a', False)
    # Processing the call keyword arguments (line 704)
    kwargs_34432 = {}
    # Getting the type of 'np' (line 704)
    np_34427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 16), 'np', False)
    # Obtaining the member 'linalg' of a type (line 704)
    linalg_34428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 16), np_34427, 'linalg')
    # Obtaining the member 'matrix_power' of a type (line 704)
    matrix_power_34429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 16), linalg_34428, 'matrix_power')
    # Calling matrix_power(args, kwargs) (line 704)
    matrix_power_call_result_34433 = invoke(stypy.reporting.localization.Localization(__file__, 704, 16), matrix_power_34429, *[A_34430, a_34431], **kwargs_34432)
    
    # Assigning a type to the variable 'Q' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 12), 'Q', matrix_power_call_result_34433)
    
    # Call to dot(...): (line 705)
    # Processing the call arguments (line 705)
    # Getting the type of 'R' (line 705)
    R_34436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 25), 'R', False)
    # Processing the call keyword arguments (line 705)
    kwargs_34437 = {}
    # Getting the type of 'Q' (line 705)
    Q_34434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 19), 'Q', False)
    # Obtaining the member 'dot' of a type (line 705)
    dot_34435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 19), Q_34434, 'dot')
    # Calling dot(args, kwargs) (line 705)
    dot_call_result_34438 = invoke(stypy.reporting.localization.Localization(__file__, 705, 19), dot_34435, *[R_34436], **kwargs_34437)
    
    # Assigning a type to the variable 'stypy_return_type' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 12), 'stypy_return_type', dot_call_result_34438)
    # SSA branch for the except part of a try statement (line 702)
    # SSA branch for the except 'Attribute' branch of a try statement (line 702)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 702)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 690)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'p' (line 710)
    p_34439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 7), 'p')
    int_34440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 11), 'int')
    # Applying the binary operator '<' (line 710)
    result_lt_34441 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 7), '<', p_34439, int_34440)
    
    # Testing the type of an if condition (line 710)
    if_condition_34442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 710, 4), result_lt_34441)
    # Assigning a type to the variable 'if_condition_34442' (line 710)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'if_condition_34442', if_condition_34442)
    # SSA begins for if statement (line 710)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 711):
    
    # Assigning a Call to a Name (line 711):
    
    # Call to empty_like(...): (line 711)
    # Processing the call arguments (line 711)
    # Getting the type of 'A' (line 711)
    A_34445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 26), 'A', False)
    # Processing the call keyword arguments (line 711)
    kwargs_34446 = {}
    # Getting the type of 'np' (line 711)
    np_34443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 12), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 711)
    empty_like_34444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 12), np_34443, 'empty_like')
    # Calling empty_like(args, kwargs) (line 711)
    empty_like_call_result_34447 = invoke(stypy.reporting.localization.Localization(__file__, 711, 12), empty_like_34444, *[A_34445], **kwargs_34446)
    
    # Assigning a type to the variable 'X' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'X', empty_like_call_result_34447)
    
    # Call to fill(...): (line 712)
    # Processing the call arguments (line 712)
    # Getting the type of 'np' (line 712)
    np_34450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 15), 'np', False)
    # Obtaining the member 'nan' of a type (line 712)
    nan_34451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 15), np_34450, 'nan')
    # Processing the call keyword arguments (line 712)
    kwargs_34452 = {}
    # Getting the type of 'X' (line 712)
    X_34448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'X', False)
    # Obtaining the member 'fill' of a type (line 712)
    fill_34449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 8), X_34448, 'fill')
    # Calling fill(args, kwargs) (line 712)
    fill_call_result_34453 = invoke(stypy.reporting.localization.Localization(__file__, 712, 8), fill_34449, *[nan_34451], **kwargs_34452)
    
    # Getting the type of 'X' (line 713)
    X_34454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 15), 'X')
    # Assigning a type to the variable 'stypy_return_type' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'stypy_return_type', X_34454)
    # SSA branch for the else part of an if statement (line 710)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 715):
    
    # Assigning a BinOp to a Name (line 715):
    # Getting the type of 'p' (line 715)
    p_34455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 13), 'p')
    
    # Call to floor(...): (line 715)
    # Processing the call arguments (line 715)
    # Getting the type of 'p' (line 715)
    p_34458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 26), 'p', False)
    # Processing the call keyword arguments (line 715)
    kwargs_34459 = {}
    # Getting the type of 'np' (line 715)
    np_34456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 17), 'np', False)
    # Obtaining the member 'floor' of a type (line 715)
    floor_34457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 17), np_34456, 'floor')
    # Calling floor(args, kwargs) (line 715)
    floor_call_result_34460 = invoke(stypy.reporting.localization.Localization(__file__, 715, 17), floor_34457, *[p_34458], **kwargs_34459)
    
    # Applying the binary operator '-' (line 715)
    result_sub_34461 = python_operator(stypy.reporting.localization.Localization(__file__, 715, 13), '-', p_34455, floor_call_result_34460)
    
    # Assigning a type to the variable 'p1' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'p1', result_sub_34461)
    
    # Assigning a Call to a Name (line 716):
    
    # Assigning a Call to a Name (line 716):
    
    # Call to int(...): (line 716)
    # Processing the call arguments (line 716)
    
    # Call to floor(...): (line 716)
    # Processing the call arguments (line 716)
    # Getting the type of 'p' (line 716)
    p_34465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 25), 'p', False)
    # Processing the call keyword arguments (line 716)
    kwargs_34466 = {}
    # Getting the type of 'np' (line 716)
    np_34463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 16), 'np', False)
    # Obtaining the member 'floor' of a type (line 716)
    floor_34464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 16), np_34463, 'floor')
    # Calling floor(args, kwargs) (line 716)
    floor_call_result_34467 = invoke(stypy.reporting.localization.Localization(__file__, 716, 16), floor_34464, *[p_34465], **kwargs_34466)
    
    # Processing the call keyword arguments (line 716)
    kwargs_34468 = {}
    # Getting the type of 'int' (line 716)
    int_34462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 12), 'int', False)
    # Calling int(args, kwargs) (line 716)
    int_call_result_34469 = invoke(stypy.reporting.localization.Localization(__file__, 716, 12), int_34462, *[floor_call_result_34467], **kwargs_34468)
    
    # Assigning a type to the variable 'a' (line 716)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'a', int_call_result_34469)
    
    # Assigning a Name to a Name (line 717):
    
    # Assigning a Name to a Name (line 717):
    # Getting the type of 'p1' (line 717)
    p1_34470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 12), 'p1')
    # Assigning a type to the variable 'b' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'b', p1_34470)
    
    # Assigning a Call to a Tuple (line 718):
    
    # Assigning a Subscript to a Name (line 718):
    
    # Obtaining the type of the subscript
    int_34471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 8), 'int')
    
    # Call to funm(...): (line 718)
    # Processing the call arguments (line 718)
    # Getting the type of 'A' (line 718)
    A_34473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 23), 'A', False)

    @norecursion
    def _stypy_temp_lambda_18(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_18'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_18', 718, 26, True)
        # Passed parameters checking function
        _stypy_temp_lambda_18.stypy_localization = localization
        _stypy_temp_lambda_18.stypy_type_of_self = None
        _stypy_temp_lambda_18.stypy_type_store = module_type_store
        _stypy_temp_lambda_18.stypy_function_name = '_stypy_temp_lambda_18'
        _stypy_temp_lambda_18.stypy_param_names_list = ['x']
        _stypy_temp_lambda_18.stypy_varargs_param_name = None
        _stypy_temp_lambda_18.stypy_kwargs_param_name = None
        _stypy_temp_lambda_18.stypy_call_defaults = defaults
        _stypy_temp_lambda_18.stypy_call_varargs = varargs
        _stypy_temp_lambda_18.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_18', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_18', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to pow(...): (line 718)
        # Processing the call arguments (line 718)
        # Getting the type of 'x' (line 718)
        x_34475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 40), 'x', False)
        # Getting the type of 'b' (line 718)
        b_34476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 43), 'b', False)
        # Processing the call keyword arguments (line 718)
        kwargs_34477 = {}
        # Getting the type of 'pow' (line 718)
        pow_34474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 36), 'pow', False)
        # Calling pow(args, kwargs) (line 718)
        pow_call_result_34478 = invoke(stypy.reporting.localization.Localization(__file__, 718, 36), pow_34474, *[x_34475, b_34476], **kwargs_34477)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 26), 'stypy_return_type', pow_call_result_34478)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_18' in the type store
        # Getting the type of 'stypy_return_type' (line 718)
        stypy_return_type_34479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 26), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34479)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_18'
        return stypy_return_type_34479

    # Assigning a type to the variable '_stypy_temp_lambda_18' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 26), '_stypy_temp_lambda_18', _stypy_temp_lambda_18)
    # Getting the type of '_stypy_temp_lambda_18' (line 718)
    _stypy_temp_lambda_18_34480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 26), '_stypy_temp_lambda_18')
    # Processing the call keyword arguments (line 718)
    # Getting the type of 'False' (line 718)
    False_34481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 52), 'False', False)
    keyword_34482 = False_34481
    kwargs_34483 = {'disp': keyword_34482}
    # Getting the type of 'funm' (line 718)
    funm_34472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 18), 'funm', False)
    # Calling funm(args, kwargs) (line 718)
    funm_call_result_34484 = invoke(stypy.reporting.localization.Localization(__file__, 718, 18), funm_34472, *[A_34473, _stypy_temp_lambda_18_34480], **kwargs_34483)
    
    # Obtaining the member '__getitem__' of a type (line 718)
    getitem___34485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 8), funm_call_result_34484, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 718)
    subscript_call_result_34486 = invoke(stypy.reporting.localization.Localization(__file__, 718, 8), getitem___34485, int_34471)
    
    # Assigning a type to the variable 'tuple_var_assignment_32855' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'tuple_var_assignment_32855', subscript_call_result_34486)
    
    # Assigning a Subscript to a Name (line 718):
    
    # Obtaining the type of the subscript
    int_34487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 8), 'int')
    
    # Call to funm(...): (line 718)
    # Processing the call arguments (line 718)
    # Getting the type of 'A' (line 718)
    A_34489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 23), 'A', False)

    @norecursion
    def _stypy_temp_lambda_19(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_19'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_19', 718, 26, True)
        # Passed parameters checking function
        _stypy_temp_lambda_19.stypy_localization = localization
        _stypy_temp_lambda_19.stypy_type_of_self = None
        _stypy_temp_lambda_19.stypy_type_store = module_type_store
        _stypy_temp_lambda_19.stypy_function_name = '_stypy_temp_lambda_19'
        _stypy_temp_lambda_19.stypy_param_names_list = ['x']
        _stypy_temp_lambda_19.stypy_varargs_param_name = None
        _stypy_temp_lambda_19.stypy_kwargs_param_name = None
        _stypy_temp_lambda_19.stypy_call_defaults = defaults
        _stypy_temp_lambda_19.stypy_call_varargs = varargs
        _stypy_temp_lambda_19.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_19', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_19', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to pow(...): (line 718)
        # Processing the call arguments (line 718)
        # Getting the type of 'x' (line 718)
        x_34491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 40), 'x', False)
        # Getting the type of 'b' (line 718)
        b_34492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 43), 'b', False)
        # Processing the call keyword arguments (line 718)
        kwargs_34493 = {}
        # Getting the type of 'pow' (line 718)
        pow_34490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 36), 'pow', False)
        # Calling pow(args, kwargs) (line 718)
        pow_call_result_34494 = invoke(stypy.reporting.localization.Localization(__file__, 718, 36), pow_34490, *[x_34491, b_34492], **kwargs_34493)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 26), 'stypy_return_type', pow_call_result_34494)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_19' in the type store
        # Getting the type of 'stypy_return_type' (line 718)
        stypy_return_type_34495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 26), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34495)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_19'
        return stypy_return_type_34495

    # Assigning a type to the variable '_stypy_temp_lambda_19' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 26), '_stypy_temp_lambda_19', _stypy_temp_lambda_19)
    # Getting the type of '_stypy_temp_lambda_19' (line 718)
    _stypy_temp_lambda_19_34496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 26), '_stypy_temp_lambda_19')
    # Processing the call keyword arguments (line 718)
    # Getting the type of 'False' (line 718)
    False_34497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 52), 'False', False)
    keyword_34498 = False_34497
    kwargs_34499 = {'disp': keyword_34498}
    # Getting the type of 'funm' (line 718)
    funm_34488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 18), 'funm', False)
    # Calling funm(args, kwargs) (line 718)
    funm_call_result_34500 = invoke(stypy.reporting.localization.Localization(__file__, 718, 18), funm_34488, *[A_34489, _stypy_temp_lambda_19_34496], **kwargs_34499)
    
    # Obtaining the member '__getitem__' of a type (line 718)
    getitem___34501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 8), funm_call_result_34500, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 718)
    subscript_call_result_34502 = invoke(stypy.reporting.localization.Localization(__file__, 718, 8), getitem___34501, int_34487)
    
    # Assigning a type to the variable 'tuple_var_assignment_32856' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'tuple_var_assignment_32856', subscript_call_result_34502)
    
    # Assigning a Name to a Name (line 718):
    # Getting the type of 'tuple_var_assignment_32855' (line 718)
    tuple_var_assignment_32855_34503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'tuple_var_assignment_32855')
    # Assigning a type to the variable 'R' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'R', tuple_var_assignment_32855_34503)
    
    # Assigning a Name to a Name (line 718):
    # Getting the type of 'tuple_var_assignment_32856' (line 718)
    tuple_var_assignment_32856_34504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'tuple_var_assignment_32856')
    # Assigning a type to the variable 'info' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 11), 'info', tuple_var_assignment_32856_34504)
    
    # Assigning a Call to a Name (line 719):
    
    # Assigning a Call to a Name (line 719):
    
    # Call to matrix_power(...): (line 719)
    # Processing the call arguments (line 719)
    # Getting the type of 'A' (line 719)
    A_34508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 35), 'A', False)
    # Getting the type of 'a' (line 719)
    a_34509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 38), 'a', False)
    # Processing the call keyword arguments (line 719)
    kwargs_34510 = {}
    # Getting the type of 'np' (line 719)
    np_34505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'np', False)
    # Obtaining the member 'linalg' of a type (line 719)
    linalg_34506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 12), np_34505, 'linalg')
    # Obtaining the member 'matrix_power' of a type (line 719)
    matrix_power_34507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 12), linalg_34506, 'matrix_power')
    # Calling matrix_power(args, kwargs) (line 719)
    matrix_power_call_result_34511 = invoke(stypy.reporting.localization.Localization(__file__, 719, 12), matrix_power_34507, *[A_34508, a_34509], **kwargs_34510)
    
    # Assigning a type to the variable 'Q' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'Q', matrix_power_call_result_34511)
    
    # Call to dot(...): (line 720)
    # Processing the call arguments (line 720)
    # Getting the type of 'R' (line 720)
    R_34514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 21), 'R', False)
    # Processing the call keyword arguments (line 720)
    kwargs_34515 = {}
    # Getting the type of 'Q' (line 720)
    Q_34512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 15), 'Q', False)
    # Obtaining the member 'dot' of a type (line 720)
    dot_34513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 15), Q_34512, 'dot')
    # Calling dot(args, kwargs) (line 720)
    dot_call_result_34516 = invoke(stypy.reporting.localization.Localization(__file__, 720, 15), dot_34513, *[R_34514], **kwargs_34515)
    
    # Assigning a type to the variable 'stypy_return_type' (line 720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'stypy_return_type', dot_call_result_34516)
    # SSA join for if statement (line 710)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_fractional_matrix_power(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fractional_matrix_power' in the type store
    # Getting the type of 'stypy_return_type' (line 673)
    stypy_return_type_34517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34517)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fractional_matrix_power'
    return stypy_return_type_34517

# Assigning a type to the variable '_fractional_matrix_power' (line 673)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 0), '_fractional_matrix_power', _fractional_matrix_power)

@norecursion
def _logm_triu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_logm_triu'
    module_type_store = module_type_store.open_function_context('_logm_triu', 723, 0, False)
    
    # Passed parameters checking function
    _logm_triu.stypy_localization = localization
    _logm_triu.stypy_type_of_self = None
    _logm_triu.stypy_type_store = module_type_store
    _logm_triu.stypy_function_name = '_logm_triu'
    _logm_triu.stypy_param_names_list = ['T']
    _logm_triu.stypy_varargs_param_name = None
    _logm_triu.stypy_kwargs_param_name = None
    _logm_triu.stypy_call_defaults = defaults
    _logm_triu.stypy_call_varargs = varargs
    _logm_triu.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_logm_triu', ['T'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_logm_triu', localization, ['T'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_logm_triu(...)' code ##################

    str_34518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, (-1)), 'str', '\n    Compute matrix logarithm of an upper triangular matrix.\n\n    The matrix logarithm is the inverse of\n    expm: expm(logm(`T`)) == `T`\n\n    Parameters\n    ----------\n    T : (N, N) array_like\n        Upper triangular matrix whose logarithm to evaluate\n\n    Returns\n    -------\n    logm : (N, N) ndarray\n        Matrix logarithm of `T`\n\n    References\n    ----------\n    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)\n           "Improved Inverse Scaling and Squaring Algorithms\n           for the Matrix Logarithm."\n           SIAM Journal on Scientific Computing, 34 (4). C152-C169.\n           ISSN 1095-7197\n\n    .. [2] Nicholas J. Higham (2008)\n           "Functions of Matrices: Theory and Computation"\n           ISBN 978-0-898716-46-7\n\n    .. [3] Nicholas J. Higham and Lijing lin (2011)\n           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."\n           SIAM Journal on Matrix Analysis and Applications,\n           32 (3). pp. 1056-1078. ISSN 0895-4798\n\n    ')
    
    # Assigning a Call to a Name (line 758):
    
    # Assigning a Call to a Name (line 758):
    
    # Call to asarray(...): (line 758)
    # Processing the call arguments (line 758)
    # Getting the type of 'T' (line 758)
    T_34521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 19), 'T', False)
    # Processing the call keyword arguments (line 758)
    kwargs_34522 = {}
    # Getting the type of 'np' (line 758)
    np_34519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 758)
    asarray_34520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 8), np_34519, 'asarray')
    # Calling asarray(args, kwargs) (line 758)
    asarray_call_result_34523 = invoke(stypy.reporting.localization.Localization(__file__, 758, 8), asarray_34520, *[T_34521], **kwargs_34522)
    
    # Assigning a type to the variable 'T' (line 758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 4), 'T', asarray_call_result_34523)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 759)
    # Processing the call arguments (line 759)
    # Getting the type of 'T' (line 759)
    T_34525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 11), 'T', False)
    # Obtaining the member 'shape' of a type (line 759)
    shape_34526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 11), T_34525, 'shape')
    # Processing the call keyword arguments (line 759)
    kwargs_34527 = {}
    # Getting the type of 'len' (line 759)
    len_34524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 7), 'len', False)
    # Calling len(args, kwargs) (line 759)
    len_call_result_34528 = invoke(stypy.reporting.localization.Localization(__file__, 759, 7), len_34524, *[shape_34526], **kwargs_34527)
    
    int_34529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 23), 'int')
    # Applying the binary operator '!=' (line 759)
    result_ne_34530 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 7), '!=', len_call_result_34528, int_34529)
    
    
    
    # Obtaining the type of the subscript
    int_34531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 36), 'int')
    # Getting the type of 'T' (line 759)
    T_34532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 28), 'T')
    # Obtaining the member 'shape' of a type (line 759)
    shape_34533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 28), T_34532, 'shape')
    # Obtaining the member '__getitem__' of a type (line 759)
    getitem___34534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 28), shape_34533, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 759)
    subscript_call_result_34535 = invoke(stypy.reporting.localization.Localization(__file__, 759, 28), getitem___34534, int_34531)
    
    
    # Obtaining the type of the subscript
    int_34536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 50), 'int')
    # Getting the type of 'T' (line 759)
    T_34537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 42), 'T')
    # Obtaining the member 'shape' of a type (line 759)
    shape_34538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 42), T_34537, 'shape')
    # Obtaining the member '__getitem__' of a type (line 759)
    getitem___34539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 42), shape_34538, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 759)
    subscript_call_result_34540 = invoke(stypy.reporting.localization.Localization(__file__, 759, 42), getitem___34539, int_34536)
    
    # Applying the binary operator '!=' (line 759)
    result_ne_34541 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 28), '!=', subscript_call_result_34535, subscript_call_result_34540)
    
    # Applying the binary operator 'or' (line 759)
    result_or_keyword_34542 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 7), 'or', result_ne_34530, result_ne_34541)
    
    # Testing the type of an if condition (line 759)
    if_condition_34543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 759, 4), result_or_keyword_34542)
    # Assigning a type to the variable 'if_condition_34543' (line 759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 4), 'if_condition_34543', if_condition_34543)
    # SSA begins for if statement (line 759)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 760)
    # Processing the call arguments (line 760)
    str_34545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 25), 'str', 'expected an upper triangular square matrix')
    # Processing the call keyword arguments (line 760)
    kwargs_34546 = {}
    # Getting the type of 'ValueError' (line 760)
    ValueError_34544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 760)
    ValueError_call_result_34547 = invoke(stypy.reporting.localization.Localization(__file__, 760, 14), ValueError_34544, *[str_34545], **kwargs_34546)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 760, 8), ValueError_call_result_34547, 'raise parameter', BaseException)
    # SSA join for if statement (line 759)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 761):
    
    # Assigning a Subscript to a Name (line 761):
    
    # Obtaining the type of the subscript
    int_34548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 4), 'int')
    # Getting the type of 'T' (line 761)
    T_34549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 11), 'T')
    # Obtaining the member 'shape' of a type (line 761)
    shape_34550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 11), T_34549, 'shape')
    # Obtaining the member '__getitem__' of a type (line 761)
    getitem___34551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 4), shape_34550, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 761)
    subscript_call_result_34552 = invoke(stypy.reporting.localization.Localization(__file__, 761, 4), getitem___34551, int_34548)
    
    # Assigning a type to the variable 'tuple_var_assignment_32857' (line 761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 4), 'tuple_var_assignment_32857', subscript_call_result_34552)
    
    # Assigning a Subscript to a Name (line 761):
    
    # Obtaining the type of the subscript
    int_34553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 4), 'int')
    # Getting the type of 'T' (line 761)
    T_34554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 11), 'T')
    # Obtaining the member 'shape' of a type (line 761)
    shape_34555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 11), T_34554, 'shape')
    # Obtaining the member '__getitem__' of a type (line 761)
    getitem___34556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 4), shape_34555, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 761)
    subscript_call_result_34557 = invoke(stypy.reporting.localization.Localization(__file__, 761, 4), getitem___34556, int_34553)
    
    # Assigning a type to the variable 'tuple_var_assignment_32858' (line 761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 4), 'tuple_var_assignment_32858', subscript_call_result_34557)
    
    # Assigning a Name to a Name (line 761):
    # Getting the type of 'tuple_var_assignment_32857' (line 761)
    tuple_var_assignment_32857_34558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 4), 'tuple_var_assignment_32857')
    # Assigning a type to the variable 'n' (line 761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 4), 'n', tuple_var_assignment_32857_34558)
    
    # Assigning a Name to a Name (line 761):
    # Getting the type of 'tuple_var_assignment_32858' (line 761)
    tuple_var_assignment_32858_34559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 4), 'tuple_var_assignment_32858')
    # Assigning a type to the variable 'n' (line 761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 7), 'n', tuple_var_assignment_32858_34559)
    
    # Assigning a Call to a Name (line 765):
    
    # Assigning a Call to a Name (line 765):
    
    # Call to diag(...): (line 765)
    # Processing the call arguments (line 765)
    # Getting the type of 'T' (line 765)
    T_34562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 21), 'T', False)
    # Processing the call keyword arguments (line 765)
    kwargs_34563 = {}
    # Getting the type of 'np' (line 765)
    np_34560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 13), 'np', False)
    # Obtaining the member 'diag' of a type (line 765)
    diag_34561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 13), np_34560, 'diag')
    # Calling diag(args, kwargs) (line 765)
    diag_call_result_34564 = invoke(stypy.reporting.localization.Localization(__file__, 765, 13), diag_34561, *[T_34562], **kwargs_34563)
    
    # Assigning a type to the variable 'T_diag' (line 765)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 4), 'T_diag', diag_call_result_34564)
    
    # Assigning a BoolOp to a Name (line 766):
    
    # Assigning a BoolOp to a Name (line 766):
    
    # Evaluating a boolean operation
    
    # Call to isrealobj(...): (line 766)
    # Processing the call arguments (line 766)
    # Getting the type of 'T' (line 766)
    T_34567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 32), 'T', False)
    # Processing the call keyword arguments (line 766)
    kwargs_34568 = {}
    # Getting the type of 'np' (line 766)
    np_34565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 19), 'np', False)
    # Obtaining the member 'isrealobj' of a type (line 766)
    isrealobj_34566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 19), np_34565, 'isrealobj')
    # Calling isrealobj(args, kwargs) (line 766)
    isrealobj_call_result_34569 = invoke(stypy.reporting.localization.Localization(__file__, 766, 19), isrealobj_34566, *[T_34567], **kwargs_34568)
    
    
    
    # Call to min(...): (line 766)
    # Processing the call arguments (line 766)
    # Getting the type of 'T_diag' (line 766)
    T_diag_34572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 46), 'T_diag', False)
    # Processing the call keyword arguments (line 766)
    kwargs_34573 = {}
    # Getting the type of 'np' (line 766)
    np_34570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 39), 'np', False)
    # Obtaining the member 'min' of a type (line 766)
    min_34571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 39), np_34570, 'min')
    # Calling min(args, kwargs) (line 766)
    min_call_result_34574 = invoke(stypy.reporting.localization.Localization(__file__, 766, 39), min_34571, *[T_diag_34572], **kwargs_34573)
    
    int_34575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 57), 'int')
    # Applying the binary operator '>=' (line 766)
    result_ge_34576 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 39), '>=', min_call_result_34574, int_34575)
    
    # Applying the binary operator 'and' (line 766)
    result_and_keyword_34577 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 19), 'and', isrealobj_call_result_34569, result_ge_34576)
    
    # Assigning a type to the variable 'keep_it_real' (line 766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 4), 'keep_it_real', result_and_keyword_34577)
    
    # Getting the type of 'keep_it_real' (line 767)
    keep_it_real_34578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 7), 'keep_it_real')
    # Testing the type of an if condition (line 767)
    if_condition_34579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 767, 4), keep_it_real_34578)
    # Assigning a type to the variable 'if_condition_34579' (line 767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 4), 'if_condition_34579', if_condition_34579)
    # SSA begins for if statement (line 767)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 768):
    
    # Assigning a Name to a Name (line 768):
    # Getting the type of 'T' (line 768)
    T_34580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 13), 'T')
    # Assigning a type to the variable 'T0' (line 768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'T0', T_34580)
    # SSA branch for the else part of an if statement (line 767)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 770):
    
    # Assigning a Call to a Name (line 770):
    
    # Call to astype(...): (line 770)
    # Processing the call arguments (line 770)
    # Getting the type of 'complex' (line 770)
    complex_34583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 22), 'complex', False)
    # Processing the call keyword arguments (line 770)
    kwargs_34584 = {}
    # Getting the type of 'T' (line 770)
    T_34581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 13), 'T', False)
    # Obtaining the member 'astype' of a type (line 770)
    astype_34582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 13), T_34581, 'astype')
    # Calling astype(args, kwargs) (line 770)
    astype_call_result_34585 = invoke(stypy.reporting.localization.Localization(__file__, 770, 13), astype_34582, *[complex_34583], **kwargs_34584)
    
    # Assigning a type to the variable 'T0' (line 770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 8), 'T0', astype_call_result_34585)
    # SSA join for if statement (line 767)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Name (line 773):
    
    # Assigning a Tuple to a Name (line 773):
    
    # Obtaining an instance of the builtin type 'tuple' (line 773)
    tuple_34586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 773)
    # Adding element type (line 773)
    # Getting the type of 'None' (line 773)
    None_34587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 13), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, None_34587)
    # Adding element type (line 773)
    float_34588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 12), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34588)
    # Adding element type (line 773)
    float_34589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34589)
    # Adding element type (line 773)
    float_34590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34590)
    # Adding element type (line 773)
    float_34591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34591)
    # Adding element type (line 773)
    float_34592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 12), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34592)
    # Adding element type (line 773)
    float_34593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34593)
    # Adding element type (line 773)
    float_34594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34594)
    # Adding element type (line 773)
    float_34595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34595)
    # Adding element type (line 773)
    float_34596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 12), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34596)
    # Adding element type (line 773)
    float_34597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34597)
    # Adding element type (line 773)
    float_34598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34598)
    # Adding element type (line 773)
    float_34599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34599)
    # Adding element type (line 773)
    float_34600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 12), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34600)
    # Adding element type (line 773)
    float_34601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34601)
    # Adding element type (line 773)
    float_34602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34602)
    # Adding element type (line 773)
    float_34603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 13), tuple_34586, float_34603)
    
    # Assigning a type to the variable 'theta' (line 773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 4), 'theta', tuple_34586)
    
    # Assigning a Call to a Tuple (line 779):
    
    # Assigning a Subscript to a Name (line 779):
    
    # Obtaining the type of the subscript
    int_34604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 4), 'int')
    
    # Call to _inverse_squaring_helper(...): (line 779)
    # Processing the call arguments (line 779)
    # Getting the type of 'T0' (line 779)
    T0_34606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 39), 'T0', False)
    # Getting the type of 'theta' (line 779)
    theta_34607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 43), 'theta', False)
    # Processing the call keyword arguments (line 779)
    kwargs_34608 = {}
    # Getting the type of '_inverse_squaring_helper' (line 779)
    _inverse_squaring_helper_34605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 14), '_inverse_squaring_helper', False)
    # Calling _inverse_squaring_helper(args, kwargs) (line 779)
    _inverse_squaring_helper_call_result_34609 = invoke(stypy.reporting.localization.Localization(__file__, 779, 14), _inverse_squaring_helper_34605, *[T0_34606, theta_34607], **kwargs_34608)
    
    # Obtaining the member '__getitem__' of a type (line 779)
    getitem___34610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 4), _inverse_squaring_helper_call_result_34609, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 779)
    subscript_call_result_34611 = invoke(stypy.reporting.localization.Localization(__file__, 779, 4), getitem___34610, int_34604)
    
    # Assigning a type to the variable 'tuple_var_assignment_32859' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'tuple_var_assignment_32859', subscript_call_result_34611)
    
    # Assigning a Subscript to a Name (line 779):
    
    # Obtaining the type of the subscript
    int_34612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 4), 'int')
    
    # Call to _inverse_squaring_helper(...): (line 779)
    # Processing the call arguments (line 779)
    # Getting the type of 'T0' (line 779)
    T0_34614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 39), 'T0', False)
    # Getting the type of 'theta' (line 779)
    theta_34615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 43), 'theta', False)
    # Processing the call keyword arguments (line 779)
    kwargs_34616 = {}
    # Getting the type of '_inverse_squaring_helper' (line 779)
    _inverse_squaring_helper_34613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 14), '_inverse_squaring_helper', False)
    # Calling _inverse_squaring_helper(args, kwargs) (line 779)
    _inverse_squaring_helper_call_result_34617 = invoke(stypy.reporting.localization.Localization(__file__, 779, 14), _inverse_squaring_helper_34613, *[T0_34614, theta_34615], **kwargs_34616)
    
    # Obtaining the member '__getitem__' of a type (line 779)
    getitem___34618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 4), _inverse_squaring_helper_call_result_34617, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 779)
    subscript_call_result_34619 = invoke(stypy.reporting.localization.Localization(__file__, 779, 4), getitem___34618, int_34612)
    
    # Assigning a type to the variable 'tuple_var_assignment_32860' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'tuple_var_assignment_32860', subscript_call_result_34619)
    
    # Assigning a Subscript to a Name (line 779):
    
    # Obtaining the type of the subscript
    int_34620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 4), 'int')
    
    # Call to _inverse_squaring_helper(...): (line 779)
    # Processing the call arguments (line 779)
    # Getting the type of 'T0' (line 779)
    T0_34622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 39), 'T0', False)
    # Getting the type of 'theta' (line 779)
    theta_34623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 43), 'theta', False)
    # Processing the call keyword arguments (line 779)
    kwargs_34624 = {}
    # Getting the type of '_inverse_squaring_helper' (line 779)
    _inverse_squaring_helper_34621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 14), '_inverse_squaring_helper', False)
    # Calling _inverse_squaring_helper(args, kwargs) (line 779)
    _inverse_squaring_helper_call_result_34625 = invoke(stypy.reporting.localization.Localization(__file__, 779, 14), _inverse_squaring_helper_34621, *[T0_34622, theta_34623], **kwargs_34624)
    
    # Obtaining the member '__getitem__' of a type (line 779)
    getitem___34626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 4), _inverse_squaring_helper_call_result_34625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 779)
    subscript_call_result_34627 = invoke(stypy.reporting.localization.Localization(__file__, 779, 4), getitem___34626, int_34620)
    
    # Assigning a type to the variable 'tuple_var_assignment_32861' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'tuple_var_assignment_32861', subscript_call_result_34627)
    
    # Assigning a Name to a Name (line 779):
    # Getting the type of 'tuple_var_assignment_32859' (line 779)
    tuple_var_assignment_32859_34628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'tuple_var_assignment_32859')
    # Assigning a type to the variable 'R' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'R', tuple_var_assignment_32859_34628)
    
    # Assigning a Name to a Name (line 779):
    # Getting the type of 'tuple_var_assignment_32860' (line 779)
    tuple_var_assignment_32860_34629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'tuple_var_assignment_32860')
    # Assigning a type to the variable 's' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 7), 's', tuple_var_assignment_32860_34629)
    
    # Assigning a Name to a Name (line 779):
    # Getting the type of 'tuple_var_assignment_32861' (line 779)
    tuple_var_assignment_32861_34630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'tuple_var_assignment_32861')
    # Assigning a type to the variable 'm' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 10), 'm', tuple_var_assignment_32861_34630)
    
    # Assigning a Call to a Tuple (line 786):
    
    # Assigning a Subscript to a Name (line 786):
    
    # Obtaining the type of the subscript
    int_34631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 4), 'int')
    
    # Call to p_roots(...): (line 786)
    # Processing the call arguments (line 786)
    # Getting the type of 'm' (line 786)
    m_34635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 43), 'm', False)
    # Processing the call keyword arguments (line 786)
    kwargs_34636 = {}
    # Getting the type of 'scipy' (line 786)
    scipy_34632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 21), 'scipy', False)
    # Obtaining the member 'special' of a type (line 786)
    special_34633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 21), scipy_34632, 'special')
    # Obtaining the member 'p_roots' of a type (line 786)
    p_roots_34634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 21), special_34633, 'p_roots')
    # Calling p_roots(args, kwargs) (line 786)
    p_roots_call_result_34637 = invoke(stypy.reporting.localization.Localization(__file__, 786, 21), p_roots_34634, *[m_34635], **kwargs_34636)
    
    # Obtaining the member '__getitem__' of a type (line 786)
    getitem___34638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 4), p_roots_call_result_34637, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 786)
    subscript_call_result_34639 = invoke(stypy.reporting.localization.Localization(__file__, 786, 4), getitem___34638, int_34631)
    
    # Assigning a type to the variable 'tuple_var_assignment_32862' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 4), 'tuple_var_assignment_32862', subscript_call_result_34639)
    
    # Assigning a Subscript to a Name (line 786):
    
    # Obtaining the type of the subscript
    int_34640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 4), 'int')
    
    # Call to p_roots(...): (line 786)
    # Processing the call arguments (line 786)
    # Getting the type of 'm' (line 786)
    m_34644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 43), 'm', False)
    # Processing the call keyword arguments (line 786)
    kwargs_34645 = {}
    # Getting the type of 'scipy' (line 786)
    scipy_34641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 21), 'scipy', False)
    # Obtaining the member 'special' of a type (line 786)
    special_34642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 21), scipy_34641, 'special')
    # Obtaining the member 'p_roots' of a type (line 786)
    p_roots_34643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 21), special_34642, 'p_roots')
    # Calling p_roots(args, kwargs) (line 786)
    p_roots_call_result_34646 = invoke(stypy.reporting.localization.Localization(__file__, 786, 21), p_roots_34643, *[m_34644], **kwargs_34645)
    
    # Obtaining the member '__getitem__' of a type (line 786)
    getitem___34647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 4), p_roots_call_result_34646, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 786)
    subscript_call_result_34648 = invoke(stypy.reporting.localization.Localization(__file__, 786, 4), getitem___34647, int_34640)
    
    # Assigning a type to the variable 'tuple_var_assignment_32863' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 4), 'tuple_var_assignment_32863', subscript_call_result_34648)
    
    # Assigning a Name to a Name (line 786):
    # Getting the type of 'tuple_var_assignment_32862' (line 786)
    tuple_var_assignment_32862_34649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 4), 'tuple_var_assignment_32862')
    # Assigning a type to the variable 'nodes' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 4), 'nodes', tuple_var_assignment_32862_34649)
    
    # Assigning a Name to a Name (line 786):
    # Getting the type of 'tuple_var_assignment_32863' (line 786)
    tuple_var_assignment_32863_34650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 4), 'tuple_var_assignment_32863')
    # Assigning a type to the variable 'weights' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 11), 'weights', tuple_var_assignment_32863_34650)
    
    # Assigning a Attribute to a Name (line 787):
    
    # Assigning a Attribute to a Name (line 787):
    # Getting the type of 'nodes' (line 787)
    nodes_34651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 12), 'nodes')
    # Obtaining the member 'real' of a type (line 787)
    real_34652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 12), nodes_34651, 'real')
    # Assigning a type to the variable 'nodes' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'nodes', real_34652)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'nodes' (line 788)
    nodes_34653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 7), 'nodes')
    # Obtaining the member 'shape' of a type (line 788)
    shape_34654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 7), nodes_34653, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 788)
    tuple_34655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 788)
    # Adding element type (line 788)
    # Getting the type of 'm' (line 788)
    m_34656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 23), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 23), tuple_34655, m_34656)
    
    # Applying the binary operator '!=' (line 788)
    result_ne_34657 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 7), '!=', shape_34654, tuple_34655)
    
    
    # Getting the type of 'weights' (line 788)
    weights_34658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 30), 'weights')
    # Obtaining the member 'shape' of a type (line 788)
    shape_34659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 30), weights_34658, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 788)
    tuple_34660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 788)
    # Adding element type (line 788)
    # Getting the type of 'm' (line 788)
    m_34661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 48), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 48), tuple_34660, m_34661)
    
    # Applying the binary operator '!=' (line 788)
    result_ne_34662 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 30), '!=', shape_34659, tuple_34660)
    
    # Applying the binary operator 'or' (line 788)
    result_or_keyword_34663 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 7), 'or', result_ne_34657, result_ne_34662)
    
    # Testing the type of an if condition (line 788)
    if_condition_34664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 788, 4), result_or_keyword_34663)
    # Assigning a type to the variable 'if_condition_34664' (line 788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 4), 'if_condition_34664', if_condition_34664)
    # SSA begins for if statement (line 788)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 789)
    # Processing the call arguments (line 789)
    str_34666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 24), 'str', 'internal error')
    # Processing the call keyword arguments (line 789)
    kwargs_34667 = {}
    # Getting the type of 'Exception' (line 789)
    Exception_34665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 789)
    Exception_call_result_34668 = invoke(stypy.reporting.localization.Localization(__file__, 789, 14), Exception_34665, *[str_34666], **kwargs_34667)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 789, 8), Exception_call_result_34668, 'raise parameter', BaseException)
    # SSA join for if statement (line 788)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 790):
    
    # Assigning a BinOp to a Name (line 790):
    float_34669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 12), 'float')
    float_34670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 18), 'float')
    # Getting the type of 'nodes' (line 790)
    nodes_34671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 24), 'nodes')
    # Applying the binary operator '*' (line 790)
    result_mul_34672 = python_operator(stypy.reporting.localization.Localization(__file__, 790, 18), '*', float_34670, nodes_34671)
    
    # Applying the binary operator '+' (line 790)
    result_add_34673 = python_operator(stypy.reporting.localization.Localization(__file__, 790, 12), '+', float_34669, result_mul_34672)
    
    # Assigning a type to the variable 'nodes' (line 790)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 4), 'nodes', result_add_34673)
    
    # Assigning a BinOp to a Name (line 791):
    
    # Assigning a BinOp to a Name (line 791):
    float_34674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 14), 'float')
    # Getting the type of 'weights' (line 791)
    weights_34675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 20), 'weights')
    # Applying the binary operator '*' (line 791)
    result_mul_34676 = python_operator(stypy.reporting.localization.Localization(__file__, 791, 14), '*', float_34674, weights_34675)
    
    # Assigning a type to the variable 'weights' (line 791)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 4), 'weights', result_mul_34676)
    
    # Assigning a Call to a Name (line 792):
    
    # Assigning a Call to a Name (line 792):
    
    # Call to identity(...): (line 792)
    # Processing the call arguments (line 792)
    # Getting the type of 'n' (line 792)
    n_34679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 24), 'n', False)
    # Processing the call keyword arguments (line 792)
    kwargs_34680 = {}
    # Getting the type of 'np' (line 792)
    np_34677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 12), 'np', False)
    # Obtaining the member 'identity' of a type (line 792)
    identity_34678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 12), np_34677, 'identity')
    # Calling identity(args, kwargs) (line 792)
    identity_call_result_34681 = invoke(stypy.reporting.localization.Localization(__file__, 792, 12), identity_34678, *[n_34679], **kwargs_34680)
    
    # Assigning a type to the variable 'ident' (line 792)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 792, 4), 'ident', identity_call_result_34681)
    
    # Assigning a Call to a Name (line 793):
    
    # Assigning a Call to a Name (line 793):
    
    # Call to zeros_like(...): (line 793)
    # Processing the call arguments (line 793)
    # Getting the type of 'R' (line 793)
    R_34684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 22), 'R', False)
    # Processing the call keyword arguments (line 793)
    kwargs_34685 = {}
    # Getting the type of 'np' (line 793)
    np_34682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 8), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 793)
    zeros_like_34683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 8), np_34682, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 793)
    zeros_like_call_result_34686 = invoke(stypy.reporting.localization.Localization(__file__, 793, 8), zeros_like_34683, *[R_34684], **kwargs_34685)
    
    # Assigning a type to the variable 'U' (line 793)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 4), 'U', zeros_like_call_result_34686)
    
    
    # Call to zip(...): (line 794)
    # Processing the call arguments (line 794)
    # Getting the type of 'weights' (line 794)
    weights_34688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 27), 'weights', False)
    # Getting the type of 'nodes' (line 794)
    nodes_34689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 36), 'nodes', False)
    # Processing the call keyword arguments (line 794)
    kwargs_34690 = {}
    # Getting the type of 'zip' (line 794)
    zip_34687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 23), 'zip', False)
    # Calling zip(args, kwargs) (line 794)
    zip_call_result_34691 = invoke(stypy.reporting.localization.Localization(__file__, 794, 23), zip_34687, *[weights_34688, nodes_34689], **kwargs_34690)
    
    # Testing the type of a for loop iterable (line 794)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 794, 4), zip_call_result_34691)
    # Getting the type of the for loop variable (line 794)
    for_loop_var_34692 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 794, 4), zip_call_result_34691)
    # Assigning a type to the variable 'alpha' (line 794)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 4), 'alpha', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 794, 4), for_loop_var_34692))
    # Assigning a type to the variable 'beta' (line 794)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 4), 'beta', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 794, 4), for_loop_var_34692))
    # SSA begins for a for statement (line 794)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'U' (line 795)
    U_34693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 8), 'U')
    
    # Call to solve_triangular(...): (line 795)
    # Processing the call arguments (line 795)
    # Getting the type of 'ident' (line 795)
    ident_34695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 30), 'ident', False)
    # Getting the type of 'beta' (line 795)
    beta_34696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 38), 'beta', False)
    # Getting the type of 'R' (line 795)
    R_34697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 43), 'R', False)
    # Applying the binary operator '*' (line 795)
    result_mul_34698 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 38), '*', beta_34696, R_34697)
    
    # Applying the binary operator '+' (line 795)
    result_add_34699 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 30), '+', ident_34695, result_mul_34698)
    
    # Getting the type of 'alpha' (line 795)
    alpha_34700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 46), 'alpha', False)
    # Getting the type of 'R' (line 795)
    R_34701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 52), 'R', False)
    # Applying the binary operator '*' (line 795)
    result_mul_34702 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 46), '*', alpha_34700, R_34701)
    
    # Processing the call keyword arguments (line 795)
    kwargs_34703 = {}
    # Getting the type of 'solve_triangular' (line 795)
    solve_triangular_34694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 13), 'solve_triangular', False)
    # Calling solve_triangular(args, kwargs) (line 795)
    solve_triangular_call_result_34704 = invoke(stypy.reporting.localization.Localization(__file__, 795, 13), solve_triangular_34694, *[result_add_34699, result_mul_34702], **kwargs_34703)
    
    # Applying the binary operator '+=' (line 795)
    result_iadd_34705 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 8), '+=', U_34693, solve_triangular_call_result_34704)
    # Assigning a type to the variable 'U' (line 795)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 8), 'U', result_iadd_34705)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'U' (line 796)
    U_34706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'U')
    
    # Call to exp2(...): (line 796)
    # Processing the call arguments (line 796)
    # Getting the type of 's' (line 796)
    s_34709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 17), 's', False)
    # Processing the call keyword arguments (line 796)
    kwargs_34710 = {}
    # Getting the type of 'np' (line 796)
    np_34707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 9), 'np', False)
    # Obtaining the member 'exp2' of a type (line 796)
    exp2_34708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 9), np_34707, 'exp2')
    # Calling exp2(args, kwargs) (line 796)
    exp2_call_result_34711 = invoke(stypy.reporting.localization.Localization(__file__, 796, 9), exp2_34708, *[s_34709], **kwargs_34710)
    
    # Applying the binary operator '*=' (line 796)
    result_imul_34712 = python_operator(stypy.reporting.localization.Localization(__file__, 796, 4), '*=', U_34706, exp2_call_result_34711)
    # Assigning a type to the variable 'U' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'U', result_imul_34712)
    
    
    # Assigning a Call to a Name (line 801):
    
    # Assigning a Call to a Name (line 801):
    
    # Call to all(...): (line 801)
    # Processing the call arguments (line 801)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 801, 31, True)
    # Calculating comprehension expression
    
    # Call to diag(...): (line 801)
    # Processing the call arguments (line 801)
    # Getting the type of 'T0' (line 801)
    T0_34725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 74), 'T0', False)
    # Processing the call keyword arguments (line 801)
    kwargs_34726 = {}
    # Getting the type of 'np' (line 801)
    np_34723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 66), 'np', False)
    # Obtaining the member 'diag' of a type (line 801)
    diag_34724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 66), np_34723, 'diag')
    # Calling diag(args, kwargs) (line 801)
    diag_call_result_34727 = invoke(stypy.reporting.localization.Localization(__file__, 801, 66), diag_34724, *[T0_34725], **kwargs_34726)
    
    comprehension_34728 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 801, 31), diag_call_result_34727)
    # Assigning a type to the variable 'x' (line 801)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 31), 'x', comprehension_34728)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'x' (line 801)
    x_34714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 31), 'x', False)
    # Obtaining the member 'real' of a type (line 801)
    real_34715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 31), x_34714, 'real')
    int_34716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 40), 'int')
    # Applying the binary operator '>' (line 801)
    result_gt_34717 = python_operator(stypy.reporting.localization.Localization(__file__, 801, 31), '>', real_34715, int_34716)
    
    
    # Getting the type of 'x' (line 801)
    x_34718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 45), 'x', False)
    # Obtaining the member 'imag' of a type (line 801)
    imag_34719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 45), x_34718, 'imag')
    int_34720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 55), 'int')
    # Applying the binary operator '!=' (line 801)
    result_ne_34721 = python_operator(stypy.reporting.localization.Localization(__file__, 801, 45), '!=', imag_34719, int_34720)
    
    # Applying the binary operator 'or' (line 801)
    result_or_keyword_34722 = python_operator(stypy.reporting.localization.Localization(__file__, 801, 31), 'or', result_gt_34717, result_ne_34721)
    
    list_34729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 31), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 801, 31), list_34729, result_or_keyword_34722)
    # Processing the call keyword arguments (line 801)
    kwargs_34730 = {}
    # Getting the type of 'all' (line 801)
    all_34713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 27), 'all', False)
    # Calling all(args, kwargs) (line 801)
    all_call_result_34731 = invoke(stypy.reporting.localization.Localization(__file__, 801, 27), all_34713, *[list_34729], **kwargs_34730)
    
    # Assigning a type to the variable 'has_principal_branch' (line 801)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 4), 'has_principal_branch', all_call_result_34731)
    
    # Getting the type of 'has_principal_branch' (line 802)
    has_principal_branch_34732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 7), 'has_principal_branch')
    # Testing the type of an if condition (line 802)
    if_condition_34733 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 802, 4), has_principal_branch_34732)
    # Assigning a type to the variable 'if_condition_34733' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 4), 'if_condition_34733', if_condition_34733)
    # SSA begins for if statement (line 802)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 805):
    
    # Assigning a Call to a Subscript (line 805):
    
    # Call to log(...): (line 805)
    # Processing the call arguments (line 805)
    
    # Call to diag(...): (line 805)
    # Processing the call arguments (line 805)
    # Getting the type of 'T0' (line 805)
    T0_34738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 47), 'T0', False)
    # Processing the call keyword arguments (line 805)
    kwargs_34739 = {}
    # Getting the type of 'np' (line 805)
    np_34736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 39), 'np', False)
    # Obtaining the member 'diag' of a type (line 805)
    diag_34737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 39), np_34736, 'diag')
    # Calling diag(args, kwargs) (line 805)
    diag_call_result_34740 = invoke(stypy.reporting.localization.Localization(__file__, 805, 39), diag_34737, *[T0_34738], **kwargs_34739)
    
    # Processing the call keyword arguments (line 805)
    kwargs_34741 = {}
    # Getting the type of 'np' (line 805)
    np_34734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 32), 'np', False)
    # Obtaining the member 'log' of a type (line 805)
    log_34735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 32), np_34734, 'log')
    # Calling log(args, kwargs) (line 805)
    log_call_result_34742 = invoke(stypy.reporting.localization.Localization(__file__, 805, 32), log_34735, *[diag_call_result_34740], **kwargs_34741)
    
    # Getting the type of 'U' (line 805)
    U_34743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 8), 'U')
    
    # Call to diag_indices(...): (line 805)
    # Processing the call arguments (line 805)
    # Getting the type of 'n' (line 805)
    n_34746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 26), 'n', False)
    # Processing the call keyword arguments (line 805)
    kwargs_34747 = {}
    # Getting the type of 'np' (line 805)
    np_34744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 10), 'np', False)
    # Obtaining the member 'diag_indices' of a type (line 805)
    diag_indices_34745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 10), np_34744, 'diag_indices')
    # Calling diag_indices(args, kwargs) (line 805)
    diag_indices_call_result_34748 = invoke(stypy.reporting.localization.Localization(__file__, 805, 10), diag_indices_34745, *[n_34746], **kwargs_34747)
    
    # Storing an element on a container (line 805)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 805, 8), U_34743, (diag_indices_call_result_34748, log_call_result_34742))
    
    
    # Call to range(...): (line 810)
    # Processing the call arguments (line 810)
    # Getting the type of 'n' (line 810)
    n_34750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 23), 'n', False)
    int_34751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 25), 'int')
    # Applying the binary operator '-' (line 810)
    result_sub_34752 = python_operator(stypy.reporting.localization.Localization(__file__, 810, 23), '-', n_34750, int_34751)
    
    # Processing the call keyword arguments (line 810)
    kwargs_34753 = {}
    # Getting the type of 'range' (line 810)
    range_34749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 17), 'range', False)
    # Calling range(args, kwargs) (line 810)
    range_call_result_34754 = invoke(stypy.reporting.localization.Localization(__file__, 810, 17), range_34749, *[result_sub_34752], **kwargs_34753)
    
    # Testing the type of a for loop iterable (line 810)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 810, 8), range_call_result_34754)
    # Getting the type of the for loop variable (line 810)
    for_loop_var_34755 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 810, 8), range_call_result_34754)
    # Assigning a type to the variable 'i' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'i', for_loop_var_34755)
    # SSA begins for a for statement (line 810)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 811):
    
    # Assigning a Subscript to a Name (line 811):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 811)
    tuple_34756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 811)
    # Adding element type (line 811)
    # Getting the type of 'i' (line 811)
    i_34757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 20), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 811, 20), tuple_34756, i_34757)
    # Adding element type (line 811)
    # Getting the type of 'i' (line 811)
    i_34758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 23), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 811, 20), tuple_34756, i_34758)
    
    # Getting the type of 'T0' (line 811)
    T0_34759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 17), 'T0')
    # Obtaining the member '__getitem__' of a type (line 811)
    getitem___34760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 17), T0_34759, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 811)
    subscript_call_result_34761 = invoke(stypy.reporting.localization.Localization(__file__, 811, 17), getitem___34760, tuple_34756)
    
    # Assigning a type to the variable 'l1' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 12), 'l1', subscript_call_result_34761)
    
    # Assigning a Subscript to a Name (line 812):
    
    # Assigning a Subscript to a Name (line 812):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 812)
    tuple_34762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 812)
    # Adding element type (line 812)
    # Getting the type of 'i' (line 812)
    i_34763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 20), 'i')
    int_34764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 22), 'int')
    # Applying the binary operator '+' (line 812)
    result_add_34765 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 20), '+', i_34763, int_34764)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 812, 20), tuple_34762, result_add_34765)
    # Adding element type (line 812)
    # Getting the type of 'i' (line 812)
    i_34766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 25), 'i')
    int_34767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 27), 'int')
    # Applying the binary operator '+' (line 812)
    result_add_34768 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 25), '+', i_34766, int_34767)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 812, 20), tuple_34762, result_add_34768)
    
    # Getting the type of 'T0' (line 812)
    T0_34769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 17), 'T0')
    # Obtaining the member '__getitem__' of a type (line 812)
    getitem___34770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 17), T0_34769, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 812)
    subscript_call_result_34771 = invoke(stypy.reporting.localization.Localization(__file__, 812, 17), getitem___34770, tuple_34762)
    
    # Assigning a type to the variable 'l2' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 12), 'l2', subscript_call_result_34771)
    
    # Assigning a Subscript to a Name (line 813):
    
    # Assigning a Subscript to a Name (line 813):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 813)
    tuple_34772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 813)
    # Adding element type (line 813)
    # Getting the type of 'i' (line 813)
    i_34773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 21), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 21), tuple_34772, i_34773)
    # Adding element type (line 813)
    # Getting the type of 'i' (line 813)
    i_34774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 24), 'i')
    int_34775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 26), 'int')
    # Applying the binary operator '+' (line 813)
    result_add_34776 = python_operator(stypy.reporting.localization.Localization(__file__, 813, 24), '+', i_34774, int_34775)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 21), tuple_34772, result_add_34776)
    
    # Getting the type of 'T0' (line 813)
    T0_34777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 18), 'T0')
    # Obtaining the member '__getitem__' of a type (line 813)
    getitem___34778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 18), T0_34777, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 813)
    subscript_call_result_34779 = invoke(stypy.reporting.localization.Localization(__file__, 813, 18), getitem___34778, tuple_34772)
    
    # Assigning a type to the variable 't12' (line 813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 12), 't12', subscript_call_result_34779)
    
    # Assigning a Call to a Subscript (line 814):
    
    # Assigning a Call to a Subscript (line 814):
    
    # Call to _logm_superdiag_entry(...): (line 814)
    # Processing the call arguments (line 814)
    # Getting the type of 'l1' (line 814)
    l1_34781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 46), 'l1', False)
    # Getting the type of 'l2' (line 814)
    l2_34782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 50), 'l2', False)
    # Getting the type of 't12' (line 814)
    t12_34783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 54), 't12', False)
    # Processing the call keyword arguments (line 814)
    kwargs_34784 = {}
    # Getting the type of '_logm_superdiag_entry' (line 814)
    _logm_superdiag_entry_34780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 24), '_logm_superdiag_entry', False)
    # Calling _logm_superdiag_entry(args, kwargs) (line 814)
    _logm_superdiag_entry_call_result_34785 = invoke(stypy.reporting.localization.Localization(__file__, 814, 24), _logm_superdiag_entry_34780, *[l1_34781, l2_34782, t12_34783], **kwargs_34784)
    
    # Getting the type of 'U' (line 814)
    U_34786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 12), 'U')
    
    # Obtaining an instance of the builtin type 'tuple' (line 814)
    tuple_34787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 814)
    # Adding element type (line 814)
    # Getting the type of 'i' (line 814)
    i_34788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 14), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 814, 14), tuple_34787, i_34788)
    # Adding element type (line 814)
    # Getting the type of 'i' (line 814)
    i_34789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 17), 'i')
    int_34790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 19), 'int')
    # Applying the binary operator '+' (line 814)
    result_add_34791 = python_operator(stypy.reporting.localization.Localization(__file__, 814, 17), '+', i_34789, int_34790)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 814, 14), tuple_34787, result_add_34791)
    
    # Storing an element on a container (line 814)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 814, 12), U_34786, (tuple_34787, _logm_superdiag_entry_call_result_34785))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 802)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to array_equal(...): (line 817)
    # Processing the call arguments (line 817)
    # Getting the type of 'U' (line 817)
    U_34794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 26), 'U', False)
    
    # Call to triu(...): (line 817)
    # Processing the call arguments (line 817)
    # Getting the type of 'U' (line 817)
    U_34797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 37), 'U', False)
    # Processing the call keyword arguments (line 817)
    kwargs_34798 = {}
    # Getting the type of 'np' (line 817)
    np_34795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 29), 'np', False)
    # Obtaining the member 'triu' of a type (line 817)
    triu_34796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 29), np_34795, 'triu')
    # Calling triu(args, kwargs) (line 817)
    triu_call_result_34799 = invoke(stypy.reporting.localization.Localization(__file__, 817, 29), triu_34796, *[U_34797], **kwargs_34798)
    
    # Processing the call keyword arguments (line 817)
    kwargs_34800 = {}
    # Getting the type of 'np' (line 817)
    np_34792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 11), 'np', False)
    # Obtaining the member 'array_equal' of a type (line 817)
    array_equal_34793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 11), np_34792, 'array_equal')
    # Calling array_equal(args, kwargs) (line 817)
    array_equal_call_result_34801 = invoke(stypy.reporting.localization.Localization(__file__, 817, 11), array_equal_34793, *[U_34794, triu_call_result_34799], **kwargs_34800)
    
    # Applying the 'not' unary operator (line 817)
    result_not__34802 = python_operator(stypy.reporting.localization.Localization(__file__, 817, 7), 'not', array_equal_call_result_34801)
    
    # Testing the type of an if condition (line 817)
    if_condition_34803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 817, 4), result_not__34802)
    # Assigning a type to the variable 'if_condition_34803' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 4), 'if_condition_34803', if_condition_34803)
    # SSA begins for if statement (line 817)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 818)
    # Processing the call arguments (line 818)
    str_34805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 24), 'str', 'internal inconsistency')
    # Processing the call keyword arguments (line 818)
    kwargs_34806 = {}
    # Getting the type of 'Exception' (line 818)
    Exception_34804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 818)
    Exception_call_result_34807 = invoke(stypy.reporting.localization.Localization(__file__, 818, 14), Exception_34804, *[str_34805], **kwargs_34806)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 818, 8), Exception_call_result_34807, 'raise parameter', BaseException)
    # SSA join for if statement (line 817)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'U' (line 819)
    U_34808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 11), 'U')
    # Assigning a type to the variable 'stypy_return_type' (line 819)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 4), 'stypy_return_type', U_34808)
    
    # ################# End of '_logm_triu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_logm_triu' in the type store
    # Getting the type of 'stypy_return_type' (line 723)
    stypy_return_type_34809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34809)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_logm_triu'
    return stypy_return_type_34809

# Assigning a type to the variable '_logm_triu' (line 723)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 0), '_logm_triu', _logm_triu)

@norecursion
def _logm_force_nonsingular_triangular_matrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 822)
    False_34810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 57), 'False')
    defaults = [False_34810]
    # Create a new context for function '_logm_force_nonsingular_triangular_matrix'
    module_type_store = module_type_store.open_function_context('_logm_force_nonsingular_triangular_matrix', 822, 0, False)
    
    # Passed parameters checking function
    _logm_force_nonsingular_triangular_matrix.stypy_localization = localization
    _logm_force_nonsingular_triangular_matrix.stypy_type_of_self = None
    _logm_force_nonsingular_triangular_matrix.stypy_type_store = module_type_store
    _logm_force_nonsingular_triangular_matrix.stypy_function_name = '_logm_force_nonsingular_triangular_matrix'
    _logm_force_nonsingular_triangular_matrix.stypy_param_names_list = ['T', 'inplace']
    _logm_force_nonsingular_triangular_matrix.stypy_varargs_param_name = None
    _logm_force_nonsingular_triangular_matrix.stypy_kwargs_param_name = None
    _logm_force_nonsingular_triangular_matrix.stypy_call_defaults = defaults
    _logm_force_nonsingular_triangular_matrix.stypy_call_varargs = varargs
    _logm_force_nonsingular_triangular_matrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_logm_force_nonsingular_triangular_matrix', ['T', 'inplace'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_logm_force_nonsingular_triangular_matrix', localization, ['T', 'inplace'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_logm_force_nonsingular_triangular_matrix(...)' code ##################

    
    # Assigning a Num to a Name (line 825):
    
    # Assigning a Num to a Name (line 825):
    float_34811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 14), 'float')
    # Assigning a type to the variable 'tri_eps' (line 825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 4), 'tri_eps', float_34811)
    
    # Assigning a Call to a Name (line 826):
    
    # Assigning a Call to a Name (line 826):
    
    # Call to absolute(...): (line 826)
    # Processing the call arguments (line 826)
    
    # Call to diag(...): (line 826)
    # Processing the call arguments (line 826)
    # Getting the type of 'T' (line 826)
    T_34816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 35), 'T', False)
    # Processing the call keyword arguments (line 826)
    kwargs_34817 = {}
    # Getting the type of 'np' (line 826)
    np_34814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 27), 'np', False)
    # Obtaining the member 'diag' of a type (line 826)
    diag_34815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 27), np_34814, 'diag')
    # Calling diag(args, kwargs) (line 826)
    diag_call_result_34818 = invoke(stypy.reporting.localization.Localization(__file__, 826, 27), diag_34815, *[T_34816], **kwargs_34817)
    
    # Processing the call keyword arguments (line 826)
    kwargs_34819 = {}
    # Getting the type of 'np' (line 826)
    np_34812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 15), 'np', False)
    # Obtaining the member 'absolute' of a type (line 826)
    absolute_34813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 15), np_34812, 'absolute')
    # Calling absolute(args, kwargs) (line 826)
    absolute_call_result_34820 = invoke(stypy.reporting.localization.Localization(__file__, 826, 15), absolute_34813, *[diag_call_result_34818], **kwargs_34819)
    
    # Assigning a type to the variable 'abs_diag' (line 826)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 4), 'abs_diag', absolute_call_result_34820)
    
    
    # Call to any(...): (line 827)
    # Processing the call arguments (line 827)
    
    # Getting the type of 'abs_diag' (line 827)
    abs_diag_34823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 14), 'abs_diag', False)
    int_34824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 26), 'int')
    # Applying the binary operator '==' (line 827)
    result_eq_34825 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 14), '==', abs_diag_34823, int_34824)
    
    # Processing the call keyword arguments (line 827)
    kwargs_34826 = {}
    # Getting the type of 'np' (line 827)
    np_34821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 827)
    any_34822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 7), np_34821, 'any')
    # Calling any(args, kwargs) (line 827)
    any_call_result_34827 = invoke(stypy.reporting.localization.Localization(__file__, 827, 7), any_34822, *[result_eq_34825], **kwargs_34826)
    
    # Testing the type of an if condition (line 827)
    if_condition_34828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 827, 4), any_call_result_34827)
    # Assigning a type to the variable 'if_condition_34828' (line 827)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 4), 'if_condition_34828', if_condition_34828)
    # SSA begins for if statement (line 827)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 828):
    
    # Assigning a Str to a Name (line 828):
    str_34829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 32), 'str', 'The logm input matrix is exactly singular.')
    # Assigning a type to the variable 'exact_singularity_msg' (line 828)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 8), 'exact_singularity_msg', str_34829)
    
    # Call to warn(...): (line 829)
    # Processing the call arguments (line 829)
    # Getting the type of 'exact_singularity_msg' (line 829)
    exact_singularity_msg_34832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 22), 'exact_singularity_msg', False)
    # Getting the type of 'LogmExactlySingularWarning' (line 829)
    LogmExactlySingularWarning_34833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 45), 'LogmExactlySingularWarning', False)
    # Processing the call keyword arguments (line 829)
    kwargs_34834 = {}
    # Getting the type of 'warnings' (line 829)
    warnings_34830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 829)
    warn_34831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 8), warnings_34830, 'warn')
    # Calling warn(args, kwargs) (line 829)
    warn_call_result_34835 = invoke(stypy.reporting.localization.Localization(__file__, 829, 8), warn_34831, *[exact_singularity_msg_34832, LogmExactlySingularWarning_34833], **kwargs_34834)
    
    
    
    # Getting the type of 'inplace' (line 830)
    inplace_34836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 15), 'inplace')
    # Applying the 'not' unary operator (line 830)
    result_not__34837 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 11), 'not', inplace_34836)
    
    # Testing the type of an if condition (line 830)
    if_condition_34838 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 830, 8), result_not__34837)
    # Assigning a type to the variable 'if_condition_34838' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 8), 'if_condition_34838', if_condition_34838)
    # SSA begins for if statement (line 830)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 831):
    
    # Assigning a Call to a Name (line 831):
    
    # Call to copy(...): (line 831)
    # Processing the call keyword arguments (line 831)
    kwargs_34841 = {}
    # Getting the type of 'T' (line 831)
    T_34839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 16), 'T', False)
    # Obtaining the member 'copy' of a type (line 831)
    copy_34840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 16), T_34839, 'copy')
    # Calling copy(args, kwargs) (line 831)
    copy_call_result_34842 = invoke(stypy.reporting.localization.Localization(__file__, 831, 16), copy_34840, *[], **kwargs_34841)
    
    # Assigning a type to the variable 'T' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 12), 'T', copy_call_result_34842)
    # SSA join for if statement (line 830)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 832):
    
    # Assigning a Subscript to a Name (line 832):
    
    # Obtaining the type of the subscript
    int_34843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 20), 'int')
    # Getting the type of 'T' (line 832)
    T_34844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 12), 'T')
    # Obtaining the member 'shape' of a type (line 832)
    shape_34845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 12), T_34844, 'shape')
    # Obtaining the member '__getitem__' of a type (line 832)
    getitem___34846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 12), shape_34845, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 832)
    subscript_call_result_34847 = invoke(stypy.reporting.localization.Localization(__file__, 832, 12), getitem___34846, int_34843)
    
    # Assigning a type to the variable 'n' (line 832)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 8), 'n', subscript_call_result_34847)
    
    
    # Call to range(...): (line 833)
    # Processing the call arguments (line 833)
    # Getting the type of 'n' (line 833)
    n_34849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 23), 'n', False)
    # Processing the call keyword arguments (line 833)
    kwargs_34850 = {}
    # Getting the type of 'range' (line 833)
    range_34848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 17), 'range', False)
    # Calling range(args, kwargs) (line 833)
    range_call_result_34851 = invoke(stypy.reporting.localization.Localization(__file__, 833, 17), range_34848, *[n_34849], **kwargs_34850)
    
    # Testing the type of a for loop iterable (line 833)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 833, 8), range_call_result_34851)
    # Getting the type of the for loop variable (line 833)
    for_loop_var_34852 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 833, 8), range_call_result_34851)
    # Assigning a type to the variable 'i' (line 833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 'i', for_loop_var_34852)
    # SSA begins for a for statement (line 833)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 834)
    tuple_34853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 834)
    # Adding element type (line 834)
    # Getting the type of 'i' (line 834)
    i_34854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 21), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 834, 21), tuple_34853, i_34854)
    # Adding element type (line 834)
    # Getting the type of 'i' (line 834)
    i_34855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 24), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 834, 21), tuple_34853, i_34855)
    
    # Getting the type of 'T' (line 834)
    T_34856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 19), 'T')
    # Obtaining the member '__getitem__' of a type (line 834)
    getitem___34857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 19), T_34856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 834)
    subscript_call_result_34858 = invoke(stypy.reporting.localization.Localization(__file__, 834, 19), getitem___34857, tuple_34853)
    
    # Applying the 'not' unary operator (line 834)
    result_not__34859 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 15), 'not', subscript_call_result_34858)
    
    # Testing the type of an if condition (line 834)
    if_condition_34860 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 834, 12), result_not__34859)
    # Assigning a type to the variable 'if_condition_34860' (line 834)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 12), 'if_condition_34860', if_condition_34860)
    # SSA begins for if statement (line 834)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 835):
    
    # Assigning a Name to a Subscript (line 835):
    # Getting the type of 'tri_eps' (line 835)
    tri_eps_34861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 26), 'tri_eps')
    # Getting the type of 'T' (line 835)
    T_34862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 16), 'T')
    
    # Obtaining an instance of the builtin type 'tuple' (line 835)
    tuple_34863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 835)
    # Adding element type (line 835)
    # Getting the type of 'i' (line 835)
    i_34864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 18), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 835, 18), tuple_34863, i_34864)
    # Adding element type (line 835)
    # Getting the type of 'i' (line 835)
    i_34865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 21), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 835, 18), tuple_34863, i_34865)
    
    # Storing an element on a container (line 835)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 835, 16), T_34862, (tuple_34863, tri_eps_34861))
    # SSA join for if statement (line 834)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 827)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to any(...): (line 836)
    # Processing the call arguments (line 836)
    
    # Getting the type of 'abs_diag' (line 836)
    abs_diag_34868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 16), 'abs_diag', False)
    # Getting the type of 'tri_eps' (line 836)
    tri_eps_34869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 27), 'tri_eps', False)
    # Applying the binary operator '<' (line 836)
    result_lt_34870 = python_operator(stypy.reporting.localization.Localization(__file__, 836, 16), '<', abs_diag_34868, tri_eps_34869)
    
    # Processing the call keyword arguments (line 836)
    kwargs_34871 = {}
    # Getting the type of 'np' (line 836)
    np_34866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 9), 'np', False)
    # Obtaining the member 'any' of a type (line 836)
    any_34867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 836, 9), np_34866, 'any')
    # Calling any(args, kwargs) (line 836)
    any_call_result_34872 = invoke(stypy.reporting.localization.Localization(__file__, 836, 9), any_34867, *[result_lt_34870], **kwargs_34871)
    
    # Testing the type of an if condition (line 836)
    if_condition_34873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 836, 9), any_call_result_34872)
    # Assigning a type to the variable 'if_condition_34873' (line 836)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 9), 'if_condition_34873', if_condition_34873)
    # SSA begins for if statement (line 836)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 837):
    
    # Assigning a Str to a Name (line 837):
    str_34874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 31), 'str', 'The logm input matrix may be nearly singular.')
    # Assigning a type to the variable 'near_singularity_msg' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 8), 'near_singularity_msg', str_34874)
    
    # Call to warn(...): (line 838)
    # Processing the call arguments (line 838)
    # Getting the type of 'near_singularity_msg' (line 838)
    near_singularity_msg_34877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 22), 'near_singularity_msg', False)
    # Getting the type of 'LogmNearlySingularWarning' (line 838)
    LogmNearlySingularWarning_34878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 44), 'LogmNearlySingularWarning', False)
    # Processing the call keyword arguments (line 838)
    kwargs_34879 = {}
    # Getting the type of 'warnings' (line 838)
    warnings_34875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 838)
    warn_34876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 8), warnings_34875, 'warn')
    # Calling warn(args, kwargs) (line 838)
    warn_call_result_34880 = invoke(stypy.reporting.localization.Localization(__file__, 838, 8), warn_34876, *[near_singularity_msg_34877, LogmNearlySingularWarning_34878], **kwargs_34879)
    
    # SSA join for if statement (line 836)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 827)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'T' (line 839)
    T_34881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 11), 'T')
    # Assigning a type to the variable 'stypy_return_type' (line 839)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 4), 'stypy_return_type', T_34881)
    
    # ################# End of '_logm_force_nonsingular_triangular_matrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_logm_force_nonsingular_triangular_matrix' in the type store
    # Getting the type of 'stypy_return_type' (line 822)
    stypy_return_type_34882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34882)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_logm_force_nonsingular_triangular_matrix'
    return stypy_return_type_34882

# Assigning a type to the variable '_logm_force_nonsingular_triangular_matrix' (line 822)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 0), '_logm_force_nonsingular_triangular_matrix', _logm_force_nonsingular_triangular_matrix)

@norecursion
def _logm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_logm'
    module_type_store = module_type_store.open_function_context('_logm', 842, 0, False)
    
    # Passed parameters checking function
    _logm.stypy_localization = localization
    _logm.stypy_type_of_self = None
    _logm.stypy_type_store = module_type_store
    _logm.stypy_function_name = '_logm'
    _logm.stypy_param_names_list = ['A']
    _logm.stypy_varargs_param_name = None
    _logm.stypy_kwargs_param_name = None
    _logm.stypy_call_defaults = defaults
    _logm.stypy_call_varargs = varargs
    _logm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_logm', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_logm', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_logm(...)' code ##################

    str_34883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 858, (-1)), 'str', '\n    Compute the matrix logarithm.\n\n    See the logm docstring in matfuncs.py for more info.\n\n    Notes\n    -----\n    In this function we look at triangular matrices that are similar\n    to the input matrix.  If any diagonal entry of such a triangular matrix\n    is exactly zero then the original matrix is singular.\n    The matrix logarithm does not exist for such matrices,\n    but in such cases we will pretend that the diagonal entries that are zero\n    are actually slightly positive by an ad-hoc amount, in the interest\n    of returning something more useful than NaN.  This will cause a warning.\n\n    ')
    
    # Assigning a Call to a Name (line 859):
    
    # Assigning a Call to a Name (line 859):
    
    # Call to asarray(...): (line 859)
    # Processing the call arguments (line 859)
    # Getting the type of 'A' (line 859)
    A_34886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 19), 'A', False)
    # Processing the call keyword arguments (line 859)
    kwargs_34887 = {}
    # Getting the type of 'np' (line 859)
    np_34884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 859)
    asarray_34885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 8), np_34884, 'asarray')
    # Calling asarray(args, kwargs) (line 859)
    asarray_call_result_34888 = invoke(stypy.reporting.localization.Localization(__file__, 859, 8), asarray_34885, *[A_34886], **kwargs_34887)
    
    # Assigning a type to the variable 'A' (line 859)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 4), 'A', asarray_call_result_34888)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 860)
    # Processing the call arguments (line 860)
    # Getting the type of 'A' (line 860)
    A_34890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 860)
    shape_34891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 11), A_34890, 'shape')
    # Processing the call keyword arguments (line 860)
    kwargs_34892 = {}
    # Getting the type of 'len' (line 860)
    len_34889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 7), 'len', False)
    # Calling len(args, kwargs) (line 860)
    len_call_result_34893 = invoke(stypy.reporting.localization.Localization(__file__, 860, 7), len_34889, *[shape_34891], **kwargs_34892)
    
    int_34894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 23), 'int')
    # Applying the binary operator '!=' (line 860)
    result_ne_34895 = python_operator(stypy.reporting.localization.Localization(__file__, 860, 7), '!=', len_call_result_34893, int_34894)
    
    
    
    # Obtaining the type of the subscript
    int_34896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 36), 'int')
    # Getting the type of 'A' (line 860)
    A_34897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 28), 'A')
    # Obtaining the member 'shape' of a type (line 860)
    shape_34898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 28), A_34897, 'shape')
    # Obtaining the member '__getitem__' of a type (line 860)
    getitem___34899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 28), shape_34898, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 860)
    subscript_call_result_34900 = invoke(stypy.reporting.localization.Localization(__file__, 860, 28), getitem___34899, int_34896)
    
    
    # Obtaining the type of the subscript
    int_34901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 50), 'int')
    # Getting the type of 'A' (line 860)
    A_34902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 42), 'A')
    # Obtaining the member 'shape' of a type (line 860)
    shape_34903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 42), A_34902, 'shape')
    # Obtaining the member '__getitem__' of a type (line 860)
    getitem___34904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 42), shape_34903, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 860)
    subscript_call_result_34905 = invoke(stypy.reporting.localization.Localization(__file__, 860, 42), getitem___34904, int_34901)
    
    # Applying the binary operator '!=' (line 860)
    result_ne_34906 = python_operator(stypy.reporting.localization.Localization(__file__, 860, 28), '!=', subscript_call_result_34900, subscript_call_result_34905)
    
    # Applying the binary operator 'or' (line 860)
    result_or_keyword_34907 = python_operator(stypy.reporting.localization.Localization(__file__, 860, 7), 'or', result_ne_34895, result_ne_34906)
    
    # Testing the type of an if condition (line 860)
    if_condition_34908 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 860, 4), result_or_keyword_34907)
    # Assigning a type to the variable 'if_condition_34908' (line 860)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 4), 'if_condition_34908', if_condition_34908)
    # SSA begins for if statement (line 860)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 861)
    # Processing the call arguments (line 861)
    str_34910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 861, 25), 'str', 'expected a square matrix')
    # Processing the call keyword arguments (line 861)
    kwargs_34911 = {}
    # Getting the type of 'ValueError' (line 861)
    ValueError_34909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 861)
    ValueError_call_result_34912 = invoke(stypy.reporting.localization.Localization(__file__, 861, 14), ValueError_34909, *[str_34910], **kwargs_34911)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 861, 8), ValueError_call_result_34912, 'raise parameter', BaseException)
    # SSA join for if statement (line 860)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to issubclass(...): (line 864)
    # Processing the call arguments (line 864)
    # Getting the type of 'A' (line 864)
    A_34914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 18), 'A', False)
    # Obtaining the member 'dtype' of a type (line 864)
    dtype_34915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 18), A_34914, 'dtype')
    # Obtaining the member 'type' of a type (line 864)
    type_34916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 18), dtype_34915, 'type')
    # Getting the type of 'np' (line 864)
    np_34917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 32), 'np', False)
    # Obtaining the member 'integer' of a type (line 864)
    integer_34918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 32), np_34917, 'integer')
    # Processing the call keyword arguments (line 864)
    kwargs_34919 = {}
    # Getting the type of 'issubclass' (line 864)
    issubclass_34913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 864)
    issubclass_call_result_34920 = invoke(stypy.reporting.localization.Localization(__file__, 864, 7), issubclass_34913, *[type_34916, integer_34918], **kwargs_34919)
    
    # Testing the type of an if condition (line 864)
    if_condition_34921 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 864, 4), issubclass_call_result_34920)
    # Assigning a type to the variable 'if_condition_34921' (line 864)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 4), 'if_condition_34921', if_condition_34921)
    # SSA begins for if statement (line 864)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 865):
    
    # Assigning a Call to a Name (line 865):
    
    # Call to asarray(...): (line 865)
    # Processing the call arguments (line 865)
    # Getting the type of 'A' (line 865)
    A_34924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 23), 'A', False)
    # Processing the call keyword arguments (line 865)
    # Getting the type of 'float' (line 865)
    float_34925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 32), 'float', False)
    keyword_34926 = float_34925
    kwargs_34927 = {'dtype': keyword_34926}
    # Getting the type of 'np' (line 865)
    np_34922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 865)
    asarray_34923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 865, 12), np_34922, 'asarray')
    # Calling asarray(args, kwargs) (line 865)
    asarray_call_result_34928 = invoke(stypy.reporting.localization.Localization(__file__, 865, 12), asarray_34923, *[A_34924], **kwargs_34927)
    
    # Assigning a type to the variable 'A' (line 865)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 8), 'A', asarray_call_result_34928)
    # SSA join for if statement (line 864)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 867):
    
    # Assigning a Call to a Name (line 867):
    
    # Call to isrealobj(...): (line 867)
    # Processing the call arguments (line 867)
    # Getting the type of 'A' (line 867)
    A_34931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 32), 'A', False)
    # Processing the call keyword arguments (line 867)
    kwargs_34932 = {}
    # Getting the type of 'np' (line 867)
    np_34929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 19), 'np', False)
    # Obtaining the member 'isrealobj' of a type (line 867)
    isrealobj_34930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 19), np_34929, 'isrealobj')
    # Calling isrealobj(args, kwargs) (line 867)
    isrealobj_call_result_34933 = invoke(stypy.reporting.localization.Localization(__file__, 867, 19), isrealobj_34930, *[A_34931], **kwargs_34932)
    
    # Assigning a type to the variable 'keep_it_real' (line 867)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 4), 'keep_it_real', isrealobj_call_result_34933)
    
    
    # SSA begins for try-except statement (line 868)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Call to array_equal(...): (line 869)
    # Processing the call arguments (line 869)
    # Getting the type of 'A' (line 869)
    A_34936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 26), 'A', False)
    
    # Call to triu(...): (line 869)
    # Processing the call arguments (line 869)
    # Getting the type of 'A' (line 869)
    A_34939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 37), 'A', False)
    # Processing the call keyword arguments (line 869)
    kwargs_34940 = {}
    # Getting the type of 'np' (line 869)
    np_34937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 29), 'np', False)
    # Obtaining the member 'triu' of a type (line 869)
    triu_34938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 29), np_34937, 'triu')
    # Calling triu(args, kwargs) (line 869)
    triu_call_result_34941 = invoke(stypy.reporting.localization.Localization(__file__, 869, 29), triu_34938, *[A_34939], **kwargs_34940)
    
    # Processing the call keyword arguments (line 869)
    kwargs_34942 = {}
    # Getting the type of 'np' (line 869)
    np_34934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 11), 'np', False)
    # Obtaining the member 'array_equal' of a type (line 869)
    array_equal_34935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 11), np_34934, 'array_equal')
    # Calling array_equal(args, kwargs) (line 869)
    array_equal_call_result_34943 = invoke(stypy.reporting.localization.Localization(__file__, 869, 11), array_equal_34935, *[A_34936, triu_call_result_34941], **kwargs_34942)
    
    # Testing the type of an if condition (line 869)
    if_condition_34944 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 869, 8), array_equal_call_result_34943)
    # Assigning a type to the variable 'if_condition_34944' (line 869)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 8), 'if_condition_34944', if_condition_34944)
    # SSA begins for if statement (line 869)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 870):
    
    # Assigning a Call to a Name (line 870):
    
    # Call to _logm_force_nonsingular_triangular_matrix(...): (line 870)
    # Processing the call arguments (line 870)
    # Getting the type of 'A' (line 870)
    A_34946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 58), 'A', False)
    # Processing the call keyword arguments (line 870)
    kwargs_34947 = {}
    # Getting the type of '_logm_force_nonsingular_triangular_matrix' (line 870)
    _logm_force_nonsingular_triangular_matrix_34945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 16), '_logm_force_nonsingular_triangular_matrix', False)
    # Calling _logm_force_nonsingular_triangular_matrix(args, kwargs) (line 870)
    _logm_force_nonsingular_triangular_matrix_call_result_34948 = invoke(stypy.reporting.localization.Localization(__file__, 870, 16), _logm_force_nonsingular_triangular_matrix_34945, *[A_34946], **kwargs_34947)
    
    # Assigning a type to the variable 'A' (line 870)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 12), 'A', _logm_force_nonsingular_triangular_matrix_call_result_34948)
    
    
    
    # Call to min(...): (line 871)
    # Processing the call arguments (line 871)
    
    # Call to diag(...): (line 871)
    # Processing the call arguments (line 871)
    # Getting the type of 'A' (line 871)
    A_34953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 30), 'A', False)
    # Processing the call keyword arguments (line 871)
    kwargs_34954 = {}
    # Getting the type of 'np' (line 871)
    np_34951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 22), 'np', False)
    # Obtaining the member 'diag' of a type (line 871)
    diag_34952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 22), np_34951, 'diag')
    # Calling diag(args, kwargs) (line 871)
    diag_call_result_34955 = invoke(stypy.reporting.localization.Localization(__file__, 871, 22), diag_34952, *[A_34953], **kwargs_34954)
    
    # Processing the call keyword arguments (line 871)
    kwargs_34956 = {}
    # Getting the type of 'np' (line 871)
    np_34949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 15), 'np', False)
    # Obtaining the member 'min' of a type (line 871)
    min_34950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 15), np_34949, 'min')
    # Calling min(args, kwargs) (line 871)
    min_call_result_34957 = invoke(stypy.reporting.localization.Localization(__file__, 871, 15), min_34950, *[diag_call_result_34955], **kwargs_34956)
    
    int_34958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 36), 'int')
    # Applying the binary operator '<' (line 871)
    result_lt_34959 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 15), '<', min_call_result_34957, int_34958)
    
    # Testing the type of an if condition (line 871)
    if_condition_34960 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 871, 12), result_lt_34959)
    # Assigning a type to the variable 'if_condition_34960' (line 871)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 12), 'if_condition_34960', if_condition_34960)
    # SSA begins for if statement (line 871)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 872):
    
    # Assigning a Call to a Name (line 872):
    
    # Call to astype(...): (line 872)
    # Processing the call arguments (line 872)
    # Getting the type of 'complex' (line 872)
    complex_34963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 29), 'complex', False)
    # Processing the call keyword arguments (line 872)
    kwargs_34964 = {}
    # Getting the type of 'A' (line 872)
    A_34961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 20), 'A', False)
    # Obtaining the member 'astype' of a type (line 872)
    astype_34962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 20), A_34961, 'astype')
    # Calling astype(args, kwargs) (line 872)
    astype_call_result_34965 = invoke(stypy.reporting.localization.Localization(__file__, 872, 20), astype_34962, *[complex_34963], **kwargs_34964)
    
    # Assigning a type to the variable 'A' (line 872)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 16), 'A', astype_call_result_34965)
    # SSA join for if statement (line 871)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _logm_triu(...): (line 873)
    # Processing the call arguments (line 873)
    # Getting the type of 'A' (line 873)
    A_34967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 30), 'A', False)
    # Processing the call keyword arguments (line 873)
    kwargs_34968 = {}
    # Getting the type of '_logm_triu' (line 873)
    _logm_triu_34966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 19), '_logm_triu', False)
    # Calling _logm_triu(args, kwargs) (line 873)
    _logm_triu_call_result_34969 = invoke(stypy.reporting.localization.Localization(__file__, 873, 19), _logm_triu_34966, *[A_34967], **kwargs_34968)
    
    # Assigning a type to the variable 'stypy_return_type' (line 873)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 12), 'stypy_return_type', _logm_triu_call_result_34969)
    # SSA branch for the else part of an if statement (line 869)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'keep_it_real' (line 875)
    keep_it_real_34970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 15), 'keep_it_real')
    # Testing the type of an if condition (line 875)
    if_condition_34971 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 875, 12), keep_it_real_34970)
    # Assigning a type to the variable 'if_condition_34971' (line 875)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 12), 'if_condition_34971', if_condition_34971)
    # SSA begins for if statement (line 875)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 876):
    
    # Assigning a Subscript to a Name (line 876):
    
    # Obtaining the type of the subscript
    int_34972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 16), 'int')
    
    # Call to schur(...): (line 876)
    # Processing the call arguments (line 876)
    # Getting the type of 'A' (line 876)
    A_34974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 29), 'A', False)
    # Processing the call keyword arguments (line 876)
    kwargs_34975 = {}
    # Getting the type of 'schur' (line 876)
    schur_34973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 23), 'schur', False)
    # Calling schur(args, kwargs) (line 876)
    schur_call_result_34976 = invoke(stypy.reporting.localization.Localization(__file__, 876, 23), schur_34973, *[A_34974], **kwargs_34975)
    
    # Obtaining the member '__getitem__' of a type (line 876)
    getitem___34977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 16), schur_call_result_34976, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 876)
    subscript_call_result_34978 = invoke(stypy.reporting.localization.Localization(__file__, 876, 16), getitem___34977, int_34972)
    
    # Assigning a type to the variable 'tuple_var_assignment_32864' (line 876)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 16), 'tuple_var_assignment_32864', subscript_call_result_34978)
    
    # Assigning a Subscript to a Name (line 876):
    
    # Obtaining the type of the subscript
    int_34979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 16), 'int')
    
    # Call to schur(...): (line 876)
    # Processing the call arguments (line 876)
    # Getting the type of 'A' (line 876)
    A_34981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 29), 'A', False)
    # Processing the call keyword arguments (line 876)
    kwargs_34982 = {}
    # Getting the type of 'schur' (line 876)
    schur_34980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 23), 'schur', False)
    # Calling schur(args, kwargs) (line 876)
    schur_call_result_34983 = invoke(stypy.reporting.localization.Localization(__file__, 876, 23), schur_34980, *[A_34981], **kwargs_34982)
    
    # Obtaining the member '__getitem__' of a type (line 876)
    getitem___34984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 16), schur_call_result_34983, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 876)
    subscript_call_result_34985 = invoke(stypy.reporting.localization.Localization(__file__, 876, 16), getitem___34984, int_34979)
    
    # Assigning a type to the variable 'tuple_var_assignment_32865' (line 876)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 16), 'tuple_var_assignment_32865', subscript_call_result_34985)
    
    # Assigning a Name to a Name (line 876):
    # Getting the type of 'tuple_var_assignment_32864' (line 876)
    tuple_var_assignment_32864_34986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 16), 'tuple_var_assignment_32864')
    # Assigning a type to the variable 'T' (line 876)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 16), 'T', tuple_var_assignment_32864_34986)
    
    # Assigning a Name to a Name (line 876):
    # Getting the type of 'tuple_var_assignment_32865' (line 876)
    tuple_var_assignment_32865_34987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 16), 'tuple_var_assignment_32865')
    # Assigning a type to the variable 'Z' (line 876)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 19), 'Z', tuple_var_assignment_32865_34987)
    
    
    
    # Call to array_equal(...): (line 877)
    # Processing the call arguments (line 877)
    # Getting the type of 'T' (line 877)
    T_34990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 38), 'T', False)
    
    # Call to triu(...): (line 877)
    # Processing the call arguments (line 877)
    # Getting the type of 'T' (line 877)
    T_34993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 49), 'T', False)
    # Processing the call keyword arguments (line 877)
    kwargs_34994 = {}
    # Getting the type of 'np' (line 877)
    np_34991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 41), 'np', False)
    # Obtaining the member 'triu' of a type (line 877)
    triu_34992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 41), np_34991, 'triu')
    # Calling triu(args, kwargs) (line 877)
    triu_call_result_34995 = invoke(stypy.reporting.localization.Localization(__file__, 877, 41), triu_34992, *[T_34993], **kwargs_34994)
    
    # Processing the call keyword arguments (line 877)
    kwargs_34996 = {}
    # Getting the type of 'np' (line 877)
    np_34988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 23), 'np', False)
    # Obtaining the member 'array_equal' of a type (line 877)
    array_equal_34989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 23), np_34988, 'array_equal')
    # Calling array_equal(args, kwargs) (line 877)
    array_equal_call_result_34997 = invoke(stypy.reporting.localization.Localization(__file__, 877, 23), array_equal_34989, *[T_34990, triu_call_result_34995], **kwargs_34996)
    
    # Applying the 'not' unary operator (line 877)
    result_not__34998 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 19), 'not', array_equal_call_result_34997)
    
    # Testing the type of an if condition (line 877)
    if_condition_34999 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 877, 16), result_not__34998)
    # Assigning a type to the variable 'if_condition_34999' (line 877)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 16), 'if_condition_34999', if_condition_34999)
    # SSA begins for if statement (line 877)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 878):
    
    # Assigning a Subscript to a Name (line 878):
    
    # Obtaining the type of the subscript
    int_35000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 878, 20), 'int')
    
    # Call to rsf2csf(...): (line 878)
    # Processing the call arguments (line 878)
    # Getting the type of 'T' (line 878)
    T_35002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 35), 'T', False)
    # Getting the type of 'Z' (line 878)
    Z_35003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 38), 'Z', False)
    # Processing the call keyword arguments (line 878)
    kwargs_35004 = {}
    # Getting the type of 'rsf2csf' (line 878)
    rsf2csf_35001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 27), 'rsf2csf', False)
    # Calling rsf2csf(args, kwargs) (line 878)
    rsf2csf_call_result_35005 = invoke(stypy.reporting.localization.Localization(__file__, 878, 27), rsf2csf_35001, *[T_35002, Z_35003], **kwargs_35004)
    
    # Obtaining the member '__getitem__' of a type (line 878)
    getitem___35006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 878, 20), rsf2csf_call_result_35005, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 878)
    subscript_call_result_35007 = invoke(stypy.reporting.localization.Localization(__file__, 878, 20), getitem___35006, int_35000)
    
    # Assigning a type to the variable 'tuple_var_assignment_32866' (line 878)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 20), 'tuple_var_assignment_32866', subscript_call_result_35007)
    
    # Assigning a Subscript to a Name (line 878):
    
    # Obtaining the type of the subscript
    int_35008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 878, 20), 'int')
    
    # Call to rsf2csf(...): (line 878)
    # Processing the call arguments (line 878)
    # Getting the type of 'T' (line 878)
    T_35010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 35), 'T', False)
    # Getting the type of 'Z' (line 878)
    Z_35011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 38), 'Z', False)
    # Processing the call keyword arguments (line 878)
    kwargs_35012 = {}
    # Getting the type of 'rsf2csf' (line 878)
    rsf2csf_35009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 27), 'rsf2csf', False)
    # Calling rsf2csf(args, kwargs) (line 878)
    rsf2csf_call_result_35013 = invoke(stypy.reporting.localization.Localization(__file__, 878, 27), rsf2csf_35009, *[T_35010, Z_35011], **kwargs_35012)
    
    # Obtaining the member '__getitem__' of a type (line 878)
    getitem___35014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 878, 20), rsf2csf_call_result_35013, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 878)
    subscript_call_result_35015 = invoke(stypy.reporting.localization.Localization(__file__, 878, 20), getitem___35014, int_35008)
    
    # Assigning a type to the variable 'tuple_var_assignment_32867' (line 878)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 20), 'tuple_var_assignment_32867', subscript_call_result_35015)
    
    # Assigning a Name to a Name (line 878):
    # Getting the type of 'tuple_var_assignment_32866' (line 878)
    tuple_var_assignment_32866_35016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 20), 'tuple_var_assignment_32866')
    # Assigning a type to the variable 'T' (line 878)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 20), 'T', tuple_var_assignment_32866_35016)
    
    # Assigning a Name to a Name (line 878):
    # Getting the type of 'tuple_var_assignment_32867' (line 878)
    tuple_var_assignment_32867_35017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 20), 'tuple_var_assignment_32867')
    # Assigning a type to the variable 'Z' (line 878)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 23), 'Z', tuple_var_assignment_32867_35017)
    # SSA join for if statement (line 877)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 875)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 880):
    
    # Assigning a Subscript to a Name (line 880):
    
    # Obtaining the type of the subscript
    int_35018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 16), 'int')
    
    # Call to schur(...): (line 880)
    # Processing the call arguments (line 880)
    # Getting the type of 'A' (line 880)
    A_35020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 29), 'A', False)
    # Processing the call keyword arguments (line 880)
    str_35021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 39), 'str', 'complex')
    keyword_35022 = str_35021
    kwargs_35023 = {'output': keyword_35022}
    # Getting the type of 'schur' (line 880)
    schur_35019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 23), 'schur', False)
    # Calling schur(args, kwargs) (line 880)
    schur_call_result_35024 = invoke(stypy.reporting.localization.Localization(__file__, 880, 23), schur_35019, *[A_35020], **kwargs_35023)
    
    # Obtaining the member '__getitem__' of a type (line 880)
    getitem___35025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 16), schur_call_result_35024, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 880)
    subscript_call_result_35026 = invoke(stypy.reporting.localization.Localization(__file__, 880, 16), getitem___35025, int_35018)
    
    # Assigning a type to the variable 'tuple_var_assignment_32868' (line 880)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 16), 'tuple_var_assignment_32868', subscript_call_result_35026)
    
    # Assigning a Subscript to a Name (line 880):
    
    # Obtaining the type of the subscript
    int_35027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 16), 'int')
    
    # Call to schur(...): (line 880)
    # Processing the call arguments (line 880)
    # Getting the type of 'A' (line 880)
    A_35029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 29), 'A', False)
    # Processing the call keyword arguments (line 880)
    str_35030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 39), 'str', 'complex')
    keyword_35031 = str_35030
    kwargs_35032 = {'output': keyword_35031}
    # Getting the type of 'schur' (line 880)
    schur_35028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 23), 'schur', False)
    # Calling schur(args, kwargs) (line 880)
    schur_call_result_35033 = invoke(stypy.reporting.localization.Localization(__file__, 880, 23), schur_35028, *[A_35029], **kwargs_35032)
    
    # Obtaining the member '__getitem__' of a type (line 880)
    getitem___35034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 16), schur_call_result_35033, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 880)
    subscript_call_result_35035 = invoke(stypy.reporting.localization.Localization(__file__, 880, 16), getitem___35034, int_35027)
    
    # Assigning a type to the variable 'tuple_var_assignment_32869' (line 880)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 16), 'tuple_var_assignment_32869', subscript_call_result_35035)
    
    # Assigning a Name to a Name (line 880):
    # Getting the type of 'tuple_var_assignment_32868' (line 880)
    tuple_var_assignment_32868_35036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 16), 'tuple_var_assignment_32868')
    # Assigning a type to the variable 'T' (line 880)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 16), 'T', tuple_var_assignment_32868_35036)
    
    # Assigning a Name to a Name (line 880):
    # Getting the type of 'tuple_var_assignment_32869' (line 880)
    tuple_var_assignment_32869_35037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 16), 'tuple_var_assignment_32869')
    # Assigning a type to the variable 'Z' (line 880)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 19), 'Z', tuple_var_assignment_32869_35037)
    # SSA join for if statement (line 875)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 881):
    
    # Assigning a Call to a Name (line 881):
    
    # Call to _logm_force_nonsingular_triangular_matrix(...): (line 881)
    # Processing the call arguments (line 881)
    # Getting the type of 'T' (line 881)
    T_35039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 58), 'T', False)
    # Processing the call keyword arguments (line 881)
    # Getting the type of 'True' (line 881)
    True_35040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 69), 'True', False)
    keyword_35041 = True_35040
    kwargs_35042 = {'inplace': keyword_35041}
    # Getting the type of '_logm_force_nonsingular_triangular_matrix' (line 881)
    _logm_force_nonsingular_triangular_matrix_35038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 16), '_logm_force_nonsingular_triangular_matrix', False)
    # Calling _logm_force_nonsingular_triangular_matrix(args, kwargs) (line 881)
    _logm_force_nonsingular_triangular_matrix_call_result_35043 = invoke(stypy.reporting.localization.Localization(__file__, 881, 16), _logm_force_nonsingular_triangular_matrix_35038, *[T_35039], **kwargs_35042)
    
    # Assigning a type to the variable 'T' (line 881)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 12), 'T', _logm_force_nonsingular_triangular_matrix_call_result_35043)
    
    # Assigning a Call to a Name (line 882):
    
    # Assigning a Call to a Name (line 882):
    
    # Call to _logm_triu(...): (line 882)
    # Processing the call arguments (line 882)
    # Getting the type of 'T' (line 882)
    T_35045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 27), 'T', False)
    # Processing the call keyword arguments (line 882)
    kwargs_35046 = {}
    # Getting the type of '_logm_triu' (line 882)
    _logm_triu_35044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 16), '_logm_triu', False)
    # Calling _logm_triu(args, kwargs) (line 882)
    _logm_triu_call_result_35047 = invoke(stypy.reporting.localization.Localization(__file__, 882, 16), _logm_triu_35044, *[T_35045], **kwargs_35046)
    
    # Assigning a type to the variable 'U' (line 882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 12), 'U', _logm_triu_call_result_35047)
    
    # Assigning a Attribute to a Name (line 883):
    
    # Assigning a Attribute to a Name (line 883):
    
    # Call to conjugate(...): (line 883)
    # Processing the call arguments (line 883)
    # Getting the type of 'Z' (line 883)
    Z_35050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 30), 'Z', False)
    # Processing the call keyword arguments (line 883)
    kwargs_35051 = {}
    # Getting the type of 'np' (line 883)
    np_35048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 17), 'np', False)
    # Obtaining the member 'conjugate' of a type (line 883)
    conjugate_35049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 883, 17), np_35048, 'conjugate')
    # Calling conjugate(args, kwargs) (line 883)
    conjugate_call_result_35052 = invoke(stypy.reporting.localization.Localization(__file__, 883, 17), conjugate_35049, *[Z_35050], **kwargs_35051)
    
    # Obtaining the member 'T' of a type (line 883)
    T_35053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 883, 17), conjugate_call_result_35052, 'T')
    # Assigning a type to the variable 'ZH' (line 883)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 883, 12), 'ZH', T_35053)
    
    # Call to dot(...): (line 884)
    # Processing the call arguments (line 884)
    # Getting the type of 'ZH' (line 884)
    ZH_35060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 32), 'ZH', False)
    # Processing the call keyword arguments (line 884)
    kwargs_35061 = {}
    
    # Call to dot(...): (line 884)
    # Processing the call arguments (line 884)
    # Getting the type of 'U' (line 884)
    U_35056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 25), 'U', False)
    # Processing the call keyword arguments (line 884)
    kwargs_35057 = {}
    # Getting the type of 'Z' (line 884)
    Z_35054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 19), 'Z', False)
    # Obtaining the member 'dot' of a type (line 884)
    dot_35055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 19), Z_35054, 'dot')
    # Calling dot(args, kwargs) (line 884)
    dot_call_result_35058 = invoke(stypy.reporting.localization.Localization(__file__, 884, 19), dot_35055, *[U_35056], **kwargs_35057)
    
    # Obtaining the member 'dot' of a type (line 884)
    dot_35059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 19), dot_call_result_35058, 'dot')
    # Calling dot(args, kwargs) (line 884)
    dot_call_result_35062 = invoke(stypy.reporting.localization.Localization(__file__, 884, 19), dot_35059, *[ZH_35060], **kwargs_35061)
    
    # Assigning a type to the variable 'stypy_return_type' (line 884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 12), 'stypy_return_type', dot_call_result_35062)
    # SSA join for if statement (line 869)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 868)
    # SSA branch for the except 'Tuple' branch of a try statement (line 868)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 886):
    
    # Assigning a Call to a Name (line 886):
    
    # Call to empty_like(...): (line 886)
    # Processing the call arguments (line 886)
    # Getting the type of 'A' (line 886)
    A_35065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 26), 'A', False)
    # Processing the call keyword arguments (line 886)
    kwargs_35066 = {}
    # Getting the type of 'np' (line 886)
    np_35063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 12), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 886)
    empty_like_35064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 12), np_35063, 'empty_like')
    # Calling empty_like(args, kwargs) (line 886)
    empty_like_call_result_35067 = invoke(stypy.reporting.localization.Localization(__file__, 886, 12), empty_like_35064, *[A_35065], **kwargs_35066)
    
    # Assigning a type to the variable 'X' (line 886)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 886, 8), 'X', empty_like_call_result_35067)
    
    # Call to fill(...): (line 887)
    # Processing the call arguments (line 887)
    # Getting the type of 'np' (line 887)
    np_35070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 15), 'np', False)
    # Obtaining the member 'nan' of a type (line 887)
    nan_35071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 15), np_35070, 'nan')
    # Processing the call keyword arguments (line 887)
    kwargs_35072 = {}
    # Getting the type of 'X' (line 887)
    X_35068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 8), 'X', False)
    # Obtaining the member 'fill' of a type (line 887)
    fill_35069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 8), X_35068, 'fill')
    # Calling fill(args, kwargs) (line 887)
    fill_call_result_35073 = invoke(stypy.reporting.localization.Localization(__file__, 887, 8), fill_35069, *[nan_35071], **kwargs_35072)
    
    # Getting the type of 'X' (line 888)
    X_35074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 15), 'X')
    # Assigning a type to the variable 'stypy_return_type' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'stypy_return_type', X_35074)
    # SSA join for try-except statement (line 868)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_logm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_logm' in the type store
    # Getting the type of 'stypy_return_type' (line 842)
    stypy_return_type_35075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35075)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_logm'
    return stypy_return_type_35075

# Assigning a type to the variable '_logm' (line 842)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 0), '_logm', _logm)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
