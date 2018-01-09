
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #
2: # Author: Travis Oliphant, March 2002
3: #
4: 
5: from __future__ import division, print_function, absolute_import
6: 
7: __all__ = ['expm','cosm','sinm','tanm','coshm','sinhm',
8:            'tanhm','logm','funm','signm','sqrtm',
9:            'expm_frechet', 'expm_cond', 'fractional_matrix_power']
10: 
11: from numpy import (Inf, dot, diag, product, logical_not, ravel,
12:         transpose, conjugate, absolute, amax, sign, isfinite, single)
13: import numpy as np
14: 
15: # Local imports
16: from .misc import norm
17: from .basic import solve, inv
18: from .special_matrices import triu
19: from .decomp_svd import svd
20: from .decomp_schur import schur, rsf2csf
21: from ._expm_frechet import expm_frechet, expm_cond
22: from ._matfuncs_sqrtm import sqrtm
23: 
24: eps = np.finfo(float).eps
25: feps = np.finfo(single).eps
26: 
27: _array_precision = {'i': 1, 'l': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}
28: 
29: 
30: ###############################################################################
31: # Utility functions.
32: 
33: 
34: def _asarray_square(A):
35:     '''
36:     Wraps asarray with the extra requirement that the input be a square matrix.
37: 
38:     The motivation is that the matfuncs module has real functions that have
39:     been lifted to square matrix functions.
40: 
41:     Parameters
42:     ----------
43:     A : array_like
44:         A square matrix.
45: 
46:     Returns
47:     -------
48:     out : ndarray
49:         An ndarray copy or view or other representation of A.
50: 
51:     '''
52:     A = np.asarray(A)
53:     if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
54:         raise ValueError('expected square array_like input')
55:     return A
56: 
57: 
58: def _maybe_real(A, B, tol=None):
59:     '''
60:     Return either B or the real part of B, depending on properties of A and B.
61: 
62:     The motivation is that B has been computed as a complicated function of A,
63:     and B may be perturbed by negligible imaginary components.
64:     If A is real and B is complex with small imaginary components,
65:     then return a real copy of B.  The assumption in that case would be that
66:     the imaginary components of B are numerical artifacts.
67: 
68:     Parameters
69:     ----------
70:     A : ndarray
71:         Input array whose type is to be checked as real vs. complex.
72:     B : ndarray
73:         Array to be returned, possibly without its imaginary part.
74:     tol : float
75:         Absolute tolerance.
76: 
77:     Returns
78:     -------
79:     out : real or complex array
80:         Either the input array B or only the real part of the input array B.
81: 
82:     '''
83:     # Note that booleans and integers compare as real.
84:     if np.isrealobj(A) and np.iscomplexobj(B):
85:         if tol is None:
86:             tol = {0:feps*1e3, 1:eps*1e6}[_array_precision[B.dtype.char]]
87:         if np.allclose(B.imag, 0.0, atol=tol):
88:             B = B.real
89:     return B
90: 
91: 
92: ###############################################################################
93: # Matrix functions.
94: 
95: 
96: def fractional_matrix_power(A, t):
97:     '''
98:     Compute the fractional power of a matrix.
99: 
100:     Proceeds according to the discussion in section (6) of [1]_.
101: 
102:     Parameters
103:     ----------
104:     A : (N, N) array_like
105:         Matrix whose fractional power to evaluate.
106:     t : float
107:         Fractional power.
108: 
109:     Returns
110:     -------
111:     X : (N, N) array_like
112:         The fractional power of the matrix.
113: 
114:     References
115:     ----------
116:     .. [1] Nicholas J. Higham and Lijing lin (2011)
117:            "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
118:            SIAM Journal on Matrix Analysis and Applications,
119:            32 (3). pp. 1056-1078. ISSN 0895-4798
120: 
121:     Examples
122:     --------
123:     >>> from scipy.linalg import fractional_matrix_power
124:     >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
125:     >>> b = fractional_matrix_power(a, 0.5)
126:     >>> b
127:     array([[ 0.75592895,  1.13389342],
128:            [ 0.37796447,  1.88982237]])
129:     >>> np.dot(b, b)      # Verify square root
130:     array([[ 1.,  3.],
131:            [ 1.,  4.]])
132: 
133:     '''
134:     # This fixes some issue with imports;
135:     # this function calls onenormest which is in scipy.sparse.
136:     A = _asarray_square(A)
137:     import scipy.linalg._matfuncs_inv_ssq
138:     return scipy.linalg._matfuncs_inv_ssq._fractional_matrix_power(A, t)
139: 
140: 
141: def logm(A, disp=True):
142:     '''
143:     Compute matrix logarithm.
144: 
145:     The matrix logarithm is the inverse of
146:     expm: expm(logm(`A`)) == `A`
147: 
148:     Parameters
149:     ----------
150:     A : (N, N) array_like
151:         Matrix whose logarithm to evaluate
152:     disp : bool, optional
153:         Print warning if error in the result is estimated large
154:         instead of returning estimated error. (Default: True)
155: 
156:     Returns
157:     -------
158:     logm : (N, N) ndarray
159:         Matrix logarithm of `A`
160:     errest : float
161:         (if disp == False)
162: 
163:         1-norm of the estimated error, ||err||_1 / ||A||_1
164: 
165:     References
166:     ----------
167:     .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
168:            "Improved Inverse Scaling and Squaring Algorithms
169:            for the Matrix Logarithm."
170:            SIAM Journal on Scientific Computing, 34 (4). C152-C169.
171:            ISSN 1095-7197
172: 
173:     .. [2] Nicholas J. Higham (2008)
174:            "Functions of Matrices: Theory and Computation"
175:            ISBN 978-0-898716-46-7
176: 
177:     .. [3] Nicholas J. Higham and Lijing lin (2011)
178:            "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
179:            SIAM Journal on Matrix Analysis and Applications,
180:            32 (3). pp. 1056-1078. ISSN 0895-4798
181: 
182:     Examples
183:     --------
184:     >>> from scipy.linalg import logm, expm
185:     >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
186:     >>> b = logm(a)
187:     >>> b
188:     array([[-1.02571087,  2.05142174],
189:            [ 0.68380725,  1.02571087]])
190:     >>> expm(b)         # Verify expm(logm(a)) returns a
191:     array([[ 1.,  3.],
192:            [ 1.,  4.]])
193: 
194:     '''
195:     A = _asarray_square(A)
196:     # Avoid circular import ... this is OK, right?
197:     import scipy.linalg._matfuncs_inv_ssq
198:     F = scipy.linalg._matfuncs_inv_ssq._logm(A)
199:     F = _maybe_real(A, F)
200:     errtol = 1000*eps
201:     #TODO use a better error approximation
202:     errest = norm(expm(F)-A,1) / norm(A,1)
203:     if disp:
204:         if not isfinite(errest) or errest >= errtol:
205:             print("logm result may be inaccurate, approximate err =", errest)
206:         return F
207:     else:
208:         return F, errest
209: 
210: 
211: def expm(A):
212:     '''
213:     Compute the matrix exponential using Pade approximation.
214: 
215:     Parameters
216:     ----------
217:     A : (N, N) array_like or sparse matrix
218:         Matrix to be exponentiated.
219: 
220:     Returns
221:     -------
222:     expm : (N, N) ndarray
223:         Matrix exponential of `A`.
224: 
225:     References
226:     ----------
227:     .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
228:            "A New Scaling and Squaring Algorithm for the Matrix Exponential."
229:            SIAM Journal on Matrix Analysis and Applications.
230:            31 (3). pp. 970-989. ISSN 1095-7162
231: 
232:     Examples
233:     --------
234:     >>> from scipy.linalg import expm, sinm, cosm
235: 
236:     Matrix version of the formula exp(0) = 1:
237: 
238:     >>> expm(np.zeros((2,2)))
239:     array([[ 1.,  0.],
240:            [ 0.,  1.]])
241: 
242:     Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
243:     applied to a matrix:
244: 
245:     >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
246:     >>> expm(1j*a)
247:     array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
248:            [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
249:     >>> cosm(a) + 1j*sinm(a)
250:     array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
251:            [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
252: 
253:     '''
254:     # Input checking and conversion is provided by sparse.linalg.expm().
255:     import scipy.sparse.linalg
256:     return scipy.sparse.linalg.expm(A)
257: 
258: 
259: def cosm(A):
260:     '''
261:     Compute the matrix cosine.
262: 
263:     This routine uses expm to compute the matrix exponentials.
264: 
265:     Parameters
266:     ----------
267:     A : (N, N) array_like
268:         Input array
269: 
270:     Returns
271:     -------
272:     cosm : (N, N) ndarray
273:         Matrix cosine of A
274: 
275:     Examples
276:     --------
277:     >>> from scipy.linalg import expm, sinm, cosm
278: 
279:     Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
280:     applied to a matrix:
281: 
282:     >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
283:     >>> expm(1j*a)
284:     array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
285:            [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
286:     >>> cosm(a) + 1j*sinm(a)
287:     array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
288:            [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
289: 
290:     '''
291:     A = _asarray_square(A)
292:     if np.iscomplexobj(A):
293:         return 0.5*(expm(1j*A) + expm(-1j*A))
294:     else:
295:         return expm(1j*A).real
296: 
297: 
298: def sinm(A):
299:     '''
300:     Compute the matrix sine.
301: 
302:     This routine uses expm to compute the matrix exponentials.
303: 
304:     Parameters
305:     ----------
306:     A : (N, N) array_like
307:         Input array.
308: 
309:     Returns
310:     -------
311:     sinm : (N, N) ndarray
312:         Matrix sine of `A`
313: 
314:     Examples
315:     --------
316:     >>> from scipy.linalg import expm, sinm, cosm
317: 
318:     Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
319:     applied to a matrix:
320: 
321:     >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
322:     >>> expm(1j*a)
323:     array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
324:            [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
325:     >>> cosm(a) + 1j*sinm(a)
326:     array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
327:            [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
328: 
329:     '''
330:     A = _asarray_square(A)
331:     if np.iscomplexobj(A):
332:         return -0.5j*(expm(1j*A) - expm(-1j*A))
333:     else:
334:         return expm(1j*A).imag
335: 
336: 
337: def tanm(A):
338:     '''
339:     Compute the matrix tangent.
340: 
341:     This routine uses expm to compute the matrix exponentials.
342: 
343:     Parameters
344:     ----------
345:     A : (N, N) array_like
346:         Input array.
347: 
348:     Returns
349:     -------
350:     tanm : (N, N) ndarray
351:         Matrix tangent of `A`
352: 
353:     Examples
354:     --------
355:     >>> from scipy.linalg import tanm, sinm, cosm
356:     >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
357:     >>> t = tanm(a)
358:     >>> t
359:     array([[ -2.00876993,  -8.41880636],
360:            [ -2.80626879, -10.42757629]])
361: 
362:     Verify tanm(a) = sinm(a).dot(inv(cosm(a)))
363: 
364:     >>> s = sinm(a)
365:     >>> c = cosm(a)
366:     >>> s.dot(np.linalg.inv(c))
367:     array([[ -2.00876993,  -8.41880636],
368:            [ -2.80626879, -10.42757629]])
369: 
370:     '''
371:     A = _asarray_square(A)
372:     return _maybe_real(A, solve(cosm(A), sinm(A)))
373: 
374: 
375: def coshm(A):
376:     '''
377:     Compute the hyperbolic matrix cosine.
378: 
379:     This routine uses expm to compute the matrix exponentials.
380: 
381:     Parameters
382:     ----------
383:     A : (N, N) array_like
384:         Input array.
385: 
386:     Returns
387:     -------
388:     coshm : (N, N) ndarray
389:         Hyperbolic matrix cosine of `A`
390: 
391:     Examples
392:     --------
393:     >>> from scipy.linalg import tanhm, sinhm, coshm
394:     >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
395:     >>> c = coshm(a)
396:     >>> c
397:     array([[ 11.24592233,  38.76236492],
398:            [ 12.92078831,  50.00828725]])
399: 
400:     Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))
401: 
402:     >>> t = tanhm(a)
403:     >>> s = sinhm(a)
404:     >>> t - s.dot(np.linalg.inv(c))
405:     array([[  2.72004641e-15,   4.55191440e-15],
406:            [  0.00000000e+00,  -5.55111512e-16]])
407: 
408:     '''
409:     A = _asarray_square(A)
410:     return _maybe_real(A, 0.5 * (expm(A) + expm(-A)))
411: 
412: 
413: def sinhm(A):
414:     '''
415:     Compute the hyperbolic matrix sine.
416: 
417:     This routine uses expm to compute the matrix exponentials.
418: 
419:     Parameters
420:     ----------
421:     A : (N, N) array_like
422:         Input array.
423: 
424:     Returns
425:     -------
426:     sinhm : (N, N) ndarray
427:         Hyperbolic matrix sine of `A`
428: 
429:     Examples
430:     --------
431:     >>> from scipy.linalg import tanhm, sinhm, coshm
432:     >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
433:     >>> s = sinhm(a)
434:     >>> s
435:     array([[ 10.57300653,  39.28826594],
436:            [ 13.09608865,  49.86127247]])
437: 
438:     Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))
439: 
440:     >>> t = tanhm(a)
441:     >>> c = coshm(a)
442:     >>> t - s.dot(np.linalg.inv(c))
443:     array([[  2.72004641e-15,   4.55191440e-15],
444:            [  0.00000000e+00,  -5.55111512e-16]])
445: 
446:     '''
447:     A = _asarray_square(A)
448:     return _maybe_real(A, 0.5 * (expm(A) - expm(-A)))
449: 
450: 
451: def tanhm(A):
452:     '''
453:     Compute the hyperbolic matrix tangent.
454: 
455:     This routine uses expm to compute the matrix exponentials.
456: 
457:     Parameters
458:     ----------
459:     A : (N, N) array_like
460:         Input array
461: 
462:     Returns
463:     -------
464:     tanhm : (N, N) ndarray
465:         Hyperbolic matrix tangent of `A`
466: 
467:     Examples
468:     --------
469:     >>> from scipy.linalg import tanhm, sinhm, coshm
470:     >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
471:     >>> t = tanhm(a)
472:     >>> t
473:     array([[ 0.3428582 ,  0.51987926],
474:            [ 0.17329309,  0.86273746]])
475: 
476:     Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))
477: 
478:     >>> s = sinhm(a)
479:     >>> c = coshm(a)
480:     >>> t - s.dot(np.linalg.inv(c))
481:     array([[  2.72004641e-15,   4.55191440e-15],
482:            [  0.00000000e+00,  -5.55111512e-16]])
483: 
484:     '''
485:     A = _asarray_square(A)
486:     return _maybe_real(A, solve(coshm(A), sinhm(A)))
487: 
488: 
489: def funm(A, func, disp=True):
490:     '''
491:     Evaluate a matrix function specified by a callable.
492: 
493:     Returns the value of matrix-valued function ``f`` at `A`. The
494:     function ``f`` is an extension of the scalar-valued function `func`
495:     to matrices.
496: 
497:     Parameters
498:     ----------
499:     A : (N, N) array_like
500:         Matrix at which to evaluate the function
501:     func : callable
502:         Callable object that evaluates a scalar function f.
503:         Must be vectorized (eg. using vectorize).
504:     disp : bool, optional
505:         Print warning if error in the result is estimated large
506:         instead of returning estimated error. (Default: True)
507: 
508:     Returns
509:     -------
510:     funm : (N, N) ndarray
511:         Value of the matrix function specified by func evaluated at `A`
512:     errest : float
513:         (if disp == False)
514: 
515:         1-norm of the estimated error, ||err||_1 / ||A||_1
516: 
517:     Examples
518:     --------
519:     >>> from scipy.linalg import funm
520:     >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
521:     >>> funm(a, lambda x: x*x)
522:     array([[  4.,  15.],
523:            [  5.,  19.]])
524:     >>> a.dot(a)
525:     array([[  4.,  15.],
526:            [  5.,  19.]])
527: 
528:     Notes
529:     -----
530:     This function implements the general algorithm based on Schur decomposition
531:     (Algorithm 9.1.1. in [1]_).
532: 
533:     If the input matrix is known to be diagonalizable, then relying on the
534:     eigendecomposition is likely to be faster. For example, if your matrix is
535:     Hermitian, you can do
536: 
537:     >>> from scipy.linalg import eigh
538:     >>> def funm_herm(a, func, check_finite=False):
539:     ...     w, v = eigh(a, check_finite=check_finite)
540:     ...     ## if you further know that your matrix is positive semidefinite,
541:     ...     ## you can optionally guard against precision errors by doing
542:     ...     # w = np.maximum(w, 0)
543:     ...     w = func(w)
544:     ...     return (v * w).dot(v.conj().T)
545: 
546:     References
547:     ----------
548:     .. [1] Gene H. Golub, Charles F. van Loan, Matrix Computations 4th ed.
549: 
550:     '''
551:     A = _asarray_square(A)
552:     # Perform Shur decomposition (lapack ?gees)
553:     T, Z = schur(A)
554:     T, Z = rsf2csf(T,Z)
555:     n,n = T.shape
556:     F = diag(func(diag(T)))  # apply function to diagonal elements
557:     F = F.astype(T.dtype.char)  # e.g. when F is real but T is complex
558: 
559:     minden = abs(T[0,0])
560: 
561:     # implement Algorithm 11.1.1 from Golub and Van Loan
562:     #                 "matrix Computations."
563:     for p in range(1,n):
564:         for i in range(1,n-p+1):
565:             j = i + p
566:             s = T[i-1,j-1] * (F[j-1,j-1] - F[i-1,i-1])
567:             ksl = slice(i,j-1)
568:             val = dot(T[i-1,ksl],F[ksl,j-1]) - dot(F[i-1,ksl],T[ksl,j-1])
569:             s = s + val
570:             den = T[j-1,j-1] - T[i-1,i-1]
571:             if den != 0.0:
572:                 s = s / den
573:             F[i-1,j-1] = s
574:             minden = min(minden,abs(den))
575: 
576:     F = dot(dot(Z, F), transpose(conjugate(Z)))
577:     F = _maybe_real(A, F)
578: 
579:     tol = {0:feps, 1:eps}[_array_precision[F.dtype.char]]
580:     if minden == 0.0:
581:         minden = tol
582:     err = min(1, max(tol,(tol/minden)*norm(triu(T,1),1)))
583:     if product(ravel(logical_not(isfinite(F))),axis=0):
584:         err = Inf
585:     if disp:
586:         if err > 1000*tol:
587:             print("funm result may be inaccurate, approximate err =", err)
588:         return F
589:     else:
590:         return F, err
591: 
592: 
593: def signm(A, disp=True):
594:     '''
595:     Matrix sign function.
596: 
597:     Extension of the scalar sign(x) to matrices.
598: 
599:     Parameters
600:     ----------
601:     A : (N, N) array_like
602:         Matrix at which to evaluate the sign function
603:     disp : bool, optional
604:         Print warning if error in the result is estimated large
605:         instead of returning estimated error. (Default: True)
606: 
607:     Returns
608:     -------
609:     signm : (N, N) ndarray
610:         Value of the sign function at `A`
611:     errest : float
612:         (if disp == False)
613: 
614:         1-norm of the estimated error, ||err||_1 / ||A||_1
615: 
616:     Examples
617:     --------
618:     >>> from scipy.linalg import signm, eigvals
619:     >>> a = [[1,2,3], [1,2,1], [1,1,1]]
620:     >>> eigvals(a)
621:     array([ 4.12488542+0.j, -0.76155718+0.j,  0.63667176+0.j])
622:     >>> eigvals(signm(a))
623:     array([-1.+0.j,  1.+0.j,  1.+0.j])
624: 
625:     '''
626:     A = _asarray_square(A)
627: 
628:     def rounded_sign(x):
629:         rx = np.real(x)
630:         if rx.dtype.char == 'f':
631:             c = 1e3*feps*amax(x)
632:         else:
633:             c = 1e3*eps*amax(x)
634:         return sign((absolute(rx) > c) * rx)
635:     result, errest = funm(A, rounded_sign, disp=0)
636:     errtol = {0:1e3*feps, 1:1e3*eps}[_array_precision[result.dtype.char]]
637:     if errest < errtol:
638:         return result
639: 
640:     # Handle signm of defective matrices:
641: 
642:     # See "E.D.Denman and J.Leyva-Ramos, Appl.Math.Comp.,
643:     # 8:237-250,1981" for how to improve the following (currently a
644:     # rather naive) iteration process:
645: 
646:     # a = result # sometimes iteration converges faster but where??
647: 
648:     # Shifting to avoid zero eigenvalues. How to ensure that shifting does
649:     # not change the spectrum too much?
650:     vals = svd(A, compute_uv=0)
651:     max_sv = np.amax(vals)
652:     # min_nonzero_sv = vals[(vals>max_sv*errtol).tolist().count(1)-1]
653:     # c = 0.5/min_nonzero_sv
654:     c = 0.5/max_sv
655:     S0 = A + c*np.identity(A.shape[0])
656:     prev_errest = errest
657:     for i in range(100):
658:         iS0 = inv(S0)
659:         S0 = 0.5*(S0 + iS0)
660:         Pp = 0.5*(dot(S0,S0)+S0)
661:         errest = norm(dot(Pp,Pp)-Pp,1)
662:         if errest < errtol or prev_errest == errest:
663:             break
664:         prev_errest = errest
665:     if disp:
666:         if not isfinite(errest) or errest >= errtol:
667:             print("signm result may be inaccurate, approximate err =", errest)
668:         return S0
669:     else:
670:         return S0, errest
671: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['expm', 'cosm', 'sinm', 'tanm', 'coshm', 'sinhm', 'tanhm', 'logm', 'funm', 'signm', 'sqrtm', 'expm_frechet', 'expm_cond', 'fractional_matrix_power']
module_type_store.set_exportable_members(['expm', 'cosm', 'sinm', 'tanm', 'coshm', 'sinhm', 'tanhm', 'logm', 'funm', 'signm', 'sqrtm', 'expm_frechet', 'expm_cond', 'fractional_matrix_power'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_22548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_22549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'expm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22549)
# Adding element type (line 7)
str_22550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 18), 'str', 'cosm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22550)
# Adding element type (line 7)
str_22551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'str', 'sinm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22551)
# Adding element type (line 7)
str_22552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 32), 'str', 'tanm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22552)
# Adding element type (line 7)
str_22553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 39), 'str', 'coshm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22553)
# Adding element type (line 7)
str_22554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 47), 'str', 'sinhm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22554)
# Adding element type (line 7)
str_22555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'tanhm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22555)
# Adding element type (line 7)
str_22556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'str', 'logm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22556)
# Adding element type (line 7)
str_22557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 26), 'str', 'funm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22557)
# Adding element type (line 7)
str_22558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 33), 'str', 'signm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22558)
# Adding element type (line 7)
str_22559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 41), 'str', 'sqrtm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22559)
# Adding element type (line 7)
str_22560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'expm_frechet')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22560)
# Adding element type (line 7)
str_22561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 27), 'str', 'expm_cond')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22561)
# Adding element type (line 7)
str_22562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 40), 'str', 'fractional_matrix_power')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_22548, str_22562)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_22548)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy import Inf, dot, diag, product, logical_not, ravel, transpose, conjugate, absolute, amax, sign, isfinite, single' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22563 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_22563) is not StypyTypeError):

    if (import_22563 != 'pyd_module'):
        __import__(import_22563)
        sys_modules_22564 = sys.modules[import_22563]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', sys_modules_22564.module_type_store, module_type_store, ['Inf', 'dot', 'diag', 'product', 'logical_not', 'ravel', 'transpose', 'conjugate', 'absolute', 'amax', 'sign', 'isfinite', 'single'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_22564, sys_modules_22564.module_type_store, module_type_store)
    else:
        from numpy import Inf, dot, diag, product, logical_not, ravel, transpose, conjugate, absolute, amax, sign, isfinite, single

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', None, module_type_store, ['Inf', 'dot', 'diag', 'product', 'logical_not', 'ravel', 'transpose', 'conjugate', 'absolute', 'amax', 'sign', 'isfinite', 'single'], [Inf, dot, diag, product, logical_not, ravel, transpose, conjugate, absolute, amax, sign, isfinite, single])

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_22563)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22565 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_22565) is not StypyTypeError):

    if (import_22565 != 'pyd_module'):
        __import__(import_22565)
        sys_modules_22566 = sys.modules[import_22565]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', sys_modules_22566.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_22565)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.linalg.misc import norm' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22567 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.linalg.misc')

if (type(import_22567) is not StypyTypeError):

    if (import_22567 != 'pyd_module'):
        __import__(import_22567)
        sys_modules_22568 = sys.modules[import_22567]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.linalg.misc', sys_modules_22568.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_22568, sys_modules_22568.module_type_store, module_type_store)
    else:
        from scipy.linalg.misc import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.linalg.misc', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'scipy.linalg.misc' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.linalg.misc', import_22567)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.linalg.basic import solve, inv' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22569 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg.basic')

if (type(import_22569) is not StypyTypeError):

    if (import_22569 != 'pyd_module'):
        __import__(import_22569)
        sys_modules_22570 = sys.modules[import_22569]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg.basic', sys_modules_22570.module_type_store, module_type_store, ['solve', 'inv'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_22570, sys_modules_22570.module_type_store, module_type_store)
    else:
        from scipy.linalg.basic import solve, inv

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg.basic', None, module_type_store, ['solve', 'inv'], [solve, inv])

else:
    # Assigning a type to the variable 'scipy.linalg.basic' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg.basic', import_22569)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.linalg.special_matrices import triu' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22571 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.linalg.special_matrices')

if (type(import_22571) is not StypyTypeError):

    if (import_22571 != 'pyd_module'):
        __import__(import_22571)
        sys_modules_22572 = sys.modules[import_22571]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.linalg.special_matrices', sys_modules_22572.module_type_store, module_type_store, ['triu'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_22572, sys_modules_22572.module_type_store, module_type_store)
    else:
        from scipy.linalg.special_matrices import triu

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.linalg.special_matrices', None, module_type_store, ['triu'], [triu])

else:
    # Assigning a type to the variable 'scipy.linalg.special_matrices' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.linalg.special_matrices', import_22571)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy.linalg.decomp_svd import svd' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22573 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.linalg.decomp_svd')

if (type(import_22573) is not StypyTypeError):

    if (import_22573 != 'pyd_module'):
        __import__(import_22573)
        sys_modules_22574 = sys.modules[import_22573]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.linalg.decomp_svd', sys_modules_22574.module_type_store, module_type_store, ['svd'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_22574, sys_modules_22574.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp_svd import svd

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.linalg.decomp_svd', None, module_type_store, ['svd'], [svd])

else:
    # Assigning a type to the variable 'scipy.linalg.decomp_svd' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.linalg.decomp_svd', import_22573)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy.linalg.decomp_schur import schur, rsf2csf' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22575 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg.decomp_schur')

if (type(import_22575) is not StypyTypeError):

    if (import_22575 != 'pyd_module'):
        __import__(import_22575)
        sys_modules_22576 = sys.modules[import_22575]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg.decomp_schur', sys_modules_22576.module_type_store, module_type_store, ['schur', 'rsf2csf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_22576, sys_modules_22576.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp_schur import schur, rsf2csf

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg.decomp_schur', None, module_type_store, ['schur', 'rsf2csf'], [schur, rsf2csf])

else:
    # Assigning a type to the variable 'scipy.linalg.decomp_schur' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg.decomp_schur', import_22575)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from scipy.linalg._expm_frechet import expm_frechet, expm_cond' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22577 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.linalg._expm_frechet')

if (type(import_22577) is not StypyTypeError):

    if (import_22577 != 'pyd_module'):
        __import__(import_22577)
        sys_modules_22578 = sys.modules[import_22577]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.linalg._expm_frechet', sys_modules_22578.module_type_store, module_type_store, ['expm_frechet', 'expm_cond'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_22578, sys_modules_22578.module_type_store, module_type_store)
    else:
        from scipy.linalg._expm_frechet import expm_frechet, expm_cond

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.linalg._expm_frechet', None, module_type_store, ['expm_frechet', 'expm_cond'], [expm_frechet, expm_cond])

else:
    # Assigning a type to the variable 'scipy.linalg._expm_frechet' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.linalg._expm_frechet', import_22577)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from scipy.linalg._matfuncs_sqrtm import sqrtm' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22579 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.linalg._matfuncs_sqrtm')

if (type(import_22579) is not StypyTypeError):

    if (import_22579 != 'pyd_module'):
        __import__(import_22579)
        sys_modules_22580 = sys.modules[import_22579]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.linalg._matfuncs_sqrtm', sys_modules_22580.module_type_store, module_type_store, ['sqrtm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_22580, sys_modules_22580.module_type_store, module_type_store)
    else:
        from scipy.linalg._matfuncs_sqrtm import sqrtm

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.linalg._matfuncs_sqrtm', None, module_type_store, ['sqrtm'], [sqrtm])

else:
    # Assigning a type to the variable 'scipy.linalg._matfuncs_sqrtm' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.linalg._matfuncs_sqrtm', import_22579)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a Attribute to a Name (line 24):

# Assigning a Attribute to a Name (line 24):

# Call to finfo(...): (line 24)
# Processing the call arguments (line 24)
# Getting the type of 'float' (line 24)
float_22583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 15), 'float', False)
# Processing the call keyword arguments (line 24)
kwargs_22584 = {}
# Getting the type of 'np' (line 24)
np_22581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 6), 'np', False)
# Obtaining the member 'finfo' of a type (line 24)
finfo_22582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 6), np_22581, 'finfo')
# Calling finfo(args, kwargs) (line 24)
finfo_call_result_22585 = invoke(stypy.reporting.localization.Localization(__file__, 24, 6), finfo_22582, *[float_22583], **kwargs_22584)

# Obtaining the member 'eps' of a type (line 24)
eps_22586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 6), finfo_call_result_22585, 'eps')
# Assigning a type to the variable 'eps' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'eps', eps_22586)

# Assigning a Attribute to a Name (line 25):

# Assigning a Attribute to a Name (line 25):

# Call to finfo(...): (line 25)
# Processing the call arguments (line 25)
# Getting the type of 'single' (line 25)
single_22589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'single', False)
# Processing the call keyword arguments (line 25)
kwargs_22590 = {}
# Getting the type of 'np' (line 25)
np_22587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 7), 'np', False)
# Obtaining the member 'finfo' of a type (line 25)
finfo_22588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 7), np_22587, 'finfo')
# Calling finfo(args, kwargs) (line 25)
finfo_call_result_22591 = invoke(stypy.reporting.localization.Localization(__file__, 25, 7), finfo_22588, *[single_22589], **kwargs_22590)

# Obtaining the member 'eps' of a type (line 25)
eps_22592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 7), finfo_call_result_22591, 'eps')
# Assigning a type to the variable 'feps' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'feps', eps_22592)

# Assigning a Dict to a Name (line 27):

# Assigning a Dict to a Name (line 27):

# Obtaining an instance of the builtin type 'dict' (line 27)
dict_22593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 27)
# Adding element type (key, value) (line 27)
str_22594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 20), 'str', 'i')
int_22595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 19), dict_22593, (str_22594, int_22595))
# Adding element type (key, value) (line 27)
str_22596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'str', 'l')
int_22597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 33), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 19), dict_22593, (str_22596, int_22597))
# Adding element type (key, value) (line 27)
str_22598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 36), 'str', 'f')
int_22599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 41), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 19), dict_22593, (str_22598, int_22599))
# Adding element type (key, value) (line 27)
str_22600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 44), 'str', 'd')
int_22601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 49), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 19), dict_22593, (str_22600, int_22601))
# Adding element type (key, value) (line 27)
str_22602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 52), 'str', 'F')
int_22603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 57), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 19), dict_22593, (str_22602, int_22603))
# Adding element type (key, value) (line 27)
str_22604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 60), 'str', 'D')
int_22605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 65), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 19), dict_22593, (str_22604, int_22605))

# Assigning a type to the variable '_array_precision' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), '_array_precision', dict_22593)

@norecursion
def _asarray_square(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_asarray_square'
    module_type_store = module_type_store.open_function_context('_asarray_square', 34, 0, False)
    
    # Passed parameters checking function
    _asarray_square.stypy_localization = localization
    _asarray_square.stypy_type_of_self = None
    _asarray_square.stypy_type_store = module_type_store
    _asarray_square.stypy_function_name = '_asarray_square'
    _asarray_square.stypy_param_names_list = ['A']
    _asarray_square.stypy_varargs_param_name = None
    _asarray_square.stypy_kwargs_param_name = None
    _asarray_square.stypy_call_defaults = defaults
    _asarray_square.stypy_call_varargs = varargs
    _asarray_square.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_asarray_square', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_asarray_square', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_asarray_square(...)' code ##################

    str_22606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, (-1)), 'str', '\n    Wraps asarray with the extra requirement that the input be a square matrix.\n\n    The motivation is that the matfuncs module has real functions that have\n    been lifted to square matrix functions.\n\n    Parameters\n    ----------\n    A : array_like\n        A square matrix.\n\n    Returns\n    -------\n    out : ndarray\n        An ndarray copy or view or other representation of A.\n\n    ')
    
    # Assigning a Call to a Name (line 52):
    
    # Assigning a Call to a Name (line 52):
    
    # Call to asarray(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'A' (line 52)
    A_22609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'A', False)
    # Processing the call keyword arguments (line 52)
    kwargs_22610 = {}
    # Getting the type of 'np' (line 52)
    np_22607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 52)
    asarray_22608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), np_22607, 'asarray')
    # Calling asarray(args, kwargs) (line 52)
    asarray_call_result_22611 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), asarray_22608, *[A_22609], **kwargs_22610)
    
    # Assigning a type to the variable 'A' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'A', asarray_call_result_22611)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'A' (line 53)
    A_22613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 53)
    shape_22614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 11), A_22613, 'shape')
    # Processing the call keyword arguments (line 53)
    kwargs_22615 = {}
    # Getting the type of 'len' (line 53)
    len_22612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 7), 'len', False)
    # Calling len(args, kwargs) (line 53)
    len_call_result_22616 = invoke(stypy.reporting.localization.Localization(__file__, 53, 7), len_22612, *[shape_22614], **kwargs_22615)
    
    int_22617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 23), 'int')
    # Applying the binary operator '!=' (line 53)
    result_ne_22618 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 7), '!=', len_call_result_22616, int_22617)
    
    
    
    # Obtaining the type of the subscript
    int_22619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 36), 'int')
    # Getting the type of 'A' (line 53)
    A_22620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 28), 'A')
    # Obtaining the member 'shape' of a type (line 53)
    shape_22621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 28), A_22620, 'shape')
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___22622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 28), shape_22621, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_22623 = invoke(stypy.reporting.localization.Localization(__file__, 53, 28), getitem___22622, int_22619)
    
    
    # Obtaining the type of the subscript
    int_22624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 50), 'int')
    # Getting the type of 'A' (line 53)
    A_22625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 42), 'A')
    # Obtaining the member 'shape' of a type (line 53)
    shape_22626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 42), A_22625, 'shape')
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___22627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 42), shape_22626, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_22628 = invoke(stypy.reporting.localization.Localization(__file__, 53, 42), getitem___22627, int_22624)
    
    # Applying the binary operator '!=' (line 53)
    result_ne_22629 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 28), '!=', subscript_call_result_22623, subscript_call_result_22628)
    
    # Applying the binary operator 'or' (line 53)
    result_or_keyword_22630 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 7), 'or', result_ne_22618, result_ne_22629)
    
    # Testing the type of an if condition (line 53)
    if_condition_22631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 4), result_or_keyword_22630)
    # Assigning a type to the variable 'if_condition_22631' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'if_condition_22631', if_condition_22631)
    # SSA begins for if statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 54)
    # Processing the call arguments (line 54)
    str_22633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'str', 'expected square array_like input')
    # Processing the call keyword arguments (line 54)
    kwargs_22634 = {}
    # Getting the type of 'ValueError' (line 54)
    ValueError_22632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 54)
    ValueError_call_result_22635 = invoke(stypy.reporting.localization.Localization(__file__, 54, 14), ValueError_22632, *[str_22633], **kwargs_22634)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 54, 8), ValueError_call_result_22635, 'raise parameter', BaseException)
    # SSA join for if statement (line 53)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'A' (line 55)
    A_22636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'A')
    # Assigning a type to the variable 'stypy_return_type' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type', A_22636)
    
    # ################# End of '_asarray_square(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_asarray_square' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_22637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22637)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_asarray_square'
    return stypy_return_type_22637

# Assigning a type to the variable '_asarray_square' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), '_asarray_square', _asarray_square)

@norecursion
def _maybe_real(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 58)
    None_22638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'None')
    defaults = [None_22638]
    # Create a new context for function '_maybe_real'
    module_type_store = module_type_store.open_function_context('_maybe_real', 58, 0, False)
    
    # Passed parameters checking function
    _maybe_real.stypy_localization = localization
    _maybe_real.stypy_type_of_self = None
    _maybe_real.stypy_type_store = module_type_store
    _maybe_real.stypy_function_name = '_maybe_real'
    _maybe_real.stypy_param_names_list = ['A', 'B', 'tol']
    _maybe_real.stypy_varargs_param_name = None
    _maybe_real.stypy_kwargs_param_name = None
    _maybe_real.stypy_call_defaults = defaults
    _maybe_real.stypy_call_varargs = varargs
    _maybe_real.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_maybe_real', ['A', 'B', 'tol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_maybe_real', localization, ['A', 'B', 'tol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_maybe_real(...)' code ##################

    str_22639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, (-1)), 'str', '\n    Return either B or the real part of B, depending on properties of A and B.\n\n    The motivation is that B has been computed as a complicated function of A,\n    and B may be perturbed by negligible imaginary components.\n    If A is real and B is complex with small imaginary components,\n    then return a real copy of B.  The assumption in that case would be that\n    the imaginary components of B are numerical artifacts.\n\n    Parameters\n    ----------\n    A : ndarray\n        Input array whose type is to be checked as real vs. complex.\n    B : ndarray\n        Array to be returned, possibly without its imaginary part.\n    tol : float\n        Absolute tolerance.\n\n    Returns\n    -------\n    out : real or complex array\n        Either the input array B or only the real part of the input array B.\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Call to isrealobj(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'A' (line 84)
    A_22642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'A', False)
    # Processing the call keyword arguments (line 84)
    kwargs_22643 = {}
    # Getting the type of 'np' (line 84)
    np_22640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 7), 'np', False)
    # Obtaining the member 'isrealobj' of a type (line 84)
    isrealobj_22641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 7), np_22640, 'isrealobj')
    # Calling isrealobj(args, kwargs) (line 84)
    isrealobj_call_result_22644 = invoke(stypy.reporting.localization.Localization(__file__, 84, 7), isrealobj_22641, *[A_22642], **kwargs_22643)
    
    
    # Call to iscomplexobj(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'B' (line 84)
    B_22647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 43), 'B', False)
    # Processing the call keyword arguments (line 84)
    kwargs_22648 = {}
    # Getting the type of 'np' (line 84)
    np_22645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 84)
    iscomplexobj_22646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 27), np_22645, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 84)
    iscomplexobj_call_result_22649 = invoke(stypy.reporting.localization.Localization(__file__, 84, 27), iscomplexobj_22646, *[B_22647], **kwargs_22648)
    
    # Applying the binary operator 'and' (line 84)
    result_and_keyword_22650 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 7), 'and', isrealobj_call_result_22644, iscomplexobj_call_result_22649)
    
    # Testing the type of an if condition (line 84)
    if_condition_22651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 4), result_and_keyword_22650)
    # Assigning a type to the variable 'if_condition_22651' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'if_condition_22651', if_condition_22651)
    # SSA begins for if statement (line 84)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 85)
    # Getting the type of 'tol' (line 85)
    tol_22652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'tol')
    # Getting the type of 'None' (line 85)
    None_22653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 18), 'None')
    
    (may_be_22654, more_types_in_union_22655) = may_be_none(tol_22652, None_22653)

    if may_be_22654:

        if more_types_in_union_22655:
            # Runtime conditional SSA (line 85)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 86):
        
        # Assigning a Subscript to a Name (line 86):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        # Getting the type of 'B' (line 86)
        B_22656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 59), 'B')
        # Obtaining the member 'dtype' of a type (line 86)
        dtype_22657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 59), B_22656, 'dtype')
        # Obtaining the member 'char' of a type (line 86)
        char_22658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 59), dtype_22657, 'char')
        # Getting the type of '_array_precision' (line 86)
        _array_precision_22659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 42), '_array_precision')
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___22660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 42), _array_precision_22659, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_22661 = invoke(stypy.reporting.localization.Localization(__file__, 86, 42), getitem___22660, char_22658)
        
        
        # Obtaining an instance of the builtin type 'dict' (line 86)
        dict_22662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 86)
        # Adding element type (key, value) (line 86)
        int_22663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 19), 'int')
        # Getting the type of 'feps' (line 86)
        feps_22664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'feps')
        float_22665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 26), 'float')
        # Applying the binary operator '*' (line 86)
        result_mul_22666 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 21), '*', feps_22664, float_22665)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 18), dict_22662, (int_22663, result_mul_22666))
        # Adding element type (key, value) (line 86)
        int_22667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 31), 'int')
        # Getting the type of 'eps' (line 86)
        eps_22668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 33), 'eps')
        float_22669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 37), 'float')
        # Applying the binary operator '*' (line 86)
        result_mul_22670 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 33), '*', eps_22668, float_22669)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 18), dict_22662, (int_22667, result_mul_22670))
        
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___22671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 18), dict_22662, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_22672 = invoke(stypy.reporting.localization.Localization(__file__, 86, 18), getitem___22671, subscript_call_result_22661)
        
        # Assigning a type to the variable 'tol' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'tol', subscript_call_result_22672)

        if more_types_in_union_22655:
            # SSA join for if statement (line 85)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to allclose(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'B' (line 87)
    B_22675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'B', False)
    # Obtaining the member 'imag' of a type (line 87)
    imag_22676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 23), B_22675, 'imag')
    float_22677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 31), 'float')
    # Processing the call keyword arguments (line 87)
    # Getting the type of 'tol' (line 87)
    tol_22678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 41), 'tol', False)
    keyword_22679 = tol_22678
    kwargs_22680 = {'atol': keyword_22679}
    # Getting the type of 'np' (line 87)
    np_22673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'np', False)
    # Obtaining the member 'allclose' of a type (line 87)
    allclose_22674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 11), np_22673, 'allclose')
    # Calling allclose(args, kwargs) (line 87)
    allclose_call_result_22681 = invoke(stypy.reporting.localization.Localization(__file__, 87, 11), allclose_22674, *[imag_22676, float_22677], **kwargs_22680)
    
    # Testing the type of an if condition (line 87)
    if_condition_22682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 8), allclose_call_result_22681)
    # Assigning a type to the variable 'if_condition_22682' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'if_condition_22682', if_condition_22682)
    # SSA begins for if statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 88):
    
    # Assigning a Attribute to a Name (line 88):
    # Getting the type of 'B' (line 88)
    B_22683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'B')
    # Obtaining the member 'real' of a type (line 88)
    real_22684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 16), B_22683, 'real')
    # Assigning a type to the variable 'B' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'B', real_22684)
    # SSA join for if statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 84)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'B' (line 89)
    B_22685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'B')
    # Assigning a type to the variable 'stypy_return_type' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type', B_22685)
    
    # ################# End of '_maybe_real(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_maybe_real' in the type store
    # Getting the type of 'stypy_return_type' (line 58)
    stypy_return_type_22686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22686)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_maybe_real'
    return stypy_return_type_22686

# Assigning a type to the variable '_maybe_real' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), '_maybe_real', _maybe_real)

@norecursion
def fractional_matrix_power(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fractional_matrix_power'
    module_type_store = module_type_store.open_function_context('fractional_matrix_power', 96, 0, False)
    
    # Passed parameters checking function
    fractional_matrix_power.stypy_localization = localization
    fractional_matrix_power.stypy_type_of_self = None
    fractional_matrix_power.stypy_type_store = module_type_store
    fractional_matrix_power.stypy_function_name = 'fractional_matrix_power'
    fractional_matrix_power.stypy_param_names_list = ['A', 't']
    fractional_matrix_power.stypy_varargs_param_name = None
    fractional_matrix_power.stypy_kwargs_param_name = None
    fractional_matrix_power.stypy_call_defaults = defaults
    fractional_matrix_power.stypy_call_varargs = varargs
    fractional_matrix_power.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fractional_matrix_power', ['A', 't'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fractional_matrix_power', localization, ['A', 't'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fractional_matrix_power(...)' code ##################

    str_22687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, (-1)), 'str', '\n    Compute the fractional power of a matrix.\n\n    Proceeds according to the discussion in section (6) of [1]_.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Matrix whose fractional power to evaluate.\n    t : float\n        Fractional power.\n\n    Returns\n    -------\n    X : (N, N) array_like\n        The fractional power of the matrix.\n\n    References\n    ----------\n    .. [1] Nicholas J. Higham and Lijing lin (2011)\n           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."\n           SIAM Journal on Matrix Analysis and Applications,\n           32 (3). pp. 1056-1078. ISSN 0895-4798\n\n    Examples\n    --------\n    >>> from scipy.linalg import fractional_matrix_power\n    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])\n    >>> b = fractional_matrix_power(a, 0.5)\n    >>> b\n    array([[ 0.75592895,  1.13389342],\n           [ 0.37796447,  1.88982237]])\n    >>> np.dot(b, b)      # Verify square root\n    array([[ 1.,  3.],\n           [ 1.,  4.]])\n\n    ')
    
    # Assigning a Call to a Name (line 136):
    
    # Assigning a Call to a Name (line 136):
    
    # Call to _asarray_square(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'A' (line 136)
    A_22689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'A', False)
    # Processing the call keyword arguments (line 136)
    kwargs_22690 = {}
    # Getting the type of '_asarray_square' (line 136)
    _asarray_square_22688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), '_asarray_square', False)
    # Calling _asarray_square(args, kwargs) (line 136)
    _asarray_square_call_result_22691 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), _asarray_square_22688, *[A_22689], **kwargs_22690)
    
    # Assigning a type to the variable 'A' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'A', _asarray_square_call_result_22691)
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 137, 4))
    
    # 'import scipy.linalg._matfuncs_inv_ssq' statement (line 137)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_22692 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 137, 4), 'scipy.linalg._matfuncs_inv_ssq')

    if (type(import_22692) is not StypyTypeError):

        if (import_22692 != 'pyd_module'):
            __import__(import_22692)
            sys_modules_22693 = sys.modules[import_22692]
            import_module(stypy.reporting.localization.Localization(__file__, 137, 4), 'scipy.linalg._matfuncs_inv_ssq', sys_modules_22693.module_type_store, module_type_store)
        else:
            import scipy.linalg._matfuncs_inv_ssq

            import_module(stypy.reporting.localization.Localization(__file__, 137, 4), 'scipy.linalg._matfuncs_inv_ssq', scipy.linalg._matfuncs_inv_ssq, module_type_store)

    else:
        # Assigning a type to the variable 'scipy.linalg._matfuncs_inv_ssq' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'scipy.linalg._matfuncs_inv_ssq', import_22692)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    # Call to _fractional_matrix_power(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'A' (line 138)
    A_22698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 67), 'A', False)
    # Getting the type of 't' (line 138)
    t_22699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 70), 't', False)
    # Processing the call keyword arguments (line 138)
    kwargs_22700 = {}
    # Getting the type of 'scipy' (line 138)
    scipy_22694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 138)
    linalg_22695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 11), scipy_22694, 'linalg')
    # Obtaining the member '_matfuncs_inv_ssq' of a type (line 138)
    _matfuncs_inv_ssq_22696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 11), linalg_22695, '_matfuncs_inv_ssq')
    # Obtaining the member '_fractional_matrix_power' of a type (line 138)
    _fractional_matrix_power_22697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 11), _matfuncs_inv_ssq_22696, '_fractional_matrix_power')
    # Calling _fractional_matrix_power(args, kwargs) (line 138)
    _fractional_matrix_power_call_result_22701 = invoke(stypy.reporting.localization.Localization(__file__, 138, 11), _fractional_matrix_power_22697, *[A_22698, t_22699], **kwargs_22700)
    
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type', _fractional_matrix_power_call_result_22701)
    
    # ################# End of 'fractional_matrix_power(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fractional_matrix_power' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_22702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22702)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fractional_matrix_power'
    return stypy_return_type_22702

# Assigning a type to the variable 'fractional_matrix_power' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'fractional_matrix_power', fractional_matrix_power)

@norecursion
def logm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 141)
    True_22703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 17), 'True')
    defaults = [True_22703]
    # Create a new context for function 'logm'
    module_type_store = module_type_store.open_function_context('logm', 141, 0, False)
    
    # Passed parameters checking function
    logm.stypy_localization = localization
    logm.stypy_type_of_self = None
    logm.stypy_type_store = module_type_store
    logm.stypy_function_name = 'logm'
    logm.stypy_param_names_list = ['A', 'disp']
    logm.stypy_varargs_param_name = None
    logm.stypy_kwargs_param_name = None
    logm.stypy_call_defaults = defaults
    logm.stypy_call_varargs = varargs
    logm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'logm', ['A', 'disp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'logm', localization, ['A', 'disp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'logm(...)' code ##################

    str_22704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, (-1)), 'str', '\n    Compute matrix logarithm.\n\n    The matrix logarithm is the inverse of\n    expm: expm(logm(`A`)) == `A`\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Matrix whose logarithm to evaluate\n    disp : bool, optional\n        Print warning if error in the result is estimated large\n        instead of returning estimated error. (Default: True)\n\n    Returns\n    -------\n    logm : (N, N) ndarray\n        Matrix logarithm of `A`\n    errest : float\n        (if disp == False)\n\n        1-norm of the estimated error, ||err||_1 / ||A||_1\n\n    References\n    ----------\n    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)\n           "Improved Inverse Scaling and Squaring Algorithms\n           for the Matrix Logarithm."\n           SIAM Journal on Scientific Computing, 34 (4). C152-C169.\n           ISSN 1095-7197\n\n    .. [2] Nicholas J. Higham (2008)\n           "Functions of Matrices: Theory and Computation"\n           ISBN 978-0-898716-46-7\n\n    .. [3] Nicholas J. Higham and Lijing lin (2011)\n           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."\n           SIAM Journal on Matrix Analysis and Applications,\n           32 (3). pp. 1056-1078. ISSN 0895-4798\n\n    Examples\n    --------\n    >>> from scipy.linalg import logm, expm\n    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])\n    >>> b = logm(a)\n    >>> b\n    array([[-1.02571087,  2.05142174],\n           [ 0.68380725,  1.02571087]])\n    >>> expm(b)         # Verify expm(logm(a)) returns a\n    array([[ 1.,  3.],\n           [ 1.,  4.]])\n\n    ')
    
    # Assigning a Call to a Name (line 195):
    
    # Assigning a Call to a Name (line 195):
    
    # Call to _asarray_square(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'A' (line 195)
    A_22706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 24), 'A', False)
    # Processing the call keyword arguments (line 195)
    kwargs_22707 = {}
    # Getting the type of '_asarray_square' (line 195)
    _asarray_square_22705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), '_asarray_square', False)
    # Calling _asarray_square(args, kwargs) (line 195)
    _asarray_square_call_result_22708 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), _asarray_square_22705, *[A_22706], **kwargs_22707)
    
    # Assigning a type to the variable 'A' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'A', _asarray_square_call_result_22708)
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 197, 4))
    
    # 'import scipy.linalg._matfuncs_inv_ssq' statement (line 197)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_22709 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 197, 4), 'scipy.linalg._matfuncs_inv_ssq')

    if (type(import_22709) is not StypyTypeError):

        if (import_22709 != 'pyd_module'):
            __import__(import_22709)
            sys_modules_22710 = sys.modules[import_22709]
            import_module(stypy.reporting.localization.Localization(__file__, 197, 4), 'scipy.linalg._matfuncs_inv_ssq', sys_modules_22710.module_type_store, module_type_store)
        else:
            import scipy.linalg._matfuncs_inv_ssq

            import_module(stypy.reporting.localization.Localization(__file__, 197, 4), 'scipy.linalg._matfuncs_inv_ssq', scipy.linalg._matfuncs_inv_ssq, module_type_store)

    else:
        # Assigning a type to the variable 'scipy.linalg._matfuncs_inv_ssq' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'scipy.linalg._matfuncs_inv_ssq', import_22709)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to _logm(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'A' (line 198)
    A_22715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 45), 'A', False)
    # Processing the call keyword arguments (line 198)
    kwargs_22716 = {}
    # Getting the type of 'scipy' (line 198)
    scipy_22711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 198)
    linalg_22712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), scipy_22711, 'linalg')
    # Obtaining the member '_matfuncs_inv_ssq' of a type (line 198)
    _matfuncs_inv_ssq_22713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), linalg_22712, '_matfuncs_inv_ssq')
    # Obtaining the member '_logm' of a type (line 198)
    _logm_22714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), _matfuncs_inv_ssq_22713, '_logm')
    # Calling _logm(args, kwargs) (line 198)
    _logm_call_result_22717 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), _logm_22714, *[A_22715], **kwargs_22716)
    
    # Assigning a type to the variable 'F' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'F', _logm_call_result_22717)
    
    # Assigning a Call to a Name (line 199):
    
    # Assigning a Call to a Name (line 199):
    
    # Call to _maybe_real(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'A' (line 199)
    A_22719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'A', False)
    # Getting the type of 'F' (line 199)
    F_22720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'F', False)
    # Processing the call keyword arguments (line 199)
    kwargs_22721 = {}
    # Getting the type of '_maybe_real' (line 199)
    _maybe_real_22718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), '_maybe_real', False)
    # Calling _maybe_real(args, kwargs) (line 199)
    _maybe_real_call_result_22722 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), _maybe_real_22718, *[A_22719, F_22720], **kwargs_22721)
    
    # Assigning a type to the variable 'F' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'F', _maybe_real_call_result_22722)
    
    # Assigning a BinOp to a Name (line 200):
    
    # Assigning a BinOp to a Name (line 200):
    int_22723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 13), 'int')
    # Getting the type of 'eps' (line 200)
    eps_22724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), 'eps')
    # Applying the binary operator '*' (line 200)
    result_mul_22725 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 13), '*', int_22723, eps_22724)
    
    # Assigning a type to the variable 'errtol' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'errtol', result_mul_22725)
    
    # Assigning a BinOp to a Name (line 202):
    
    # Assigning a BinOp to a Name (line 202):
    
    # Call to norm(...): (line 202)
    # Processing the call arguments (line 202)
    
    # Call to expm(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'F' (line 202)
    F_22728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'F', False)
    # Processing the call keyword arguments (line 202)
    kwargs_22729 = {}
    # Getting the type of 'expm' (line 202)
    expm_22727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'expm', False)
    # Calling expm(args, kwargs) (line 202)
    expm_call_result_22730 = invoke(stypy.reporting.localization.Localization(__file__, 202, 18), expm_22727, *[F_22728], **kwargs_22729)
    
    # Getting the type of 'A' (line 202)
    A_22731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 26), 'A', False)
    # Applying the binary operator '-' (line 202)
    result_sub_22732 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 18), '-', expm_call_result_22730, A_22731)
    
    int_22733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 28), 'int')
    # Processing the call keyword arguments (line 202)
    kwargs_22734 = {}
    # Getting the type of 'norm' (line 202)
    norm_22726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 13), 'norm', False)
    # Calling norm(args, kwargs) (line 202)
    norm_call_result_22735 = invoke(stypy.reporting.localization.Localization(__file__, 202, 13), norm_22726, *[result_sub_22732, int_22733], **kwargs_22734)
    
    
    # Call to norm(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'A' (line 202)
    A_22737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 38), 'A', False)
    int_22738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 40), 'int')
    # Processing the call keyword arguments (line 202)
    kwargs_22739 = {}
    # Getting the type of 'norm' (line 202)
    norm_22736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 33), 'norm', False)
    # Calling norm(args, kwargs) (line 202)
    norm_call_result_22740 = invoke(stypy.reporting.localization.Localization(__file__, 202, 33), norm_22736, *[A_22737, int_22738], **kwargs_22739)
    
    # Applying the binary operator 'div' (line 202)
    result_div_22741 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 13), 'div', norm_call_result_22735, norm_call_result_22740)
    
    # Assigning a type to the variable 'errest' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'errest', result_div_22741)
    
    # Getting the type of 'disp' (line 203)
    disp_22742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 7), 'disp')
    # Testing the type of an if condition (line 203)
    if_condition_22743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 4), disp_22742)
    # Assigning a type to the variable 'if_condition_22743' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'if_condition_22743', if_condition_22743)
    # SSA begins for if statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    
    # Call to isfinite(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'errest' (line 204)
    errest_22745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 24), 'errest', False)
    # Processing the call keyword arguments (line 204)
    kwargs_22746 = {}
    # Getting the type of 'isfinite' (line 204)
    isfinite_22744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 15), 'isfinite', False)
    # Calling isfinite(args, kwargs) (line 204)
    isfinite_call_result_22747 = invoke(stypy.reporting.localization.Localization(__file__, 204, 15), isfinite_22744, *[errest_22745], **kwargs_22746)
    
    # Applying the 'not' unary operator (line 204)
    result_not__22748 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 11), 'not', isfinite_call_result_22747)
    
    
    # Getting the type of 'errest' (line 204)
    errest_22749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 35), 'errest')
    # Getting the type of 'errtol' (line 204)
    errtol_22750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 45), 'errtol')
    # Applying the binary operator '>=' (line 204)
    result_ge_22751 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 35), '>=', errest_22749, errtol_22750)
    
    # Applying the binary operator 'or' (line 204)
    result_or_keyword_22752 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 11), 'or', result_not__22748, result_ge_22751)
    
    # Testing the type of an if condition (line 204)
    if_condition_22753 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 8), result_or_keyword_22752)
    # Assigning a type to the variable 'if_condition_22753' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'if_condition_22753', if_condition_22753)
    # SSA begins for if statement (line 204)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 205)
    # Processing the call arguments (line 205)
    str_22755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 18), 'str', 'logm result may be inaccurate, approximate err =')
    # Getting the type of 'errest' (line 205)
    errest_22756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 70), 'errest', False)
    # Processing the call keyword arguments (line 205)
    kwargs_22757 = {}
    # Getting the type of 'print' (line 205)
    print_22754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'print', False)
    # Calling print(args, kwargs) (line 205)
    print_call_result_22758 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), print_22754, *[str_22755, errest_22756], **kwargs_22757)
    
    # SSA join for if statement (line 204)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'F' (line 206)
    F_22759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), 'F')
    # Assigning a type to the variable 'stypy_return_type' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'stypy_return_type', F_22759)
    # SSA branch for the else part of an if statement (line 203)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 208)
    tuple_22760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 208)
    # Adding element type (line 208)
    # Getting the type of 'F' (line 208)
    F_22761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'F')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), tuple_22760, F_22761)
    # Adding element type (line 208)
    # Getting the type of 'errest' (line 208)
    errest_22762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 18), 'errest')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), tuple_22760, errest_22762)
    
    # Assigning a type to the variable 'stypy_return_type' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', tuple_22760)
    # SSA join for if statement (line 203)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'logm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'logm' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_22763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22763)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'logm'
    return stypy_return_type_22763

# Assigning a type to the variable 'logm' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'logm', logm)

@norecursion
def expm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'expm'
    module_type_store = module_type_store.open_function_context('expm', 211, 0, False)
    
    # Passed parameters checking function
    expm.stypy_localization = localization
    expm.stypy_type_of_self = None
    expm.stypy_type_store = module_type_store
    expm.stypy_function_name = 'expm'
    expm.stypy_param_names_list = ['A']
    expm.stypy_varargs_param_name = None
    expm.stypy_kwargs_param_name = None
    expm.stypy_call_defaults = defaults
    expm.stypy_call_varargs = varargs
    expm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expm', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expm', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expm(...)' code ##################

    str_22764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, (-1)), 'str', '\n    Compute the matrix exponential using Pade approximation.\n\n    Parameters\n    ----------\n    A : (N, N) array_like or sparse matrix\n        Matrix to be exponentiated.\n\n    Returns\n    -------\n    expm : (N, N) ndarray\n        Matrix exponential of `A`.\n\n    References\n    ----------\n    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)\n           "A New Scaling and Squaring Algorithm for the Matrix Exponential."\n           SIAM Journal on Matrix Analysis and Applications.\n           31 (3). pp. 970-989. ISSN 1095-7162\n\n    Examples\n    --------\n    >>> from scipy.linalg import expm, sinm, cosm\n\n    Matrix version of the formula exp(0) = 1:\n\n    >>> expm(np.zeros((2,2)))\n    array([[ 1.,  0.],\n           [ 0.,  1.]])\n\n    Euler\'s identity (exp(i*theta) = cos(theta) + i*sin(theta))\n    applied to a matrix:\n\n    >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])\n    >>> expm(1j*a)\n    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],\n           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])\n    >>> cosm(a) + 1j*sinm(a)\n    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],\n           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])\n\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 255, 4))
    
    # 'import scipy.sparse.linalg' statement (line 255)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_22765 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 255, 4), 'scipy.sparse.linalg')

    if (type(import_22765) is not StypyTypeError):

        if (import_22765 != 'pyd_module'):
            __import__(import_22765)
            sys_modules_22766 = sys.modules[import_22765]
            import_module(stypy.reporting.localization.Localization(__file__, 255, 4), 'scipy.sparse.linalg', sys_modules_22766.module_type_store, module_type_store)
        else:
            import scipy.sparse.linalg

            import_module(stypy.reporting.localization.Localization(__file__, 255, 4), 'scipy.sparse.linalg', scipy.sparse.linalg, module_type_store)

    else:
        # Assigning a type to the variable 'scipy.sparse.linalg' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'scipy.sparse.linalg', import_22765)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    # Call to expm(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'A' (line 256)
    A_22771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 36), 'A', False)
    # Processing the call keyword arguments (line 256)
    kwargs_22772 = {}
    # Getting the type of 'scipy' (line 256)
    scipy_22767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 256)
    sparse_22768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 11), scipy_22767, 'sparse')
    # Obtaining the member 'linalg' of a type (line 256)
    linalg_22769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 11), sparse_22768, 'linalg')
    # Obtaining the member 'expm' of a type (line 256)
    expm_22770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 11), linalg_22769, 'expm')
    # Calling expm(args, kwargs) (line 256)
    expm_call_result_22773 = invoke(stypy.reporting.localization.Localization(__file__, 256, 11), expm_22770, *[A_22771], **kwargs_22772)
    
    # Assigning a type to the variable 'stypy_return_type' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type', expm_call_result_22773)
    
    # ################# End of 'expm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expm' in the type store
    # Getting the type of 'stypy_return_type' (line 211)
    stypy_return_type_22774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22774)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expm'
    return stypy_return_type_22774

# Assigning a type to the variable 'expm' (line 211)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), 'expm', expm)

@norecursion
def cosm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cosm'
    module_type_store = module_type_store.open_function_context('cosm', 259, 0, False)
    
    # Passed parameters checking function
    cosm.stypy_localization = localization
    cosm.stypy_type_of_self = None
    cosm.stypy_type_store = module_type_store
    cosm.stypy_function_name = 'cosm'
    cosm.stypy_param_names_list = ['A']
    cosm.stypy_varargs_param_name = None
    cosm.stypy_kwargs_param_name = None
    cosm.stypy_call_defaults = defaults
    cosm.stypy_call_varargs = varargs
    cosm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cosm', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cosm', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cosm(...)' code ##################

    str_22775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, (-1)), 'str', "\n    Compute the matrix cosine.\n\n    This routine uses expm to compute the matrix exponentials.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Input array\n\n    Returns\n    -------\n    cosm : (N, N) ndarray\n        Matrix cosine of A\n\n    Examples\n    --------\n    >>> from scipy.linalg import expm, sinm, cosm\n\n    Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))\n    applied to a matrix:\n\n    >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])\n    >>> expm(1j*a)\n    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],\n           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])\n    >>> cosm(a) + 1j*sinm(a)\n    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],\n           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])\n\n    ")
    
    # Assigning a Call to a Name (line 291):
    
    # Assigning a Call to a Name (line 291):
    
    # Call to _asarray_square(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'A' (line 291)
    A_22777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'A', False)
    # Processing the call keyword arguments (line 291)
    kwargs_22778 = {}
    # Getting the type of '_asarray_square' (line 291)
    _asarray_square_22776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), '_asarray_square', False)
    # Calling _asarray_square(args, kwargs) (line 291)
    _asarray_square_call_result_22779 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), _asarray_square_22776, *[A_22777], **kwargs_22778)
    
    # Assigning a type to the variable 'A' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'A', _asarray_square_call_result_22779)
    
    
    # Call to iscomplexobj(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'A' (line 292)
    A_22782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 23), 'A', False)
    # Processing the call keyword arguments (line 292)
    kwargs_22783 = {}
    # Getting the type of 'np' (line 292)
    np_22780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 7), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 292)
    iscomplexobj_22781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 7), np_22780, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 292)
    iscomplexobj_call_result_22784 = invoke(stypy.reporting.localization.Localization(__file__, 292, 7), iscomplexobj_22781, *[A_22782], **kwargs_22783)
    
    # Testing the type of an if condition (line 292)
    if_condition_22785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 4), iscomplexobj_call_result_22784)
    # Assigning a type to the variable 'if_condition_22785' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'if_condition_22785', if_condition_22785)
    # SSA begins for if statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    float_22786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 15), 'float')
    
    # Call to expm(...): (line 293)
    # Processing the call arguments (line 293)
    complex_22788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 25), 'complex')
    # Getting the type of 'A' (line 293)
    A_22789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 28), 'A', False)
    # Applying the binary operator '*' (line 293)
    result_mul_22790 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 25), '*', complex_22788, A_22789)
    
    # Processing the call keyword arguments (line 293)
    kwargs_22791 = {}
    # Getting the type of 'expm' (line 293)
    expm_22787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 20), 'expm', False)
    # Calling expm(args, kwargs) (line 293)
    expm_call_result_22792 = invoke(stypy.reporting.localization.Localization(__file__, 293, 20), expm_22787, *[result_mul_22790], **kwargs_22791)
    
    
    # Call to expm(...): (line 293)
    # Processing the call arguments (line 293)
    complex_22794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 38), 'complex')
    # Getting the type of 'A' (line 293)
    A_22795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 42), 'A', False)
    # Applying the binary operator '*' (line 293)
    result_mul_22796 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 38), '*', complex_22794, A_22795)
    
    # Processing the call keyword arguments (line 293)
    kwargs_22797 = {}
    # Getting the type of 'expm' (line 293)
    expm_22793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 33), 'expm', False)
    # Calling expm(args, kwargs) (line 293)
    expm_call_result_22798 = invoke(stypy.reporting.localization.Localization(__file__, 293, 33), expm_22793, *[result_mul_22796], **kwargs_22797)
    
    # Applying the binary operator '+' (line 293)
    result_add_22799 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 20), '+', expm_call_result_22792, expm_call_result_22798)
    
    # Applying the binary operator '*' (line 293)
    result_mul_22800 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 15), '*', float_22786, result_add_22799)
    
    # Assigning a type to the variable 'stypy_return_type' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'stypy_return_type', result_mul_22800)
    # SSA branch for the else part of an if statement (line 292)
    module_type_store.open_ssa_branch('else')
    
    # Call to expm(...): (line 295)
    # Processing the call arguments (line 295)
    complex_22802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 20), 'complex')
    # Getting the type of 'A' (line 295)
    A_22803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 23), 'A', False)
    # Applying the binary operator '*' (line 295)
    result_mul_22804 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 20), '*', complex_22802, A_22803)
    
    # Processing the call keyword arguments (line 295)
    kwargs_22805 = {}
    # Getting the type of 'expm' (line 295)
    expm_22801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'expm', False)
    # Calling expm(args, kwargs) (line 295)
    expm_call_result_22806 = invoke(stypy.reporting.localization.Localization(__file__, 295, 15), expm_22801, *[result_mul_22804], **kwargs_22805)
    
    # Obtaining the member 'real' of a type (line 295)
    real_22807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 15), expm_call_result_22806, 'real')
    # Assigning a type to the variable 'stypy_return_type' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'stypy_return_type', real_22807)
    # SSA join for if statement (line 292)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'cosm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cosm' in the type store
    # Getting the type of 'stypy_return_type' (line 259)
    stypy_return_type_22808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22808)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cosm'
    return stypy_return_type_22808

# Assigning a type to the variable 'cosm' (line 259)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), 'cosm', cosm)

@norecursion
def sinm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sinm'
    module_type_store = module_type_store.open_function_context('sinm', 298, 0, False)
    
    # Passed parameters checking function
    sinm.stypy_localization = localization
    sinm.stypy_type_of_self = None
    sinm.stypy_type_store = module_type_store
    sinm.stypy_function_name = 'sinm'
    sinm.stypy_param_names_list = ['A']
    sinm.stypy_varargs_param_name = None
    sinm.stypy_kwargs_param_name = None
    sinm.stypy_call_defaults = defaults
    sinm.stypy_call_varargs = varargs
    sinm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sinm', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sinm', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sinm(...)' code ##################

    str_22809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, (-1)), 'str', "\n    Compute the matrix sine.\n\n    This routine uses expm to compute the matrix exponentials.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Input array.\n\n    Returns\n    -------\n    sinm : (N, N) ndarray\n        Matrix sine of `A`\n\n    Examples\n    --------\n    >>> from scipy.linalg import expm, sinm, cosm\n\n    Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))\n    applied to a matrix:\n\n    >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])\n    >>> expm(1j*a)\n    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],\n           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])\n    >>> cosm(a) + 1j*sinm(a)\n    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],\n           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])\n\n    ")
    
    # Assigning a Call to a Name (line 330):
    
    # Assigning a Call to a Name (line 330):
    
    # Call to _asarray_square(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'A' (line 330)
    A_22811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 24), 'A', False)
    # Processing the call keyword arguments (line 330)
    kwargs_22812 = {}
    # Getting the type of '_asarray_square' (line 330)
    _asarray_square_22810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), '_asarray_square', False)
    # Calling _asarray_square(args, kwargs) (line 330)
    _asarray_square_call_result_22813 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), _asarray_square_22810, *[A_22811], **kwargs_22812)
    
    # Assigning a type to the variable 'A' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'A', _asarray_square_call_result_22813)
    
    
    # Call to iscomplexobj(...): (line 331)
    # Processing the call arguments (line 331)
    # Getting the type of 'A' (line 331)
    A_22816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 23), 'A', False)
    # Processing the call keyword arguments (line 331)
    kwargs_22817 = {}
    # Getting the type of 'np' (line 331)
    np_22814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 7), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 331)
    iscomplexobj_22815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 7), np_22814, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 331)
    iscomplexobj_call_result_22818 = invoke(stypy.reporting.localization.Localization(__file__, 331, 7), iscomplexobj_22815, *[A_22816], **kwargs_22817)
    
    # Testing the type of an if condition (line 331)
    if_condition_22819 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 4), iscomplexobj_call_result_22818)
    # Assigning a type to the variable 'if_condition_22819' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'if_condition_22819', if_condition_22819)
    # SSA begins for if statement (line 331)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    complex_22820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 15), 'complex')
    
    # Call to expm(...): (line 332)
    # Processing the call arguments (line 332)
    complex_22822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 27), 'complex')
    # Getting the type of 'A' (line 332)
    A_22823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 30), 'A', False)
    # Applying the binary operator '*' (line 332)
    result_mul_22824 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 27), '*', complex_22822, A_22823)
    
    # Processing the call keyword arguments (line 332)
    kwargs_22825 = {}
    # Getting the type of 'expm' (line 332)
    expm_22821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 22), 'expm', False)
    # Calling expm(args, kwargs) (line 332)
    expm_call_result_22826 = invoke(stypy.reporting.localization.Localization(__file__, 332, 22), expm_22821, *[result_mul_22824], **kwargs_22825)
    
    
    # Call to expm(...): (line 332)
    # Processing the call arguments (line 332)
    complex_22828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 40), 'complex')
    # Getting the type of 'A' (line 332)
    A_22829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 44), 'A', False)
    # Applying the binary operator '*' (line 332)
    result_mul_22830 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 40), '*', complex_22828, A_22829)
    
    # Processing the call keyword arguments (line 332)
    kwargs_22831 = {}
    # Getting the type of 'expm' (line 332)
    expm_22827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 35), 'expm', False)
    # Calling expm(args, kwargs) (line 332)
    expm_call_result_22832 = invoke(stypy.reporting.localization.Localization(__file__, 332, 35), expm_22827, *[result_mul_22830], **kwargs_22831)
    
    # Applying the binary operator '-' (line 332)
    result_sub_22833 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 22), '-', expm_call_result_22826, expm_call_result_22832)
    
    # Applying the binary operator '*' (line 332)
    result_mul_22834 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 15), '*', complex_22820, result_sub_22833)
    
    # Assigning a type to the variable 'stypy_return_type' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'stypy_return_type', result_mul_22834)
    # SSA branch for the else part of an if statement (line 331)
    module_type_store.open_ssa_branch('else')
    
    # Call to expm(...): (line 334)
    # Processing the call arguments (line 334)
    complex_22836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 20), 'complex')
    # Getting the type of 'A' (line 334)
    A_22837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 23), 'A', False)
    # Applying the binary operator '*' (line 334)
    result_mul_22838 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 20), '*', complex_22836, A_22837)
    
    # Processing the call keyword arguments (line 334)
    kwargs_22839 = {}
    # Getting the type of 'expm' (line 334)
    expm_22835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 15), 'expm', False)
    # Calling expm(args, kwargs) (line 334)
    expm_call_result_22840 = invoke(stypy.reporting.localization.Localization(__file__, 334, 15), expm_22835, *[result_mul_22838], **kwargs_22839)
    
    # Obtaining the member 'imag' of a type (line 334)
    imag_22841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 15), expm_call_result_22840, 'imag')
    # Assigning a type to the variable 'stypy_return_type' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'stypy_return_type', imag_22841)
    # SSA join for if statement (line 331)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'sinm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sinm' in the type store
    # Getting the type of 'stypy_return_type' (line 298)
    stypy_return_type_22842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22842)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sinm'
    return stypy_return_type_22842

# Assigning a type to the variable 'sinm' (line 298)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'sinm', sinm)

@norecursion
def tanm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tanm'
    module_type_store = module_type_store.open_function_context('tanm', 337, 0, False)
    
    # Passed parameters checking function
    tanm.stypy_localization = localization
    tanm.stypy_type_of_self = None
    tanm.stypy_type_store = module_type_store
    tanm.stypy_function_name = 'tanm'
    tanm.stypy_param_names_list = ['A']
    tanm.stypy_varargs_param_name = None
    tanm.stypy_kwargs_param_name = None
    tanm.stypy_call_defaults = defaults
    tanm.stypy_call_varargs = varargs
    tanm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tanm', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tanm', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tanm(...)' code ##################

    str_22843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, (-1)), 'str', '\n    Compute the matrix tangent.\n\n    This routine uses expm to compute the matrix exponentials.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Input array.\n\n    Returns\n    -------\n    tanm : (N, N) ndarray\n        Matrix tangent of `A`\n\n    Examples\n    --------\n    >>> from scipy.linalg import tanm, sinm, cosm\n    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])\n    >>> t = tanm(a)\n    >>> t\n    array([[ -2.00876993,  -8.41880636],\n           [ -2.80626879, -10.42757629]])\n\n    Verify tanm(a) = sinm(a).dot(inv(cosm(a)))\n\n    >>> s = sinm(a)\n    >>> c = cosm(a)\n    >>> s.dot(np.linalg.inv(c))\n    array([[ -2.00876993,  -8.41880636],\n           [ -2.80626879, -10.42757629]])\n\n    ')
    
    # Assigning a Call to a Name (line 371):
    
    # Assigning a Call to a Name (line 371):
    
    # Call to _asarray_square(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of 'A' (line 371)
    A_22845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'A', False)
    # Processing the call keyword arguments (line 371)
    kwargs_22846 = {}
    # Getting the type of '_asarray_square' (line 371)
    _asarray_square_22844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), '_asarray_square', False)
    # Calling _asarray_square(args, kwargs) (line 371)
    _asarray_square_call_result_22847 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), _asarray_square_22844, *[A_22845], **kwargs_22846)
    
    # Assigning a type to the variable 'A' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'A', _asarray_square_call_result_22847)
    
    # Call to _maybe_real(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'A' (line 372)
    A_22849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 23), 'A', False)
    
    # Call to solve(...): (line 372)
    # Processing the call arguments (line 372)
    
    # Call to cosm(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'A' (line 372)
    A_22852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 37), 'A', False)
    # Processing the call keyword arguments (line 372)
    kwargs_22853 = {}
    # Getting the type of 'cosm' (line 372)
    cosm_22851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 32), 'cosm', False)
    # Calling cosm(args, kwargs) (line 372)
    cosm_call_result_22854 = invoke(stypy.reporting.localization.Localization(__file__, 372, 32), cosm_22851, *[A_22852], **kwargs_22853)
    
    
    # Call to sinm(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'A' (line 372)
    A_22856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 46), 'A', False)
    # Processing the call keyword arguments (line 372)
    kwargs_22857 = {}
    # Getting the type of 'sinm' (line 372)
    sinm_22855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 41), 'sinm', False)
    # Calling sinm(args, kwargs) (line 372)
    sinm_call_result_22858 = invoke(stypy.reporting.localization.Localization(__file__, 372, 41), sinm_22855, *[A_22856], **kwargs_22857)
    
    # Processing the call keyword arguments (line 372)
    kwargs_22859 = {}
    # Getting the type of 'solve' (line 372)
    solve_22850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 26), 'solve', False)
    # Calling solve(args, kwargs) (line 372)
    solve_call_result_22860 = invoke(stypy.reporting.localization.Localization(__file__, 372, 26), solve_22850, *[cosm_call_result_22854, sinm_call_result_22858], **kwargs_22859)
    
    # Processing the call keyword arguments (line 372)
    kwargs_22861 = {}
    # Getting the type of '_maybe_real' (line 372)
    _maybe_real_22848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 11), '_maybe_real', False)
    # Calling _maybe_real(args, kwargs) (line 372)
    _maybe_real_call_result_22862 = invoke(stypy.reporting.localization.Localization(__file__, 372, 11), _maybe_real_22848, *[A_22849, solve_call_result_22860], **kwargs_22861)
    
    # Assigning a type to the variable 'stypy_return_type' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'stypy_return_type', _maybe_real_call_result_22862)
    
    # ################# End of 'tanm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tanm' in the type store
    # Getting the type of 'stypy_return_type' (line 337)
    stypy_return_type_22863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22863)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tanm'
    return stypy_return_type_22863

# Assigning a type to the variable 'tanm' (line 337)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 0), 'tanm', tanm)

@norecursion
def coshm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'coshm'
    module_type_store = module_type_store.open_function_context('coshm', 375, 0, False)
    
    # Passed parameters checking function
    coshm.stypy_localization = localization
    coshm.stypy_type_of_self = None
    coshm.stypy_type_store = module_type_store
    coshm.stypy_function_name = 'coshm'
    coshm.stypy_param_names_list = ['A']
    coshm.stypy_varargs_param_name = None
    coshm.stypy_kwargs_param_name = None
    coshm.stypy_call_defaults = defaults
    coshm.stypy_call_varargs = varargs
    coshm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'coshm', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'coshm', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'coshm(...)' code ##################

    str_22864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, (-1)), 'str', '\n    Compute the hyperbolic matrix cosine.\n\n    This routine uses expm to compute the matrix exponentials.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Input array.\n\n    Returns\n    -------\n    coshm : (N, N) ndarray\n        Hyperbolic matrix cosine of `A`\n\n    Examples\n    --------\n    >>> from scipy.linalg import tanhm, sinhm, coshm\n    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])\n    >>> c = coshm(a)\n    >>> c\n    array([[ 11.24592233,  38.76236492],\n           [ 12.92078831,  50.00828725]])\n\n    Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))\n\n    >>> t = tanhm(a)\n    >>> s = sinhm(a)\n    >>> t - s.dot(np.linalg.inv(c))\n    array([[  2.72004641e-15,   4.55191440e-15],\n           [  0.00000000e+00,  -5.55111512e-16]])\n\n    ')
    
    # Assigning a Call to a Name (line 409):
    
    # Assigning a Call to a Name (line 409):
    
    # Call to _asarray_square(...): (line 409)
    # Processing the call arguments (line 409)
    # Getting the type of 'A' (line 409)
    A_22866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 24), 'A', False)
    # Processing the call keyword arguments (line 409)
    kwargs_22867 = {}
    # Getting the type of '_asarray_square' (line 409)
    _asarray_square_22865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), '_asarray_square', False)
    # Calling _asarray_square(args, kwargs) (line 409)
    _asarray_square_call_result_22868 = invoke(stypy.reporting.localization.Localization(__file__, 409, 8), _asarray_square_22865, *[A_22866], **kwargs_22867)
    
    # Assigning a type to the variable 'A' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'A', _asarray_square_call_result_22868)
    
    # Call to _maybe_real(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 'A' (line 410)
    A_22870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 23), 'A', False)
    float_22871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 26), 'float')
    
    # Call to expm(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 'A' (line 410)
    A_22873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 38), 'A', False)
    # Processing the call keyword arguments (line 410)
    kwargs_22874 = {}
    # Getting the type of 'expm' (line 410)
    expm_22872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 33), 'expm', False)
    # Calling expm(args, kwargs) (line 410)
    expm_call_result_22875 = invoke(stypy.reporting.localization.Localization(__file__, 410, 33), expm_22872, *[A_22873], **kwargs_22874)
    
    
    # Call to expm(...): (line 410)
    # Processing the call arguments (line 410)
    
    # Getting the type of 'A' (line 410)
    A_22877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 49), 'A', False)
    # Applying the 'usub' unary operator (line 410)
    result___neg___22878 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 48), 'usub', A_22877)
    
    # Processing the call keyword arguments (line 410)
    kwargs_22879 = {}
    # Getting the type of 'expm' (line 410)
    expm_22876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 43), 'expm', False)
    # Calling expm(args, kwargs) (line 410)
    expm_call_result_22880 = invoke(stypy.reporting.localization.Localization(__file__, 410, 43), expm_22876, *[result___neg___22878], **kwargs_22879)
    
    # Applying the binary operator '+' (line 410)
    result_add_22881 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 33), '+', expm_call_result_22875, expm_call_result_22880)
    
    # Applying the binary operator '*' (line 410)
    result_mul_22882 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 26), '*', float_22871, result_add_22881)
    
    # Processing the call keyword arguments (line 410)
    kwargs_22883 = {}
    # Getting the type of '_maybe_real' (line 410)
    _maybe_real_22869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 11), '_maybe_real', False)
    # Calling _maybe_real(args, kwargs) (line 410)
    _maybe_real_call_result_22884 = invoke(stypy.reporting.localization.Localization(__file__, 410, 11), _maybe_real_22869, *[A_22870, result_mul_22882], **kwargs_22883)
    
    # Assigning a type to the variable 'stypy_return_type' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'stypy_return_type', _maybe_real_call_result_22884)
    
    # ################# End of 'coshm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'coshm' in the type store
    # Getting the type of 'stypy_return_type' (line 375)
    stypy_return_type_22885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22885)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'coshm'
    return stypy_return_type_22885

# Assigning a type to the variable 'coshm' (line 375)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 0), 'coshm', coshm)

@norecursion
def sinhm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sinhm'
    module_type_store = module_type_store.open_function_context('sinhm', 413, 0, False)
    
    # Passed parameters checking function
    sinhm.stypy_localization = localization
    sinhm.stypy_type_of_self = None
    sinhm.stypy_type_store = module_type_store
    sinhm.stypy_function_name = 'sinhm'
    sinhm.stypy_param_names_list = ['A']
    sinhm.stypy_varargs_param_name = None
    sinhm.stypy_kwargs_param_name = None
    sinhm.stypy_call_defaults = defaults
    sinhm.stypy_call_varargs = varargs
    sinhm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sinhm', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sinhm', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sinhm(...)' code ##################

    str_22886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, (-1)), 'str', '\n    Compute the hyperbolic matrix sine.\n\n    This routine uses expm to compute the matrix exponentials.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Input array.\n\n    Returns\n    -------\n    sinhm : (N, N) ndarray\n        Hyperbolic matrix sine of `A`\n\n    Examples\n    --------\n    >>> from scipy.linalg import tanhm, sinhm, coshm\n    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])\n    >>> s = sinhm(a)\n    >>> s\n    array([[ 10.57300653,  39.28826594],\n           [ 13.09608865,  49.86127247]])\n\n    Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))\n\n    >>> t = tanhm(a)\n    >>> c = coshm(a)\n    >>> t - s.dot(np.linalg.inv(c))\n    array([[  2.72004641e-15,   4.55191440e-15],\n           [  0.00000000e+00,  -5.55111512e-16]])\n\n    ')
    
    # Assigning a Call to a Name (line 447):
    
    # Assigning a Call to a Name (line 447):
    
    # Call to _asarray_square(...): (line 447)
    # Processing the call arguments (line 447)
    # Getting the type of 'A' (line 447)
    A_22888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 24), 'A', False)
    # Processing the call keyword arguments (line 447)
    kwargs_22889 = {}
    # Getting the type of '_asarray_square' (line 447)
    _asarray_square_22887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), '_asarray_square', False)
    # Calling _asarray_square(args, kwargs) (line 447)
    _asarray_square_call_result_22890 = invoke(stypy.reporting.localization.Localization(__file__, 447, 8), _asarray_square_22887, *[A_22888], **kwargs_22889)
    
    # Assigning a type to the variable 'A' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'A', _asarray_square_call_result_22890)
    
    # Call to _maybe_real(...): (line 448)
    # Processing the call arguments (line 448)
    # Getting the type of 'A' (line 448)
    A_22892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 23), 'A', False)
    float_22893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 26), 'float')
    
    # Call to expm(...): (line 448)
    # Processing the call arguments (line 448)
    # Getting the type of 'A' (line 448)
    A_22895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 38), 'A', False)
    # Processing the call keyword arguments (line 448)
    kwargs_22896 = {}
    # Getting the type of 'expm' (line 448)
    expm_22894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 33), 'expm', False)
    # Calling expm(args, kwargs) (line 448)
    expm_call_result_22897 = invoke(stypy.reporting.localization.Localization(__file__, 448, 33), expm_22894, *[A_22895], **kwargs_22896)
    
    
    # Call to expm(...): (line 448)
    # Processing the call arguments (line 448)
    
    # Getting the type of 'A' (line 448)
    A_22899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 49), 'A', False)
    # Applying the 'usub' unary operator (line 448)
    result___neg___22900 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 48), 'usub', A_22899)
    
    # Processing the call keyword arguments (line 448)
    kwargs_22901 = {}
    # Getting the type of 'expm' (line 448)
    expm_22898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 43), 'expm', False)
    # Calling expm(args, kwargs) (line 448)
    expm_call_result_22902 = invoke(stypy.reporting.localization.Localization(__file__, 448, 43), expm_22898, *[result___neg___22900], **kwargs_22901)
    
    # Applying the binary operator '-' (line 448)
    result_sub_22903 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 33), '-', expm_call_result_22897, expm_call_result_22902)
    
    # Applying the binary operator '*' (line 448)
    result_mul_22904 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 26), '*', float_22893, result_sub_22903)
    
    # Processing the call keyword arguments (line 448)
    kwargs_22905 = {}
    # Getting the type of '_maybe_real' (line 448)
    _maybe_real_22891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 11), '_maybe_real', False)
    # Calling _maybe_real(args, kwargs) (line 448)
    _maybe_real_call_result_22906 = invoke(stypy.reporting.localization.Localization(__file__, 448, 11), _maybe_real_22891, *[A_22892, result_mul_22904], **kwargs_22905)
    
    # Assigning a type to the variable 'stypy_return_type' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'stypy_return_type', _maybe_real_call_result_22906)
    
    # ################# End of 'sinhm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sinhm' in the type store
    # Getting the type of 'stypy_return_type' (line 413)
    stypy_return_type_22907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22907)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sinhm'
    return stypy_return_type_22907

# Assigning a type to the variable 'sinhm' (line 413)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 0), 'sinhm', sinhm)

@norecursion
def tanhm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tanhm'
    module_type_store = module_type_store.open_function_context('tanhm', 451, 0, False)
    
    # Passed parameters checking function
    tanhm.stypy_localization = localization
    tanhm.stypy_type_of_self = None
    tanhm.stypy_type_store = module_type_store
    tanhm.stypy_function_name = 'tanhm'
    tanhm.stypy_param_names_list = ['A']
    tanhm.stypy_varargs_param_name = None
    tanhm.stypy_kwargs_param_name = None
    tanhm.stypy_call_defaults = defaults
    tanhm.stypy_call_varargs = varargs
    tanhm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tanhm', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tanhm', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tanhm(...)' code ##################

    str_22908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, (-1)), 'str', '\n    Compute the hyperbolic matrix tangent.\n\n    This routine uses expm to compute the matrix exponentials.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Input array\n\n    Returns\n    -------\n    tanhm : (N, N) ndarray\n        Hyperbolic matrix tangent of `A`\n\n    Examples\n    --------\n    >>> from scipy.linalg import tanhm, sinhm, coshm\n    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])\n    >>> t = tanhm(a)\n    >>> t\n    array([[ 0.3428582 ,  0.51987926],\n           [ 0.17329309,  0.86273746]])\n\n    Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))\n\n    >>> s = sinhm(a)\n    >>> c = coshm(a)\n    >>> t - s.dot(np.linalg.inv(c))\n    array([[  2.72004641e-15,   4.55191440e-15],\n           [  0.00000000e+00,  -5.55111512e-16]])\n\n    ')
    
    # Assigning a Call to a Name (line 485):
    
    # Assigning a Call to a Name (line 485):
    
    # Call to _asarray_square(...): (line 485)
    # Processing the call arguments (line 485)
    # Getting the type of 'A' (line 485)
    A_22910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 24), 'A', False)
    # Processing the call keyword arguments (line 485)
    kwargs_22911 = {}
    # Getting the type of '_asarray_square' (line 485)
    _asarray_square_22909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), '_asarray_square', False)
    # Calling _asarray_square(args, kwargs) (line 485)
    _asarray_square_call_result_22912 = invoke(stypy.reporting.localization.Localization(__file__, 485, 8), _asarray_square_22909, *[A_22910], **kwargs_22911)
    
    # Assigning a type to the variable 'A' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'A', _asarray_square_call_result_22912)
    
    # Call to _maybe_real(...): (line 486)
    # Processing the call arguments (line 486)
    # Getting the type of 'A' (line 486)
    A_22914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 23), 'A', False)
    
    # Call to solve(...): (line 486)
    # Processing the call arguments (line 486)
    
    # Call to coshm(...): (line 486)
    # Processing the call arguments (line 486)
    # Getting the type of 'A' (line 486)
    A_22917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 38), 'A', False)
    # Processing the call keyword arguments (line 486)
    kwargs_22918 = {}
    # Getting the type of 'coshm' (line 486)
    coshm_22916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 32), 'coshm', False)
    # Calling coshm(args, kwargs) (line 486)
    coshm_call_result_22919 = invoke(stypy.reporting.localization.Localization(__file__, 486, 32), coshm_22916, *[A_22917], **kwargs_22918)
    
    
    # Call to sinhm(...): (line 486)
    # Processing the call arguments (line 486)
    # Getting the type of 'A' (line 486)
    A_22921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 48), 'A', False)
    # Processing the call keyword arguments (line 486)
    kwargs_22922 = {}
    # Getting the type of 'sinhm' (line 486)
    sinhm_22920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 42), 'sinhm', False)
    # Calling sinhm(args, kwargs) (line 486)
    sinhm_call_result_22923 = invoke(stypy.reporting.localization.Localization(__file__, 486, 42), sinhm_22920, *[A_22921], **kwargs_22922)
    
    # Processing the call keyword arguments (line 486)
    kwargs_22924 = {}
    # Getting the type of 'solve' (line 486)
    solve_22915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 26), 'solve', False)
    # Calling solve(args, kwargs) (line 486)
    solve_call_result_22925 = invoke(stypy.reporting.localization.Localization(__file__, 486, 26), solve_22915, *[coshm_call_result_22919, sinhm_call_result_22923], **kwargs_22924)
    
    # Processing the call keyword arguments (line 486)
    kwargs_22926 = {}
    # Getting the type of '_maybe_real' (line 486)
    _maybe_real_22913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 11), '_maybe_real', False)
    # Calling _maybe_real(args, kwargs) (line 486)
    _maybe_real_call_result_22927 = invoke(stypy.reporting.localization.Localization(__file__, 486, 11), _maybe_real_22913, *[A_22914, solve_call_result_22925], **kwargs_22926)
    
    # Assigning a type to the variable 'stypy_return_type' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'stypy_return_type', _maybe_real_call_result_22927)
    
    # ################# End of 'tanhm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tanhm' in the type store
    # Getting the type of 'stypy_return_type' (line 451)
    stypy_return_type_22928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22928)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tanhm'
    return stypy_return_type_22928

# Assigning a type to the variable 'tanhm' (line 451)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 0), 'tanhm', tanhm)

@norecursion
def funm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 489)
    True_22929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 23), 'True')
    defaults = [True_22929]
    # Create a new context for function 'funm'
    module_type_store = module_type_store.open_function_context('funm', 489, 0, False)
    
    # Passed parameters checking function
    funm.stypy_localization = localization
    funm.stypy_type_of_self = None
    funm.stypy_type_store = module_type_store
    funm.stypy_function_name = 'funm'
    funm.stypy_param_names_list = ['A', 'func', 'disp']
    funm.stypy_varargs_param_name = None
    funm.stypy_kwargs_param_name = None
    funm.stypy_call_defaults = defaults
    funm.stypy_call_varargs = varargs
    funm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'funm', ['A', 'func', 'disp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'funm', localization, ['A', 'func', 'disp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'funm(...)' code ##################

    str_22930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, (-1)), 'str', '\n    Evaluate a matrix function specified by a callable.\n\n    Returns the value of matrix-valued function ``f`` at `A`. The\n    function ``f`` is an extension of the scalar-valued function `func`\n    to matrices.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Matrix at which to evaluate the function\n    func : callable\n        Callable object that evaluates a scalar function f.\n        Must be vectorized (eg. using vectorize).\n    disp : bool, optional\n        Print warning if error in the result is estimated large\n        instead of returning estimated error. (Default: True)\n\n    Returns\n    -------\n    funm : (N, N) ndarray\n        Value of the matrix function specified by func evaluated at `A`\n    errest : float\n        (if disp == False)\n\n        1-norm of the estimated error, ||err||_1 / ||A||_1\n\n    Examples\n    --------\n    >>> from scipy.linalg import funm\n    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])\n    >>> funm(a, lambda x: x*x)\n    array([[  4.,  15.],\n           [  5.,  19.]])\n    >>> a.dot(a)\n    array([[  4.,  15.],\n           [  5.,  19.]])\n\n    Notes\n    -----\n    This function implements the general algorithm based on Schur decomposition\n    (Algorithm 9.1.1. in [1]_).\n\n    If the input matrix is known to be diagonalizable, then relying on the\n    eigendecomposition is likely to be faster. For example, if your matrix is\n    Hermitian, you can do\n\n    >>> from scipy.linalg import eigh\n    >>> def funm_herm(a, func, check_finite=False):\n    ...     w, v = eigh(a, check_finite=check_finite)\n    ...     ## if you further know that your matrix is positive semidefinite,\n    ...     ## you can optionally guard against precision errors by doing\n    ...     # w = np.maximum(w, 0)\n    ...     w = func(w)\n    ...     return (v * w).dot(v.conj().T)\n\n    References\n    ----------\n    .. [1] Gene H. Golub, Charles F. van Loan, Matrix Computations 4th ed.\n\n    ')
    
    # Assigning a Call to a Name (line 551):
    
    # Assigning a Call to a Name (line 551):
    
    # Call to _asarray_square(...): (line 551)
    # Processing the call arguments (line 551)
    # Getting the type of 'A' (line 551)
    A_22932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 24), 'A', False)
    # Processing the call keyword arguments (line 551)
    kwargs_22933 = {}
    # Getting the type of '_asarray_square' (line 551)
    _asarray_square_22931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), '_asarray_square', False)
    # Calling _asarray_square(args, kwargs) (line 551)
    _asarray_square_call_result_22934 = invoke(stypy.reporting.localization.Localization(__file__, 551, 8), _asarray_square_22931, *[A_22932], **kwargs_22933)
    
    # Assigning a type to the variable 'A' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'A', _asarray_square_call_result_22934)
    
    # Assigning a Call to a Tuple (line 553):
    
    # Assigning a Subscript to a Name (line 553):
    
    # Obtaining the type of the subscript
    int_22935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 4), 'int')
    
    # Call to schur(...): (line 553)
    # Processing the call arguments (line 553)
    # Getting the type of 'A' (line 553)
    A_22937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 17), 'A', False)
    # Processing the call keyword arguments (line 553)
    kwargs_22938 = {}
    # Getting the type of 'schur' (line 553)
    schur_22936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 11), 'schur', False)
    # Calling schur(args, kwargs) (line 553)
    schur_call_result_22939 = invoke(stypy.reporting.localization.Localization(__file__, 553, 11), schur_22936, *[A_22937], **kwargs_22938)
    
    # Obtaining the member '__getitem__' of a type (line 553)
    getitem___22940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 4), schur_call_result_22939, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 553)
    subscript_call_result_22941 = invoke(stypy.reporting.localization.Localization(__file__, 553, 4), getitem___22940, int_22935)
    
    # Assigning a type to the variable 'tuple_var_assignment_22540' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'tuple_var_assignment_22540', subscript_call_result_22941)
    
    # Assigning a Subscript to a Name (line 553):
    
    # Obtaining the type of the subscript
    int_22942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 4), 'int')
    
    # Call to schur(...): (line 553)
    # Processing the call arguments (line 553)
    # Getting the type of 'A' (line 553)
    A_22944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 17), 'A', False)
    # Processing the call keyword arguments (line 553)
    kwargs_22945 = {}
    # Getting the type of 'schur' (line 553)
    schur_22943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 11), 'schur', False)
    # Calling schur(args, kwargs) (line 553)
    schur_call_result_22946 = invoke(stypy.reporting.localization.Localization(__file__, 553, 11), schur_22943, *[A_22944], **kwargs_22945)
    
    # Obtaining the member '__getitem__' of a type (line 553)
    getitem___22947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 4), schur_call_result_22946, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 553)
    subscript_call_result_22948 = invoke(stypy.reporting.localization.Localization(__file__, 553, 4), getitem___22947, int_22942)
    
    # Assigning a type to the variable 'tuple_var_assignment_22541' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'tuple_var_assignment_22541', subscript_call_result_22948)
    
    # Assigning a Name to a Name (line 553):
    # Getting the type of 'tuple_var_assignment_22540' (line 553)
    tuple_var_assignment_22540_22949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'tuple_var_assignment_22540')
    # Assigning a type to the variable 'T' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'T', tuple_var_assignment_22540_22949)
    
    # Assigning a Name to a Name (line 553):
    # Getting the type of 'tuple_var_assignment_22541' (line 553)
    tuple_var_assignment_22541_22950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'tuple_var_assignment_22541')
    # Assigning a type to the variable 'Z' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 7), 'Z', tuple_var_assignment_22541_22950)
    
    # Assigning a Call to a Tuple (line 554):
    
    # Assigning a Subscript to a Name (line 554):
    
    # Obtaining the type of the subscript
    int_22951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 4), 'int')
    
    # Call to rsf2csf(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'T' (line 554)
    T_22953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 19), 'T', False)
    # Getting the type of 'Z' (line 554)
    Z_22954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 21), 'Z', False)
    # Processing the call keyword arguments (line 554)
    kwargs_22955 = {}
    # Getting the type of 'rsf2csf' (line 554)
    rsf2csf_22952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 11), 'rsf2csf', False)
    # Calling rsf2csf(args, kwargs) (line 554)
    rsf2csf_call_result_22956 = invoke(stypy.reporting.localization.Localization(__file__, 554, 11), rsf2csf_22952, *[T_22953, Z_22954], **kwargs_22955)
    
    # Obtaining the member '__getitem__' of a type (line 554)
    getitem___22957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 4), rsf2csf_call_result_22956, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 554)
    subscript_call_result_22958 = invoke(stypy.reporting.localization.Localization(__file__, 554, 4), getitem___22957, int_22951)
    
    # Assigning a type to the variable 'tuple_var_assignment_22542' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'tuple_var_assignment_22542', subscript_call_result_22958)
    
    # Assigning a Subscript to a Name (line 554):
    
    # Obtaining the type of the subscript
    int_22959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 4), 'int')
    
    # Call to rsf2csf(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'T' (line 554)
    T_22961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 19), 'T', False)
    # Getting the type of 'Z' (line 554)
    Z_22962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 21), 'Z', False)
    # Processing the call keyword arguments (line 554)
    kwargs_22963 = {}
    # Getting the type of 'rsf2csf' (line 554)
    rsf2csf_22960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 11), 'rsf2csf', False)
    # Calling rsf2csf(args, kwargs) (line 554)
    rsf2csf_call_result_22964 = invoke(stypy.reporting.localization.Localization(__file__, 554, 11), rsf2csf_22960, *[T_22961, Z_22962], **kwargs_22963)
    
    # Obtaining the member '__getitem__' of a type (line 554)
    getitem___22965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 4), rsf2csf_call_result_22964, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 554)
    subscript_call_result_22966 = invoke(stypy.reporting.localization.Localization(__file__, 554, 4), getitem___22965, int_22959)
    
    # Assigning a type to the variable 'tuple_var_assignment_22543' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'tuple_var_assignment_22543', subscript_call_result_22966)
    
    # Assigning a Name to a Name (line 554):
    # Getting the type of 'tuple_var_assignment_22542' (line 554)
    tuple_var_assignment_22542_22967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'tuple_var_assignment_22542')
    # Assigning a type to the variable 'T' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'T', tuple_var_assignment_22542_22967)
    
    # Assigning a Name to a Name (line 554):
    # Getting the type of 'tuple_var_assignment_22543' (line 554)
    tuple_var_assignment_22543_22968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'tuple_var_assignment_22543')
    # Assigning a type to the variable 'Z' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 7), 'Z', tuple_var_assignment_22543_22968)
    
    # Assigning a Attribute to a Tuple (line 555):
    
    # Assigning a Subscript to a Name (line 555):
    
    # Obtaining the type of the subscript
    int_22969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 4), 'int')
    # Getting the type of 'T' (line 555)
    T_22970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 10), 'T')
    # Obtaining the member 'shape' of a type (line 555)
    shape_22971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 10), T_22970, 'shape')
    # Obtaining the member '__getitem__' of a type (line 555)
    getitem___22972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 4), shape_22971, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 555)
    subscript_call_result_22973 = invoke(stypy.reporting.localization.Localization(__file__, 555, 4), getitem___22972, int_22969)
    
    # Assigning a type to the variable 'tuple_var_assignment_22544' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'tuple_var_assignment_22544', subscript_call_result_22973)
    
    # Assigning a Subscript to a Name (line 555):
    
    # Obtaining the type of the subscript
    int_22974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 4), 'int')
    # Getting the type of 'T' (line 555)
    T_22975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 10), 'T')
    # Obtaining the member 'shape' of a type (line 555)
    shape_22976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 10), T_22975, 'shape')
    # Obtaining the member '__getitem__' of a type (line 555)
    getitem___22977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 4), shape_22976, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 555)
    subscript_call_result_22978 = invoke(stypy.reporting.localization.Localization(__file__, 555, 4), getitem___22977, int_22974)
    
    # Assigning a type to the variable 'tuple_var_assignment_22545' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'tuple_var_assignment_22545', subscript_call_result_22978)
    
    # Assigning a Name to a Name (line 555):
    # Getting the type of 'tuple_var_assignment_22544' (line 555)
    tuple_var_assignment_22544_22979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'tuple_var_assignment_22544')
    # Assigning a type to the variable 'n' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'n', tuple_var_assignment_22544_22979)
    
    # Assigning a Name to a Name (line 555):
    # Getting the type of 'tuple_var_assignment_22545' (line 555)
    tuple_var_assignment_22545_22980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'tuple_var_assignment_22545')
    # Assigning a type to the variable 'n' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 6), 'n', tuple_var_assignment_22545_22980)
    
    # Assigning a Call to a Name (line 556):
    
    # Assigning a Call to a Name (line 556):
    
    # Call to diag(...): (line 556)
    # Processing the call arguments (line 556)
    
    # Call to func(...): (line 556)
    # Processing the call arguments (line 556)
    
    # Call to diag(...): (line 556)
    # Processing the call arguments (line 556)
    # Getting the type of 'T' (line 556)
    T_22984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 23), 'T', False)
    # Processing the call keyword arguments (line 556)
    kwargs_22985 = {}
    # Getting the type of 'diag' (line 556)
    diag_22983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 18), 'diag', False)
    # Calling diag(args, kwargs) (line 556)
    diag_call_result_22986 = invoke(stypy.reporting.localization.Localization(__file__, 556, 18), diag_22983, *[T_22984], **kwargs_22985)
    
    # Processing the call keyword arguments (line 556)
    kwargs_22987 = {}
    # Getting the type of 'func' (line 556)
    func_22982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 13), 'func', False)
    # Calling func(args, kwargs) (line 556)
    func_call_result_22988 = invoke(stypy.reporting.localization.Localization(__file__, 556, 13), func_22982, *[diag_call_result_22986], **kwargs_22987)
    
    # Processing the call keyword arguments (line 556)
    kwargs_22989 = {}
    # Getting the type of 'diag' (line 556)
    diag_22981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'diag', False)
    # Calling diag(args, kwargs) (line 556)
    diag_call_result_22990 = invoke(stypy.reporting.localization.Localization(__file__, 556, 8), diag_22981, *[func_call_result_22988], **kwargs_22989)
    
    # Assigning a type to the variable 'F' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'F', diag_call_result_22990)
    
    # Assigning a Call to a Name (line 557):
    
    # Assigning a Call to a Name (line 557):
    
    # Call to astype(...): (line 557)
    # Processing the call arguments (line 557)
    # Getting the type of 'T' (line 557)
    T_22993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 17), 'T', False)
    # Obtaining the member 'dtype' of a type (line 557)
    dtype_22994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 17), T_22993, 'dtype')
    # Obtaining the member 'char' of a type (line 557)
    char_22995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 17), dtype_22994, 'char')
    # Processing the call keyword arguments (line 557)
    kwargs_22996 = {}
    # Getting the type of 'F' (line 557)
    F_22991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'F', False)
    # Obtaining the member 'astype' of a type (line 557)
    astype_22992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 8), F_22991, 'astype')
    # Calling astype(args, kwargs) (line 557)
    astype_call_result_22997 = invoke(stypy.reporting.localization.Localization(__file__, 557, 8), astype_22992, *[char_22995], **kwargs_22996)
    
    # Assigning a type to the variable 'F' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'F', astype_call_result_22997)
    
    # Assigning a Call to a Name (line 559):
    
    # Assigning a Call to a Name (line 559):
    
    # Call to abs(...): (line 559)
    # Processing the call arguments (line 559)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 559)
    tuple_22999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 559)
    # Adding element type (line 559)
    int_23000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 19), tuple_22999, int_23000)
    # Adding element type (line 559)
    int_23001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 19), tuple_22999, int_23001)
    
    # Getting the type of 'T' (line 559)
    T_23002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 17), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 559)
    getitem___23003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 17), T_23002, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 559)
    subscript_call_result_23004 = invoke(stypy.reporting.localization.Localization(__file__, 559, 17), getitem___23003, tuple_22999)
    
    # Processing the call keyword arguments (line 559)
    kwargs_23005 = {}
    # Getting the type of 'abs' (line 559)
    abs_22998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 13), 'abs', False)
    # Calling abs(args, kwargs) (line 559)
    abs_call_result_23006 = invoke(stypy.reporting.localization.Localization(__file__, 559, 13), abs_22998, *[subscript_call_result_23004], **kwargs_23005)
    
    # Assigning a type to the variable 'minden' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'minden', abs_call_result_23006)
    
    
    # Call to range(...): (line 563)
    # Processing the call arguments (line 563)
    int_23008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 19), 'int')
    # Getting the type of 'n' (line 563)
    n_23009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 21), 'n', False)
    # Processing the call keyword arguments (line 563)
    kwargs_23010 = {}
    # Getting the type of 'range' (line 563)
    range_23007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 13), 'range', False)
    # Calling range(args, kwargs) (line 563)
    range_call_result_23011 = invoke(stypy.reporting.localization.Localization(__file__, 563, 13), range_23007, *[int_23008, n_23009], **kwargs_23010)
    
    # Testing the type of a for loop iterable (line 563)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 563, 4), range_call_result_23011)
    # Getting the type of the for loop variable (line 563)
    for_loop_var_23012 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 563, 4), range_call_result_23011)
    # Assigning a type to the variable 'p' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'p', for_loop_var_23012)
    # SSA begins for a for statement (line 563)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 564)
    # Processing the call arguments (line 564)
    int_23014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 23), 'int')
    # Getting the type of 'n' (line 564)
    n_23015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 25), 'n', False)
    # Getting the type of 'p' (line 564)
    p_23016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 27), 'p', False)
    # Applying the binary operator '-' (line 564)
    result_sub_23017 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 25), '-', n_23015, p_23016)
    
    int_23018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 29), 'int')
    # Applying the binary operator '+' (line 564)
    result_add_23019 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 28), '+', result_sub_23017, int_23018)
    
    # Processing the call keyword arguments (line 564)
    kwargs_23020 = {}
    # Getting the type of 'range' (line 564)
    range_23013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 17), 'range', False)
    # Calling range(args, kwargs) (line 564)
    range_call_result_23021 = invoke(stypy.reporting.localization.Localization(__file__, 564, 17), range_23013, *[int_23014, result_add_23019], **kwargs_23020)
    
    # Testing the type of a for loop iterable (line 564)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 564, 8), range_call_result_23021)
    # Getting the type of the for loop variable (line 564)
    for_loop_var_23022 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 564, 8), range_call_result_23021)
    # Assigning a type to the variable 'i' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'i', for_loop_var_23022)
    # SSA begins for a for statement (line 564)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 565):
    
    # Assigning a BinOp to a Name (line 565):
    # Getting the type of 'i' (line 565)
    i_23023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 16), 'i')
    # Getting the type of 'p' (line 565)
    p_23024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 20), 'p')
    # Applying the binary operator '+' (line 565)
    result_add_23025 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 16), '+', i_23023, p_23024)
    
    # Assigning a type to the variable 'j' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'j', result_add_23025)
    
    # Assigning a BinOp to a Name (line 566):
    
    # Assigning a BinOp to a Name (line 566):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 566)
    tuple_23026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 566)
    # Adding element type (line 566)
    # Getting the type of 'i' (line 566)
    i_23027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 18), 'i')
    int_23028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 20), 'int')
    # Applying the binary operator '-' (line 566)
    result_sub_23029 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 18), '-', i_23027, int_23028)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 18), tuple_23026, result_sub_23029)
    # Adding element type (line 566)
    # Getting the type of 'j' (line 566)
    j_23030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 22), 'j')
    int_23031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 24), 'int')
    # Applying the binary operator '-' (line 566)
    result_sub_23032 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 22), '-', j_23030, int_23031)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 18), tuple_23026, result_sub_23032)
    
    # Getting the type of 'T' (line 566)
    T_23033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 16), 'T')
    # Obtaining the member '__getitem__' of a type (line 566)
    getitem___23034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 16), T_23033, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 566)
    subscript_call_result_23035 = invoke(stypy.reporting.localization.Localization(__file__, 566, 16), getitem___23034, tuple_23026)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 566)
    tuple_23036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 566)
    # Adding element type (line 566)
    # Getting the type of 'j' (line 566)
    j_23037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 32), 'j')
    int_23038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 34), 'int')
    # Applying the binary operator '-' (line 566)
    result_sub_23039 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 32), '-', j_23037, int_23038)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 32), tuple_23036, result_sub_23039)
    # Adding element type (line 566)
    # Getting the type of 'j' (line 566)
    j_23040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 36), 'j')
    int_23041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 38), 'int')
    # Applying the binary operator '-' (line 566)
    result_sub_23042 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 36), '-', j_23040, int_23041)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 32), tuple_23036, result_sub_23042)
    
    # Getting the type of 'F' (line 566)
    F_23043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 30), 'F')
    # Obtaining the member '__getitem__' of a type (line 566)
    getitem___23044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 30), F_23043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 566)
    subscript_call_result_23045 = invoke(stypy.reporting.localization.Localization(__file__, 566, 30), getitem___23044, tuple_23036)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 566)
    tuple_23046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 566)
    # Adding element type (line 566)
    # Getting the type of 'i' (line 566)
    i_23047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 45), 'i')
    int_23048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 47), 'int')
    # Applying the binary operator '-' (line 566)
    result_sub_23049 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 45), '-', i_23047, int_23048)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 45), tuple_23046, result_sub_23049)
    # Adding element type (line 566)
    # Getting the type of 'i' (line 566)
    i_23050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 49), 'i')
    int_23051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 51), 'int')
    # Applying the binary operator '-' (line 566)
    result_sub_23052 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 49), '-', i_23050, int_23051)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 45), tuple_23046, result_sub_23052)
    
    # Getting the type of 'F' (line 566)
    F_23053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 43), 'F')
    # Obtaining the member '__getitem__' of a type (line 566)
    getitem___23054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 43), F_23053, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 566)
    subscript_call_result_23055 = invoke(stypy.reporting.localization.Localization(__file__, 566, 43), getitem___23054, tuple_23046)
    
    # Applying the binary operator '-' (line 566)
    result_sub_23056 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 30), '-', subscript_call_result_23045, subscript_call_result_23055)
    
    # Applying the binary operator '*' (line 566)
    result_mul_23057 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 16), '*', subscript_call_result_23035, result_sub_23056)
    
    # Assigning a type to the variable 's' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 's', result_mul_23057)
    
    # Assigning a Call to a Name (line 567):
    
    # Assigning a Call to a Name (line 567):
    
    # Call to slice(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'i' (line 567)
    i_23059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 24), 'i', False)
    # Getting the type of 'j' (line 567)
    j_23060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 26), 'j', False)
    int_23061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 28), 'int')
    # Applying the binary operator '-' (line 567)
    result_sub_23062 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 26), '-', j_23060, int_23061)
    
    # Processing the call keyword arguments (line 567)
    kwargs_23063 = {}
    # Getting the type of 'slice' (line 567)
    slice_23058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 18), 'slice', False)
    # Calling slice(args, kwargs) (line 567)
    slice_call_result_23064 = invoke(stypy.reporting.localization.Localization(__file__, 567, 18), slice_23058, *[i_23059, result_sub_23062], **kwargs_23063)
    
    # Assigning a type to the variable 'ksl' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'ksl', slice_call_result_23064)
    
    # Assigning a BinOp to a Name (line 568):
    
    # Assigning a BinOp to a Name (line 568):
    
    # Call to dot(...): (line 568)
    # Processing the call arguments (line 568)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 568)
    tuple_23066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 568)
    # Adding element type (line 568)
    # Getting the type of 'i' (line 568)
    i_23067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 24), 'i', False)
    int_23068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 26), 'int')
    # Applying the binary operator '-' (line 568)
    result_sub_23069 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 24), '-', i_23067, int_23068)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 24), tuple_23066, result_sub_23069)
    # Adding element type (line 568)
    # Getting the type of 'ksl' (line 568)
    ksl_23070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 28), 'ksl', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 24), tuple_23066, ksl_23070)
    
    # Getting the type of 'T' (line 568)
    T_23071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 22), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 568)
    getitem___23072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 22), T_23071, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 568)
    subscript_call_result_23073 = invoke(stypy.reporting.localization.Localization(__file__, 568, 22), getitem___23072, tuple_23066)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 568)
    tuple_23074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 568)
    # Adding element type (line 568)
    # Getting the type of 'ksl' (line 568)
    ksl_23075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 35), 'ksl', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 35), tuple_23074, ksl_23075)
    # Adding element type (line 568)
    # Getting the type of 'j' (line 568)
    j_23076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 39), 'j', False)
    int_23077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 41), 'int')
    # Applying the binary operator '-' (line 568)
    result_sub_23078 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 39), '-', j_23076, int_23077)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 35), tuple_23074, result_sub_23078)
    
    # Getting the type of 'F' (line 568)
    F_23079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 33), 'F', False)
    # Obtaining the member '__getitem__' of a type (line 568)
    getitem___23080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 33), F_23079, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 568)
    subscript_call_result_23081 = invoke(stypy.reporting.localization.Localization(__file__, 568, 33), getitem___23080, tuple_23074)
    
    # Processing the call keyword arguments (line 568)
    kwargs_23082 = {}
    # Getting the type of 'dot' (line 568)
    dot_23065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 18), 'dot', False)
    # Calling dot(args, kwargs) (line 568)
    dot_call_result_23083 = invoke(stypy.reporting.localization.Localization(__file__, 568, 18), dot_23065, *[subscript_call_result_23073, subscript_call_result_23081], **kwargs_23082)
    
    
    # Call to dot(...): (line 568)
    # Processing the call arguments (line 568)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 568)
    tuple_23085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 568)
    # Adding element type (line 568)
    # Getting the type of 'i' (line 568)
    i_23086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 53), 'i', False)
    int_23087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 55), 'int')
    # Applying the binary operator '-' (line 568)
    result_sub_23088 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 53), '-', i_23086, int_23087)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 53), tuple_23085, result_sub_23088)
    # Adding element type (line 568)
    # Getting the type of 'ksl' (line 568)
    ksl_23089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 57), 'ksl', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 53), tuple_23085, ksl_23089)
    
    # Getting the type of 'F' (line 568)
    F_23090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 51), 'F', False)
    # Obtaining the member '__getitem__' of a type (line 568)
    getitem___23091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 51), F_23090, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 568)
    subscript_call_result_23092 = invoke(stypy.reporting.localization.Localization(__file__, 568, 51), getitem___23091, tuple_23085)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 568)
    tuple_23093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 64), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 568)
    # Adding element type (line 568)
    # Getting the type of 'ksl' (line 568)
    ksl_23094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 64), 'ksl', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 64), tuple_23093, ksl_23094)
    # Adding element type (line 568)
    # Getting the type of 'j' (line 568)
    j_23095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 68), 'j', False)
    int_23096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 70), 'int')
    # Applying the binary operator '-' (line 568)
    result_sub_23097 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 68), '-', j_23095, int_23096)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 64), tuple_23093, result_sub_23097)
    
    # Getting the type of 'T' (line 568)
    T_23098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 62), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 568)
    getitem___23099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 62), T_23098, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 568)
    subscript_call_result_23100 = invoke(stypy.reporting.localization.Localization(__file__, 568, 62), getitem___23099, tuple_23093)
    
    # Processing the call keyword arguments (line 568)
    kwargs_23101 = {}
    # Getting the type of 'dot' (line 568)
    dot_23084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 47), 'dot', False)
    # Calling dot(args, kwargs) (line 568)
    dot_call_result_23102 = invoke(stypy.reporting.localization.Localization(__file__, 568, 47), dot_23084, *[subscript_call_result_23092, subscript_call_result_23100], **kwargs_23101)
    
    # Applying the binary operator '-' (line 568)
    result_sub_23103 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 18), '-', dot_call_result_23083, dot_call_result_23102)
    
    # Assigning a type to the variable 'val' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'val', result_sub_23103)
    
    # Assigning a BinOp to a Name (line 569):
    
    # Assigning a BinOp to a Name (line 569):
    # Getting the type of 's' (line 569)
    s_23104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 16), 's')
    # Getting the type of 'val' (line 569)
    val_23105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 20), 'val')
    # Applying the binary operator '+' (line 569)
    result_add_23106 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 16), '+', s_23104, val_23105)
    
    # Assigning a type to the variable 's' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 's', result_add_23106)
    
    # Assigning a BinOp to a Name (line 570):
    
    # Assigning a BinOp to a Name (line 570):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 570)
    tuple_23107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 570)
    # Adding element type (line 570)
    # Getting the type of 'j' (line 570)
    j_23108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 20), 'j')
    int_23109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 22), 'int')
    # Applying the binary operator '-' (line 570)
    result_sub_23110 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 20), '-', j_23108, int_23109)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 20), tuple_23107, result_sub_23110)
    # Adding element type (line 570)
    # Getting the type of 'j' (line 570)
    j_23111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 24), 'j')
    int_23112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 26), 'int')
    # Applying the binary operator '-' (line 570)
    result_sub_23113 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 24), '-', j_23111, int_23112)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 20), tuple_23107, result_sub_23113)
    
    # Getting the type of 'T' (line 570)
    T_23114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 18), 'T')
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___23115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 18), T_23114, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_23116 = invoke(stypy.reporting.localization.Localization(__file__, 570, 18), getitem___23115, tuple_23107)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 570)
    tuple_23117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 570)
    # Adding element type (line 570)
    # Getting the type of 'i' (line 570)
    i_23118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 33), 'i')
    int_23119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 35), 'int')
    # Applying the binary operator '-' (line 570)
    result_sub_23120 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 33), '-', i_23118, int_23119)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 33), tuple_23117, result_sub_23120)
    # Adding element type (line 570)
    # Getting the type of 'i' (line 570)
    i_23121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 37), 'i')
    int_23122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 39), 'int')
    # Applying the binary operator '-' (line 570)
    result_sub_23123 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 37), '-', i_23121, int_23122)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 33), tuple_23117, result_sub_23123)
    
    # Getting the type of 'T' (line 570)
    T_23124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 31), 'T')
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___23125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 31), T_23124, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_23126 = invoke(stypy.reporting.localization.Localization(__file__, 570, 31), getitem___23125, tuple_23117)
    
    # Applying the binary operator '-' (line 570)
    result_sub_23127 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 18), '-', subscript_call_result_23116, subscript_call_result_23126)
    
    # Assigning a type to the variable 'den' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'den', result_sub_23127)
    
    
    # Getting the type of 'den' (line 571)
    den_23128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 15), 'den')
    float_23129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 22), 'float')
    # Applying the binary operator '!=' (line 571)
    result_ne_23130 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 15), '!=', den_23128, float_23129)
    
    # Testing the type of an if condition (line 571)
    if_condition_23131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 571, 12), result_ne_23130)
    # Assigning a type to the variable 'if_condition_23131' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 12), 'if_condition_23131', if_condition_23131)
    # SSA begins for if statement (line 571)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 572):
    
    # Assigning a BinOp to a Name (line 572):
    # Getting the type of 's' (line 572)
    s_23132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 20), 's')
    # Getting the type of 'den' (line 572)
    den_23133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 24), 'den')
    # Applying the binary operator 'div' (line 572)
    result_div_23134 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 20), 'div', s_23132, den_23133)
    
    # Assigning a type to the variable 's' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 16), 's', result_div_23134)
    # SSA join for if statement (line 571)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 573):
    
    # Assigning a Name to a Subscript (line 573):
    # Getting the type of 's' (line 573)
    s_23135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 25), 's')
    # Getting the type of 'F' (line 573)
    F_23136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'F')
    
    # Obtaining an instance of the builtin type 'tuple' (line 573)
    tuple_23137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 573)
    # Adding element type (line 573)
    # Getting the type of 'i' (line 573)
    i_23138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 14), 'i')
    int_23139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 16), 'int')
    # Applying the binary operator '-' (line 573)
    result_sub_23140 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 14), '-', i_23138, int_23139)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 14), tuple_23137, result_sub_23140)
    # Adding element type (line 573)
    # Getting the type of 'j' (line 573)
    j_23141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 18), 'j')
    int_23142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 20), 'int')
    # Applying the binary operator '-' (line 573)
    result_sub_23143 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 18), '-', j_23141, int_23142)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 14), tuple_23137, result_sub_23143)
    
    # Storing an element on a container (line 573)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 12), F_23136, (tuple_23137, s_23135))
    
    # Assigning a Call to a Name (line 574):
    
    # Assigning a Call to a Name (line 574):
    
    # Call to min(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'minden' (line 574)
    minden_23145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 25), 'minden', False)
    
    # Call to abs(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'den' (line 574)
    den_23147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 36), 'den', False)
    # Processing the call keyword arguments (line 574)
    kwargs_23148 = {}
    # Getting the type of 'abs' (line 574)
    abs_23146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 32), 'abs', False)
    # Calling abs(args, kwargs) (line 574)
    abs_call_result_23149 = invoke(stypy.reporting.localization.Localization(__file__, 574, 32), abs_23146, *[den_23147], **kwargs_23148)
    
    # Processing the call keyword arguments (line 574)
    kwargs_23150 = {}
    # Getting the type of 'min' (line 574)
    min_23144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 21), 'min', False)
    # Calling min(args, kwargs) (line 574)
    min_call_result_23151 = invoke(stypy.reporting.localization.Localization(__file__, 574, 21), min_23144, *[minden_23145, abs_call_result_23149], **kwargs_23150)
    
    # Assigning a type to the variable 'minden' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'minden', min_call_result_23151)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 576):
    
    # Assigning a Call to a Name (line 576):
    
    # Call to dot(...): (line 576)
    # Processing the call arguments (line 576)
    
    # Call to dot(...): (line 576)
    # Processing the call arguments (line 576)
    # Getting the type of 'Z' (line 576)
    Z_23154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 16), 'Z', False)
    # Getting the type of 'F' (line 576)
    F_23155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 19), 'F', False)
    # Processing the call keyword arguments (line 576)
    kwargs_23156 = {}
    # Getting the type of 'dot' (line 576)
    dot_23153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'dot', False)
    # Calling dot(args, kwargs) (line 576)
    dot_call_result_23157 = invoke(stypy.reporting.localization.Localization(__file__, 576, 12), dot_23153, *[Z_23154, F_23155], **kwargs_23156)
    
    
    # Call to transpose(...): (line 576)
    # Processing the call arguments (line 576)
    
    # Call to conjugate(...): (line 576)
    # Processing the call arguments (line 576)
    # Getting the type of 'Z' (line 576)
    Z_23160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 43), 'Z', False)
    # Processing the call keyword arguments (line 576)
    kwargs_23161 = {}
    # Getting the type of 'conjugate' (line 576)
    conjugate_23159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 33), 'conjugate', False)
    # Calling conjugate(args, kwargs) (line 576)
    conjugate_call_result_23162 = invoke(stypy.reporting.localization.Localization(__file__, 576, 33), conjugate_23159, *[Z_23160], **kwargs_23161)
    
    # Processing the call keyword arguments (line 576)
    kwargs_23163 = {}
    # Getting the type of 'transpose' (line 576)
    transpose_23158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 23), 'transpose', False)
    # Calling transpose(args, kwargs) (line 576)
    transpose_call_result_23164 = invoke(stypy.reporting.localization.Localization(__file__, 576, 23), transpose_23158, *[conjugate_call_result_23162], **kwargs_23163)
    
    # Processing the call keyword arguments (line 576)
    kwargs_23165 = {}
    # Getting the type of 'dot' (line 576)
    dot_23152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'dot', False)
    # Calling dot(args, kwargs) (line 576)
    dot_call_result_23166 = invoke(stypy.reporting.localization.Localization(__file__, 576, 8), dot_23152, *[dot_call_result_23157, transpose_call_result_23164], **kwargs_23165)
    
    # Assigning a type to the variable 'F' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'F', dot_call_result_23166)
    
    # Assigning a Call to a Name (line 577):
    
    # Assigning a Call to a Name (line 577):
    
    # Call to _maybe_real(...): (line 577)
    # Processing the call arguments (line 577)
    # Getting the type of 'A' (line 577)
    A_23168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 20), 'A', False)
    # Getting the type of 'F' (line 577)
    F_23169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 23), 'F', False)
    # Processing the call keyword arguments (line 577)
    kwargs_23170 = {}
    # Getting the type of '_maybe_real' (line 577)
    _maybe_real_23167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), '_maybe_real', False)
    # Calling _maybe_real(args, kwargs) (line 577)
    _maybe_real_call_result_23171 = invoke(stypy.reporting.localization.Localization(__file__, 577, 8), _maybe_real_23167, *[A_23168, F_23169], **kwargs_23170)
    
    # Assigning a type to the variable 'F' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'F', _maybe_real_call_result_23171)
    
    # Assigning a Subscript to a Name (line 579):
    
    # Assigning a Subscript to a Name (line 579):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'F' (line 579)
    F_23172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 43), 'F')
    # Obtaining the member 'dtype' of a type (line 579)
    dtype_23173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 43), F_23172, 'dtype')
    # Obtaining the member 'char' of a type (line 579)
    char_23174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 43), dtype_23173, 'char')
    # Getting the type of '_array_precision' (line 579)
    _array_precision_23175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 26), '_array_precision')
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___23176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 26), _array_precision_23175, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_23177 = invoke(stypy.reporting.localization.Localization(__file__, 579, 26), getitem___23176, char_23174)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 579)
    dict_23178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 579)
    # Adding element type (key, value) (line 579)
    int_23179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 11), 'int')
    # Getting the type of 'feps' (line 579)
    feps_23180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 13), 'feps')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 10), dict_23178, (int_23179, feps_23180))
    # Adding element type (key, value) (line 579)
    int_23181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 19), 'int')
    # Getting the type of 'eps' (line 579)
    eps_23182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 21), 'eps')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 10), dict_23178, (int_23181, eps_23182))
    
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___23183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 10), dict_23178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_23184 = invoke(stypy.reporting.localization.Localization(__file__, 579, 10), getitem___23183, subscript_call_result_23177)
    
    # Assigning a type to the variable 'tol' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'tol', subscript_call_result_23184)
    
    
    # Getting the type of 'minden' (line 580)
    minden_23185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 7), 'minden')
    float_23186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 17), 'float')
    # Applying the binary operator '==' (line 580)
    result_eq_23187 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 7), '==', minden_23185, float_23186)
    
    # Testing the type of an if condition (line 580)
    if_condition_23188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 580, 4), result_eq_23187)
    # Assigning a type to the variable 'if_condition_23188' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'if_condition_23188', if_condition_23188)
    # SSA begins for if statement (line 580)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 581):
    
    # Assigning a Name to a Name (line 581):
    # Getting the type of 'tol' (line 581)
    tol_23189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 17), 'tol')
    # Assigning a type to the variable 'minden' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'minden', tol_23189)
    # SSA join for if statement (line 580)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 582):
    
    # Assigning a Call to a Name (line 582):
    
    # Call to min(...): (line 582)
    # Processing the call arguments (line 582)
    int_23191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 14), 'int')
    
    # Call to max(...): (line 582)
    # Processing the call arguments (line 582)
    # Getting the type of 'tol' (line 582)
    tol_23193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 21), 'tol', False)
    # Getting the type of 'tol' (line 582)
    tol_23194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 26), 'tol', False)
    # Getting the type of 'minden' (line 582)
    minden_23195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 30), 'minden', False)
    # Applying the binary operator 'div' (line 582)
    result_div_23196 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 26), 'div', tol_23194, minden_23195)
    
    
    # Call to norm(...): (line 582)
    # Processing the call arguments (line 582)
    
    # Call to triu(...): (line 582)
    # Processing the call arguments (line 582)
    # Getting the type of 'T' (line 582)
    T_23199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 48), 'T', False)
    int_23200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 50), 'int')
    # Processing the call keyword arguments (line 582)
    kwargs_23201 = {}
    # Getting the type of 'triu' (line 582)
    triu_23198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 43), 'triu', False)
    # Calling triu(args, kwargs) (line 582)
    triu_call_result_23202 = invoke(stypy.reporting.localization.Localization(__file__, 582, 43), triu_23198, *[T_23199, int_23200], **kwargs_23201)
    
    int_23203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 53), 'int')
    # Processing the call keyword arguments (line 582)
    kwargs_23204 = {}
    # Getting the type of 'norm' (line 582)
    norm_23197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 38), 'norm', False)
    # Calling norm(args, kwargs) (line 582)
    norm_call_result_23205 = invoke(stypy.reporting.localization.Localization(__file__, 582, 38), norm_23197, *[triu_call_result_23202, int_23203], **kwargs_23204)
    
    # Applying the binary operator '*' (line 582)
    result_mul_23206 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 25), '*', result_div_23196, norm_call_result_23205)
    
    # Processing the call keyword arguments (line 582)
    kwargs_23207 = {}
    # Getting the type of 'max' (line 582)
    max_23192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 17), 'max', False)
    # Calling max(args, kwargs) (line 582)
    max_call_result_23208 = invoke(stypy.reporting.localization.Localization(__file__, 582, 17), max_23192, *[tol_23193, result_mul_23206], **kwargs_23207)
    
    # Processing the call keyword arguments (line 582)
    kwargs_23209 = {}
    # Getting the type of 'min' (line 582)
    min_23190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 10), 'min', False)
    # Calling min(args, kwargs) (line 582)
    min_call_result_23210 = invoke(stypy.reporting.localization.Localization(__file__, 582, 10), min_23190, *[int_23191, max_call_result_23208], **kwargs_23209)
    
    # Assigning a type to the variable 'err' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'err', min_call_result_23210)
    
    
    # Call to product(...): (line 583)
    # Processing the call arguments (line 583)
    
    # Call to ravel(...): (line 583)
    # Processing the call arguments (line 583)
    
    # Call to logical_not(...): (line 583)
    # Processing the call arguments (line 583)
    
    # Call to isfinite(...): (line 583)
    # Processing the call arguments (line 583)
    # Getting the type of 'F' (line 583)
    F_23215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 42), 'F', False)
    # Processing the call keyword arguments (line 583)
    kwargs_23216 = {}
    # Getting the type of 'isfinite' (line 583)
    isfinite_23214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 33), 'isfinite', False)
    # Calling isfinite(args, kwargs) (line 583)
    isfinite_call_result_23217 = invoke(stypy.reporting.localization.Localization(__file__, 583, 33), isfinite_23214, *[F_23215], **kwargs_23216)
    
    # Processing the call keyword arguments (line 583)
    kwargs_23218 = {}
    # Getting the type of 'logical_not' (line 583)
    logical_not_23213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 21), 'logical_not', False)
    # Calling logical_not(args, kwargs) (line 583)
    logical_not_call_result_23219 = invoke(stypy.reporting.localization.Localization(__file__, 583, 21), logical_not_23213, *[isfinite_call_result_23217], **kwargs_23218)
    
    # Processing the call keyword arguments (line 583)
    kwargs_23220 = {}
    # Getting the type of 'ravel' (line 583)
    ravel_23212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 15), 'ravel', False)
    # Calling ravel(args, kwargs) (line 583)
    ravel_call_result_23221 = invoke(stypy.reporting.localization.Localization(__file__, 583, 15), ravel_23212, *[logical_not_call_result_23219], **kwargs_23220)
    
    # Processing the call keyword arguments (line 583)
    int_23222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 52), 'int')
    keyword_23223 = int_23222
    kwargs_23224 = {'axis': keyword_23223}
    # Getting the type of 'product' (line 583)
    product_23211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 7), 'product', False)
    # Calling product(args, kwargs) (line 583)
    product_call_result_23225 = invoke(stypy.reporting.localization.Localization(__file__, 583, 7), product_23211, *[ravel_call_result_23221], **kwargs_23224)
    
    # Testing the type of an if condition (line 583)
    if_condition_23226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 583, 4), product_call_result_23225)
    # Assigning a type to the variable 'if_condition_23226' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'if_condition_23226', if_condition_23226)
    # SSA begins for if statement (line 583)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 584):
    
    # Assigning a Name to a Name (line 584):
    # Getting the type of 'Inf' (line 584)
    Inf_23227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 14), 'Inf')
    # Assigning a type to the variable 'err' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'err', Inf_23227)
    # SSA join for if statement (line 583)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'disp' (line 585)
    disp_23228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 7), 'disp')
    # Testing the type of an if condition (line 585)
    if_condition_23229 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 4), disp_23228)
    # Assigning a type to the variable 'if_condition_23229' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'if_condition_23229', if_condition_23229)
    # SSA begins for if statement (line 585)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'err' (line 586)
    err_23230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 11), 'err')
    int_23231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 17), 'int')
    # Getting the type of 'tol' (line 586)
    tol_23232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 22), 'tol')
    # Applying the binary operator '*' (line 586)
    result_mul_23233 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 17), '*', int_23231, tol_23232)
    
    # Applying the binary operator '>' (line 586)
    result_gt_23234 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 11), '>', err_23230, result_mul_23233)
    
    # Testing the type of an if condition (line 586)
    if_condition_23235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 586, 8), result_gt_23234)
    # Assigning a type to the variable 'if_condition_23235' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'if_condition_23235', if_condition_23235)
    # SSA begins for if statement (line 586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 587)
    # Processing the call arguments (line 587)
    str_23237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 18), 'str', 'funm result may be inaccurate, approximate err =')
    # Getting the type of 'err' (line 587)
    err_23238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 70), 'err', False)
    # Processing the call keyword arguments (line 587)
    kwargs_23239 = {}
    # Getting the type of 'print' (line 587)
    print_23236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 12), 'print', False)
    # Calling print(args, kwargs) (line 587)
    print_call_result_23240 = invoke(stypy.reporting.localization.Localization(__file__, 587, 12), print_23236, *[str_23237, err_23238], **kwargs_23239)
    
    # SSA join for if statement (line 586)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'F' (line 588)
    F_23241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 15), 'F')
    # Assigning a type to the variable 'stypy_return_type' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'stypy_return_type', F_23241)
    # SSA branch for the else part of an if statement (line 585)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 590)
    tuple_23242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 590)
    # Adding element type (line 590)
    # Getting the type of 'F' (line 590)
    F_23243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 15), 'F')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 15), tuple_23242, F_23243)
    # Adding element type (line 590)
    # Getting the type of 'err' (line 590)
    err_23244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 18), 'err')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 15), tuple_23242, err_23244)
    
    # Assigning a type to the variable 'stypy_return_type' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'stypy_return_type', tuple_23242)
    # SSA join for if statement (line 585)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'funm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'funm' in the type store
    # Getting the type of 'stypy_return_type' (line 489)
    stypy_return_type_23245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23245)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'funm'
    return stypy_return_type_23245

# Assigning a type to the variable 'funm' (line 489)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 0), 'funm', funm)

@norecursion
def signm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 593)
    True_23246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 18), 'True')
    defaults = [True_23246]
    # Create a new context for function 'signm'
    module_type_store = module_type_store.open_function_context('signm', 593, 0, False)
    
    # Passed parameters checking function
    signm.stypy_localization = localization
    signm.stypy_type_of_self = None
    signm.stypy_type_store = module_type_store
    signm.stypy_function_name = 'signm'
    signm.stypy_param_names_list = ['A', 'disp']
    signm.stypy_varargs_param_name = None
    signm.stypy_kwargs_param_name = None
    signm.stypy_call_defaults = defaults
    signm.stypy_call_varargs = varargs
    signm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'signm', ['A', 'disp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'signm', localization, ['A', 'disp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'signm(...)' code ##################

    str_23247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, (-1)), 'str', '\n    Matrix sign function.\n\n    Extension of the scalar sign(x) to matrices.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Matrix at which to evaluate the sign function\n    disp : bool, optional\n        Print warning if error in the result is estimated large\n        instead of returning estimated error. (Default: True)\n\n    Returns\n    -------\n    signm : (N, N) ndarray\n        Value of the sign function at `A`\n    errest : float\n        (if disp == False)\n\n        1-norm of the estimated error, ||err||_1 / ||A||_1\n\n    Examples\n    --------\n    >>> from scipy.linalg import signm, eigvals\n    >>> a = [[1,2,3], [1,2,1], [1,1,1]]\n    >>> eigvals(a)\n    array([ 4.12488542+0.j, -0.76155718+0.j,  0.63667176+0.j])\n    >>> eigvals(signm(a))\n    array([-1.+0.j,  1.+0.j,  1.+0.j])\n\n    ')
    
    # Assigning a Call to a Name (line 626):
    
    # Assigning a Call to a Name (line 626):
    
    # Call to _asarray_square(...): (line 626)
    # Processing the call arguments (line 626)
    # Getting the type of 'A' (line 626)
    A_23249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 24), 'A', False)
    # Processing the call keyword arguments (line 626)
    kwargs_23250 = {}
    # Getting the type of '_asarray_square' (line 626)
    _asarray_square_23248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), '_asarray_square', False)
    # Calling _asarray_square(args, kwargs) (line 626)
    _asarray_square_call_result_23251 = invoke(stypy.reporting.localization.Localization(__file__, 626, 8), _asarray_square_23248, *[A_23249], **kwargs_23250)
    
    # Assigning a type to the variable 'A' (line 626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 4), 'A', _asarray_square_call_result_23251)

    @norecursion
    def rounded_sign(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'rounded_sign'
        module_type_store = module_type_store.open_function_context('rounded_sign', 628, 4, False)
        
        # Passed parameters checking function
        rounded_sign.stypy_localization = localization
        rounded_sign.stypy_type_of_self = None
        rounded_sign.stypy_type_store = module_type_store
        rounded_sign.stypy_function_name = 'rounded_sign'
        rounded_sign.stypy_param_names_list = ['x']
        rounded_sign.stypy_varargs_param_name = None
        rounded_sign.stypy_kwargs_param_name = None
        rounded_sign.stypy_call_defaults = defaults
        rounded_sign.stypy_call_varargs = varargs
        rounded_sign.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'rounded_sign', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rounded_sign', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rounded_sign(...)' code ##################

        
        # Assigning a Call to a Name (line 629):
        
        # Assigning a Call to a Name (line 629):
        
        # Call to real(...): (line 629)
        # Processing the call arguments (line 629)
        # Getting the type of 'x' (line 629)
        x_23254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 21), 'x', False)
        # Processing the call keyword arguments (line 629)
        kwargs_23255 = {}
        # Getting the type of 'np' (line 629)
        np_23252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 13), 'np', False)
        # Obtaining the member 'real' of a type (line 629)
        real_23253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 13), np_23252, 'real')
        # Calling real(args, kwargs) (line 629)
        real_call_result_23256 = invoke(stypy.reporting.localization.Localization(__file__, 629, 13), real_23253, *[x_23254], **kwargs_23255)
        
        # Assigning a type to the variable 'rx' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'rx', real_call_result_23256)
        
        
        # Getting the type of 'rx' (line 630)
        rx_23257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 11), 'rx')
        # Obtaining the member 'dtype' of a type (line 630)
        dtype_23258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 11), rx_23257, 'dtype')
        # Obtaining the member 'char' of a type (line 630)
        char_23259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 11), dtype_23258, 'char')
        str_23260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 28), 'str', 'f')
        # Applying the binary operator '==' (line 630)
        result_eq_23261 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 11), '==', char_23259, str_23260)
        
        # Testing the type of an if condition (line 630)
        if_condition_23262 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 630, 8), result_eq_23261)
        # Assigning a type to the variable 'if_condition_23262' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 8), 'if_condition_23262', if_condition_23262)
        # SSA begins for if statement (line 630)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 631):
        
        # Assigning a BinOp to a Name (line 631):
        float_23263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 16), 'float')
        # Getting the type of 'feps' (line 631)
        feps_23264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 20), 'feps')
        # Applying the binary operator '*' (line 631)
        result_mul_23265 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 16), '*', float_23263, feps_23264)
        
        
        # Call to amax(...): (line 631)
        # Processing the call arguments (line 631)
        # Getting the type of 'x' (line 631)
        x_23267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 30), 'x', False)
        # Processing the call keyword arguments (line 631)
        kwargs_23268 = {}
        # Getting the type of 'amax' (line 631)
        amax_23266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 25), 'amax', False)
        # Calling amax(args, kwargs) (line 631)
        amax_call_result_23269 = invoke(stypy.reporting.localization.Localization(__file__, 631, 25), amax_23266, *[x_23267], **kwargs_23268)
        
        # Applying the binary operator '*' (line 631)
        result_mul_23270 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 24), '*', result_mul_23265, amax_call_result_23269)
        
        # Assigning a type to the variable 'c' (line 631)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 12), 'c', result_mul_23270)
        # SSA branch for the else part of an if statement (line 630)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 633):
        
        # Assigning a BinOp to a Name (line 633):
        float_23271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 16), 'float')
        # Getting the type of 'eps' (line 633)
        eps_23272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 20), 'eps')
        # Applying the binary operator '*' (line 633)
        result_mul_23273 = python_operator(stypy.reporting.localization.Localization(__file__, 633, 16), '*', float_23271, eps_23272)
        
        
        # Call to amax(...): (line 633)
        # Processing the call arguments (line 633)
        # Getting the type of 'x' (line 633)
        x_23275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 29), 'x', False)
        # Processing the call keyword arguments (line 633)
        kwargs_23276 = {}
        # Getting the type of 'amax' (line 633)
        amax_23274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 24), 'amax', False)
        # Calling amax(args, kwargs) (line 633)
        amax_call_result_23277 = invoke(stypy.reporting.localization.Localization(__file__, 633, 24), amax_23274, *[x_23275], **kwargs_23276)
        
        # Applying the binary operator '*' (line 633)
        result_mul_23278 = python_operator(stypy.reporting.localization.Localization(__file__, 633, 23), '*', result_mul_23273, amax_call_result_23277)
        
        # Assigning a type to the variable 'c' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 12), 'c', result_mul_23278)
        # SSA join for if statement (line 630)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to sign(...): (line 634)
        # Processing the call arguments (line 634)
        
        
        # Call to absolute(...): (line 634)
        # Processing the call arguments (line 634)
        # Getting the type of 'rx' (line 634)
        rx_23281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 30), 'rx', False)
        # Processing the call keyword arguments (line 634)
        kwargs_23282 = {}
        # Getting the type of 'absolute' (line 634)
        absolute_23280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 21), 'absolute', False)
        # Calling absolute(args, kwargs) (line 634)
        absolute_call_result_23283 = invoke(stypy.reporting.localization.Localization(__file__, 634, 21), absolute_23280, *[rx_23281], **kwargs_23282)
        
        # Getting the type of 'c' (line 634)
        c_23284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 36), 'c', False)
        # Applying the binary operator '>' (line 634)
        result_gt_23285 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 21), '>', absolute_call_result_23283, c_23284)
        
        # Getting the type of 'rx' (line 634)
        rx_23286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 41), 'rx', False)
        # Applying the binary operator '*' (line 634)
        result_mul_23287 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 20), '*', result_gt_23285, rx_23286)
        
        # Processing the call keyword arguments (line 634)
        kwargs_23288 = {}
        # Getting the type of 'sign' (line 634)
        sign_23279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 15), 'sign', False)
        # Calling sign(args, kwargs) (line 634)
        sign_call_result_23289 = invoke(stypy.reporting.localization.Localization(__file__, 634, 15), sign_23279, *[result_mul_23287], **kwargs_23288)
        
        # Assigning a type to the variable 'stypy_return_type' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'stypy_return_type', sign_call_result_23289)
        
        # ################# End of 'rounded_sign(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rounded_sign' in the type store
        # Getting the type of 'stypy_return_type' (line 628)
        stypy_return_type_23290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23290)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rounded_sign'
        return stypy_return_type_23290

    # Assigning a type to the variable 'rounded_sign' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'rounded_sign', rounded_sign)
    
    # Assigning a Call to a Tuple (line 635):
    
    # Assigning a Subscript to a Name (line 635):
    
    # Obtaining the type of the subscript
    int_23291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 4), 'int')
    
    # Call to funm(...): (line 635)
    # Processing the call arguments (line 635)
    # Getting the type of 'A' (line 635)
    A_23293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 26), 'A', False)
    # Getting the type of 'rounded_sign' (line 635)
    rounded_sign_23294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 29), 'rounded_sign', False)
    # Processing the call keyword arguments (line 635)
    int_23295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 48), 'int')
    keyword_23296 = int_23295
    kwargs_23297 = {'disp': keyword_23296}
    # Getting the type of 'funm' (line 635)
    funm_23292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 21), 'funm', False)
    # Calling funm(args, kwargs) (line 635)
    funm_call_result_23298 = invoke(stypy.reporting.localization.Localization(__file__, 635, 21), funm_23292, *[A_23293, rounded_sign_23294], **kwargs_23297)
    
    # Obtaining the member '__getitem__' of a type (line 635)
    getitem___23299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 4), funm_call_result_23298, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 635)
    subscript_call_result_23300 = invoke(stypy.reporting.localization.Localization(__file__, 635, 4), getitem___23299, int_23291)
    
    # Assigning a type to the variable 'tuple_var_assignment_22546' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_var_assignment_22546', subscript_call_result_23300)
    
    # Assigning a Subscript to a Name (line 635):
    
    # Obtaining the type of the subscript
    int_23301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 4), 'int')
    
    # Call to funm(...): (line 635)
    # Processing the call arguments (line 635)
    # Getting the type of 'A' (line 635)
    A_23303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 26), 'A', False)
    # Getting the type of 'rounded_sign' (line 635)
    rounded_sign_23304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 29), 'rounded_sign', False)
    # Processing the call keyword arguments (line 635)
    int_23305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 48), 'int')
    keyword_23306 = int_23305
    kwargs_23307 = {'disp': keyword_23306}
    # Getting the type of 'funm' (line 635)
    funm_23302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 21), 'funm', False)
    # Calling funm(args, kwargs) (line 635)
    funm_call_result_23308 = invoke(stypy.reporting.localization.Localization(__file__, 635, 21), funm_23302, *[A_23303, rounded_sign_23304], **kwargs_23307)
    
    # Obtaining the member '__getitem__' of a type (line 635)
    getitem___23309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 4), funm_call_result_23308, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 635)
    subscript_call_result_23310 = invoke(stypy.reporting.localization.Localization(__file__, 635, 4), getitem___23309, int_23301)
    
    # Assigning a type to the variable 'tuple_var_assignment_22547' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_var_assignment_22547', subscript_call_result_23310)
    
    # Assigning a Name to a Name (line 635):
    # Getting the type of 'tuple_var_assignment_22546' (line 635)
    tuple_var_assignment_22546_23311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_var_assignment_22546')
    # Assigning a type to the variable 'result' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'result', tuple_var_assignment_22546_23311)
    
    # Assigning a Name to a Name (line 635):
    # Getting the type of 'tuple_var_assignment_22547' (line 635)
    tuple_var_assignment_22547_23312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_var_assignment_22547')
    # Assigning a type to the variable 'errest' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 12), 'errest', tuple_var_assignment_22547_23312)
    
    # Assigning a Subscript to a Name (line 636):
    
    # Assigning a Subscript to a Name (line 636):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'result' (line 636)
    result_23313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 54), 'result')
    # Obtaining the member 'dtype' of a type (line 636)
    dtype_23314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 54), result_23313, 'dtype')
    # Obtaining the member 'char' of a type (line 636)
    char_23315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 54), dtype_23314, 'char')
    # Getting the type of '_array_precision' (line 636)
    _array_precision_23316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 37), '_array_precision')
    # Obtaining the member '__getitem__' of a type (line 636)
    getitem___23317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 37), _array_precision_23316, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 636)
    subscript_call_result_23318 = invoke(stypy.reporting.localization.Localization(__file__, 636, 37), getitem___23317, char_23315)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 636)
    dict_23319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 636)
    # Adding element type (key, value) (line 636)
    int_23320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 14), 'int')
    float_23321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 16), 'float')
    # Getting the type of 'feps' (line 636)
    feps_23322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 20), 'feps')
    # Applying the binary operator '*' (line 636)
    result_mul_23323 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 16), '*', float_23321, feps_23322)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 13), dict_23319, (int_23320, result_mul_23323))
    # Adding element type (key, value) (line 636)
    int_23324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 26), 'int')
    float_23325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 28), 'float')
    # Getting the type of 'eps' (line 636)
    eps_23326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 32), 'eps')
    # Applying the binary operator '*' (line 636)
    result_mul_23327 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 28), '*', float_23325, eps_23326)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 13), dict_23319, (int_23324, result_mul_23327))
    
    # Obtaining the member '__getitem__' of a type (line 636)
    getitem___23328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 13), dict_23319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 636)
    subscript_call_result_23329 = invoke(stypy.reporting.localization.Localization(__file__, 636, 13), getitem___23328, subscript_call_result_23318)
    
    # Assigning a type to the variable 'errtol' (line 636)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 4), 'errtol', subscript_call_result_23329)
    
    
    # Getting the type of 'errest' (line 637)
    errest_23330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 7), 'errest')
    # Getting the type of 'errtol' (line 637)
    errtol_23331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 16), 'errtol')
    # Applying the binary operator '<' (line 637)
    result_lt_23332 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 7), '<', errest_23330, errtol_23331)
    
    # Testing the type of an if condition (line 637)
    if_condition_23333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 637, 4), result_lt_23332)
    # Assigning a type to the variable 'if_condition_23333' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'if_condition_23333', if_condition_23333)
    # SSA begins for if statement (line 637)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'result' (line 638)
    result_23334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 15), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'stypy_return_type', result_23334)
    # SSA join for if statement (line 637)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 650):
    
    # Assigning a Call to a Name (line 650):
    
    # Call to svd(...): (line 650)
    # Processing the call arguments (line 650)
    # Getting the type of 'A' (line 650)
    A_23336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 15), 'A', False)
    # Processing the call keyword arguments (line 650)
    int_23337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 29), 'int')
    keyword_23338 = int_23337
    kwargs_23339 = {'compute_uv': keyword_23338}
    # Getting the type of 'svd' (line 650)
    svd_23335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 11), 'svd', False)
    # Calling svd(args, kwargs) (line 650)
    svd_call_result_23340 = invoke(stypy.reporting.localization.Localization(__file__, 650, 11), svd_23335, *[A_23336], **kwargs_23339)
    
    # Assigning a type to the variable 'vals' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'vals', svd_call_result_23340)
    
    # Assigning a Call to a Name (line 651):
    
    # Assigning a Call to a Name (line 651):
    
    # Call to amax(...): (line 651)
    # Processing the call arguments (line 651)
    # Getting the type of 'vals' (line 651)
    vals_23343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 21), 'vals', False)
    # Processing the call keyword arguments (line 651)
    kwargs_23344 = {}
    # Getting the type of 'np' (line 651)
    np_23341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 13), 'np', False)
    # Obtaining the member 'amax' of a type (line 651)
    amax_23342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 13), np_23341, 'amax')
    # Calling amax(args, kwargs) (line 651)
    amax_call_result_23345 = invoke(stypy.reporting.localization.Localization(__file__, 651, 13), amax_23342, *[vals_23343], **kwargs_23344)
    
    # Assigning a type to the variable 'max_sv' (line 651)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 4), 'max_sv', amax_call_result_23345)
    
    # Assigning a BinOp to a Name (line 654):
    
    # Assigning a BinOp to a Name (line 654):
    float_23346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 8), 'float')
    # Getting the type of 'max_sv' (line 654)
    max_sv_23347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'max_sv')
    # Applying the binary operator 'div' (line 654)
    result_div_23348 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 8), 'div', float_23346, max_sv_23347)
    
    # Assigning a type to the variable 'c' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 'c', result_div_23348)
    
    # Assigning a BinOp to a Name (line 655):
    
    # Assigning a BinOp to a Name (line 655):
    # Getting the type of 'A' (line 655)
    A_23349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 9), 'A')
    # Getting the type of 'c' (line 655)
    c_23350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 13), 'c')
    
    # Call to identity(...): (line 655)
    # Processing the call arguments (line 655)
    
    # Obtaining the type of the subscript
    int_23353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 35), 'int')
    # Getting the type of 'A' (line 655)
    A_23354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 27), 'A', False)
    # Obtaining the member 'shape' of a type (line 655)
    shape_23355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 27), A_23354, 'shape')
    # Obtaining the member '__getitem__' of a type (line 655)
    getitem___23356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 27), shape_23355, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 655)
    subscript_call_result_23357 = invoke(stypy.reporting.localization.Localization(__file__, 655, 27), getitem___23356, int_23353)
    
    # Processing the call keyword arguments (line 655)
    kwargs_23358 = {}
    # Getting the type of 'np' (line 655)
    np_23351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 15), 'np', False)
    # Obtaining the member 'identity' of a type (line 655)
    identity_23352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 15), np_23351, 'identity')
    # Calling identity(args, kwargs) (line 655)
    identity_call_result_23359 = invoke(stypy.reporting.localization.Localization(__file__, 655, 15), identity_23352, *[subscript_call_result_23357], **kwargs_23358)
    
    # Applying the binary operator '*' (line 655)
    result_mul_23360 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 13), '*', c_23350, identity_call_result_23359)
    
    # Applying the binary operator '+' (line 655)
    result_add_23361 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 9), '+', A_23349, result_mul_23360)
    
    # Assigning a type to the variable 'S0' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'S0', result_add_23361)
    
    # Assigning a Name to a Name (line 656):
    
    # Assigning a Name to a Name (line 656):
    # Getting the type of 'errest' (line 656)
    errest_23362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 18), 'errest')
    # Assigning a type to the variable 'prev_errest' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'prev_errest', errest_23362)
    
    
    # Call to range(...): (line 657)
    # Processing the call arguments (line 657)
    int_23364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 19), 'int')
    # Processing the call keyword arguments (line 657)
    kwargs_23365 = {}
    # Getting the type of 'range' (line 657)
    range_23363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 13), 'range', False)
    # Calling range(args, kwargs) (line 657)
    range_call_result_23366 = invoke(stypy.reporting.localization.Localization(__file__, 657, 13), range_23363, *[int_23364], **kwargs_23365)
    
    # Testing the type of a for loop iterable (line 657)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 657, 4), range_call_result_23366)
    # Getting the type of the for loop variable (line 657)
    for_loop_var_23367 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 657, 4), range_call_result_23366)
    # Assigning a type to the variable 'i' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'i', for_loop_var_23367)
    # SSA begins for a for statement (line 657)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 658):
    
    # Assigning a Call to a Name (line 658):
    
    # Call to inv(...): (line 658)
    # Processing the call arguments (line 658)
    # Getting the type of 'S0' (line 658)
    S0_23369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 18), 'S0', False)
    # Processing the call keyword arguments (line 658)
    kwargs_23370 = {}
    # Getting the type of 'inv' (line 658)
    inv_23368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 14), 'inv', False)
    # Calling inv(args, kwargs) (line 658)
    inv_call_result_23371 = invoke(stypy.reporting.localization.Localization(__file__, 658, 14), inv_23368, *[S0_23369], **kwargs_23370)
    
    # Assigning a type to the variable 'iS0' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 8), 'iS0', inv_call_result_23371)
    
    # Assigning a BinOp to a Name (line 659):
    
    # Assigning a BinOp to a Name (line 659):
    float_23372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 13), 'float')
    # Getting the type of 'S0' (line 659)
    S0_23373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 18), 'S0')
    # Getting the type of 'iS0' (line 659)
    iS0_23374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 23), 'iS0')
    # Applying the binary operator '+' (line 659)
    result_add_23375 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 18), '+', S0_23373, iS0_23374)
    
    # Applying the binary operator '*' (line 659)
    result_mul_23376 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 13), '*', float_23372, result_add_23375)
    
    # Assigning a type to the variable 'S0' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 8), 'S0', result_mul_23376)
    
    # Assigning a BinOp to a Name (line 660):
    
    # Assigning a BinOp to a Name (line 660):
    float_23377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 13), 'float')
    
    # Call to dot(...): (line 660)
    # Processing the call arguments (line 660)
    # Getting the type of 'S0' (line 660)
    S0_23379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 22), 'S0', False)
    # Getting the type of 'S0' (line 660)
    S0_23380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 25), 'S0', False)
    # Processing the call keyword arguments (line 660)
    kwargs_23381 = {}
    # Getting the type of 'dot' (line 660)
    dot_23378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 18), 'dot', False)
    # Calling dot(args, kwargs) (line 660)
    dot_call_result_23382 = invoke(stypy.reporting.localization.Localization(__file__, 660, 18), dot_23378, *[S0_23379, S0_23380], **kwargs_23381)
    
    # Getting the type of 'S0' (line 660)
    S0_23383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 29), 'S0')
    # Applying the binary operator '+' (line 660)
    result_add_23384 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 18), '+', dot_call_result_23382, S0_23383)
    
    # Applying the binary operator '*' (line 660)
    result_mul_23385 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 13), '*', float_23377, result_add_23384)
    
    # Assigning a type to the variable 'Pp' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'Pp', result_mul_23385)
    
    # Assigning a Call to a Name (line 661):
    
    # Assigning a Call to a Name (line 661):
    
    # Call to norm(...): (line 661)
    # Processing the call arguments (line 661)
    
    # Call to dot(...): (line 661)
    # Processing the call arguments (line 661)
    # Getting the type of 'Pp' (line 661)
    Pp_23388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 26), 'Pp', False)
    # Getting the type of 'Pp' (line 661)
    Pp_23389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 29), 'Pp', False)
    # Processing the call keyword arguments (line 661)
    kwargs_23390 = {}
    # Getting the type of 'dot' (line 661)
    dot_23387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 22), 'dot', False)
    # Calling dot(args, kwargs) (line 661)
    dot_call_result_23391 = invoke(stypy.reporting.localization.Localization(__file__, 661, 22), dot_23387, *[Pp_23388, Pp_23389], **kwargs_23390)
    
    # Getting the type of 'Pp' (line 661)
    Pp_23392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 33), 'Pp', False)
    # Applying the binary operator '-' (line 661)
    result_sub_23393 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 22), '-', dot_call_result_23391, Pp_23392)
    
    int_23394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 36), 'int')
    # Processing the call keyword arguments (line 661)
    kwargs_23395 = {}
    # Getting the type of 'norm' (line 661)
    norm_23386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 17), 'norm', False)
    # Calling norm(args, kwargs) (line 661)
    norm_call_result_23396 = invoke(stypy.reporting.localization.Localization(__file__, 661, 17), norm_23386, *[result_sub_23393, int_23394], **kwargs_23395)
    
    # Assigning a type to the variable 'errest' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 8), 'errest', norm_call_result_23396)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'errest' (line 662)
    errest_23397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 11), 'errest')
    # Getting the type of 'errtol' (line 662)
    errtol_23398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 20), 'errtol')
    # Applying the binary operator '<' (line 662)
    result_lt_23399 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 11), '<', errest_23397, errtol_23398)
    
    
    # Getting the type of 'prev_errest' (line 662)
    prev_errest_23400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 30), 'prev_errest')
    # Getting the type of 'errest' (line 662)
    errest_23401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 45), 'errest')
    # Applying the binary operator '==' (line 662)
    result_eq_23402 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 30), '==', prev_errest_23400, errest_23401)
    
    # Applying the binary operator 'or' (line 662)
    result_or_keyword_23403 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 11), 'or', result_lt_23399, result_eq_23402)
    
    # Testing the type of an if condition (line 662)
    if_condition_23404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 662, 8), result_or_keyword_23403)
    # Assigning a type to the variable 'if_condition_23404' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'if_condition_23404', if_condition_23404)
    # SSA begins for if statement (line 662)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 662)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 664):
    
    # Assigning a Name to a Name (line 664):
    # Getting the type of 'errest' (line 664)
    errest_23405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 22), 'errest')
    # Assigning a type to the variable 'prev_errest' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 8), 'prev_errest', errest_23405)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'disp' (line 665)
    disp_23406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 7), 'disp')
    # Testing the type of an if condition (line 665)
    if_condition_23407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 665, 4), disp_23406)
    # Assigning a type to the variable 'if_condition_23407' (line 665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'if_condition_23407', if_condition_23407)
    # SSA begins for if statement (line 665)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    
    # Call to isfinite(...): (line 666)
    # Processing the call arguments (line 666)
    # Getting the type of 'errest' (line 666)
    errest_23409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 24), 'errest', False)
    # Processing the call keyword arguments (line 666)
    kwargs_23410 = {}
    # Getting the type of 'isfinite' (line 666)
    isfinite_23408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 15), 'isfinite', False)
    # Calling isfinite(args, kwargs) (line 666)
    isfinite_call_result_23411 = invoke(stypy.reporting.localization.Localization(__file__, 666, 15), isfinite_23408, *[errest_23409], **kwargs_23410)
    
    # Applying the 'not' unary operator (line 666)
    result_not__23412 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 11), 'not', isfinite_call_result_23411)
    
    
    # Getting the type of 'errest' (line 666)
    errest_23413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 35), 'errest')
    # Getting the type of 'errtol' (line 666)
    errtol_23414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 45), 'errtol')
    # Applying the binary operator '>=' (line 666)
    result_ge_23415 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 35), '>=', errest_23413, errtol_23414)
    
    # Applying the binary operator 'or' (line 666)
    result_or_keyword_23416 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 11), 'or', result_not__23412, result_ge_23415)
    
    # Testing the type of an if condition (line 666)
    if_condition_23417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 666, 8), result_or_keyword_23416)
    # Assigning a type to the variable 'if_condition_23417' (line 666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'if_condition_23417', if_condition_23417)
    # SSA begins for if statement (line 666)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 667)
    # Processing the call arguments (line 667)
    str_23419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 18), 'str', 'signm result may be inaccurate, approximate err =')
    # Getting the type of 'errest' (line 667)
    errest_23420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 71), 'errest', False)
    # Processing the call keyword arguments (line 667)
    kwargs_23421 = {}
    # Getting the type of 'print' (line 667)
    print_23418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'print', False)
    # Calling print(args, kwargs) (line 667)
    print_call_result_23422 = invoke(stypy.reporting.localization.Localization(__file__, 667, 12), print_23418, *[str_23419, errest_23420], **kwargs_23421)
    
    # SSA join for if statement (line 666)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'S0' (line 668)
    S0_23423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 15), 'S0')
    # Assigning a type to the variable 'stypy_return_type' (line 668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'stypy_return_type', S0_23423)
    # SSA branch for the else part of an if statement (line 665)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 670)
    tuple_23424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 670)
    # Adding element type (line 670)
    # Getting the type of 'S0' (line 670)
    S0_23425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 15), 'S0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 670, 15), tuple_23424, S0_23425)
    # Adding element type (line 670)
    # Getting the type of 'errest' (line 670)
    errest_23426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 19), 'errest')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 670, 15), tuple_23424, errest_23426)
    
    # Assigning a type to the variable 'stypy_return_type' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 8), 'stypy_return_type', tuple_23424)
    # SSA join for if statement (line 665)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'signm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'signm' in the type store
    # Getting the type of 'stypy_return_type' (line 593)
    stypy_return_type_23427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23427)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'signm'
    return stypy_return_type_23427

# Assigning a type to the variable 'signm' (line 593)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 0), 'signm', signm)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
