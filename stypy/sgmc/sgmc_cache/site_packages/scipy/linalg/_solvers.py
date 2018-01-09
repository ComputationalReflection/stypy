
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Matrix equation solver routines'''
2: # Author: Jeffrey Armstrong <jeff@approximatrix.com>
3: # February 24, 2012
4: 
5: # Modified: Chad Fulton <ChadFulton@gmail.com>
6: # June 19, 2014
7: 
8: # Modified: Ilhan Polat <ilhanpolat@gmail.com>
9: # September 13, 2016
10: 
11: from __future__ import division, print_function, absolute_import
12: 
13: import warnings
14: import numpy as np
15: from numpy.linalg import inv, LinAlgError, norm, cond, svd
16: 
17: from .basic import solve, solve_triangular, matrix_balance
18: from .lapack import get_lapack_funcs
19: from .decomp_schur import schur
20: from .decomp_lu import lu
21: from .decomp_qr import qr
22: from ._decomp_qz import ordqz
23: from .decomp import _asarray_validated
24: from .special_matrices import kron, block_diag
25: 
26: __all__ = ['solve_sylvester',
27:            'solve_continuous_lyapunov', 'solve_discrete_lyapunov',
28:            'solve_continuous_are', 'solve_discrete_are']
29: 
30: 
31: def solve_sylvester(a, b, q):
32:     '''
33:     Computes a solution (X) to the Sylvester equation :math:`AX + XB = Q`.
34: 
35:     Parameters
36:     ----------
37:     a : (M, M) array_like
38:         Leading matrix of the Sylvester equation
39:     b : (N, N) array_like
40:         Trailing matrix of the Sylvester equation
41:     q : (M, N) array_like
42:         Right-hand side
43: 
44:     Returns
45:     -------
46:     x : (M, N) ndarray
47:         The solution to the Sylvester equation.
48: 
49:     Raises
50:     ------
51:     LinAlgError
52:         If solution was not found
53: 
54:     Notes
55:     -----
56:     Computes a solution to the Sylvester matrix equation via the Bartels-
57:     Stewart algorithm.  The A and B matrices first undergo Schur
58:     decompositions.  The resulting matrices are used to construct an
59:     alternative Sylvester equation (``RY + YS^T = F``) where the R and S
60:     matrices are in quasi-triangular form (or, when R, S or F are complex,
61:     triangular form).  The simplified equation is then solved using
62:     ``*TRSYL`` from LAPACK directly.
63: 
64:     .. versionadded:: 0.11.0
65: 
66:     Examples
67:     --------
68:     Given `a`, `b`, and `q` solve for `x`:
69: 
70:     >>> from scipy import linalg
71:     >>> a = np.array([[-3, -2, 0], [-1, -1, 3], [3, -5, -1]])
72:     >>> b = np.array([[1]])
73:     >>> q = np.array([[1],[2],[3]])
74:     >>> x = linalg.solve_sylvester(a, b, q)
75:     >>> x
76:     array([[ 0.0625],
77:            [-0.5625],
78:            [ 0.6875]])
79:     >>> np.allclose(a.dot(x) + x.dot(b), q)
80:     True
81: 
82:     '''
83: 
84:     # Compute the Schur decomp form of a
85:     r, u = schur(a, output='real')
86: 
87:     # Compute the Schur decomp of b
88:     s, v = schur(b.conj().transpose(), output='real')
89: 
90:     # Construct f = u'*q*v
91:     f = np.dot(np.dot(u.conj().transpose(), q), v)
92: 
93:     # Call the Sylvester equation solver
94:     trsyl, = get_lapack_funcs(('trsyl',), (r, s, f))
95:     if trsyl is None:
96:         raise RuntimeError('LAPACK implementation does not contain a proper '
97:                            'Sylvester equation solver (TRSYL)')
98:     y, scale, info = trsyl(r, s, f, tranb='C')
99: 
100:     y = scale*y
101: 
102:     if info < 0:
103:         raise LinAlgError("Illegal value encountered in "
104:                           "the %d term" % (-info,))
105: 
106:     return np.dot(np.dot(u, y), v.conj().transpose())
107: 
108: 
109: def solve_continuous_lyapunov(a, q):
110:     '''
111:     Solves the continuous Lyapunov equation :math:`AX + XA^H = Q`.
112: 
113:     Uses the Bartels-Stewart algorithm to find :math:`X`.
114: 
115:     Parameters
116:     ----------
117:     a : array_like
118:         A square matrix
119: 
120:     q : array_like
121:         Right-hand side square matrix
122: 
123:     Returns
124:     -------
125:     x : ndarray
126:         Solution to the continuous Lyapunov equation
127: 
128:     See Also
129:     --------
130:     solve_discrete_lyapunov : computes the solution to the discrete-time
131:         Lyapunov equation
132:     solve_sylvester : computes the solution to the Sylvester equation
133: 
134:     Notes
135:     -----
136:     The continuous Lyapunov equation is a special form of the Sylvester
137:     equation, hence this solver relies on LAPACK routine ?TRSYL.
138: 
139:     .. versionadded:: 0.11.0
140: 
141:     Examples
142:     --------
143:     Given `a` and `q` solve for `x`:
144: 
145:     >>> from scipy import linalg
146:     >>> a = np.array([[-3, -2, 0], [-1, -1, 0], [0, -5, -1]])
147:     >>> b = np.array([2, 4, -1])
148:     >>> q = np.eye(3)
149:     >>> x = linalg.solve_continuous_lyapunov(a, q)
150:     >>> x
151:     array([[ -0.75  ,   0.875 ,  -3.75  ],
152:            [  0.875 ,  -1.375 ,   5.3125],
153:            [ -3.75  ,   5.3125, -27.0625]])
154:     >>> np.allclose(a.dot(x) + x.dot(a.T), q)
155:     True
156:     '''
157: 
158:     a = np.atleast_2d(_asarray_validated(a, check_finite=True))
159:     q = np.atleast_2d(_asarray_validated(q, check_finite=True))
160: 
161:     r_or_c = float
162: 
163:     for ind, _ in enumerate((a, q)):
164:         if np.iscomplexobj(_):
165:             r_or_c = complex
166: 
167:         if not np.equal(*_.shape):
168:             raise ValueError("Matrix {} should be square.".format("aq"[ind]))
169: 
170:     # Shape consistency check
171:     if a.shape != q.shape:
172:         raise ValueError("Matrix a and q should have the same shape.")
173: 
174:     # Compute the Schur decomp form of a
175:     r, u = schur(a, output='real')
176: 
177:     # Construct f = u'*q*u
178:     f = u.conj().T.dot(q.dot(u))
179: 
180:     # Call the Sylvester equation solver
181:     trsyl = get_lapack_funcs('trsyl', (r, f))
182: 
183:     dtype_string = 'T' if r_or_c == float else 'C'
184:     y, scale, info = trsyl(r, r, f, tranb=dtype_string)
185: 
186:     if info < 0:
187:         raise ValueError('?TRSYL exited with the internal error '
188:                          '"illegal value in argument number {}.". See '
189:                          'LAPACK documentation for the ?TRSYL error codes.'
190:                          ''.format(-info))
191:     elif info == 1:
192:         warnings.warn('Input "a" has an eigenvalue pair whose sum is '
193:                       'very close to or exactly zero. The solution is '
194:                       'obtained via perturbing the coefficients.',
195:                       RuntimeWarning)
196:     y *= scale
197: 
198:     return u.dot(y).dot(u.conj().T)
199: 
200: # For backwards compatibility, keep the old name
201: solve_lyapunov = solve_continuous_lyapunov
202: 
203: 
204: def _solve_discrete_lyapunov_direct(a, q):
205:     '''
206:     Solves the discrete Lyapunov equation directly.
207: 
208:     This function is called by the `solve_discrete_lyapunov` function with
209:     `method=direct`. It is not supposed to be called directly.
210:     '''
211: 
212:     lhs = kron(a, a.conj())
213:     lhs = np.eye(lhs.shape[0]) - lhs
214:     x = solve(lhs, q.flatten())
215: 
216:     return np.reshape(x, q.shape)
217: 
218: 
219: def _solve_discrete_lyapunov_bilinear(a, q):
220:     '''
221:     Solves the discrete Lyapunov equation using a bilinear transformation.
222: 
223:     This function is called by the `solve_discrete_lyapunov` function with
224:     `method=bilinear`. It is not supposed to be called directly.
225:     '''
226:     eye = np.eye(a.shape[0])
227:     aH = a.conj().transpose()
228:     aHI_inv = inv(aH + eye)
229:     b = np.dot(aH - eye, aHI_inv)
230:     c = 2*np.dot(np.dot(inv(a + eye), q), aHI_inv)
231:     return solve_lyapunov(b.conj().transpose(), -c)
232: 
233: 
234: def solve_discrete_lyapunov(a, q, method=None):
235:     '''
236:     Solves the discrete Lyapunov equation :math:`AXA^H - X + Q = 0`.
237: 
238:     Parameters
239:     ----------
240:     a, q : (M, M) array_like
241:         Square matrices corresponding to A and Q in the equation
242:         above respectively. Must have the same shape.
243: 
244:     method : {'direct', 'bilinear'}, optional
245:         Type of solver.
246: 
247:         If not given, chosen to be ``direct`` if ``M`` is less than 10 and
248:         ``bilinear`` otherwise.
249: 
250:     Returns
251:     -------
252:     x : ndarray
253:         Solution to the discrete Lyapunov equation
254: 
255:     See Also
256:     --------
257:     solve_continuous_lyapunov : computes the solution to the continuous-time
258:         Lyapunov equation
259: 
260:     Notes
261:     -----
262:     This section describes the available solvers that can be selected by the
263:     'method' parameter. The default method is *direct* if ``M`` is less than 10
264:     and ``bilinear`` otherwise.
265: 
266:     Method *direct* uses a direct analytical solution to the discrete Lyapunov
267:     equation. The algorithm is given in, for example, [1]_. However it requires
268:     the linear solution of a system with dimension :math:`M^2` so that
269:     performance degrades rapidly for even moderately sized matrices.
270: 
271:     Method *bilinear* uses a bilinear transformation to convert the discrete
272:     Lyapunov equation to a continuous Lyapunov equation :math:`(BX+XB'=-C)`
273:     where :math:`B=(A-I)(A+I)^{-1}` and
274:     :math:`C=2(A' + I)^{-1} Q (A + I)^{-1}`. The continuous equation can be
275:     efficiently solved since it is a special case of a Sylvester equation.
276:     The transformation algorithm is from Popov (1964) as described in [2]_.
277: 
278:     .. versionadded:: 0.11.0
279: 
280:     References
281:     ----------
282:     .. [1] Hamilton, James D. Time Series Analysis, Princeton: Princeton
283:        University Press, 1994.  265.  Print.
284:        http://doc1.lbfl.li/aca/FLMF037168.pdf
285:     .. [2] Gajic, Z., and M.T.J. Qureshi. 2008.
286:        Lyapunov Matrix Equation in System Stability and Control.
287:        Dover Books on Engineering Series. Dover Publications.
288: 
289:     Examples
290:     --------
291:     Given `a` and `q` solve for `x`:
292: 
293:     >>> from scipy import linalg
294:     >>> a = np.array([[0.2, 0.5],[0.7, -0.9]])
295:     >>> q = np.eye(2)
296:     >>> x = linalg.solve_discrete_lyapunov(a, q)
297:     >>> x
298:     array([[ 0.70872893,  1.43518822],
299:            [ 1.43518822, -2.4266315 ]])
300:     >>> np.allclose(a.dot(x).dot(a.T)-x, -q)
301:     True
302: 
303:     '''
304:     a = np.asarray(a)
305:     q = np.asarray(q)
306:     if method is None:
307:         # Select automatically based on size of matrices
308:         if a.shape[0] >= 10:
309:             method = 'bilinear'
310:         else:
311:             method = 'direct'
312: 
313:     meth = method.lower()
314: 
315:     if meth == 'direct':
316:         x = _solve_discrete_lyapunov_direct(a, q)
317:     elif meth == 'bilinear':
318:         x = _solve_discrete_lyapunov_bilinear(a, q)
319:     else:
320:         raise ValueError('Unknown solver %s' % method)
321: 
322:     return x
323: 
324: 
325: def solve_continuous_are(a, b, q, r, e=None, s=None, balanced=True):
326:     r'''
327:     Solves the continuous-time algebraic Riccati equation (CARE).
328: 
329:     The CARE is defined as
330: 
331:     .. math::
332: 
333:           X A + A^H X - X B R^{-1} B^H X + Q = 0
334: 
335:     The limitations for a solution to exist are :
336: 
337:         * All eigenvalues of :math:`A` on the right half plane, should be
338:           controllable.
339: 
340:         * The associated hamiltonian pencil (See Notes), should have
341:           eigenvalues sufficiently away from the imaginary axis.
342: 
343:     Moreover, if ``e`` or ``s`` is not precisely ``None``, then the
344:     generalized version of CARE
345: 
346:     .. math::
347: 
348:           E^HXA + A^HXE - (E^HXB + S) R^{-1} (B^HXE + S^H) + Q = 0
349: 
350:     is solved. When omitted, ``e`` is assumed to be the identity and ``s``
351:     is assumed to be the zero matrix with sizes compatible with ``a`` and
352:     ``b`` respectively.
353: 
354:     Parameters
355:     ----------
356:     a : (M, M) array_like
357:         Square matrix
358:     b : (M, N) array_like
359:         Input
360:     q : (M, M) array_like
361:         Input
362:     r : (N, N) array_like
363:         Nonsingular square matrix
364:     e : (M, M) array_like, optional
365:         Nonsingular square matrix
366:     s : (M, N) array_like, optional
367:         Input
368:     balanced : bool, optional
369:         The boolean that indicates whether a balancing step is performed
370:         on the data. The default is set to True.
371: 
372:     Returns
373:     -------
374:     x : (M, M) ndarray
375:         Solution to the continuous-time algebraic Riccati equation.
376: 
377:     Raises
378:     ------
379:     LinAlgError
380:         For cases where the stable subspace of the pencil could not be
381:         isolated. See Notes section and the references for details.
382: 
383:     See Also
384:     --------
385:     solve_discrete_are : Solves the discrete-time algebraic Riccati equation
386: 
387:     Notes
388:     -----
389:     The equation is solved by forming the extended hamiltonian matrix pencil,
390:     as described in [1]_, :math:`H - \lambda J` given by the block matrices ::
391: 
392:         [ A    0    B ]             [ E   0    0 ]
393:         [-Q  -A^H  -S ] - \lambda * [ 0  E^H   0 ]
394:         [ S^H B^H   R ]             [ 0   0    0 ]
395: 
396:     and using a QZ decomposition method.
397: 
398:     In this algorithm, the fail conditions are linked to the symmetry
399:     of the product :math:`U_2 U_1^{-1}` and condition number of
400:     :math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the
401:     eigenvectors spanning the stable subspace with 2m rows and partitioned
402:     into two m-row matrices. See [1]_ and [2]_ for more details.
403: 
404:     In order to improve the QZ decomposition accuracy, the pencil goes
405:     through a balancing step where the sum of absolute values of
406:     :math:`H` and :math:`J` entries (after removing the diagonal entries of
407:     the sum) is balanced following the recipe given in [3]_.
408: 
409:     .. versionadded:: 0.11.0
410: 
411:     References
412:     ----------
413:     .. [1]  P. van Dooren , "A Generalized Eigenvalue Approach For Solving
414:        Riccati Equations.", SIAM Journal on Scientific and Statistical
415:        Computing, Vol.2(2), DOI: 10.1137/0902010
416: 
417:     .. [2] A.J. Laub, "A Schur Method for Solving Algebraic Riccati
418:        Equations.", Massachusetts Institute of Technology. Laboratory for
419:        Information and Decision Systems. LIDS-R ; 859. Available online :
420:        http://hdl.handle.net/1721.1/1301
421: 
422:     .. [3] P. Benner, "Symplectic Balancing of Hamiltonian Matrices", 2001,
423:        SIAM J. Sci. Comput., 2001, Vol.22(5), DOI: 10.1137/S1064827500367993
424: 
425:     Examples
426:     --------
427:     Given `a`, `b`, `q`, and `r` solve for `x`:
428: 
429:     >>> from scipy import linalg
430:     >>> a = np.array([[4, 3], [-4.5, -3.5]])
431:     >>> b = np.array([[1], [-1]])
432:     >>> q = np.array([[9, 6], [6, 4.]])
433:     >>> r = 1
434:     >>> x = linalg.solve_continuous_are(a, b, q, r)
435:     >>> x
436:     array([[ 21.72792206,  14.48528137],
437:            [ 14.48528137,   9.65685425]])
438:     >>> np.allclose(a.T.dot(x) + x.dot(a)-x.dot(b).dot(b.T).dot(x), -q)
439:     True
440: 
441:     '''
442: 
443:     # Validate input arguments
444:     a, b, q, r, e, s, m, n, r_or_c, gen_are = _are_validate_args(
445:                                                      a, b, q, r, e, s, 'care')
446: 
447:     H = np.empty((2*m+n, 2*m+n), dtype=r_or_c)
448:     H[:m, :m] = a
449:     H[:m, m:2*m] = 0.
450:     H[:m, 2*m:] = b
451:     H[m:2*m, :m] = -q
452:     H[m:2*m, m:2*m] = -a.conj().T
453:     H[m:2*m, 2*m:] = 0. if s is None else -s
454:     H[2*m:, :m] = 0. if s is None else s.conj().T
455:     H[2*m:, m:2*m] = b.conj().T
456:     H[2*m:, 2*m:] = r
457: 
458:     if gen_are and e is not None:
459:         J = block_diag(e, e.conj().T, np.zeros_like(r, dtype=r_or_c))
460:     else:
461:         J = block_diag(np.eye(2*m), np.zeros_like(r, dtype=r_or_c))
462: 
463:     if balanced:
464:         # xGEBAL does not remove the diagonals before scaling. Also
465:         # to avoid destroying the Symplectic structure, we follow Ref.3
466:         M = np.abs(H) + np.abs(J)
467:         M[np.diag_indices_from(M)] = 0.
468:         _, (sca, _) = matrix_balance(M, separate=1, permute=0)
469:         # do we need to bother?
470:         if not np.allclose(sca, np.ones_like(sca)):
471:             # Now impose diag(D,inv(D)) from Benner where D is
472:             # square root of s_i/s_(n+i) for i=0,....
473:             sca = np.log2(sca)
474:             # NOTE: Py3 uses "Bankers Rounding: round to the nearest even" !!
475:             s = np.round((sca[m:2*m] - sca[:m])/2)
476:             sca = 2 ** np.r_[s, -s, sca[2*m:]]
477:             # Elementwise multiplication via broadcasting.
478:             elwisescale = sca[:, None] * np.reciprocal(sca)
479:             H *= elwisescale
480:             J *= elwisescale
481: 
482:     # Deflate the pencil to 2m x 2m ala Ref.1, eq.(55)
483:     q, r = qr(H[:, -n:])
484:     H = q[:, n:].conj().T.dot(H[:, :2*m])
485:     J = q[:2*m, n:].conj().T.dot(J[:2*m, :2*m])
486: 
487:     # Decide on which output type is needed for QZ
488:     out_str = 'real' if r_or_c == float else 'complex'
489: 
490:     _, _, _, _, _, u = ordqz(H, J, sort='lhp', overwrite_a=True,
491:                              overwrite_b=True, check_finite=False,
492:                              output=out_str)
493: 
494:     # Get the relevant parts of the stable subspace basis
495:     if e is not None:
496:         u, _ = qr(np.vstack((e.dot(u[:m, :m]), u[m:, :m])))
497:     u00 = u[:m, :m]
498:     u10 = u[m:, :m]
499: 
500:     # Solve via back-substituion after checking the condition of u00
501:     up, ul, uu = lu(u00)
502:     if 1/cond(uu) < np.spacing(1.):
503:         raise LinAlgError('Failed to find a finite solution.')
504: 
505:     # Exploit the triangular structure
506:     x = solve_triangular(ul.conj().T,
507:                          solve_triangular(uu.conj().T,
508:                                           u10.conj().T,
509:                                           lower=True),
510:                          unit_diagonal=True,
511:                          ).conj().T.dot(up.conj().T)
512:     if balanced:
513:         x *= sca[:m, None] * sca[:m]
514: 
515:     # Check the deviation from symmetry for success
516:     u_sym = u00.conj().T.dot(u10)
517:     n_u_sym = norm(u_sym, 1)
518:     u_sym = u_sym - u_sym.conj().T
519:     sym_threshold = np.max([np.spacing(1000.), n_u_sym])
520: 
521:     if norm(u_sym, 1) > sym_threshold:
522:         raise LinAlgError('The associated Hamiltonian pencil has eigenvalues '
523:                           'too close to the imaginary axis')
524: 
525:     return (x + x.conj().T)/2
526: 
527: 
528: def solve_discrete_are(a, b, q, r, e=None, s=None, balanced=True):
529:     r'''
530:     Solves the discrete-time algebraic Riccati equation (DARE).
531: 
532:     The DARE is defined as
533: 
534:     .. math::
535: 
536:           A^HXA - X - (A^HXB) (R + B^HXB)^{-1} (B^HXA) + Q = 0
537: 
538:     The limitations for a solution to exist are :
539: 
540:         * All eigenvalues of :math:`A` outside the unit disc, should be
541:           controllable.
542: 
543:         * The associated symplectic pencil (See Notes), should have
544:           eigenvalues sufficiently away from the unit circle.
545: 
546:     Moreover, if ``e`` and ``s`` are not both precisely ``None``, then the
547:     generalized version of DARE
548: 
549:     .. math::
550: 
551:           A^HXA - E^HXE - (A^HXB+S) (R+B^HXB)^{-1} (B^HXA+S^H) + Q = 0
552: 
553:     is solved. When omitted, ``e`` is assumed to be the identity and ``s``
554:     is assumed to be the zero matrix.
555: 
556:     Parameters
557:     ----------
558:     a : (M, M) array_like
559:         Square matrix
560:     b : (M, N) array_like
561:         Input
562:     q : (M, M) array_like
563:         Input
564:     r : (N, N) array_like
565:         Square matrix
566:     e : (M, M) array_like, optional
567:         Nonsingular square matrix
568:     s : (M, N) array_like, optional
569:         Input
570:     balanced : bool
571:         The boolean that indicates whether a balancing step is performed
572:         on the data. The default is set to True.
573: 
574:     Returns
575:     -------
576:     x : (M, M) ndarray
577:         Solution to the discrete algebraic Riccati equation.
578: 
579:     Raises
580:     ------
581:     LinAlgError
582:         For cases where the stable subspace of the pencil could not be
583:         isolated. See Notes section and the references for details.
584: 
585:     See Also
586:     --------
587:     solve_continuous_are : Solves the continuous algebraic Riccati equation
588: 
589:     Notes
590:     -----
591:     The equation is solved by forming the extended symplectic matrix pencil,
592:     as described in [1]_, :math:`H - \lambda J` given by the block matrices ::
593: 
594:            [  A   0   B ]             [ E   0   B ]
595:            [ -Q  E^H -S ] - \lambda * [ 0  A^H  0 ]
596:            [ S^H  0   R ]             [ 0 -B^H  0 ]
597: 
598:     and using a QZ decomposition method.
599: 
600:     In this algorithm, the fail conditions are linked to the symmetry
601:     of the product :math:`U_2 U_1^{-1}` and condition number of
602:     :math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the
603:     eigenvectors spanning the stable subspace with 2m rows and partitioned
604:     into two m-row matrices. See [1]_ and [2]_ for more details.
605: 
606:     In order to improve the QZ decomposition accuracy, the pencil goes
607:     through a balancing step where the sum of absolute values of
608:     :math:`H` and :math:`J` rows/cols (after removing the diagonal entries)
609:     is balanced following the recipe given in [3]_. If the data has small
610:     numerical noise, balancing may amplify their effects and some clean up
611:     is required.
612: 
613:     .. versionadded:: 0.11.0
614: 
615:     References
616:     ----------
617:     .. [1]  P. van Dooren , "A Generalized Eigenvalue Approach For Solving
618:        Riccati Equations.", SIAM Journal on Scientific and Statistical
619:        Computing, Vol.2(2), DOI: 10.1137/0902010
620: 
621:     .. [2] A.J. Laub, "A Schur Method for Solving Algebraic Riccati
622:        Equations.", Massachusetts Institute of Technology. Laboratory for
623:        Information and Decision Systems. LIDS-R ; 859. Available online :
624:        http://hdl.handle.net/1721.1/1301
625: 
626:     .. [3] P. Benner, "Symplectic Balancing of Hamiltonian Matrices", 2001,
627:        SIAM J. Sci. Comput., 2001, Vol.22(5), DOI: 10.1137/S1064827500367993
628: 
629:     Examples
630:     --------
631:     Given `a`, `b`, `q`, and `r` solve for `x`:
632: 
633:     >>> from scipy import linalg as la
634:     >>> a = np.array([[0, 1], [0, -1]])
635:     >>> b = np.array([[1, 0], [2, 1]])
636:     >>> q = np.array([[-4, -4], [-4, 7]])
637:     >>> r = np.array([[9, 3], [3, 1]])
638:     >>> x = la.solve_discrete_are(a, b, q, r)
639:     >>> x
640:     array([[-4., -4.],
641:            [-4.,  7.]])
642:     >>> R = la.solve(r + b.T.dot(x).dot(b), b.T.dot(x).dot(a))
643:     >>> np.allclose(a.T.dot(x).dot(a) - x - a.T.dot(x).dot(b).dot(R), -q)
644:     True
645: 
646:     '''
647: 
648:     # Validate input arguments
649:     a, b, q, r, e, s, m, n, r_or_c, gen_are = _are_validate_args(
650:                                                      a, b, q, r, e, s, 'dare')
651: 
652:     # Form the matrix pencil
653:     H = np.zeros((2*m+n, 2*m+n), dtype=r_or_c)
654:     H[:m, :m] = a
655:     H[:m, 2*m:] = b
656:     H[m:2*m, :m] = -q
657:     H[m:2*m, m:2*m] = np.eye(m) if e is None else e.conj().T
658:     H[m:2*m, 2*m:] = 0. if s is None else -s
659:     H[2*m:, :m] = 0. if s is None else s.conj().T
660:     H[2*m:, 2*m:] = r
661: 
662:     J = np.zeros_like(H, dtype=r_or_c)
663:     J[:m, :m] = np.eye(m) if e is None else e
664:     J[m:2*m, m:2*m] = a.conj().T
665:     J[2*m:, m:2*m] = -b.conj().T
666: 
667:     if balanced:
668:         # xGEBAL does not remove the diagonals before scaling. Also
669:         # to avoid destroying the Symplectic structure, we follow Ref.3
670:         M = np.abs(H) + np.abs(J)
671:         M[np.diag_indices_from(M)] = 0.
672:         _, (sca, _) = matrix_balance(M, separate=1, permute=0)
673:         # do we need to bother?
674:         if not np.allclose(sca, np.ones_like(sca)):
675:             # Now impose diag(D,inv(D)) from Benner where D is
676:             # square root of s_i/s_(n+i) for i=0,....
677:             sca = np.log2(sca)
678:             # NOTE: Py3 uses "Bankers Rounding: round to the nearest even" !!
679:             s = np.round((sca[m:2*m] - sca[:m])/2)
680:             sca = 2 ** np.r_[s, -s, sca[2*m:]]
681:             # Elementwise multiplication via broadcasting.
682:             elwisescale = sca[:, None] * np.reciprocal(sca)
683:             H *= elwisescale
684:             J *= elwisescale
685: 
686:     # Deflate the pencil by the R column ala Ref.1
687:     q_of_qr, _ = qr(H[:, -n:])
688:     H = q_of_qr[:, n:].conj().T.dot(H[:, :2*m])
689:     J = q_of_qr[:, n:].conj().T.dot(J[:, :2*m])
690: 
691:     # Decide on which output type is needed for QZ
692:     out_str = 'real' if r_or_c == float else 'complex'
693: 
694:     _, _, _, _, _, u = ordqz(H, J, sort='iuc',
695:                              overwrite_a=True,
696:                              overwrite_b=True,
697:                              check_finite=False,
698:                              output=out_str)
699: 
700:     # Get the relevant parts of the stable subspace basis
701:     if e is not None:
702:         u, _ = qr(np.vstack((e.dot(u[:m, :m]), u[m:, :m])))
703:     u00 = u[:m, :m]
704:     u10 = u[m:, :m]
705: 
706:     # Solve via back-substituion after checking the condition of u00
707:     up, ul, uu = lu(u00)
708: 
709:     if 1/cond(uu) < np.spacing(1.):
710:         raise LinAlgError('Failed to find a finite solution.')
711: 
712:     # Exploit the triangular structure
713:     x = solve_triangular(ul.conj().T,
714:                          solve_triangular(uu.conj().T,
715:                                           u10.conj().T,
716:                                           lower=True),
717:                          unit_diagonal=True,
718:                          ).conj().T.dot(up.conj().T)
719:     if balanced:
720:         x *= sca[:m, None] * sca[:m]
721: 
722:     # Check the deviation from symmetry for success
723:     u_sym = u00.conj().T.dot(u10)
724:     n_u_sym = norm(u_sym, 1)
725:     u_sym = u_sym - u_sym.conj().T
726:     sym_threshold = np.max([np.spacing(1000.), n_u_sym])
727: 
728:     if norm(u_sym, 1) > sym_threshold:
729:         raise LinAlgError('The associated symplectic pencil has eigenvalues'
730:                           'too close to the unit circle')
731: 
732:     return (x + x.conj().T)/2
733: 
734: 
735: def _are_validate_args(a, b, q, r, e, s, eq_type='care'):
736:     '''
737:     A helper function to validate the arguments supplied to the
738:     Riccati equation solvers. Any discrepancy found in the input
739:     matrices leads to a ``ValueError`` exception.
740: 
741:     Essentially, it performs:
742: 
743:         - a check whether the input is free of NaN and Infs.
744:         - a pass for the data through ``numpy.atleast_2d()``
745:         - squareness check of the relevant arrays,
746:         - shape consistency check of the arrays,
747:         - singularity check of the relevant arrays,
748:         - symmetricity check of the relevant matrices,
749:         - a check whether the regular or the generalized version is asked.
750: 
751:     This function is used by ``solve_continuous_are`` and
752:     ``solve_discrete_are``.
753: 
754:     Parameters
755:     ----------
756:     a, b, q, r, e, s : array_like
757:         Input data
758:     eq_type : str
759:         Accepted arguments are 'care' and 'dare'.
760: 
761:     Returns
762:     -------
763:     a, b, q, r, e, s : ndarray
764:         Regularized input data
765:     m, n : int
766:         shape of the problem
767:     r_or_c : type
768:         Data type of the problem, returns float or complex
769:     gen_or_not : bool
770:         Type of the equation, True for generalized and False for regular ARE.
771: 
772:     '''
773: 
774:     if not eq_type.lower() in ('dare', 'care'):
775:         raise ValueError("Equation type unknown. "
776:                          "Only 'care' and 'dare' is understood")
777: 
778:     a = np.atleast_2d(_asarray_validated(a, check_finite=True))
779:     b = np.atleast_2d(_asarray_validated(b, check_finite=True))
780:     q = np.atleast_2d(_asarray_validated(q, check_finite=True))
781:     r = np.atleast_2d(_asarray_validated(r, check_finite=True))
782: 
783:     # Get the correct data types otherwise Numpy complains
784:     # about pushing complex numbers into real arrays.
785:     r_or_c = complex if np.iscomplexobj(b) else float
786: 
787:     for ind, mat in enumerate((a, q, r)):
788:         if np.iscomplexobj(mat):
789:             r_or_c = complex
790: 
791:         if not np.equal(*mat.shape):
792:             raise ValueError("Matrix {} should be square.".format("aqr"[ind]))
793: 
794:     # Shape consistency checks
795:     m, n = b.shape
796:     if m != a.shape[0]:
797:         raise ValueError("Matrix a and b should have the same number of rows.")
798:     if m != q.shape[0]:
799:         raise ValueError("Matrix a and q should have the same shape.")
800:     if n != r.shape[0]:
801:         raise ValueError("Matrix b and r should have the same number of cols.")
802: 
803:     # Check if the data matrices q, r are (sufficiently) hermitian
804:     for ind, mat in enumerate((q, r)):
805:         if norm(mat - mat.conj().T, 1) > np.spacing(norm(mat, 1))*100:
806:             raise ValueError("Matrix {} should be symmetric/hermitian."
807:                              "".format("qr"[ind]))
808: 
809:     # Continuous time ARE should have a nonsingular r matrix.
810:     if eq_type == 'care':
811:         min_sv = svd(r, compute_uv=False)[-1]
812:         if min_sv == 0. or min_sv < np.spacing(1.)*norm(r, 1):
813:             raise ValueError('Matrix r is numerically singular.')
814: 
815:     # Check if the generalized case is required with omitted arguments
816:     # perform late shape checking etc.
817:     generalized_case = e is not None or s is not None
818: 
819:     if generalized_case:
820:         if e is not None:
821:             e = np.atleast_2d(_asarray_validated(e, check_finite=True))
822:             if not np.equal(*e.shape):
823:                 raise ValueError("Matrix e should be square.")
824:             if m != e.shape[0]:
825:                 raise ValueError("Matrix a and e should have the same shape.")
826:             # numpy.linalg.cond doesn't check for exact zeros and
827:             # emits a runtime warning. Hence the following manual check.
828:             min_sv = svd(e, compute_uv=False)[-1]
829:             if min_sv == 0. or min_sv < np.spacing(1.) * norm(e, 1):
830:                 raise ValueError('Matrix e is numerically singular.')
831:             if np.iscomplexobj(e):
832:                 r_or_c = complex
833:         if s is not None:
834:             s = np.atleast_2d(_asarray_validated(s, check_finite=True))
835:             if s.shape != b.shape:
836:                 raise ValueError("Matrix b and s should have the same shape.")
837:             if np.iscomplexobj(s):
838:                 r_or_c = complex
839: 
840:     return a, b, q, r, e, s, m, n, r_or_c, generalized_case
841: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_35939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Matrix equation solver routines')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import warnings' statement (line 13)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import numpy' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35940 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy')

if (type(import_35940) is not StypyTypeError):

    if (import_35940 != 'pyd_module'):
        __import__(import_35940)
        sys_modules_35941 = sys.modules[import_35940]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'np', sys_modules_35941.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', import_35940)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.linalg import inv, LinAlgError, norm, cond, svd' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35942 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.linalg')

if (type(import_35942) is not StypyTypeError):

    if (import_35942 != 'pyd_module'):
        __import__(import_35942)
        sys_modules_35943 = sys.modules[import_35942]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.linalg', sys_modules_35943.module_type_store, module_type_store, ['inv', 'LinAlgError', 'norm', 'cond', 'svd'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_35943, sys_modules_35943.module_type_store, module_type_store)
    else:
        from numpy.linalg import inv, LinAlgError, norm, cond, svd

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.linalg', None, module_type_store, ['inv', 'LinAlgError', 'norm', 'cond', 'svd'], [inv, LinAlgError, norm, cond, svd])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.linalg', import_35942)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.linalg.basic import solve, solve_triangular, matrix_balance' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35944 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg.basic')

if (type(import_35944) is not StypyTypeError):

    if (import_35944 != 'pyd_module'):
        __import__(import_35944)
        sys_modules_35945 = sys.modules[import_35944]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg.basic', sys_modules_35945.module_type_store, module_type_store, ['solve', 'solve_triangular', 'matrix_balance'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_35945, sys_modules_35945.module_type_store, module_type_store)
    else:
        from scipy.linalg.basic import solve, solve_triangular, matrix_balance

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg.basic', None, module_type_store, ['solve', 'solve_triangular', 'matrix_balance'], [solve, solve_triangular, matrix_balance])

else:
    # Assigning a type to the variable 'scipy.linalg.basic' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg.basic', import_35944)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.linalg.lapack import get_lapack_funcs' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35946 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.linalg.lapack')

if (type(import_35946) is not StypyTypeError):

    if (import_35946 != 'pyd_module'):
        __import__(import_35946)
        sys_modules_35947 = sys.modules[import_35946]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.linalg.lapack', sys_modules_35947.module_type_store, module_type_store, ['get_lapack_funcs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_35947, sys_modules_35947.module_type_store, module_type_store)
    else:
        from scipy.linalg.lapack import get_lapack_funcs

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.linalg.lapack', None, module_type_store, ['get_lapack_funcs'], [get_lapack_funcs])

else:
    # Assigning a type to the variable 'scipy.linalg.lapack' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.linalg.lapack', import_35946)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy.linalg.decomp_schur import schur' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35948 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.linalg.decomp_schur')

if (type(import_35948) is not StypyTypeError):

    if (import_35948 != 'pyd_module'):
        __import__(import_35948)
        sys_modules_35949 = sys.modules[import_35948]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.linalg.decomp_schur', sys_modules_35949.module_type_store, module_type_store, ['schur'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_35949, sys_modules_35949.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp_schur import schur

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.linalg.decomp_schur', None, module_type_store, ['schur'], [schur])

else:
    # Assigning a type to the variable 'scipy.linalg.decomp_schur' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.linalg.decomp_schur', import_35948)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy.linalg.decomp_lu import lu' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35950 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg.decomp_lu')

if (type(import_35950) is not StypyTypeError):

    if (import_35950 != 'pyd_module'):
        __import__(import_35950)
        sys_modules_35951 = sys.modules[import_35950]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg.decomp_lu', sys_modules_35951.module_type_store, module_type_store, ['lu'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_35951, sys_modules_35951.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp_lu import lu

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg.decomp_lu', None, module_type_store, ['lu'], [lu])

else:
    # Assigning a type to the variable 'scipy.linalg.decomp_lu' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg.decomp_lu', import_35950)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from scipy.linalg.decomp_qr import qr' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35952 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.linalg.decomp_qr')

if (type(import_35952) is not StypyTypeError):

    if (import_35952 != 'pyd_module'):
        __import__(import_35952)
        sys_modules_35953 = sys.modules[import_35952]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.linalg.decomp_qr', sys_modules_35953.module_type_store, module_type_store, ['qr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_35953, sys_modules_35953.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp_qr import qr

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.linalg.decomp_qr', None, module_type_store, ['qr'], [qr])

else:
    # Assigning a type to the variable 'scipy.linalg.decomp_qr' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.linalg.decomp_qr', import_35952)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from scipy.linalg._decomp_qz import ordqz' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35954 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.linalg._decomp_qz')

if (type(import_35954) is not StypyTypeError):

    if (import_35954 != 'pyd_module'):
        __import__(import_35954)
        sys_modules_35955 = sys.modules[import_35954]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.linalg._decomp_qz', sys_modules_35955.module_type_store, module_type_store, ['ordqz'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_35955, sys_modules_35955.module_type_store, module_type_store)
    else:
        from scipy.linalg._decomp_qz import ordqz

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.linalg._decomp_qz', None, module_type_store, ['ordqz'], [ordqz])

else:
    # Assigning a type to the variable 'scipy.linalg._decomp_qz' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.linalg._decomp_qz', import_35954)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from scipy.linalg.decomp import _asarray_validated' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35956 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy.linalg.decomp')

if (type(import_35956) is not StypyTypeError):

    if (import_35956 != 'pyd_module'):
        __import__(import_35956)
        sys_modules_35957 = sys.modules[import_35956]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy.linalg.decomp', sys_modules_35957.module_type_store, module_type_store, ['_asarray_validated'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_35957, sys_modules_35957.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp import _asarray_validated

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy.linalg.decomp', None, module_type_store, ['_asarray_validated'], [_asarray_validated])

else:
    # Assigning a type to the variable 'scipy.linalg.decomp' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy.linalg.decomp', import_35956)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from scipy.linalg.special_matrices import kron, block_diag' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35958 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.linalg.special_matrices')

if (type(import_35958) is not StypyTypeError):

    if (import_35958 != 'pyd_module'):
        __import__(import_35958)
        sys_modules_35959 = sys.modules[import_35958]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.linalg.special_matrices', sys_modules_35959.module_type_store, module_type_store, ['kron', 'block_diag'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_35959, sys_modules_35959.module_type_store, module_type_store)
    else:
        from scipy.linalg.special_matrices import kron, block_diag

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.linalg.special_matrices', None, module_type_store, ['kron', 'block_diag'], [kron, block_diag])

else:
    # Assigning a type to the variable 'scipy.linalg.special_matrices' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.linalg.special_matrices', import_35958)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 26):

# Assigning a List to a Name (line 26):

# Assigning a List to a Name (line 26):
__all__ = ['solve_sylvester', 'solve_continuous_lyapunov', 'solve_discrete_lyapunov', 'solve_continuous_are', 'solve_discrete_are']
module_type_store.set_exportable_members(['solve_sylvester', 'solve_continuous_lyapunov', 'solve_discrete_lyapunov', 'solve_continuous_are', 'solve_discrete_are'])

# Obtaining an instance of the builtin type 'list' (line 26)
list_35960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_35961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'str', 'solve_sylvester')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_35960, str_35961)
# Adding element type (line 26)
str_35962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 11), 'str', 'solve_continuous_lyapunov')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_35960, str_35962)
# Adding element type (line 26)
str_35963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 40), 'str', 'solve_discrete_lyapunov')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_35960, str_35963)
# Adding element type (line 26)
str_35964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'str', 'solve_continuous_are')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_35960, str_35964)
# Adding element type (line 26)
str_35965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 35), 'str', 'solve_discrete_are')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_35960, str_35965)

# Assigning a type to the variable '__all__' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '__all__', list_35960)

@norecursion
def solve_sylvester(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'solve_sylvester'
    module_type_store = module_type_store.open_function_context('solve_sylvester', 31, 0, False)
    
    # Passed parameters checking function
    solve_sylvester.stypy_localization = localization
    solve_sylvester.stypy_type_of_self = None
    solve_sylvester.stypy_type_store = module_type_store
    solve_sylvester.stypy_function_name = 'solve_sylvester'
    solve_sylvester.stypy_param_names_list = ['a', 'b', 'q']
    solve_sylvester.stypy_varargs_param_name = None
    solve_sylvester.stypy_kwargs_param_name = None
    solve_sylvester.stypy_call_defaults = defaults
    solve_sylvester.stypy_call_varargs = varargs
    solve_sylvester.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_sylvester', ['a', 'b', 'q'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_sylvester', localization, ['a', 'b', 'q'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_sylvester(...)' code ##################

    str_35966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, (-1)), 'str', '\n    Computes a solution (X) to the Sylvester equation :math:`AX + XB = Q`.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Leading matrix of the Sylvester equation\n    b : (N, N) array_like\n        Trailing matrix of the Sylvester equation\n    q : (M, N) array_like\n        Right-hand side\n\n    Returns\n    -------\n    x : (M, N) ndarray\n        The solution to the Sylvester equation.\n\n    Raises\n    ------\n    LinAlgError\n        If solution was not found\n\n    Notes\n    -----\n    Computes a solution to the Sylvester matrix equation via the Bartels-\n    Stewart algorithm.  The A and B matrices first undergo Schur\n    decompositions.  The resulting matrices are used to construct an\n    alternative Sylvester equation (``RY + YS^T = F``) where the R and S\n    matrices are in quasi-triangular form (or, when R, S or F are complex,\n    triangular form).  The simplified equation is then solved using\n    ``*TRSYL`` from LAPACK directly.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    Given `a`, `b`, and `q` solve for `x`:\n\n    >>> from scipy import linalg\n    >>> a = np.array([[-3, -2, 0], [-1, -1, 3], [3, -5, -1]])\n    >>> b = np.array([[1]])\n    >>> q = np.array([[1],[2],[3]])\n    >>> x = linalg.solve_sylvester(a, b, q)\n    >>> x\n    array([[ 0.0625],\n           [-0.5625],\n           [ 0.6875]])\n    >>> np.allclose(a.dot(x) + x.dot(b), q)\n    True\n\n    ')
    
    # Assigning a Call to a Tuple (line 85):
    
    # Assigning a Subscript to a Name (line 85):
    
    # Assigning a Subscript to a Name (line 85):
    
    # Obtaining the type of the subscript
    int_35967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'int')
    
    # Call to schur(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'a' (line 85)
    a_35969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'a', False)
    # Processing the call keyword arguments (line 85)
    str_35970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 27), 'str', 'real')
    keyword_35971 = str_35970
    kwargs_35972 = {'output': keyword_35971}
    # Getting the type of 'schur' (line 85)
    schur_35968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'schur', False)
    # Calling schur(args, kwargs) (line 85)
    schur_call_result_35973 = invoke(stypy.reporting.localization.Localization(__file__, 85, 11), schur_35968, *[a_35969], **kwargs_35972)
    
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___35974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), schur_call_result_35973, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_35975 = invoke(stypy.reporting.localization.Localization(__file__, 85, 4), getitem___35974, int_35967)
    
    # Assigning a type to the variable 'tuple_var_assignment_35870' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tuple_var_assignment_35870', subscript_call_result_35975)
    
    # Assigning a Subscript to a Name (line 85):
    
    # Assigning a Subscript to a Name (line 85):
    
    # Obtaining the type of the subscript
    int_35976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'int')
    
    # Call to schur(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'a' (line 85)
    a_35978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'a', False)
    # Processing the call keyword arguments (line 85)
    str_35979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 27), 'str', 'real')
    keyword_35980 = str_35979
    kwargs_35981 = {'output': keyword_35980}
    # Getting the type of 'schur' (line 85)
    schur_35977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'schur', False)
    # Calling schur(args, kwargs) (line 85)
    schur_call_result_35982 = invoke(stypy.reporting.localization.Localization(__file__, 85, 11), schur_35977, *[a_35978], **kwargs_35981)
    
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___35983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), schur_call_result_35982, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_35984 = invoke(stypy.reporting.localization.Localization(__file__, 85, 4), getitem___35983, int_35976)
    
    # Assigning a type to the variable 'tuple_var_assignment_35871' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tuple_var_assignment_35871', subscript_call_result_35984)
    
    # Assigning a Name to a Name (line 85):
    
    # Assigning a Name to a Name (line 85):
    # Getting the type of 'tuple_var_assignment_35870' (line 85)
    tuple_var_assignment_35870_35985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tuple_var_assignment_35870')
    # Assigning a type to the variable 'r' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'r', tuple_var_assignment_35870_35985)
    
    # Assigning a Name to a Name (line 85):
    
    # Assigning a Name to a Name (line 85):
    # Getting the type of 'tuple_var_assignment_35871' (line 85)
    tuple_var_assignment_35871_35986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tuple_var_assignment_35871')
    # Assigning a type to the variable 'u' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 7), 'u', tuple_var_assignment_35871_35986)
    
    # Assigning a Call to a Tuple (line 88):
    
    # Assigning a Subscript to a Name (line 88):
    
    # Assigning a Subscript to a Name (line 88):
    
    # Obtaining the type of the subscript
    int_35987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 4), 'int')
    
    # Call to schur(...): (line 88)
    # Processing the call arguments (line 88)
    
    # Call to transpose(...): (line 88)
    # Processing the call keyword arguments (line 88)
    kwargs_35994 = {}
    
    # Call to conj(...): (line 88)
    # Processing the call keyword arguments (line 88)
    kwargs_35991 = {}
    # Getting the type of 'b' (line 88)
    b_35989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'b', False)
    # Obtaining the member 'conj' of a type (line 88)
    conj_35990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 17), b_35989, 'conj')
    # Calling conj(args, kwargs) (line 88)
    conj_call_result_35992 = invoke(stypy.reporting.localization.Localization(__file__, 88, 17), conj_35990, *[], **kwargs_35991)
    
    # Obtaining the member 'transpose' of a type (line 88)
    transpose_35993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 17), conj_call_result_35992, 'transpose')
    # Calling transpose(args, kwargs) (line 88)
    transpose_call_result_35995 = invoke(stypy.reporting.localization.Localization(__file__, 88, 17), transpose_35993, *[], **kwargs_35994)
    
    # Processing the call keyword arguments (line 88)
    str_35996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 46), 'str', 'real')
    keyword_35997 = str_35996
    kwargs_35998 = {'output': keyword_35997}
    # Getting the type of 'schur' (line 88)
    schur_35988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'schur', False)
    # Calling schur(args, kwargs) (line 88)
    schur_call_result_35999 = invoke(stypy.reporting.localization.Localization(__file__, 88, 11), schur_35988, *[transpose_call_result_35995], **kwargs_35998)
    
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___36000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 4), schur_call_result_35999, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_36001 = invoke(stypy.reporting.localization.Localization(__file__, 88, 4), getitem___36000, int_35987)
    
    # Assigning a type to the variable 'tuple_var_assignment_35872' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'tuple_var_assignment_35872', subscript_call_result_36001)
    
    # Assigning a Subscript to a Name (line 88):
    
    # Assigning a Subscript to a Name (line 88):
    
    # Obtaining the type of the subscript
    int_36002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 4), 'int')
    
    # Call to schur(...): (line 88)
    # Processing the call arguments (line 88)
    
    # Call to transpose(...): (line 88)
    # Processing the call keyword arguments (line 88)
    kwargs_36009 = {}
    
    # Call to conj(...): (line 88)
    # Processing the call keyword arguments (line 88)
    kwargs_36006 = {}
    # Getting the type of 'b' (line 88)
    b_36004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'b', False)
    # Obtaining the member 'conj' of a type (line 88)
    conj_36005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 17), b_36004, 'conj')
    # Calling conj(args, kwargs) (line 88)
    conj_call_result_36007 = invoke(stypy.reporting.localization.Localization(__file__, 88, 17), conj_36005, *[], **kwargs_36006)
    
    # Obtaining the member 'transpose' of a type (line 88)
    transpose_36008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 17), conj_call_result_36007, 'transpose')
    # Calling transpose(args, kwargs) (line 88)
    transpose_call_result_36010 = invoke(stypy.reporting.localization.Localization(__file__, 88, 17), transpose_36008, *[], **kwargs_36009)
    
    # Processing the call keyword arguments (line 88)
    str_36011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 46), 'str', 'real')
    keyword_36012 = str_36011
    kwargs_36013 = {'output': keyword_36012}
    # Getting the type of 'schur' (line 88)
    schur_36003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'schur', False)
    # Calling schur(args, kwargs) (line 88)
    schur_call_result_36014 = invoke(stypy.reporting.localization.Localization(__file__, 88, 11), schur_36003, *[transpose_call_result_36010], **kwargs_36013)
    
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___36015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 4), schur_call_result_36014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_36016 = invoke(stypy.reporting.localization.Localization(__file__, 88, 4), getitem___36015, int_36002)
    
    # Assigning a type to the variable 'tuple_var_assignment_35873' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'tuple_var_assignment_35873', subscript_call_result_36016)
    
    # Assigning a Name to a Name (line 88):
    
    # Assigning a Name to a Name (line 88):
    # Getting the type of 'tuple_var_assignment_35872' (line 88)
    tuple_var_assignment_35872_36017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'tuple_var_assignment_35872')
    # Assigning a type to the variable 's' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 's', tuple_var_assignment_35872_36017)
    
    # Assigning a Name to a Name (line 88):
    
    # Assigning a Name to a Name (line 88):
    # Getting the type of 'tuple_var_assignment_35873' (line 88)
    tuple_var_assignment_35873_36018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'tuple_var_assignment_35873')
    # Assigning a type to the variable 'v' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 7), 'v', tuple_var_assignment_35873_36018)
    
    # Assigning a Call to a Name (line 91):
    
    # Assigning a Call to a Name (line 91):
    
    # Assigning a Call to a Name (line 91):
    
    # Call to dot(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Call to dot(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Call to transpose(...): (line 91)
    # Processing the call keyword arguments (line 91)
    kwargs_36028 = {}
    
    # Call to conj(...): (line 91)
    # Processing the call keyword arguments (line 91)
    kwargs_36025 = {}
    # Getting the type of 'u' (line 91)
    u_36023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'u', False)
    # Obtaining the member 'conj' of a type (line 91)
    conj_36024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 22), u_36023, 'conj')
    # Calling conj(args, kwargs) (line 91)
    conj_call_result_36026 = invoke(stypy.reporting.localization.Localization(__file__, 91, 22), conj_36024, *[], **kwargs_36025)
    
    # Obtaining the member 'transpose' of a type (line 91)
    transpose_36027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 22), conj_call_result_36026, 'transpose')
    # Calling transpose(args, kwargs) (line 91)
    transpose_call_result_36029 = invoke(stypy.reporting.localization.Localization(__file__, 91, 22), transpose_36027, *[], **kwargs_36028)
    
    # Getting the type of 'q' (line 91)
    q_36030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 44), 'q', False)
    # Processing the call keyword arguments (line 91)
    kwargs_36031 = {}
    # Getting the type of 'np' (line 91)
    np_36021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'np', False)
    # Obtaining the member 'dot' of a type (line 91)
    dot_36022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 15), np_36021, 'dot')
    # Calling dot(args, kwargs) (line 91)
    dot_call_result_36032 = invoke(stypy.reporting.localization.Localization(__file__, 91, 15), dot_36022, *[transpose_call_result_36029, q_36030], **kwargs_36031)
    
    # Getting the type of 'v' (line 91)
    v_36033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 48), 'v', False)
    # Processing the call keyword arguments (line 91)
    kwargs_36034 = {}
    # Getting the type of 'np' (line 91)
    np_36019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'np', False)
    # Obtaining the member 'dot' of a type (line 91)
    dot_36020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), np_36019, 'dot')
    # Calling dot(args, kwargs) (line 91)
    dot_call_result_36035 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), dot_36020, *[dot_call_result_36032, v_36033], **kwargs_36034)
    
    # Assigning a type to the variable 'f' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'f', dot_call_result_36035)
    
    # Assigning a Call to a Tuple (line 94):
    
    # Assigning a Subscript to a Name (line 94):
    
    # Assigning a Subscript to a Name (line 94):
    
    # Obtaining the type of the subscript
    int_36036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 94)
    # Processing the call arguments (line 94)
    
    # Obtaining an instance of the builtin type 'tuple' (line 94)
    tuple_36038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 94)
    # Adding element type (line 94)
    str_36039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 31), 'str', 'trsyl')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 31), tuple_36038, str_36039)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 94)
    tuple_36040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 94)
    # Adding element type (line 94)
    # Getting the type of 'r' (line 94)
    r_36041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 43), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 43), tuple_36040, r_36041)
    # Adding element type (line 94)
    # Getting the type of 's' (line 94)
    s_36042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 46), 's', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 43), tuple_36040, s_36042)
    # Adding element type (line 94)
    # Getting the type of 'f' (line 94)
    f_36043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 49), 'f', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 43), tuple_36040, f_36043)
    
    # Processing the call keyword arguments (line 94)
    kwargs_36044 = {}
    # Getting the type of 'get_lapack_funcs' (line 94)
    get_lapack_funcs_36037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 13), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 94)
    get_lapack_funcs_call_result_36045 = invoke(stypy.reporting.localization.Localization(__file__, 94, 13), get_lapack_funcs_36037, *[tuple_36038, tuple_36040], **kwargs_36044)
    
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___36046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 4), get_lapack_funcs_call_result_36045, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_36047 = invoke(stypy.reporting.localization.Localization(__file__, 94, 4), getitem___36046, int_36036)
    
    # Assigning a type to the variable 'tuple_var_assignment_35874' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_var_assignment_35874', subscript_call_result_36047)
    
    # Assigning a Name to a Name (line 94):
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'tuple_var_assignment_35874' (line 94)
    tuple_var_assignment_35874_36048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_var_assignment_35874')
    # Assigning a type to the variable 'trsyl' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'trsyl', tuple_var_assignment_35874_36048)
    
    # Type idiom detected: calculating its left and rigth part (line 95)
    # Getting the type of 'trsyl' (line 95)
    trsyl_36049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 7), 'trsyl')
    # Getting the type of 'None' (line 95)
    None_36050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'None')
    
    (may_be_36051, more_types_in_union_36052) = may_be_none(trsyl_36049, None_36050)

    if may_be_36051:

        if more_types_in_union_36052:
            # Runtime conditional SSA (line 95)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to RuntimeError(...): (line 96)
        # Processing the call arguments (line 96)
        str_36054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 27), 'str', 'LAPACK implementation does not contain a proper Sylvester equation solver (TRSYL)')
        # Processing the call keyword arguments (line 96)
        kwargs_36055 = {}
        # Getting the type of 'RuntimeError' (line 96)
        RuntimeError_36053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 14), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 96)
        RuntimeError_call_result_36056 = invoke(stypy.reporting.localization.Localization(__file__, 96, 14), RuntimeError_36053, *[str_36054], **kwargs_36055)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 96, 8), RuntimeError_call_result_36056, 'raise parameter', BaseException)

        if more_types_in_union_36052:
            # SSA join for if statement (line 95)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 98):
    
    # Assigning a Subscript to a Name (line 98):
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_36057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'int')
    
    # Call to trsyl(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'r' (line 98)
    r_36059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'r', False)
    # Getting the type of 's' (line 98)
    s_36060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), 's', False)
    # Getting the type of 'f' (line 98)
    f_36061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'f', False)
    # Processing the call keyword arguments (line 98)
    str_36062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 42), 'str', 'C')
    keyword_36063 = str_36062
    kwargs_36064 = {'tranb': keyword_36063}
    # Getting the type of 'trsyl' (line 98)
    trsyl_36058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'trsyl', False)
    # Calling trsyl(args, kwargs) (line 98)
    trsyl_call_result_36065 = invoke(stypy.reporting.localization.Localization(__file__, 98, 21), trsyl_36058, *[r_36059, s_36060, f_36061], **kwargs_36064)
    
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___36066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), trsyl_call_result_36065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_36067 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), getitem___36066, int_36057)
    
    # Assigning a type to the variable 'tuple_var_assignment_35875' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_35875', subscript_call_result_36067)
    
    # Assigning a Subscript to a Name (line 98):
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_36068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'int')
    
    # Call to trsyl(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'r' (line 98)
    r_36070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'r', False)
    # Getting the type of 's' (line 98)
    s_36071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), 's', False)
    # Getting the type of 'f' (line 98)
    f_36072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'f', False)
    # Processing the call keyword arguments (line 98)
    str_36073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 42), 'str', 'C')
    keyword_36074 = str_36073
    kwargs_36075 = {'tranb': keyword_36074}
    # Getting the type of 'trsyl' (line 98)
    trsyl_36069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'trsyl', False)
    # Calling trsyl(args, kwargs) (line 98)
    trsyl_call_result_36076 = invoke(stypy.reporting.localization.Localization(__file__, 98, 21), trsyl_36069, *[r_36070, s_36071, f_36072], **kwargs_36075)
    
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___36077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), trsyl_call_result_36076, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_36078 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), getitem___36077, int_36068)
    
    # Assigning a type to the variable 'tuple_var_assignment_35876' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_35876', subscript_call_result_36078)
    
    # Assigning a Subscript to a Name (line 98):
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_36079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'int')
    
    # Call to trsyl(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'r' (line 98)
    r_36081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'r', False)
    # Getting the type of 's' (line 98)
    s_36082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), 's', False)
    # Getting the type of 'f' (line 98)
    f_36083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'f', False)
    # Processing the call keyword arguments (line 98)
    str_36084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 42), 'str', 'C')
    keyword_36085 = str_36084
    kwargs_36086 = {'tranb': keyword_36085}
    # Getting the type of 'trsyl' (line 98)
    trsyl_36080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'trsyl', False)
    # Calling trsyl(args, kwargs) (line 98)
    trsyl_call_result_36087 = invoke(stypy.reporting.localization.Localization(__file__, 98, 21), trsyl_36080, *[r_36081, s_36082, f_36083], **kwargs_36086)
    
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___36088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), trsyl_call_result_36087, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_36089 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), getitem___36088, int_36079)
    
    # Assigning a type to the variable 'tuple_var_assignment_35877' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_35877', subscript_call_result_36089)
    
    # Assigning a Name to a Name (line 98):
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'tuple_var_assignment_35875' (line 98)
    tuple_var_assignment_35875_36090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_35875')
    # Assigning a type to the variable 'y' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'y', tuple_var_assignment_35875_36090)
    
    # Assigning a Name to a Name (line 98):
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'tuple_var_assignment_35876' (line 98)
    tuple_var_assignment_35876_36091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_35876')
    # Assigning a type to the variable 'scale' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 7), 'scale', tuple_var_assignment_35876_36091)
    
    # Assigning a Name to a Name (line 98):
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'tuple_var_assignment_35877' (line 98)
    tuple_var_assignment_35877_36092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_35877')
    # Assigning a type to the variable 'info' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'info', tuple_var_assignment_35877_36092)
    
    # Assigning a BinOp to a Name (line 100):
    
    # Assigning a BinOp to a Name (line 100):
    
    # Assigning a BinOp to a Name (line 100):
    # Getting the type of 'scale' (line 100)
    scale_36093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'scale')
    # Getting the type of 'y' (line 100)
    y_36094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 'y')
    # Applying the binary operator '*' (line 100)
    result_mul_36095 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 8), '*', scale_36093, y_36094)
    
    # Assigning a type to the variable 'y' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'y', result_mul_36095)
    
    
    # Getting the type of 'info' (line 102)
    info_36096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 7), 'info')
    int_36097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 14), 'int')
    # Applying the binary operator '<' (line 102)
    result_lt_36098 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 7), '<', info_36096, int_36097)
    
    # Testing the type of an if condition (line 102)
    if_condition_36099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 4), result_lt_36098)
    # Assigning a type to the variable 'if_condition_36099' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'if_condition_36099', if_condition_36099)
    # SSA begins for if statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 103)
    # Processing the call arguments (line 103)
    str_36101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 26), 'str', 'Illegal value encountered in the %d term')
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_36102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    
    # Getting the type of 'info' (line 104)
    info_36103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 44), 'info', False)
    # Applying the 'usub' unary operator (line 104)
    result___neg___36104 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 43), 'usub', info_36103)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 43), tuple_36102, result___neg___36104)
    
    # Applying the binary operator '%' (line 103)
    result_mod_36105 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 26), '%', str_36101, tuple_36102)
    
    # Processing the call keyword arguments (line 103)
    kwargs_36106 = {}
    # Getting the type of 'LinAlgError' (line 103)
    LinAlgError_36100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 103)
    LinAlgError_call_result_36107 = invoke(stypy.reporting.localization.Localization(__file__, 103, 14), LinAlgError_36100, *[result_mod_36105], **kwargs_36106)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 103, 8), LinAlgError_call_result_36107, 'raise parameter', BaseException)
    # SSA join for if statement (line 102)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to dot(...): (line 106)
    # Processing the call arguments (line 106)
    
    # Call to dot(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'u' (line 106)
    u_36112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'u', False)
    # Getting the type of 'y' (line 106)
    y_36113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 28), 'y', False)
    # Processing the call keyword arguments (line 106)
    kwargs_36114 = {}
    # Getting the type of 'np' (line 106)
    np_36110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 106)
    dot_36111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 18), np_36110, 'dot')
    # Calling dot(args, kwargs) (line 106)
    dot_call_result_36115 = invoke(stypy.reporting.localization.Localization(__file__, 106, 18), dot_36111, *[u_36112, y_36113], **kwargs_36114)
    
    
    # Call to transpose(...): (line 106)
    # Processing the call keyword arguments (line 106)
    kwargs_36121 = {}
    
    # Call to conj(...): (line 106)
    # Processing the call keyword arguments (line 106)
    kwargs_36118 = {}
    # Getting the type of 'v' (line 106)
    v_36116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 32), 'v', False)
    # Obtaining the member 'conj' of a type (line 106)
    conj_36117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 32), v_36116, 'conj')
    # Calling conj(args, kwargs) (line 106)
    conj_call_result_36119 = invoke(stypy.reporting.localization.Localization(__file__, 106, 32), conj_36117, *[], **kwargs_36118)
    
    # Obtaining the member 'transpose' of a type (line 106)
    transpose_36120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 32), conj_call_result_36119, 'transpose')
    # Calling transpose(args, kwargs) (line 106)
    transpose_call_result_36122 = invoke(stypy.reporting.localization.Localization(__file__, 106, 32), transpose_36120, *[], **kwargs_36121)
    
    # Processing the call keyword arguments (line 106)
    kwargs_36123 = {}
    # Getting the type of 'np' (line 106)
    np_36108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'np', False)
    # Obtaining the member 'dot' of a type (line 106)
    dot_36109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 11), np_36108, 'dot')
    # Calling dot(args, kwargs) (line 106)
    dot_call_result_36124 = invoke(stypy.reporting.localization.Localization(__file__, 106, 11), dot_36109, *[dot_call_result_36115, transpose_call_result_36122], **kwargs_36123)
    
    # Assigning a type to the variable 'stypy_return_type' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type', dot_call_result_36124)
    
    # ################# End of 'solve_sylvester(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_sylvester' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_36125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36125)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_sylvester'
    return stypy_return_type_36125

# Assigning a type to the variable 'solve_sylvester' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'solve_sylvester', solve_sylvester)

@norecursion
def solve_continuous_lyapunov(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'solve_continuous_lyapunov'
    module_type_store = module_type_store.open_function_context('solve_continuous_lyapunov', 109, 0, False)
    
    # Passed parameters checking function
    solve_continuous_lyapunov.stypy_localization = localization
    solve_continuous_lyapunov.stypy_type_of_self = None
    solve_continuous_lyapunov.stypy_type_store = module_type_store
    solve_continuous_lyapunov.stypy_function_name = 'solve_continuous_lyapunov'
    solve_continuous_lyapunov.stypy_param_names_list = ['a', 'q']
    solve_continuous_lyapunov.stypy_varargs_param_name = None
    solve_continuous_lyapunov.stypy_kwargs_param_name = None
    solve_continuous_lyapunov.stypy_call_defaults = defaults
    solve_continuous_lyapunov.stypy_call_varargs = varargs
    solve_continuous_lyapunov.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_continuous_lyapunov', ['a', 'q'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_continuous_lyapunov', localization, ['a', 'q'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_continuous_lyapunov(...)' code ##################

    str_36126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, (-1)), 'str', '\n    Solves the continuous Lyapunov equation :math:`AX + XA^H = Q`.\n\n    Uses the Bartels-Stewart algorithm to find :math:`X`.\n\n    Parameters\n    ----------\n    a : array_like\n        A square matrix\n\n    q : array_like\n        Right-hand side square matrix\n\n    Returns\n    -------\n    x : ndarray\n        Solution to the continuous Lyapunov equation\n\n    See Also\n    --------\n    solve_discrete_lyapunov : computes the solution to the discrete-time\n        Lyapunov equation\n    solve_sylvester : computes the solution to the Sylvester equation\n\n    Notes\n    -----\n    The continuous Lyapunov equation is a special form of the Sylvester\n    equation, hence this solver relies on LAPACK routine ?TRSYL.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    Given `a` and `q` solve for `x`:\n\n    >>> from scipy import linalg\n    >>> a = np.array([[-3, -2, 0], [-1, -1, 0], [0, -5, -1]])\n    >>> b = np.array([2, 4, -1])\n    >>> q = np.eye(3)\n    >>> x = linalg.solve_continuous_lyapunov(a, q)\n    >>> x\n    array([[ -0.75  ,   0.875 ,  -3.75  ],\n           [  0.875 ,  -1.375 ,   5.3125],\n           [ -3.75  ,   5.3125, -27.0625]])\n    >>> np.allclose(a.dot(x) + x.dot(a.T), q)\n    True\n    ')
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to atleast_2d(...): (line 158)
    # Processing the call arguments (line 158)
    
    # Call to _asarray_validated(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'a' (line 158)
    a_36130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 41), 'a', False)
    # Processing the call keyword arguments (line 158)
    # Getting the type of 'True' (line 158)
    True_36131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 57), 'True', False)
    keyword_36132 = True_36131
    kwargs_36133 = {'check_finite': keyword_36132}
    # Getting the type of '_asarray_validated' (line 158)
    _asarray_validated_36129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 158)
    _asarray_validated_call_result_36134 = invoke(stypy.reporting.localization.Localization(__file__, 158, 22), _asarray_validated_36129, *[a_36130], **kwargs_36133)
    
    # Processing the call keyword arguments (line 158)
    kwargs_36135 = {}
    # Getting the type of 'np' (line 158)
    np_36127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 158)
    atleast_2d_36128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), np_36127, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 158)
    atleast_2d_call_result_36136 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), atleast_2d_36128, *[_asarray_validated_call_result_36134], **kwargs_36135)
    
    # Assigning a type to the variable 'a' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'a', atleast_2d_call_result_36136)
    
    # Assigning a Call to a Name (line 159):
    
    # Assigning a Call to a Name (line 159):
    
    # Assigning a Call to a Name (line 159):
    
    # Call to atleast_2d(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Call to _asarray_validated(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'q' (line 159)
    q_36140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 41), 'q', False)
    # Processing the call keyword arguments (line 159)
    # Getting the type of 'True' (line 159)
    True_36141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 57), 'True', False)
    keyword_36142 = True_36141
    kwargs_36143 = {'check_finite': keyword_36142}
    # Getting the type of '_asarray_validated' (line 159)
    _asarray_validated_36139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 159)
    _asarray_validated_call_result_36144 = invoke(stypy.reporting.localization.Localization(__file__, 159, 22), _asarray_validated_36139, *[q_36140], **kwargs_36143)
    
    # Processing the call keyword arguments (line 159)
    kwargs_36145 = {}
    # Getting the type of 'np' (line 159)
    np_36137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 159)
    atleast_2d_36138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), np_36137, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 159)
    atleast_2d_call_result_36146 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), atleast_2d_36138, *[_asarray_validated_call_result_36144], **kwargs_36145)
    
    # Assigning a type to the variable 'q' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'q', atleast_2d_call_result_36146)
    
    # Assigning a Name to a Name (line 161):
    
    # Assigning a Name to a Name (line 161):
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'float' (line 161)
    float_36147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), 'float')
    # Assigning a type to the variable 'r_or_c' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'r_or_c', float_36147)
    
    
    # Call to enumerate(...): (line 163)
    # Processing the call arguments (line 163)
    
    # Obtaining an instance of the builtin type 'tuple' (line 163)
    tuple_36149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 163)
    # Adding element type (line 163)
    # Getting the type of 'a' (line 163)
    a_36150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 29), tuple_36149, a_36150)
    # Adding element type (line 163)
    # Getting the type of 'q' (line 163)
    q_36151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 32), 'q', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 29), tuple_36149, q_36151)
    
    # Processing the call keyword arguments (line 163)
    kwargs_36152 = {}
    # Getting the type of 'enumerate' (line 163)
    enumerate_36148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 18), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 163)
    enumerate_call_result_36153 = invoke(stypy.reporting.localization.Localization(__file__, 163, 18), enumerate_36148, *[tuple_36149], **kwargs_36152)
    
    # Testing the type of a for loop iterable (line 163)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 163, 4), enumerate_call_result_36153)
    # Getting the type of the for loop variable (line 163)
    for_loop_var_36154 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 163, 4), enumerate_call_result_36153)
    # Assigning a type to the variable 'ind' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'ind', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 4), for_loop_var_36154))
    # Assigning a type to the variable '_' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), '_', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 4), for_loop_var_36154))
    # SSA begins for a for statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to iscomplexobj(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of '_' (line 164)
    __36157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), '_', False)
    # Processing the call keyword arguments (line 164)
    kwargs_36158 = {}
    # Getting the type of 'np' (line 164)
    np_36155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 164)
    iscomplexobj_36156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 11), np_36155, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 164)
    iscomplexobj_call_result_36159 = invoke(stypy.reporting.localization.Localization(__file__, 164, 11), iscomplexobj_36156, *[__36157], **kwargs_36158)
    
    # Testing the type of an if condition (line 164)
    if_condition_36160 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 8), iscomplexobj_call_result_36159)
    # Assigning a type to the variable 'if_condition_36160' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'if_condition_36160', if_condition_36160)
    # SSA begins for if statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 165):
    
    # Assigning a Name to a Name (line 165):
    
    # Assigning a Name to a Name (line 165):
    # Getting the type of 'complex' (line 165)
    complex_36161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 21), 'complex')
    # Assigning a type to the variable 'r_or_c' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'r_or_c', complex_36161)
    # SSA join for if statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to equal(...): (line 167)
    # Getting the type of '_' (line 167)
    __36164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), '_', False)
    # Obtaining the member 'shape' of a type (line 167)
    shape_36165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 25), __36164, 'shape')
    # Processing the call keyword arguments (line 167)
    kwargs_36166 = {}
    # Getting the type of 'np' (line 167)
    np_36162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), 'np', False)
    # Obtaining the member 'equal' of a type (line 167)
    equal_36163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 15), np_36162, 'equal')
    # Calling equal(args, kwargs) (line 167)
    equal_call_result_36167 = invoke(stypy.reporting.localization.Localization(__file__, 167, 15), equal_36163, *[shape_36165], **kwargs_36166)
    
    # Applying the 'not' unary operator (line 167)
    result_not__36168 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 11), 'not', equal_call_result_36167)
    
    # Testing the type of an if condition (line 167)
    if_condition_36169 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 8), result_not__36168)
    # Assigning a type to the variable 'if_condition_36169' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'if_condition_36169', if_condition_36169)
    # SSA begins for if statement (line 167)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 168)
    # Processing the call arguments (line 168)
    
    # Call to format(...): (line 168)
    # Processing the call arguments (line 168)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 168)
    ind_36173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 71), 'ind', False)
    str_36174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 66), 'str', 'aq')
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___36175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 66), str_36174, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 168)
    subscript_call_result_36176 = invoke(stypy.reporting.localization.Localization(__file__, 168, 66), getitem___36175, ind_36173)
    
    # Processing the call keyword arguments (line 168)
    kwargs_36177 = {}
    str_36171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 29), 'str', 'Matrix {} should be square.')
    # Obtaining the member 'format' of a type (line 168)
    format_36172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 29), str_36171, 'format')
    # Calling format(args, kwargs) (line 168)
    format_call_result_36178 = invoke(stypy.reporting.localization.Localization(__file__, 168, 29), format_36172, *[subscript_call_result_36176], **kwargs_36177)
    
    # Processing the call keyword arguments (line 168)
    kwargs_36179 = {}
    # Getting the type of 'ValueError' (line 168)
    ValueError_36170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 168)
    ValueError_call_result_36180 = invoke(stypy.reporting.localization.Localization(__file__, 168, 18), ValueError_36170, *[format_call_result_36178], **kwargs_36179)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 168, 12), ValueError_call_result_36180, 'raise parameter', BaseException)
    # SSA join for if statement (line 167)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a' (line 171)
    a_36181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 7), 'a')
    # Obtaining the member 'shape' of a type (line 171)
    shape_36182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 7), a_36181, 'shape')
    # Getting the type of 'q' (line 171)
    q_36183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'q')
    # Obtaining the member 'shape' of a type (line 171)
    shape_36184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 18), q_36183, 'shape')
    # Applying the binary operator '!=' (line 171)
    result_ne_36185 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 7), '!=', shape_36182, shape_36184)
    
    # Testing the type of an if condition (line 171)
    if_condition_36186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 4), result_ne_36185)
    # Assigning a type to the variable 'if_condition_36186' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'if_condition_36186', if_condition_36186)
    # SSA begins for if statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 172)
    # Processing the call arguments (line 172)
    str_36188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 25), 'str', 'Matrix a and q should have the same shape.')
    # Processing the call keyword arguments (line 172)
    kwargs_36189 = {}
    # Getting the type of 'ValueError' (line 172)
    ValueError_36187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 172)
    ValueError_call_result_36190 = invoke(stypy.reporting.localization.Localization(__file__, 172, 14), ValueError_36187, *[str_36188], **kwargs_36189)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 172, 8), ValueError_call_result_36190, 'raise parameter', BaseException)
    # SSA join for if statement (line 171)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 175):
    
    # Assigning a Subscript to a Name (line 175):
    
    # Assigning a Subscript to a Name (line 175):
    
    # Obtaining the type of the subscript
    int_36191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 4), 'int')
    
    # Call to schur(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'a' (line 175)
    a_36193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 17), 'a', False)
    # Processing the call keyword arguments (line 175)
    str_36194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 27), 'str', 'real')
    keyword_36195 = str_36194
    kwargs_36196 = {'output': keyword_36195}
    # Getting the type of 'schur' (line 175)
    schur_36192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 11), 'schur', False)
    # Calling schur(args, kwargs) (line 175)
    schur_call_result_36197 = invoke(stypy.reporting.localization.Localization(__file__, 175, 11), schur_36192, *[a_36193], **kwargs_36196)
    
    # Obtaining the member '__getitem__' of a type (line 175)
    getitem___36198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 4), schur_call_result_36197, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 175)
    subscript_call_result_36199 = invoke(stypy.reporting.localization.Localization(__file__, 175, 4), getitem___36198, int_36191)
    
    # Assigning a type to the variable 'tuple_var_assignment_35878' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'tuple_var_assignment_35878', subscript_call_result_36199)
    
    # Assigning a Subscript to a Name (line 175):
    
    # Assigning a Subscript to a Name (line 175):
    
    # Obtaining the type of the subscript
    int_36200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 4), 'int')
    
    # Call to schur(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'a' (line 175)
    a_36202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 17), 'a', False)
    # Processing the call keyword arguments (line 175)
    str_36203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 27), 'str', 'real')
    keyword_36204 = str_36203
    kwargs_36205 = {'output': keyword_36204}
    # Getting the type of 'schur' (line 175)
    schur_36201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 11), 'schur', False)
    # Calling schur(args, kwargs) (line 175)
    schur_call_result_36206 = invoke(stypy.reporting.localization.Localization(__file__, 175, 11), schur_36201, *[a_36202], **kwargs_36205)
    
    # Obtaining the member '__getitem__' of a type (line 175)
    getitem___36207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 4), schur_call_result_36206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 175)
    subscript_call_result_36208 = invoke(stypy.reporting.localization.Localization(__file__, 175, 4), getitem___36207, int_36200)
    
    # Assigning a type to the variable 'tuple_var_assignment_35879' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'tuple_var_assignment_35879', subscript_call_result_36208)
    
    # Assigning a Name to a Name (line 175):
    
    # Assigning a Name to a Name (line 175):
    # Getting the type of 'tuple_var_assignment_35878' (line 175)
    tuple_var_assignment_35878_36209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'tuple_var_assignment_35878')
    # Assigning a type to the variable 'r' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'r', tuple_var_assignment_35878_36209)
    
    # Assigning a Name to a Name (line 175):
    
    # Assigning a Name to a Name (line 175):
    # Getting the type of 'tuple_var_assignment_35879' (line 175)
    tuple_var_assignment_35879_36210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'tuple_var_assignment_35879')
    # Assigning a type to the variable 'u' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 7), 'u', tuple_var_assignment_35879_36210)
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to dot(...): (line 178)
    # Processing the call arguments (line 178)
    
    # Call to dot(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'u' (line 178)
    u_36219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 29), 'u', False)
    # Processing the call keyword arguments (line 178)
    kwargs_36220 = {}
    # Getting the type of 'q' (line 178)
    q_36217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'q', False)
    # Obtaining the member 'dot' of a type (line 178)
    dot_36218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 23), q_36217, 'dot')
    # Calling dot(args, kwargs) (line 178)
    dot_call_result_36221 = invoke(stypy.reporting.localization.Localization(__file__, 178, 23), dot_36218, *[u_36219], **kwargs_36220)
    
    # Processing the call keyword arguments (line 178)
    kwargs_36222 = {}
    
    # Call to conj(...): (line 178)
    # Processing the call keyword arguments (line 178)
    kwargs_36213 = {}
    # Getting the type of 'u' (line 178)
    u_36211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'u', False)
    # Obtaining the member 'conj' of a type (line 178)
    conj_36212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), u_36211, 'conj')
    # Calling conj(args, kwargs) (line 178)
    conj_call_result_36214 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), conj_36212, *[], **kwargs_36213)
    
    # Obtaining the member 'T' of a type (line 178)
    T_36215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), conj_call_result_36214, 'T')
    # Obtaining the member 'dot' of a type (line 178)
    dot_36216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), T_36215, 'dot')
    # Calling dot(args, kwargs) (line 178)
    dot_call_result_36223 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), dot_36216, *[dot_call_result_36221], **kwargs_36222)
    
    # Assigning a type to the variable 'f' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'f', dot_call_result_36223)
    
    # Assigning a Call to a Name (line 181):
    
    # Assigning a Call to a Name (line 181):
    
    # Assigning a Call to a Name (line 181):
    
    # Call to get_lapack_funcs(...): (line 181)
    # Processing the call arguments (line 181)
    str_36225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 29), 'str', 'trsyl')
    
    # Obtaining an instance of the builtin type 'tuple' (line 181)
    tuple_36226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 181)
    # Adding element type (line 181)
    # Getting the type of 'r' (line 181)
    r_36227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 39), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 39), tuple_36226, r_36227)
    # Adding element type (line 181)
    # Getting the type of 'f' (line 181)
    f_36228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 42), 'f', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 39), tuple_36226, f_36228)
    
    # Processing the call keyword arguments (line 181)
    kwargs_36229 = {}
    # Getting the type of 'get_lapack_funcs' (line 181)
    get_lapack_funcs_36224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 181)
    get_lapack_funcs_call_result_36230 = invoke(stypy.reporting.localization.Localization(__file__, 181, 12), get_lapack_funcs_36224, *[str_36225, tuple_36226], **kwargs_36229)
    
    # Assigning a type to the variable 'trsyl' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'trsyl', get_lapack_funcs_call_result_36230)
    
    # Assigning a IfExp to a Name (line 183):
    
    # Assigning a IfExp to a Name (line 183):
    
    # Assigning a IfExp to a Name (line 183):
    
    
    # Getting the type of 'r_or_c' (line 183)
    r_or_c_36231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 26), 'r_or_c')
    # Getting the type of 'float' (line 183)
    float_36232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 36), 'float')
    # Applying the binary operator '==' (line 183)
    result_eq_36233 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 26), '==', r_or_c_36231, float_36232)
    
    # Testing the type of an if expression (line 183)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 19), result_eq_36233)
    # SSA begins for if expression (line 183)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    str_36234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 19), 'str', 'T')
    # SSA branch for the else part of an if expression (line 183)
    module_type_store.open_ssa_branch('if expression else')
    str_36235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 47), 'str', 'C')
    # SSA join for if expression (line 183)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_36236 = union_type.UnionType.add(str_36234, str_36235)
    
    # Assigning a type to the variable 'dtype_string' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'dtype_string', if_exp_36236)
    
    # Assigning a Call to a Tuple (line 184):
    
    # Assigning a Subscript to a Name (line 184):
    
    # Assigning a Subscript to a Name (line 184):
    
    # Obtaining the type of the subscript
    int_36237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 4), 'int')
    
    # Call to trsyl(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'r' (line 184)
    r_36239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'r', False)
    # Getting the type of 'r' (line 184)
    r_36240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 30), 'r', False)
    # Getting the type of 'f' (line 184)
    f_36241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 33), 'f', False)
    # Processing the call keyword arguments (line 184)
    # Getting the type of 'dtype_string' (line 184)
    dtype_string_36242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 42), 'dtype_string', False)
    keyword_36243 = dtype_string_36242
    kwargs_36244 = {'tranb': keyword_36243}
    # Getting the type of 'trsyl' (line 184)
    trsyl_36238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'trsyl', False)
    # Calling trsyl(args, kwargs) (line 184)
    trsyl_call_result_36245 = invoke(stypy.reporting.localization.Localization(__file__, 184, 21), trsyl_36238, *[r_36239, r_36240, f_36241], **kwargs_36244)
    
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___36246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 4), trsyl_call_result_36245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_36247 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), getitem___36246, int_36237)
    
    # Assigning a type to the variable 'tuple_var_assignment_35880' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'tuple_var_assignment_35880', subscript_call_result_36247)
    
    # Assigning a Subscript to a Name (line 184):
    
    # Assigning a Subscript to a Name (line 184):
    
    # Obtaining the type of the subscript
    int_36248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 4), 'int')
    
    # Call to trsyl(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'r' (line 184)
    r_36250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'r', False)
    # Getting the type of 'r' (line 184)
    r_36251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 30), 'r', False)
    # Getting the type of 'f' (line 184)
    f_36252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 33), 'f', False)
    # Processing the call keyword arguments (line 184)
    # Getting the type of 'dtype_string' (line 184)
    dtype_string_36253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 42), 'dtype_string', False)
    keyword_36254 = dtype_string_36253
    kwargs_36255 = {'tranb': keyword_36254}
    # Getting the type of 'trsyl' (line 184)
    trsyl_36249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'trsyl', False)
    # Calling trsyl(args, kwargs) (line 184)
    trsyl_call_result_36256 = invoke(stypy.reporting.localization.Localization(__file__, 184, 21), trsyl_36249, *[r_36250, r_36251, f_36252], **kwargs_36255)
    
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___36257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 4), trsyl_call_result_36256, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_36258 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), getitem___36257, int_36248)
    
    # Assigning a type to the variable 'tuple_var_assignment_35881' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'tuple_var_assignment_35881', subscript_call_result_36258)
    
    # Assigning a Subscript to a Name (line 184):
    
    # Assigning a Subscript to a Name (line 184):
    
    # Obtaining the type of the subscript
    int_36259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 4), 'int')
    
    # Call to trsyl(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'r' (line 184)
    r_36261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'r', False)
    # Getting the type of 'r' (line 184)
    r_36262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 30), 'r', False)
    # Getting the type of 'f' (line 184)
    f_36263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 33), 'f', False)
    # Processing the call keyword arguments (line 184)
    # Getting the type of 'dtype_string' (line 184)
    dtype_string_36264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 42), 'dtype_string', False)
    keyword_36265 = dtype_string_36264
    kwargs_36266 = {'tranb': keyword_36265}
    # Getting the type of 'trsyl' (line 184)
    trsyl_36260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'trsyl', False)
    # Calling trsyl(args, kwargs) (line 184)
    trsyl_call_result_36267 = invoke(stypy.reporting.localization.Localization(__file__, 184, 21), trsyl_36260, *[r_36261, r_36262, f_36263], **kwargs_36266)
    
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___36268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 4), trsyl_call_result_36267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_36269 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), getitem___36268, int_36259)
    
    # Assigning a type to the variable 'tuple_var_assignment_35882' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'tuple_var_assignment_35882', subscript_call_result_36269)
    
    # Assigning a Name to a Name (line 184):
    
    # Assigning a Name to a Name (line 184):
    # Getting the type of 'tuple_var_assignment_35880' (line 184)
    tuple_var_assignment_35880_36270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'tuple_var_assignment_35880')
    # Assigning a type to the variable 'y' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'y', tuple_var_assignment_35880_36270)
    
    # Assigning a Name to a Name (line 184):
    
    # Assigning a Name to a Name (line 184):
    # Getting the type of 'tuple_var_assignment_35881' (line 184)
    tuple_var_assignment_35881_36271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'tuple_var_assignment_35881')
    # Assigning a type to the variable 'scale' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 7), 'scale', tuple_var_assignment_35881_36271)
    
    # Assigning a Name to a Name (line 184):
    
    # Assigning a Name to a Name (line 184):
    # Getting the type of 'tuple_var_assignment_35882' (line 184)
    tuple_var_assignment_35882_36272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'tuple_var_assignment_35882')
    # Assigning a type to the variable 'info' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 14), 'info', tuple_var_assignment_35882_36272)
    
    
    # Getting the type of 'info' (line 186)
    info_36273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 7), 'info')
    int_36274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 14), 'int')
    # Applying the binary operator '<' (line 186)
    result_lt_36275 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 7), '<', info_36273, int_36274)
    
    # Testing the type of an if condition (line 186)
    if_condition_36276 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 4), result_lt_36275)
    # Assigning a type to the variable 'if_condition_36276' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'if_condition_36276', if_condition_36276)
    # SSA begins for if statement (line 186)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 187)
    # Processing the call arguments (line 187)
    
    # Call to format(...): (line 187)
    # Processing the call arguments (line 187)
    
    # Getting the type of 'info' (line 190)
    info_36280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 36), 'info', False)
    # Applying the 'usub' unary operator (line 190)
    result___neg___36281 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 35), 'usub', info_36280)
    
    # Processing the call keyword arguments (line 187)
    kwargs_36282 = {}
    str_36278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 25), 'str', '?TRSYL exited with the internal error "illegal value in argument number {}.". See LAPACK documentation for the ?TRSYL error codes.')
    # Obtaining the member 'format' of a type (line 187)
    format_36279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 25), str_36278, 'format')
    # Calling format(args, kwargs) (line 187)
    format_call_result_36283 = invoke(stypy.reporting.localization.Localization(__file__, 187, 25), format_36279, *[result___neg___36281], **kwargs_36282)
    
    # Processing the call keyword arguments (line 187)
    kwargs_36284 = {}
    # Getting the type of 'ValueError' (line 187)
    ValueError_36277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 187)
    ValueError_call_result_36285 = invoke(stypy.reporting.localization.Localization(__file__, 187, 14), ValueError_36277, *[format_call_result_36283], **kwargs_36284)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 187, 8), ValueError_call_result_36285, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 186)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'info' (line 191)
    info_36286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 9), 'info')
    int_36287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 17), 'int')
    # Applying the binary operator '==' (line 191)
    result_eq_36288 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 9), '==', info_36286, int_36287)
    
    # Testing the type of an if condition (line 191)
    if_condition_36289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 9), result_eq_36288)
    # Assigning a type to the variable 'if_condition_36289' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 9), 'if_condition_36289', if_condition_36289)
    # SSA begins for if statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 192)
    # Processing the call arguments (line 192)
    str_36292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 22), 'str', 'Input "a" has an eigenvalue pair whose sum is very close to or exactly zero. The solution is obtained via perturbing the coefficients.')
    # Getting the type of 'RuntimeWarning' (line 195)
    RuntimeWarning_36293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 192)
    kwargs_36294 = {}
    # Getting the type of 'warnings' (line 192)
    warnings_36290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 192)
    warn_36291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), warnings_36290, 'warn')
    # Calling warn(args, kwargs) (line 192)
    warn_call_result_36295 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), warn_36291, *[str_36292, RuntimeWarning_36293], **kwargs_36294)
    
    # SSA join for if statement (line 191)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 186)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'y' (line 196)
    y_36296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'y')
    # Getting the type of 'scale' (line 196)
    scale_36297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 9), 'scale')
    # Applying the binary operator '*=' (line 196)
    result_imul_36298 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 4), '*=', y_36296, scale_36297)
    # Assigning a type to the variable 'y' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'y', result_imul_36298)
    
    
    # Call to dot(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Call to conj(...): (line 198)
    # Processing the call keyword arguments (line 198)
    kwargs_36307 = {}
    # Getting the type of 'u' (line 198)
    u_36305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 24), 'u', False)
    # Obtaining the member 'conj' of a type (line 198)
    conj_36306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 24), u_36305, 'conj')
    # Calling conj(args, kwargs) (line 198)
    conj_call_result_36308 = invoke(stypy.reporting.localization.Localization(__file__, 198, 24), conj_36306, *[], **kwargs_36307)
    
    # Obtaining the member 'T' of a type (line 198)
    T_36309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 24), conj_call_result_36308, 'T')
    # Processing the call keyword arguments (line 198)
    kwargs_36310 = {}
    
    # Call to dot(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'y' (line 198)
    y_36301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'y', False)
    # Processing the call keyword arguments (line 198)
    kwargs_36302 = {}
    # Getting the type of 'u' (line 198)
    u_36299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'u', False)
    # Obtaining the member 'dot' of a type (line 198)
    dot_36300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 11), u_36299, 'dot')
    # Calling dot(args, kwargs) (line 198)
    dot_call_result_36303 = invoke(stypy.reporting.localization.Localization(__file__, 198, 11), dot_36300, *[y_36301], **kwargs_36302)
    
    # Obtaining the member 'dot' of a type (line 198)
    dot_36304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 11), dot_call_result_36303, 'dot')
    # Calling dot(args, kwargs) (line 198)
    dot_call_result_36311 = invoke(stypy.reporting.localization.Localization(__file__, 198, 11), dot_36304, *[T_36309], **kwargs_36310)
    
    # Assigning a type to the variable 'stypy_return_type' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type', dot_call_result_36311)
    
    # ################# End of 'solve_continuous_lyapunov(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_continuous_lyapunov' in the type store
    # Getting the type of 'stypy_return_type' (line 109)
    stypy_return_type_36312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36312)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_continuous_lyapunov'
    return stypy_return_type_36312

# Assigning a type to the variable 'solve_continuous_lyapunov' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'solve_continuous_lyapunov', solve_continuous_lyapunov)

# Assigning a Name to a Name (line 201):

# Assigning a Name to a Name (line 201):

# Assigning a Name to a Name (line 201):
# Getting the type of 'solve_continuous_lyapunov' (line 201)
solve_continuous_lyapunov_36313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 17), 'solve_continuous_lyapunov')
# Assigning a type to the variable 'solve_lyapunov' (line 201)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'solve_lyapunov', solve_continuous_lyapunov_36313)

@norecursion
def _solve_discrete_lyapunov_direct(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_solve_discrete_lyapunov_direct'
    module_type_store = module_type_store.open_function_context('_solve_discrete_lyapunov_direct', 204, 0, False)
    
    # Passed parameters checking function
    _solve_discrete_lyapunov_direct.stypy_localization = localization
    _solve_discrete_lyapunov_direct.stypy_type_of_self = None
    _solve_discrete_lyapunov_direct.stypy_type_store = module_type_store
    _solve_discrete_lyapunov_direct.stypy_function_name = '_solve_discrete_lyapunov_direct'
    _solve_discrete_lyapunov_direct.stypy_param_names_list = ['a', 'q']
    _solve_discrete_lyapunov_direct.stypy_varargs_param_name = None
    _solve_discrete_lyapunov_direct.stypy_kwargs_param_name = None
    _solve_discrete_lyapunov_direct.stypy_call_defaults = defaults
    _solve_discrete_lyapunov_direct.stypy_call_varargs = varargs
    _solve_discrete_lyapunov_direct.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_solve_discrete_lyapunov_direct', ['a', 'q'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_solve_discrete_lyapunov_direct', localization, ['a', 'q'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_solve_discrete_lyapunov_direct(...)' code ##################

    str_36314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, (-1)), 'str', '\n    Solves the discrete Lyapunov equation directly.\n\n    This function is called by the `solve_discrete_lyapunov` function with\n    `method=direct`. It is not supposed to be called directly.\n    ')
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Call to kron(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'a' (line 212)
    a_36316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'a', False)
    
    # Call to conj(...): (line 212)
    # Processing the call keyword arguments (line 212)
    kwargs_36319 = {}
    # Getting the type of 'a' (line 212)
    a_36317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 18), 'a', False)
    # Obtaining the member 'conj' of a type (line 212)
    conj_36318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 18), a_36317, 'conj')
    # Calling conj(args, kwargs) (line 212)
    conj_call_result_36320 = invoke(stypy.reporting.localization.Localization(__file__, 212, 18), conj_36318, *[], **kwargs_36319)
    
    # Processing the call keyword arguments (line 212)
    kwargs_36321 = {}
    # Getting the type of 'kron' (line 212)
    kron_36315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 10), 'kron', False)
    # Calling kron(args, kwargs) (line 212)
    kron_call_result_36322 = invoke(stypy.reporting.localization.Localization(__file__, 212, 10), kron_36315, *[a_36316, conj_call_result_36320], **kwargs_36321)
    
    # Assigning a type to the variable 'lhs' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'lhs', kron_call_result_36322)
    
    # Assigning a BinOp to a Name (line 213):
    
    # Assigning a BinOp to a Name (line 213):
    
    # Assigning a BinOp to a Name (line 213):
    
    # Call to eye(...): (line 213)
    # Processing the call arguments (line 213)
    
    # Obtaining the type of the subscript
    int_36325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 27), 'int')
    # Getting the type of 'lhs' (line 213)
    lhs_36326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 17), 'lhs', False)
    # Obtaining the member 'shape' of a type (line 213)
    shape_36327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 17), lhs_36326, 'shape')
    # Obtaining the member '__getitem__' of a type (line 213)
    getitem___36328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 17), shape_36327, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 213)
    subscript_call_result_36329 = invoke(stypy.reporting.localization.Localization(__file__, 213, 17), getitem___36328, int_36325)
    
    # Processing the call keyword arguments (line 213)
    kwargs_36330 = {}
    # Getting the type of 'np' (line 213)
    np_36323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 10), 'np', False)
    # Obtaining the member 'eye' of a type (line 213)
    eye_36324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 10), np_36323, 'eye')
    # Calling eye(args, kwargs) (line 213)
    eye_call_result_36331 = invoke(stypy.reporting.localization.Localization(__file__, 213, 10), eye_36324, *[subscript_call_result_36329], **kwargs_36330)
    
    # Getting the type of 'lhs' (line 213)
    lhs_36332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 33), 'lhs')
    # Applying the binary operator '-' (line 213)
    result_sub_36333 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 10), '-', eye_call_result_36331, lhs_36332)
    
    # Assigning a type to the variable 'lhs' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'lhs', result_sub_36333)
    
    # Assigning a Call to a Name (line 214):
    
    # Assigning a Call to a Name (line 214):
    
    # Assigning a Call to a Name (line 214):
    
    # Call to solve(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'lhs' (line 214)
    lhs_36335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 14), 'lhs', False)
    
    # Call to flatten(...): (line 214)
    # Processing the call keyword arguments (line 214)
    kwargs_36338 = {}
    # Getting the type of 'q' (line 214)
    q_36336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'q', False)
    # Obtaining the member 'flatten' of a type (line 214)
    flatten_36337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 19), q_36336, 'flatten')
    # Calling flatten(args, kwargs) (line 214)
    flatten_call_result_36339 = invoke(stypy.reporting.localization.Localization(__file__, 214, 19), flatten_36337, *[], **kwargs_36338)
    
    # Processing the call keyword arguments (line 214)
    kwargs_36340 = {}
    # Getting the type of 'solve' (line 214)
    solve_36334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'solve', False)
    # Calling solve(args, kwargs) (line 214)
    solve_call_result_36341 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), solve_36334, *[lhs_36335, flatten_call_result_36339], **kwargs_36340)
    
    # Assigning a type to the variable 'x' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'x', solve_call_result_36341)
    
    # Call to reshape(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'x' (line 216)
    x_36344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 22), 'x', False)
    # Getting the type of 'q' (line 216)
    q_36345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 25), 'q', False)
    # Obtaining the member 'shape' of a type (line 216)
    shape_36346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 25), q_36345, 'shape')
    # Processing the call keyword arguments (line 216)
    kwargs_36347 = {}
    # Getting the type of 'np' (line 216)
    np_36342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 11), 'np', False)
    # Obtaining the member 'reshape' of a type (line 216)
    reshape_36343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 11), np_36342, 'reshape')
    # Calling reshape(args, kwargs) (line 216)
    reshape_call_result_36348 = invoke(stypy.reporting.localization.Localization(__file__, 216, 11), reshape_36343, *[x_36344, shape_36346], **kwargs_36347)
    
    # Assigning a type to the variable 'stypy_return_type' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type', reshape_call_result_36348)
    
    # ################# End of '_solve_discrete_lyapunov_direct(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_solve_discrete_lyapunov_direct' in the type store
    # Getting the type of 'stypy_return_type' (line 204)
    stypy_return_type_36349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36349)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_solve_discrete_lyapunov_direct'
    return stypy_return_type_36349

# Assigning a type to the variable '_solve_discrete_lyapunov_direct' (line 204)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 0), '_solve_discrete_lyapunov_direct', _solve_discrete_lyapunov_direct)

@norecursion
def _solve_discrete_lyapunov_bilinear(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_solve_discrete_lyapunov_bilinear'
    module_type_store = module_type_store.open_function_context('_solve_discrete_lyapunov_bilinear', 219, 0, False)
    
    # Passed parameters checking function
    _solve_discrete_lyapunov_bilinear.stypy_localization = localization
    _solve_discrete_lyapunov_bilinear.stypy_type_of_self = None
    _solve_discrete_lyapunov_bilinear.stypy_type_store = module_type_store
    _solve_discrete_lyapunov_bilinear.stypy_function_name = '_solve_discrete_lyapunov_bilinear'
    _solve_discrete_lyapunov_bilinear.stypy_param_names_list = ['a', 'q']
    _solve_discrete_lyapunov_bilinear.stypy_varargs_param_name = None
    _solve_discrete_lyapunov_bilinear.stypy_kwargs_param_name = None
    _solve_discrete_lyapunov_bilinear.stypy_call_defaults = defaults
    _solve_discrete_lyapunov_bilinear.stypy_call_varargs = varargs
    _solve_discrete_lyapunov_bilinear.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_solve_discrete_lyapunov_bilinear', ['a', 'q'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_solve_discrete_lyapunov_bilinear', localization, ['a', 'q'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_solve_discrete_lyapunov_bilinear(...)' code ##################

    str_36350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, (-1)), 'str', '\n    Solves the discrete Lyapunov equation using a bilinear transformation.\n\n    This function is called by the `solve_discrete_lyapunov` function with\n    `method=bilinear`. It is not supposed to be called directly.\n    ')
    
    # Assigning a Call to a Name (line 226):
    
    # Assigning a Call to a Name (line 226):
    
    # Assigning a Call to a Name (line 226):
    
    # Call to eye(...): (line 226)
    # Processing the call arguments (line 226)
    
    # Obtaining the type of the subscript
    int_36353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 25), 'int')
    # Getting the type of 'a' (line 226)
    a_36354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), 'a', False)
    # Obtaining the member 'shape' of a type (line 226)
    shape_36355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 17), a_36354, 'shape')
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___36356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 17), shape_36355, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_36357 = invoke(stypy.reporting.localization.Localization(__file__, 226, 17), getitem___36356, int_36353)
    
    # Processing the call keyword arguments (line 226)
    kwargs_36358 = {}
    # Getting the type of 'np' (line 226)
    np_36351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 10), 'np', False)
    # Obtaining the member 'eye' of a type (line 226)
    eye_36352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 10), np_36351, 'eye')
    # Calling eye(args, kwargs) (line 226)
    eye_call_result_36359 = invoke(stypy.reporting.localization.Localization(__file__, 226, 10), eye_36352, *[subscript_call_result_36357], **kwargs_36358)
    
    # Assigning a type to the variable 'eye' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'eye', eye_call_result_36359)
    
    # Assigning a Call to a Name (line 227):
    
    # Assigning a Call to a Name (line 227):
    
    # Assigning a Call to a Name (line 227):
    
    # Call to transpose(...): (line 227)
    # Processing the call keyword arguments (line 227)
    kwargs_36365 = {}
    
    # Call to conj(...): (line 227)
    # Processing the call keyword arguments (line 227)
    kwargs_36362 = {}
    # Getting the type of 'a' (line 227)
    a_36360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 9), 'a', False)
    # Obtaining the member 'conj' of a type (line 227)
    conj_36361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 9), a_36360, 'conj')
    # Calling conj(args, kwargs) (line 227)
    conj_call_result_36363 = invoke(stypy.reporting.localization.Localization(__file__, 227, 9), conj_36361, *[], **kwargs_36362)
    
    # Obtaining the member 'transpose' of a type (line 227)
    transpose_36364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 9), conj_call_result_36363, 'transpose')
    # Calling transpose(args, kwargs) (line 227)
    transpose_call_result_36366 = invoke(stypy.reporting.localization.Localization(__file__, 227, 9), transpose_36364, *[], **kwargs_36365)
    
    # Assigning a type to the variable 'aH' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'aH', transpose_call_result_36366)
    
    # Assigning a Call to a Name (line 228):
    
    # Assigning a Call to a Name (line 228):
    
    # Assigning a Call to a Name (line 228):
    
    # Call to inv(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'aH' (line 228)
    aH_36368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'aH', False)
    # Getting the type of 'eye' (line 228)
    eye_36369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 23), 'eye', False)
    # Applying the binary operator '+' (line 228)
    result_add_36370 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 18), '+', aH_36368, eye_36369)
    
    # Processing the call keyword arguments (line 228)
    kwargs_36371 = {}
    # Getting the type of 'inv' (line 228)
    inv_36367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 14), 'inv', False)
    # Calling inv(args, kwargs) (line 228)
    inv_call_result_36372 = invoke(stypy.reporting.localization.Localization(__file__, 228, 14), inv_36367, *[result_add_36370], **kwargs_36371)
    
    # Assigning a type to the variable 'aHI_inv' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'aHI_inv', inv_call_result_36372)
    
    # Assigning a Call to a Name (line 229):
    
    # Assigning a Call to a Name (line 229):
    
    # Assigning a Call to a Name (line 229):
    
    # Call to dot(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'aH' (line 229)
    aH_36375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'aH', False)
    # Getting the type of 'eye' (line 229)
    eye_36376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'eye', False)
    # Applying the binary operator '-' (line 229)
    result_sub_36377 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 15), '-', aH_36375, eye_36376)
    
    # Getting the type of 'aHI_inv' (line 229)
    aHI_inv_36378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 25), 'aHI_inv', False)
    # Processing the call keyword arguments (line 229)
    kwargs_36379 = {}
    # Getting the type of 'np' (line 229)
    np_36373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'np', False)
    # Obtaining the member 'dot' of a type (line 229)
    dot_36374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), np_36373, 'dot')
    # Calling dot(args, kwargs) (line 229)
    dot_call_result_36380 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), dot_36374, *[result_sub_36377, aHI_inv_36378], **kwargs_36379)
    
    # Assigning a type to the variable 'b' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'b', dot_call_result_36380)
    
    # Assigning a BinOp to a Name (line 230):
    
    # Assigning a BinOp to a Name (line 230):
    
    # Assigning a BinOp to a Name (line 230):
    int_36381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 8), 'int')
    
    # Call to dot(...): (line 230)
    # Processing the call arguments (line 230)
    
    # Call to dot(...): (line 230)
    # Processing the call arguments (line 230)
    
    # Call to inv(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'a' (line 230)
    a_36387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 28), 'a', False)
    # Getting the type of 'eye' (line 230)
    eye_36388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 32), 'eye', False)
    # Applying the binary operator '+' (line 230)
    result_add_36389 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 28), '+', a_36387, eye_36388)
    
    # Processing the call keyword arguments (line 230)
    kwargs_36390 = {}
    # Getting the type of 'inv' (line 230)
    inv_36386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 24), 'inv', False)
    # Calling inv(args, kwargs) (line 230)
    inv_call_result_36391 = invoke(stypy.reporting.localization.Localization(__file__, 230, 24), inv_36386, *[result_add_36389], **kwargs_36390)
    
    # Getting the type of 'q' (line 230)
    q_36392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 38), 'q', False)
    # Processing the call keyword arguments (line 230)
    kwargs_36393 = {}
    # Getting the type of 'np' (line 230)
    np_36384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 17), 'np', False)
    # Obtaining the member 'dot' of a type (line 230)
    dot_36385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 17), np_36384, 'dot')
    # Calling dot(args, kwargs) (line 230)
    dot_call_result_36394 = invoke(stypy.reporting.localization.Localization(__file__, 230, 17), dot_36385, *[inv_call_result_36391, q_36392], **kwargs_36393)
    
    # Getting the type of 'aHI_inv' (line 230)
    aHI_inv_36395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 42), 'aHI_inv', False)
    # Processing the call keyword arguments (line 230)
    kwargs_36396 = {}
    # Getting the type of 'np' (line 230)
    np_36382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 10), 'np', False)
    # Obtaining the member 'dot' of a type (line 230)
    dot_36383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 10), np_36382, 'dot')
    # Calling dot(args, kwargs) (line 230)
    dot_call_result_36397 = invoke(stypy.reporting.localization.Localization(__file__, 230, 10), dot_36383, *[dot_call_result_36394, aHI_inv_36395], **kwargs_36396)
    
    # Applying the binary operator '*' (line 230)
    result_mul_36398 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 8), '*', int_36381, dot_call_result_36397)
    
    # Assigning a type to the variable 'c' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'c', result_mul_36398)
    
    # Call to solve_lyapunov(...): (line 231)
    # Processing the call arguments (line 231)
    
    # Call to transpose(...): (line 231)
    # Processing the call keyword arguments (line 231)
    kwargs_36405 = {}
    
    # Call to conj(...): (line 231)
    # Processing the call keyword arguments (line 231)
    kwargs_36402 = {}
    # Getting the type of 'b' (line 231)
    b_36400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 26), 'b', False)
    # Obtaining the member 'conj' of a type (line 231)
    conj_36401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 26), b_36400, 'conj')
    # Calling conj(args, kwargs) (line 231)
    conj_call_result_36403 = invoke(stypy.reporting.localization.Localization(__file__, 231, 26), conj_36401, *[], **kwargs_36402)
    
    # Obtaining the member 'transpose' of a type (line 231)
    transpose_36404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 26), conj_call_result_36403, 'transpose')
    # Calling transpose(args, kwargs) (line 231)
    transpose_call_result_36406 = invoke(stypy.reporting.localization.Localization(__file__, 231, 26), transpose_36404, *[], **kwargs_36405)
    
    
    # Getting the type of 'c' (line 231)
    c_36407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 49), 'c', False)
    # Applying the 'usub' unary operator (line 231)
    result___neg___36408 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 48), 'usub', c_36407)
    
    # Processing the call keyword arguments (line 231)
    kwargs_36409 = {}
    # Getting the type of 'solve_lyapunov' (line 231)
    solve_lyapunov_36399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 11), 'solve_lyapunov', False)
    # Calling solve_lyapunov(args, kwargs) (line 231)
    solve_lyapunov_call_result_36410 = invoke(stypy.reporting.localization.Localization(__file__, 231, 11), solve_lyapunov_36399, *[transpose_call_result_36406, result___neg___36408], **kwargs_36409)
    
    # Assigning a type to the variable 'stypy_return_type' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type', solve_lyapunov_call_result_36410)
    
    # ################# End of '_solve_discrete_lyapunov_bilinear(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_solve_discrete_lyapunov_bilinear' in the type store
    # Getting the type of 'stypy_return_type' (line 219)
    stypy_return_type_36411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36411)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_solve_discrete_lyapunov_bilinear'
    return stypy_return_type_36411

# Assigning a type to the variable '_solve_discrete_lyapunov_bilinear' (line 219)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), '_solve_discrete_lyapunov_bilinear', _solve_discrete_lyapunov_bilinear)

@norecursion
def solve_discrete_lyapunov(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 234)
    None_36412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 41), 'None')
    defaults = [None_36412]
    # Create a new context for function 'solve_discrete_lyapunov'
    module_type_store = module_type_store.open_function_context('solve_discrete_lyapunov', 234, 0, False)
    
    # Passed parameters checking function
    solve_discrete_lyapunov.stypy_localization = localization
    solve_discrete_lyapunov.stypy_type_of_self = None
    solve_discrete_lyapunov.stypy_type_store = module_type_store
    solve_discrete_lyapunov.stypy_function_name = 'solve_discrete_lyapunov'
    solve_discrete_lyapunov.stypy_param_names_list = ['a', 'q', 'method']
    solve_discrete_lyapunov.stypy_varargs_param_name = None
    solve_discrete_lyapunov.stypy_kwargs_param_name = None
    solve_discrete_lyapunov.stypy_call_defaults = defaults
    solve_discrete_lyapunov.stypy_call_varargs = varargs
    solve_discrete_lyapunov.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_discrete_lyapunov', ['a', 'q', 'method'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_discrete_lyapunov', localization, ['a', 'q', 'method'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_discrete_lyapunov(...)' code ##################

    str_36413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, (-1)), 'str', "\n    Solves the discrete Lyapunov equation :math:`AXA^H - X + Q = 0`.\n\n    Parameters\n    ----------\n    a, q : (M, M) array_like\n        Square matrices corresponding to A and Q in the equation\n        above respectively. Must have the same shape.\n\n    method : {'direct', 'bilinear'}, optional\n        Type of solver.\n\n        If not given, chosen to be ``direct`` if ``M`` is less than 10 and\n        ``bilinear`` otherwise.\n\n    Returns\n    -------\n    x : ndarray\n        Solution to the discrete Lyapunov equation\n\n    See Also\n    --------\n    solve_continuous_lyapunov : computes the solution to the continuous-time\n        Lyapunov equation\n\n    Notes\n    -----\n    This section describes the available solvers that can be selected by the\n    'method' parameter. The default method is *direct* if ``M`` is less than 10\n    and ``bilinear`` otherwise.\n\n    Method *direct* uses a direct analytical solution to the discrete Lyapunov\n    equation. The algorithm is given in, for example, [1]_. However it requires\n    the linear solution of a system with dimension :math:`M^2` so that\n    performance degrades rapidly for even moderately sized matrices.\n\n    Method *bilinear* uses a bilinear transformation to convert the discrete\n    Lyapunov equation to a continuous Lyapunov equation :math:`(BX+XB'=-C)`\n    where :math:`B=(A-I)(A+I)^{-1}` and\n    :math:`C=2(A' + I)^{-1} Q (A + I)^{-1}`. The continuous equation can be\n    efficiently solved since it is a special case of a Sylvester equation.\n    The transformation algorithm is from Popov (1964) as described in [2]_.\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1] Hamilton, James D. Time Series Analysis, Princeton: Princeton\n       University Press, 1994.  265.  Print.\n       http://doc1.lbfl.li/aca/FLMF037168.pdf\n    .. [2] Gajic, Z., and M.T.J. Qureshi. 2008.\n       Lyapunov Matrix Equation in System Stability and Control.\n       Dover Books on Engineering Series. Dover Publications.\n\n    Examples\n    --------\n    Given `a` and `q` solve for `x`:\n\n    >>> from scipy import linalg\n    >>> a = np.array([[0.2, 0.5],[0.7, -0.9]])\n    >>> q = np.eye(2)\n    >>> x = linalg.solve_discrete_lyapunov(a, q)\n    >>> x\n    array([[ 0.70872893,  1.43518822],\n           [ 1.43518822, -2.4266315 ]])\n    >>> np.allclose(a.dot(x).dot(a.T)-x, -q)\n    True\n\n    ")
    
    # Assigning a Call to a Name (line 304):
    
    # Assigning a Call to a Name (line 304):
    
    # Assigning a Call to a Name (line 304):
    
    # Call to asarray(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'a' (line 304)
    a_36416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 19), 'a', False)
    # Processing the call keyword arguments (line 304)
    kwargs_36417 = {}
    # Getting the type of 'np' (line 304)
    np_36414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 304)
    asarray_36415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), np_36414, 'asarray')
    # Calling asarray(args, kwargs) (line 304)
    asarray_call_result_36418 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), asarray_36415, *[a_36416], **kwargs_36417)
    
    # Assigning a type to the variable 'a' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'a', asarray_call_result_36418)
    
    # Assigning a Call to a Name (line 305):
    
    # Assigning a Call to a Name (line 305):
    
    # Assigning a Call to a Name (line 305):
    
    # Call to asarray(...): (line 305)
    # Processing the call arguments (line 305)
    # Getting the type of 'q' (line 305)
    q_36421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 19), 'q', False)
    # Processing the call keyword arguments (line 305)
    kwargs_36422 = {}
    # Getting the type of 'np' (line 305)
    np_36419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 305)
    asarray_36420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 8), np_36419, 'asarray')
    # Calling asarray(args, kwargs) (line 305)
    asarray_call_result_36423 = invoke(stypy.reporting.localization.Localization(__file__, 305, 8), asarray_36420, *[q_36421], **kwargs_36422)
    
    # Assigning a type to the variable 'q' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'q', asarray_call_result_36423)
    
    # Type idiom detected: calculating its left and rigth part (line 306)
    # Getting the type of 'method' (line 306)
    method_36424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 7), 'method')
    # Getting the type of 'None' (line 306)
    None_36425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 17), 'None')
    
    (may_be_36426, more_types_in_union_36427) = may_be_none(method_36424, None_36425)

    if may_be_36426:

        if more_types_in_union_36427:
            # Runtime conditional SSA (line 306)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Obtaining the type of the subscript
        int_36428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 19), 'int')
        # Getting the type of 'a' (line 308)
        a_36429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 11), 'a')
        # Obtaining the member 'shape' of a type (line 308)
        shape_36430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 11), a_36429, 'shape')
        # Obtaining the member '__getitem__' of a type (line 308)
        getitem___36431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 11), shape_36430, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 308)
        subscript_call_result_36432 = invoke(stypy.reporting.localization.Localization(__file__, 308, 11), getitem___36431, int_36428)
        
        int_36433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 25), 'int')
        # Applying the binary operator '>=' (line 308)
        result_ge_36434 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 11), '>=', subscript_call_result_36432, int_36433)
        
        # Testing the type of an if condition (line 308)
        if_condition_36435 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 8), result_ge_36434)
        # Assigning a type to the variable 'if_condition_36435' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'if_condition_36435', if_condition_36435)
        # SSA begins for if statement (line 308)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 309):
        
        # Assigning a Str to a Name (line 309):
        
        # Assigning a Str to a Name (line 309):
        str_36436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 21), 'str', 'bilinear')
        # Assigning a type to the variable 'method' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'method', str_36436)
        # SSA branch for the else part of an if statement (line 308)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 311):
        
        # Assigning a Str to a Name (line 311):
        
        # Assigning a Str to a Name (line 311):
        str_36437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 21), 'str', 'direct')
        # Assigning a type to the variable 'method' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'method', str_36437)
        # SSA join for if statement (line 308)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_36427:
            # SSA join for if statement (line 306)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 313):
    
    # Assigning a Call to a Name (line 313):
    
    # Assigning a Call to a Name (line 313):
    
    # Call to lower(...): (line 313)
    # Processing the call keyword arguments (line 313)
    kwargs_36440 = {}
    # Getting the type of 'method' (line 313)
    method_36438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 11), 'method', False)
    # Obtaining the member 'lower' of a type (line 313)
    lower_36439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 11), method_36438, 'lower')
    # Calling lower(args, kwargs) (line 313)
    lower_call_result_36441 = invoke(stypy.reporting.localization.Localization(__file__, 313, 11), lower_36439, *[], **kwargs_36440)
    
    # Assigning a type to the variable 'meth' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'meth', lower_call_result_36441)
    
    
    # Getting the type of 'meth' (line 315)
    meth_36442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 7), 'meth')
    str_36443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 15), 'str', 'direct')
    # Applying the binary operator '==' (line 315)
    result_eq_36444 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 7), '==', meth_36442, str_36443)
    
    # Testing the type of an if condition (line 315)
    if_condition_36445 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 315, 4), result_eq_36444)
    # Assigning a type to the variable 'if_condition_36445' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'if_condition_36445', if_condition_36445)
    # SSA begins for if statement (line 315)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 316):
    
    # Assigning a Call to a Name (line 316):
    
    # Assigning a Call to a Name (line 316):
    
    # Call to _solve_discrete_lyapunov_direct(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'a' (line 316)
    a_36447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 44), 'a', False)
    # Getting the type of 'q' (line 316)
    q_36448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 47), 'q', False)
    # Processing the call keyword arguments (line 316)
    kwargs_36449 = {}
    # Getting the type of '_solve_discrete_lyapunov_direct' (line 316)
    _solve_discrete_lyapunov_direct_36446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), '_solve_discrete_lyapunov_direct', False)
    # Calling _solve_discrete_lyapunov_direct(args, kwargs) (line 316)
    _solve_discrete_lyapunov_direct_call_result_36450 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), _solve_discrete_lyapunov_direct_36446, *[a_36447, q_36448], **kwargs_36449)
    
    # Assigning a type to the variable 'x' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'x', _solve_discrete_lyapunov_direct_call_result_36450)
    # SSA branch for the else part of an if statement (line 315)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 317)
    meth_36451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 9), 'meth')
    str_36452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 17), 'str', 'bilinear')
    # Applying the binary operator '==' (line 317)
    result_eq_36453 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 9), '==', meth_36451, str_36452)
    
    # Testing the type of an if condition (line 317)
    if_condition_36454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 9), result_eq_36453)
    # Assigning a type to the variable 'if_condition_36454' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 9), 'if_condition_36454', if_condition_36454)
    # SSA begins for if statement (line 317)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 318):
    
    # Assigning a Call to a Name (line 318):
    
    # Assigning a Call to a Name (line 318):
    
    # Call to _solve_discrete_lyapunov_bilinear(...): (line 318)
    # Processing the call arguments (line 318)
    # Getting the type of 'a' (line 318)
    a_36456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 46), 'a', False)
    # Getting the type of 'q' (line 318)
    q_36457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 49), 'q', False)
    # Processing the call keyword arguments (line 318)
    kwargs_36458 = {}
    # Getting the type of '_solve_discrete_lyapunov_bilinear' (line 318)
    _solve_discrete_lyapunov_bilinear_36455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), '_solve_discrete_lyapunov_bilinear', False)
    # Calling _solve_discrete_lyapunov_bilinear(args, kwargs) (line 318)
    _solve_discrete_lyapunov_bilinear_call_result_36459 = invoke(stypy.reporting.localization.Localization(__file__, 318, 12), _solve_discrete_lyapunov_bilinear_36455, *[a_36456, q_36457], **kwargs_36458)
    
    # Assigning a type to the variable 'x' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'x', _solve_discrete_lyapunov_bilinear_call_result_36459)
    # SSA branch for the else part of an if statement (line 317)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 320)
    # Processing the call arguments (line 320)
    str_36461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 25), 'str', 'Unknown solver %s')
    # Getting the type of 'method' (line 320)
    method_36462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 47), 'method', False)
    # Applying the binary operator '%' (line 320)
    result_mod_36463 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 25), '%', str_36461, method_36462)
    
    # Processing the call keyword arguments (line 320)
    kwargs_36464 = {}
    # Getting the type of 'ValueError' (line 320)
    ValueError_36460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 320)
    ValueError_call_result_36465 = invoke(stypy.reporting.localization.Localization(__file__, 320, 14), ValueError_36460, *[result_mod_36463], **kwargs_36464)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 320, 8), ValueError_call_result_36465, 'raise parameter', BaseException)
    # SSA join for if statement (line 317)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 315)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 322)
    x_36466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'stypy_return_type', x_36466)
    
    # ################# End of 'solve_discrete_lyapunov(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_discrete_lyapunov' in the type store
    # Getting the type of 'stypy_return_type' (line 234)
    stypy_return_type_36467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36467)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_discrete_lyapunov'
    return stypy_return_type_36467

# Assigning a type to the variable 'solve_discrete_lyapunov' (line 234)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'solve_discrete_lyapunov', solve_discrete_lyapunov)

@norecursion
def solve_continuous_are(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 325)
    None_36468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 39), 'None')
    # Getting the type of 'None' (line 325)
    None_36469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 47), 'None')
    # Getting the type of 'True' (line 325)
    True_36470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 62), 'True')
    defaults = [None_36468, None_36469, True_36470]
    # Create a new context for function 'solve_continuous_are'
    module_type_store = module_type_store.open_function_context('solve_continuous_are', 325, 0, False)
    
    # Passed parameters checking function
    solve_continuous_are.stypy_localization = localization
    solve_continuous_are.stypy_type_of_self = None
    solve_continuous_are.stypy_type_store = module_type_store
    solve_continuous_are.stypy_function_name = 'solve_continuous_are'
    solve_continuous_are.stypy_param_names_list = ['a', 'b', 'q', 'r', 'e', 's', 'balanced']
    solve_continuous_are.stypy_varargs_param_name = None
    solve_continuous_are.stypy_kwargs_param_name = None
    solve_continuous_are.stypy_call_defaults = defaults
    solve_continuous_are.stypy_call_varargs = varargs
    solve_continuous_are.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_continuous_are', ['a', 'b', 'q', 'r', 'e', 's', 'balanced'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_continuous_are', localization, ['a', 'b', 'q', 'r', 'e', 's', 'balanced'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_continuous_are(...)' code ##################

    str_36471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, (-1)), 'str', '\n    Solves the continuous-time algebraic Riccati equation (CARE).\n\n    The CARE is defined as\n\n    .. math::\n\n          X A + A^H X - X B R^{-1} B^H X + Q = 0\n\n    The limitations for a solution to exist are :\n\n        * All eigenvalues of :math:`A` on the right half plane, should be\n          controllable.\n\n        * The associated hamiltonian pencil (See Notes), should have\n          eigenvalues sufficiently away from the imaginary axis.\n\n    Moreover, if ``e`` or ``s`` is not precisely ``None``, then the\n    generalized version of CARE\n\n    .. math::\n\n          E^HXA + A^HXE - (E^HXB + S) R^{-1} (B^HXE + S^H) + Q = 0\n\n    is solved. When omitted, ``e`` is assumed to be the identity and ``s``\n    is assumed to be the zero matrix with sizes compatible with ``a`` and\n    ``b`` respectively.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Square matrix\n    b : (M, N) array_like\n        Input\n    q : (M, M) array_like\n        Input\n    r : (N, N) array_like\n        Nonsingular square matrix\n    e : (M, M) array_like, optional\n        Nonsingular square matrix\n    s : (M, N) array_like, optional\n        Input\n    balanced : bool, optional\n        The boolean that indicates whether a balancing step is performed\n        on the data. The default is set to True.\n\n    Returns\n    -------\n    x : (M, M) ndarray\n        Solution to the continuous-time algebraic Riccati equation.\n\n    Raises\n    ------\n    LinAlgError\n        For cases where the stable subspace of the pencil could not be\n        isolated. See Notes section and the references for details.\n\n    See Also\n    --------\n    solve_discrete_are : Solves the discrete-time algebraic Riccati equation\n\n    Notes\n    -----\n    The equation is solved by forming the extended hamiltonian matrix pencil,\n    as described in [1]_, :math:`H - \\lambda J` given by the block matrices ::\n\n        [ A    0    B ]             [ E   0    0 ]\n        [-Q  -A^H  -S ] - \\lambda * [ 0  E^H   0 ]\n        [ S^H B^H   R ]             [ 0   0    0 ]\n\n    and using a QZ decomposition method.\n\n    In this algorithm, the fail conditions are linked to the symmetry\n    of the product :math:`U_2 U_1^{-1}` and condition number of\n    :math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the\n    eigenvectors spanning the stable subspace with 2m rows and partitioned\n    into two m-row matrices. See [1]_ and [2]_ for more details.\n\n    In order to improve the QZ decomposition accuracy, the pencil goes\n    through a balancing step where the sum of absolute values of\n    :math:`H` and :math:`J` entries (after removing the diagonal entries of\n    the sum) is balanced following the recipe given in [3]_.\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1]  P. van Dooren , "A Generalized Eigenvalue Approach For Solving\n       Riccati Equations.", SIAM Journal on Scientific and Statistical\n       Computing, Vol.2(2), DOI: 10.1137/0902010\n\n    .. [2] A.J. Laub, "A Schur Method for Solving Algebraic Riccati\n       Equations.", Massachusetts Institute of Technology. Laboratory for\n       Information and Decision Systems. LIDS-R ; 859. Available online :\n       http://hdl.handle.net/1721.1/1301\n\n    .. [3] P. Benner, "Symplectic Balancing of Hamiltonian Matrices", 2001,\n       SIAM J. Sci. Comput., 2001, Vol.22(5), DOI: 10.1137/S1064827500367993\n\n    Examples\n    --------\n    Given `a`, `b`, `q`, and `r` solve for `x`:\n\n    >>> from scipy import linalg\n    >>> a = np.array([[4, 3], [-4.5, -3.5]])\n    >>> b = np.array([[1], [-1]])\n    >>> q = np.array([[9, 6], [6, 4.]])\n    >>> r = 1\n    >>> x = linalg.solve_continuous_are(a, b, q, r)\n    >>> x\n    array([[ 21.72792206,  14.48528137],\n           [ 14.48528137,   9.65685425]])\n    >>> np.allclose(a.T.dot(x) + x.dot(a)-x.dot(b).dot(b.T).dot(x), -q)\n    True\n\n    ')
    
    # Assigning a Call to a Tuple (line 444):
    
    # Assigning a Subscript to a Name (line 444):
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_36472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 4), 'int')
    
    # Call to _are_validate_args(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'a' (line 445)
    a_36474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 53), 'a', False)
    # Getting the type of 'b' (line 445)
    b_36475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 56), 'b', False)
    # Getting the type of 'q' (line 445)
    q_36476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 59), 'q', False)
    # Getting the type of 'r' (line 445)
    r_36477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 62), 'r', False)
    # Getting the type of 'e' (line 445)
    e_36478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 65), 'e', False)
    # Getting the type of 's' (line 445)
    s_36479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 68), 's', False)
    str_36480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 71), 'str', 'care')
    # Processing the call keyword arguments (line 444)
    kwargs_36481 = {}
    # Getting the type of '_are_validate_args' (line 444)
    _are_validate_args_36473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 444)
    _are_validate_args_call_result_36482 = invoke(stypy.reporting.localization.Localization(__file__, 444, 46), _are_validate_args_36473, *[a_36474, b_36475, q_36476, r_36477, e_36478, s_36479, str_36480], **kwargs_36481)
    
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___36483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), _are_validate_args_call_result_36482, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_36484 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), getitem___36483, int_36472)
    
    # Assigning a type to the variable 'tuple_var_assignment_35883' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35883', subscript_call_result_36484)
    
    # Assigning a Subscript to a Name (line 444):
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_36485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 4), 'int')
    
    # Call to _are_validate_args(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'a' (line 445)
    a_36487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 53), 'a', False)
    # Getting the type of 'b' (line 445)
    b_36488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 56), 'b', False)
    # Getting the type of 'q' (line 445)
    q_36489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 59), 'q', False)
    # Getting the type of 'r' (line 445)
    r_36490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 62), 'r', False)
    # Getting the type of 'e' (line 445)
    e_36491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 65), 'e', False)
    # Getting the type of 's' (line 445)
    s_36492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 68), 's', False)
    str_36493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 71), 'str', 'care')
    # Processing the call keyword arguments (line 444)
    kwargs_36494 = {}
    # Getting the type of '_are_validate_args' (line 444)
    _are_validate_args_36486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 444)
    _are_validate_args_call_result_36495 = invoke(stypy.reporting.localization.Localization(__file__, 444, 46), _are_validate_args_36486, *[a_36487, b_36488, q_36489, r_36490, e_36491, s_36492, str_36493], **kwargs_36494)
    
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___36496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), _are_validate_args_call_result_36495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_36497 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), getitem___36496, int_36485)
    
    # Assigning a type to the variable 'tuple_var_assignment_35884' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35884', subscript_call_result_36497)
    
    # Assigning a Subscript to a Name (line 444):
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_36498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 4), 'int')
    
    # Call to _are_validate_args(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'a' (line 445)
    a_36500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 53), 'a', False)
    # Getting the type of 'b' (line 445)
    b_36501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 56), 'b', False)
    # Getting the type of 'q' (line 445)
    q_36502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 59), 'q', False)
    # Getting the type of 'r' (line 445)
    r_36503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 62), 'r', False)
    # Getting the type of 'e' (line 445)
    e_36504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 65), 'e', False)
    # Getting the type of 's' (line 445)
    s_36505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 68), 's', False)
    str_36506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 71), 'str', 'care')
    # Processing the call keyword arguments (line 444)
    kwargs_36507 = {}
    # Getting the type of '_are_validate_args' (line 444)
    _are_validate_args_36499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 444)
    _are_validate_args_call_result_36508 = invoke(stypy.reporting.localization.Localization(__file__, 444, 46), _are_validate_args_36499, *[a_36500, b_36501, q_36502, r_36503, e_36504, s_36505, str_36506], **kwargs_36507)
    
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___36509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), _are_validate_args_call_result_36508, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_36510 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), getitem___36509, int_36498)
    
    # Assigning a type to the variable 'tuple_var_assignment_35885' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35885', subscript_call_result_36510)
    
    # Assigning a Subscript to a Name (line 444):
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_36511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 4), 'int')
    
    # Call to _are_validate_args(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'a' (line 445)
    a_36513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 53), 'a', False)
    # Getting the type of 'b' (line 445)
    b_36514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 56), 'b', False)
    # Getting the type of 'q' (line 445)
    q_36515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 59), 'q', False)
    # Getting the type of 'r' (line 445)
    r_36516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 62), 'r', False)
    # Getting the type of 'e' (line 445)
    e_36517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 65), 'e', False)
    # Getting the type of 's' (line 445)
    s_36518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 68), 's', False)
    str_36519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 71), 'str', 'care')
    # Processing the call keyword arguments (line 444)
    kwargs_36520 = {}
    # Getting the type of '_are_validate_args' (line 444)
    _are_validate_args_36512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 444)
    _are_validate_args_call_result_36521 = invoke(stypy.reporting.localization.Localization(__file__, 444, 46), _are_validate_args_36512, *[a_36513, b_36514, q_36515, r_36516, e_36517, s_36518, str_36519], **kwargs_36520)
    
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___36522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), _are_validate_args_call_result_36521, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_36523 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), getitem___36522, int_36511)
    
    # Assigning a type to the variable 'tuple_var_assignment_35886' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35886', subscript_call_result_36523)
    
    # Assigning a Subscript to a Name (line 444):
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_36524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 4), 'int')
    
    # Call to _are_validate_args(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'a' (line 445)
    a_36526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 53), 'a', False)
    # Getting the type of 'b' (line 445)
    b_36527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 56), 'b', False)
    # Getting the type of 'q' (line 445)
    q_36528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 59), 'q', False)
    # Getting the type of 'r' (line 445)
    r_36529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 62), 'r', False)
    # Getting the type of 'e' (line 445)
    e_36530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 65), 'e', False)
    # Getting the type of 's' (line 445)
    s_36531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 68), 's', False)
    str_36532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 71), 'str', 'care')
    # Processing the call keyword arguments (line 444)
    kwargs_36533 = {}
    # Getting the type of '_are_validate_args' (line 444)
    _are_validate_args_36525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 444)
    _are_validate_args_call_result_36534 = invoke(stypy.reporting.localization.Localization(__file__, 444, 46), _are_validate_args_36525, *[a_36526, b_36527, q_36528, r_36529, e_36530, s_36531, str_36532], **kwargs_36533)
    
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___36535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), _are_validate_args_call_result_36534, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_36536 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), getitem___36535, int_36524)
    
    # Assigning a type to the variable 'tuple_var_assignment_35887' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35887', subscript_call_result_36536)
    
    # Assigning a Subscript to a Name (line 444):
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_36537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 4), 'int')
    
    # Call to _are_validate_args(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'a' (line 445)
    a_36539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 53), 'a', False)
    # Getting the type of 'b' (line 445)
    b_36540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 56), 'b', False)
    # Getting the type of 'q' (line 445)
    q_36541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 59), 'q', False)
    # Getting the type of 'r' (line 445)
    r_36542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 62), 'r', False)
    # Getting the type of 'e' (line 445)
    e_36543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 65), 'e', False)
    # Getting the type of 's' (line 445)
    s_36544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 68), 's', False)
    str_36545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 71), 'str', 'care')
    # Processing the call keyword arguments (line 444)
    kwargs_36546 = {}
    # Getting the type of '_are_validate_args' (line 444)
    _are_validate_args_36538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 444)
    _are_validate_args_call_result_36547 = invoke(stypy.reporting.localization.Localization(__file__, 444, 46), _are_validate_args_36538, *[a_36539, b_36540, q_36541, r_36542, e_36543, s_36544, str_36545], **kwargs_36546)
    
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___36548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), _are_validate_args_call_result_36547, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_36549 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), getitem___36548, int_36537)
    
    # Assigning a type to the variable 'tuple_var_assignment_35888' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35888', subscript_call_result_36549)
    
    # Assigning a Subscript to a Name (line 444):
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_36550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 4), 'int')
    
    # Call to _are_validate_args(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'a' (line 445)
    a_36552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 53), 'a', False)
    # Getting the type of 'b' (line 445)
    b_36553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 56), 'b', False)
    # Getting the type of 'q' (line 445)
    q_36554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 59), 'q', False)
    # Getting the type of 'r' (line 445)
    r_36555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 62), 'r', False)
    # Getting the type of 'e' (line 445)
    e_36556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 65), 'e', False)
    # Getting the type of 's' (line 445)
    s_36557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 68), 's', False)
    str_36558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 71), 'str', 'care')
    # Processing the call keyword arguments (line 444)
    kwargs_36559 = {}
    # Getting the type of '_are_validate_args' (line 444)
    _are_validate_args_36551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 444)
    _are_validate_args_call_result_36560 = invoke(stypy.reporting.localization.Localization(__file__, 444, 46), _are_validate_args_36551, *[a_36552, b_36553, q_36554, r_36555, e_36556, s_36557, str_36558], **kwargs_36559)
    
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___36561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), _are_validate_args_call_result_36560, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_36562 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), getitem___36561, int_36550)
    
    # Assigning a type to the variable 'tuple_var_assignment_35889' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35889', subscript_call_result_36562)
    
    # Assigning a Subscript to a Name (line 444):
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_36563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 4), 'int')
    
    # Call to _are_validate_args(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'a' (line 445)
    a_36565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 53), 'a', False)
    # Getting the type of 'b' (line 445)
    b_36566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 56), 'b', False)
    # Getting the type of 'q' (line 445)
    q_36567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 59), 'q', False)
    # Getting the type of 'r' (line 445)
    r_36568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 62), 'r', False)
    # Getting the type of 'e' (line 445)
    e_36569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 65), 'e', False)
    # Getting the type of 's' (line 445)
    s_36570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 68), 's', False)
    str_36571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 71), 'str', 'care')
    # Processing the call keyword arguments (line 444)
    kwargs_36572 = {}
    # Getting the type of '_are_validate_args' (line 444)
    _are_validate_args_36564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 444)
    _are_validate_args_call_result_36573 = invoke(stypy.reporting.localization.Localization(__file__, 444, 46), _are_validate_args_36564, *[a_36565, b_36566, q_36567, r_36568, e_36569, s_36570, str_36571], **kwargs_36572)
    
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___36574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), _are_validate_args_call_result_36573, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_36575 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), getitem___36574, int_36563)
    
    # Assigning a type to the variable 'tuple_var_assignment_35890' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35890', subscript_call_result_36575)
    
    # Assigning a Subscript to a Name (line 444):
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_36576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 4), 'int')
    
    # Call to _are_validate_args(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'a' (line 445)
    a_36578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 53), 'a', False)
    # Getting the type of 'b' (line 445)
    b_36579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 56), 'b', False)
    # Getting the type of 'q' (line 445)
    q_36580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 59), 'q', False)
    # Getting the type of 'r' (line 445)
    r_36581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 62), 'r', False)
    # Getting the type of 'e' (line 445)
    e_36582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 65), 'e', False)
    # Getting the type of 's' (line 445)
    s_36583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 68), 's', False)
    str_36584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 71), 'str', 'care')
    # Processing the call keyword arguments (line 444)
    kwargs_36585 = {}
    # Getting the type of '_are_validate_args' (line 444)
    _are_validate_args_36577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 444)
    _are_validate_args_call_result_36586 = invoke(stypy.reporting.localization.Localization(__file__, 444, 46), _are_validate_args_36577, *[a_36578, b_36579, q_36580, r_36581, e_36582, s_36583, str_36584], **kwargs_36585)
    
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___36587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), _are_validate_args_call_result_36586, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_36588 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), getitem___36587, int_36576)
    
    # Assigning a type to the variable 'tuple_var_assignment_35891' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35891', subscript_call_result_36588)
    
    # Assigning a Subscript to a Name (line 444):
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_36589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 4), 'int')
    
    # Call to _are_validate_args(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'a' (line 445)
    a_36591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 53), 'a', False)
    # Getting the type of 'b' (line 445)
    b_36592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 56), 'b', False)
    # Getting the type of 'q' (line 445)
    q_36593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 59), 'q', False)
    # Getting the type of 'r' (line 445)
    r_36594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 62), 'r', False)
    # Getting the type of 'e' (line 445)
    e_36595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 65), 'e', False)
    # Getting the type of 's' (line 445)
    s_36596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 68), 's', False)
    str_36597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 71), 'str', 'care')
    # Processing the call keyword arguments (line 444)
    kwargs_36598 = {}
    # Getting the type of '_are_validate_args' (line 444)
    _are_validate_args_36590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 444)
    _are_validate_args_call_result_36599 = invoke(stypy.reporting.localization.Localization(__file__, 444, 46), _are_validate_args_36590, *[a_36591, b_36592, q_36593, r_36594, e_36595, s_36596, str_36597], **kwargs_36598)
    
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___36600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), _are_validate_args_call_result_36599, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_36601 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), getitem___36600, int_36589)
    
    # Assigning a type to the variable 'tuple_var_assignment_35892' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35892', subscript_call_result_36601)
    
    # Assigning a Name to a Name (line 444):
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_35883' (line 444)
    tuple_var_assignment_35883_36602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35883')
    # Assigning a type to the variable 'a' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'a', tuple_var_assignment_35883_36602)
    
    # Assigning a Name to a Name (line 444):
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_35884' (line 444)
    tuple_var_assignment_35884_36603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35884')
    # Assigning a type to the variable 'b' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 7), 'b', tuple_var_assignment_35884_36603)
    
    # Assigning a Name to a Name (line 444):
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_35885' (line 444)
    tuple_var_assignment_35885_36604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35885')
    # Assigning a type to the variable 'q' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 10), 'q', tuple_var_assignment_35885_36604)
    
    # Assigning a Name to a Name (line 444):
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_35886' (line 444)
    tuple_var_assignment_35886_36605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35886')
    # Assigning a type to the variable 'r' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 13), 'r', tuple_var_assignment_35886_36605)
    
    # Assigning a Name to a Name (line 444):
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_35887' (line 444)
    tuple_var_assignment_35887_36606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35887')
    # Assigning a type to the variable 'e' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 'e', tuple_var_assignment_35887_36606)
    
    # Assigning a Name to a Name (line 444):
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_35888' (line 444)
    tuple_var_assignment_35888_36607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35888')
    # Assigning a type to the variable 's' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 19), 's', tuple_var_assignment_35888_36607)
    
    # Assigning a Name to a Name (line 444):
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_35889' (line 444)
    tuple_var_assignment_35889_36608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35889')
    # Assigning a type to the variable 'm' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 22), 'm', tuple_var_assignment_35889_36608)
    
    # Assigning a Name to a Name (line 444):
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_35890' (line 444)
    tuple_var_assignment_35890_36609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35890')
    # Assigning a type to the variable 'n' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 25), 'n', tuple_var_assignment_35890_36609)
    
    # Assigning a Name to a Name (line 444):
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_35891' (line 444)
    tuple_var_assignment_35891_36610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35891')
    # Assigning a type to the variable 'r_or_c' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 28), 'r_or_c', tuple_var_assignment_35891_36610)
    
    # Assigning a Name to a Name (line 444):
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_35892' (line 444)
    tuple_var_assignment_35892_36611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_35892')
    # Assigning a type to the variable 'gen_are' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 36), 'gen_are', tuple_var_assignment_35892_36611)
    
    # Assigning a Call to a Name (line 447):
    
    # Assigning a Call to a Name (line 447):
    
    # Assigning a Call to a Name (line 447):
    
    # Call to empty(...): (line 447)
    # Processing the call arguments (line 447)
    
    # Obtaining an instance of the builtin type 'tuple' (line 447)
    tuple_36614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 447)
    # Adding element type (line 447)
    int_36615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 18), 'int')
    # Getting the type of 'm' (line 447)
    m_36616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 20), 'm', False)
    # Applying the binary operator '*' (line 447)
    result_mul_36617 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 18), '*', int_36615, m_36616)
    
    # Getting the type of 'n' (line 447)
    n_36618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 22), 'n', False)
    # Applying the binary operator '+' (line 447)
    result_add_36619 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 18), '+', result_mul_36617, n_36618)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 18), tuple_36614, result_add_36619)
    # Adding element type (line 447)
    int_36620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 25), 'int')
    # Getting the type of 'm' (line 447)
    m_36621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 27), 'm', False)
    # Applying the binary operator '*' (line 447)
    result_mul_36622 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 25), '*', int_36620, m_36621)
    
    # Getting the type of 'n' (line 447)
    n_36623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 29), 'n', False)
    # Applying the binary operator '+' (line 447)
    result_add_36624 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 25), '+', result_mul_36622, n_36623)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 18), tuple_36614, result_add_36624)
    
    # Processing the call keyword arguments (line 447)
    # Getting the type of 'r_or_c' (line 447)
    r_or_c_36625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 39), 'r_or_c', False)
    keyword_36626 = r_or_c_36625
    kwargs_36627 = {'dtype': keyword_36626}
    # Getting the type of 'np' (line 447)
    np_36612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 447)
    empty_36613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 8), np_36612, 'empty')
    # Calling empty(args, kwargs) (line 447)
    empty_call_result_36628 = invoke(stypy.reporting.localization.Localization(__file__, 447, 8), empty_36613, *[tuple_36614], **kwargs_36627)
    
    # Assigning a type to the variable 'H' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'H', empty_call_result_36628)
    
    # Assigning a Name to a Subscript (line 448):
    
    # Assigning a Name to a Subscript (line 448):
    
    # Assigning a Name to a Subscript (line 448):
    # Getting the type of 'a' (line 448)
    a_36629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 16), 'a')
    # Getting the type of 'H' (line 448)
    H_36630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'H')
    # Getting the type of 'm' (line 448)
    m_36631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 7), 'm')
    slice_36632 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 448, 4), None, m_36631, None)
    # Getting the type of 'm' (line 448)
    m_36633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 11), 'm')
    slice_36634 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 448, 4), None, m_36633, None)
    # Storing an element on a container (line 448)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 4), H_36630, ((slice_36632, slice_36634), a_36629))
    
    # Assigning a Num to a Subscript (line 449):
    
    # Assigning a Num to a Subscript (line 449):
    
    # Assigning a Num to a Subscript (line 449):
    float_36635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 19), 'float')
    # Getting the type of 'H' (line 449)
    H_36636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'H')
    # Getting the type of 'm' (line 449)
    m_36637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 7), 'm')
    slice_36638 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 449, 4), None, m_36637, None)
    # Getting the type of 'm' (line 449)
    m_36639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 10), 'm')
    int_36640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 12), 'int')
    # Getting the type of 'm' (line 449)
    m_36641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 14), 'm')
    # Applying the binary operator '*' (line 449)
    result_mul_36642 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 12), '*', int_36640, m_36641)
    
    slice_36643 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 449, 4), m_36639, result_mul_36642, None)
    # Storing an element on a container (line 449)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 4), H_36636, ((slice_36638, slice_36643), float_36635))
    
    # Assigning a Name to a Subscript (line 450):
    
    # Assigning a Name to a Subscript (line 450):
    
    # Assigning a Name to a Subscript (line 450):
    # Getting the type of 'b' (line 450)
    b_36644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 18), 'b')
    # Getting the type of 'H' (line 450)
    H_36645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'H')
    # Getting the type of 'm' (line 450)
    m_36646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 7), 'm')
    slice_36647 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 450, 4), None, m_36646, None)
    int_36648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 10), 'int')
    # Getting the type of 'm' (line 450)
    m_36649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'm')
    # Applying the binary operator '*' (line 450)
    result_mul_36650 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 10), '*', int_36648, m_36649)
    
    slice_36651 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 450, 4), result_mul_36650, None, None)
    # Storing an element on a container (line 450)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 4), H_36645, ((slice_36647, slice_36651), b_36644))
    
    # Assigning a UnaryOp to a Subscript (line 451):
    
    # Assigning a UnaryOp to a Subscript (line 451):
    
    # Assigning a UnaryOp to a Subscript (line 451):
    
    # Getting the type of 'q' (line 451)
    q_36652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 20), 'q')
    # Applying the 'usub' unary operator (line 451)
    result___neg___36653 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 19), 'usub', q_36652)
    
    # Getting the type of 'H' (line 451)
    H_36654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'H')
    # Getting the type of 'm' (line 451)
    m_36655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 6), 'm')
    int_36656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 8), 'int')
    # Getting the type of 'm' (line 451)
    m_36657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 10), 'm')
    # Applying the binary operator '*' (line 451)
    result_mul_36658 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 8), '*', int_36656, m_36657)
    
    slice_36659 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 451, 4), m_36655, result_mul_36658, None)
    # Getting the type of 'm' (line 451)
    m_36660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 14), 'm')
    slice_36661 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 451, 4), None, m_36660, None)
    # Storing an element on a container (line 451)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 4), H_36654, ((slice_36659, slice_36661), result___neg___36653))
    
    # Assigning a UnaryOp to a Subscript (line 452):
    
    # Assigning a UnaryOp to a Subscript (line 452):
    
    # Assigning a UnaryOp to a Subscript (line 452):
    
    
    # Call to conj(...): (line 452)
    # Processing the call keyword arguments (line 452)
    kwargs_36664 = {}
    # Getting the type of 'a' (line 452)
    a_36662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 23), 'a', False)
    # Obtaining the member 'conj' of a type (line 452)
    conj_36663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 23), a_36662, 'conj')
    # Calling conj(args, kwargs) (line 452)
    conj_call_result_36665 = invoke(stypy.reporting.localization.Localization(__file__, 452, 23), conj_36663, *[], **kwargs_36664)
    
    # Obtaining the member 'T' of a type (line 452)
    T_36666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 23), conj_call_result_36665, 'T')
    # Applying the 'usub' unary operator (line 452)
    result___neg___36667 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 22), 'usub', T_36666)
    
    # Getting the type of 'H' (line 452)
    H_36668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'H')
    # Getting the type of 'm' (line 452)
    m_36669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 6), 'm')
    int_36670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 8), 'int')
    # Getting the type of 'm' (line 452)
    m_36671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 10), 'm')
    # Applying the binary operator '*' (line 452)
    result_mul_36672 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 8), '*', int_36670, m_36671)
    
    slice_36673 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 452, 4), m_36669, result_mul_36672, None)
    # Getting the type of 'm' (line 452)
    m_36674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 13), 'm')
    int_36675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 15), 'int')
    # Getting the type of 'm' (line 452)
    m_36676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 17), 'm')
    # Applying the binary operator '*' (line 452)
    result_mul_36677 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 15), '*', int_36675, m_36676)
    
    slice_36678 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 452, 4), m_36674, result_mul_36677, None)
    # Storing an element on a container (line 452)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 4), H_36668, ((slice_36673, slice_36678), result___neg___36667))
    
    # Assigning a IfExp to a Subscript (line 453):
    
    # Assigning a IfExp to a Subscript (line 453):
    
    # Assigning a IfExp to a Subscript (line 453):
    
    
    # Getting the type of 's' (line 453)
    s_36679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 27), 's')
    # Getting the type of 'None' (line 453)
    None_36680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 32), 'None')
    # Applying the binary operator 'is' (line 453)
    result_is__36681 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 27), 'is', s_36679, None_36680)
    
    # Testing the type of an if expression (line 453)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 21), result_is__36681)
    # SSA begins for if expression (line 453)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    float_36682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 21), 'float')
    # SSA branch for the else part of an if expression (line 453)
    module_type_store.open_ssa_branch('if expression else')
    
    # Getting the type of 's' (line 453)
    s_36683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 43), 's')
    # Applying the 'usub' unary operator (line 453)
    result___neg___36684 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 42), 'usub', s_36683)
    
    # SSA join for if expression (line 453)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_36685 = union_type.UnionType.add(float_36682, result___neg___36684)
    
    # Getting the type of 'H' (line 453)
    H_36686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'H')
    # Getting the type of 'm' (line 453)
    m_36687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 6), 'm')
    int_36688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 8), 'int')
    # Getting the type of 'm' (line 453)
    m_36689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 10), 'm')
    # Applying the binary operator '*' (line 453)
    result_mul_36690 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 8), '*', int_36688, m_36689)
    
    slice_36691 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 453, 4), m_36687, result_mul_36690, None)
    int_36692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 13), 'int')
    # Getting the type of 'm' (line 453)
    m_36693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 15), 'm')
    # Applying the binary operator '*' (line 453)
    result_mul_36694 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 13), '*', int_36692, m_36693)
    
    slice_36695 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 453, 4), result_mul_36694, None, None)
    # Storing an element on a container (line 453)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 4), H_36686, ((slice_36691, slice_36695), if_exp_36685))
    
    # Assigning a IfExp to a Subscript (line 454):
    
    # Assigning a IfExp to a Subscript (line 454):
    
    # Assigning a IfExp to a Subscript (line 454):
    
    
    # Getting the type of 's' (line 454)
    s_36696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 24), 's')
    # Getting the type of 'None' (line 454)
    None_36697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 29), 'None')
    # Applying the binary operator 'is' (line 454)
    result_is__36698 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 24), 'is', s_36696, None_36697)
    
    # Testing the type of an if expression (line 454)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 18), result_is__36698)
    # SSA begins for if expression (line 454)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    float_36699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 18), 'float')
    # SSA branch for the else part of an if expression (line 454)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to conj(...): (line 454)
    # Processing the call keyword arguments (line 454)
    kwargs_36702 = {}
    # Getting the type of 's' (line 454)
    s_36700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 39), 's', False)
    # Obtaining the member 'conj' of a type (line 454)
    conj_36701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 39), s_36700, 'conj')
    # Calling conj(args, kwargs) (line 454)
    conj_call_result_36703 = invoke(stypy.reporting.localization.Localization(__file__, 454, 39), conj_36701, *[], **kwargs_36702)
    
    # Obtaining the member 'T' of a type (line 454)
    T_36704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 39), conj_call_result_36703, 'T')
    # SSA join for if expression (line 454)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_36705 = union_type.UnionType.add(float_36699, T_36704)
    
    # Getting the type of 'H' (line 454)
    H_36706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'H')
    int_36707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 6), 'int')
    # Getting the type of 'm' (line 454)
    m_36708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'm')
    # Applying the binary operator '*' (line 454)
    result_mul_36709 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 6), '*', int_36707, m_36708)
    
    slice_36710 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 454, 4), result_mul_36709, None, None)
    # Getting the type of 'm' (line 454)
    m_36711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 13), 'm')
    slice_36712 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 454, 4), None, m_36711, None)
    # Storing an element on a container (line 454)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 4), H_36706, ((slice_36710, slice_36712), if_exp_36705))
    
    # Assigning a Attribute to a Subscript (line 455):
    
    # Assigning a Attribute to a Subscript (line 455):
    
    # Assigning a Attribute to a Subscript (line 455):
    
    # Call to conj(...): (line 455)
    # Processing the call keyword arguments (line 455)
    kwargs_36715 = {}
    # Getting the type of 'b' (line 455)
    b_36713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 21), 'b', False)
    # Obtaining the member 'conj' of a type (line 455)
    conj_36714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 21), b_36713, 'conj')
    # Calling conj(args, kwargs) (line 455)
    conj_call_result_36716 = invoke(stypy.reporting.localization.Localization(__file__, 455, 21), conj_36714, *[], **kwargs_36715)
    
    # Obtaining the member 'T' of a type (line 455)
    T_36717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 21), conj_call_result_36716, 'T')
    # Getting the type of 'H' (line 455)
    H_36718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'H')
    int_36719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 6), 'int')
    # Getting the type of 'm' (line 455)
    m_36720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'm')
    # Applying the binary operator '*' (line 455)
    result_mul_36721 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 6), '*', int_36719, m_36720)
    
    slice_36722 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 455, 4), result_mul_36721, None, None)
    # Getting the type of 'm' (line 455)
    m_36723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'm')
    int_36724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 14), 'int')
    # Getting the type of 'm' (line 455)
    m_36725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'm')
    # Applying the binary operator '*' (line 455)
    result_mul_36726 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 14), '*', int_36724, m_36725)
    
    slice_36727 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 455, 4), m_36723, result_mul_36726, None)
    # Storing an element on a container (line 455)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 4), H_36718, ((slice_36722, slice_36727), T_36717))
    
    # Assigning a Name to a Subscript (line 456):
    
    # Assigning a Name to a Subscript (line 456):
    
    # Assigning a Name to a Subscript (line 456):
    # Getting the type of 'r' (line 456)
    r_36728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 20), 'r')
    # Getting the type of 'H' (line 456)
    H_36729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'H')
    int_36730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 6), 'int')
    # Getting the type of 'm' (line 456)
    m_36731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'm')
    # Applying the binary operator '*' (line 456)
    result_mul_36732 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 6), '*', int_36730, m_36731)
    
    slice_36733 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 456, 4), result_mul_36732, None, None)
    int_36734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 12), 'int')
    # Getting the type of 'm' (line 456)
    m_36735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 14), 'm')
    # Applying the binary operator '*' (line 456)
    result_mul_36736 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 12), '*', int_36734, m_36735)
    
    slice_36737 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 456, 4), result_mul_36736, None, None)
    # Storing an element on a container (line 456)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 4), H_36729, ((slice_36733, slice_36737), r_36728))
    
    
    # Evaluating a boolean operation
    # Getting the type of 'gen_are' (line 458)
    gen_are_36738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 7), 'gen_are')
    
    # Getting the type of 'e' (line 458)
    e_36739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 19), 'e')
    # Getting the type of 'None' (line 458)
    None_36740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 28), 'None')
    # Applying the binary operator 'isnot' (line 458)
    result_is_not_36741 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 19), 'isnot', e_36739, None_36740)
    
    # Applying the binary operator 'and' (line 458)
    result_and_keyword_36742 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 7), 'and', gen_are_36738, result_is_not_36741)
    
    # Testing the type of an if condition (line 458)
    if_condition_36743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 458, 4), result_and_keyword_36742)
    # Assigning a type to the variable 'if_condition_36743' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'if_condition_36743', if_condition_36743)
    # SSA begins for if statement (line 458)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Call to block_diag(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'e' (line 459)
    e_36745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 23), 'e', False)
    
    # Call to conj(...): (line 459)
    # Processing the call keyword arguments (line 459)
    kwargs_36748 = {}
    # Getting the type of 'e' (line 459)
    e_36746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 26), 'e', False)
    # Obtaining the member 'conj' of a type (line 459)
    conj_36747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 26), e_36746, 'conj')
    # Calling conj(args, kwargs) (line 459)
    conj_call_result_36749 = invoke(stypy.reporting.localization.Localization(__file__, 459, 26), conj_36747, *[], **kwargs_36748)
    
    # Obtaining the member 'T' of a type (line 459)
    T_36750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 26), conj_call_result_36749, 'T')
    
    # Call to zeros_like(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'r' (line 459)
    r_36753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 52), 'r', False)
    # Processing the call keyword arguments (line 459)
    # Getting the type of 'r_or_c' (line 459)
    r_or_c_36754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 61), 'r_or_c', False)
    keyword_36755 = r_or_c_36754
    kwargs_36756 = {'dtype': keyword_36755}
    # Getting the type of 'np' (line 459)
    np_36751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 38), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 459)
    zeros_like_36752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 38), np_36751, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 459)
    zeros_like_call_result_36757 = invoke(stypy.reporting.localization.Localization(__file__, 459, 38), zeros_like_36752, *[r_36753], **kwargs_36756)
    
    # Processing the call keyword arguments (line 459)
    kwargs_36758 = {}
    # Getting the type of 'block_diag' (line 459)
    block_diag_36744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'block_diag', False)
    # Calling block_diag(args, kwargs) (line 459)
    block_diag_call_result_36759 = invoke(stypy.reporting.localization.Localization(__file__, 459, 12), block_diag_36744, *[e_36745, T_36750, zeros_like_call_result_36757], **kwargs_36758)
    
    # Assigning a type to the variable 'J' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'J', block_diag_call_result_36759)
    # SSA branch for the else part of an if statement (line 458)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 461):
    
    # Assigning a Call to a Name (line 461):
    
    # Assigning a Call to a Name (line 461):
    
    # Call to block_diag(...): (line 461)
    # Processing the call arguments (line 461)
    
    # Call to eye(...): (line 461)
    # Processing the call arguments (line 461)
    int_36763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 30), 'int')
    # Getting the type of 'm' (line 461)
    m_36764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 32), 'm', False)
    # Applying the binary operator '*' (line 461)
    result_mul_36765 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 30), '*', int_36763, m_36764)
    
    # Processing the call keyword arguments (line 461)
    kwargs_36766 = {}
    # Getting the type of 'np' (line 461)
    np_36761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 23), 'np', False)
    # Obtaining the member 'eye' of a type (line 461)
    eye_36762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 23), np_36761, 'eye')
    # Calling eye(args, kwargs) (line 461)
    eye_call_result_36767 = invoke(stypy.reporting.localization.Localization(__file__, 461, 23), eye_36762, *[result_mul_36765], **kwargs_36766)
    
    
    # Call to zeros_like(...): (line 461)
    # Processing the call arguments (line 461)
    # Getting the type of 'r' (line 461)
    r_36770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 50), 'r', False)
    # Processing the call keyword arguments (line 461)
    # Getting the type of 'r_or_c' (line 461)
    r_or_c_36771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 59), 'r_or_c', False)
    keyword_36772 = r_or_c_36771
    kwargs_36773 = {'dtype': keyword_36772}
    # Getting the type of 'np' (line 461)
    np_36768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 36), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 461)
    zeros_like_36769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 36), np_36768, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 461)
    zeros_like_call_result_36774 = invoke(stypy.reporting.localization.Localization(__file__, 461, 36), zeros_like_36769, *[r_36770], **kwargs_36773)
    
    # Processing the call keyword arguments (line 461)
    kwargs_36775 = {}
    # Getting the type of 'block_diag' (line 461)
    block_diag_36760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'block_diag', False)
    # Calling block_diag(args, kwargs) (line 461)
    block_diag_call_result_36776 = invoke(stypy.reporting.localization.Localization(__file__, 461, 12), block_diag_36760, *[eye_call_result_36767, zeros_like_call_result_36774], **kwargs_36775)
    
    # Assigning a type to the variable 'J' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'J', block_diag_call_result_36776)
    # SSA join for if statement (line 458)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'balanced' (line 463)
    balanced_36777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 7), 'balanced')
    # Testing the type of an if condition (line 463)
    if_condition_36778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 4), balanced_36777)
    # Assigning a type to the variable 'if_condition_36778' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'if_condition_36778', if_condition_36778)
    # SSA begins for if statement (line 463)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 466):
    
    # Assigning a BinOp to a Name (line 466):
    
    # Assigning a BinOp to a Name (line 466):
    
    # Call to abs(...): (line 466)
    # Processing the call arguments (line 466)
    # Getting the type of 'H' (line 466)
    H_36781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 19), 'H', False)
    # Processing the call keyword arguments (line 466)
    kwargs_36782 = {}
    # Getting the type of 'np' (line 466)
    np_36779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'np', False)
    # Obtaining the member 'abs' of a type (line 466)
    abs_36780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 12), np_36779, 'abs')
    # Calling abs(args, kwargs) (line 466)
    abs_call_result_36783 = invoke(stypy.reporting.localization.Localization(__file__, 466, 12), abs_36780, *[H_36781], **kwargs_36782)
    
    
    # Call to abs(...): (line 466)
    # Processing the call arguments (line 466)
    # Getting the type of 'J' (line 466)
    J_36786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 31), 'J', False)
    # Processing the call keyword arguments (line 466)
    kwargs_36787 = {}
    # Getting the type of 'np' (line 466)
    np_36784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 24), 'np', False)
    # Obtaining the member 'abs' of a type (line 466)
    abs_36785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 24), np_36784, 'abs')
    # Calling abs(args, kwargs) (line 466)
    abs_call_result_36788 = invoke(stypy.reporting.localization.Localization(__file__, 466, 24), abs_36785, *[J_36786], **kwargs_36787)
    
    # Applying the binary operator '+' (line 466)
    result_add_36789 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 12), '+', abs_call_result_36783, abs_call_result_36788)
    
    # Assigning a type to the variable 'M' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'M', result_add_36789)
    
    # Assigning a Num to a Subscript (line 467):
    
    # Assigning a Num to a Subscript (line 467):
    
    # Assigning a Num to a Subscript (line 467):
    float_36790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 37), 'float')
    # Getting the type of 'M' (line 467)
    M_36791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'M')
    
    # Call to diag_indices_from(...): (line 467)
    # Processing the call arguments (line 467)
    # Getting the type of 'M' (line 467)
    M_36794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 31), 'M', False)
    # Processing the call keyword arguments (line 467)
    kwargs_36795 = {}
    # Getting the type of 'np' (line 467)
    np_36792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 10), 'np', False)
    # Obtaining the member 'diag_indices_from' of a type (line 467)
    diag_indices_from_36793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 10), np_36792, 'diag_indices_from')
    # Calling diag_indices_from(args, kwargs) (line 467)
    diag_indices_from_call_result_36796 = invoke(stypy.reporting.localization.Localization(__file__, 467, 10), diag_indices_from_36793, *[M_36794], **kwargs_36795)
    
    # Storing an element on a container (line 467)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 8), M_36791, (diag_indices_from_call_result_36796, float_36790))
    
    # Assigning a Call to a Tuple (line 468):
    
    # Assigning a Subscript to a Name (line 468):
    
    # Assigning a Subscript to a Name (line 468):
    
    # Obtaining the type of the subscript
    int_36797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 8), 'int')
    
    # Call to matrix_balance(...): (line 468)
    # Processing the call arguments (line 468)
    # Getting the type of 'M' (line 468)
    M_36799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 37), 'M', False)
    # Processing the call keyword arguments (line 468)
    int_36800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 49), 'int')
    keyword_36801 = int_36800
    int_36802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 60), 'int')
    keyword_36803 = int_36802
    kwargs_36804 = {'permute': keyword_36803, 'separate': keyword_36801}
    # Getting the type of 'matrix_balance' (line 468)
    matrix_balance_36798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 22), 'matrix_balance', False)
    # Calling matrix_balance(args, kwargs) (line 468)
    matrix_balance_call_result_36805 = invoke(stypy.reporting.localization.Localization(__file__, 468, 22), matrix_balance_36798, *[M_36799], **kwargs_36804)
    
    # Obtaining the member '__getitem__' of a type (line 468)
    getitem___36806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 8), matrix_balance_call_result_36805, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 468)
    subscript_call_result_36807 = invoke(stypy.reporting.localization.Localization(__file__, 468, 8), getitem___36806, int_36797)
    
    # Assigning a type to the variable 'tuple_var_assignment_35893' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'tuple_var_assignment_35893', subscript_call_result_36807)
    
    # Assigning a Subscript to a Name (line 468):
    
    # Assigning a Subscript to a Name (line 468):
    
    # Obtaining the type of the subscript
    int_36808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 8), 'int')
    
    # Call to matrix_balance(...): (line 468)
    # Processing the call arguments (line 468)
    # Getting the type of 'M' (line 468)
    M_36810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 37), 'M', False)
    # Processing the call keyword arguments (line 468)
    int_36811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 49), 'int')
    keyword_36812 = int_36811
    int_36813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 60), 'int')
    keyword_36814 = int_36813
    kwargs_36815 = {'permute': keyword_36814, 'separate': keyword_36812}
    # Getting the type of 'matrix_balance' (line 468)
    matrix_balance_36809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 22), 'matrix_balance', False)
    # Calling matrix_balance(args, kwargs) (line 468)
    matrix_balance_call_result_36816 = invoke(stypy.reporting.localization.Localization(__file__, 468, 22), matrix_balance_36809, *[M_36810], **kwargs_36815)
    
    # Obtaining the member '__getitem__' of a type (line 468)
    getitem___36817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 8), matrix_balance_call_result_36816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 468)
    subscript_call_result_36818 = invoke(stypy.reporting.localization.Localization(__file__, 468, 8), getitem___36817, int_36808)
    
    # Assigning a type to the variable 'tuple_var_assignment_35894' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'tuple_var_assignment_35894', subscript_call_result_36818)
    
    # Assigning a Name to a Name (line 468):
    
    # Assigning a Name to a Name (line 468):
    # Getting the type of 'tuple_var_assignment_35893' (line 468)
    tuple_var_assignment_35893_36819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'tuple_var_assignment_35893')
    # Assigning a type to the variable '_' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), '_', tuple_var_assignment_35893_36819)
    
    # Assigning a Name to a Tuple (line 468):
    
    # Assigning a Subscript to a Name (line 468):
    
    # Obtaining the type of the subscript
    int_36820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 8), 'int')
    # Getting the type of 'tuple_var_assignment_35894' (line 468)
    tuple_var_assignment_35894_36821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'tuple_var_assignment_35894')
    # Obtaining the member '__getitem__' of a type (line 468)
    getitem___36822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 8), tuple_var_assignment_35894_36821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 468)
    subscript_call_result_36823 = invoke(stypy.reporting.localization.Localization(__file__, 468, 8), getitem___36822, int_36820)
    
    # Assigning a type to the variable 'tuple_var_assignment_35935' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'tuple_var_assignment_35935', subscript_call_result_36823)
    
    # Assigning a Subscript to a Name (line 468):
    
    # Obtaining the type of the subscript
    int_36824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 8), 'int')
    # Getting the type of 'tuple_var_assignment_35894' (line 468)
    tuple_var_assignment_35894_36825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'tuple_var_assignment_35894')
    # Obtaining the member '__getitem__' of a type (line 468)
    getitem___36826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 8), tuple_var_assignment_35894_36825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 468)
    subscript_call_result_36827 = invoke(stypy.reporting.localization.Localization(__file__, 468, 8), getitem___36826, int_36824)
    
    # Assigning a type to the variable 'tuple_var_assignment_35936' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'tuple_var_assignment_35936', subscript_call_result_36827)
    
    # Assigning a Name to a Name (line 468):
    # Getting the type of 'tuple_var_assignment_35935' (line 468)
    tuple_var_assignment_35935_36828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'tuple_var_assignment_35935')
    # Assigning a type to the variable 'sca' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'sca', tuple_var_assignment_35935_36828)
    
    # Assigning a Name to a Name (line 468):
    # Getting the type of 'tuple_var_assignment_35936' (line 468)
    tuple_var_assignment_35936_36829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'tuple_var_assignment_35936')
    # Assigning a type to the variable '_' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 17), '_', tuple_var_assignment_35936_36829)
    
    
    
    # Call to allclose(...): (line 470)
    # Processing the call arguments (line 470)
    # Getting the type of 'sca' (line 470)
    sca_36832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 27), 'sca', False)
    
    # Call to ones_like(...): (line 470)
    # Processing the call arguments (line 470)
    # Getting the type of 'sca' (line 470)
    sca_36835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 45), 'sca', False)
    # Processing the call keyword arguments (line 470)
    kwargs_36836 = {}
    # Getting the type of 'np' (line 470)
    np_36833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 32), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 470)
    ones_like_36834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 32), np_36833, 'ones_like')
    # Calling ones_like(args, kwargs) (line 470)
    ones_like_call_result_36837 = invoke(stypy.reporting.localization.Localization(__file__, 470, 32), ones_like_36834, *[sca_36835], **kwargs_36836)
    
    # Processing the call keyword arguments (line 470)
    kwargs_36838 = {}
    # Getting the type of 'np' (line 470)
    np_36830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 15), 'np', False)
    # Obtaining the member 'allclose' of a type (line 470)
    allclose_36831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 15), np_36830, 'allclose')
    # Calling allclose(args, kwargs) (line 470)
    allclose_call_result_36839 = invoke(stypy.reporting.localization.Localization(__file__, 470, 15), allclose_36831, *[sca_36832, ones_like_call_result_36837], **kwargs_36838)
    
    # Applying the 'not' unary operator (line 470)
    result_not__36840 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 11), 'not', allclose_call_result_36839)
    
    # Testing the type of an if condition (line 470)
    if_condition_36841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 470, 8), result_not__36840)
    # Assigning a type to the variable 'if_condition_36841' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'if_condition_36841', if_condition_36841)
    # SSA begins for if statement (line 470)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 473):
    
    # Assigning a Call to a Name (line 473):
    
    # Assigning a Call to a Name (line 473):
    
    # Call to log2(...): (line 473)
    # Processing the call arguments (line 473)
    # Getting the type of 'sca' (line 473)
    sca_36844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 26), 'sca', False)
    # Processing the call keyword arguments (line 473)
    kwargs_36845 = {}
    # Getting the type of 'np' (line 473)
    np_36842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 18), 'np', False)
    # Obtaining the member 'log2' of a type (line 473)
    log2_36843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 18), np_36842, 'log2')
    # Calling log2(args, kwargs) (line 473)
    log2_call_result_36846 = invoke(stypy.reporting.localization.Localization(__file__, 473, 18), log2_36843, *[sca_36844], **kwargs_36845)
    
    # Assigning a type to the variable 'sca' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'sca', log2_call_result_36846)
    
    # Assigning a Call to a Name (line 475):
    
    # Assigning a Call to a Name (line 475):
    
    # Assigning a Call to a Name (line 475):
    
    # Call to round(...): (line 475)
    # Processing the call arguments (line 475)
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 475)
    m_36849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 30), 'm', False)
    int_36850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 32), 'int')
    # Getting the type of 'm' (line 475)
    m_36851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 34), 'm', False)
    # Applying the binary operator '*' (line 475)
    result_mul_36852 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 32), '*', int_36850, m_36851)
    
    slice_36853 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 475, 26), m_36849, result_mul_36852, None)
    # Getting the type of 'sca' (line 475)
    sca_36854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 26), 'sca', False)
    # Obtaining the member '__getitem__' of a type (line 475)
    getitem___36855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 26), sca_36854, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 475)
    subscript_call_result_36856 = invoke(stypy.reporting.localization.Localization(__file__, 475, 26), getitem___36855, slice_36853)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 475)
    m_36857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 44), 'm', False)
    slice_36858 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 475, 39), None, m_36857, None)
    # Getting the type of 'sca' (line 475)
    sca_36859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 39), 'sca', False)
    # Obtaining the member '__getitem__' of a type (line 475)
    getitem___36860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 39), sca_36859, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 475)
    subscript_call_result_36861 = invoke(stypy.reporting.localization.Localization(__file__, 475, 39), getitem___36860, slice_36858)
    
    # Applying the binary operator '-' (line 475)
    result_sub_36862 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 26), '-', subscript_call_result_36856, subscript_call_result_36861)
    
    int_36863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 48), 'int')
    # Applying the binary operator 'div' (line 475)
    result_div_36864 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 25), 'div', result_sub_36862, int_36863)
    
    # Processing the call keyword arguments (line 475)
    kwargs_36865 = {}
    # Getting the type of 'np' (line 475)
    np_36847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 16), 'np', False)
    # Obtaining the member 'round' of a type (line 475)
    round_36848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 16), np_36847, 'round')
    # Calling round(args, kwargs) (line 475)
    round_call_result_36866 = invoke(stypy.reporting.localization.Localization(__file__, 475, 16), round_36848, *[result_div_36864], **kwargs_36865)
    
    # Assigning a type to the variable 's' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 's', round_call_result_36866)
    
    # Assigning a BinOp to a Name (line 476):
    
    # Assigning a BinOp to a Name (line 476):
    
    # Assigning a BinOp to a Name (line 476):
    int_36867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 18), 'int')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 476)
    tuple_36868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 476)
    # Adding element type (line 476)
    # Getting the type of 's' (line 476)
    s_36869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 29), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 29), tuple_36868, s_36869)
    # Adding element type (line 476)
    
    # Getting the type of 's' (line 476)
    s_36870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 33), 's')
    # Applying the 'usub' unary operator (line 476)
    result___neg___36871 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 32), 'usub', s_36870)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 29), tuple_36868, result___neg___36871)
    # Adding element type (line 476)
    
    # Obtaining the type of the subscript
    int_36872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 40), 'int')
    # Getting the type of 'm' (line 476)
    m_36873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 42), 'm')
    # Applying the binary operator '*' (line 476)
    result_mul_36874 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 40), '*', int_36872, m_36873)
    
    slice_36875 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 476, 36), result_mul_36874, None, None)
    # Getting the type of 'sca' (line 476)
    sca_36876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 36), 'sca')
    # Obtaining the member '__getitem__' of a type (line 476)
    getitem___36877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 36), sca_36876, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 476)
    subscript_call_result_36878 = invoke(stypy.reporting.localization.Localization(__file__, 476, 36), getitem___36877, slice_36875)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 29), tuple_36868, subscript_call_result_36878)
    
    # Getting the type of 'np' (line 476)
    np_36879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 23), 'np')
    # Obtaining the member 'r_' of a type (line 476)
    r__36880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 23), np_36879, 'r_')
    # Obtaining the member '__getitem__' of a type (line 476)
    getitem___36881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 23), r__36880, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 476)
    subscript_call_result_36882 = invoke(stypy.reporting.localization.Localization(__file__, 476, 23), getitem___36881, tuple_36868)
    
    # Applying the binary operator '**' (line 476)
    result_pow_36883 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 18), '**', int_36867, subscript_call_result_36882)
    
    # Assigning a type to the variable 'sca' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'sca', result_pow_36883)
    
    # Assigning a BinOp to a Name (line 478):
    
    # Assigning a BinOp to a Name (line 478):
    
    # Assigning a BinOp to a Name (line 478):
    
    # Obtaining the type of the subscript
    slice_36884 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 478, 26), None, None, None)
    # Getting the type of 'None' (line 478)
    None_36885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 33), 'None')
    # Getting the type of 'sca' (line 478)
    sca_36886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 26), 'sca')
    # Obtaining the member '__getitem__' of a type (line 478)
    getitem___36887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 26), sca_36886, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 478)
    subscript_call_result_36888 = invoke(stypy.reporting.localization.Localization(__file__, 478, 26), getitem___36887, (slice_36884, None_36885))
    
    
    # Call to reciprocal(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'sca' (line 478)
    sca_36891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 55), 'sca', False)
    # Processing the call keyword arguments (line 478)
    kwargs_36892 = {}
    # Getting the type of 'np' (line 478)
    np_36889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 41), 'np', False)
    # Obtaining the member 'reciprocal' of a type (line 478)
    reciprocal_36890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 41), np_36889, 'reciprocal')
    # Calling reciprocal(args, kwargs) (line 478)
    reciprocal_call_result_36893 = invoke(stypy.reporting.localization.Localization(__file__, 478, 41), reciprocal_36890, *[sca_36891], **kwargs_36892)
    
    # Applying the binary operator '*' (line 478)
    result_mul_36894 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 26), '*', subscript_call_result_36888, reciprocal_call_result_36893)
    
    # Assigning a type to the variable 'elwisescale' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'elwisescale', result_mul_36894)
    
    # Getting the type of 'H' (line 479)
    H_36895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'H')
    # Getting the type of 'elwisescale' (line 479)
    elwisescale_36896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 17), 'elwisescale')
    # Applying the binary operator '*=' (line 479)
    result_imul_36897 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 12), '*=', H_36895, elwisescale_36896)
    # Assigning a type to the variable 'H' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'H', result_imul_36897)
    
    
    # Getting the type of 'J' (line 480)
    J_36898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'J')
    # Getting the type of 'elwisescale' (line 480)
    elwisescale_36899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 17), 'elwisescale')
    # Applying the binary operator '*=' (line 480)
    result_imul_36900 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 12), '*=', J_36898, elwisescale_36899)
    # Assigning a type to the variable 'J' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'J', result_imul_36900)
    
    # SSA join for if statement (line 470)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 463)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 483):
    
    # Assigning a Subscript to a Name (line 483):
    
    # Assigning a Subscript to a Name (line 483):
    
    # Obtaining the type of the subscript
    int_36901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 4), 'int')
    
    # Call to qr(...): (line 483)
    # Processing the call arguments (line 483)
    
    # Obtaining the type of the subscript
    slice_36903 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 483, 14), None, None, None)
    
    # Getting the type of 'n' (line 483)
    n_36904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 20), 'n', False)
    # Applying the 'usub' unary operator (line 483)
    result___neg___36905 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 19), 'usub', n_36904)
    
    slice_36906 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 483, 14), result___neg___36905, None, None)
    # Getting the type of 'H' (line 483)
    H_36907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 14), 'H', False)
    # Obtaining the member '__getitem__' of a type (line 483)
    getitem___36908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 14), H_36907, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 483)
    subscript_call_result_36909 = invoke(stypy.reporting.localization.Localization(__file__, 483, 14), getitem___36908, (slice_36903, slice_36906))
    
    # Processing the call keyword arguments (line 483)
    kwargs_36910 = {}
    # Getting the type of 'qr' (line 483)
    qr_36902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 11), 'qr', False)
    # Calling qr(args, kwargs) (line 483)
    qr_call_result_36911 = invoke(stypy.reporting.localization.Localization(__file__, 483, 11), qr_36902, *[subscript_call_result_36909], **kwargs_36910)
    
    # Obtaining the member '__getitem__' of a type (line 483)
    getitem___36912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 4), qr_call_result_36911, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 483)
    subscript_call_result_36913 = invoke(stypy.reporting.localization.Localization(__file__, 483, 4), getitem___36912, int_36901)
    
    # Assigning a type to the variable 'tuple_var_assignment_35895' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'tuple_var_assignment_35895', subscript_call_result_36913)
    
    # Assigning a Subscript to a Name (line 483):
    
    # Assigning a Subscript to a Name (line 483):
    
    # Obtaining the type of the subscript
    int_36914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 4), 'int')
    
    # Call to qr(...): (line 483)
    # Processing the call arguments (line 483)
    
    # Obtaining the type of the subscript
    slice_36916 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 483, 14), None, None, None)
    
    # Getting the type of 'n' (line 483)
    n_36917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 20), 'n', False)
    # Applying the 'usub' unary operator (line 483)
    result___neg___36918 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 19), 'usub', n_36917)
    
    slice_36919 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 483, 14), result___neg___36918, None, None)
    # Getting the type of 'H' (line 483)
    H_36920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 14), 'H', False)
    # Obtaining the member '__getitem__' of a type (line 483)
    getitem___36921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 14), H_36920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 483)
    subscript_call_result_36922 = invoke(stypy.reporting.localization.Localization(__file__, 483, 14), getitem___36921, (slice_36916, slice_36919))
    
    # Processing the call keyword arguments (line 483)
    kwargs_36923 = {}
    # Getting the type of 'qr' (line 483)
    qr_36915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 11), 'qr', False)
    # Calling qr(args, kwargs) (line 483)
    qr_call_result_36924 = invoke(stypy.reporting.localization.Localization(__file__, 483, 11), qr_36915, *[subscript_call_result_36922], **kwargs_36923)
    
    # Obtaining the member '__getitem__' of a type (line 483)
    getitem___36925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 4), qr_call_result_36924, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 483)
    subscript_call_result_36926 = invoke(stypy.reporting.localization.Localization(__file__, 483, 4), getitem___36925, int_36914)
    
    # Assigning a type to the variable 'tuple_var_assignment_35896' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'tuple_var_assignment_35896', subscript_call_result_36926)
    
    # Assigning a Name to a Name (line 483):
    
    # Assigning a Name to a Name (line 483):
    # Getting the type of 'tuple_var_assignment_35895' (line 483)
    tuple_var_assignment_35895_36927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'tuple_var_assignment_35895')
    # Assigning a type to the variable 'q' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'q', tuple_var_assignment_35895_36927)
    
    # Assigning a Name to a Name (line 483):
    
    # Assigning a Name to a Name (line 483):
    # Getting the type of 'tuple_var_assignment_35896' (line 483)
    tuple_var_assignment_35896_36928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'tuple_var_assignment_35896')
    # Assigning a type to the variable 'r' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 7), 'r', tuple_var_assignment_35896_36928)
    
    # Assigning a Call to a Name (line 484):
    
    # Assigning a Call to a Name (line 484):
    
    # Assigning a Call to a Name (line 484):
    
    # Call to dot(...): (line 484)
    # Processing the call arguments (line 484)
    
    # Obtaining the type of the subscript
    slice_36940 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 484, 30), None, None, None)
    int_36941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 36), 'int')
    # Getting the type of 'm' (line 484)
    m_36942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 38), 'm', False)
    # Applying the binary operator '*' (line 484)
    result_mul_36943 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 36), '*', int_36941, m_36942)
    
    slice_36944 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 484, 30), None, result_mul_36943, None)
    # Getting the type of 'H' (line 484)
    H_36945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 30), 'H', False)
    # Obtaining the member '__getitem__' of a type (line 484)
    getitem___36946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 30), H_36945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 484)
    subscript_call_result_36947 = invoke(stypy.reporting.localization.Localization(__file__, 484, 30), getitem___36946, (slice_36940, slice_36944))
    
    # Processing the call keyword arguments (line 484)
    kwargs_36948 = {}
    
    # Call to conj(...): (line 484)
    # Processing the call keyword arguments (line 484)
    kwargs_36936 = {}
    
    # Obtaining the type of the subscript
    slice_36929 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 484, 8), None, None, None)
    # Getting the type of 'n' (line 484)
    n_36930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 13), 'n', False)
    slice_36931 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 484, 8), n_36930, None, None)
    # Getting the type of 'q' (line 484)
    q_36932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'q', False)
    # Obtaining the member '__getitem__' of a type (line 484)
    getitem___36933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 8), q_36932, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 484)
    subscript_call_result_36934 = invoke(stypy.reporting.localization.Localization(__file__, 484, 8), getitem___36933, (slice_36929, slice_36931))
    
    # Obtaining the member 'conj' of a type (line 484)
    conj_36935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 8), subscript_call_result_36934, 'conj')
    # Calling conj(args, kwargs) (line 484)
    conj_call_result_36937 = invoke(stypy.reporting.localization.Localization(__file__, 484, 8), conj_36935, *[], **kwargs_36936)
    
    # Obtaining the member 'T' of a type (line 484)
    T_36938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 8), conj_call_result_36937, 'T')
    # Obtaining the member 'dot' of a type (line 484)
    dot_36939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 8), T_36938, 'dot')
    # Calling dot(args, kwargs) (line 484)
    dot_call_result_36949 = invoke(stypy.reporting.localization.Localization(__file__, 484, 8), dot_36939, *[subscript_call_result_36947], **kwargs_36948)
    
    # Assigning a type to the variable 'H' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'H', dot_call_result_36949)
    
    # Assigning a Call to a Name (line 485):
    
    # Assigning a Call to a Name (line 485):
    
    # Assigning a Call to a Name (line 485):
    
    # Call to dot(...): (line 485)
    # Processing the call arguments (line 485)
    
    # Obtaining the type of the subscript
    int_36964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 36), 'int')
    # Getting the type of 'm' (line 485)
    m_36965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 38), 'm', False)
    # Applying the binary operator '*' (line 485)
    result_mul_36966 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 36), '*', int_36964, m_36965)
    
    slice_36967 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 485, 33), None, result_mul_36966, None)
    int_36968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 42), 'int')
    # Getting the type of 'm' (line 485)
    m_36969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 44), 'm', False)
    # Applying the binary operator '*' (line 485)
    result_mul_36970 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 42), '*', int_36968, m_36969)
    
    slice_36971 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 485, 33), None, result_mul_36970, None)
    # Getting the type of 'J' (line 485)
    J_36972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 33), 'J', False)
    # Obtaining the member '__getitem__' of a type (line 485)
    getitem___36973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 33), J_36972, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 485)
    subscript_call_result_36974 = invoke(stypy.reporting.localization.Localization(__file__, 485, 33), getitem___36973, (slice_36967, slice_36971))
    
    # Processing the call keyword arguments (line 485)
    kwargs_36975 = {}
    
    # Call to conj(...): (line 485)
    # Processing the call keyword arguments (line 485)
    kwargs_36960 = {}
    
    # Obtaining the type of the subscript
    int_36950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 11), 'int')
    # Getting the type of 'm' (line 485)
    m_36951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 13), 'm', False)
    # Applying the binary operator '*' (line 485)
    result_mul_36952 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 11), '*', int_36950, m_36951)
    
    slice_36953 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 485, 8), None, result_mul_36952, None)
    # Getting the type of 'n' (line 485)
    n_36954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 16), 'n', False)
    slice_36955 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 485, 8), n_36954, None, None)
    # Getting the type of 'q' (line 485)
    q_36956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'q', False)
    # Obtaining the member '__getitem__' of a type (line 485)
    getitem___36957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 8), q_36956, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 485)
    subscript_call_result_36958 = invoke(stypy.reporting.localization.Localization(__file__, 485, 8), getitem___36957, (slice_36953, slice_36955))
    
    # Obtaining the member 'conj' of a type (line 485)
    conj_36959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 8), subscript_call_result_36958, 'conj')
    # Calling conj(args, kwargs) (line 485)
    conj_call_result_36961 = invoke(stypy.reporting.localization.Localization(__file__, 485, 8), conj_36959, *[], **kwargs_36960)
    
    # Obtaining the member 'T' of a type (line 485)
    T_36962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 8), conj_call_result_36961, 'T')
    # Obtaining the member 'dot' of a type (line 485)
    dot_36963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 8), T_36962, 'dot')
    # Calling dot(args, kwargs) (line 485)
    dot_call_result_36976 = invoke(stypy.reporting.localization.Localization(__file__, 485, 8), dot_36963, *[subscript_call_result_36974], **kwargs_36975)
    
    # Assigning a type to the variable 'J' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'J', dot_call_result_36976)
    
    # Assigning a IfExp to a Name (line 488):
    
    # Assigning a IfExp to a Name (line 488):
    
    # Assigning a IfExp to a Name (line 488):
    
    
    # Getting the type of 'r_or_c' (line 488)
    r_or_c_36977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 24), 'r_or_c')
    # Getting the type of 'float' (line 488)
    float_36978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 34), 'float')
    # Applying the binary operator '==' (line 488)
    result_eq_36979 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 24), '==', r_or_c_36977, float_36978)
    
    # Testing the type of an if expression (line 488)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 488, 14), result_eq_36979)
    # SSA begins for if expression (line 488)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    str_36980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 14), 'str', 'real')
    # SSA branch for the else part of an if expression (line 488)
    module_type_store.open_ssa_branch('if expression else')
    str_36981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 45), 'str', 'complex')
    # SSA join for if expression (line 488)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_36982 = union_type.UnionType.add(str_36980, str_36981)
    
    # Assigning a type to the variable 'out_str' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'out_str', if_exp_36982)
    
    # Assigning a Call to a Tuple (line 490):
    
    # Assigning a Subscript to a Name (line 490):
    
    # Assigning a Subscript to a Name (line 490):
    
    # Obtaining the type of the subscript
    int_36983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 4), 'int')
    
    # Call to ordqz(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'H' (line 490)
    H_36985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 29), 'H', False)
    # Getting the type of 'J' (line 490)
    J_36986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 32), 'J', False)
    # Processing the call keyword arguments (line 490)
    str_36987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 40), 'str', 'lhp')
    keyword_36988 = str_36987
    # Getting the type of 'True' (line 490)
    True_36989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 59), 'True', False)
    keyword_36990 = True_36989
    # Getting the type of 'True' (line 491)
    True_36991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 41), 'True', False)
    keyword_36992 = True_36991
    # Getting the type of 'False' (line 491)
    False_36993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 60), 'False', False)
    keyword_36994 = False_36993
    # Getting the type of 'out_str' (line 492)
    out_str_36995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 36), 'out_str', False)
    keyword_36996 = out_str_36995
    kwargs_36997 = {'sort': keyword_36988, 'output': keyword_36996, 'overwrite_a': keyword_36990, 'check_finite': keyword_36994, 'overwrite_b': keyword_36992}
    # Getting the type of 'ordqz' (line 490)
    ordqz_36984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 'ordqz', False)
    # Calling ordqz(args, kwargs) (line 490)
    ordqz_call_result_36998 = invoke(stypy.reporting.localization.Localization(__file__, 490, 23), ordqz_36984, *[H_36985, J_36986], **kwargs_36997)
    
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___36999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 4), ordqz_call_result_36998, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 490)
    subscript_call_result_37000 = invoke(stypy.reporting.localization.Localization(__file__, 490, 4), getitem___36999, int_36983)
    
    # Assigning a type to the variable 'tuple_var_assignment_35897' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'tuple_var_assignment_35897', subscript_call_result_37000)
    
    # Assigning a Subscript to a Name (line 490):
    
    # Assigning a Subscript to a Name (line 490):
    
    # Obtaining the type of the subscript
    int_37001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 4), 'int')
    
    # Call to ordqz(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'H' (line 490)
    H_37003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 29), 'H', False)
    # Getting the type of 'J' (line 490)
    J_37004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 32), 'J', False)
    # Processing the call keyword arguments (line 490)
    str_37005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 40), 'str', 'lhp')
    keyword_37006 = str_37005
    # Getting the type of 'True' (line 490)
    True_37007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 59), 'True', False)
    keyword_37008 = True_37007
    # Getting the type of 'True' (line 491)
    True_37009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 41), 'True', False)
    keyword_37010 = True_37009
    # Getting the type of 'False' (line 491)
    False_37011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 60), 'False', False)
    keyword_37012 = False_37011
    # Getting the type of 'out_str' (line 492)
    out_str_37013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 36), 'out_str', False)
    keyword_37014 = out_str_37013
    kwargs_37015 = {'sort': keyword_37006, 'output': keyword_37014, 'overwrite_a': keyword_37008, 'check_finite': keyword_37012, 'overwrite_b': keyword_37010}
    # Getting the type of 'ordqz' (line 490)
    ordqz_37002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 'ordqz', False)
    # Calling ordqz(args, kwargs) (line 490)
    ordqz_call_result_37016 = invoke(stypy.reporting.localization.Localization(__file__, 490, 23), ordqz_37002, *[H_37003, J_37004], **kwargs_37015)
    
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___37017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 4), ordqz_call_result_37016, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 490)
    subscript_call_result_37018 = invoke(stypy.reporting.localization.Localization(__file__, 490, 4), getitem___37017, int_37001)
    
    # Assigning a type to the variable 'tuple_var_assignment_35898' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'tuple_var_assignment_35898', subscript_call_result_37018)
    
    # Assigning a Subscript to a Name (line 490):
    
    # Assigning a Subscript to a Name (line 490):
    
    # Obtaining the type of the subscript
    int_37019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 4), 'int')
    
    # Call to ordqz(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'H' (line 490)
    H_37021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 29), 'H', False)
    # Getting the type of 'J' (line 490)
    J_37022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 32), 'J', False)
    # Processing the call keyword arguments (line 490)
    str_37023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 40), 'str', 'lhp')
    keyword_37024 = str_37023
    # Getting the type of 'True' (line 490)
    True_37025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 59), 'True', False)
    keyword_37026 = True_37025
    # Getting the type of 'True' (line 491)
    True_37027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 41), 'True', False)
    keyword_37028 = True_37027
    # Getting the type of 'False' (line 491)
    False_37029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 60), 'False', False)
    keyword_37030 = False_37029
    # Getting the type of 'out_str' (line 492)
    out_str_37031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 36), 'out_str', False)
    keyword_37032 = out_str_37031
    kwargs_37033 = {'sort': keyword_37024, 'output': keyword_37032, 'overwrite_a': keyword_37026, 'check_finite': keyword_37030, 'overwrite_b': keyword_37028}
    # Getting the type of 'ordqz' (line 490)
    ordqz_37020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 'ordqz', False)
    # Calling ordqz(args, kwargs) (line 490)
    ordqz_call_result_37034 = invoke(stypy.reporting.localization.Localization(__file__, 490, 23), ordqz_37020, *[H_37021, J_37022], **kwargs_37033)
    
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___37035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 4), ordqz_call_result_37034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 490)
    subscript_call_result_37036 = invoke(stypy.reporting.localization.Localization(__file__, 490, 4), getitem___37035, int_37019)
    
    # Assigning a type to the variable 'tuple_var_assignment_35899' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'tuple_var_assignment_35899', subscript_call_result_37036)
    
    # Assigning a Subscript to a Name (line 490):
    
    # Assigning a Subscript to a Name (line 490):
    
    # Obtaining the type of the subscript
    int_37037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 4), 'int')
    
    # Call to ordqz(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'H' (line 490)
    H_37039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 29), 'H', False)
    # Getting the type of 'J' (line 490)
    J_37040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 32), 'J', False)
    # Processing the call keyword arguments (line 490)
    str_37041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 40), 'str', 'lhp')
    keyword_37042 = str_37041
    # Getting the type of 'True' (line 490)
    True_37043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 59), 'True', False)
    keyword_37044 = True_37043
    # Getting the type of 'True' (line 491)
    True_37045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 41), 'True', False)
    keyword_37046 = True_37045
    # Getting the type of 'False' (line 491)
    False_37047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 60), 'False', False)
    keyword_37048 = False_37047
    # Getting the type of 'out_str' (line 492)
    out_str_37049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 36), 'out_str', False)
    keyword_37050 = out_str_37049
    kwargs_37051 = {'sort': keyword_37042, 'output': keyword_37050, 'overwrite_a': keyword_37044, 'check_finite': keyword_37048, 'overwrite_b': keyword_37046}
    # Getting the type of 'ordqz' (line 490)
    ordqz_37038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 'ordqz', False)
    # Calling ordqz(args, kwargs) (line 490)
    ordqz_call_result_37052 = invoke(stypy.reporting.localization.Localization(__file__, 490, 23), ordqz_37038, *[H_37039, J_37040], **kwargs_37051)
    
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___37053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 4), ordqz_call_result_37052, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 490)
    subscript_call_result_37054 = invoke(stypy.reporting.localization.Localization(__file__, 490, 4), getitem___37053, int_37037)
    
    # Assigning a type to the variable 'tuple_var_assignment_35900' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'tuple_var_assignment_35900', subscript_call_result_37054)
    
    # Assigning a Subscript to a Name (line 490):
    
    # Assigning a Subscript to a Name (line 490):
    
    # Obtaining the type of the subscript
    int_37055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 4), 'int')
    
    # Call to ordqz(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'H' (line 490)
    H_37057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 29), 'H', False)
    # Getting the type of 'J' (line 490)
    J_37058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 32), 'J', False)
    # Processing the call keyword arguments (line 490)
    str_37059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 40), 'str', 'lhp')
    keyword_37060 = str_37059
    # Getting the type of 'True' (line 490)
    True_37061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 59), 'True', False)
    keyword_37062 = True_37061
    # Getting the type of 'True' (line 491)
    True_37063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 41), 'True', False)
    keyword_37064 = True_37063
    # Getting the type of 'False' (line 491)
    False_37065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 60), 'False', False)
    keyword_37066 = False_37065
    # Getting the type of 'out_str' (line 492)
    out_str_37067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 36), 'out_str', False)
    keyword_37068 = out_str_37067
    kwargs_37069 = {'sort': keyword_37060, 'output': keyword_37068, 'overwrite_a': keyword_37062, 'check_finite': keyword_37066, 'overwrite_b': keyword_37064}
    # Getting the type of 'ordqz' (line 490)
    ordqz_37056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 'ordqz', False)
    # Calling ordqz(args, kwargs) (line 490)
    ordqz_call_result_37070 = invoke(stypy.reporting.localization.Localization(__file__, 490, 23), ordqz_37056, *[H_37057, J_37058], **kwargs_37069)
    
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___37071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 4), ordqz_call_result_37070, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 490)
    subscript_call_result_37072 = invoke(stypy.reporting.localization.Localization(__file__, 490, 4), getitem___37071, int_37055)
    
    # Assigning a type to the variable 'tuple_var_assignment_35901' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'tuple_var_assignment_35901', subscript_call_result_37072)
    
    # Assigning a Subscript to a Name (line 490):
    
    # Assigning a Subscript to a Name (line 490):
    
    # Obtaining the type of the subscript
    int_37073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 4), 'int')
    
    # Call to ordqz(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'H' (line 490)
    H_37075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 29), 'H', False)
    # Getting the type of 'J' (line 490)
    J_37076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 32), 'J', False)
    # Processing the call keyword arguments (line 490)
    str_37077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 40), 'str', 'lhp')
    keyword_37078 = str_37077
    # Getting the type of 'True' (line 490)
    True_37079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 59), 'True', False)
    keyword_37080 = True_37079
    # Getting the type of 'True' (line 491)
    True_37081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 41), 'True', False)
    keyword_37082 = True_37081
    # Getting the type of 'False' (line 491)
    False_37083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 60), 'False', False)
    keyword_37084 = False_37083
    # Getting the type of 'out_str' (line 492)
    out_str_37085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 36), 'out_str', False)
    keyword_37086 = out_str_37085
    kwargs_37087 = {'sort': keyword_37078, 'output': keyword_37086, 'overwrite_a': keyword_37080, 'check_finite': keyword_37084, 'overwrite_b': keyword_37082}
    # Getting the type of 'ordqz' (line 490)
    ordqz_37074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 'ordqz', False)
    # Calling ordqz(args, kwargs) (line 490)
    ordqz_call_result_37088 = invoke(stypy.reporting.localization.Localization(__file__, 490, 23), ordqz_37074, *[H_37075, J_37076], **kwargs_37087)
    
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___37089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 4), ordqz_call_result_37088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 490)
    subscript_call_result_37090 = invoke(stypy.reporting.localization.Localization(__file__, 490, 4), getitem___37089, int_37073)
    
    # Assigning a type to the variable 'tuple_var_assignment_35902' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'tuple_var_assignment_35902', subscript_call_result_37090)
    
    # Assigning a Name to a Name (line 490):
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'tuple_var_assignment_35897' (line 490)
    tuple_var_assignment_35897_37091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'tuple_var_assignment_35897')
    # Assigning a type to the variable '_' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), '_', tuple_var_assignment_35897_37091)
    
    # Assigning a Name to a Name (line 490):
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'tuple_var_assignment_35898' (line 490)
    tuple_var_assignment_35898_37092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'tuple_var_assignment_35898')
    # Assigning a type to the variable '_' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 7), '_', tuple_var_assignment_35898_37092)
    
    # Assigning a Name to a Name (line 490):
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'tuple_var_assignment_35899' (line 490)
    tuple_var_assignment_35899_37093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'tuple_var_assignment_35899')
    # Assigning a type to the variable '_' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 10), '_', tuple_var_assignment_35899_37093)
    
    # Assigning a Name to a Name (line 490):
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'tuple_var_assignment_35900' (line 490)
    tuple_var_assignment_35900_37094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'tuple_var_assignment_35900')
    # Assigning a type to the variable '_' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 13), '_', tuple_var_assignment_35900_37094)
    
    # Assigning a Name to a Name (line 490):
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'tuple_var_assignment_35901' (line 490)
    tuple_var_assignment_35901_37095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'tuple_var_assignment_35901')
    # Assigning a type to the variable '_' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 16), '_', tuple_var_assignment_35901_37095)
    
    # Assigning a Name to a Name (line 490):
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'tuple_var_assignment_35902' (line 490)
    tuple_var_assignment_35902_37096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'tuple_var_assignment_35902')
    # Assigning a type to the variable 'u' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 19), 'u', tuple_var_assignment_35902_37096)
    
    # Type idiom detected: calculating its left and rigth part (line 495)
    # Getting the type of 'e' (line 495)
    e_37097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'e')
    # Getting the type of 'None' (line 495)
    None_37098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 16), 'None')
    
    (may_be_37099, more_types_in_union_37100) = may_not_be_none(e_37097, None_37098)

    if may_be_37099:

        if more_types_in_union_37100:
            # Runtime conditional SSA (line 495)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Tuple (line 496):
        
        # Assigning a Subscript to a Name (line 496):
        
        # Assigning a Subscript to a Name (line 496):
        
        # Obtaining the type of the subscript
        int_37101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 8), 'int')
        
        # Call to qr(...): (line 496)
        # Processing the call arguments (line 496)
        
        # Call to vstack(...): (line 496)
        # Processing the call arguments (line 496)
        
        # Obtaining an instance of the builtin type 'tuple' (line 496)
        tuple_37105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 496)
        # Adding element type (line 496)
        
        # Call to dot(...): (line 496)
        # Processing the call arguments (line 496)
        
        # Obtaining the type of the subscript
        # Getting the type of 'm' (line 496)
        m_37108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 38), 'm', False)
        slice_37109 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 496, 35), None, m_37108, None)
        # Getting the type of 'm' (line 496)
        m_37110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 42), 'm', False)
        slice_37111 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 496, 35), None, m_37110, None)
        # Getting the type of 'u' (line 496)
        u_37112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 35), 'u', False)
        # Obtaining the member '__getitem__' of a type (line 496)
        getitem___37113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 35), u_37112, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 496)
        subscript_call_result_37114 = invoke(stypy.reporting.localization.Localization(__file__, 496, 35), getitem___37113, (slice_37109, slice_37111))
        
        # Processing the call keyword arguments (line 496)
        kwargs_37115 = {}
        # Getting the type of 'e' (line 496)
        e_37106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 29), 'e', False)
        # Obtaining the member 'dot' of a type (line 496)
        dot_37107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 29), e_37106, 'dot')
        # Calling dot(args, kwargs) (line 496)
        dot_call_result_37116 = invoke(stypy.reporting.localization.Localization(__file__, 496, 29), dot_37107, *[subscript_call_result_37114], **kwargs_37115)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 29), tuple_37105, dot_call_result_37116)
        # Adding element type (line 496)
        
        # Obtaining the type of the subscript
        # Getting the type of 'm' (line 496)
        m_37117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 49), 'm', False)
        slice_37118 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 496, 47), m_37117, None, None)
        # Getting the type of 'm' (line 496)
        m_37119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 54), 'm', False)
        slice_37120 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 496, 47), None, m_37119, None)
        # Getting the type of 'u' (line 496)
        u_37121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 47), 'u', False)
        # Obtaining the member '__getitem__' of a type (line 496)
        getitem___37122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 47), u_37121, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 496)
        subscript_call_result_37123 = invoke(stypy.reporting.localization.Localization(__file__, 496, 47), getitem___37122, (slice_37118, slice_37120))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 29), tuple_37105, subscript_call_result_37123)
        
        # Processing the call keyword arguments (line 496)
        kwargs_37124 = {}
        # Getting the type of 'np' (line 496)
        np_37103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 18), 'np', False)
        # Obtaining the member 'vstack' of a type (line 496)
        vstack_37104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 18), np_37103, 'vstack')
        # Calling vstack(args, kwargs) (line 496)
        vstack_call_result_37125 = invoke(stypy.reporting.localization.Localization(__file__, 496, 18), vstack_37104, *[tuple_37105], **kwargs_37124)
        
        # Processing the call keyword arguments (line 496)
        kwargs_37126 = {}
        # Getting the type of 'qr' (line 496)
        qr_37102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'qr', False)
        # Calling qr(args, kwargs) (line 496)
        qr_call_result_37127 = invoke(stypy.reporting.localization.Localization(__file__, 496, 15), qr_37102, *[vstack_call_result_37125], **kwargs_37126)
        
        # Obtaining the member '__getitem__' of a type (line 496)
        getitem___37128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 8), qr_call_result_37127, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 496)
        subscript_call_result_37129 = invoke(stypy.reporting.localization.Localization(__file__, 496, 8), getitem___37128, int_37101)
        
        # Assigning a type to the variable 'tuple_var_assignment_35903' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'tuple_var_assignment_35903', subscript_call_result_37129)
        
        # Assigning a Subscript to a Name (line 496):
        
        # Assigning a Subscript to a Name (line 496):
        
        # Obtaining the type of the subscript
        int_37130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 8), 'int')
        
        # Call to qr(...): (line 496)
        # Processing the call arguments (line 496)
        
        # Call to vstack(...): (line 496)
        # Processing the call arguments (line 496)
        
        # Obtaining an instance of the builtin type 'tuple' (line 496)
        tuple_37134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 496)
        # Adding element type (line 496)
        
        # Call to dot(...): (line 496)
        # Processing the call arguments (line 496)
        
        # Obtaining the type of the subscript
        # Getting the type of 'm' (line 496)
        m_37137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 38), 'm', False)
        slice_37138 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 496, 35), None, m_37137, None)
        # Getting the type of 'm' (line 496)
        m_37139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 42), 'm', False)
        slice_37140 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 496, 35), None, m_37139, None)
        # Getting the type of 'u' (line 496)
        u_37141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 35), 'u', False)
        # Obtaining the member '__getitem__' of a type (line 496)
        getitem___37142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 35), u_37141, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 496)
        subscript_call_result_37143 = invoke(stypy.reporting.localization.Localization(__file__, 496, 35), getitem___37142, (slice_37138, slice_37140))
        
        # Processing the call keyword arguments (line 496)
        kwargs_37144 = {}
        # Getting the type of 'e' (line 496)
        e_37135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 29), 'e', False)
        # Obtaining the member 'dot' of a type (line 496)
        dot_37136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 29), e_37135, 'dot')
        # Calling dot(args, kwargs) (line 496)
        dot_call_result_37145 = invoke(stypy.reporting.localization.Localization(__file__, 496, 29), dot_37136, *[subscript_call_result_37143], **kwargs_37144)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 29), tuple_37134, dot_call_result_37145)
        # Adding element type (line 496)
        
        # Obtaining the type of the subscript
        # Getting the type of 'm' (line 496)
        m_37146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 49), 'm', False)
        slice_37147 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 496, 47), m_37146, None, None)
        # Getting the type of 'm' (line 496)
        m_37148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 54), 'm', False)
        slice_37149 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 496, 47), None, m_37148, None)
        # Getting the type of 'u' (line 496)
        u_37150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 47), 'u', False)
        # Obtaining the member '__getitem__' of a type (line 496)
        getitem___37151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 47), u_37150, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 496)
        subscript_call_result_37152 = invoke(stypy.reporting.localization.Localization(__file__, 496, 47), getitem___37151, (slice_37147, slice_37149))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 29), tuple_37134, subscript_call_result_37152)
        
        # Processing the call keyword arguments (line 496)
        kwargs_37153 = {}
        # Getting the type of 'np' (line 496)
        np_37132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 18), 'np', False)
        # Obtaining the member 'vstack' of a type (line 496)
        vstack_37133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 18), np_37132, 'vstack')
        # Calling vstack(args, kwargs) (line 496)
        vstack_call_result_37154 = invoke(stypy.reporting.localization.Localization(__file__, 496, 18), vstack_37133, *[tuple_37134], **kwargs_37153)
        
        # Processing the call keyword arguments (line 496)
        kwargs_37155 = {}
        # Getting the type of 'qr' (line 496)
        qr_37131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'qr', False)
        # Calling qr(args, kwargs) (line 496)
        qr_call_result_37156 = invoke(stypy.reporting.localization.Localization(__file__, 496, 15), qr_37131, *[vstack_call_result_37154], **kwargs_37155)
        
        # Obtaining the member '__getitem__' of a type (line 496)
        getitem___37157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 8), qr_call_result_37156, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 496)
        subscript_call_result_37158 = invoke(stypy.reporting.localization.Localization(__file__, 496, 8), getitem___37157, int_37130)
        
        # Assigning a type to the variable 'tuple_var_assignment_35904' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'tuple_var_assignment_35904', subscript_call_result_37158)
        
        # Assigning a Name to a Name (line 496):
        
        # Assigning a Name to a Name (line 496):
        # Getting the type of 'tuple_var_assignment_35903' (line 496)
        tuple_var_assignment_35903_37159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'tuple_var_assignment_35903')
        # Assigning a type to the variable 'u' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'u', tuple_var_assignment_35903_37159)
        
        # Assigning a Name to a Name (line 496):
        
        # Assigning a Name to a Name (line 496):
        # Getting the type of 'tuple_var_assignment_35904' (line 496)
        tuple_var_assignment_35904_37160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'tuple_var_assignment_35904')
        # Assigning a type to the variable '_' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 11), '_', tuple_var_assignment_35904_37160)

        if more_types_in_union_37100:
            # SSA join for if statement (line 495)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Subscript to a Name (line 497):
    
    # Assigning a Subscript to a Name (line 497):
    
    # Assigning a Subscript to a Name (line 497):
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 497)
    m_37161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 13), 'm')
    slice_37162 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 497, 10), None, m_37161, None)
    # Getting the type of 'm' (line 497)
    m_37163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 17), 'm')
    slice_37164 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 497, 10), None, m_37163, None)
    # Getting the type of 'u' (line 497)
    u_37165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 10), 'u')
    # Obtaining the member '__getitem__' of a type (line 497)
    getitem___37166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 10), u_37165, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 497)
    subscript_call_result_37167 = invoke(stypy.reporting.localization.Localization(__file__, 497, 10), getitem___37166, (slice_37162, slice_37164))
    
    # Assigning a type to the variable 'u00' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'u00', subscript_call_result_37167)
    
    # Assigning a Subscript to a Name (line 498):
    
    # Assigning a Subscript to a Name (line 498):
    
    # Assigning a Subscript to a Name (line 498):
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 498)
    m_37168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'm')
    slice_37169 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 498, 10), m_37168, None, None)
    # Getting the type of 'm' (line 498)
    m_37170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 17), 'm')
    slice_37171 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 498, 10), None, m_37170, None)
    # Getting the type of 'u' (line 498)
    u_37172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 10), 'u')
    # Obtaining the member '__getitem__' of a type (line 498)
    getitem___37173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 10), u_37172, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 498)
    subscript_call_result_37174 = invoke(stypy.reporting.localization.Localization(__file__, 498, 10), getitem___37173, (slice_37169, slice_37171))
    
    # Assigning a type to the variable 'u10' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'u10', subscript_call_result_37174)
    
    # Assigning a Call to a Tuple (line 501):
    
    # Assigning a Subscript to a Name (line 501):
    
    # Assigning a Subscript to a Name (line 501):
    
    # Obtaining the type of the subscript
    int_37175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 4), 'int')
    
    # Call to lu(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'u00' (line 501)
    u00_37177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 20), 'u00', False)
    # Processing the call keyword arguments (line 501)
    kwargs_37178 = {}
    # Getting the type of 'lu' (line 501)
    lu_37176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 17), 'lu', False)
    # Calling lu(args, kwargs) (line 501)
    lu_call_result_37179 = invoke(stypy.reporting.localization.Localization(__file__, 501, 17), lu_37176, *[u00_37177], **kwargs_37178)
    
    # Obtaining the member '__getitem__' of a type (line 501)
    getitem___37180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 4), lu_call_result_37179, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 501)
    subscript_call_result_37181 = invoke(stypy.reporting.localization.Localization(__file__, 501, 4), getitem___37180, int_37175)
    
    # Assigning a type to the variable 'tuple_var_assignment_35905' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'tuple_var_assignment_35905', subscript_call_result_37181)
    
    # Assigning a Subscript to a Name (line 501):
    
    # Assigning a Subscript to a Name (line 501):
    
    # Obtaining the type of the subscript
    int_37182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 4), 'int')
    
    # Call to lu(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'u00' (line 501)
    u00_37184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 20), 'u00', False)
    # Processing the call keyword arguments (line 501)
    kwargs_37185 = {}
    # Getting the type of 'lu' (line 501)
    lu_37183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 17), 'lu', False)
    # Calling lu(args, kwargs) (line 501)
    lu_call_result_37186 = invoke(stypy.reporting.localization.Localization(__file__, 501, 17), lu_37183, *[u00_37184], **kwargs_37185)
    
    # Obtaining the member '__getitem__' of a type (line 501)
    getitem___37187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 4), lu_call_result_37186, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 501)
    subscript_call_result_37188 = invoke(stypy.reporting.localization.Localization(__file__, 501, 4), getitem___37187, int_37182)
    
    # Assigning a type to the variable 'tuple_var_assignment_35906' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'tuple_var_assignment_35906', subscript_call_result_37188)
    
    # Assigning a Subscript to a Name (line 501):
    
    # Assigning a Subscript to a Name (line 501):
    
    # Obtaining the type of the subscript
    int_37189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 4), 'int')
    
    # Call to lu(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'u00' (line 501)
    u00_37191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 20), 'u00', False)
    # Processing the call keyword arguments (line 501)
    kwargs_37192 = {}
    # Getting the type of 'lu' (line 501)
    lu_37190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 17), 'lu', False)
    # Calling lu(args, kwargs) (line 501)
    lu_call_result_37193 = invoke(stypy.reporting.localization.Localization(__file__, 501, 17), lu_37190, *[u00_37191], **kwargs_37192)
    
    # Obtaining the member '__getitem__' of a type (line 501)
    getitem___37194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 4), lu_call_result_37193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 501)
    subscript_call_result_37195 = invoke(stypy.reporting.localization.Localization(__file__, 501, 4), getitem___37194, int_37189)
    
    # Assigning a type to the variable 'tuple_var_assignment_35907' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'tuple_var_assignment_35907', subscript_call_result_37195)
    
    # Assigning a Name to a Name (line 501):
    
    # Assigning a Name to a Name (line 501):
    # Getting the type of 'tuple_var_assignment_35905' (line 501)
    tuple_var_assignment_35905_37196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'tuple_var_assignment_35905')
    # Assigning a type to the variable 'up' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'up', tuple_var_assignment_35905_37196)
    
    # Assigning a Name to a Name (line 501):
    
    # Assigning a Name to a Name (line 501):
    # Getting the type of 'tuple_var_assignment_35906' (line 501)
    tuple_var_assignment_35906_37197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'tuple_var_assignment_35906')
    # Assigning a type to the variable 'ul' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'ul', tuple_var_assignment_35906_37197)
    
    # Assigning a Name to a Name (line 501):
    
    # Assigning a Name to a Name (line 501):
    # Getting the type of 'tuple_var_assignment_35907' (line 501)
    tuple_var_assignment_35907_37198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'tuple_var_assignment_35907')
    # Assigning a type to the variable 'uu' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'uu', tuple_var_assignment_35907_37198)
    
    
    int_37199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 7), 'int')
    
    # Call to cond(...): (line 502)
    # Processing the call arguments (line 502)
    # Getting the type of 'uu' (line 502)
    uu_37201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 14), 'uu', False)
    # Processing the call keyword arguments (line 502)
    kwargs_37202 = {}
    # Getting the type of 'cond' (line 502)
    cond_37200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 9), 'cond', False)
    # Calling cond(args, kwargs) (line 502)
    cond_call_result_37203 = invoke(stypy.reporting.localization.Localization(__file__, 502, 9), cond_37200, *[uu_37201], **kwargs_37202)
    
    # Applying the binary operator 'div' (line 502)
    result_div_37204 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 7), 'div', int_37199, cond_call_result_37203)
    
    
    # Call to spacing(...): (line 502)
    # Processing the call arguments (line 502)
    float_37207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 31), 'float')
    # Processing the call keyword arguments (line 502)
    kwargs_37208 = {}
    # Getting the type of 'np' (line 502)
    np_37205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 20), 'np', False)
    # Obtaining the member 'spacing' of a type (line 502)
    spacing_37206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 20), np_37205, 'spacing')
    # Calling spacing(args, kwargs) (line 502)
    spacing_call_result_37209 = invoke(stypy.reporting.localization.Localization(__file__, 502, 20), spacing_37206, *[float_37207], **kwargs_37208)
    
    # Applying the binary operator '<' (line 502)
    result_lt_37210 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 7), '<', result_div_37204, spacing_call_result_37209)
    
    # Testing the type of an if condition (line 502)
    if_condition_37211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 4), result_lt_37210)
    # Assigning a type to the variable 'if_condition_37211' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'if_condition_37211', if_condition_37211)
    # SSA begins for if statement (line 502)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 503)
    # Processing the call arguments (line 503)
    str_37213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 26), 'str', 'Failed to find a finite solution.')
    # Processing the call keyword arguments (line 503)
    kwargs_37214 = {}
    # Getting the type of 'LinAlgError' (line 503)
    LinAlgError_37212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 503)
    LinAlgError_call_result_37215 = invoke(stypy.reporting.localization.Localization(__file__, 503, 14), LinAlgError_37212, *[str_37213], **kwargs_37214)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 503, 8), LinAlgError_call_result_37215, 'raise parameter', BaseException)
    # SSA join for if statement (line 502)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 506):
    
    # Assigning a Call to a Name (line 506):
    
    # Assigning a Call to a Name (line 506):
    
    # Call to dot(...): (line 506)
    # Processing the call arguments (line 506)
    
    # Call to conj(...): (line 511)
    # Processing the call keyword arguments (line 511)
    kwargs_37248 = {}
    # Getting the type of 'up' (line 511)
    up_37246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 40), 'up', False)
    # Obtaining the member 'conj' of a type (line 511)
    conj_37247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 40), up_37246, 'conj')
    # Calling conj(args, kwargs) (line 511)
    conj_call_result_37249 = invoke(stypy.reporting.localization.Localization(__file__, 511, 40), conj_37247, *[], **kwargs_37248)
    
    # Obtaining the member 'T' of a type (line 511)
    T_37250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 40), conj_call_result_37249, 'T')
    # Processing the call keyword arguments (line 506)
    kwargs_37251 = {}
    
    # Call to conj(...): (line 506)
    # Processing the call keyword arguments (line 506)
    kwargs_37242 = {}
    
    # Call to solve_triangular(...): (line 506)
    # Processing the call arguments (line 506)
    
    # Call to conj(...): (line 506)
    # Processing the call keyword arguments (line 506)
    kwargs_37219 = {}
    # Getting the type of 'ul' (line 506)
    ul_37217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 25), 'ul', False)
    # Obtaining the member 'conj' of a type (line 506)
    conj_37218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 25), ul_37217, 'conj')
    # Calling conj(args, kwargs) (line 506)
    conj_call_result_37220 = invoke(stypy.reporting.localization.Localization(__file__, 506, 25), conj_37218, *[], **kwargs_37219)
    
    # Obtaining the member 'T' of a type (line 506)
    T_37221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 25), conj_call_result_37220, 'T')
    
    # Call to solve_triangular(...): (line 507)
    # Processing the call arguments (line 507)
    
    # Call to conj(...): (line 507)
    # Processing the call keyword arguments (line 507)
    kwargs_37225 = {}
    # Getting the type of 'uu' (line 507)
    uu_37223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 42), 'uu', False)
    # Obtaining the member 'conj' of a type (line 507)
    conj_37224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 42), uu_37223, 'conj')
    # Calling conj(args, kwargs) (line 507)
    conj_call_result_37226 = invoke(stypy.reporting.localization.Localization(__file__, 507, 42), conj_37224, *[], **kwargs_37225)
    
    # Obtaining the member 'T' of a type (line 507)
    T_37227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 42), conj_call_result_37226, 'T')
    
    # Call to conj(...): (line 508)
    # Processing the call keyword arguments (line 508)
    kwargs_37230 = {}
    # Getting the type of 'u10' (line 508)
    u10_37228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 42), 'u10', False)
    # Obtaining the member 'conj' of a type (line 508)
    conj_37229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 42), u10_37228, 'conj')
    # Calling conj(args, kwargs) (line 508)
    conj_call_result_37231 = invoke(stypy.reporting.localization.Localization(__file__, 508, 42), conj_37229, *[], **kwargs_37230)
    
    # Obtaining the member 'T' of a type (line 508)
    T_37232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 42), conj_call_result_37231, 'T')
    # Processing the call keyword arguments (line 507)
    # Getting the type of 'True' (line 509)
    True_37233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 48), 'True', False)
    keyword_37234 = True_37233
    kwargs_37235 = {'lower': keyword_37234}
    # Getting the type of 'solve_triangular' (line 507)
    solve_triangular_37222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 25), 'solve_triangular', False)
    # Calling solve_triangular(args, kwargs) (line 507)
    solve_triangular_call_result_37236 = invoke(stypy.reporting.localization.Localization(__file__, 507, 25), solve_triangular_37222, *[T_37227, T_37232], **kwargs_37235)
    
    # Processing the call keyword arguments (line 506)
    # Getting the type of 'True' (line 510)
    True_37237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 39), 'True', False)
    keyword_37238 = True_37237
    kwargs_37239 = {'unit_diagonal': keyword_37238}
    # Getting the type of 'solve_triangular' (line 506)
    solve_triangular_37216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'solve_triangular', False)
    # Calling solve_triangular(args, kwargs) (line 506)
    solve_triangular_call_result_37240 = invoke(stypy.reporting.localization.Localization(__file__, 506, 8), solve_triangular_37216, *[T_37221, solve_triangular_call_result_37236], **kwargs_37239)
    
    # Obtaining the member 'conj' of a type (line 506)
    conj_37241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 8), solve_triangular_call_result_37240, 'conj')
    # Calling conj(args, kwargs) (line 506)
    conj_call_result_37243 = invoke(stypy.reporting.localization.Localization(__file__, 506, 8), conj_37241, *[], **kwargs_37242)
    
    # Obtaining the member 'T' of a type (line 506)
    T_37244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 8), conj_call_result_37243, 'T')
    # Obtaining the member 'dot' of a type (line 506)
    dot_37245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 8), T_37244, 'dot')
    # Calling dot(args, kwargs) (line 506)
    dot_call_result_37252 = invoke(stypy.reporting.localization.Localization(__file__, 506, 8), dot_37245, *[T_37250], **kwargs_37251)
    
    # Assigning a type to the variable 'x' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'x', dot_call_result_37252)
    
    # Getting the type of 'balanced' (line 512)
    balanced_37253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 7), 'balanced')
    # Testing the type of an if condition (line 512)
    if_condition_37254 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 512, 4), balanced_37253)
    # Assigning a type to the variable 'if_condition_37254' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'if_condition_37254', if_condition_37254)
    # SSA begins for if statement (line 512)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'x' (line 513)
    x_37255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'x')
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 513)
    m_37256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 18), 'm')
    slice_37257 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 513, 13), None, m_37256, None)
    # Getting the type of 'None' (line 513)
    None_37258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 21), 'None')
    # Getting the type of 'sca' (line 513)
    sca_37259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 13), 'sca')
    # Obtaining the member '__getitem__' of a type (line 513)
    getitem___37260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 13), sca_37259, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 513)
    subscript_call_result_37261 = invoke(stypy.reporting.localization.Localization(__file__, 513, 13), getitem___37260, (slice_37257, None_37258))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 513)
    m_37262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 34), 'm')
    slice_37263 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 513, 29), None, m_37262, None)
    # Getting the type of 'sca' (line 513)
    sca_37264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 29), 'sca')
    # Obtaining the member '__getitem__' of a type (line 513)
    getitem___37265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 29), sca_37264, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 513)
    subscript_call_result_37266 = invoke(stypy.reporting.localization.Localization(__file__, 513, 29), getitem___37265, slice_37263)
    
    # Applying the binary operator '*' (line 513)
    result_mul_37267 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 13), '*', subscript_call_result_37261, subscript_call_result_37266)
    
    # Applying the binary operator '*=' (line 513)
    result_imul_37268 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 8), '*=', x_37255, result_mul_37267)
    # Assigning a type to the variable 'x' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'x', result_imul_37268)
    
    # SSA join for if statement (line 512)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 516):
    
    # Assigning a Call to a Name (line 516):
    
    # Assigning a Call to a Name (line 516):
    
    # Call to dot(...): (line 516)
    # Processing the call arguments (line 516)
    # Getting the type of 'u10' (line 516)
    u10_37275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 29), 'u10', False)
    # Processing the call keyword arguments (line 516)
    kwargs_37276 = {}
    
    # Call to conj(...): (line 516)
    # Processing the call keyword arguments (line 516)
    kwargs_37271 = {}
    # Getting the type of 'u00' (line 516)
    u00_37269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'u00', False)
    # Obtaining the member 'conj' of a type (line 516)
    conj_37270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 12), u00_37269, 'conj')
    # Calling conj(args, kwargs) (line 516)
    conj_call_result_37272 = invoke(stypy.reporting.localization.Localization(__file__, 516, 12), conj_37270, *[], **kwargs_37271)
    
    # Obtaining the member 'T' of a type (line 516)
    T_37273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 12), conj_call_result_37272, 'T')
    # Obtaining the member 'dot' of a type (line 516)
    dot_37274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 12), T_37273, 'dot')
    # Calling dot(args, kwargs) (line 516)
    dot_call_result_37277 = invoke(stypy.reporting.localization.Localization(__file__, 516, 12), dot_37274, *[u10_37275], **kwargs_37276)
    
    # Assigning a type to the variable 'u_sym' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'u_sym', dot_call_result_37277)
    
    # Assigning a Call to a Name (line 517):
    
    # Assigning a Call to a Name (line 517):
    
    # Assigning a Call to a Name (line 517):
    
    # Call to norm(...): (line 517)
    # Processing the call arguments (line 517)
    # Getting the type of 'u_sym' (line 517)
    u_sym_37279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 19), 'u_sym', False)
    int_37280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 26), 'int')
    # Processing the call keyword arguments (line 517)
    kwargs_37281 = {}
    # Getting the type of 'norm' (line 517)
    norm_37278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 14), 'norm', False)
    # Calling norm(args, kwargs) (line 517)
    norm_call_result_37282 = invoke(stypy.reporting.localization.Localization(__file__, 517, 14), norm_37278, *[u_sym_37279, int_37280], **kwargs_37281)
    
    # Assigning a type to the variable 'n_u_sym' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'n_u_sym', norm_call_result_37282)
    
    # Assigning a BinOp to a Name (line 518):
    
    # Assigning a BinOp to a Name (line 518):
    
    # Assigning a BinOp to a Name (line 518):
    # Getting the type of 'u_sym' (line 518)
    u_sym_37283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'u_sym')
    
    # Call to conj(...): (line 518)
    # Processing the call keyword arguments (line 518)
    kwargs_37286 = {}
    # Getting the type of 'u_sym' (line 518)
    u_sym_37284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 20), 'u_sym', False)
    # Obtaining the member 'conj' of a type (line 518)
    conj_37285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 20), u_sym_37284, 'conj')
    # Calling conj(args, kwargs) (line 518)
    conj_call_result_37287 = invoke(stypy.reporting.localization.Localization(__file__, 518, 20), conj_37285, *[], **kwargs_37286)
    
    # Obtaining the member 'T' of a type (line 518)
    T_37288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 20), conj_call_result_37287, 'T')
    # Applying the binary operator '-' (line 518)
    result_sub_37289 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 12), '-', u_sym_37283, T_37288)
    
    # Assigning a type to the variable 'u_sym' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'u_sym', result_sub_37289)
    
    # Assigning a Call to a Name (line 519):
    
    # Assigning a Call to a Name (line 519):
    
    # Assigning a Call to a Name (line 519):
    
    # Call to max(...): (line 519)
    # Processing the call arguments (line 519)
    
    # Obtaining an instance of the builtin type 'list' (line 519)
    list_37292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 519)
    # Adding element type (line 519)
    
    # Call to spacing(...): (line 519)
    # Processing the call arguments (line 519)
    float_37295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 39), 'float')
    # Processing the call keyword arguments (line 519)
    kwargs_37296 = {}
    # Getting the type of 'np' (line 519)
    np_37293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 28), 'np', False)
    # Obtaining the member 'spacing' of a type (line 519)
    spacing_37294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 28), np_37293, 'spacing')
    # Calling spacing(args, kwargs) (line 519)
    spacing_call_result_37297 = invoke(stypy.reporting.localization.Localization(__file__, 519, 28), spacing_37294, *[float_37295], **kwargs_37296)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 27), list_37292, spacing_call_result_37297)
    # Adding element type (line 519)
    # Getting the type of 'n_u_sym' (line 519)
    n_u_sym_37298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 47), 'n_u_sym', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 27), list_37292, n_u_sym_37298)
    
    # Processing the call keyword arguments (line 519)
    kwargs_37299 = {}
    # Getting the type of 'np' (line 519)
    np_37290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 20), 'np', False)
    # Obtaining the member 'max' of a type (line 519)
    max_37291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 20), np_37290, 'max')
    # Calling max(args, kwargs) (line 519)
    max_call_result_37300 = invoke(stypy.reporting.localization.Localization(__file__, 519, 20), max_37291, *[list_37292], **kwargs_37299)
    
    # Assigning a type to the variable 'sym_threshold' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'sym_threshold', max_call_result_37300)
    
    
    
    # Call to norm(...): (line 521)
    # Processing the call arguments (line 521)
    # Getting the type of 'u_sym' (line 521)
    u_sym_37302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'u_sym', False)
    int_37303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 19), 'int')
    # Processing the call keyword arguments (line 521)
    kwargs_37304 = {}
    # Getting the type of 'norm' (line 521)
    norm_37301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 7), 'norm', False)
    # Calling norm(args, kwargs) (line 521)
    norm_call_result_37305 = invoke(stypy.reporting.localization.Localization(__file__, 521, 7), norm_37301, *[u_sym_37302, int_37303], **kwargs_37304)
    
    # Getting the type of 'sym_threshold' (line 521)
    sym_threshold_37306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 24), 'sym_threshold')
    # Applying the binary operator '>' (line 521)
    result_gt_37307 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 7), '>', norm_call_result_37305, sym_threshold_37306)
    
    # Testing the type of an if condition (line 521)
    if_condition_37308 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 521, 4), result_gt_37307)
    # Assigning a type to the variable 'if_condition_37308' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'if_condition_37308', if_condition_37308)
    # SSA begins for if statement (line 521)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 522)
    # Processing the call arguments (line 522)
    str_37310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 26), 'str', 'The associated Hamiltonian pencil has eigenvalues too close to the imaginary axis')
    # Processing the call keyword arguments (line 522)
    kwargs_37311 = {}
    # Getting the type of 'LinAlgError' (line 522)
    LinAlgError_37309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 522)
    LinAlgError_call_result_37312 = invoke(stypy.reporting.localization.Localization(__file__, 522, 14), LinAlgError_37309, *[str_37310], **kwargs_37311)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 522, 8), LinAlgError_call_result_37312, 'raise parameter', BaseException)
    # SSA join for if statement (line 521)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 525)
    x_37313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'x')
    
    # Call to conj(...): (line 525)
    # Processing the call keyword arguments (line 525)
    kwargs_37316 = {}
    # Getting the type of 'x' (line 525)
    x_37314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'x', False)
    # Obtaining the member 'conj' of a type (line 525)
    conj_37315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 16), x_37314, 'conj')
    # Calling conj(args, kwargs) (line 525)
    conj_call_result_37317 = invoke(stypy.reporting.localization.Localization(__file__, 525, 16), conj_37315, *[], **kwargs_37316)
    
    # Obtaining the member 'T' of a type (line 525)
    T_37318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 16), conj_call_result_37317, 'T')
    # Applying the binary operator '+' (line 525)
    result_add_37319 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 12), '+', x_37313, T_37318)
    
    int_37320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 28), 'int')
    # Applying the binary operator 'div' (line 525)
    result_div_37321 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 11), 'div', result_add_37319, int_37320)
    
    # Assigning a type to the variable 'stypy_return_type' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'stypy_return_type', result_div_37321)
    
    # ################# End of 'solve_continuous_are(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_continuous_are' in the type store
    # Getting the type of 'stypy_return_type' (line 325)
    stypy_return_type_37322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_37322)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_continuous_are'
    return stypy_return_type_37322

# Assigning a type to the variable 'solve_continuous_are' (line 325)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 0), 'solve_continuous_are', solve_continuous_are)

@norecursion
def solve_discrete_are(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 528)
    None_37323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 37), 'None')
    # Getting the type of 'None' (line 528)
    None_37324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 45), 'None')
    # Getting the type of 'True' (line 528)
    True_37325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 60), 'True')
    defaults = [None_37323, None_37324, True_37325]
    # Create a new context for function 'solve_discrete_are'
    module_type_store = module_type_store.open_function_context('solve_discrete_are', 528, 0, False)
    
    # Passed parameters checking function
    solve_discrete_are.stypy_localization = localization
    solve_discrete_are.stypy_type_of_self = None
    solve_discrete_are.stypy_type_store = module_type_store
    solve_discrete_are.stypy_function_name = 'solve_discrete_are'
    solve_discrete_are.stypy_param_names_list = ['a', 'b', 'q', 'r', 'e', 's', 'balanced']
    solve_discrete_are.stypy_varargs_param_name = None
    solve_discrete_are.stypy_kwargs_param_name = None
    solve_discrete_are.stypy_call_defaults = defaults
    solve_discrete_are.stypy_call_varargs = varargs
    solve_discrete_are.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_discrete_are', ['a', 'b', 'q', 'r', 'e', 's', 'balanced'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_discrete_are', localization, ['a', 'b', 'q', 'r', 'e', 's', 'balanced'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_discrete_are(...)' code ##################

    str_37326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, (-1)), 'str', '\n    Solves the discrete-time algebraic Riccati equation (DARE).\n\n    The DARE is defined as\n\n    .. math::\n\n          A^HXA - X - (A^HXB) (R + B^HXB)^{-1} (B^HXA) + Q = 0\n\n    The limitations for a solution to exist are :\n\n        * All eigenvalues of :math:`A` outside the unit disc, should be\n          controllable.\n\n        * The associated symplectic pencil (See Notes), should have\n          eigenvalues sufficiently away from the unit circle.\n\n    Moreover, if ``e`` and ``s`` are not both precisely ``None``, then the\n    generalized version of DARE\n\n    .. math::\n\n          A^HXA - E^HXE - (A^HXB+S) (R+B^HXB)^{-1} (B^HXA+S^H) + Q = 0\n\n    is solved. When omitted, ``e`` is assumed to be the identity and ``s``\n    is assumed to be the zero matrix.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Square matrix\n    b : (M, N) array_like\n        Input\n    q : (M, M) array_like\n        Input\n    r : (N, N) array_like\n        Square matrix\n    e : (M, M) array_like, optional\n        Nonsingular square matrix\n    s : (M, N) array_like, optional\n        Input\n    balanced : bool\n        The boolean that indicates whether a balancing step is performed\n        on the data. The default is set to True.\n\n    Returns\n    -------\n    x : (M, M) ndarray\n        Solution to the discrete algebraic Riccati equation.\n\n    Raises\n    ------\n    LinAlgError\n        For cases where the stable subspace of the pencil could not be\n        isolated. See Notes section and the references for details.\n\n    See Also\n    --------\n    solve_continuous_are : Solves the continuous algebraic Riccati equation\n\n    Notes\n    -----\n    The equation is solved by forming the extended symplectic matrix pencil,\n    as described in [1]_, :math:`H - \\lambda J` given by the block matrices ::\n\n           [  A   0   B ]             [ E   0   B ]\n           [ -Q  E^H -S ] - \\lambda * [ 0  A^H  0 ]\n           [ S^H  0   R ]             [ 0 -B^H  0 ]\n\n    and using a QZ decomposition method.\n\n    In this algorithm, the fail conditions are linked to the symmetry\n    of the product :math:`U_2 U_1^{-1}` and condition number of\n    :math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the\n    eigenvectors spanning the stable subspace with 2m rows and partitioned\n    into two m-row matrices. See [1]_ and [2]_ for more details.\n\n    In order to improve the QZ decomposition accuracy, the pencil goes\n    through a balancing step where the sum of absolute values of\n    :math:`H` and :math:`J` rows/cols (after removing the diagonal entries)\n    is balanced following the recipe given in [3]_. If the data has small\n    numerical noise, balancing may amplify their effects and some clean up\n    is required.\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1]  P. van Dooren , "A Generalized Eigenvalue Approach For Solving\n       Riccati Equations.", SIAM Journal on Scientific and Statistical\n       Computing, Vol.2(2), DOI: 10.1137/0902010\n\n    .. [2] A.J. Laub, "A Schur Method for Solving Algebraic Riccati\n       Equations.", Massachusetts Institute of Technology. Laboratory for\n       Information and Decision Systems. LIDS-R ; 859. Available online :\n       http://hdl.handle.net/1721.1/1301\n\n    .. [3] P. Benner, "Symplectic Balancing of Hamiltonian Matrices", 2001,\n       SIAM J. Sci. Comput., 2001, Vol.22(5), DOI: 10.1137/S1064827500367993\n\n    Examples\n    --------\n    Given `a`, `b`, `q`, and `r` solve for `x`:\n\n    >>> from scipy import linalg as la\n    >>> a = np.array([[0, 1], [0, -1]])\n    >>> b = np.array([[1, 0], [2, 1]])\n    >>> q = np.array([[-4, -4], [-4, 7]])\n    >>> r = np.array([[9, 3], [3, 1]])\n    >>> x = la.solve_discrete_are(a, b, q, r)\n    >>> x\n    array([[-4., -4.],\n           [-4.,  7.]])\n    >>> R = la.solve(r + b.T.dot(x).dot(b), b.T.dot(x).dot(a))\n    >>> np.allclose(a.T.dot(x).dot(a) - x - a.T.dot(x).dot(b).dot(R), -q)\n    True\n\n    ')
    
    # Assigning a Call to a Tuple (line 649):
    
    # Assigning a Subscript to a Name (line 649):
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    int_37327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    
    # Call to _are_validate_args(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'a' (line 650)
    a_37329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 53), 'a', False)
    # Getting the type of 'b' (line 650)
    b_37330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 56), 'b', False)
    # Getting the type of 'q' (line 650)
    q_37331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 59), 'q', False)
    # Getting the type of 'r' (line 650)
    r_37332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 62), 'r', False)
    # Getting the type of 'e' (line 650)
    e_37333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 65), 'e', False)
    # Getting the type of 's' (line 650)
    s_37334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 68), 's', False)
    str_37335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 71), 'str', 'dare')
    # Processing the call keyword arguments (line 649)
    kwargs_37336 = {}
    # Getting the type of '_are_validate_args' (line 649)
    _are_validate_args_37328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 649)
    _are_validate_args_call_result_37337 = invoke(stypy.reporting.localization.Localization(__file__, 649, 46), _are_validate_args_37328, *[a_37329, b_37330, q_37331, r_37332, e_37333, s_37334, str_37335], **kwargs_37336)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___37338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), _are_validate_args_call_result_37337, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_37339 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), getitem___37338, int_37327)
    
    # Assigning a type to the variable 'tuple_var_assignment_35908' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35908', subscript_call_result_37339)
    
    # Assigning a Subscript to a Name (line 649):
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    int_37340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    
    # Call to _are_validate_args(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'a' (line 650)
    a_37342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 53), 'a', False)
    # Getting the type of 'b' (line 650)
    b_37343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 56), 'b', False)
    # Getting the type of 'q' (line 650)
    q_37344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 59), 'q', False)
    # Getting the type of 'r' (line 650)
    r_37345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 62), 'r', False)
    # Getting the type of 'e' (line 650)
    e_37346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 65), 'e', False)
    # Getting the type of 's' (line 650)
    s_37347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 68), 's', False)
    str_37348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 71), 'str', 'dare')
    # Processing the call keyword arguments (line 649)
    kwargs_37349 = {}
    # Getting the type of '_are_validate_args' (line 649)
    _are_validate_args_37341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 649)
    _are_validate_args_call_result_37350 = invoke(stypy.reporting.localization.Localization(__file__, 649, 46), _are_validate_args_37341, *[a_37342, b_37343, q_37344, r_37345, e_37346, s_37347, str_37348], **kwargs_37349)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___37351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), _are_validate_args_call_result_37350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_37352 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), getitem___37351, int_37340)
    
    # Assigning a type to the variable 'tuple_var_assignment_35909' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35909', subscript_call_result_37352)
    
    # Assigning a Subscript to a Name (line 649):
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    int_37353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    
    # Call to _are_validate_args(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'a' (line 650)
    a_37355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 53), 'a', False)
    # Getting the type of 'b' (line 650)
    b_37356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 56), 'b', False)
    # Getting the type of 'q' (line 650)
    q_37357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 59), 'q', False)
    # Getting the type of 'r' (line 650)
    r_37358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 62), 'r', False)
    # Getting the type of 'e' (line 650)
    e_37359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 65), 'e', False)
    # Getting the type of 's' (line 650)
    s_37360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 68), 's', False)
    str_37361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 71), 'str', 'dare')
    # Processing the call keyword arguments (line 649)
    kwargs_37362 = {}
    # Getting the type of '_are_validate_args' (line 649)
    _are_validate_args_37354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 649)
    _are_validate_args_call_result_37363 = invoke(stypy.reporting.localization.Localization(__file__, 649, 46), _are_validate_args_37354, *[a_37355, b_37356, q_37357, r_37358, e_37359, s_37360, str_37361], **kwargs_37362)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___37364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), _are_validate_args_call_result_37363, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_37365 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), getitem___37364, int_37353)
    
    # Assigning a type to the variable 'tuple_var_assignment_35910' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35910', subscript_call_result_37365)
    
    # Assigning a Subscript to a Name (line 649):
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    int_37366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    
    # Call to _are_validate_args(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'a' (line 650)
    a_37368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 53), 'a', False)
    # Getting the type of 'b' (line 650)
    b_37369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 56), 'b', False)
    # Getting the type of 'q' (line 650)
    q_37370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 59), 'q', False)
    # Getting the type of 'r' (line 650)
    r_37371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 62), 'r', False)
    # Getting the type of 'e' (line 650)
    e_37372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 65), 'e', False)
    # Getting the type of 's' (line 650)
    s_37373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 68), 's', False)
    str_37374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 71), 'str', 'dare')
    # Processing the call keyword arguments (line 649)
    kwargs_37375 = {}
    # Getting the type of '_are_validate_args' (line 649)
    _are_validate_args_37367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 649)
    _are_validate_args_call_result_37376 = invoke(stypy.reporting.localization.Localization(__file__, 649, 46), _are_validate_args_37367, *[a_37368, b_37369, q_37370, r_37371, e_37372, s_37373, str_37374], **kwargs_37375)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___37377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), _are_validate_args_call_result_37376, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_37378 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), getitem___37377, int_37366)
    
    # Assigning a type to the variable 'tuple_var_assignment_35911' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35911', subscript_call_result_37378)
    
    # Assigning a Subscript to a Name (line 649):
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    int_37379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    
    # Call to _are_validate_args(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'a' (line 650)
    a_37381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 53), 'a', False)
    # Getting the type of 'b' (line 650)
    b_37382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 56), 'b', False)
    # Getting the type of 'q' (line 650)
    q_37383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 59), 'q', False)
    # Getting the type of 'r' (line 650)
    r_37384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 62), 'r', False)
    # Getting the type of 'e' (line 650)
    e_37385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 65), 'e', False)
    # Getting the type of 's' (line 650)
    s_37386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 68), 's', False)
    str_37387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 71), 'str', 'dare')
    # Processing the call keyword arguments (line 649)
    kwargs_37388 = {}
    # Getting the type of '_are_validate_args' (line 649)
    _are_validate_args_37380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 649)
    _are_validate_args_call_result_37389 = invoke(stypy.reporting.localization.Localization(__file__, 649, 46), _are_validate_args_37380, *[a_37381, b_37382, q_37383, r_37384, e_37385, s_37386, str_37387], **kwargs_37388)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___37390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), _are_validate_args_call_result_37389, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_37391 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), getitem___37390, int_37379)
    
    # Assigning a type to the variable 'tuple_var_assignment_35912' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35912', subscript_call_result_37391)
    
    # Assigning a Subscript to a Name (line 649):
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    int_37392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    
    # Call to _are_validate_args(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'a' (line 650)
    a_37394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 53), 'a', False)
    # Getting the type of 'b' (line 650)
    b_37395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 56), 'b', False)
    # Getting the type of 'q' (line 650)
    q_37396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 59), 'q', False)
    # Getting the type of 'r' (line 650)
    r_37397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 62), 'r', False)
    # Getting the type of 'e' (line 650)
    e_37398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 65), 'e', False)
    # Getting the type of 's' (line 650)
    s_37399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 68), 's', False)
    str_37400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 71), 'str', 'dare')
    # Processing the call keyword arguments (line 649)
    kwargs_37401 = {}
    # Getting the type of '_are_validate_args' (line 649)
    _are_validate_args_37393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 649)
    _are_validate_args_call_result_37402 = invoke(stypy.reporting.localization.Localization(__file__, 649, 46), _are_validate_args_37393, *[a_37394, b_37395, q_37396, r_37397, e_37398, s_37399, str_37400], **kwargs_37401)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___37403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), _are_validate_args_call_result_37402, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_37404 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), getitem___37403, int_37392)
    
    # Assigning a type to the variable 'tuple_var_assignment_35913' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35913', subscript_call_result_37404)
    
    # Assigning a Subscript to a Name (line 649):
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    int_37405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    
    # Call to _are_validate_args(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'a' (line 650)
    a_37407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 53), 'a', False)
    # Getting the type of 'b' (line 650)
    b_37408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 56), 'b', False)
    # Getting the type of 'q' (line 650)
    q_37409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 59), 'q', False)
    # Getting the type of 'r' (line 650)
    r_37410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 62), 'r', False)
    # Getting the type of 'e' (line 650)
    e_37411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 65), 'e', False)
    # Getting the type of 's' (line 650)
    s_37412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 68), 's', False)
    str_37413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 71), 'str', 'dare')
    # Processing the call keyword arguments (line 649)
    kwargs_37414 = {}
    # Getting the type of '_are_validate_args' (line 649)
    _are_validate_args_37406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 649)
    _are_validate_args_call_result_37415 = invoke(stypy.reporting.localization.Localization(__file__, 649, 46), _are_validate_args_37406, *[a_37407, b_37408, q_37409, r_37410, e_37411, s_37412, str_37413], **kwargs_37414)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___37416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), _are_validate_args_call_result_37415, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_37417 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), getitem___37416, int_37405)
    
    # Assigning a type to the variable 'tuple_var_assignment_35914' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35914', subscript_call_result_37417)
    
    # Assigning a Subscript to a Name (line 649):
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    int_37418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    
    # Call to _are_validate_args(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'a' (line 650)
    a_37420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 53), 'a', False)
    # Getting the type of 'b' (line 650)
    b_37421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 56), 'b', False)
    # Getting the type of 'q' (line 650)
    q_37422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 59), 'q', False)
    # Getting the type of 'r' (line 650)
    r_37423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 62), 'r', False)
    # Getting the type of 'e' (line 650)
    e_37424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 65), 'e', False)
    # Getting the type of 's' (line 650)
    s_37425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 68), 's', False)
    str_37426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 71), 'str', 'dare')
    # Processing the call keyword arguments (line 649)
    kwargs_37427 = {}
    # Getting the type of '_are_validate_args' (line 649)
    _are_validate_args_37419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 649)
    _are_validate_args_call_result_37428 = invoke(stypy.reporting.localization.Localization(__file__, 649, 46), _are_validate_args_37419, *[a_37420, b_37421, q_37422, r_37423, e_37424, s_37425, str_37426], **kwargs_37427)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___37429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), _are_validate_args_call_result_37428, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_37430 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), getitem___37429, int_37418)
    
    # Assigning a type to the variable 'tuple_var_assignment_35915' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35915', subscript_call_result_37430)
    
    # Assigning a Subscript to a Name (line 649):
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    int_37431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    
    # Call to _are_validate_args(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'a' (line 650)
    a_37433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 53), 'a', False)
    # Getting the type of 'b' (line 650)
    b_37434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 56), 'b', False)
    # Getting the type of 'q' (line 650)
    q_37435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 59), 'q', False)
    # Getting the type of 'r' (line 650)
    r_37436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 62), 'r', False)
    # Getting the type of 'e' (line 650)
    e_37437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 65), 'e', False)
    # Getting the type of 's' (line 650)
    s_37438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 68), 's', False)
    str_37439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 71), 'str', 'dare')
    # Processing the call keyword arguments (line 649)
    kwargs_37440 = {}
    # Getting the type of '_are_validate_args' (line 649)
    _are_validate_args_37432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 649)
    _are_validate_args_call_result_37441 = invoke(stypy.reporting.localization.Localization(__file__, 649, 46), _are_validate_args_37432, *[a_37433, b_37434, q_37435, r_37436, e_37437, s_37438, str_37439], **kwargs_37440)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___37442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), _are_validate_args_call_result_37441, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_37443 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), getitem___37442, int_37431)
    
    # Assigning a type to the variable 'tuple_var_assignment_35916' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35916', subscript_call_result_37443)
    
    # Assigning a Subscript to a Name (line 649):
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    int_37444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    
    # Call to _are_validate_args(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'a' (line 650)
    a_37446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 53), 'a', False)
    # Getting the type of 'b' (line 650)
    b_37447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 56), 'b', False)
    # Getting the type of 'q' (line 650)
    q_37448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 59), 'q', False)
    # Getting the type of 'r' (line 650)
    r_37449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 62), 'r', False)
    # Getting the type of 'e' (line 650)
    e_37450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 65), 'e', False)
    # Getting the type of 's' (line 650)
    s_37451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 68), 's', False)
    str_37452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 71), 'str', 'dare')
    # Processing the call keyword arguments (line 649)
    kwargs_37453 = {}
    # Getting the type of '_are_validate_args' (line 649)
    _are_validate_args_37445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 46), '_are_validate_args', False)
    # Calling _are_validate_args(args, kwargs) (line 649)
    _are_validate_args_call_result_37454 = invoke(stypy.reporting.localization.Localization(__file__, 649, 46), _are_validate_args_37445, *[a_37446, b_37447, q_37448, r_37449, e_37450, s_37451, str_37452], **kwargs_37453)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___37455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), _are_validate_args_call_result_37454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_37456 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), getitem___37455, int_37444)
    
    # Assigning a type to the variable 'tuple_var_assignment_35917' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35917', subscript_call_result_37456)
    
    # Assigning a Name to a Name (line 649):
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_var_assignment_35908' (line 649)
    tuple_var_assignment_35908_37457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35908')
    # Assigning a type to the variable 'a' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'a', tuple_var_assignment_35908_37457)
    
    # Assigning a Name to a Name (line 649):
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_var_assignment_35909' (line 649)
    tuple_var_assignment_35909_37458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35909')
    # Assigning a type to the variable 'b' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 7), 'b', tuple_var_assignment_35909_37458)
    
    # Assigning a Name to a Name (line 649):
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_var_assignment_35910' (line 649)
    tuple_var_assignment_35910_37459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35910')
    # Assigning a type to the variable 'q' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 10), 'q', tuple_var_assignment_35910_37459)
    
    # Assigning a Name to a Name (line 649):
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_var_assignment_35911' (line 649)
    tuple_var_assignment_35911_37460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35911')
    # Assigning a type to the variable 'r' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 13), 'r', tuple_var_assignment_35911_37460)
    
    # Assigning a Name to a Name (line 649):
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_var_assignment_35912' (line 649)
    tuple_var_assignment_35912_37461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35912')
    # Assigning a type to the variable 'e' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 16), 'e', tuple_var_assignment_35912_37461)
    
    # Assigning a Name to a Name (line 649):
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_var_assignment_35913' (line 649)
    tuple_var_assignment_35913_37462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35913')
    # Assigning a type to the variable 's' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 19), 's', tuple_var_assignment_35913_37462)
    
    # Assigning a Name to a Name (line 649):
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_var_assignment_35914' (line 649)
    tuple_var_assignment_35914_37463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35914')
    # Assigning a type to the variable 'm' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 22), 'm', tuple_var_assignment_35914_37463)
    
    # Assigning a Name to a Name (line 649):
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_var_assignment_35915' (line 649)
    tuple_var_assignment_35915_37464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35915')
    # Assigning a type to the variable 'n' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 25), 'n', tuple_var_assignment_35915_37464)
    
    # Assigning a Name to a Name (line 649):
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_var_assignment_35916' (line 649)
    tuple_var_assignment_35916_37465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35916')
    # Assigning a type to the variable 'r_or_c' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 28), 'r_or_c', tuple_var_assignment_35916_37465)
    
    # Assigning a Name to a Name (line 649):
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_var_assignment_35917' (line 649)
    tuple_var_assignment_35917_37466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_35917')
    # Assigning a type to the variable 'gen_are' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 36), 'gen_are', tuple_var_assignment_35917_37466)
    
    # Assigning a Call to a Name (line 653):
    
    # Assigning a Call to a Name (line 653):
    
    # Assigning a Call to a Name (line 653):
    
    # Call to zeros(...): (line 653)
    # Processing the call arguments (line 653)
    
    # Obtaining an instance of the builtin type 'tuple' (line 653)
    tuple_37469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 653)
    # Adding element type (line 653)
    int_37470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 18), 'int')
    # Getting the type of 'm' (line 653)
    m_37471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 20), 'm', False)
    # Applying the binary operator '*' (line 653)
    result_mul_37472 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 18), '*', int_37470, m_37471)
    
    # Getting the type of 'n' (line 653)
    n_37473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 22), 'n', False)
    # Applying the binary operator '+' (line 653)
    result_add_37474 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 18), '+', result_mul_37472, n_37473)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 653, 18), tuple_37469, result_add_37474)
    # Adding element type (line 653)
    int_37475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 25), 'int')
    # Getting the type of 'm' (line 653)
    m_37476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 27), 'm', False)
    # Applying the binary operator '*' (line 653)
    result_mul_37477 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 25), '*', int_37475, m_37476)
    
    # Getting the type of 'n' (line 653)
    n_37478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 29), 'n', False)
    # Applying the binary operator '+' (line 653)
    result_add_37479 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 25), '+', result_mul_37477, n_37478)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 653, 18), tuple_37469, result_add_37479)
    
    # Processing the call keyword arguments (line 653)
    # Getting the type of 'r_or_c' (line 653)
    r_or_c_37480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 39), 'r_or_c', False)
    keyword_37481 = r_or_c_37480
    kwargs_37482 = {'dtype': keyword_37481}
    # Getting the type of 'np' (line 653)
    np_37467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 653)
    zeros_37468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 8), np_37467, 'zeros')
    # Calling zeros(args, kwargs) (line 653)
    zeros_call_result_37483 = invoke(stypy.reporting.localization.Localization(__file__, 653, 8), zeros_37468, *[tuple_37469], **kwargs_37482)
    
    # Assigning a type to the variable 'H' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'H', zeros_call_result_37483)
    
    # Assigning a Name to a Subscript (line 654):
    
    # Assigning a Name to a Subscript (line 654):
    
    # Assigning a Name to a Subscript (line 654):
    # Getting the type of 'a' (line 654)
    a_37484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 16), 'a')
    # Getting the type of 'H' (line 654)
    H_37485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 'H')
    # Getting the type of 'm' (line 654)
    m_37486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 7), 'm')
    slice_37487 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 654, 4), None, m_37486, None)
    # Getting the type of 'm' (line 654)
    m_37488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 11), 'm')
    slice_37489 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 654, 4), None, m_37488, None)
    # Storing an element on a container (line 654)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 4), H_37485, ((slice_37487, slice_37489), a_37484))
    
    # Assigning a Name to a Subscript (line 655):
    
    # Assigning a Name to a Subscript (line 655):
    
    # Assigning a Name to a Subscript (line 655):
    # Getting the type of 'b' (line 655)
    b_37490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 18), 'b')
    # Getting the type of 'H' (line 655)
    H_37491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'H')
    # Getting the type of 'm' (line 655)
    m_37492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 7), 'm')
    slice_37493 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 655, 4), None, m_37492, None)
    int_37494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 10), 'int')
    # Getting the type of 'm' (line 655)
    m_37495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 12), 'm')
    # Applying the binary operator '*' (line 655)
    result_mul_37496 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 10), '*', int_37494, m_37495)
    
    slice_37497 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 655, 4), result_mul_37496, None, None)
    # Storing an element on a container (line 655)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 655, 4), H_37491, ((slice_37493, slice_37497), b_37490))
    
    # Assigning a UnaryOp to a Subscript (line 656):
    
    # Assigning a UnaryOp to a Subscript (line 656):
    
    # Assigning a UnaryOp to a Subscript (line 656):
    
    # Getting the type of 'q' (line 656)
    q_37498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 20), 'q')
    # Applying the 'usub' unary operator (line 656)
    result___neg___37499 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 19), 'usub', q_37498)
    
    # Getting the type of 'H' (line 656)
    H_37500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'H')
    # Getting the type of 'm' (line 656)
    m_37501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 6), 'm')
    int_37502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 8), 'int')
    # Getting the type of 'm' (line 656)
    m_37503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 10), 'm')
    # Applying the binary operator '*' (line 656)
    result_mul_37504 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 8), '*', int_37502, m_37503)
    
    slice_37505 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 656, 4), m_37501, result_mul_37504, None)
    # Getting the type of 'm' (line 656)
    m_37506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 14), 'm')
    slice_37507 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 656, 4), None, m_37506, None)
    # Storing an element on a container (line 656)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 4), H_37500, ((slice_37505, slice_37507), result___neg___37499))
    
    # Assigning a IfExp to a Subscript (line 657):
    
    # Assigning a IfExp to a Subscript (line 657):
    
    # Assigning a IfExp to a Subscript (line 657):
    
    
    # Getting the type of 'e' (line 657)
    e_37508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 35), 'e')
    # Getting the type of 'None' (line 657)
    None_37509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 40), 'None')
    # Applying the binary operator 'is' (line 657)
    result_is__37510 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 35), 'is', e_37508, None_37509)
    
    # Testing the type of an if expression (line 657)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 657, 22), result_is__37510)
    # SSA begins for if expression (line 657)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to eye(...): (line 657)
    # Processing the call arguments (line 657)
    # Getting the type of 'm' (line 657)
    m_37513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 29), 'm', False)
    # Processing the call keyword arguments (line 657)
    kwargs_37514 = {}
    # Getting the type of 'np' (line 657)
    np_37511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 22), 'np', False)
    # Obtaining the member 'eye' of a type (line 657)
    eye_37512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 22), np_37511, 'eye')
    # Calling eye(args, kwargs) (line 657)
    eye_call_result_37515 = invoke(stypy.reporting.localization.Localization(__file__, 657, 22), eye_37512, *[m_37513], **kwargs_37514)
    
    # SSA branch for the else part of an if expression (line 657)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to conj(...): (line 657)
    # Processing the call keyword arguments (line 657)
    kwargs_37518 = {}
    # Getting the type of 'e' (line 657)
    e_37516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 50), 'e', False)
    # Obtaining the member 'conj' of a type (line 657)
    conj_37517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 50), e_37516, 'conj')
    # Calling conj(args, kwargs) (line 657)
    conj_call_result_37519 = invoke(stypy.reporting.localization.Localization(__file__, 657, 50), conj_37517, *[], **kwargs_37518)
    
    # Obtaining the member 'T' of a type (line 657)
    T_37520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 50), conj_call_result_37519, 'T')
    # SSA join for if expression (line 657)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_37521 = union_type.UnionType.add(eye_call_result_37515, T_37520)
    
    # Getting the type of 'H' (line 657)
    H_37522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'H')
    # Getting the type of 'm' (line 657)
    m_37523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 6), 'm')
    int_37524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 8), 'int')
    # Getting the type of 'm' (line 657)
    m_37525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 10), 'm')
    # Applying the binary operator '*' (line 657)
    result_mul_37526 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 8), '*', int_37524, m_37525)
    
    slice_37527 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 657, 4), m_37523, result_mul_37526, None)
    # Getting the type of 'm' (line 657)
    m_37528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 13), 'm')
    int_37529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 15), 'int')
    # Getting the type of 'm' (line 657)
    m_37530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 17), 'm')
    # Applying the binary operator '*' (line 657)
    result_mul_37531 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 15), '*', int_37529, m_37530)
    
    slice_37532 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 657, 4), m_37528, result_mul_37531, None)
    # Storing an element on a container (line 657)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 657, 4), H_37522, ((slice_37527, slice_37532), if_exp_37521))
    
    # Assigning a IfExp to a Subscript (line 658):
    
    # Assigning a IfExp to a Subscript (line 658):
    
    # Assigning a IfExp to a Subscript (line 658):
    
    
    # Getting the type of 's' (line 658)
    s_37533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 27), 's')
    # Getting the type of 'None' (line 658)
    None_37534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 32), 'None')
    # Applying the binary operator 'is' (line 658)
    result_is__37535 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 27), 'is', s_37533, None_37534)
    
    # Testing the type of an if expression (line 658)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 658, 21), result_is__37535)
    # SSA begins for if expression (line 658)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    float_37536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 21), 'float')
    # SSA branch for the else part of an if expression (line 658)
    module_type_store.open_ssa_branch('if expression else')
    
    # Getting the type of 's' (line 658)
    s_37537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 43), 's')
    # Applying the 'usub' unary operator (line 658)
    result___neg___37538 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 42), 'usub', s_37537)
    
    # SSA join for if expression (line 658)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_37539 = union_type.UnionType.add(float_37536, result___neg___37538)
    
    # Getting the type of 'H' (line 658)
    H_37540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'H')
    # Getting the type of 'm' (line 658)
    m_37541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 6), 'm')
    int_37542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 8), 'int')
    # Getting the type of 'm' (line 658)
    m_37543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 10), 'm')
    # Applying the binary operator '*' (line 658)
    result_mul_37544 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 8), '*', int_37542, m_37543)
    
    slice_37545 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 658, 4), m_37541, result_mul_37544, None)
    int_37546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 13), 'int')
    # Getting the type of 'm' (line 658)
    m_37547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 15), 'm')
    # Applying the binary operator '*' (line 658)
    result_mul_37548 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 13), '*', int_37546, m_37547)
    
    slice_37549 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 658, 4), result_mul_37548, None, None)
    # Storing an element on a container (line 658)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 4), H_37540, ((slice_37545, slice_37549), if_exp_37539))
    
    # Assigning a IfExp to a Subscript (line 659):
    
    # Assigning a IfExp to a Subscript (line 659):
    
    # Assigning a IfExp to a Subscript (line 659):
    
    
    # Getting the type of 's' (line 659)
    s_37550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 24), 's')
    # Getting the type of 'None' (line 659)
    None_37551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 29), 'None')
    # Applying the binary operator 'is' (line 659)
    result_is__37552 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 24), 'is', s_37550, None_37551)
    
    # Testing the type of an if expression (line 659)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 659, 18), result_is__37552)
    # SSA begins for if expression (line 659)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    float_37553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 18), 'float')
    # SSA branch for the else part of an if expression (line 659)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to conj(...): (line 659)
    # Processing the call keyword arguments (line 659)
    kwargs_37556 = {}
    # Getting the type of 's' (line 659)
    s_37554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 39), 's', False)
    # Obtaining the member 'conj' of a type (line 659)
    conj_37555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 39), s_37554, 'conj')
    # Calling conj(args, kwargs) (line 659)
    conj_call_result_37557 = invoke(stypy.reporting.localization.Localization(__file__, 659, 39), conj_37555, *[], **kwargs_37556)
    
    # Obtaining the member 'T' of a type (line 659)
    T_37558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 39), conj_call_result_37557, 'T')
    # SSA join for if expression (line 659)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_37559 = union_type.UnionType.add(float_37553, T_37558)
    
    # Getting the type of 'H' (line 659)
    H_37560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'H')
    int_37561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 6), 'int')
    # Getting the type of 'm' (line 659)
    m_37562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 8), 'm')
    # Applying the binary operator '*' (line 659)
    result_mul_37563 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 6), '*', int_37561, m_37562)
    
    slice_37564 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 659, 4), result_mul_37563, None, None)
    # Getting the type of 'm' (line 659)
    m_37565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 13), 'm')
    slice_37566 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 659, 4), None, m_37565, None)
    # Storing an element on a container (line 659)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 4), H_37560, ((slice_37564, slice_37566), if_exp_37559))
    
    # Assigning a Name to a Subscript (line 660):
    
    # Assigning a Name to a Subscript (line 660):
    
    # Assigning a Name to a Subscript (line 660):
    # Getting the type of 'r' (line 660)
    r_37567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 20), 'r')
    # Getting the type of 'H' (line 660)
    H_37568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'H')
    int_37569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 6), 'int')
    # Getting the type of 'm' (line 660)
    m_37570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'm')
    # Applying the binary operator '*' (line 660)
    result_mul_37571 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 6), '*', int_37569, m_37570)
    
    slice_37572 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 660, 4), result_mul_37571, None, None)
    int_37573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 12), 'int')
    # Getting the type of 'm' (line 660)
    m_37574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 14), 'm')
    # Applying the binary operator '*' (line 660)
    result_mul_37575 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 12), '*', int_37573, m_37574)
    
    slice_37576 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 660, 4), result_mul_37575, None, None)
    # Storing an element on a container (line 660)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 660, 4), H_37568, ((slice_37572, slice_37576), r_37567))
    
    # Assigning a Call to a Name (line 662):
    
    # Assigning a Call to a Name (line 662):
    
    # Assigning a Call to a Name (line 662):
    
    # Call to zeros_like(...): (line 662)
    # Processing the call arguments (line 662)
    # Getting the type of 'H' (line 662)
    H_37579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 22), 'H', False)
    # Processing the call keyword arguments (line 662)
    # Getting the type of 'r_or_c' (line 662)
    r_or_c_37580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 31), 'r_or_c', False)
    keyword_37581 = r_or_c_37580
    kwargs_37582 = {'dtype': keyword_37581}
    # Getting the type of 'np' (line 662)
    np_37577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 662)
    zeros_like_37578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 8), np_37577, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 662)
    zeros_like_call_result_37583 = invoke(stypy.reporting.localization.Localization(__file__, 662, 8), zeros_like_37578, *[H_37579], **kwargs_37582)
    
    # Assigning a type to the variable 'J' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 'J', zeros_like_call_result_37583)
    
    # Assigning a IfExp to a Subscript (line 663):
    
    # Assigning a IfExp to a Subscript (line 663):
    
    # Assigning a IfExp to a Subscript (line 663):
    
    
    # Getting the type of 'e' (line 663)
    e_37584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 29), 'e')
    # Getting the type of 'None' (line 663)
    None_37585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 34), 'None')
    # Applying the binary operator 'is' (line 663)
    result_is__37586 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 29), 'is', e_37584, None_37585)
    
    # Testing the type of an if expression (line 663)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 663, 16), result_is__37586)
    # SSA begins for if expression (line 663)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to eye(...): (line 663)
    # Processing the call arguments (line 663)
    # Getting the type of 'm' (line 663)
    m_37589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 23), 'm', False)
    # Processing the call keyword arguments (line 663)
    kwargs_37590 = {}
    # Getting the type of 'np' (line 663)
    np_37587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 16), 'np', False)
    # Obtaining the member 'eye' of a type (line 663)
    eye_37588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 16), np_37587, 'eye')
    # Calling eye(args, kwargs) (line 663)
    eye_call_result_37591 = invoke(stypy.reporting.localization.Localization(__file__, 663, 16), eye_37588, *[m_37589], **kwargs_37590)
    
    # SSA branch for the else part of an if expression (line 663)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'e' (line 663)
    e_37592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 44), 'e')
    # SSA join for if expression (line 663)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_37593 = union_type.UnionType.add(eye_call_result_37591, e_37592)
    
    # Getting the type of 'J' (line 663)
    J_37594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 4), 'J')
    # Getting the type of 'm' (line 663)
    m_37595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 7), 'm')
    slice_37596 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 663, 4), None, m_37595, None)
    # Getting the type of 'm' (line 663)
    m_37597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 11), 'm')
    slice_37598 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 663, 4), None, m_37597, None)
    # Storing an element on a container (line 663)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 4), J_37594, ((slice_37596, slice_37598), if_exp_37593))
    
    # Assigning a Attribute to a Subscript (line 664):
    
    # Assigning a Attribute to a Subscript (line 664):
    
    # Assigning a Attribute to a Subscript (line 664):
    
    # Call to conj(...): (line 664)
    # Processing the call keyword arguments (line 664)
    kwargs_37601 = {}
    # Getting the type of 'a' (line 664)
    a_37599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 22), 'a', False)
    # Obtaining the member 'conj' of a type (line 664)
    conj_37600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 22), a_37599, 'conj')
    # Calling conj(args, kwargs) (line 664)
    conj_call_result_37602 = invoke(stypy.reporting.localization.Localization(__file__, 664, 22), conj_37600, *[], **kwargs_37601)
    
    # Obtaining the member 'T' of a type (line 664)
    T_37603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 22), conj_call_result_37602, 'T')
    # Getting the type of 'J' (line 664)
    J_37604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 4), 'J')
    # Getting the type of 'm' (line 664)
    m_37605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 6), 'm')
    int_37606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 8), 'int')
    # Getting the type of 'm' (line 664)
    m_37607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 10), 'm')
    # Applying the binary operator '*' (line 664)
    result_mul_37608 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 8), '*', int_37606, m_37607)
    
    slice_37609 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 664, 4), m_37605, result_mul_37608, None)
    # Getting the type of 'm' (line 664)
    m_37610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 13), 'm')
    int_37611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 15), 'int')
    # Getting the type of 'm' (line 664)
    m_37612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 17), 'm')
    # Applying the binary operator '*' (line 664)
    result_mul_37613 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 15), '*', int_37611, m_37612)
    
    slice_37614 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 664, 4), m_37610, result_mul_37613, None)
    # Storing an element on a container (line 664)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 4), J_37604, ((slice_37609, slice_37614), T_37603))
    
    # Assigning a UnaryOp to a Subscript (line 665):
    
    # Assigning a UnaryOp to a Subscript (line 665):
    
    # Assigning a UnaryOp to a Subscript (line 665):
    
    
    # Call to conj(...): (line 665)
    # Processing the call keyword arguments (line 665)
    kwargs_37617 = {}
    # Getting the type of 'b' (line 665)
    b_37615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 22), 'b', False)
    # Obtaining the member 'conj' of a type (line 665)
    conj_37616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 22), b_37615, 'conj')
    # Calling conj(args, kwargs) (line 665)
    conj_call_result_37618 = invoke(stypy.reporting.localization.Localization(__file__, 665, 22), conj_37616, *[], **kwargs_37617)
    
    # Obtaining the member 'T' of a type (line 665)
    T_37619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 22), conj_call_result_37618, 'T')
    # Applying the 'usub' unary operator (line 665)
    result___neg___37620 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 21), 'usub', T_37619)
    
    # Getting the type of 'J' (line 665)
    J_37621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'J')
    int_37622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 6), 'int')
    # Getting the type of 'm' (line 665)
    m_37623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 8), 'm')
    # Applying the binary operator '*' (line 665)
    result_mul_37624 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 6), '*', int_37622, m_37623)
    
    slice_37625 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 665, 4), result_mul_37624, None, None)
    # Getting the type of 'm' (line 665)
    m_37626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'm')
    int_37627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 14), 'int')
    # Getting the type of 'm' (line 665)
    m_37628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 16), 'm')
    # Applying the binary operator '*' (line 665)
    result_mul_37629 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 14), '*', int_37627, m_37628)
    
    slice_37630 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 665, 4), m_37626, result_mul_37629, None)
    # Storing an element on a container (line 665)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 4), J_37621, ((slice_37625, slice_37630), result___neg___37620))
    
    # Getting the type of 'balanced' (line 667)
    balanced_37631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 7), 'balanced')
    # Testing the type of an if condition (line 667)
    if_condition_37632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 667, 4), balanced_37631)
    # Assigning a type to the variable 'if_condition_37632' (line 667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 4), 'if_condition_37632', if_condition_37632)
    # SSA begins for if statement (line 667)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 670):
    
    # Assigning a BinOp to a Name (line 670):
    
    # Assigning a BinOp to a Name (line 670):
    
    # Call to abs(...): (line 670)
    # Processing the call arguments (line 670)
    # Getting the type of 'H' (line 670)
    H_37635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 19), 'H', False)
    # Processing the call keyword arguments (line 670)
    kwargs_37636 = {}
    # Getting the type of 'np' (line 670)
    np_37633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 12), 'np', False)
    # Obtaining the member 'abs' of a type (line 670)
    abs_37634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 12), np_37633, 'abs')
    # Calling abs(args, kwargs) (line 670)
    abs_call_result_37637 = invoke(stypy.reporting.localization.Localization(__file__, 670, 12), abs_37634, *[H_37635], **kwargs_37636)
    
    
    # Call to abs(...): (line 670)
    # Processing the call arguments (line 670)
    # Getting the type of 'J' (line 670)
    J_37640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 31), 'J', False)
    # Processing the call keyword arguments (line 670)
    kwargs_37641 = {}
    # Getting the type of 'np' (line 670)
    np_37638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 24), 'np', False)
    # Obtaining the member 'abs' of a type (line 670)
    abs_37639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 24), np_37638, 'abs')
    # Calling abs(args, kwargs) (line 670)
    abs_call_result_37642 = invoke(stypy.reporting.localization.Localization(__file__, 670, 24), abs_37639, *[J_37640], **kwargs_37641)
    
    # Applying the binary operator '+' (line 670)
    result_add_37643 = python_operator(stypy.reporting.localization.Localization(__file__, 670, 12), '+', abs_call_result_37637, abs_call_result_37642)
    
    # Assigning a type to the variable 'M' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 8), 'M', result_add_37643)
    
    # Assigning a Num to a Subscript (line 671):
    
    # Assigning a Num to a Subscript (line 671):
    
    # Assigning a Num to a Subscript (line 671):
    float_37644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 37), 'float')
    # Getting the type of 'M' (line 671)
    M_37645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'M')
    
    # Call to diag_indices_from(...): (line 671)
    # Processing the call arguments (line 671)
    # Getting the type of 'M' (line 671)
    M_37648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 31), 'M', False)
    # Processing the call keyword arguments (line 671)
    kwargs_37649 = {}
    # Getting the type of 'np' (line 671)
    np_37646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 10), 'np', False)
    # Obtaining the member 'diag_indices_from' of a type (line 671)
    diag_indices_from_37647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 10), np_37646, 'diag_indices_from')
    # Calling diag_indices_from(args, kwargs) (line 671)
    diag_indices_from_call_result_37650 = invoke(stypy.reporting.localization.Localization(__file__, 671, 10), diag_indices_from_37647, *[M_37648], **kwargs_37649)
    
    # Storing an element on a container (line 671)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 671, 8), M_37645, (diag_indices_from_call_result_37650, float_37644))
    
    # Assigning a Call to a Tuple (line 672):
    
    # Assigning a Subscript to a Name (line 672):
    
    # Assigning a Subscript to a Name (line 672):
    
    # Obtaining the type of the subscript
    int_37651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 8), 'int')
    
    # Call to matrix_balance(...): (line 672)
    # Processing the call arguments (line 672)
    # Getting the type of 'M' (line 672)
    M_37653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 37), 'M', False)
    # Processing the call keyword arguments (line 672)
    int_37654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 49), 'int')
    keyword_37655 = int_37654
    int_37656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 60), 'int')
    keyword_37657 = int_37656
    kwargs_37658 = {'permute': keyword_37657, 'separate': keyword_37655}
    # Getting the type of 'matrix_balance' (line 672)
    matrix_balance_37652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 22), 'matrix_balance', False)
    # Calling matrix_balance(args, kwargs) (line 672)
    matrix_balance_call_result_37659 = invoke(stypy.reporting.localization.Localization(__file__, 672, 22), matrix_balance_37652, *[M_37653], **kwargs_37658)
    
    # Obtaining the member '__getitem__' of a type (line 672)
    getitem___37660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 8), matrix_balance_call_result_37659, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 672)
    subscript_call_result_37661 = invoke(stypy.reporting.localization.Localization(__file__, 672, 8), getitem___37660, int_37651)
    
    # Assigning a type to the variable 'tuple_var_assignment_35918' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'tuple_var_assignment_35918', subscript_call_result_37661)
    
    # Assigning a Subscript to a Name (line 672):
    
    # Assigning a Subscript to a Name (line 672):
    
    # Obtaining the type of the subscript
    int_37662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 8), 'int')
    
    # Call to matrix_balance(...): (line 672)
    # Processing the call arguments (line 672)
    # Getting the type of 'M' (line 672)
    M_37664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 37), 'M', False)
    # Processing the call keyword arguments (line 672)
    int_37665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 49), 'int')
    keyword_37666 = int_37665
    int_37667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 60), 'int')
    keyword_37668 = int_37667
    kwargs_37669 = {'permute': keyword_37668, 'separate': keyword_37666}
    # Getting the type of 'matrix_balance' (line 672)
    matrix_balance_37663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 22), 'matrix_balance', False)
    # Calling matrix_balance(args, kwargs) (line 672)
    matrix_balance_call_result_37670 = invoke(stypy.reporting.localization.Localization(__file__, 672, 22), matrix_balance_37663, *[M_37664], **kwargs_37669)
    
    # Obtaining the member '__getitem__' of a type (line 672)
    getitem___37671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 8), matrix_balance_call_result_37670, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 672)
    subscript_call_result_37672 = invoke(stypy.reporting.localization.Localization(__file__, 672, 8), getitem___37671, int_37662)
    
    # Assigning a type to the variable 'tuple_var_assignment_35919' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'tuple_var_assignment_35919', subscript_call_result_37672)
    
    # Assigning a Name to a Name (line 672):
    
    # Assigning a Name to a Name (line 672):
    # Getting the type of 'tuple_var_assignment_35918' (line 672)
    tuple_var_assignment_35918_37673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'tuple_var_assignment_35918')
    # Assigning a type to the variable '_' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), '_', tuple_var_assignment_35918_37673)
    
    # Assigning a Name to a Tuple (line 672):
    
    # Assigning a Subscript to a Name (line 672):
    
    # Obtaining the type of the subscript
    int_37674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 8), 'int')
    # Getting the type of 'tuple_var_assignment_35919' (line 672)
    tuple_var_assignment_35919_37675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'tuple_var_assignment_35919')
    # Obtaining the member '__getitem__' of a type (line 672)
    getitem___37676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 8), tuple_var_assignment_35919_37675, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 672)
    subscript_call_result_37677 = invoke(stypy.reporting.localization.Localization(__file__, 672, 8), getitem___37676, int_37674)
    
    # Assigning a type to the variable 'tuple_var_assignment_35937' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'tuple_var_assignment_35937', subscript_call_result_37677)
    
    # Assigning a Subscript to a Name (line 672):
    
    # Obtaining the type of the subscript
    int_37678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 8), 'int')
    # Getting the type of 'tuple_var_assignment_35919' (line 672)
    tuple_var_assignment_35919_37679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'tuple_var_assignment_35919')
    # Obtaining the member '__getitem__' of a type (line 672)
    getitem___37680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 8), tuple_var_assignment_35919_37679, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 672)
    subscript_call_result_37681 = invoke(stypy.reporting.localization.Localization(__file__, 672, 8), getitem___37680, int_37678)
    
    # Assigning a type to the variable 'tuple_var_assignment_35938' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'tuple_var_assignment_35938', subscript_call_result_37681)
    
    # Assigning a Name to a Name (line 672):
    # Getting the type of 'tuple_var_assignment_35937' (line 672)
    tuple_var_assignment_35937_37682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'tuple_var_assignment_35937')
    # Assigning a type to the variable 'sca' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'sca', tuple_var_assignment_35937_37682)
    
    # Assigning a Name to a Name (line 672):
    # Getting the type of 'tuple_var_assignment_35938' (line 672)
    tuple_var_assignment_35938_37683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'tuple_var_assignment_35938')
    # Assigning a type to the variable '_' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 17), '_', tuple_var_assignment_35938_37683)
    
    
    
    # Call to allclose(...): (line 674)
    # Processing the call arguments (line 674)
    # Getting the type of 'sca' (line 674)
    sca_37686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 27), 'sca', False)
    
    # Call to ones_like(...): (line 674)
    # Processing the call arguments (line 674)
    # Getting the type of 'sca' (line 674)
    sca_37689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 45), 'sca', False)
    # Processing the call keyword arguments (line 674)
    kwargs_37690 = {}
    # Getting the type of 'np' (line 674)
    np_37687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 32), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 674)
    ones_like_37688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 32), np_37687, 'ones_like')
    # Calling ones_like(args, kwargs) (line 674)
    ones_like_call_result_37691 = invoke(stypy.reporting.localization.Localization(__file__, 674, 32), ones_like_37688, *[sca_37689], **kwargs_37690)
    
    # Processing the call keyword arguments (line 674)
    kwargs_37692 = {}
    # Getting the type of 'np' (line 674)
    np_37684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 15), 'np', False)
    # Obtaining the member 'allclose' of a type (line 674)
    allclose_37685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 15), np_37684, 'allclose')
    # Calling allclose(args, kwargs) (line 674)
    allclose_call_result_37693 = invoke(stypy.reporting.localization.Localization(__file__, 674, 15), allclose_37685, *[sca_37686, ones_like_call_result_37691], **kwargs_37692)
    
    # Applying the 'not' unary operator (line 674)
    result_not__37694 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 11), 'not', allclose_call_result_37693)
    
    # Testing the type of an if condition (line 674)
    if_condition_37695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 674, 8), result_not__37694)
    # Assigning a type to the variable 'if_condition_37695' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'if_condition_37695', if_condition_37695)
    # SSA begins for if statement (line 674)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 677):
    
    # Assigning a Call to a Name (line 677):
    
    # Assigning a Call to a Name (line 677):
    
    # Call to log2(...): (line 677)
    # Processing the call arguments (line 677)
    # Getting the type of 'sca' (line 677)
    sca_37698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 26), 'sca', False)
    # Processing the call keyword arguments (line 677)
    kwargs_37699 = {}
    # Getting the type of 'np' (line 677)
    np_37696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 18), 'np', False)
    # Obtaining the member 'log2' of a type (line 677)
    log2_37697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 18), np_37696, 'log2')
    # Calling log2(args, kwargs) (line 677)
    log2_call_result_37700 = invoke(stypy.reporting.localization.Localization(__file__, 677, 18), log2_37697, *[sca_37698], **kwargs_37699)
    
    # Assigning a type to the variable 'sca' (line 677)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 12), 'sca', log2_call_result_37700)
    
    # Assigning a Call to a Name (line 679):
    
    # Assigning a Call to a Name (line 679):
    
    # Assigning a Call to a Name (line 679):
    
    # Call to round(...): (line 679)
    # Processing the call arguments (line 679)
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 679)
    m_37703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 30), 'm', False)
    int_37704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 32), 'int')
    # Getting the type of 'm' (line 679)
    m_37705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 34), 'm', False)
    # Applying the binary operator '*' (line 679)
    result_mul_37706 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 32), '*', int_37704, m_37705)
    
    slice_37707 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 679, 26), m_37703, result_mul_37706, None)
    # Getting the type of 'sca' (line 679)
    sca_37708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 26), 'sca', False)
    # Obtaining the member '__getitem__' of a type (line 679)
    getitem___37709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 26), sca_37708, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 679)
    subscript_call_result_37710 = invoke(stypy.reporting.localization.Localization(__file__, 679, 26), getitem___37709, slice_37707)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 679)
    m_37711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 44), 'm', False)
    slice_37712 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 679, 39), None, m_37711, None)
    # Getting the type of 'sca' (line 679)
    sca_37713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 39), 'sca', False)
    # Obtaining the member '__getitem__' of a type (line 679)
    getitem___37714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 39), sca_37713, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 679)
    subscript_call_result_37715 = invoke(stypy.reporting.localization.Localization(__file__, 679, 39), getitem___37714, slice_37712)
    
    # Applying the binary operator '-' (line 679)
    result_sub_37716 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 26), '-', subscript_call_result_37710, subscript_call_result_37715)
    
    int_37717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 48), 'int')
    # Applying the binary operator 'div' (line 679)
    result_div_37718 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 25), 'div', result_sub_37716, int_37717)
    
    # Processing the call keyword arguments (line 679)
    kwargs_37719 = {}
    # Getting the type of 'np' (line 679)
    np_37701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'np', False)
    # Obtaining the member 'round' of a type (line 679)
    round_37702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 16), np_37701, 'round')
    # Calling round(args, kwargs) (line 679)
    round_call_result_37720 = invoke(stypy.reporting.localization.Localization(__file__, 679, 16), round_37702, *[result_div_37718], **kwargs_37719)
    
    # Assigning a type to the variable 's' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 12), 's', round_call_result_37720)
    
    # Assigning a BinOp to a Name (line 680):
    
    # Assigning a BinOp to a Name (line 680):
    
    # Assigning a BinOp to a Name (line 680):
    int_37721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 18), 'int')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 680)
    tuple_37722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 680)
    # Adding element type (line 680)
    # Getting the type of 's' (line 680)
    s_37723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 29), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 29), tuple_37722, s_37723)
    # Adding element type (line 680)
    
    # Getting the type of 's' (line 680)
    s_37724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 33), 's')
    # Applying the 'usub' unary operator (line 680)
    result___neg___37725 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 32), 'usub', s_37724)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 29), tuple_37722, result___neg___37725)
    # Adding element type (line 680)
    
    # Obtaining the type of the subscript
    int_37726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 40), 'int')
    # Getting the type of 'm' (line 680)
    m_37727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 42), 'm')
    # Applying the binary operator '*' (line 680)
    result_mul_37728 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 40), '*', int_37726, m_37727)
    
    slice_37729 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 680, 36), result_mul_37728, None, None)
    # Getting the type of 'sca' (line 680)
    sca_37730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 36), 'sca')
    # Obtaining the member '__getitem__' of a type (line 680)
    getitem___37731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 36), sca_37730, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 680)
    subscript_call_result_37732 = invoke(stypy.reporting.localization.Localization(__file__, 680, 36), getitem___37731, slice_37729)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 29), tuple_37722, subscript_call_result_37732)
    
    # Getting the type of 'np' (line 680)
    np_37733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 23), 'np')
    # Obtaining the member 'r_' of a type (line 680)
    r__37734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 23), np_37733, 'r_')
    # Obtaining the member '__getitem__' of a type (line 680)
    getitem___37735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 23), r__37734, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 680)
    subscript_call_result_37736 = invoke(stypy.reporting.localization.Localization(__file__, 680, 23), getitem___37735, tuple_37722)
    
    # Applying the binary operator '**' (line 680)
    result_pow_37737 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 18), '**', int_37721, subscript_call_result_37736)
    
    # Assigning a type to the variable 'sca' (line 680)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 12), 'sca', result_pow_37737)
    
    # Assigning a BinOp to a Name (line 682):
    
    # Assigning a BinOp to a Name (line 682):
    
    # Assigning a BinOp to a Name (line 682):
    
    # Obtaining the type of the subscript
    slice_37738 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 682, 26), None, None, None)
    # Getting the type of 'None' (line 682)
    None_37739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 33), 'None')
    # Getting the type of 'sca' (line 682)
    sca_37740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 26), 'sca')
    # Obtaining the member '__getitem__' of a type (line 682)
    getitem___37741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 26), sca_37740, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 682)
    subscript_call_result_37742 = invoke(stypy.reporting.localization.Localization(__file__, 682, 26), getitem___37741, (slice_37738, None_37739))
    
    
    # Call to reciprocal(...): (line 682)
    # Processing the call arguments (line 682)
    # Getting the type of 'sca' (line 682)
    sca_37745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 55), 'sca', False)
    # Processing the call keyword arguments (line 682)
    kwargs_37746 = {}
    # Getting the type of 'np' (line 682)
    np_37743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 41), 'np', False)
    # Obtaining the member 'reciprocal' of a type (line 682)
    reciprocal_37744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 41), np_37743, 'reciprocal')
    # Calling reciprocal(args, kwargs) (line 682)
    reciprocal_call_result_37747 = invoke(stypy.reporting.localization.Localization(__file__, 682, 41), reciprocal_37744, *[sca_37745], **kwargs_37746)
    
    # Applying the binary operator '*' (line 682)
    result_mul_37748 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 26), '*', subscript_call_result_37742, reciprocal_call_result_37747)
    
    # Assigning a type to the variable 'elwisescale' (line 682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 12), 'elwisescale', result_mul_37748)
    
    # Getting the type of 'H' (line 683)
    H_37749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 12), 'H')
    # Getting the type of 'elwisescale' (line 683)
    elwisescale_37750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 17), 'elwisescale')
    # Applying the binary operator '*=' (line 683)
    result_imul_37751 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 12), '*=', H_37749, elwisescale_37750)
    # Assigning a type to the variable 'H' (line 683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 12), 'H', result_imul_37751)
    
    
    # Getting the type of 'J' (line 684)
    J_37752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 12), 'J')
    # Getting the type of 'elwisescale' (line 684)
    elwisescale_37753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 17), 'elwisescale')
    # Applying the binary operator '*=' (line 684)
    result_imul_37754 = python_operator(stypy.reporting.localization.Localization(__file__, 684, 12), '*=', J_37752, elwisescale_37753)
    # Assigning a type to the variable 'J' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 12), 'J', result_imul_37754)
    
    # SSA join for if statement (line 674)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 667)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 687):
    
    # Assigning a Subscript to a Name (line 687):
    
    # Assigning a Subscript to a Name (line 687):
    
    # Obtaining the type of the subscript
    int_37755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 4), 'int')
    
    # Call to qr(...): (line 687)
    # Processing the call arguments (line 687)
    
    # Obtaining the type of the subscript
    slice_37757 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 687, 20), None, None, None)
    
    # Getting the type of 'n' (line 687)
    n_37758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 26), 'n', False)
    # Applying the 'usub' unary operator (line 687)
    result___neg___37759 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 25), 'usub', n_37758)
    
    slice_37760 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 687, 20), result___neg___37759, None, None)
    # Getting the type of 'H' (line 687)
    H_37761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 20), 'H', False)
    # Obtaining the member '__getitem__' of a type (line 687)
    getitem___37762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 20), H_37761, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 687)
    subscript_call_result_37763 = invoke(stypy.reporting.localization.Localization(__file__, 687, 20), getitem___37762, (slice_37757, slice_37760))
    
    # Processing the call keyword arguments (line 687)
    kwargs_37764 = {}
    # Getting the type of 'qr' (line 687)
    qr_37756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 17), 'qr', False)
    # Calling qr(args, kwargs) (line 687)
    qr_call_result_37765 = invoke(stypy.reporting.localization.Localization(__file__, 687, 17), qr_37756, *[subscript_call_result_37763], **kwargs_37764)
    
    # Obtaining the member '__getitem__' of a type (line 687)
    getitem___37766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 4), qr_call_result_37765, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 687)
    subscript_call_result_37767 = invoke(stypy.reporting.localization.Localization(__file__, 687, 4), getitem___37766, int_37755)
    
    # Assigning a type to the variable 'tuple_var_assignment_35920' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'tuple_var_assignment_35920', subscript_call_result_37767)
    
    # Assigning a Subscript to a Name (line 687):
    
    # Assigning a Subscript to a Name (line 687):
    
    # Obtaining the type of the subscript
    int_37768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 4), 'int')
    
    # Call to qr(...): (line 687)
    # Processing the call arguments (line 687)
    
    # Obtaining the type of the subscript
    slice_37770 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 687, 20), None, None, None)
    
    # Getting the type of 'n' (line 687)
    n_37771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 26), 'n', False)
    # Applying the 'usub' unary operator (line 687)
    result___neg___37772 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 25), 'usub', n_37771)
    
    slice_37773 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 687, 20), result___neg___37772, None, None)
    # Getting the type of 'H' (line 687)
    H_37774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 20), 'H', False)
    # Obtaining the member '__getitem__' of a type (line 687)
    getitem___37775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 20), H_37774, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 687)
    subscript_call_result_37776 = invoke(stypy.reporting.localization.Localization(__file__, 687, 20), getitem___37775, (slice_37770, slice_37773))
    
    # Processing the call keyword arguments (line 687)
    kwargs_37777 = {}
    # Getting the type of 'qr' (line 687)
    qr_37769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 17), 'qr', False)
    # Calling qr(args, kwargs) (line 687)
    qr_call_result_37778 = invoke(stypy.reporting.localization.Localization(__file__, 687, 17), qr_37769, *[subscript_call_result_37776], **kwargs_37777)
    
    # Obtaining the member '__getitem__' of a type (line 687)
    getitem___37779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 4), qr_call_result_37778, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 687)
    subscript_call_result_37780 = invoke(stypy.reporting.localization.Localization(__file__, 687, 4), getitem___37779, int_37768)
    
    # Assigning a type to the variable 'tuple_var_assignment_35921' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'tuple_var_assignment_35921', subscript_call_result_37780)
    
    # Assigning a Name to a Name (line 687):
    
    # Assigning a Name to a Name (line 687):
    # Getting the type of 'tuple_var_assignment_35920' (line 687)
    tuple_var_assignment_35920_37781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'tuple_var_assignment_35920')
    # Assigning a type to the variable 'q_of_qr' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'q_of_qr', tuple_var_assignment_35920_37781)
    
    # Assigning a Name to a Name (line 687):
    
    # Assigning a Name to a Name (line 687):
    # Getting the type of 'tuple_var_assignment_35921' (line 687)
    tuple_var_assignment_35921_37782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'tuple_var_assignment_35921')
    # Assigning a type to the variable '_' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 13), '_', tuple_var_assignment_35921_37782)
    
    # Assigning a Call to a Name (line 688):
    
    # Assigning a Call to a Name (line 688):
    
    # Assigning a Call to a Name (line 688):
    
    # Call to dot(...): (line 688)
    # Processing the call arguments (line 688)
    
    # Obtaining the type of the subscript
    slice_37794 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 688, 36), None, None, None)
    int_37795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 42), 'int')
    # Getting the type of 'm' (line 688)
    m_37796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 44), 'm', False)
    # Applying the binary operator '*' (line 688)
    result_mul_37797 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 42), '*', int_37795, m_37796)
    
    slice_37798 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 688, 36), None, result_mul_37797, None)
    # Getting the type of 'H' (line 688)
    H_37799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 36), 'H', False)
    # Obtaining the member '__getitem__' of a type (line 688)
    getitem___37800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 36), H_37799, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 688)
    subscript_call_result_37801 = invoke(stypy.reporting.localization.Localization(__file__, 688, 36), getitem___37800, (slice_37794, slice_37798))
    
    # Processing the call keyword arguments (line 688)
    kwargs_37802 = {}
    
    # Call to conj(...): (line 688)
    # Processing the call keyword arguments (line 688)
    kwargs_37790 = {}
    
    # Obtaining the type of the subscript
    slice_37783 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 688, 8), None, None, None)
    # Getting the type of 'n' (line 688)
    n_37784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 19), 'n', False)
    slice_37785 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 688, 8), n_37784, None, None)
    # Getting the type of 'q_of_qr' (line 688)
    q_of_qr_37786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 8), 'q_of_qr', False)
    # Obtaining the member '__getitem__' of a type (line 688)
    getitem___37787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 8), q_of_qr_37786, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 688)
    subscript_call_result_37788 = invoke(stypy.reporting.localization.Localization(__file__, 688, 8), getitem___37787, (slice_37783, slice_37785))
    
    # Obtaining the member 'conj' of a type (line 688)
    conj_37789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 8), subscript_call_result_37788, 'conj')
    # Calling conj(args, kwargs) (line 688)
    conj_call_result_37791 = invoke(stypy.reporting.localization.Localization(__file__, 688, 8), conj_37789, *[], **kwargs_37790)
    
    # Obtaining the member 'T' of a type (line 688)
    T_37792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 8), conj_call_result_37791, 'T')
    # Obtaining the member 'dot' of a type (line 688)
    dot_37793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 8), T_37792, 'dot')
    # Calling dot(args, kwargs) (line 688)
    dot_call_result_37803 = invoke(stypy.reporting.localization.Localization(__file__, 688, 8), dot_37793, *[subscript_call_result_37801], **kwargs_37802)
    
    # Assigning a type to the variable 'H' (line 688)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 4), 'H', dot_call_result_37803)
    
    # Assigning a Call to a Name (line 689):
    
    # Assigning a Call to a Name (line 689):
    
    # Assigning a Call to a Name (line 689):
    
    # Call to dot(...): (line 689)
    # Processing the call arguments (line 689)
    
    # Obtaining the type of the subscript
    slice_37815 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 689, 36), None, None, None)
    int_37816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 42), 'int')
    # Getting the type of 'm' (line 689)
    m_37817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 44), 'm', False)
    # Applying the binary operator '*' (line 689)
    result_mul_37818 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 42), '*', int_37816, m_37817)
    
    slice_37819 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 689, 36), None, result_mul_37818, None)
    # Getting the type of 'J' (line 689)
    J_37820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 36), 'J', False)
    # Obtaining the member '__getitem__' of a type (line 689)
    getitem___37821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 36), J_37820, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 689)
    subscript_call_result_37822 = invoke(stypy.reporting.localization.Localization(__file__, 689, 36), getitem___37821, (slice_37815, slice_37819))
    
    # Processing the call keyword arguments (line 689)
    kwargs_37823 = {}
    
    # Call to conj(...): (line 689)
    # Processing the call keyword arguments (line 689)
    kwargs_37811 = {}
    
    # Obtaining the type of the subscript
    slice_37804 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 689, 8), None, None, None)
    # Getting the type of 'n' (line 689)
    n_37805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 19), 'n', False)
    slice_37806 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 689, 8), n_37805, None, None)
    # Getting the type of 'q_of_qr' (line 689)
    q_of_qr_37807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'q_of_qr', False)
    # Obtaining the member '__getitem__' of a type (line 689)
    getitem___37808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), q_of_qr_37807, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 689)
    subscript_call_result_37809 = invoke(stypy.reporting.localization.Localization(__file__, 689, 8), getitem___37808, (slice_37804, slice_37806))
    
    # Obtaining the member 'conj' of a type (line 689)
    conj_37810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), subscript_call_result_37809, 'conj')
    # Calling conj(args, kwargs) (line 689)
    conj_call_result_37812 = invoke(stypy.reporting.localization.Localization(__file__, 689, 8), conj_37810, *[], **kwargs_37811)
    
    # Obtaining the member 'T' of a type (line 689)
    T_37813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), conj_call_result_37812, 'T')
    # Obtaining the member 'dot' of a type (line 689)
    dot_37814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), T_37813, 'dot')
    # Calling dot(args, kwargs) (line 689)
    dot_call_result_37824 = invoke(stypy.reporting.localization.Localization(__file__, 689, 8), dot_37814, *[subscript_call_result_37822], **kwargs_37823)
    
    # Assigning a type to the variable 'J' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'J', dot_call_result_37824)
    
    # Assigning a IfExp to a Name (line 692):
    
    # Assigning a IfExp to a Name (line 692):
    
    # Assigning a IfExp to a Name (line 692):
    
    
    # Getting the type of 'r_or_c' (line 692)
    r_or_c_37825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 24), 'r_or_c')
    # Getting the type of 'float' (line 692)
    float_37826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 34), 'float')
    # Applying the binary operator '==' (line 692)
    result_eq_37827 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 24), '==', r_or_c_37825, float_37826)
    
    # Testing the type of an if expression (line 692)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 692, 14), result_eq_37827)
    # SSA begins for if expression (line 692)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    str_37828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 14), 'str', 'real')
    # SSA branch for the else part of an if expression (line 692)
    module_type_store.open_ssa_branch('if expression else')
    str_37829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 45), 'str', 'complex')
    # SSA join for if expression (line 692)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_37830 = union_type.UnionType.add(str_37828, str_37829)
    
    # Assigning a type to the variable 'out_str' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'out_str', if_exp_37830)
    
    # Assigning a Call to a Tuple (line 694):
    
    # Assigning a Subscript to a Name (line 694):
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_37831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    
    # Call to ordqz(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'H' (line 694)
    H_37833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 29), 'H', False)
    # Getting the type of 'J' (line 694)
    J_37834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 32), 'J', False)
    # Processing the call keyword arguments (line 694)
    str_37835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 40), 'str', 'iuc')
    keyword_37836 = str_37835
    # Getting the type of 'True' (line 695)
    True_37837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 41), 'True', False)
    keyword_37838 = True_37837
    # Getting the type of 'True' (line 696)
    True_37839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 41), 'True', False)
    keyword_37840 = True_37839
    # Getting the type of 'False' (line 697)
    False_37841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 42), 'False', False)
    keyword_37842 = False_37841
    # Getting the type of 'out_str' (line 698)
    out_str_37843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 36), 'out_str', False)
    keyword_37844 = out_str_37843
    kwargs_37845 = {'sort': keyword_37836, 'output': keyword_37844, 'overwrite_a': keyword_37838, 'check_finite': keyword_37842, 'overwrite_b': keyword_37840}
    # Getting the type of 'ordqz' (line 694)
    ordqz_37832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 23), 'ordqz', False)
    # Calling ordqz(args, kwargs) (line 694)
    ordqz_call_result_37846 = invoke(stypy.reporting.localization.Localization(__file__, 694, 23), ordqz_37832, *[H_37833, J_37834], **kwargs_37845)
    
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___37847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), ordqz_call_result_37846, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_37848 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___37847, int_37831)
    
    # Assigning a type to the variable 'tuple_var_assignment_35922' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_35922', subscript_call_result_37848)
    
    # Assigning a Subscript to a Name (line 694):
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_37849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    
    # Call to ordqz(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'H' (line 694)
    H_37851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 29), 'H', False)
    # Getting the type of 'J' (line 694)
    J_37852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 32), 'J', False)
    # Processing the call keyword arguments (line 694)
    str_37853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 40), 'str', 'iuc')
    keyword_37854 = str_37853
    # Getting the type of 'True' (line 695)
    True_37855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 41), 'True', False)
    keyword_37856 = True_37855
    # Getting the type of 'True' (line 696)
    True_37857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 41), 'True', False)
    keyword_37858 = True_37857
    # Getting the type of 'False' (line 697)
    False_37859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 42), 'False', False)
    keyword_37860 = False_37859
    # Getting the type of 'out_str' (line 698)
    out_str_37861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 36), 'out_str', False)
    keyword_37862 = out_str_37861
    kwargs_37863 = {'sort': keyword_37854, 'output': keyword_37862, 'overwrite_a': keyword_37856, 'check_finite': keyword_37860, 'overwrite_b': keyword_37858}
    # Getting the type of 'ordqz' (line 694)
    ordqz_37850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 23), 'ordqz', False)
    # Calling ordqz(args, kwargs) (line 694)
    ordqz_call_result_37864 = invoke(stypy.reporting.localization.Localization(__file__, 694, 23), ordqz_37850, *[H_37851, J_37852], **kwargs_37863)
    
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___37865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), ordqz_call_result_37864, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_37866 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___37865, int_37849)
    
    # Assigning a type to the variable 'tuple_var_assignment_35923' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_35923', subscript_call_result_37866)
    
    # Assigning a Subscript to a Name (line 694):
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_37867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    
    # Call to ordqz(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'H' (line 694)
    H_37869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 29), 'H', False)
    # Getting the type of 'J' (line 694)
    J_37870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 32), 'J', False)
    # Processing the call keyword arguments (line 694)
    str_37871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 40), 'str', 'iuc')
    keyword_37872 = str_37871
    # Getting the type of 'True' (line 695)
    True_37873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 41), 'True', False)
    keyword_37874 = True_37873
    # Getting the type of 'True' (line 696)
    True_37875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 41), 'True', False)
    keyword_37876 = True_37875
    # Getting the type of 'False' (line 697)
    False_37877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 42), 'False', False)
    keyword_37878 = False_37877
    # Getting the type of 'out_str' (line 698)
    out_str_37879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 36), 'out_str', False)
    keyword_37880 = out_str_37879
    kwargs_37881 = {'sort': keyword_37872, 'output': keyword_37880, 'overwrite_a': keyword_37874, 'check_finite': keyword_37878, 'overwrite_b': keyword_37876}
    # Getting the type of 'ordqz' (line 694)
    ordqz_37868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 23), 'ordqz', False)
    # Calling ordqz(args, kwargs) (line 694)
    ordqz_call_result_37882 = invoke(stypy.reporting.localization.Localization(__file__, 694, 23), ordqz_37868, *[H_37869, J_37870], **kwargs_37881)
    
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___37883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), ordqz_call_result_37882, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_37884 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___37883, int_37867)
    
    # Assigning a type to the variable 'tuple_var_assignment_35924' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_35924', subscript_call_result_37884)
    
    # Assigning a Subscript to a Name (line 694):
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_37885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    
    # Call to ordqz(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'H' (line 694)
    H_37887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 29), 'H', False)
    # Getting the type of 'J' (line 694)
    J_37888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 32), 'J', False)
    # Processing the call keyword arguments (line 694)
    str_37889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 40), 'str', 'iuc')
    keyword_37890 = str_37889
    # Getting the type of 'True' (line 695)
    True_37891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 41), 'True', False)
    keyword_37892 = True_37891
    # Getting the type of 'True' (line 696)
    True_37893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 41), 'True', False)
    keyword_37894 = True_37893
    # Getting the type of 'False' (line 697)
    False_37895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 42), 'False', False)
    keyword_37896 = False_37895
    # Getting the type of 'out_str' (line 698)
    out_str_37897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 36), 'out_str', False)
    keyword_37898 = out_str_37897
    kwargs_37899 = {'sort': keyword_37890, 'output': keyword_37898, 'overwrite_a': keyword_37892, 'check_finite': keyword_37896, 'overwrite_b': keyword_37894}
    # Getting the type of 'ordqz' (line 694)
    ordqz_37886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 23), 'ordqz', False)
    # Calling ordqz(args, kwargs) (line 694)
    ordqz_call_result_37900 = invoke(stypy.reporting.localization.Localization(__file__, 694, 23), ordqz_37886, *[H_37887, J_37888], **kwargs_37899)
    
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___37901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), ordqz_call_result_37900, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_37902 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___37901, int_37885)
    
    # Assigning a type to the variable 'tuple_var_assignment_35925' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_35925', subscript_call_result_37902)
    
    # Assigning a Subscript to a Name (line 694):
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_37903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    
    # Call to ordqz(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'H' (line 694)
    H_37905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 29), 'H', False)
    # Getting the type of 'J' (line 694)
    J_37906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 32), 'J', False)
    # Processing the call keyword arguments (line 694)
    str_37907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 40), 'str', 'iuc')
    keyword_37908 = str_37907
    # Getting the type of 'True' (line 695)
    True_37909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 41), 'True', False)
    keyword_37910 = True_37909
    # Getting the type of 'True' (line 696)
    True_37911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 41), 'True', False)
    keyword_37912 = True_37911
    # Getting the type of 'False' (line 697)
    False_37913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 42), 'False', False)
    keyword_37914 = False_37913
    # Getting the type of 'out_str' (line 698)
    out_str_37915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 36), 'out_str', False)
    keyword_37916 = out_str_37915
    kwargs_37917 = {'sort': keyword_37908, 'output': keyword_37916, 'overwrite_a': keyword_37910, 'check_finite': keyword_37914, 'overwrite_b': keyword_37912}
    # Getting the type of 'ordqz' (line 694)
    ordqz_37904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 23), 'ordqz', False)
    # Calling ordqz(args, kwargs) (line 694)
    ordqz_call_result_37918 = invoke(stypy.reporting.localization.Localization(__file__, 694, 23), ordqz_37904, *[H_37905, J_37906], **kwargs_37917)
    
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___37919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), ordqz_call_result_37918, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_37920 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___37919, int_37903)
    
    # Assigning a type to the variable 'tuple_var_assignment_35926' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_35926', subscript_call_result_37920)
    
    # Assigning a Subscript to a Name (line 694):
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_37921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    
    # Call to ordqz(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'H' (line 694)
    H_37923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 29), 'H', False)
    # Getting the type of 'J' (line 694)
    J_37924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 32), 'J', False)
    # Processing the call keyword arguments (line 694)
    str_37925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 40), 'str', 'iuc')
    keyword_37926 = str_37925
    # Getting the type of 'True' (line 695)
    True_37927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 41), 'True', False)
    keyword_37928 = True_37927
    # Getting the type of 'True' (line 696)
    True_37929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 41), 'True', False)
    keyword_37930 = True_37929
    # Getting the type of 'False' (line 697)
    False_37931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 42), 'False', False)
    keyword_37932 = False_37931
    # Getting the type of 'out_str' (line 698)
    out_str_37933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 36), 'out_str', False)
    keyword_37934 = out_str_37933
    kwargs_37935 = {'sort': keyword_37926, 'output': keyword_37934, 'overwrite_a': keyword_37928, 'check_finite': keyword_37932, 'overwrite_b': keyword_37930}
    # Getting the type of 'ordqz' (line 694)
    ordqz_37922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 23), 'ordqz', False)
    # Calling ordqz(args, kwargs) (line 694)
    ordqz_call_result_37936 = invoke(stypy.reporting.localization.Localization(__file__, 694, 23), ordqz_37922, *[H_37923, J_37924], **kwargs_37935)
    
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___37937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), ordqz_call_result_37936, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_37938 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___37937, int_37921)
    
    # Assigning a type to the variable 'tuple_var_assignment_35927' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_35927', subscript_call_result_37938)
    
    # Assigning a Name to a Name (line 694):
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_35922' (line 694)
    tuple_var_assignment_35922_37939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_35922')
    # Assigning a type to the variable '_' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), '_', tuple_var_assignment_35922_37939)
    
    # Assigning a Name to a Name (line 694):
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_35923' (line 694)
    tuple_var_assignment_35923_37940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_35923')
    # Assigning a type to the variable '_' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 7), '_', tuple_var_assignment_35923_37940)
    
    # Assigning a Name to a Name (line 694):
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_35924' (line 694)
    tuple_var_assignment_35924_37941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_35924')
    # Assigning a type to the variable '_' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 10), '_', tuple_var_assignment_35924_37941)
    
    # Assigning a Name to a Name (line 694):
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_35925' (line 694)
    tuple_var_assignment_35925_37942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_35925')
    # Assigning a type to the variable '_' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 13), '_', tuple_var_assignment_35925_37942)
    
    # Assigning a Name to a Name (line 694):
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_35926' (line 694)
    tuple_var_assignment_35926_37943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_35926')
    # Assigning a type to the variable '_' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), '_', tuple_var_assignment_35926_37943)
    
    # Assigning a Name to a Name (line 694):
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_35927' (line 694)
    tuple_var_assignment_35927_37944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_35927')
    # Assigning a type to the variable 'u' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 19), 'u', tuple_var_assignment_35927_37944)
    
    # Type idiom detected: calculating its left and rigth part (line 701)
    # Getting the type of 'e' (line 701)
    e_37945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'e')
    # Getting the type of 'None' (line 701)
    None_37946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 16), 'None')
    
    (may_be_37947, more_types_in_union_37948) = may_not_be_none(e_37945, None_37946)

    if may_be_37947:

        if more_types_in_union_37948:
            # Runtime conditional SSA (line 701)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Tuple (line 702):
        
        # Assigning a Subscript to a Name (line 702):
        
        # Assigning a Subscript to a Name (line 702):
        
        # Obtaining the type of the subscript
        int_37949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 8), 'int')
        
        # Call to qr(...): (line 702)
        # Processing the call arguments (line 702)
        
        # Call to vstack(...): (line 702)
        # Processing the call arguments (line 702)
        
        # Obtaining an instance of the builtin type 'tuple' (line 702)
        tuple_37953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 702)
        # Adding element type (line 702)
        
        # Call to dot(...): (line 702)
        # Processing the call arguments (line 702)
        
        # Obtaining the type of the subscript
        # Getting the type of 'm' (line 702)
        m_37956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 38), 'm', False)
        slice_37957 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 702, 35), None, m_37956, None)
        # Getting the type of 'm' (line 702)
        m_37958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 42), 'm', False)
        slice_37959 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 702, 35), None, m_37958, None)
        # Getting the type of 'u' (line 702)
        u_37960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 35), 'u', False)
        # Obtaining the member '__getitem__' of a type (line 702)
        getitem___37961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 35), u_37960, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 702)
        subscript_call_result_37962 = invoke(stypy.reporting.localization.Localization(__file__, 702, 35), getitem___37961, (slice_37957, slice_37959))
        
        # Processing the call keyword arguments (line 702)
        kwargs_37963 = {}
        # Getting the type of 'e' (line 702)
        e_37954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 29), 'e', False)
        # Obtaining the member 'dot' of a type (line 702)
        dot_37955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 29), e_37954, 'dot')
        # Calling dot(args, kwargs) (line 702)
        dot_call_result_37964 = invoke(stypy.reporting.localization.Localization(__file__, 702, 29), dot_37955, *[subscript_call_result_37962], **kwargs_37963)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 29), tuple_37953, dot_call_result_37964)
        # Adding element type (line 702)
        
        # Obtaining the type of the subscript
        # Getting the type of 'm' (line 702)
        m_37965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 49), 'm', False)
        slice_37966 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 702, 47), m_37965, None, None)
        # Getting the type of 'm' (line 702)
        m_37967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 54), 'm', False)
        slice_37968 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 702, 47), None, m_37967, None)
        # Getting the type of 'u' (line 702)
        u_37969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 47), 'u', False)
        # Obtaining the member '__getitem__' of a type (line 702)
        getitem___37970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 47), u_37969, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 702)
        subscript_call_result_37971 = invoke(stypy.reporting.localization.Localization(__file__, 702, 47), getitem___37970, (slice_37966, slice_37968))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 29), tuple_37953, subscript_call_result_37971)
        
        # Processing the call keyword arguments (line 702)
        kwargs_37972 = {}
        # Getting the type of 'np' (line 702)
        np_37951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 18), 'np', False)
        # Obtaining the member 'vstack' of a type (line 702)
        vstack_37952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 18), np_37951, 'vstack')
        # Calling vstack(args, kwargs) (line 702)
        vstack_call_result_37973 = invoke(stypy.reporting.localization.Localization(__file__, 702, 18), vstack_37952, *[tuple_37953], **kwargs_37972)
        
        # Processing the call keyword arguments (line 702)
        kwargs_37974 = {}
        # Getting the type of 'qr' (line 702)
        qr_37950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 15), 'qr', False)
        # Calling qr(args, kwargs) (line 702)
        qr_call_result_37975 = invoke(stypy.reporting.localization.Localization(__file__, 702, 15), qr_37950, *[vstack_call_result_37973], **kwargs_37974)
        
        # Obtaining the member '__getitem__' of a type (line 702)
        getitem___37976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 8), qr_call_result_37975, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 702)
        subscript_call_result_37977 = invoke(stypy.reporting.localization.Localization(__file__, 702, 8), getitem___37976, int_37949)
        
        # Assigning a type to the variable 'tuple_var_assignment_35928' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'tuple_var_assignment_35928', subscript_call_result_37977)
        
        # Assigning a Subscript to a Name (line 702):
        
        # Assigning a Subscript to a Name (line 702):
        
        # Obtaining the type of the subscript
        int_37978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 8), 'int')
        
        # Call to qr(...): (line 702)
        # Processing the call arguments (line 702)
        
        # Call to vstack(...): (line 702)
        # Processing the call arguments (line 702)
        
        # Obtaining an instance of the builtin type 'tuple' (line 702)
        tuple_37982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 702)
        # Adding element type (line 702)
        
        # Call to dot(...): (line 702)
        # Processing the call arguments (line 702)
        
        # Obtaining the type of the subscript
        # Getting the type of 'm' (line 702)
        m_37985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 38), 'm', False)
        slice_37986 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 702, 35), None, m_37985, None)
        # Getting the type of 'm' (line 702)
        m_37987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 42), 'm', False)
        slice_37988 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 702, 35), None, m_37987, None)
        # Getting the type of 'u' (line 702)
        u_37989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 35), 'u', False)
        # Obtaining the member '__getitem__' of a type (line 702)
        getitem___37990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 35), u_37989, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 702)
        subscript_call_result_37991 = invoke(stypy.reporting.localization.Localization(__file__, 702, 35), getitem___37990, (slice_37986, slice_37988))
        
        # Processing the call keyword arguments (line 702)
        kwargs_37992 = {}
        # Getting the type of 'e' (line 702)
        e_37983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 29), 'e', False)
        # Obtaining the member 'dot' of a type (line 702)
        dot_37984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 29), e_37983, 'dot')
        # Calling dot(args, kwargs) (line 702)
        dot_call_result_37993 = invoke(stypy.reporting.localization.Localization(__file__, 702, 29), dot_37984, *[subscript_call_result_37991], **kwargs_37992)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 29), tuple_37982, dot_call_result_37993)
        # Adding element type (line 702)
        
        # Obtaining the type of the subscript
        # Getting the type of 'm' (line 702)
        m_37994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 49), 'm', False)
        slice_37995 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 702, 47), m_37994, None, None)
        # Getting the type of 'm' (line 702)
        m_37996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 54), 'm', False)
        slice_37997 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 702, 47), None, m_37996, None)
        # Getting the type of 'u' (line 702)
        u_37998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 47), 'u', False)
        # Obtaining the member '__getitem__' of a type (line 702)
        getitem___37999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 47), u_37998, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 702)
        subscript_call_result_38000 = invoke(stypy.reporting.localization.Localization(__file__, 702, 47), getitem___37999, (slice_37995, slice_37997))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 29), tuple_37982, subscript_call_result_38000)
        
        # Processing the call keyword arguments (line 702)
        kwargs_38001 = {}
        # Getting the type of 'np' (line 702)
        np_37980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 18), 'np', False)
        # Obtaining the member 'vstack' of a type (line 702)
        vstack_37981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 18), np_37980, 'vstack')
        # Calling vstack(args, kwargs) (line 702)
        vstack_call_result_38002 = invoke(stypy.reporting.localization.Localization(__file__, 702, 18), vstack_37981, *[tuple_37982], **kwargs_38001)
        
        # Processing the call keyword arguments (line 702)
        kwargs_38003 = {}
        # Getting the type of 'qr' (line 702)
        qr_37979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 15), 'qr', False)
        # Calling qr(args, kwargs) (line 702)
        qr_call_result_38004 = invoke(stypy.reporting.localization.Localization(__file__, 702, 15), qr_37979, *[vstack_call_result_38002], **kwargs_38003)
        
        # Obtaining the member '__getitem__' of a type (line 702)
        getitem___38005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 8), qr_call_result_38004, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 702)
        subscript_call_result_38006 = invoke(stypy.reporting.localization.Localization(__file__, 702, 8), getitem___38005, int_37978)
        
        # Assigning a type to the variable 'tuple_var_assignment_35929' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'tuple_var_assignment_35929', subscript_call_result_38006)
        
        # Assigning a Name to a Name (line 702):
        
        # Assigning a Name to a Name (line 702):
        # Getting the type of 'tuple_var_assignment_35928' (line 702)
        tuple_var_assignment_35928_38007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'tuple_var_assignment_35928')
        # Assigning a type to the variable 'u' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'u', tuple_var_assignment_35928_38007)
        
        # Assigning a Name to a Name (line 702):
        
        # Assigning a Name to a Name (line 702):
        # Getting the type of 'tuple_var_assignment_35929' (line 702)
        tuple_var_assignment_35929_38008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'tuple_var_assignment_35929')
        # Assigning a type to the variable '_' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 11), '_', tuple_var_assignment_35929_38008)

        if more_types_in_union_37948:
            # SSA join for if statement (line 701)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Subscript to a Name (line 703):
    
    # Assigning a Subscript to a Name (line 703):
    
    # Assigning a Subscript to a Name (line 703):
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 703)
    m_38009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 13), 'm')
    slice_38010 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 703, 10), None, m_38009, None)
    # Getting the type of 'm' (line 703)
    m_38011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 17), 'm')
    slice_38012 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 703, 10), None, m_38011, None)
    # Getting the type of 'u' (line 703)
    u_38013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 10), 'u')
    # Obtaining the member '__getitem__' of a type (line 703)
    getitem___38014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 10), u_38013, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 703)
    subscript_call_result_38015 = invoke(stypy.reporting.localization.Localization(__file__, 703, 10), getitem___38014, (slice_38010, slice_38012))
    
    # Assigning a type to the variable 'u00' (line 703)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 4), 'u00', subscript_call_result_38015)
    
    # Assigning a Subscript to a Name (line 704):
    
    # Assigning a Subscript to a Name (line 704):
    
    # Assigning a Subscript to a Name (line 704):
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 704)
    m_38016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 12), 'm')
    slice_38017 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 704, 10), m_38016, None, None)
    # Getting the type of 'm' (line 704)
    m_38018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 17), 'm')
    slice_38019 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 704, 10), None, m_38018, None)
    # Getting the type of 'u' (line 704)
    u_38020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 10), 'u')
    # Obtaining the member '__getitem__' of a type (line 704)
    getitem___38021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 10), u_38020, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 704)
    subscript_call_result_38022 = invoke(stypy.reporting.localization.Localization(__file__, 704, 10), getitem___38021, (slice_38017, slice_38019))
    
    # Assigning a type to the variable 'u10' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'u10', subscript_call_result_38022)
    
    # Assigning a Call to a Tuple (line 707):
    
    # Assigning a Subscript to a Name (line 707):
    
    # Assigning a Subscript to a Name (line 707):
    
    # Obtaining the type of the subscript
    int_38023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 4), 'int')
    
    # Call to lu(...): (line 707)
    # Processing the call arguments (line 707)
    # Getting the type of 'u00' (line 707)
    u00_38025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 20), 'u00', False)
    # Processing the call keyword arguments (line 707)
    kwargs_38026 = {}
    # Getting the type of 'lu' (line 707)
    lu_38024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 17), 'lu', False)
    # Calling lu(args, kwargs) (line 707)
    lu_call_result_38027 = invoke(stypy.reporting.localization.Localization(__file__, 707, 17), lu_38024, *[u00_38025], **kwargs_38026)
    
    # Obtaining the member '__getitem__' of a type (line 707)
    getitem___38028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 4), lu_call_result_38027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 707)
    subscript_call_result_38029 = invoke(stypy.reporting.localization.Localization(__file__, 707, 4), getitem___38028, int_38023)
    
    # Assigning a type to the variable 'tuple_var_assignment_35930' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 4), 'tuple_var_assignment_35930', subscript_call_result_38029)
    
    # Assigning a Subscript to a Name (line 707):
    
    # Assigning a Subscript to a Name (line 707):
    
    # Obtaining the type of the subscript
    int_38030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 4), 'int')
    
    # Call to lu(...): (line 707)
    # Processing the call arguments (line 707)
    # Getting the type of 'u00' (line 707)
    u00_38032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 20), 'u00', False)
    # Processing the call keyword arguments (line 707)
    kwargs_38033 = {}
    # Getting the type of 'lu' (line 707)
    lu_38031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 17), 'lu', False)
    # Calling lu(args, kwargs) (line 707)
    lu_call_result_38034 = invoke(stypy.reporting.localization.Localization(__file__, 707, 17), lu_38031, *[u00_38032], **kwargs_38033)
    
    # Obtaining the member '__getitem__' of a type (line 707)
    getitem___38035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 4), lu_call_result_38034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 707)
    subscript_call_result_38036 = invoke(stypy.reporting.localization.Localization(__file__, 707, 4), getitem___38035, int_38030)
    
    # Assigning a type to the variable 'tuple_var_assignment_35931' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 4), 'tuple_var_assignment_35931', subscript_call_result_38036)
    
    # Assigning a Subscript to a Name (line 707):
    
    # Assigning a Subscript to a Name (line 707):
    
    # Obtaining the type of the subscript
    int_38037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 4), 'int')
    
    # Call to lu(...): (line 707)
    # Processing the call arguments (line 707)
    # Getting the type of 'u00' (line 707)
    u00_38039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 20), 'u00', False)
    # Processing the call keyword arguments (line 707)
    kwargs_38040 = {}
    # Getting the type of 'lu' (line 707)
    lu_38038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 17), 'lu', False)
    # Calling lu(args, kwargs) (line 707)
    lu_call_result_38041 = invoke(stypy.reporting.localization.Localization(__file__, 707, 17), lu_38038, *[u00_38039], **kwargs_38040)
    
    # Obtaining the member '__getitem__' of a type (line 707)
    getitem___38042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 4), lu_call_result_38041, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 707)
    subscript_call_result_38043 = invoke(stypy.reporting.localization.Localization(__file__, 707, 4), getitem___38042, int_38037)
    
    # Assigning a type to the variable 'tuple_var_assignment_35932' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 4), 'tuple_var_assignment_35932', subscript_call_result_38043)
    
    # Assigning a Name to a Name (line 707):
    
    # Assigning a Name to a Name (line 707):
    # Getting the type of 'tuple_var_assignment_35930' (line 707)
    tuple_var_assignment_35930_38044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 4), 'tuple_var_assignment_35930')
    # Assigning a type to the variable 'up' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 4), 'up', tuple_var_assignment_35930_38044)
    
    # Assigning a Name to a Name (line 707):
    
    # Assigning a Name to a Name (line 707):
    # Getting the type of 'tuple_var_assignment_35931' (line 707)
    tuple_var_assignment_35931_38045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 4), 'tuple_var_assignment_35931')
    # Assigning a type to the variable 'ul' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'ul', tuple_var_assignment_35931_38045)
    
    # Assigning a Name to a Name (line 707):
    
    # Assigning a Name to a Name (line 707):
    # Getting the type of 'tuple_var_assignment_35932' (line 707)
    tuple_var_assignment_35932_38046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 4), 'tuple_var_assignment_35932')
    # Assigning a type to the variable 'uu' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 12), 'uu', tuple_var_assignment_35932_38046)
    
    
    int_38047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 7), 'int')
    
    # Call to cond(...): (line 709)
    # Processing the call arguments (line 709)
    # Getting the type of 'uu' (line 709)
    uu_38049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 14), 'uu', False)
    # Processing the call keyword arguments (line 709)
    kwargs_38050 = {}
    # Getting the type of 'cond' (line 709)
    cond_38048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 9), 'cond', False)
    # Calling cond(args, kwargs) (line 709)
    cond_call_result_38051 = invoke(stypy.reporting.localization.Localization(__file__, 709, 9), cond_38048, *[uu_38049], **kwargs_38050)
    
    # Applying the binary operator 'div' (line 709)
    result_div_38052 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 7), 'div', int_38047, cond_call_result_38051)
    
    
    # Call to spacing(...): (line 709)
    # Processing the call arguments (line 709)
    float_38055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 31), 'float')
    # Processing the call keyword arguments (line 709)
    kwargs_38056 = {}
    # Getting the type of 'np' (line 709)
    np_38053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 20), 'np', False)
    # Obtaining the member 'spacing' of a type (line 709)
    spacing_38054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 20), np_38053, 'spacing')
    # Calling spacing(args, kwargs) (line 709)
    spacing_call_result_38057 = invoke(stypy.reporting.localization.Localization(__file__, 709, 20), spacing_38054, *[float_38055], **kwargs_38056)
    
    # Applying the binary operator '<' (line 709)
    result_lt_38058 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 7), '<', result_div_38052, spacing_call_result_38057)
    
    # Testing the type of an if condition (line 709)
    if_condition_38059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 709, 4), result_lt_38058)
    # Assigning a type to the variable 'if_condition_38059' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'if_condition_38059', if_condition_38059)
    # SSA begins for if statement (line 709)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 710)
    # Processing the call arguments (line 710)
    str_38061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 26), 'str', 'Failed to find a finite solution.')
    # Processing the call keyword arguments (line 710)
    kwargs_38062 = {}
    # Getting the type of 'LinAlgError' (line 710)
    LinAlgError_38060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 710)
    LinAlgError_call_result_38063 = invoke(stypy.reporting.localization.Localization(__file__, 710, 14), LinAlgError_38060, *[str_38061], **kwargs_38062)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 710, 8), LinAlgError_call_result_38063, 'raise parameter', BaseException)
    # SSA join for if statement (line 709)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 713):
    
    # Assigning a Call to a Name (line 713):
    
    # Assigning a Call to a Name (line 713):
    
    # Call to dot(...): (line 713)
    # Processing the call arguments (line 713)
    
    # Call to conj(...): (line 718)
    # Processing the call keyword arguments (line 718)
    kwargs_38096 = {}
    # Getting the type of 'up' (line 718)
    up_38094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 40), 'up', False)
    # Obtaining the member 'conj' of a type (line 718)
    conj_38095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 40), up_38094, 'conj')
    # Calling conj(args, kwargs) (line 718)
    conj_call_result_38097 = invoke(stypy.reporting.localization.Localization(__file__, 718, 40), conj_38095, *[], **kwargs_38096)
    
    # Obtaining the member 'T' of a type (line 718)
    T_38098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 40), conj_call_result_38097, 'T')
    # Processing the call keyword arguments (line 713)
    kwargs_38099 = {}
    
    # Call to conj(...): (line 713)
    # Processing the call keyword arguments (line 713)
    kwargs_38090 = {}
    
    # Call to solve_triangular(...): (line 713)
    # Processing the call arguments (line 713)
    
    # Call to conj(...): (line 713)
    # Processing the call keyword arguments (line 713)
    kwargs_38067 = {}
    # Getting the type of 'ul' (line 713)
    ul_38065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 25), 'ul', False)
    # Obtaining the member 'conj' of a type (line 713)
    conj_38066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 25), ul_38065, 'conj')
    # Calling conj(args, kwargs) (line 713)
    conj_call_result_38068 = invoke(stypy.reporting.localization.Localization(__file__, 713, 25), conj_38066, *[], **kwargs_38067)
    
    # Obtaining the member 'T' of a type (line 713)
    T_38069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 25), conj_call_result_38068, 'T')
    
    # Call to solve_triangular(...): (line 714)
    # Processing the call arguments (line 714)
    
    # Call to conj(...): (line 714)
    # Processing the call keyword arguments (line 714)
    kwargs_38073 = {}
    # Getting the type of 'uu' (line 714)
    uu_38071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 42), 'uu', False)
    # Obtaining the member 'conj' of a type (line 714)
    conj_38072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 42), uu_38071, 'conj')
    # Calling conj(args, kwargs) (line 714)
    conj_call_result_38074 = invoke(stypy.reporting.localization.Localization(__file__, 714, 42), conj_38072, *[], **kwargs_38073)
    
    # Obtaining the member 'T' of a type (line 714)
    T_38075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 42), conj_call_result_38074, 'T')
    
    # Call to conj(...): (line 715)
    # Processing the call keyword arguments (line 715)
    kwargs_38078 = {}
    # Getting the type of 'u10' (line 715)
    u10_38076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 42), 'u10', False)
    # Obtaining the member 'conj' of a type (line 715)
    conj_38077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 42), u10_38076, 'conj')
    # Calling conj(args, kwargs) (line 715)
    conj_call_result_38079 = invoke(stypy.reporting.localization.Localization(__file__, 715, 42), conj_38077, *[], **kwargs_38078)
    
    # Obtaining the member 'T' of a type (line 715)
    T_38080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 42), conj_call_result_38079, 'T')
    # Processing the call keyword arguments (line 714)
    # Getting the type of 'True' (line 716)
    True_38081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 48), 'True', False)
    keyword_38082 = True_38081
    kwargs_38083 = {'lower': keyword_38082}
    # Getting the type of 'solve_triangular' (line 714)
    solve_triangular_38070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 25), 'solve_triangular', False)
    # Calling solve_triangular(args, kwargs) (line 714)
    solve_triangular_call_result_38084 = invoke(stypy.reporting.localization.Localization(__file__, 714, 25), solve_triangular_38070, *[T_38075, T_38080], **kwargs_38083)
    
    # Processing the call keyword arguments (line 713)
    # Getting the type of 'True' (line 717)
    True_38085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 39), 'True', False)
    keyword_38086 = True_38085
    kwargs_38087 = {'unit_diagonal': keyword_38086}
    # Getting the type of 'solve_triangular' (line 713)
    solve_triangular_38064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'solve_triangular', False)
    # Calling solve_triangular(args, kwargs) (line 713)
    solve_triangular_call_result_38088 = invoke(stypy.reporting.localization.Localization(__file__, 713, 8), solve_triangular_38064, *[T_38069, solve_triangular_call_result_38084], **kwargs_38087)
    
    # Obtaining the member 'conj' of a type (line 713)
    conj_38089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 8), solve_triangular_call_result_38088, 'conj')
    # Calling conj(args, kwargs) (line 713)
    conj_call_result_38091 = invoke(stypy.reporting.localization.Localization(__file__, 713, 8), conj_38089, *[], **kwargs_38090)
    
    # Obtaining the member 'T' of a type (line 713)
    T_38092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 8), conj_call_result_38091, 'T')
    # Obtaining the member 'dot' of a type (line 713)
    dot_38093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 8), T_38092, 'dot')
    # Calling dot(args, kwargs) (line 713)
    dot_call_result_38100 = invoke(stypy.reporting.localization.Localization(__file__, 713, 8), dot_38093, *[T_38098], **kwargs_38099)
    
    # Assigning a type to the variable 'x' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 4), 'x', dot_call_result_38100)
    
    # Getting the type of 'balanced' (line 719)
    balanced_38101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 7), 'balanced')
    # Testing the type of an if condition (line 719)
    if_condition_38102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 719, 4), balanced_38101)
    # Assigning a type to the variable 'if_condition_38102' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 4), 'if_condition_38102', if_condition_38102)
    # SSA begins for if statement (line 719)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'x' (line 720)
    x_38103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'x')
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 720)
    m_38104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 18), 'm')
    slice_38105 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 720, 13), None, m_38104, None)
    # Getting the type of 'None' (line 720)
    None_38106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 21), 'None')
    # Getting the type of 'sca' (line 720)
    sca_38107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 13), 'sca')
    # Obtaining the member '__getitem__' of a type (line 720)
    getitem___38108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 13), sca_38107, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 720)
    subscript_call_result_38109 = invoke(stypy.reporting.localization.Localization(__file__, 720, 13), getitem___38108, (slice_38105, None_38106))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 720)
    m_38110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 34), 'm')
    slice_38111 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 720, 29), None, m_38110, None)
    # Getting the type of 'sca' (line 720)
    sca_38112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 29), 'sca')
    # Obtaining the member '__getitem__' of a type (line 720)
    getitem___38113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 29), sca_38112, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 720)
    subscript_call_result_38114 = invoke(stypy.reporting.localization.Localization(__file__, 720, 29), getitem___38113, slice_38111)
    
    # Applying the binary operator '*' (line 720)
    result_mul_38115 = python_operator(stypy.reporting.localization.Localization(__file__, 720, 13), '*', subscript_call_result_38109, subscript_call_result_38114)
    
    # Applying the binary operator '*=' (line 720)
    result_imul_38116 = python_operator(stypy.reporting.localization.Localization(__file__, 720, 8), '*=', x_38103, result_mul_38115)
    # Assigning a type to the variable 'x' (line 720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'x', result_imul_38116)
    
    # SSA join for if statement (line 719)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 723):
    
    # Assigning a Call to a Name (line 723):
    
    # Assigning a Call to a Name (line 723):
    
    # Call to dot(...): (line 723)
    # Processing the call arguments (line 723)
    # Getting the type of 'u10' (line 723)
    u10_38123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 29), 'u10', False)
    # Processing the call keyword arguments (line 723)
    kwargs_38124 = {}
    
    # Call to conj(...): (line 723)
    # Processing the call keyword arguments (line 723)
    kwargs_38119 = {}
    # Getting the type of 'u00' (line 723)
    u00_38117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 12), 'u00', False)
    # Obtaining the member 'conj' of a type (line 723)
    conj_38118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 12), u00_38117, 'conj')
    # Calling conj(args, kwargs) (line 723)
    conj_call_result_38120 = invoke(stypy.reporting.localization.Localization(__file__, 723, 12), conj_38118, *[], **kwargs_38119)
    
    # Obtaining the member 'T' of a type (line 723)
    T_38121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 12), conj_call_result_38120, 'T')
    # Obtaining the member 'dot' of a type (line 723)
    dot_38122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 12), T_38121, 'dot')
    # Calling dot(args, kwargs) (line 723)
    dot_call_result_38125 = invoke(stypy.reporting.localization.Localization(__file__, 723, 12), dot_38122, *[u10_38123], **kwargs_38124)
    
    # Assigning a type to the variable 'u_sym' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 4), 'u_sym', dot_call_result_38125)
    
    # Assigning a Call to a Name (line 724):
    
    # Assigning a Call to a Name (line 724):
    
    # Assigning a Call to a Name (line 724):
    
    # Call to norm(...): (line 724)
    # Processing the call arguments (line 724)
    # Getting the type of 'u_sym' (line 724)
    u_sym_38127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 19), 'u_sym', False)
    int_38128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 26), 'int')
    # Processing the call keyword arguments (line 724)
    kwargs_38129 = {}
    # Getting the type of 'norm' (line 724)
    norm_38126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 14), 'norm', False)
    # Calling norm(args, kwargs) (line 724)
    norm_call_result_38130 = invoke(stypy.reporting.localization.Localization(__file__, 724, 14), norm_38126, *[u_sym_38127, int_38128], **kwargs_38129)
    
    # Assigning a type to the variable 'n_u_sym' (line 724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 4), 'n_u_sym', norm_call_result_38130)
    
    # Assigning a BinOp to a Name (line 725):
    
    # Assigning a BinOp to a Name (line 725):
    
    # Assigning a BinOp to a Name (line 725):
    # Getting the type of 'u_sym' (line 725)
    u_sym_38131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 12), 'u_sym')
    
    # Call to conj(...): (line 725)
    # Processing the call keyword arguments (line 725)
    kwargs_38134 = {}
    # Getting the type of 'u_sym' (line 725)
    u_sym_38132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 20), 'u_sym', False)
    # Obtaining the member 'conj' of a type (line 725)
    conj_38133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 20), u_sym_38132, 'conj')
    # Calling conj(args, kwargs) (line 725)
    conj_call_result_38135 = invoke(stypy.reporting.localization.Localization(__file__, 725, 20), conj_38133, *[], **kwargs_38134)
    
    # Obtaining the member 'T' of a type (line 725)
    T_38136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 20), conj_call_result_38135, 'T')
    # Applying the binary operator '-' (line 725)
    result_sub_38137 = python_operator(stypy.reporting.localization.Localization(__file__, 725, 12), '-', u_sym_38131, T_38136)
    
    # Assigning a type to the variable 'u_sym' (line 725)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 4), 'u_sym', result_sub_38137)
    
    # Assigning a Call to a Name (line 726):
    
    # Assigning a Call to a Name (line 726):
    
    # Assigning a Call to a Name (line 726):
    
    # Call to max(...): (line 726)
    # Processing the call arguments (line 726)
    
    # Obtaining an instance of the builtin type 'list' (line 726)
    list_38140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 726)
    # Adding element type (line 726)
    
    # Call to spacing(...): (line 726)
    # Processing the call arguments (line 726)
    float_38143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 39), 'float')
    # Processing the call keyword arguments (line 726)
    kwargs_38144 = {}
    # Getting the type of 'np' (line 726)
    np_38141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 28), 'np', False)
    # Obtaining the member 'spacing' of a type (line 726)
    spacing_38142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 28), np_38141, 'spacing')
    # Calling spacing(args, kwargs) (line 726)
    spacing_call_result_38145 = invoke(stypy.reporting.localization.Localization(__file__, 726, 28), spacing_38142, *[float_38143], **kwargs_38144)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 726, 27), list_38140, spacing_call_result_38145)
    # Adding element type (line 726)
    # Getting the type of 'n_u_sym' (line 726)
    n_u_sym_38146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 47), 'n_u_sym', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 726, 27), list_38140, n_u_sym_38146)
    
    # Processing the call keyword arguments (line 726)
    kwargs_38147 = {}
    # Getting the type of 'np' (line 726)
    np_38138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 20), 'np', False)
    # Obtaining the member 'max' of a type (line 726)
    max_38139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 20), np_38138, 'max')
    # Calling max(args, kwargs) (line 726)
    max_call_result_38148 = invoke(stypy.reporting.localization.Localization(__file__, 726, 20), max_38139, *[list_38140], **kwargs_38147)
    
    # Assigning a type to the variable 'sym_threshold' (line 726)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'sym_threshold', max_call_result_38148)
    
    
    
    # Call to norm(...): (line 728)
    # Processing the call arguments (line 728)
    # Getting the type of 'u_sym' (line 728)
    u_sym_38150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 12), 'u_sym', False)
    int_38151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 19), 'int')
    # Processing the call keyword arguments (line 728)
    kwargs_38152 = {}
    # Getting the type of 'norm' (line 728)
    norm_38149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 7), 'norm', False)
    # Calling norm(args, kwargs) (line 728)
    norm_call_result_38153 = invoke(stypy.reporting.localization.Localization(__file__, 728, 7), norm_38149, *[u_sym_38150, int_38151], **kwargs_38152)
    
    # Getting the type of 'sym_threshold' (line 728)
    sym_threshold_38154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 24), 'sym_threshold')
    # Applying the binary operator '>' (line 728)
    result_gt_38155 = python_operator(stypy.reporting.localization.Localization(__file__, 728, 7), '>', norm_call_result_38153, sym_threshold_38154)
    
    # Testing the type of an if condition (line 728)
    if_condition_38156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 728, 4), result_gt_38155)
    # Assigning a type to the variable 'if_condition_38156' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'if_condition_38156', if_condition_38156)
    # SSA begins for if statement (line 728)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 729)
    # Processing the call arguments (line 729)
    str_38158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 26), 'str', 'The associated symplectic pencil has eigenvaluestoo close to the unit circle')
    # Processing the call keyword arguments (line 729)
    kwargs_38159 = {}
    # Getting the type of 'LinAlgError' (line 729)
    LinAlgError_38157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 729)
    LinAlgError_call_result_38160 = invoke(stypy.reporting.localization.Localization(__file__, 729, 14), LinAlgError_38157, *[str_38158], **kwargs_38159)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 729, 8), LinAlgError_call_result_38160, 'raise parameter', BaseException)
    # SSA join for if statement (line 728)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 732)
    x_38161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 12), 'x')
    
    # Call to conj(...): (line 732)
    # Processing the call keyword arguments (line 732)
    kwargs_38164 = {}
    # Getting the type of 'x' (line 732)
    x_38162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 16), 'x', False)
    # Obtaining the member 'conj' of a type (line 732)
    conj_38163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 16), x_38162, 'conj')
    # Calling conj(args, kwargs) (line 732)
    conj_call_result_38165 = invoke(stypy.reporting.localization.Localization(__file__, 732, 16), conj_38163, *[], **kwargs_38164)
    
    # Obtaining the member 'T' of a type (line 732)
    T_38166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 16), conj_call_result_38165, 'T')
    # Applying the binary operator '+' (line 732)
    result_add_38167 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 12), '+', x_38161, T_38166)
    
    int_38168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 28), 'int')
    # Applying the binary operator 'div' (line 732)
    result_div_38169 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 11), 'div', result_add_38167, int_38168)
    
    # Assigning a type to the variable 'stypy_return_type' (line 732)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'stypy_return_type', result_div_38169)
    
    # ################# End of 'solve_discrete_are(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_discrete_are' in the type store
    # Getting the type of 'stypy_return_type' (line 528)
    stypy_return_type_38170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38170)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_discrete_are'
    return stypy_return_type_38170

# Assigning a type to the variable 'solve_discrete_are' (line 528)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 0), 'solve_discrete_are', solve_discrete_are)

@norecursion
def _are_validate_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_38171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 49), 'str', 'care')
    defaults = [str_38171]
    # Create a new context for function '_are_validate_args'
    module_type_store = module_type_store.open_function_context('_are_validate_args', 735, 0, False)
    
    # Passed parameters checking function
    _are_validate_args.stypy_localization = localization
    _are_validate_args.stypy_type_of_self = None
    _are_validate_args.stypy_type_store = module_type_store
    _are_validate_args.stypy_function_name = '_are_validate_args'
    _are_validate_args.stypy_param_names_list = ['a', 'b', 'q', 'r', 'e', 's', 'eq_type']
    _are_validate_args.stypy_varargs_param_name = None
    _are_validate_args.stypy_kwargs_param_name = None
    _are_validate_args.stypy_call_defaults = defaults
    _are_validate_args.stypy_call_varargs = varargs
    _are_validate_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_are_validate_args', ['a', 'b', 'q', 'r', 'e', 's', 'eq_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_are_validate_args', localization, ['a', 'b', 'q', 'r', 'e', 's', 'eq_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_are_validate_args(...)' code ##################

    str_38172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, (-1)), 'str', "\n    A helper function to validate the arguments supplied to the\n    Riccati equation solvers. Any discrepancy found in the input\n    matrices leads to a ``ValueError`` exception.\n\n    Essentially, it performs:\n\n        - a check whether the input is free of NaN and Infs.\n        - a pass for the data through ``numpy.atleast_2d()``\n        - squareness check of the relevant arrays,\n        - shape consistency check of the arrays,\n        - singularity check of the relevant arrays,\n        - symmetricity check of the relevant matrices,\n        - a check whether the regular or the generalized version is asked.\n\n    This function is used by ``solve_continuous_are`` and\n    ``solve_discrete_are``.\n\n    Parameters\n    ----------\n    a, b, q, r, e, s : array_like\n        Input data\n    eq_type : str\n        Accepted arguments are 'care' and 'dare'.\n\n    Returns\n    -------\n    a, b, q, r, e, s : ndarray\n        Regularized input data\n    m, n : int\n        shape of the problem\n    r_or_c : type\n        Data type of the problem, returns float or complex\n    gen_or_not : bool\n        Type of the equation, True for generalized and False for regular ARE.\n\n    ")
    
    
    
    
    # Call to lower(...): (line 774)
    # Processing the call keyword arguments (line 774)
    kwargs_38175 = {}
    # Getting the type of 'eq_type' (line 774)
    eq_type_38173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 11), 'eq_type', False)
    # Obtaining the member 'lower' of a type (line 774)
    lower_38174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 11), eq_type_38173, 'lower')
    # Calling lower(args, kwargs) (line 774)
    lower_call_result_38176 = invoke(stypy.reporting.localization.Localization(__file__, 774, 11), lower_38174, *[], **kwargs_38175)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 774)
    tuple_38177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 774)
    # Adding element type (line 774)
    str_38178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 31), 'str', 'dare')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 774, 31), tuple_38177, str_38178)
    # Adding element type (line 774)
    str_38179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 39), 'str', 'care')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 774, 31), tuple_38177, str_38179)
    
    # Applying the binary operator 'in' (line 774)
    result_contains_38180 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 11), 'in', lower_call_result_38176, tuple_38177)
    
    # Applying the 'not' unary operator (line 774)
    result_not__38181 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 7), 'not', result_contains_38180)
    
    # Testing the type of an if condition (line 774)
    if_condition_38182 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 774, 4), result_not__38181)
    # Assigning a type to the variable 'if_condition_38182' (line 774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 4), 'if_condition_38182', if_condition_38182)
    # SSA begins for if statement (line 774)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 775)
    # Processing the call arguments (line 775)
    str_38184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 25), 'str', "Equation type unknown. Only 'care' and 'dare' is understood")
    # Processing the call keyword arguments (line 775)
    kwargs_38185 = {}
    # Getting the type of 'ValueError' (line 775)
    ValueError_38183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 775)
    ValueError_call_result_38186 = invoke(stypy.reporting.localization.Localization(__file__, 775, 14), ValueError_38183, *[str_38184], **kwargs_38185)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 775, 8), ValueError_call_result_38186, 'raise parameter', BaseException)
    # SSA join for if statement (line 774)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 778):
    
    # Assigning a Call to a Name (line 778):
    
    # Assigning a Call to a Name (line 778):
    
    # Call to atleast_2d(...): (line 778)
    # Processing the call arguments (line 778)
    
    # Call to _asarray_validated(...): (line 778)
    # Processing the call arguments (line 778)
    # Getting the type of 'a' (line 778)
    a_38190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 41), 'a', False)
    # Processing the call keyword arguments (line 778)
    # Getting the type of 'True' (line 778)
    True_38191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 57), 'True', False)
    keyword_38192 = True_38191
    kwargs_38193 = {'check_finite': keyword_38192}
    # Getting the type of '_asarray_validated' (line 778)
    _asarray_validated_38189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 22), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 778)
    _asarray_validated_call_result_38194 = invoke(stypy.reporting.localization.Localization(__file__, 778, 22), _asarray_validated_38189, *[a_38190], **kwargs_38193)
    
    # Processing the call keyword arguments (line 778)
    kwargs_38195 = {}
    # Getting the type of 'np' (line 778)
    np_38187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 8), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 778)
    atleast_2d_38188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 8), np_38187, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 778)
    atleast_2d_call_result_38196 = invoke(stypy.reporting.localization.Localization(__file__, 778, 8), atleast_2d_38188, *[_asarray_validated_call_result_38194], **kwargs_38195)
    
    # Assigning a type to the variable 'a' (line 778)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 4), 'a', atleast_2d_call_result_38196)
    
    # Assigning a Call to a Name (line 779):
    
    # Assigning a Call to a Name (line 779):
    
    # Assigning a Call to a Name (line 779):
    
    # Call to atleast_2d(...): (line 779)
    # Processing the call arguments (line 779)
    
    # Call to _asarray_validated(...): (line 779)
    # Processing the call arguments (line 779)
    # Getting the type of 'b' (line 779)
    b_38200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 41), 'b', False)
    # Processing the call keyword arguments (line 779)
    # Getting the type of 'True' (line 779)
    True_38201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 57), 'True', False)
    keyword_38202 = True_38201
    kwargs_38203 = {'check_finite': keyword_38202}
    # Getting the type of '_asarray_validated' (line 779)
    _asarray_validated_38199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 22), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 779)
    _asarray_validated_call_result_38204 = invoke(stypy.reporting.localization.Localization(__file__, 779, 22), _asarray_validated_38199, *[b_38200], **kwargs_38203)
    
    # Processing the call keyword arguments (line 779)
    kwargs_38205 = {}
    # Getting the type of 'np' (line 779)
    np_38197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 779)
    atleast_2d_38198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 8), np_38197, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 779)
    atleast_2d_call_result_38206 = invoke(stypy.reporting.localization.Localization(__file__, 779, 8), atleast_2d_38198, *[_asarray_validated_call_result_38204], **kwargs_38205)
    
    # Assigning a type to the variable 'b' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'b', atleast_2d_call_result_38206)
    
    # Assigning a Call to a Name (line 780):
    
    # Assigning a Call to a Name (line 780):
    
    # Assigning a Call to a Name (line 780):
    
    # Call to atleast_2d(...): (line 780)
    # Processing the call arguments (line 780)
    
    # Call to _asarray_validated(...): (line 780)
    # Processing the call arguments (line 780)
    # Getting the type of 'q' (line 780)
    q_38210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 41), 'q', False)
    # Processing the call keyword arguments (line 780)
    # Getting the type of 'True' (line 780)
    True_38211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 57), 'True', False)
    keyword_38212 = True_38211
    kwargs_38213 = {'check_finite': keyword_38212}
    # Getting the type of '_asarray_validated' (line 780)
    _asarray_validated_38209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 22), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 780)
    _asarray_validated_call_result_38214 = invoke(stypy.reporting.localization.Localization(__file__, 780, 22), _asarray_validated_38209, *[q_38210], **kwargs_38213)
    
    # Processing the call keyword arguments (line 780)
    kwargs_38215 = {}
    # Getting the type of 'np' (line 780)
    np_38207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 780)
    atleast_2d_38208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 8), np_38207, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 780)
    atleast_2d_call_result_38216 = invoke(stypy.reporting.localization.Localization(__file__, 780, 8), atleast_2d_38208, *[_asarray_validated_call_result_38214], **kwargs_38215)
    
    # Assigning a type to the variable 'q' (line 780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 4), 'q', atleast_2d_call_result_38216)
    
    # Assigning a Call to a Name (line 781):
    
    # Assigning a Call to a Name (line 781):
    
    # Assigning a Call to a Name (line 781):
    
    # Call to atleast_2d(...): (line 781)
    # Processing the call arguments (line 781)
    
    # Call to _asarray_validated(...): (line 781)
    # Processing the call arguments (line 781)
    # Getting the type of 'r' (line 781)
    r_38220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 41), 'r', False)
    # Processing the call keyword arguments (line 781)
    # Getting the type of 'True' (line 781)
    True_38221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 57), 'True', False)
    keyword_38222 = True_38221
    kwargs_38223 = {'check_finite': keyword_38222}
    # Getting the type of '_asarray_validated' (line 781)
    _asarray_validated_38219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 22), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 781)
    _asarray_validated_call_result_38224 = invoke(stypy.reporting.localization.Localization(__file__, 781, 22), _asarray_validated_38219, *[r_38220], **kwargs_38223)
    
    # Processing the call keyword arguments (line 781)
    kwargs_38225 = {}
    # Getting the type of 'np' (line 781)
    np_38217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 8), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 781)
    atleast_2d_38218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 8), np_38217, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 781)
    atleast_2d_call_result_38226 = invoke(stypy.reporting.localization.Localization(__file__, 781, 8), atleast_2d_38218, *[_asarray_validated_call_result_38224], **kwargs_38225)
    
    # Assigning a type to the variable 'r' (line 781)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 4), 'r', atleast_2d_call_result_38226)
    
    # Assigning a IfExp to a Name (line 785):
    
    # Assigning a IfExp to a Name (line 785):
    
    # Assigning a IfExp to a Name (line 785):
    
    
    # Call to iscomplexobj(...): (line 785)
    # Processing the call arguments (line 785)
    # Getting the type of 'b' (line 785)
    b_38229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 40), 'b', False)
    # Processing the call keyword arguments (line 785)
    kwargs_38230 = {}
    # Getting the type of 'np' (line 785)
    np_38227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 24), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 785)
    iscomplexobj_38228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 24), np_38227, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 785)
    iscomplexobj_call_result_38231 = invoke(stypy.reporting.localization.Localization(__file__, 785, 24), iscomplexobj_38228, *[b_38229], **kwargs_38230)
    
    # Testing the type of an if expression (line 785)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 785, 13), iscomplexobj_call_result_38231)
    # SSA begins for if expression (line 785)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'complex' (line 785)
    complex_38232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 13), 'complex')
    # SSA branch for the else part of an if expression (line 785)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'float' (line 785)
    float_38233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 48), 'float')
    # SSA join for if expression (line 785)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_38234 = union_type.UnionType.add(complex_38232, float_38233)
    
    # Assigning a type to the variable 'r_or_c' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 4), 'r_or_c', if_exp_38234)
    
    
    # Call to enumerate(...): (line 787)
    # Processing the call arguments (line 787)
    
    # Obtaining an instance of the builtin type 'tuple' (line 787)
    tuple_38236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 787)
    # Adding element type (line 787)
    # Getting the type of 'a' (line 787)
    a_38237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 31), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 31), tuple_38236, a_38237)
    # Adding element type (line 787)
    # Getting the type of 'q' (line 787)
    q_38238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 34), 'q', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 31), tuple_38236, q_38238)
    # Adding element type (line 787)
    # Getting the type of 'r' (line 787)
    r_38239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 37), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 31), tuple_38236, r_38239)
    
    # Processing the call keyword arguments (line 787)
    kwargs_38240 = {}
    # Getting the type of 'enumerate' (line 787)
    enumerate_38235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 20), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 787)
    enumerate_call_result_38241 = invoke(stypy.reporting.localization.Localization(__file__, 787, 20), enumerate_38235, *[tuple_38236], **kwargs_38240)
    
    # Testing the type of a for loop iterable (line 787)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 787, 4), enumerate_call_result_38241)
    # Getting the type of the for loop variable (line 787)
    for_loop_var_38242 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 787, 4), enumerate_call_result_38241)
    # Assigning a type to the variable 'ind' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'ind', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 4), for_loop_var_38242))
    # Assigning a type to the variable 'mat' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'mat', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 4), for_loop_var_38242))
    # SSA begins for a for statement (line 787)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to iscomplexobj(...): (line 788)
    # Processing the call arguments (line 788)
    # Getting the type of 'mat' (line 788)
    mat_38245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 27), 'mat', False)
    # Processing the call keyword arguments (line 788)
    kwargs_38246 = {}
    # Getting the type of 'np' (line 788)
    np_38243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 11), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 788)
    iscomplexobj_38244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 11), np_38243, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 788)
    iscomplexobj_call_result_38247 = invoke(stypy.reporting.localization.Localization(__file__, 788, 11), iscomplexobj_38244, *[mat_38245], **kwargs_38246)
    
    # Testing the type of an if condition (line 788)
    if_condition_38248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 788, 8), iscomplexobj_call_result_38247)
    # Assigning a type to the variable 'if_condition_38248' (line 788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 8), 'if_condition_38248', if_condition_38248)
    # SSA begins for if statement (line 788)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 789):
    
    # Assigning a Name to a Name (line 789):
    
    # Assigning a Name to a Name (line 789):
    # Getting the type of 'complex' (line 789)
    complex_38249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 21), 'complex')
    # Assigning a type to the variable 'r_or_c' (line 789)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 12), 'r_or_c', complex_38249)
    # SSA join for if statement (line 788)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to equal(...): (line 791)
    # Getting the type of 'mat' (line 791)
    mat_38252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 25), 'mat', False)
    # Obtaining the member 'shape' of a type (line 791)
    shape_38253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 25), mat_38252, 'shape')
    # Processing the call keyword arguments (line 791)
    kwargs_38254 = {}
    # Getting the type of 'np' (line 791)
    np_38250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 15), 'np', False)
    # Obtaining the member 'equal' of a type (line 791)
    equal_38251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 15), np_38250, 'equal')
    # Calling equal(args, kwargs) (line 791)
    equal_call_result_38255 = invoke(stypy.reporting.localization.Localization(__file__, 791, 15), equal_38251, *[shape_38253], **kwargs_38254)
    
    # Applying the 'not' unary operator (line 791)
    result_not__38256 = python_operator(stypy.reporting.localization.Localization(__file__, 791, 11), 'not', equal_call_result_38255)
    
    # Testing the type of an if condition (line 791)
    if_condition_38257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 791, 8), result_not__38256)
    # Assigning a type to the variable 'if_condition_38257' (line 791)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 8), 'if_condition_38257', if_condition_38257)
    # SSA begins for if statement (line 791)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 792)
    # Processing the call arguments (line 792)
    
    # Call to format(...): (line 792)
    # Processing the call arguments (line 792)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 792)
    ind_38261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 72), 'ind', False)
    str_38262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 66), 'str', 'aqr')
    # Obtaining the member '__getitem__' of a type (line 792)
    getitem___38263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 66), str_38262, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 792)
    subscript_call_result_38264 = invoke(stypy.reporting.localization.Localization(__file__, 792, 66), getitem___38263, ind_38261)
    
    # Processing the call keyword arguments (line 792)
    kwargs_38265 = {}
    str_38259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 29), 'str', 'Matrix {} should be square.')
    # Obtaining the member 'format' of a type (line 792)
    format_38260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 29), str_38259, 'format')
    # Calling format(args, kwargs) (line 792)
    format_call_result_38266 = invoke(stypy.reporting.localization.Localization(__file__, 792, 29), format_38260, *[subscript_call_result_38264], **kwargs_38265)
    
    # Processing the call keyword arguments (line 792)
    kwargs_38267 = {}
    # Getting the type of 'ValueError' (line 792)
    ValueError_38258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 792)
    ValueError_call_result_38268 = invoke(stypy.reporting.localization.Localization(__file__, 792, 18), ValueError_38258, *[format_call_result_38266], **kwargs_38267)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 792, 12), ValueError_call_result_38268, 'raise parameter', BaseException)
    # SSA join for if statement (line 791)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 795):
    
    # Assigning a Subscript to a Name (line 795):
    
    # Assigning a Subscript to a Name (line 795):
    
    # Obtaining the type of the subscript
    int_38269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 4), 'int')
    # Getting the type of 'b' (line 795)
    b_38270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 11), 'b')
    # Obtaining the member 'shape' of a type (line 795)
    shape_38271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 11), b_38270, 'shape')
    # Obtaining the member '__getitem__' of a type (line 795)
    getitem___38272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 4), shape_38271, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 795)
    subscript_call_result_38273 = invoke(stypy.reporting.localization.Localization(__file__, 795, 4), getitem___38272, int_38269)
    
    # Assigning a type to the variable 'tuple_var_assignment_35933' (line 795)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 4), 'tuple_var_assignment_35933', subscript_call_result_38273)
    
    # Assigning a Subscript to a Name (line 795):
    
    # Assigning a Subscript to a Name (line 795):
    
    # Obtaining the type of the subscript
    int_38274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 4), 'int')
    # Getting the type of 'b' (line 795)
    b_38275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 11), 'b')
    # Obtaining the member 'shape' of a type (line 795)
    shape_38276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 11), b_38275, 'shape')
    # Obtaining the member '__getitem__' of a type (line 795)
    getitem___38277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 4), shape_38276, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 795)
    subscript_call_result_38278 = invoke(stypy.reporting.localization.Localization(__file__, 795, 4), getitem___38277, int_38274)
    
    # Assigning a type to the variable 'tuple_var_assignment_35934' (line 795)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 4), 'tuple_var_assignment_35934', subscript_call_result_38278)
    
    # Assigning a Name to a Name (line 795):
    
    # Assigning a Name to a Name (line 795):
    # Getting the type of 'tuple_var_assignment_35933' (line 795)
    tuple_var_assignment_35933_38279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 4), 'tuple_var_assignment_35933')
    # Assigning a type to the variable 'm' (line 795)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 4), 'm', tuple_var_assignment_35933_38279)
    
    # Assigning a Name to a Name (line 795):
    
    # Assigning a Name to a Name (line 795):
    # Getting the type of 'tuple_var_assignment_35934' (line 795)
    tuple_var_assignment_35934_38280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 4), 'tuple_var_assignment_35934')
    # Assigning a type to the variable 'n' (line 795)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 7), 'n', tuple_var_assignment_35934_38280)
    
    
    # Getting the type of 'm' (line 796)
    m_38281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 7), 'm')
    
    # Obtaining the type of the subscript
    int_38282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 20), 'int')
    # Getting the type of 'a' (line 796)
    a_38283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 12), 'a')
    # Obtaining the member 'shape' of a type (line 796)
    shape_38284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 12), a_38283, 'shape')
    # Obtaining the member '__getitem__' of a type (line 796)
    getitem___38285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 12), shape_38284, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 796)
    subscript_call_result_38286 = invoke(stypy.reporting.localization.Localization(__file__, 796, 12), getitem___38285, int_38282)
    
    # Applying the binary operator '!=' (line 796)
    result_ne_38287 = python_operator(stypy.reporting.localization.Localization(__file__, 796, 7), '!=', m_38281, subscript_call_result_38286)
    
    # Testing the type of an if condition (line 796)
    if_condition_38288 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 796, 4), result_ne_38287)
    # Assigning a type to the variable 'if_condition_38288' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'if_condition_38288', if_condition_38288)
    # SSA begins for if statement (line 796)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 797)
    # Processing the call arguments (line 797)
    str_38290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 25), 'str', 'Matrix a and b should have the same number of rows.')
    # Processing the call keyword arguments (line 797)
    kwargs_38291 = {}
    # Getting the type of 'ValueError' (line 797)
    ValueError_38289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 797)
    ValueError_call_result_38292 = invoke(stypy.reporting.localization.Localization(__file__, 797, 14), ValueError_38289, *[str_38290], **kwargs_38291)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 797, 8), ValueError_call_result_38292, 'raise parameter', BaseException)
    # SSA join for if statement (line 796)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'm' (line 798)
    m_38293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 7), 'm')
    
    # Obtaining the type of the subscript
    int_38294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, 20), 'int')
    # Getting the type of 'q' (line 798)
    q_38295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 12), 'q')
    # Obtaining the member 'shape' of a type (line 798)
    shape_38296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 12), q_38295, 'shape')
    # Obtaining the member '__getitem__' of a type (line 798)
    getitem___38297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 12), shape_38296, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 798)
    subscript_call_result_38298 = invoke(stypy.reporting.localization.Localization(__file__, 798, 12), getitem___38297, int_38294)
    
    # Applying the binary operator '!=' (line 798)
    result_ne_38299 = python_operator(stypy.reporting.localization.Localization(__file__, 798, 7), '!=', m_38293, subscript_call_result_38298)
    
    # Testing the type of an if condition (line 798)
    if_condition_38300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 798, 4), result_ne_38299)
    # Assigning a type to the variable 'if_condition_38300' (line 798)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'if_condition_38300', if_condition_38300)
    # SSA begins for if statement (line 798)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 799)
    # Processing the call arguments (line 799)
    str_38302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 25), 'str', 'Matrix a and q should have the same shape.')
    # Processing the call keyword arguments (line 799)
    kwargs_38303 = {}
    # Getting the type of 'ValueError' (line 799)
    ValueError_38301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 799)
    ValueError_call_result_38304 = invoke(stypy.reporting.localization.Localization(__file__, 799, 14), ValueError_38301, *[str_38302], **kwargs_38303)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 799, 8), ValueError_call_result_38304, 'raise parameter', BaseException)
    # SSA join for if statement (line 798)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'n' (line 800)
    n_38305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 7), 'n')
    
    # Obtaining the type of the subscript
    int_38306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 20), 'int')
    # Getting the type of 'r' (line 800)
    r_38307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 12), 'r')
    # Obtaining the member 'shape' of a type (line 800)
    shape_38308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 12), r_38307, 'shape')
    # Obtaining the member '__getitem__' of a type (line 800)
    getitem___38309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 12), shape_38308, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 800)
    subscript_call_result_38310 = invoke(stypy.reporting.localization.Localization(__file__, 800, 12), getitem___38309, int_38306)
    
    # Applying the binary operator '!=' (line 800)
    result_ne_38311 = python_operator(stypy.reporting.localization.Localization(__file__, 800, 7), '!=', n_38305, subscript_call_result_38310)
    
    # Testing the type of an if condition (line 800)
    if_condition_38312 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 800, 4), result_ne_38311)
    # Assigning a type to the variable 'if_condition_38312' (line 800)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 800, 4), 'if_condition_38312', if_condition_38312)
    # SSA begins for if statement (line 800)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 801)
    # Processing the call arguments (line 801)
    str_38314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 25), 'str', 'Matrix b and r should have the same number of cols.')
    # Processing the call keyword arguments (line 801)
    kwargs_38315 = {}
    # Getting the type of 'ValueError' (line 801)
    ValueError_38313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 801)
    ValueError_call_result_38316 = invoke(stypy.reporting.localization.Localization(__file__, 801, 14), ValueError_38313, *[str_38314], **kwargs_38315)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 801, 8), ValueError_call_result_38316, 'raise parameter', BaseException)
    # SSA join for if statement (line 800)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to enumerate(...): (line 804)
    # Processing the call arguments (line 804)
    
    # Obtaining an instance of the builtin type 'tuple' (line 804)
    tuple_38318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 804)
    # Adding element type (line 804)
    # Getting the type of 'q' (line 804)
    q_38319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 31), 'q', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 804, 31), tuple_38318, q_38319)
    # Adding element type (line 804)
    # Getting the type of 'r' (line 804)
    r_38320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 34), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 804, 31), tuple_38318, r_38320)
    
    # Processing the call keyword arguments (line 804)
    kwargs_38321 = {}
    # Getting the type of 'enumerate' (line 804)
    enumerate_38317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 20), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 804)
    enumerate_call_result_38322 = invoke(stypy.reporting.localization.Localization(__file__, 804, 20), enumerate_38317, *[tuple_38318], **kwargs_38321)
    
    # Testing the type of a for loop iterable (line 804)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 804, 4), enumerate_call_result_38322)
    # Getting the type of the for loop variable (line 804)
    for_loop_var_38323 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 804, 4), enumerate_call_result_38322)
    # Assigning a type to the variable 'ind' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'ind', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 804, 4), for_loop_var_38323))
    # Assigning a type to the variable 'mat' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'mat', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 804, 4), for_loop_var_38323))
    # SSA begins for a for statement (line 804)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to norm(...): (line 805)
    # Processing the call arguments (line 805)
    # Getting the type of 'mat' (line 805)
    mat_38325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 16), 'mat', False)
    
    # Call to conj(...): (line 805)
    # Processing the call keyword arguments (line 805)
    kwargs_38328 = {}
    # Getting the type of 'mat' (line 805)
    mat_38326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 22), 'mat', False)
    # Obtaining the member 'conj' of a type (line 805)
    conj_38327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 22), mat_38326, 'conj')
    # Calling conj(args, kwargs) (line 805)
    conj_call_result_38329 = invoke(stypy.reporting.localization.Localization(__file__, 805, 22), conj_38327, *[], **kwargs_38328)
    
    # Obtaining the member 'T' of a type (line 805)
    T_38330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 22), conj_call_result_38329, 'T')
    # Applying the binary operator '-' (line 805)
    result_sub_38331 = python_operator(stypy.reporting.localization.Localization(__file__, 805, 16), '-', mat_38325, T_38330)
    
    int_38332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 36), 'int')
    # Processing the call keyword arguments (line 805)
    kwargs_38333 = {}
    # Getting the type of 'norm' (line 805)
    norm_38324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 11), 'norm', False)
    # Calling norm(args, kwargs) (line 805)
    norm_call_result_38334 = invoke(stypy.reporting.localization.Localization(__file__, 805, 11), norm_38324, *[result_sub_38331, int_38332], **kwargs_38333)
    
    
    # Call to spacing(...): (line 805)
    # Processing the call arguments (line 805)
    
    # Call to norm(...): (line 805)
    # Processing the call arguments (line 805)
    # Getting the type of 'mat' (line 805)
    mat_38338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 57), 'mat', False)
    int_38339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 62), 'int')
    # Processing the call keyword arguments (line 805)
    kwargs_38340 = {}
    # Getting the type of 'norm' (line 805)
    norm_38337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 52), 'norm', False)
    # Calling norm(args, kwargs) (line 805)
    norm_call_result_38341 = invoke(stypy.reporting.localization.Localization(__file__, 805, 52), norm_38337, *[mat_38338, int_38339], **kwargs_38340)
    
    # Processing the call keyword arguments (line 805)
    kwargs_38342 = {}
    # Getting the type of 'np' (line 805)
    np_38335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 41), 'np', False)
    # Obtaining the member 'spacing' of a type (line 805)
    spacing_38336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 41), np_38335, 'spacing')
    # Calling spacing(args, kwargs) (line 805)
    spacing_call_result_38343 = invoke(stypy.reporting.localization.Localization(__file__, 805, 41), spacing_38336, *[norm_call_result_38341], **kwargs_38342)
    
    int_38344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 66), 'int')
    # Applying the binary operator '*' (line 805)
    result_mul_38345 = python_operator(stypy.reporting.localization.Localization(__file__, 805, 41), '*', spacing_call_result_38343, int_38344)
    
    # Applying the binary operator '>' (line 805)
    result_gt_38346 = python_operator(stypy.reporting.localization.Localization(__file__, 805, 11), '>', norm_call_result_38334, result_mul_38345)
    
    # Testing the type of an if condition (line 805)
    if_condition_38347 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 805, 8), result_gt_38346)
    # Assigning a type to the variable 'if_condition_38347' (line 805)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 8), 'if_condition_38347', if_condition_38347)
    # SSA begins for if statement (line 805)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 806)
    # Processing the call arguments (line 806)
    
    # Call to format(...): (line 806)
    # Processing the call arguments (line 806)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 807)
    ind_38351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 44), 'ind', False)
    str_38352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 39), 'str', 'qr')
    # Obtaining the member '__getitem__' of a type (line 807)
    getitem___38353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 39), str_38352, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 807)
    subscript_call_result_38354 = invoke(stypy.reporting.localization.Localization(__file__, 807, 39), getitem___38353, ind_38351)
    
    # Processing the call keyword arguments (line 806)
    kwargs_38355 = {}
    str_38349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 29), 'str', 'Matrix {} should be symmetric/hermitian.')
    # Obtaining the member 'format' of a type (line 806)
    format_38350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 29), str_38349, 'format')
    # Calling format(args, kwargs) (line 806)
    format_call_result_38356 = invoke(stypy.reporting.localization.Localization(__file__, 806, 29), format_38350, *[subscript_call_result_38354], **kwargs_38355)
    
    # Processing the call keyword arguments (line 806)
    kwargs_38357 = {}
    # Getting the type of 'ValueError' (line 806)
    ValueError_38348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 806)
    ValueError_call_result_38358 = invoke(stypy.reporting.localization.Localization(__file__, 806, 18), ValueError_38348, *[format_call_result_38356], **kwargs_38357)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 806, 12), ValueError_call_result_38358, 'raise parameter', BaseException)
    # SSA join for if statement (line 805)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'eq_type' (line 810)
    eq_type_38359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 7), 'eq_type')
    str_38360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 18), 'str', 'care')
    # Applying the binary operator '==' (line 810)
    result_eq_38361 = python_operator(stypy.reporting.localization.Localization(__file__, 810, 7), '==', eq_type_38359, str_38360)
    
    # Testing the type of an if condition (line 810)
    if_condition_38362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 810, 4), result_eq_38361)
    # Assigning a type to the variable 'if_condition_38362' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 4), 'if_condition_38362', if_condition_38362)
    # SSA begins for if statement (line 810)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 811):
    
    # Assigning a Subscript to a Name (line 811):
    
    # Assigning a Subscript to a Name (line 811):
    
    # Obtaining the type of the subscript
    int_38363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 42), 'int')
    
    # Call to svd(...): (line 811)
    # Processing the call arguments (line 811)
    # Getting the type of 'r' (line 811)
    r_38365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 21), 'r', False)
    # Processing the call keyword arguments (line 811)
    # Getting the type of 'False' (line 811)
    False_38366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 35), 'False', False)
    keyword_38367 = False_38366
    kwargs_38368 = {'compute_uv': keyword_38367}
    # Getting the type of 'svd' (line 811)
    svd_38364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 17), 'svd', False)
    # Calling svd(args, kwargs) (line 811)
    svd_call_result_38369 = invoke(stypy.reporting.localization.Localization(__file__, 811, 17), svd_38364, *[r_38365], **kwargs_38368)
    
    # Obtaining the member '__getitem__' of a type (line 811)
    getitem___38370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 17), svd_call_result_38369, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 811)
    subscript_call_result_38371 = invoke(stypy.reporting.localization.Localization(__file__, 811, 17), getitem___38370, int_38363)
    
    # Assigning a type to the variable 'min_sv' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 8), 'min_sv', subscript_call_result_38371)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'min_sv' (line 812)
    min_sv_38372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 11), 'min_sv')
    float_38373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 21), 'float')
    # Applying the binary operator '==' (line 812)
    result_eq_38374 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 11), '==', min_sv_38372, float_38373)
    
    
    # Getting the type of 'min_sv' (line 812)
    min_sv_38375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 27), 'min_sv')
    
    # Call to spacing(...): (line 812)
    # Processing the call arguments (line 812)
    float_38378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 47), 'float')
    # Processing the call keyword arguments (line 812)
    kwargs_38379 = {}
    # Getting the type of 'np' (line 812)
    np_38376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 36), 'np', False)
    # Obtaining the member 'spacing' of a type (line 812)
    spacing_38377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 36), np_38376, 'spacing')
    # Calling spacing(args, kwargs) (line 812)
    spacing_call_result_38380 = invoke(stypy.reporting.localization.Localization(__file__, 812, 36), spacing_38377, *[float_38378], **kwargs_38379)
    
    
    # Call to norm(...): (line 812)
    # Processing the call arguments (line 812)
    # Getting the type of 'r' (line 812)
    r_38382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 56), 'r', False)
    int_38383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 59), 'int')
    # Processing the call keyword arguments (line 812)
    kwargs_38384 = {}
    # Getting the type of 'norm' (line 812)
    norm_38381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 51), 'norm', False)
    # Calling norm(args, kwargs) (line 812)
    norm_call_result_38385 = invoke(stypy.reporting.localization.Localization(__file__, 812, 51), norm_38381, *[r_38382, int_38383], **kwargs_38384)
    
    # Applying the binary operator '*' (line 812)
    result_mul_38386 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 36), '*', spacing_call_result_38380, norm_call_result_38385)
    
    # Applying the binary operator '<' (line 812)
    result_lt_38387 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 27), '<', min_sv_38375, result_mul_38386)
    
    # Applying the binary operator 'or' (line 812)
    result_or_keyword_38388 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 11), 'or', result_eq_38374, result_lt_38387)
    
    # Testing the type of an if condition (line 812)
    if_condition_38389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 812, 8), result_or_keyword_38388)
    # Assigning a type to the variable 'if_condition_38389' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 8), 'if_condition_38389', if_condition_38389)
    # SSA begins for if statement (line 812)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 813)
    # Processing the call arguments (line 813)
    str_38391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 29), 'str', 'Matrix r is numerically singular.')
    # Processing the call keyword arguments (line 813)
    kwargs_38392 = {}
    # Getting the type of 'ValueError' (line 813)
    ValueError_38390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 813)
    ValueError_call_result_38393 = invoke(stypy.reporting.localization.Localization(__file__, 813, 18), ValueError_38390, *[str_38391], **kwargs_38392)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 813, 12), ValueError_call_result_38393, 'raise parameter', BaseException)
    # SSA join for if statement (line 812)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 810)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 817):
    
    # Assigning a BoolOp to a Name (line 817):
    
    # Assigning a BoolOp to a Name (line 817):
    
    # Evaluating a boolean operation
    
    # Getting the type of 'e' (line 817)
    e_38394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 23), 'e')
    # Getting the type of 'None' (line 817)
    None_38395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 32), 'None')
    # Applying the binary operator 'isnot' (line 817)
    result_is_not_38396 = python_operator(stypy.reporting.localization.Localization(__file__, 817, 23), 'isnot', e_38394, None_38395)
    
    
    # Getting the type of 's' (line 817)
    s_38397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 40), 's')
    # Getting the type of 'None' (line 817)
    None_38398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 49), 'None')
    # Applying the binary operator 'isnot' (line 817)
    result_is_not_38399 = python_operator(stypy.reporting.localization.Localization(__file__, 817, 40), 'isnot', s_38397, None_38398)
    
    # Applying the binary operator 'or' (line 817)
    result_or_keyword_38400 = python_operator(stypy.reporting.localization.Localization(__file__, 817, 23), 'or', result_is_not_38396, result_is_not_38399)
    
    # Assigning a type to the variable 'generalized_case' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 4), 'generalized_case', result_or_keyword_38400)
    
    # Getting the type of 'generalized_case' (line 819)
    generalized_case_38401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 7), 'generalized_case')
    # Testing the type of an if condition (line 819)
    if_condition_38402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 819, 4), generalized_case_38401)
    # Assigning a type to the variable 'if_condition_38402' (line 819)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 4), 'if_condition_38402', if_condition_38402)
    # SSA begins for if statement (line 819)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 820)
    # Getting the type of 'e' (line 820)
    e_38403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 8), 'e')
    # Getting the type of 'None' (line 820)
    None_38404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 20), 'None')
    
    (may_be_38405, more_types_in_union_38406) = may_not_be_none(e_38403, None_38404)

    if may_be_38405:

        if more_types_in_union_38406:
            # Runtime conditional SSA (line 820)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 821):
        
        # Assigning a Call to a Name (line 821):
        
        # Assigning a Call to a Name (line 821):
        
        # Call to atleast_2d(...): (line 821)
        # Processing the call arguments (line 821)
        
        # Call to _asarray_validated(...): (line 821)
        # Processing the call arguments (line 821)
        # Getting the type of 'e' (line 821)
        e_38410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 49), 'e', False)
        # Processing the call keyword arguments (line 821)
        # Getting the type of 'True' (line 821)
        True_38411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 65), 'True', False)
        keyword_38412 = True_38411
        kwargs_38413 = {'check_finite': keyword_38412}
        # Getting the type of '_asarray_validated' (line 821)
        _asarray_validated_38409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 30), '_asarray_validated', False)
        # Calling _asarray_validated(args, kwargs) (line 821)
        _asarray_validated_call_result_38414 = invoke(stypy.reporting.localization.Localization(__file__, 821, 30), _asarray_validated_38409, *[e_38410], **kwargs_38413)
        
        # Processing the call keyword arguments (line 821)
        kwargs_38415 = {}
        # Getting the type of 'np' (line 821)
        np_38407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 16), 'np', False)
        # Obtaining the member 'atleast_2d' of a type (line 821)
        atleast_2d_38408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 16), np_38407, 'atleast_2d')
        # Calling atleast_2d(args, kwargs) (line 821)
        atleast_2d_call_result_38416 = invoke(stypy.reporting.localization.Localization(__file__, 821, 16), atleast_2d_38408, *[_asarray_validated_call_result_38414], **kwargs_38415)
        
        # Assigning a type to the variable 'e' (line 821)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 12), 'e', atleast_2d_call_result_38416)
        
        
        
        # Call to equal(...): (line 822)
        # Getting the type of 'e' (line 822)
        e_38419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 29), 'e', False)
        # Obtaining the member 'shape' of a type (line 822)
        shape_38420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 29), e_38419, 'shape')
        # Processing the call keyword arguments (line 822)
        kwargs_38421 = {}
        # Getting the type of 'np' (line 822)
        np_38417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 19), 'np', False)
        # Obtaining the member 'equal' of a type (line 822)
        equal_38418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 19), np_38417, 'equal')
        # Calling equal(args, kwargs) (line 822)
        equal_call_result_38422 = invoke(stypy.reporting.localization.Localization(__file__, 822, 19), equal_38418, *[shape_38420], **kwargs_38421)
        
        # Applying the 'not' unary operator (line 822)
        result_not__38423 = python_operator(stypy.reporting.localization.Localization(__file__, 822, 15), 'not', equal_call_result_38422)
        
        # Testing the type of an if condition (line 822)
        if_condition_38424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 822, 12), result_not__38423)
        # Assigning a type to the variable 'if_condition_38424' (line 822)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 12), 'if_condition_38424', if_condition_38424)
        # SSA begins for if statement (line 822)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 823)
        # Processing the call arguments (line 823)
        str_38426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 33), 'str', 'Matrix e should be square.')
        # Processing the call keyword arguments (line 823)
        kwargs_38427 = {}
        # Getting the type of 'ValueError' (line 823)
        ValueError_38425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 823)
        ValueError_call_result_38428 = invoke(stypy.reporting.localization.Localization(__file__, 823, 22), ValueError_38425, *[str_38426], **kwargs_38427)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 823, 16), ValueError_call_result_38428, 'raise parameter', BaseException)
        # SSA join for if statement (line 822)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'm' (line 824)
        m_38429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 15), 'm')
        
        # Obtaining the type of the subscript
        int_38430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, 28), 'int')
        # Getting the type of 'e' (line 824)
        e_38431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 20), 'e')
        # Obtaining the member 'shape' of a type (line 824)
        shape_38432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 20), e_38431, 'shape')
        # Obtaining the member '__getitem__' of a type (line 824)
        getitem___38433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 20), shape_38432, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 824)
        subscript_call_result_38434 = invoke(stypy.reporting.localization.Localization(__file__, 824, 20), getitem___38433, int_38430)
        
        # Applying the binary operator '!=' (line 824)
        result_ne_38435 = python_operator(stypy.reporting.localization.Localization(__file__, 824, 15), '!=', m_38429, subscript_call_result_38434)
        
        # Testing the type of an if condition (line 824)
        if_condition_38436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 824, 12), result_ne_38435)
        # Assigning a type to the variable 'if_condition_38436' (line 824)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 12), 'if_condition_38436', if_condition_38436)
        # SSA begins for if statement (line 824)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 825)
        # Processing the call arguments (line 825)
        str_38438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 33), 'str', 'Matrix a and e should have the same shape.')
        # Processing the call keyword arguments (line 825)
        kwargs_38439 = {}
        # Getting the type of 'ValueError' (line 825)
        ValueError_38437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 825)
        ValueError_call_result_38440 = invoke(stypy.reporting.localization.Localization(__file__, 825, 22), ValueError_38437, *[str_38438], **kwargs_38439)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 825, 16), ValueError_call_result_38440, 'raise parameter', BaseException)
        # SSA join for if statement (line 824)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 828):
        
        # Assigning a Subscript to a Name (line 828):
        
        # Assigning a Subscript to a Name (line 828):
        
        # Obtaining the type of the subscript
        int_38441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 46), 'int')
        
        # Call to svd(...): (line 828)
        # Processing the call arguments (line 828)
        # Getting the type of 'e' (line 828)
        e_38443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 25), 'e', False)
        # Processing the call keyword arguments (line 828)
        # Getting the type of 'False' (line 828)
        False_38444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 39), 'False', False)
        keyword_38445 = False_38444
        kwargs_38446 = {'compute_uv': keyword_38445}
        # Getting the type of 'svd' (line 828)
        svd_38442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 21), 'svd', False)
        # Calling svd(args, kwargs) (line 828)
        svd_call_result_38447 = invoke(stypy.reporting.localization.Localization(__file__, 828, 21), svd_38442, *[e_38443], **kwargs_38446)
        
        # Obtaining the member '__getitem__' of a type (line 828)
        getitem___38448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 21), svd_call_result_38447, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 828)
        subscript_call_result_38449 = invoke(stypy.reporting.localization.Localization(__file__, 828, 21), getitem___38448, int_38441)
        
        # Assigning a type to the variable 'min_sv' (line 828)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 12), 'min_sv', subscript_call_result_38449)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'min_sv' (line 829)
        min_sv_38450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 15), 'min_sv')
        float_38451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 25), 'float')
        # Applying the binary operator '==' (line 829)
        result_eq_38452 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 15), '==', min_sv_38450, float_38451)
        
        
        # Getting the type of 'min_sv' (line 829)
        min_sv_38453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 31), 'min_sv')
        
        # Call to spacing(...): (line 829)
        # Processing the call arguments (line 829)
        float_38456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 51), 'float')
        # Processing the call keyword arguments (line 829)
        kwargs_38457 = {}
        # Getting the type of 'np' (line 829)
        np_38454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 40), 'np', False)
        # Obtaining the member 'spacing' of a type (line 829)
        spacing_38455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 40), np_38454, 'spacing')
        # Calling spacing(args, kwargs) (line 829)
        spacing_call_result_38458 = invoke(stypy.reporting.localization.Localization(__file__, 829, 40), spacing_38455, *[float_38456], **kwargs_38457)
        
        
        # Call to norm(...): (line 829)
        # Processing the call arguments (line 829)
        # Getting the type of 'e' (line 829)
        e_38460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 62), 'e', False)
        int_38461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 65), 'int')
        # Processing the call keyword arguments (line 829)
        kwargs_38462 = {}
        # Getting the type of 'norm' (line 829)
        norm_38459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 57), 'norm', False)
        # Calling norm(args, kwargs) (line 829)
        norm_call_result_38463 = invoke(stypy.reporting.localization.Localization(__file__, 829, 57), norm_38459, *[e_38460, int_38461], **kwargs_38462)
        
        # Applying the binary operator '*' (line 829)
        result_mul_38464 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 40), '*', spacing_call_result_38458, norm_call_result_38463)
        
        # Applying the binary operator '<' (line 829)
        result_lt_38465 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 31), '<', min_sv_38453, result_mul_38464)
        
        # Applying the binary operator 'or' (line 829)
        result_or_keyword_38466 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 15), 'or', result_eq_38452, result_lt_38465)
        
        # Testing the type of an if condition (line 829)
        if_condition_38467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 829, 12), result_or_keyword_38466)
        # Assigning a type to the variable 'if_condition_38467' (line 829)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 12), 'if_condition_38467', if_condition_38467)
        # SSA begins for if statement (line 829)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 830)
        # Processing the call arguments (line 830)
        str_38469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 33), 'str', 'Matrix e is numerically singular.')
        # Processing the call keyword arguments (line 830)
        kwargs_38470 = {}
        # Getting the type of 'ValueError' (line 830)
        ValueError_38468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 830)
        ValueError_call_result_38471 = invoke(stypy.reporting.localization.Localization(__file__, 830, 22), ValueError_38468, *[str_38469], **kwargs_38470)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 830, 16), ValueError_call_result_38471, 'raise parameter', BaseException)
        # SSA join for if statement (line 829)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to iscomplexobj(...): (line 831)
        # Processing the call arguments (line 831)
        # Getting the type of 'e' (line 831)
        e_38474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 31), 'e', False)
        # Processing the call keyword arguments (line 831)
        kwargs_38475 = {}
        # Getting the type of 'np' (line 831)
        np_38472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 15), 'np', False)
        # Obtaining the member 'iscomplexobj' of a type (line 831)
        iscomplexobj_38473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 15), np_38472, 'iscomplexobj')
        # Calling iscomplexobj(args, kwargs) (line 831)
        iscomplexobj_call_result_38476 = invoke(stypy.reporting.localization.Localization(__file__, 831, 15), iscomplexobj_38473, *[e_38474], **kwargs_38475)
        
        # Testing the type of an if condition (line 831)
        if_condition_38477 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 831, 12), iscomplexobj_call_result_38476)
        # Assigning a type to the variable 'if_condition_38477' (line 831)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 12), 'if_condition_38477', if_condition_38477)
        # SSA begins for if statement (line 831)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 832):
        
        # Assigning a Name to a Name (line 832):
        
        # Assigning a Name to a Name (line 832):
        # Getting the type of 'complex' (line 832)
        complex_38478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 25), 'complex')
        # Assigning a type to the variable 'r_or_c' (line 832)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 16), 'r_or_c', complex_38478)
        # SSA join for if statement (line 831)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_38406:
            # SSA join for if statement (line 820)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 833)
    # Getting the type of 's' (line 833)
    s_38479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 's')
    # Getting the type of 'None' (line 833)
    None_38480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 20), 'None')
    
    (may_be_38481, more_types_in_union_38482) = may_not_be_none(s_38479, None_38480)

    if may_be_38481:

        if more_types_in_union_38482:
            # Runtime conditional SSA (line 833)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 834):
        
        # Assigning a Call to a Name (line 834):
        
        # Assigning a Call to a Name (line 834):
        
        # Call to atleast_2d(...): (line 834)
        # Processing the call arguments (line 834)
        
        # Call to _asarray_validated(...): (line 834)
        # Processing the call arguments (line 834)
        # Getting the type of 's' (line 834)
        s_38486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 49), 's', False)
        # Processing the call keyword arguments (line 834)
        # Getting the type of 'True' (line 834)
        True_38487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 65), 'True', False)
        keyword_38488 = True_38487
        kwargs_38489 = {'check_finite': keyword_38488}
        # Getting the type of '_asarray_validated' (line 834)
        _asarray_validated_38485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 30), '_asarray_validated', False)
        # Calling _asarray_validated(args, kwargs) (line 834)
        _asarray_validated_call_result_38490 = invoke(stypy.reporting.localization.Localization(__file__, 834, 30), _asarray_validated_38485, *[s_38486], **kwargs_38489)
        
        # Processing the call keyword arguments (line 834)
        kwargs_38491 = {}
        # Getting the type of 'np' (line 834)
        np_38483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 16), 'np', False)
        # Obtaining the member 'atleast_2d' of a type (line 834)
        atleast_2d_38484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 16), np_38483, 'atleast_2d')
        # Calling atleast_2d(args, kwargs) (line 834)
        atleast_2d_call_result_38492 = invoke(stypy.reporting.localization.Localization(__file__, 834, 16), atleast_2d_38484, *[_asarray_validated_call_result_38490], **kwargs_38491)
        
        # Assigning a type to the variable 's' (line 834)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 12), 's', atleast_2d_call_result_38492)
        
        
        # Getting the type of 's' (line 835)
        s_38493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 15), 's')
        # Obtaining the member 'shape' of a type (line 835)
        shape_38494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 15), s_38493, 'shape')
        # Getting the type of 'b' (line 835)
        b_38495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 26), 'b')
        # Obtaining the member 'shape' of a type (line 835)
        shape_38496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 26), b_38495, 'shape')
        # Applying the binary operator '!=' (line 835)
        result_ne_38497 = python_operator(stypy.reporting.localization.Localization(__file__, 835, 15), '!=', shape_38494, shape_38496)
        
        # Testing the type of an if condition (line 835)
        if_condition_38498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 835, 12), result_ne_38497)
        # Assigning a type to the variable 'if_condition_38498' (line 835)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 12), 'if_condition_38498', if_condition_38498)
        # SSA begins for if statement (line 835)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 836)
        # Processing the call arguments (line 836)
        str_38500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 33), 'str', 'Matrix b and s should have the same shape.')
        # Processing the call keyword arguments (line 836)
        kwargs_38501 = {}
        # Getting the type of 'ValueError' (line 836)
        ValueError_38499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 836)
        ValueError_call_result_38502 = invoke(stypy.reporting.localization.Localization(__file__, 836, 22), ValueError_38499, *[str_38500], **kwargs_38501)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 836, 16), ValueError_call_result_38502, 'raise parameter', BaseException)
        # SSA join for if statement (line 835)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to iscomplexobj(...): (line 837)
        # Processing the call arguments (line 837)
        # Getting the type of 's' (line 837)
        s_38505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 31), 's', False)
        # Processing the call keyword arguments (line 837)
        kwargs_38506 = {}
        # Getting the type of 'np' (line 837)
        np_38503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 15), 'np', False)
        # Obtaining the member 'iscomplexobj' of a type (line 837)
        iscomplexobj_38504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 15), np_38503, 'iscomplexobj')
        # Calling iscomplexobj(args, kwargs) (line 837)
        iscomplexobj_call_result_38507 = invoke(stypy.reporting.localization.Localization(__file__, 837, 15), iscomplexobj_38504, *[s_38505], **kwargs_38506)
        
        # Testing the type of an if condition (line 837)
        if_condition_38508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 837, 12), iscomplexobj_call_result_38507)
        # Assigning a type to the variable 'if_condition_38508' (line 837)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 12), 'if_condition_38508', if_condition_38508)
        # SSA begins for if statement (line 837)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 838):
        
        # Assigning a Name to a Name (line 838):
        
        # Assigning a Name to a Name (line 838):
        # Getting the type of 'complex' (line 838)
        complex_38509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 25), 'complex')
        # Assigning a type to the variable 'r_or_c' (line 838)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 16), 'r_or_c', complex_38509)
        # SSA join for if statement (line 837)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_38482:
            # SSA join for if statement (line 833)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 819)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 840)
    tuple_38510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 840)
    # Adding element type (line 840)
    # Getting the type of 'a' (line 840)
    a_38511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 11), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 11), tuple_38510, a_38511)
    # Adding element type (line 840)
    # Getting the type of 'b' (line 840)
    b_38512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 14), 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 11), tuple_38510, b_38512)
    # Adding element type (line 840)
    # Getting the type of 'q' (line 840)
    q_38513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 17), 'q')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 11), tuple_38510, q_38513)
    # Adding element type (line 840)
    # Getting the type of 'r' (line 840)
    r_38514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 20), 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 11), tuple_38510, r_38514)
    # Adding element type (line 840)
    # Getting the type of 'e' (line 840)
    e_38515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 23), 'e')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 11), tuple_38510, e_38515)
    # Adding element type (line 840)
    # Getting the type of 's' (line 840)
    s_38516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 26), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 11), tuple_38510, s_38516)
    # Adding element type (line 840)
    # Getting the type of 'm' (line 840)
    m_38517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 29), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 11), tuple_38510, m_38517)
    # Adding element type (line 840)
    # Getting the type of 'n' (line 840)
    n_38518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 32), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 11), tuple_38510, n_38518)
    # Adding element type (line 840)
    # Getting the type of 'r_or_c' (line 840)
    r_or_c_38519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 35), 'r_or_c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 11), tuple_38510, r_or_c_38519)
    # Adding element type (line 840)
    # Getting the type of 'generalized_case' (line 840)
    generalized_case_38520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 43), 'generalized_case')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 11), tuple_38510, generalized_case_38520)
    
    # Assigning a type to the variable 'stypy_return_type' (line 840)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 4), 'stypy_return_type', tuple_38510)
    
    # ################# End of '_are_validate_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_are_validate_args' in the type store
    # Getting the type of 'stypy_return_type' (line 735)
    stypy_return_type_38521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38521)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_are_validate_args'
    return stypy_return_type_38521

# Assigning a type to the variable '_are_validate_args' (line 735)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 0), '_are_validate_args', _are_validate_args)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
