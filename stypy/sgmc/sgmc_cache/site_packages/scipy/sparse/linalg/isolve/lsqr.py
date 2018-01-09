
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Sparse Equations and Least Squares.
2: 
3: The original Fortran code was written by C. C. Paige and M. A. Saunders as
4: described in
5: 
6: C. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse linear
7: equations and sparse least squares, TOMS 8(1), 43--71 (1982).
8: 
9: C. C. Paige and M. A. Saunders, Algorithm 583; LSQR: Sparse linear
10: equations and least-squares problems, TOMS 8(2), 195--209 (1982).
11: 
12: It is licensed under the following BSD license:
13: 
14: Copyright (c) 2006, Systems Optimization Laboratory
15: All rights reserved.
16: 
17: Redistribution and use in source and binary forms, with or without
18: modification, are permitted provided that the following conditions are
19: met:
20: 
21:     * Redistributions of source code must retain the above copyright
22:       notice, this list of conditions and the following disclaimer.
23: 
24:     * Redistributions in binary form must reproduce the above
25:       copyright notice, this list of conditions and the following
26:       disclaimer in the documentation and/or other materials provided
27:       with the distribution.
28: 
29:     * Neither the name of Stanford University nor the names of its
30:       contributors may be used to endorse or promote products derived
31:       from this software without specific prior written permission.
32: 
33: THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
34: "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
35: LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
36: A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
37: OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
38: SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
39: LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
40: DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
41: THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
42: (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
43: OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
44: 
45: The Fortran code was translated to Python for use in CVXOPT by Jeffery
46: Kline with contributions by Mridul Aanjaneya and Bob Myhill.
47: 
48: Adapted for SciPy by Stefan van der Walt.
49: 
50: '''
51: 
52: from __future__ import division, print_function, absolute_import
53: 
54: __all__ = ['lsqr']
55: 
56: import numpy as np
57: from math import sqrt
58: from scipy.sparse.linalg.interface import aslinearoperator
59: 
60: eps = np.finfo(np.float64).eps
61: 
62: 
63: def _sym_ortho(a, b):
64:     '''
65:     Stable implementation of Givens rotation.
66: 
67:     Notes
68:     -----
69:     The routine 'SymOrtho' was added for numerical stability. This is
70:     recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
71:     ``1/eps`` in some important places (see, for example text following
72:     "Compute the next plane rotation Qk" in minres.py).
73: 
74:     References
75:     ----------
76:     .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
77:            and Least-Squares Problems", Dissertation,
78:            http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
79: 
80:     '''
81:     if b == 0:
82:         return np.sign(a), 0, abs(a)
83:     elif a == 0:
84:         return 0, np.sign(b), abs(b)
85:     elif abs(b) > abs(a):
86:         tau = a / b
87:         s = np.sign(b) / sqrt(1 + tau * tau)
88:         c = s * tau
89:         r = b / s
90:     else:
91:         tau = b / a
92:         c = np.sign(a) / sqrt(1+tau*tau)
93:         s = c * tau
94:         r = a / c
95:     return c, s, r
96: 
97: 
98: def lsqr(A, b, damp=0.0, atol=1e-8, btol=1e-8, conlim=1e8,
99:          iter_lim=None, show=False, calc_var=False, x0=None):
100:     '''Find the least-squares solution to a large, sparse, linear system
101:     of equations.
102: 
103:     The function solves ``Ax = b``  or  ``min ||b - Ax||^2`` or
104:     ``min ||Ax - b||^2 + d^2 ||x||^2``.
105: 
106:     The matrix A may be square or rectangular (over-determined or
107:     under-determined), and may have any rank.
108: 
109:     ::
110: 
111:       1. Unsymmetric equations --    solve  A*x = b
112: 
113:       2. Linear least squares  --    solve  A*x = b
114:                                      in the least-squares sense
115: 
116:       3. Damped least squares  --    solve  (   A    )*x = ( b )
117:                                             ( damp*I )     ( 0 )
118:                                      in the least-squares sense
119: 
120:     Parameters
121:     ----------
122:     A : {sparse matrix, ndarray, LinearOperator}
123:         Representation of an m-by-n matrix.  It is required that
124:         the linear operator can produce ``Ax`` and ``A^T x``.
125:     b : array_like, shape (m,)
126:         Right-hand side vector ``b``.
127:     damp : float
128:         Damping coefficient.
129:     atol, btol : float, optional
130:         Stopping tolerances. If both are 1.0e-9 (say), the final
131:         residual norm should be accurate to about 9 digits.  (The
132:         final x will usually have fewer correct digits, depending on
133:         cond(A) and the size of damp.)
134:     conlim : float, optional
135:         Another stopping tolerance.  lsqr terminates if an estimate of
136:         ``cond(A)`` exceeds `conlim`.  For compatible systems ``Ax =
137:         b``, `conlim` could be as large as 1.0e+12 (say).  For
138:         least-squares problems, conlim should be less than 1.0e+8.
139:         Maximum precision can be obtained by setting ``atol = btol =
140:         conlim = zero``, but the number of iterations may then be
141:         excessive.
142:     iter_lim : int, optional
143:         Explicit limitation on number of iterations (for safety).
144:     show : bool, optional
145:         Display an iteration log.
146:     calc_var : bool, optional
147:         Whether to estimate diagonals of ``(A'A + damp^2*I)^{-1}``.
148:     x0 : array_like, shape (n,), optional
149:         Initial guess of x, if None zeros are used.
150: 
151:         .. versionadded:: 1.0.0
152: 
153:     Returns
154:     -------
155:     x : ndarray of float
156:         The final solution.
157:     istop : int
158:         Gives the reason for termination.
159:         1 means x is an approximate solution to Ax = b.
160:         2 means x approximately solves the least-squares problem.
161:     itn : int
162:         Iteration number upon termination.
163:     r1norm : float
164:         ``norm(r)``, where ``r = b - Ax``.
165:     r2norm : float
166:         ``sqrt( norm(r)^2  +  damp^2 * norm(x)^2 )``.  Equal to `r1norm` if
167:         ``damp == 0``.
168:     anorm : float
169:         Estimate of Frobenius norm of ``Abar = [[A]; [damp*I]]``.
170:     acond : float
171:         Estimate of ``cond(Abar)``.
172:     arnorm : float
173:         Estimate of ``norm(A'*r - damp^2*x)``.
174:     xnorm : float
175:         ``norm(x)``
176:     var : ndarray of float
177:         If ``calc_var`` is True, estimates all diagonals of
178:         ``(A'A)^{-1}`` (if ``damp == 0``) or more generally ``(A'A +
179:         damp^2*I)^{-1}``.  This is well defined if A has full column
180:         rank or ``damp > 0``.  (Not sure what var means if ``rank(A)
181:         < n`` and ``damp = 0.``)
182: 
183:     Notes
184:     -----
185:     LSQR uses an iterative method to approximate the solution.  The
186:     number of iterations required to reach a certain accuracy depends
187:     strongly on the scaling of the problem.  Poor scaling of the rows
188:     or columns of A should therefore be avoided where possible.
189: 
190:     For example, in problem 1 the solution is unaltered by
191:     row-scaling.  If a row of A is very small or large compared to
192:     the other rows of A, the corresponding row of ( A  b ) should be
193:     scaled up or down.
194: 
195:     In problems 1 and 2, the solution x is easily recovered
196:     following column-scaling.  Unless better information is known,
197:     the nonzero columns of A should be scaled so that they all have
198:     the same Euclidean norm (e.g., 1.0).
199: 
200:     In problem 3, there is no freedom to re-scale if damp is
201:     nonzero.  However, the value of damp should be assigned only
202:     after attention has been paid to the scaling of A.
203: 
204:     The parameter damp is intended to help regularize
205:     ill-conditioned systems, by preventing the true solution from
206:     being very large.  Another aid to regularization is provided by
207:     the parameter acond, which may be used to terminate iterations
208:     before the computed solution becomes very large.
209: 
210:     If some initial estimate ``x0`` is known and if ``damp == 0``,
211:     one could proceed as follows:
212: 
213:       1. Compute a residual vector ``r0 = b - A*x0``.
214:       2. Use LSQR to solve the system  ``A*dx = r0``.
215:       3. Add the correction dx to obtain a final solution ``x = x0 + dx``.
216: 
217:     This requires that ``x0`` be available before and after the call
218:     to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations
219:     to solve A*x = b and k2 iterations to solve A*dx = r0.
220:     If x0 is "good", norm(r0) will be smaller than norm(b).
221:     If the same stopping tolerances atol and btol are used for each
222:     system, k1 and k2 will be similar, but the final solution x0 + dx
223:     should be more accurate.  The only way to reduce the total work
224:     is to use a larger stopping tolerance for the second system.
225:     If some value btol is suitable for A*x = b, the larger value
226:     btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.
227: 
228:     Preconditioning is another way to reduce the number of iterations.
229:     If it is possible to solve a related system ``M*x = b``
230:     efficiently, where M approximates A in some helpful way (e.g. M -
231:     A has low rank or its elements are small relative to those of A),
232:     LSQR may converge more rapidly on the system ``A*M(inverse)*z =
233:     b``, after which x can be recovered by solving M*x = z.
234: 
235:     If A is symmetric, LSQR should not be used!
236: 
237:     Alternatives are the symmetric conjugate-gradient method (cg)
238:     and/or SYMMLQ.  SYMMLQ is an implementation of symmetric cg that
239:     applies to any symmetric A and will converge more rapidly than
240:     LSQR.  If A is positive definite, there are other implementations
241:     of symmetric cg that require slightly less work per iteration than
242:     SYMMLQ (but will take the same number of iterations).
243: 
244:     References
245:     ----------
246:     .. [1] C. C. Paige and M. A. Saunders (1982a).
247:            "LSQR: An algorithm for sparse linear equations and
248:            sparse least squares", ACM TOMS 8(1), 43-71.
249:     .. [2] C. C. Paige and M. A. Saunders (1982b).
250:            "Algorithm 583.  LSQR: Sparse linear equations and least
251:            squares problems", ACM TOMS 8(2), 195-209.
252:     .. [3] M. A. Saunders (1995).  "Solution of sparse rectangular
253:            systems using LSQR and CRAIG", BIT 35, 588-604.
254: 
255:     Examples
256:     --------
257:     >>> from scipy.sparse import csc_matrix
258:     >>> from scipy.sparse.linalg import lsqr
259:     >>> A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)
260: 
261:     The first example has the trivial solution `[0, 0]`
262: 
263:     >>> b = np.array([0., 0., 0.], dtype=float)
264:     >>> x, istop, itn, normr = lsqr(A, b)[:4]
265:     The exact solution is  x = 0
266:     >>> istop
267:     0
268:     >>> x
269:     array([ 0.,  0.])
270: 
271:     The stopping code `istop=0` returned indicates that a vector of zeros was
272:     found as a solution. The returned solution `x` indeed contains `[0., 0.]`.
273:     The next example has a non-trivial solution:
274: 
275:     >>> b = np.array([1., 0., -1.], dtype=float)
276:     >>> x, istop, itn, r1norm = lsqr(A, b)[:4]
277:     >>> istop
278:     1
279:     >>> x
280:     array([ 1., -1.])
281:     >>> itn
282:     1
283:     >>> r1norm
284:     4.440892098500627e-16
285: 
286:     As indicated by `istop=1`, `lsqr` found a solution obeying the tolerance
287:     limits. The given solution `[1., -1.]` obviously solves the equation. The
288:     remaining return values include information about the number of iterations
289:     (`itn=1`) and the remaining difference of left and right side of the solved
290:     equation.
291:     The final example demonstrates the behavior in the case where there is no
292:     solution for the equation:
293: 
294:     >>> b = np.array([1., 0.01, -1.], dtype=float)
295:     >>> x, istop, itn, r1norm = lsqr(A, b)[:4]
296:     >>> istop
297:     2
298:     >>> x
299:     array([ 1.00333333, -0.99666667])
300:     >>> A.dot(x)-b
301:     array([ 0.00333333, -0.00333333,  0.00333333])
302:     >>> r1norm
303:     0.005773502691896255
304: 
305:     `istop` indicates that the system is inconsistent and thus `x` is rather an
306:     approximate solution to the corresponding least-squares problem. `r1norm`
307:     contains the norm of the minimal residual that was found.
308:     '''
309:     A = aslinearoperator(A)
310:     b = np.atleast_1d(b)
311:     if b.ndim > 1:
312:         b = b.squeeze()
313: 
314:     m, n = A.shape
315:     if iter_lim is None:
316:         iter_lim = 2 * n
317:     var = np.zeros(n)
318: 
319:     msg = ('The exact solution is  x = 0                              ',
320:          'Ax - b is small enough, given atol, btol                  ',
321:          'The least-squares solution is good enough, given atol     ',
322:          'The estimate of cond(Abar) has exceeded conlim            ',
323:          'Ax - b is small enough for this machine                   ',
324:          'The least-squares solution is good enough for this machine',
325:          'Cond(Abar) seems to be too large for this machine         ',
326:          'The iteration limit has been reached                      ')
327: 
328:     if show:
329:         print(' ')
330:         print('LSQR            Least-squares solution of  Ax = b')
331:         str1 = 'The matrix A has %8g rows  and %8g cols' % (m, n)
332:         str2 = 'damp = %20.14e   calc_var = %8g' % (damp, calc_var)
333:         str3 = 'atol = %8.2e                 conlim = %8.2e' % (atol, conlim)
334:         str4 = 'btol = %8.2e               iter_lim = %8g' % (btol, iter_lim)
335:         print(str1)
336:         print(str2)
337:         print(str3)
338:         print(str4)
339: 
340:     itn = 0
341:     istop = 0
342:     ctol = 0
343:     if conlim > 0:
344:         ctol = 1/conlim
345:     anorm = 0
346:     acond = 0
347:     dampsq = damp**2
348:     ddnorm = 0
349:     res2 = 0
350:     xnorm = 0
351:     xxnorm = 0
352:     z = 0
353:     cs2 = -1
354:     sn2 = 0
355: 
356:     '''
357:     Set up the first vectors u and v for the bidiagonalization.
358:     These satisfy  beta*u = b - A*x,  alfa*v = A'*u.
359:     '''
360:     u = b
361:     bnorm = np.linalg.norm(b)
362:     if x0 is None:
363:         x = np.zeros(n)
364:         beta = bnorm.copy()
365:     else:
366:         x = np.asarray(x0)
367:         u = u - A.matvec(x)
368:         beta = np.linalg.norm(u)
369: 
370:     if beta > 0:
371:         u = (1/beta) * u
372:         v = A.rmatvec(u)
373:         alfa = np.linalg.norm(v)
374:     else:
375:         v = x.copy()
376:         alfa = 0
377: 
378:     if alfa > 0:
379:         v = (1/alfa) * v
380:     w = v.copy()
381: 
382:     rhobar = alfa
383:     phibar = beta
384:     rnorm = beta
385:     r1norm = rnorm
386:     r2norm = rnorm
387: 
388:     # Reverse the order here from the original matlab code because
389:     # there was an error on return when arnorm==0
390:     arnorm = alfa * beta
391:     if arnorm == 0:
392:         print(msg[0])
393:         return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var
394: 
395:     head1 = '   Itn      x[0]       r1norm     r2norm '
396:     head2 = ' Compatible    LS      Norm A   Cond A'
397: 
398:     if show:
399:         print(' ')
400:         print(head1, head2)
401:         test1 = 1
402:         test2 = alfa / beta
403:         str1 = '%6g %12.5e' % (itn, x[0])
404:         str2 = ' %10.3e %10.3e' % (r1norm, r2norm)
405:         str3 = '  %8.1e %8.1e' % (test1, test2)
406:         print(str1, str2, str3)
407: 
408:     # Main iteration loop.
409:     while itn < iter_lim:
410:         itn = itn + 1
411:         '''
412:         %     Perform the next step of the bidiagonalization to obtain the
413:         %     next  beta, u, alfa, v.  These satisfy the relations
414:         %                beta*u  =  a*v   -  alfa*u,
415:         %                alfa*v  =  A'*u  -  beta*v.
416:         '''
417:         u = A.matvec(v) - alfa * u
418:         beta = np.linalg.norm(u)
419: 
420:         if beta > 0:
421:             u = (1/beta) * u
422:             anorm = sqrt(anorm**2 + alfa**2 + beta**2 + damp**2)
423:             v = A.rmatvec(u) - beta * v
424:             alfa = np.linalg.norm(v)
425:             if alfa > 0:
426:                 v = (1 / alfa) * v
427: 
428:         # Use a plane rotation to eliminate the damping parameter.
429:         # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
430:         rhobar1 = sqrt(rhobar**2 + damp**2)
431:         cs1 = rhobar / rhobar1
432:         sn1 = damp / rhobar1
433:         psi = sn1 * phibar
434:         phibar = cs1 * phibar
435: 
436:         # Use a plane rotation to eliminate the subdiagonal element (beta)
437:         # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
438:         cs, sn, rho = _sym_ortho(rhobar1, beta)
439: 
440:         theta = sn * alfa
441:         rhobar = -cs * alfa
442:         phi = cs * phibar
443:         phibar = sn * phibar
444:         tau = sn * phi
445: 
446:         # Update x and w.
447:         t1 = phi / rho
448:         t2 = -theta / rho
449:         dk = (1 / rho) * w
450: 
451:         x = x + t1 * w
452:         w = v + t2 * w
453:         ddnorm = ddnorm + np.linalg.norm(dk)**2
454: 
455:         if calc_var:
456:             var = var + dk**2
457: 
458:         # Use a plane rotation on the right to eliminate the
459:         # super-diagonal element (theta) of the upper-bidiagonal matrix.
460:         # Then use the result to estimate norm(x).
461:         delta = sn2 * rho
462:         gambar = -cs2 * rho
463:         rhs = phi - delta * z
464:         zbar = rhs / gambar
465:         xnorm = sqrt(xxnorm + zbar**2)
466:         gamma = sqrt(gambar**2 + theta**2)
467:         cs2 = gambar / gamma
468:         sn2 = theta / gamma
469:         z = rhs / gamma
470:         xxnorm = xxnorm + z**2
471: 
472:         # Test for convergence.
473:         # First, estimate the condition of the matrix  Abar,
474:         # and the norms of  rbar  and  Abar'rbar.
475:         acond = anorm * sqrt(ddnorm)
476:         res1 = phibar**2
477:         res2 = res2 + psi**2
478:         rnorm = sqrt(res1 + res2)
479:         arnorm = alfa * abs(tau)
480: 
481:         # Distinguish between
482:         #    r1norm = ||b - Ax|| and
483:         #    r2norm = rnorm in current code
484:         #           = sqrt(r1norm^2 + damp^2*||x||^2).
485:         #    Estimate r1norm from
486:         #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
487:         # Although there is cancellation, it might be accurate enough.
488:         r1sq = rnorm**2 - dampsq * xxnorm
489:         r1norm = sqrt(abs(r1sq))
490:         if r1sq < 0:
491:             r1norm = -r1norm
492:         r2norm = rnorm
493: 
494:         # Now use these norms to estimate certain other quantities,
495:         # some of which will be small near a solution.
496:         test1 = rnorm / bnorm
497:         test2 = arnorm / (anorm * rnorm + eps)
498:         test3 = 1 / (acond + eps)
499:         t1 = test1 / (1 + anorm * xnorm / bnorm)
500:         rtol = btol + atol * anorm * xnorm / bnorm
501: 
502:         # The following tests guard against extremely small values of
503:         # atol, btol  or  ctol.  (The user may have set any or all of
504:         # the parameters  atol, btol, conlim  to 0.)
505:         # The effect is equivalent to the normal tests using
506:         # atol = eps,  btol = eps,  conlim = 1/eps.
507:         if itn >= iter_lim:
508:             istop = 7
509:         if 1 + test3 <= 1:
510:             istop = 6
511:         if 1 + test2 <= 1:
512:             istop = 5
513:         if 1 + t1 <= 1:
514:             istop = 4
515: 
516:         # Allow for tolerances set by the user.
517:         if test3 <= ctol:
518:             istop = 3
519:         if test2 <= atol:
520:             istop = 2
521:         if test1 <= rtol:
522:             istop = 1
523: 
524:         # See if it is time to print something.
525:         prnt = False
526:         if n <= 40:
527:             prnt = True
528:         if itn <= 10:
529:             prnt = True
530:         if itn >= iter_lim-10:
531:             prnt = True
532:         # if itn%10 == 0: prnt = True
533:         if test3 <= 2*ctol:
534:             prnt = True
535:         if test2 <= 10*atol:
536:             prnt = True
537:         if test1 <= 10*rtol:
538:             prnt = True
539:         if istop != 0:
540:             prnt = True
541: 
542:         if prnt:
543:             if show:
544:                 str1 = '%6g %12.5e' % (itn, x[0])
545:                 str2 = ' %10.3e %10.3e' % (r1norm, r2norm)
546:                 str3 = '  %8.1e %8.1e' % (test1, test2)
547:                 str4 = ' %8.1e %8.1e' % (anorm, acond)
548:                 print(str1, str2, str3, str4)
549: 
550:         if istop != 0:
551:             break
552: 
553:     # End of iteration loop.
554:     # Print the stopping condition.
555:     if show:
556:         print(' ')
557:         print('LSQR finished')
558:         print(msg[istop])
559:         print(' ')
560:         str1 = 'istop =%8g   r1norm =%8.1e' % (istop, r1norm)
561:         str2 = 'anorm =%8.1e   arnorm =%8.1e' % (anorm, arnorm)
562:         str3 = 'itn   =%8g   r2norm =%8.1e' % (itn, r2norm)
563:         str4 = 'acond =%8.1e   xnorm  =%8.1e' % (acond, xnorm)
564:         print(str1 + '   ' + str2)
565:         print(str3 + '   ' + str4)
566:         print(' ')
567: 
568:     return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var
569: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_412478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, (-1)), 'str', 'Sparse Equations and Least Squares.\n\nThe original Fortran code was written by C. C. Paige and M. A. Saunders as\ndescribed in\n\nC. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse linear\nequations and sparse least squares, TOMS 8(1), 43--71 (1982).\n\nC. C. Paige and M. A. Saunders, Algorithm 583; LSQR: Sparse linear\nequations and least-squares problems, TOMS 8(2), 195--209 (1982).\n\nIt is licensed under the following BSD license:\n\nCopyright (c) 2006, Systems Optimization Laboratory\nAll rights reserved.\n\nRedistribution and use in source and binary forms, with or without\nmodification, are permitted provided that the following conditions are\nmet:\n\n    * Redistributions of source code must retain the above copyright\n      notice, this list of conditions and the following disclaimer.\n\n    * Redistributions in binary form must reproduce the above\n      copyright notice, this list of conditions and the following\n      disclaimer in the documentation and/or other materials provided\n      with the distribution.\n\n    * Neither the name of Stanford University nor the names of its\n      contributors may be used to endorse or promote products derived\n      from this software without specific prior written permission.\n\nTHIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS\n"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT\nLIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR\nA PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT\nOWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,\nSPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT\nLIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,\nDATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY\nTHEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT\n(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE\nOF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n\nThe Fortran code was translated to Python for use in CVXOPT by Jeffery\nKline with contributions by Mridul Aanjaneya and Bob Myhill.\n\nAdapted for SciPy by Stefan van der Walt.\n\n')

# Assigning a List to a Name (line 54):

# Assigning a List to a Name (line 54):
__all__ = ['lsqr']
module_type_store.set_exportable_members(['lsqr'])

# Obtaining an instance of the builtin type 'list' (line 54)
list_412479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 54)
# Adding element type (line 54)
str_412480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 11), 'str', 'lsqr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 10), list_412479, str_412480)

# Assigning a type to the variable '__all__' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), '__all__', list_412479)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 56, 0))

# 'import numpy' statement (line 56)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_412481 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 56, 0), 'numpy')

if (type(import_412481) is not StypyTypeError):

    if (import_412481 != 'pyd_module'):
        __import__(import_412481)
        sys_modules_412482 = sys.modules[import_412481]
        import_module(stypy.reporting.localization.Localization(__file__, 56, 0), 'np', sys_modules_412482.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 56, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'numpy', import_412481)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 57, 0))

# 'from math import sqrt' statement (line 57)
try:
    from math import sqrt

except:
    sqrt = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'math', None, module_type_store, ['sqrt'], [sqrt])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 58, 0))

# 'from scipy.sparse.linalg.interface import aslinearoperator' statement (line 58)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_412483 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'scipy.sparse.linalg.interface')

if (type(import_412483) is not StypyTypeError):

    if (import_412483 != 'pyd_module'):
        __import__(import_412483)
        sys_modules_412484 = sys.modules[import_412483]
        import_from_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'scipy.sparse.linalg.interface', sys_modules_412484.module_type_store, module_type_store, ['aslinearoperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 58, 0), __file__, sys_modules_412484, sys_modules_412484.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.interface import aslinearoperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'scipy.sparse.linalg.interface', None, module_type_store, ['aslinearoperator'], [aslinearoperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.interface' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'scipy.sparse.linalg.interface', import_412483)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')


# Assigning a Attribute to a Name (line 60):

# Assigning a Attribute to a Name (line 60):

# Call to finfo(...): (line 60)
# Processing the call arguments (line 60)
# Getting the type of 'np' (line 60)
np_412487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'np', False)
# Obtaining the member 'float64' of a type (line 60)
float64_412488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 15), np_412487, 'float64')
# Processing the call keyword arguments (line 60)
kwargs_412489 = {}
# Getting the type of 'np' (line 60)
np_412485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 6), 'np', False)
# Obtaining the member 'finfo' of a type (line 60)
finfo_412486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 6), np_412485, 'finfo')
# Calling finfo(args, kwargs) (line 60)
finfo_call_result_412490 = invoke(stypy.reporting.localization.Localization(__file__, 60, 6), finfo_412486, *[float64_412488], **kwargs_412489)

# Obtaining the member 'eps' of a type (line 60)
eps_412491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 6), finfo_call_result_412490, 'eps')
# Assigning a type to the variable 'eps' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'eps', eps_412491)

@norecursion
def _sym_ortho(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_sym_ortho'
    module_type_store = module_type_store.open_function_context('_sym_ortho', 63, 0, False)
    
    # Passed parameters checking function
    _sym_ortho.stypy_localization = localization
    _sym_ortho.stypy_type_of_self = None
    _sym_ortho.stypy_type_store = module_type_store
    _sym_ortho.stypy_function_name = '_sym_ortho'
    _sym_ortho.stypy_param_names_list = ['a', 'b']
    _sym_ortho.stypy_varargs_param_name = None
    _sym_ortho.stypy_kwargs_param_name = None
    _sym_ortho.stypy_call_defaults = defaults
    _sym_ortho.stypy_call_varargs = varargs
    _sym_ortho.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_sym_ortho', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_sym_ortho', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_sym_ortho(...)' code ##################

    str_412492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, (-1)), 'str', '\n    Stable implementation of Givens rotation.\n\n    Notes\n    -----\n    The routine \'SymOrtho\' was added for numerical stability. This is\n    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of\n    ``1/eps`` in some important places (see, for example text following\n    "Compute the next plane rotation Qk" in minres.py).\n\n    References\n    ----------\n    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations\n           and Least-Squares Problems", Dissertation,\n           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf\n\n    ')
    
    
    # Getting the type of 'b' (line 81)
    b_412493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 7), 'b')
    int_412494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 12), 'int')
    # Applying the binary operator '==' (line 81)
    result_eq_412495 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 7), '==', b_412493, int_412494)
    
    # Testing the type of an if condition (line 81)
    if_condition_412496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 4), result_eq_412495)
    # Assigning a type to the variable 'if_condition_412496' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'if_condition_412496', if_condition_412496)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 82)
    tuple_412497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 82)
    # Adding element type (line 82)
    
    # Call to sign(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'a' (line 82)
    a_412500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'a', False)
    # Processing the call keyword arguments (line 82)
    kwargs_412501 = {}
    # Getting the type of 'np' (line 82)
    np_412498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'np', False)
    # Obtaining the member 'sign' of a type (line 82)
    sign_412499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), np_412498, 'sign')
    # Calling sign(args, kwargs) (line 82)
    sign_call_result_412502 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), sign_412499, *[a_412500], **kwargs_412501)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 15), tuple_412497, sign_call_result_412502)
    # Adding element type (line 82)
    int_412503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 15), tuple_412497, int_412503)
    # Adding element type (line 82)
    
    # Call to abs(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'a' (line 82)
    a_412505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 34), 'a', False)
    # Processing the call keyword arguments (line 82)
    kwargs_412506 = {}
    # Getting the type of 'abs' (line 82)
    abs_412504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 30), 'abs', False)
    # Calling abs(args, kwargs) (line 82)
    abs_call_result_412507 = invoke(stypy.reporting.localization.Localization(__file__, 82, 30), abs_412504, *[a_412505], **kwargs_412506)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 15), tuple_412497, abs_call_result_412507)
    
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', tuple_412497)
    # SSA branch for the else part of an if statement (line 81)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'a' (line 83)
    a_412508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 9), 'a')
    int_412509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 14), 'int')
    # Applying the binary operator '==' (line 83)
    result_eq_412510 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 9), '==', a_412508, int_412509)
    
    # Testing the type of an if condition (line 83)
    if_condition_412511 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 9), result_eq_412510)
    # Assigning a type to the variable 'if_condition_412511' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 9), 'if_condition_412511', if_condition_412511)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 84)
    tuple_412512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 84)
    # Adding element type (line 84)
    int_412513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), tuple_412512, int_412513)
    # Adding element type (line 84)
    
    # Call to sign(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'b' (line 84)
    b_412516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'b', False)
    # Processing the call keyword arguments (line 84)
    kwargs_412517 = {}
    # Getting the type of 'np' (line 84)
    np_412514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'np', False)
    # Obtaining the member 'sign' of a type (line 84)
    sign_412515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 18), np_412514, 'sign')
    # Calling sign(args, kwargs) (line 84)
    sign_call_result_412518 = invoke(stypy.reporting.localization.Localization(__file__, 84, 18), sign_412515, *[b_412516], **kwargs_412517)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), tuple_412512, sign_call_result_412518)
    # Adding element type (line 84)
    
    # Call to abs(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'b' (line 84)
    b_412520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 34), 'b', False)
    # Processing the call keyword arguments (line 84)
    kwargs_412521 = {}
    # Getting the type of 'abs' (line 84)
    abs_412519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'abs', False)
    # Calling abs(args, kwargs) (line 84)
    abs_call_result_412522 = invoke(stypy.reporting.localization.Localization(__file__, 84, 30), abs_412519, *[b_412520], **kwargs_412521)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), tuple_412512, abs_call_result_412522)
    
    # Assigning a type to the variable 'stypy_return_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'stypy_return_type', tuple_412512)
    # SSA branch for the else part of an if statement (line 83)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to abs(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'b' (line 85)
    b_412524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'b', False)
    # Processing the call keyword arguments (line 85)
    kwargs_412525 = {}
    # Getting the type of 'abs' (line 85)
    abs_412523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 9), 'abs', False)
    # Calling abs(args, kwargs) (line 85)
    abs_call_result_412526 = invoke(stypy.reporting.localization.Localization(__file__, 85, 9), abs_412523, *[b_412524], **kwargs_412525)
    
    
    # Call to abs(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'a' (line 85)
    a_412528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 22), 'a', False)
    # Processing the call keyword arguments (line 85)
    kwargs_412529 = {}
    # Getting the type of 'abs' (line 85)
    abs_412527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 18), 'abs', False)
    # Calling abs(args, kwargs) (line 85)
    abs_call_result_412530 = invoke(stypy.reporting.localization.Localization(__file__, 85, 18), abs_412527, *[a_412528], **kwargs_412529)
    
    # Applying the binary operator '>' (line 85)
    result_gt_412531 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 9), '>', abs_call_result_412526, abs_call_result_412530)
    
    # Testing the type of an if condition (line 85)
    if_condition_412532 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 9), result_gt_412531)
    # Assigning a type to the variable 'if_condition_412532' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 9), 'if_condition_412532', if_condition_412532)
    # SSA begins for if statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 86):
    
    # Assigning a BinOp to a Name (line 86):
    # Getting the type of 'a' (line 86)
    a_412533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'a')
    # Getting the type of 'b' (line 86)
    b_412534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'b')
    # Applying the binary operator 'div' (line 86)
    result_div_412535 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 14), 'div', a_412533, b_412534)
    
    # Assigning a type to the variable 'tau' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tau', result_div_412535)
    
    # Assigning a BinOp to a Name (line 87):
    
    # Assigning a BinOp to a Name (line 87):
    
    # Call to sign(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'b' (line 87)
    b_412538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'b', False)
    # Processing the call keyword arguments (line 87)
    kwargs_412539 = {}
    # Getting the type of 'np' (line 87)
    np_412536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'np', False)
    # Obtaining the member 'sign' of a type (line 87)
    sign_412537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), np_412536, 'sign')
    # Calling sign(args, kwargs) (line 87)
    sign_call_result_412540 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), sign_412537, *[b_412538], **kwargs_412539)
    
    
    # Call to sqrt(...): (line 87)
    # Processing the call arguments (line 87)
    int_412542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 30), 'int')
    # Getting the type of 'tau' (line 87)
    tau_412543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 34), 'tau', False)
    # Getting the type of 'tau' (line 87)
    tau_412544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 40), 'tau', False)
    # Applying the binary operator '*' (line 87)
    result_mul_412545 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 34), '*', tau_412543, tau_412544)
    
    # Applying the binary operator '+' (line 87)
    result_add_412546 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 30), '+', int_412542, result_mul_412545)
    
    # Processing the call keyword arguments (line 87)
    kwargs_412547 = {}
    # Getting the type of 'sqrt' (line 87)
    sqrt_412541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 87)
    sqrt_call_result_412548 = invoke(stypy.reporting.localization.Localization(__file__, 87, 25), sqrt_412541, *[result_add_412546], **kwargs_412547)
    
    # Applying the binary operator 'div' (line 87)
    result_div_412549 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 12), 'div', sign_call_result_412540, sqrt_call_result_412548)
    
    # Assigning a type to the variable 's' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 's', result_div_412549)
    
    # Assigning a BinOp to a Name (line 88):
    
    # Assigning a BinOp to a Name (line 88):
    # Getting the type of 's' (line 88)
    s_412550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 's')
    # Getting the type of 'tau' (line 88)
    tau_412551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'tau')
    # Applying the binary operator '*' (line 88)
    result_mul_412552 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 12), '*', s_412550, tau_412551)
    
    # Assigning a type to the variable 'c' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'c', result_mul_412552)
    
    # Assigning a BinOp to a Name (line 89):
    
    # Assigning a BinOp to a Name (line 89):
    # Getting the type of 'b' (line 89)
    b_412553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'b')
    # Getting the type of 's' (line 89)
    s_412554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 's')
    # Applying the binary operator 'div' (line 89)
    result_div_412555 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 12), 'div', b_412553, s_412554)
    
    # Assigning a type to the variable 'r' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'r', result_div_412555)
    # SSA branch for the else part of an if statement (line 85)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 91):
    
    # Assigning a BinOp to a Name (line 91):
    # Getting the type of 'b' (line 91)
    b_412556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 14), 'b')
    # Getting the type of 'a' (line 91)
    a_412557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'a')
    # Applying the binary operator 'div' (line 91)
    result_div_412558 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 14), 'div', b_412556, a_412557)
    
    # Assigning a type to the variable 'tau' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tau', result_div_412558)
    
    # Assigning a BinOp to a Name (line 92):
    
    # Assigning a BinOp to a Name (line 92):
    
    # Call to sign(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'a' (line 92)
    a_412561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'a', False)
    # Processing the call keyword arguments (line 92)
    kwargs_412562 = {}
    # Getting the type of 'np' (line 92)
    np_412559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'np', False)
    # Obtaining the member 'sign' of a type (line 92)
    sign_412560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), np_412559, 'sign')
    # Calling sign(args, kwargs) (line 92)
    sign_call_result_412563 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), sign_412560, *[a_412561], **kwargs_412562)
    
    
    # Call to sqrt(...): (line 92)
    # Processing the call arguments (line 92)
    int_412565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 30), 'int')
    # Getting the type of 'tau' (line 92)
    tau_412566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'tau', False)
    # Getting the type of 'tau' (line 92)
    tau_412567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 36), 'tau', False)
    # Applying the binary operator '*' (line 92)
    result_mul_412568 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 32), '*', tau_412566, tau_412567)
    
    # Applying the binary operator '+' (line 92)
    result_add_412569 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 30), '+', int_412565, result_mul_412568)
    
    # Processing the call keyword arguments (line 92)
    kwargs_412570 = {}
    # Getting the type of 'sqrt' (line 92)
    sqrt_412564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 92)
    sqrt_call_result_412571 = invoke(stypy.reporting.localization.Localization(__file__, 92, 25), sqrt_412564, *[result_add_412569], **kwargs_412570)
    
    # Applying the binary operator 'div' (line 92)
    result_div_412572 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 12), 'div', sign_call_result_412563, sqrt_call_result_412571)
    
    # Assigning a type to the variable 'c' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'c', result_div_412572)
    
    # Assigning a BinOp to a Name (line 93):
    
    # Assigning a BinOp to a Name (line 93):
    # Getting the type of 'c' (line 93)
    c_412573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'c')
    # Getting the type of 'tau' (line 93)
    tau_412574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'tau')
    # Applying the binary operator '*' (line 93)
    result_mul_412575 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 12), '*', c_412573, tau_412574)
    
    # Assigning a type to the variable 's' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 's', result_mul_412575)
    
    # Assigning a BinOp to a Name (line 94):
    
    # Assigning a BinOp to a Name (line 94):
    # Getting the type of 'a' (line 94)
    a_412576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'a')
    # Getting the type of 'c' (line 94)
    c_412577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'c')
    # Applying the binary operator 'div' (line 94)
    result_div_412578 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 12), 'div', a_412576, c_412577)
    
    # Assigning a type to the variable 'r' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'r', result_div_412578)
    # SSA join for if statement (line 85)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 95)
    tuple_412579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 95)
    # Adding element type (line 95)
    # Getting the type of 'c' (line 95)
    c_412580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 11), tuple_412579, c_412580)
    # Adding element type (line 95)
    # Getting the type of 's' (line 95)
    s_412581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 14), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 11), tuple_412579, s_412581)
    # Adding element type (line 95)
    # Getting the type of 'r' (line 95)
    r_412582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 17), 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 11), tuple_412579, r_412582)
    
    # Assigning a type to the variable 'stypy_return_type' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type', tuple_412579)
    
    # ################# End of '_sym_ortho(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_sym_ortho' in the type store
    # Getting the type of 'stypy_return_type' (line 63)
    stypy_return_type_412583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_412583)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_sym_ortho'
    return stypy_return_type_412583

# Assigning a type to the variable '_sym_ortho' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), '_sym_ortho', _sym_ortho)

@norecursion
def lsqr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_412584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 20), 'float')
    float_412585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 30), 'float')
    float_412586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 41), 'float')
    float_412587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 54), 'float')
    # Getting the type of 'None' (line 99)
    None_412588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'None')
    # Getting the type of 'False' (line 99)
    False_412589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'False')
    # Getting the type of 'False' (line 99)
    False_412590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 45), 'False')
    # Getting the type of 'None' (line 99)
    None_412591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 55), 'None')
    defaults = [float_412584, float_412585, float_412586, float_412587, None_412588, False_412589, False_412590, None_412591]
    # Create a new context for function 'lsqr'
    module_type_store = module_type_store.open_function_context('lsqr', 98, 0, False)
    
    # Passed parameters checking function
    lsqr.stypy_localization = localization
    lsqr.stypy_type_of_self = None
    lsqr.stypy_type_store = module_type_store
    lsqr.stypy_function_name = 'lsqr'
    lsqr.stypy_param_names_list = ['A', 'b', 'damp', 'atol', 'btol', 'conlim', 'iter_lim', 'show', 'calc_var', 'x0']
    lsqr.stypy_varargs_param_name = None
    lsqr.stypy_kwargs_param_name = None
    lsqr.stypy_call_defaults = defaults
    lsqr.stypy_call_varargs = varargs
    lsqr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lsqr', ['A', 'b', 'damp', 'atol', 'btol', 'conlim', 'iter_lim', 'show', 'calc_var', 'x0'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lsqr', localization, ['A', 'b', 'damp', 'atol', 'btol', 'conlim', 'iter_lim', 'show', 'calc_var', 'x0'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lsqr(...)' code ##################

    str_412592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, (-1)), 'str', 'Find the least-squares solution to a large, sparse, linear system\n    of equations.\n\n    The function solves ``Ax = b``  or  ``min ||b - Ax||^2`` or\n    ``min ||Ax - b||^2 + d^2 ||x||^2``.\n\n    The matrix A may be square or rectangular (over-determined or\n    under-determined), and may have any rank.\n\n    ::\n\n      1. Unsymmetric equations --    solve  A*x = b\n\n      2. Linear least squares  --    solve  A*x = b\n                                     in the least-squares sense\n\n      3. Damped least squares  --    solve  (   A    )*x = ( b )\n                                            ( damp*I )     ( 0 )\n                                     in the least-squares sense\n\n    Parameters\n    ----------\n    A : {sparse matrix, ndarray, LinearOperator}\n        Representation of an m-by-n matrix.  It is required that\n        the linear operator can produce ``Ax`` and ``A^T x``.\n    b : array_like, shape (m,)\n        Right-hand side vector ``b``.\n    damp : float\n        Damping coefficient.\n    atol, btol : float, optional\n        Stopping tolerances. If both are 1.0e-9 (say), the final\n        residual norm should be accurate to about 9 digits.  (The\n        final x will usually have fewer correct digits, depending on\n        cond(A) and the size of damp.)\n    conlim : float, optional\n        Another stopping tolerance.  lsqr terminates if an estimate of\n        ``cond(A)`` exceeds `conlim`.  For compatible systems ``Ax =\n        b``, `conlim` could be as large as 1.0e+12 (say).  For\n        least-squares problems, conlim should be less than 1.0e+8.\n        Maximum precision can be obtained by setting ``atol = btol =\n        conlim = zero``, but the number of iterations may then be\n        excessive.\n    iter_lim : int, optional\n        Explicit limitation on number of iterations (for safety).\n    show : bool, optional\n        Display an iteration log.\n    calc_var : bool, optional\n        Whether to estimate diagonals of ``(A\'A + damp^2*I)^{-1}``.\n    x0 : array_like, shape (n,), optional\n        Initial guess of x, if None zeros are used.\n\n        .. versionadded:: 1.0.0\n\n    Returns\n    -------\n    x : ndarray of float\n        The final solution.\n    istop : int\n        Gives the reason for termination.\n        1 means x is an approximate solution to Ax = b.\n        2 means x approximately solves the least-squares problem.\n    itn : int\n        Iteration number upon termination.\n    r1norm : float\n        ``norm(r)``, where ``r = b - Ax``.\n    r2norm : float\n        ``sqrt( norm(r)^2  +  damp^2 * norm(x)^2 )``.  Equal to `r1norm` if\n        ``damp == 0``.\n    anorm : float\n        Estimate of Frobenius norm of ``Abar = [[A]; [damp*I]]``.\n    acond : float\n        Estimate of ``cond(Abar)``.\n    arnorm : float\n        Estimate of ``norm(A\'*r - damp^2*x)``.\n    xnorm : float\n        ``norm(x)``\n    var : ndarray of float\n        If ``calc_var`` is True, estimates all diagonals of\n        ``(A\'A)^{-1}`` (if ``damp == 0``) or more generally ``(A\'A +\n        damp^2*I)^{-1}``.  This is well defined if A has full column\n        rank or ``damp > 0``.  (Not sure what var means if ``rank(A)\n        < n`` and ``damp = 0.``)\n\n    Notes\n    -----\n    LSQR uses an iterative method to approximate the solution.  The\n    number of iterations required to reach a certain accuracy depends\n    strongly on the scaling of the problem.  Poor scaling of the rows\n    or columns of A should therefore be avoided where possible.\n\n    For example, in problem 1 the solution is unaltered by\n    row-scaling.  If a row of A is very small or large compared to\n    the other rows of A, the corresponding row of ( A  b ) should be\n    scaled up or down.\n\n    In problems 1 and 2, the solution x is easily recovered\n    following column-scaling.  Unless better information is known,\n    the nonzero columns of A should be scaled so that they all have\n    the same Euclidean norm (e.g., 1.0).\n\n    In problem 3, there is no freedom to re-scale if damp is\n    nonzero.  However, the value of damp should be assigned only\n    after attention has been paid to the scaling of A.\n\n    The parameter damp is intended to help regularize\n    ill-conditioned systems, by preventing the true solution from\n    being very large.  Another aid to regularization is provided by\n    the parameter acond, which may be used to terminate iterations\n    before the computed solution becomes very large.\n\n    If some initial estimate ``x0`` is known and if ``damp == 0``,\n    one could proceed as follows:\n\n      1. Compute a residual vector ``r0 = b - A*x0``.\n      2. Use LSQR to solve the system  ``A*dx = r0``.\n      3. Add the correction dx to obtain a final solution ``x = x0 + dx``.\n\n    This requires that ``x0`` be available before and after the call\n    to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations\n    to solve A*x = b and k2 iterations to solve A*dx = r0.\n    If x0 is "good", norm(r0) will be smaller than norm(b).\n    If the same stopping tolerances atol and btol are used for each\n    system, k1 and k2 will be similar, but the final solution x0 + dx\n    should be more accurate.  The only way to reduce the total work\n    is to use a larger stopping tolerance for the second system.\n    If some value btol is suitable for A*x = b, the larger value\n    btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.\n\n    Preconditioning is another way to reduce the number of iterations.\n    If it is possible to solve a related system ``M*x = b``\n    efficiently, where M approximates A in some helpful way (e.g. M -\n    A has low rank or its elements are small relative to those of A),\n    LSQR may converge more rapidly on the system ``A*M(inverse)*z =\n    b``, after which x can be recovered by solving M*x = z.\n\n    If A is symmetric, LSQR should not be used!\n\n    Alternatives are the symmetric conjugate-gradient method (cg)\n    and/or SYMMLQ.  SYMMLQ is an implementation of symmetric cg that\n    applies to any symmetric A and will converge more rapidly than\n    LSQR.  If A is positive definite, there are other implementations\n    of symmetric cg that require slightly less work per iteration than\n    SYMMLQ (but will take the same number of iterations).\n\n    References\n    ----------\n    .. [1] C. C. Paige and M. A. Saunders (1982a).\n           "LSQR: An algorithm for sparse linear equations and\n           sparse least squares", ACM TOMS 8(1), 43-71.\n    .. [2] C. C. Paige and M. A. Saunders (1982b).\n           "Algorithm 583.  LSQR: Sparse linear equations and least\n           squares problems", ACM TOMS 8(2), 195-209.\n    .. [3] M. A. Saunders (1995).  "Solution of sparse rectangular\n           systems using LSQR and CRAIG", BIT 35, 588-604.\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import lsqr\n    >>> A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)\n\n    The first example has the trivial solution `[0, 0]`\n\n    >>> b = np.array([0., 0., 0.], dtype=float)\n    >>> x, istop, itn, normr = lsqr(A, b)[:4]\n    The exact solution is  x = 0\n    >>> istop\n    0\n    >>> x\n    array([ 0.,  0.])\n\n    The stopping code `istop=0` returned indicates that a vector of zeros was\n    found as a solution. The returned solution `x` indeed contains `[0., 0.]`.\n    The next example has a non-trivial solution:\n\n    >>> b = np.array([1., 0., -1.], dtype=float)\n    >>> x, istop, itn, r1norm = lsqr(A, b)[:4]\n    >>> istop\n    1\n    >>> x\n    array([ 1., -1.])\n    >>> itn\n    1\n    >>> r1norm\n    4.440892098500627e-16\n\n    As indicated by `istop=1`, `lsqr` found a solution obeying the tolerance\n    limits. The given solution `[1., -1.]` obviously solves the equation. The\n    remaining return values include information about the number of iterations\n    (`itn=1`) and the remaining difference of left and right side of the solved\n    equation.\n    The final example demonstrates the behavior in the case where there is no\n    solution for the equation:\n\n    >>> b = np.array([1., 0.01, -1.], dtype=float)\n    >>> x, istop, itn, r1norm = lsqr(A, b)[:4]\n    >>> istop\n    2\n    >>> x\n    array([ 1.00333333, -0.99666667])\n    >>> A.dot(x)-b\n    array([ 0.00333333, -0.00333333,  0.00333333])\n    >>> r1norm\n    0.005773502691896255\n\n    `istop` indicates that the system is inconsistent and thus `x` is rather an\n    approximate solution to the corresponding least-squares problem. `r1norm`\n    contains the norm of the minimal residual that was found.\n    ')
    
    # Assigning a Call to a Name (line 309):
    
    # Assigning a Call to a Name (line 309):
    
    # Call to aslinearoperator(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'A' (line 309)
    A_412594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 25), 'A', False)
    # Processing the call keyword arguments (line 309)
    kwargs_412595 = {}
    # Getting the type of 'aslinearoperator' (line 309)
    aslinearoperator_412593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 309)
    aslinearoperator_call_result_412596 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), aslinearoperator_412593, *[A_412594], **kwargs_412595)
    
    # Assigning a type to the variable 'A' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'A', aslinearoperator_call_result_412596)
    
    # Assigning a Call to a Name (line 310):
    
    # Assigning a Call to a Name (line 310):
    
    # Call to atleast_1d(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'b' (line 310)
    b_412599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 22), 'b', False)
    # Processing the call keyword arguments (line 310)
    kwargs_412600 = {}
    # Getting the type of 'np' (line 310)
    np_412597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 310)
    atleast_1d_412598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), np_412597, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 310)
    atleast_1d_call_result_412601 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), atleast_1d_412598, *[b_412599], **kwargs_412600)
    
    # Assigning a type to the variable 'b' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'b', atleast_1d_call_result_412601)
    
    
    # Getting the type of 'b' (line 311)
    b_412602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 7), 'b')
    # Obtaining the member 'ndim' of a type (line 311)
    ndim_412603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 7), b_412602, 'ndim')
    int_412604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 16), 'int')
    # Applying the binary operator '>' (line 311)
    result_gt_412605 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 7), '>', ndim_412603, int_412604)
    
    # Testing the type of an if condition (line 311)
    if_condition_412606 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 4), result_gt_412605)
    # Assigning a type to the variable 'if_condition_412606' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'if_condition_412606', if_condition_412606)
    # SSA begins for if statement (line 311)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 312):
    
    # Assigning a Call to a Name (line 312):
    
    # Call to squeeze(...): (line 312)
    # Processing the call keyword arguments (line 312)
    kwargs_412609 = {}
    # Getting the type of 'b' (line 312)
    b_412607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'b', False)
    # Obtaining the member 'squeeze' of a type (line 312)
    squeeze_412608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), b_412607, 'squeeze')
    # Calling squeeze(args, kwargs) (line 312)
    squeeze_call_result_412610 = invoke(stypy.reporting.localization.Localization(__file__, 312, 12), squeeze_412608, *[], **kwargs_412609)
    
    # Assigning a type to the variable 'b' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'b', squeeze_call_result_412610)
    # SSA join for if statement (line 311)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 314):
    
    # Assigning a Subscript to a Name (line 314):
    
    # Obtaining the type of the subscript
    int_412611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 4), 'int')
    # Getting the type of 'A' (line 314)
    A_412612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'A')
    # Obtaining the member 'shape' of a type (line 314)
    shape_412613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 11), A_412612, 'shape')
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___412614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 4), shape_412613, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_412615 = invoke(stypy.reporting.localization.Localization(__file__, 314, 4), getitem___412614, int_412611)
    
    # Assigning a type to the variable 'tuple_var_assignment_412473' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'tuple_var_assignment_412473', subscript_call_result_412615)
    
    # Assigning a Subscript to a Name (line 314):
    
    # Obtaining the type of the subscript
    int_412616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 4), 'int')
    # Getting the type of 'A' (line 314)
    A_412617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'A')
    # Obtaining the member 'shape' of a type (line 314)
    shape_412618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 11), A_412617, 'shape')
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___412619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 4), shape_412618, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_412620 = invoke(stypy.reporting.localization.Localization(__file__, 314, 4), getitem___412619, int_412616)
    
    # Assigning a type to the variable 'tuple_var_assignment_412474' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'tuple_var_assignment_412474', subscript_call_result_412620)
    
    # Assigning a Name to a Name (line 314):
    # Getting the type of 'tuple_var_assignment_412473' (line 314)
    tuple_var_assignment_412473_412621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'tuple_var_assignment_412473')
    # Assigning a type to the variable 'm' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'm', tuple_var_assignment_412473_412621)
    
    # Assigning a Name to a Name (line 314):
    # Getting the type of 'tuple_var_assignment_412474' (line 314)
    tuple_var_assignment_412474_412622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'tuple_var_assignment_412474')
    # Assigning a type to the variable 'n' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 7), 'n', tuple_var_assignment_412474_412622)
    
    # Type idiom detected: calculating its left and rigth part (line 315)
    # Getting the type of 'iter_lim' (line 315)
    iter_lim_412623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 7), 'iter_lim')
    # Getting the type of 'None' (line 315)
    None_412624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 19), 'None')
    
    (may_be_412625, more_types_in_union_412626) = may_be_none(iter_lim_412623, None_412624)

    if may_be_412625:

        if more_types_in_union_412626:
            # Runtime conditional SSA (line 315)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 316):
        
        # Assigning a BinOp to a Name (line 316):
        int_412627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 19), 'int')
        # Getting the type of 'n' (line 316)
        n_412628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 23), 'n')
        # Applying the binary operator '*' (line 316)
        result_mul_412629 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 19), '*', int_412627, n_412628)
        
        # Assigning a type to the variable 'iter_lim' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'iter_lim', result_mul_412629)

        if more_types_in_union_412626:
            # SSA join for if statement (line 315)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 317):
    
    # Assigning a Call to a Name (line 317):
    
    # Call to zeros(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 'n' (line 317)
    n_412632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 19), 'n', False)
    # Processing the call keyword arguments (line 317)
    kwargs_412633 = {}
    # Getting the type of 'np' (line 317)
    np_412630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 317)
    zeros_412631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 10), np_412630, 'zeros')
    # Calling zeros(args, kwargs) (line 317)
    zeros_call_result_412634 = invoke(stypy.reporting.localization.Localization(__file__, 317, 10), zeros_412631, *[n_412632], **kwargs_412633)
    
    # Assigning a type to the variable 'var' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'var', zeros_call_result_412634)
    
    # Assigning a Tuple to a Name (line 319):
    
    # Assigning a Tuple to a Name (line 319):
    
    # Obtaining an instance of the builtin type 'tuple' (line 319)
    tuple_412635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 319)
    # Adding element type (line 319)
    str_412636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 11), 'str', 'The exact solution is  x = 0                              ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 11), tuple_412635, str_412636)
    # Adding element type (line 319)
    str_412637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 9), 'str', 'Ax - b is small enough, given atol, btol                  ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 11), tuple_412635, str_412637)
    # Adding element type (line 319)
    str_412638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 9), 'str', 'The least-squares solution is good enough, given atol     ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 11), tuple_412635, str_412638)
    # Adding element type (line 319)
    str_412639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 9), 'str', 'The estimate of cond(Abar) has exceeded conlim            ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 11), tuple_412635, str_412639)
    # Adding element type (line 319)
    str_412640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 9), 'str', 'Ax - b is small enough for this machine                   ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 11), tuple_412635, str_412640)
    # Adding element type (line 319)
    str_412641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 9), 'str', 'The least-squares solution is good enough for this machine')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 11), tuple_412635, str_412641)
    # Adding element type (line 319)
    str_412642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 9), 'str', 'Cond(Abar) seems to be too large for this machine         ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 11), tuple_412635, str_412642)
    # Adding element type (line 319)
    str_412643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 9), 'str', 'The iteration limit has been reached                      ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 11), tuple_412635, str_412643)
    
    # Assigning a type to the variable 'msg' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'msg', tuple_412635)
    
    # Getting the type of 'show' (line 328)
    show_412644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 7), 'show')
    # Testing the type of an if condition (line 328)
    if_condition_412645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 4), show_412644)
    # Assigning a type to the variable 'if_condition_412645' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'if_condition_412645', if_condition_412645)
    # SSA begins for if statement (line 328)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 329)
    # Processing the call arguments (line 329)
    str_412647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 14), 'str', ' ')
    # Processing the call keyword arguments (line 329)
    kwargs_412648 = {}
    # Getting the type of 'print' (line 329)
    print_412646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'print', False)
    # Calling print(args, kwargs) (line 329)
    print_call_result_412649 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), print_412646, *[str_412647], **kwargs_412648)
    
    
    # Call to print(...): (line 330)
    # Processing the call arguments (line 330)
    str_412651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 14), 'str', 'LSQR            Least-squares solution of  Ax = b')
    # Processing the call keyword arguments (line 330)
    kwargs_412652 = {}
    # Getting the type of 'print' (line 330)
    print_412650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'print', False)
    # Calling print(args, kwargs) (line 330)
    print_call_result_412653 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), print_412650, *[str_412651], **kwargs_412652)
    
    
    # Assigning a BinOp to a Name (line 331):
    
    # Assigning a BinOp to a Name (line 331):
    str_412654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 15), 'str', 'The matrix A has %8g rows  and %8g cols')
    
    # Obtaining an instance of the builtin type 'tuple' (line 331)
    tuple_412655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 60), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 331)
    # Adding element type (line 331)
    # Getting the type of 'm' (line 331)
    m_412656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 60), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 60), tuple_412655, m_412656)
    # Adding element type (line 331)
    # Getting the type of 'n' (line 331)
    n_412657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 63), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 60), tuple_412655, n_412657)
    
    # Applying the binary operator '%' (line 331)
    result_mod_412658 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 15), '%', str_412654, tuple_412655)
    
    # Assigning a type to the variable 'str1' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'str1', result_mod_412658)
    
    # Assigning a BinOp to a Name (line 332):
    
    # Assigning a BinOp to a Name (line 332):
    str_412659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 15), 'str', 'damp = %20.14e   calc_var = %8g')
    
    # Obtaining an instance of the builtin type 'tuple' (line 332)
    tuple_412660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 52), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 332)
    # Adding element type (line 332)
    # Getting the type of 'damp' (line 332)
    damp_412661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 52), 'damp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 52), tuple_412660, damp_412661)
    # Adding element type (line 332)
    # Getting the type of 'calc_var' (line 332)
    calc_var_412662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 58), 'calc_var')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 52), tuple_412660, calc_var_412662)
    
    # Applying the binary operator '%' (line 332)
    result_mod_412663 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 15), '%', str_412659, tuple_412660)
    
    # Assigning a type to the variable 'str2' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'str2', result_mod_412663)
    
    # Assigning a BinOp to a Name (line 333):
    
    # Assigning a BinOp to a Name (line 333):
    str_412664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 15), 'str', 'atol = %8.2e                 conlim = %8.2e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 333)
    tuple_412665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 64), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 333)
    # Adding element type (line 333)
    # Getting the type of 'atol' (line 333)
    atol_412666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 64), 'atol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 64), tuple_412665, atol_412666)
    # Adding element type (line 333)
    # Getting the type of 'conlim' (line 333)
    conlim_412667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 70), 'conlim')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 64), tuple_412665, conlim_412667)
    
    # Applying the binary operator '%' (line 333)
    result_mod_412668 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 15), '%', str_412664, tuple_412665)
    
    # Assigning a type to the variable 'str3' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'str3', result_mod_412668)
    
    # Assigning a BinOp to a Name (line 334):
    
    # Assigning a BinOp to a Name (line 334):
    str_412669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 15), 'str', 'btol = %8.2e               iter_lim = %8g')
    
    # Obtaining an instance of the builtin type 'tuple' (line 334)
    tuple_412670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 62), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 334)
    # Adding element type (line 334)
    # Getting the type of 'btol' (line 334)
    btol_412671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 62), 'btol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 62), tuple_412670, btol_412671)
    # Adding element type (line 334)
    # Getting the type of 'iter_lim' (line 334)
    iter_lim_412672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 68), 'iter_lim')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 62), tuple_412670, iter_lim_412672)
    
    # Applying the binary operator '%' (line 334)
    result_mod_412673 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 15), '%', str_412669, tuple_412670)
    
    # Assigning a type to the variable 'str4' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'str4', result_mod_412673)
    
    # Call to print(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'str1' (line 335)
    str1_412675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 14), 'str1', False)
    # Processing the call keyword arguments (line 335)
    kwargs_412676 = {}
    # Getting the type of 'print' (line 335)
    print_412674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'print', False)
    # Calling print(args, kwargs) (line 335)
    print_call_result_412677 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), print_412674, *[str1_412675], **kwargs_412676)
    
    
    # Call to print(...): (line 336)
    # Processing the call arguments (line 336)
    # Getting the type of 'str2' (line 336)
    str2_412679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 14), 'str2', False)
    # Processing the call keyword arguments (line 336)
    kwargs_412680 = {}
    # Getting the type of 'print' (line 336)
    print_412678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'print', False)
    # Calling print(args, kwargs) (line 336)
    print_call_result_412681 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), print_412678, *[str2_412679], **kwargs_412680)
    
    
    # Call to print(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'str3' (line 337)
    str3_412683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 14), 'str3', False)
    # Processing the call keyword arguments (line 337)
    kwargs_412684 = {}
    # Getting the type of 'print' (line 337)
    print_412682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'print', False)
    # Calling print(args, kwargs) (line 337)
    print_call_result_412685 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), print_412682, *[str3_412683], **kwargs_412684)
    
    
    # Call to print(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'str4' (line 338)
    str4_412687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 14), 'str4', False)
    # Processing the call keyword arguments (line 338)
    kwargs_412688 = {}
    # Getting the type of 'print' (line 338)
    print_412686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'print', False)
    # Calling print(args, kwargs) (line 338)
    print_call_result_412689 = invoke(stypy.reporting.localization.Localization(__file__, 338, 8), print_412686, *[str4_412687], **kwargs_412688)
    
    # SSA join for if statement (line 328)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 340):
    
    # Assigning a Num to a Name (line 340):
    int_412690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 10), 'int')
    # Assigning a type to the variable 'itn' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'itn', int_412690)
    
    # Assigning a Num to a Name (line 341):
    
    # Assigning a Num to a Name (line 341):
    int_412691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 12), 'int')
    # Assigning a type to the variable 'istop' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'istop', int_412691)
    
    # Assigning a Num to a Name (line 342):
    
    # Assigning a Num to a Name (line 342):
    int_412692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 11), 'int')
    # Assigning a type to the variable 'ctol' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'ctol', int_412692)
    
    
    # Getting the type of 'conlim' (line 343)
    conlim_412693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 7), 'conlim')
    int_412694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 16), 'int')
    # Applying the binary operator '>' (line 343)
    result_gt_412695 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 7), '>', conlim_412693, int_412694)
    
    # Testing the type of an if condition (line 343)
    if_condition_412696 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 4), result_gt_412695)
    # Assigning a type to the variable 'if_condition_412696' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'if_condition_412696', if_condition_412696)
    # SSA begins for if statement (line 343)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 344):
    
    # Assigning a BinOp to a Name (line 344):
    int_412697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 15), 'int')
    # Getting the type of 'conlim' (line 344)
    conlim_412698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 17), 'conlim')
    # Applying the binary operator 'div' (line 344)
    result_div_412699 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 15), 'div', int_412697, conlim_412698)
    
    # Assigning a type to the variable 'ctol' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'ctol', result_div_412699)
    # SSA join for if statement (line 343)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 345):
    
    # Assigning a Num to a Name (line 345):
    int_412700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 12), 'int')
    # Assigning a type to the variable 'anorm' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'anorm', int_412700)
    
    # Assigning a Num to a Name (line 346):
    
    # Assigning a Num to a Name (line 346):
    int_412701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 12), 'int')
    # Assigning a type to the variable 'acond' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'acond', int_412701)
    
    # Assigning a BinOp to a Name (line 347):
    
    # Assigning a BinOp to a Name (line 347):
    # Getting the type of 'damp' (line 347)
    damp_412702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 13), 'damp')
    int_412703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 19), 'int')
    # Applying the binary operator '**' (line 347)
    result_pow_412704 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 13), '**', damp_412702, int_412703)
    
    # Assigning a type to the variable 'dampsq' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'dampsq', result_pow_412704)
    
    # Assigning a Num to a Name (line 348):
    
    # Assigning a Num to a Name (line 348):
    int_412705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 13), 'int')
    # Assigning a type to the variable 'ddnorm' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'ddnorm', int_412705)
    
    # Assigning a Num to a Name (line 349):
    
    # Assigning a Num to a Name (line 349):
    int_412706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 11), 'int')
    # Assigning a type to the variable 'res2' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'res2', int_412706)
    
    # Assigning a Num to a Name (line 350):
    
    # Assigning a Num to a Name (line 350):
    int_412707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 12), 'int')
    # Assigning a type to the variable 'xnorm' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'xnorm', int_412707)
    
    # Assigning a Num to a Name (line 351):
    
    # Assigning a Num to a Name (line 351):
    int_412708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 13), 'int')
    # Assigning a type to the variable 'xxnorm' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'xxnorm', int_412708)
    
    # Assigning a Num to a Name (line 352):
    
    # Assigning a Num to a Name (line 352):
    int_412709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 8), 'int')
    # Assigning a type to the variable 'z' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'z', int_412709)
    
    # Assigning a Num to a Name (line 353):
    
    # Assigning a Num to a Name (line 353):
    int_412710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 10), 'int')
    # Assigning a type to the variable 'cs2' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'cs2', int_412710)
    
    # Assigning a Num to a Name (line 354):
    
    # Assigning a Num to a Name (line 354):
    int_412711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 10), 'int')
    # Assigning a type to the variable 'sn2' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'sn2', int_412711)
    str_412712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, (-1)), 'str', "\n    Set up the first vectors u and v for the bidiagonalization.\n    These satisfy  beta*u = b - A*x,  alfa*v = A'*u.\n    ")
    
    # Assigning a Name to a Name (line 360):
    
    # Assigning a Name to a Name (line 360):
    # Getting the type of 'b' (line 360)
    b_412713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'b')
    # Assigning a type to the variable 'u' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'u', b_412713)
    
    # Assigning a Call to a Name (line 361):
    
    # Assigning a Call to a Name (line 361):
    
    # Call to norm(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'b' (line 361)
    b_412717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 27), 'b', False)
    # Processing the call keyword arguments (line 361)
    kwargs_412718 = {}
    # Getting the type of 'np' (line 361)
    np_412714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'np', False)
    # Obtaining the member 'linalg' of a type (line 361)
    linalg_412715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 12), np_412714, 'linalg')
    # Obtaining the member 'norm' of a type (line 361)
    norm_412716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 12), linalg_412715, 'norm')
    # Calling norm(args, kwargs) (line 361)
    norm_call_result_412719 = invoke(stypy.reporting.localization.Localization(__file__, 361, 12), norm_412716, *[b_412717], **kwargs_412718)
    
    # Assigning a type to the variable 'bnorm' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'bnorm', norm_call_result_412719)
    
    # Type idiom detected: calculating its left and rigth part (line 362)
    # Getting the type of 'x0' (line 362)
    x0_412720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 7), 'x0')
    # Getting the type of 'None' (line 362)
    None_412721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 13), 'None')
    
    (may_be_412722, more_types_in_union_412723) = may_be_none(x0_412720, None_412721)

    if may_be_412722:

        if more_types_in_union_412723:
            # Runtime conditional SSA (line 362)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 363):
        
        # Assigning a Call to a Name (line 363):
        
        # Call to zeros(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'n' (line 363)
        n_412726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 21), 'n', False)
        # Processing the call keyword arguments (line 363)
        kwargs_412727 = {}
        # Getting the type of 'np' (line 363)
        np_412724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 363)
        zeros_412725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), np_412724, 'zeros')
        # Calling zeros(args, kwargs) (line 363)
        zeros_call_result_412728 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), zeros_412725, *[n_412726], **kwargs_412727)
        
        # Assigning a type to the variable 'x' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'x', zeros_call_result_412728)
        
        # Assigning a Call to a Name (line 364):
        
        # Assigning a Call to a Name (line 364):
        
        # Call to copy(...): (line 364)
        # Processing the call keyword arguments (line 364)
        kwargs_412731 = {}
        # Getting the type of 'bnorm' (line 364)
        bnorm_412729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 15), 'bnorm', False)
        # Obtaining the member 'copy' of a type (line 364)
        copy_412730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 15), bnorm_412729, 'copy')
        # Calling copy(args, kwargs) (line 364)
        copy_call_result_412732 = invoke(stypy.reporting.localization.Localization(__file__, 364, 15), copy_412730, *[], **kwargs_412731)
        
        # Assigning a type to the variable 'beta' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'beta', copy_call_result_412732)

        if more_types_in_union_412723:
            # Runtime conditional SSA for else branch (line 362)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_412722) or more_types_in_union_412723):
        
        # Assigning a Call to a Name (line 366):
        
        # Assigning a Call to a Name (line 366):
        
        # Call to asarray(...): (line 366)
        # Processing the call arguments (line 366)
        # Getting the type of 'x0' (line 366)
        x0_412735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 23), 'x0', False)
        # Processing the call keyword arguments (line 366)
        kwargs_412736 = {}
        # Getting the type of 'np' (line 366)
        np_412733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 366)
        asarray_412734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 12), np_412733, 'asarray')
        # Calling asarray(args, kwargs) (line 366)
        asarray_call_result_412737 = invoke(stypy.reporting.localization.Localization(__file__, 366, 12), asarray_412734, *[x0_412735], **kwargs_412736)
        
        # Assigning a type to the variable 'x' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'x', asarray_call_result_412737)
        
        # Assigning a BinOp to a Name (line 367):
        
        # Assigning a BinOp to a Name (line 367):
        # Getting the type of 'u' (line 367)
        u_412738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'u')
        
        # Call to matvec(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'x' (line 367)
        x_412741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 25), 'x', False)
        # Processing the call keyword arguments (line 367)
        kwargs_412742 = {}
        # Getting the type of 'A' (line 367)
        A_412739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'A', False)
        # Obtaining the member 'matvec' of a type (line 367)
        matvec_412740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 16), A_412739, 'matvec')
        # Calling matvec(args, kwargs) (line 367)
        matvec_call_result_412743 = invoke(stypy.reporting.localization.Localization(__file__, 367, 16), matvec_412740, *[x_412741], **kwargs_412742)
        
        # Applying the binary operator '-' (line 367)
        result_sub_412744 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 12), '-', u_412738, matvec_call_result_412743)
        
        # Assigning a type to the variable 'u' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'u', result_sub_412744)
        
        # Assigning a Call to a Name (line 368):
        
        # Assigning a Call to a Name (line 368):
        
        # Call to norm(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'u' (line 368)
        u_412748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 30), 'u', False)
        # Processing the call keyword arguments (line 368)
        kwargs_412749 = {}
        # Getting the type of 'np' (line 368)
        np_412745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 15), 'np', False)
        # Obtaining the member 'linalg' of a type (line 368)
        linalg_412746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 15), np_412745, 'linalg')
        # Obtaining the member 'norm' of a type (line 368)
        norm_412747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 15), linalg_412746, 'norm')
        # Calling norm(args, kwargs) (line 368)
        norm_call_result_412750 = invoke(stypy.reporting.localization.Localization(__file__, 368, 15), norm_412747, *[u_412748], **kwargs_412749)
        
        # Assigning a type to the variable 'beta' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'beta', norm_call_result_412750)

        if (may_be_412722 and more_types_in_union_412723):
            # SSA join for if statement (line 362)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'beta' (line 370)
    beta_412751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 7), 'beta')
    int_412752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 14), 'int')
    # Applying the binary operator '>' (line 370)
    result_gt_412753 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 7), '>', beta_412751, int_412752)
    
    # Testing the type of an if condition (line 370)
    if_condition_412754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 4), result_gt_412753)
    # Assigning a type to the variable 'if_condition_412754' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'if_condition_412754', if_condition_412754)
    # SSA begins for if statement (line 370)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 371):
    
    # Assigning a BinOp to a Name (line 371):
    int_412755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 13), 'int')
    # Getting the type of 'beta' (line 371)
    beta_412756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 15), 'beta')
    # Applying the binary operator 'div' (line 371)
    result_div_412757 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 13), 'div', int_412755, beta_412756)
    
    # Getting the type of 'u' (line 371)
    u_412758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 23), 'u')
    # Applying the binary operator '*' (line 371)
    result_mul_412759 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 12), '*', result_div_412757, u_412758)
    
    # Assigning a type to the variable 'u' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'u', result_mul_412759)
    
    # Assigning a Call to a Name (line 372):
    
    # Assigning a Call to a Name (line 372):
    
    # Call to rmatvec(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'u' (line 372)
    u_412762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 22), 'u', False)
    # Processing the call keyword arguments (line 372)
    kwargs_412763 = {}
    # Getting the type of 'A' (line 372)
    A_412760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'A', False)
    # Obtaining the member 'rmatvec' of a type (line 372)
    rmatvec_412761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 12), A_412760, 'rmatvec')
    # Calling rmatvec(args, kwargs) (line 372)
    rmatvec_call_result_412764 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), rmatvec_412761, *[u_412762], **kwargs_412763)
    
    # Assigning a type to the variable 'v' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'v', rmatvec_call_result_412764)
    
    # Assigning a Call to a Name (line 373):
    
    # Assigning a Call to a Name (line 373):
    
    # Call to norm(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of 'v' (line 373)
    v_412768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 30), 'v', False)
    # Processing the call keyword arguments (line 373)
    kwargs_412769 = {}
    # Getting the type of 'np' (line 373)
    np_412765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 15), 'np', False)
    # Obtaining the member 'linalg' of a type (line 373)
    linalg_412766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 15), np_412765, 'linalg')
    # Obtaining the member 'norm' of a type (line 373)
    norm_412767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 15), linalg_412766, 'norm')
    # Calling norm(args, kwargs) (line 373)
    norm_call_result_412770 = invoke(stypy.reporting.localization.Localization(__file__, 373, 15), norm_412767, *[v_412768], **kwargs_412769)
    
    # Assigning a type to the variable 'alfa' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'alfa', norm_call_result_412770)
    # SSA branch for the else part of an if statement (line 370)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 375):
    
    # Assigning a Call to a Name (line 375):
    
    # Call to copy(...): (line 375)
    # Processing the call keyword arguments (line 375)
    kwargs_412773 = {}
    # Getting the type of 'x' (line 375)
    x_412771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'x', False)
    # Obtaining the member 'copy' of a type (line 375)
    copy_412772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 12), x_412771, 'copy')
    # Calling copy(args, kwargs) (line 375)
    copy_call_result_412774 = invoke(stypy.reporting.localization.Localization(__file__, 375, 12), copy_412772, *[], **kwargs_412773)
    
    # Assigning a type to the variable 'v' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'v', copy_call_result_412774)
    
    # Assigning a Num to a Name (line 376):
    
    # Assigning a Num to a Name (line 376):
    int_412775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 15), 'int')
    # Assigning a type to the variable 'alfa' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'alfa', int_412775)
    # SSA join for if statement (line 370)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'alfa' (line 378)
    alfa_412776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 7), 'alfa')
    int_412777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 14), 'int')
    # Applying the binary operator '>' (line 378)
    result_gt_412778 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 7), '>', alfa_412776, int_412777)
    
    # Testing the type of an if condition (line 378)
    if_condition_412779 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 4), result_gt_412778)
    # Assigning a type to the variable 'if_condition_412779' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'if_condition_412779', if_condition_412779)
    # SSA begins for if statement (line 378)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 379):
    
    # Assigning a BinOp to a Name (line 379):
    int_412780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 13), 'int')
    # Getting the type of 'alfa' (line 379)
    alfa_412781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'alfa')
    # Applying the binary operator 'div' (line 379)
    result_div_412782 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 13), 'div', int_412780, alfa_412781)
    
    # Getting the type of 'v' (line 379)
    v_412783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 23), 'v')
    # Applying the binary operator '*' (line 379)
    result_mul_412784 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 12), '*', result_div_412782, v_412783)
    
    # Assigning a type to the variable 'v' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'v', result_mul_412784)
    # SSA join for if statement (line 378)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 380):
    
    # Assigning a Call to a Name (line 380):
    
    # Call to copy(...): (line 380)
    # Processing the call keyword arguments (line 380)
    kwargs_412787 = {}
    # Getting the type of 'v' (line 380)
    v_412785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'v', False)
    # Obtaining the member 'copy' of a type (line 380)
    copy_412786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), v_412785, 'copy')
    # Calling copy(args, kwargs) (line 380)
    copy_call_result_412788 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), copy_412786, *[], **kwargs_412787)
    
    # Assigning a type to the variable 'w' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'w', copy_call_result_412788)
    
    # Assigning a Name to a Name (line 382):
    
    # Assigning a Name to a Name (line 382):
    # Getting the type of 'alfa' (line 382)
    alfa_412789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 13), 'alfa')
    # Assigning a type to the variable 'rhobar' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'rhobar', alfa_412789)
    
    # Assigning a Name to a Name (line 383):
    
    # Assigning a Name to a Name (line 383):
    # Getting the type of 'beta' (line 383)
    beta_412790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 13), 'beta')
    # Assigning a type to the variable 'phibar' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'phibar', beta_412790)
    
    # Assigning a Name to a Name (line 384):
    
    # Assigning a Name to a Name (line 384):
    # Getting the type of 'beta' (line 384)
    beta_412791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'beta')
    # Assigning a type to the variable 'rnorm' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'rnorm', beta_412791)
    
    # Assigning a Name to a Name (line 385):
    
    # Assigning a Name to a Name (line 385):
    # Getting the type of 'rnorm' (line 385)
    rnorm_412792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 13), 'rnorm')
    # Assigning a type to the variable 'r1norm' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'r1norm', rnorm_412792)
    
    # Assigning a Name to a Name (line 386):
    
    # Assigning a Name to a Name (line 386):
    # Getting the type of 'rnorm' (line 386)
    rnorm_412793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 13), 'rnorm')
    # Assigning a type to the variable 'r2norm' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'r2norm', rnorm_412793)
    
    # Assigning a BinOp to a Name (line 390):
    
    # Assigning a BinOp to a Name (line 390):
    # Getting the type of 'alfa' (line 390)
    alfa_412794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 13), 'alfa')
    # Getting the type of 'beta' (line 390)
    beta_412795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 20), 'beta')
    # Applying the binary operator '*' (line 390)
    result_mul_412796 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 13), '*', alfa_412794, beta_412795)
    
    # Assigning a type to the variable 'arnorm' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'arnorm', result_mul_412796)
    
    
    # Getting the type of 'arnorm' (line 391)
    arnorm_412797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 7), 'arnorm')
    int_412798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 17), 'int')
    # Applying the binary operator '==' (line 391)
    result_eq_412799 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 7), '==', arnorm_412797, int_412798)
    
    # Testing the type of an if condition (line 391)
    if_condition_412800 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 4), result_eq_412799)
    # Assigning a type to the variable 'if_condition_412800' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'if_condition_412800', if_condition_412800)
    # SSA begins for if statement (line 391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 392)
    # Processing the call arguments (line 392)
    
    # Obtaining the type of the subscript
    int_412802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 18), 'int')
    # Getting the type of 'msg' (line 392)
    msg_412803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 14), 'msg', False)
    # Obtaining the member '__getitem__' of a type (line 392)
    getitem___412804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 14), msg_412803, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 392)
    subscript_call_result_412805 = invoke(stypy.reporting.localization.Localization(__file__, 392, 14), getitem___412804, int_412802)
    
    # Processing the call keyword arguments (line 392)
    kwargs_412806 = {}
    # Getting the type of 'print' (line 392)
    print_412801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'print', False)
    # Calling print(args, kwargs) (line 392)
    print_call_result_412807 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), print_412801, *[subscript_call_result_412805], **kwargs_412806)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 393)
    tuple_412808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 393)
    # Adding element type (line 393)
    # Getting the type of 'x' (line 393)
    x_412809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_412808, x_412809)
    # Adding element type (line 393)
    # Getting the type of 'istop' (line 393)
    istop_412810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 18), 'istop')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_412808, istop_412810)
    # Adding element type (line 393)
    # Getting the type of 'itn' (line 393)
    itn_412811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 25), 'itn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_412808, itn_412811)
    # Adding element type (line 393)
    # Getting the type of 'r1norm' (line 393)
    r1norm_412812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 30), 'r1norm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_412808, r1norm_412812)
    # Adding element type (line 393)
    # Getting the type of 'r2norm' (line 393)
    r2norm_412813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 38), 'r2norm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_412808, r2norm_412813)
    # Adding element type (line 393)
    # Getting the type of 'anorm' (line 393)
    anorm_412814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 46), 'anorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_412808, anorm_412814)
    # Adding element type (line 393)
    # Getting the type of 'acond' (line 393)
    acond_412815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 53), 'acond')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_412808, acond_412815)
    # Adding element type (line 393)
    # Getting the type of 'arnorm' (line 393)
    arnorm_412816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 60), 'arnorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_412808, arnorm_412816)
    # Adding element type (line 393)
    # Getting the type of 'xnorm' (line 393)
    xnorm_412817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 68), 'xnorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_412808, xnorm_412817)
    # Adding element type (line 393)
    # Getting the type of 'var' (line 393)
    var_412818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 75), 'var')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_412808, var_412818)
    
    # Assigning a type to the variable 'stypy_return_type' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'stypy_return_type', tuple_412808)
    # SSA join for if statement (line 391)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 395):
    
    # Assigning a Str to a Name (line 395):
    str_412819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 12), 'str', '   Itn      x[0]       r1norm     r2norm ')
    # Assigning a type to the variable 'head1' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'head1', str_412819)
    
    # Assigning a Str to a Name (line 396):
    
    # Assigning a Str to a Name (line 396):
    str_412820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 12), 'str', ' Compatible    LS      Norm A   Cond A')
    # Assigning a type to the variable 'head2' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'head2', str_412820)
    
    # Getting the type of 'show' (line 398)
    show_412821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 7), 'show')
    # Testing the type of an if condition (line 398)
    if_condition_412822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 398, 4), show_412821)
    # Assigning a type to the variable 'if_condition_412822' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'if_condition_412822', if_condition_412822)
    # SSA begins for if statement (line 398)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 399)
    # Processing the call arguments (line 399)
    str_412824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 14), 'str', ' ')
    # Processing the call keyword arguments (line 399)
    kwargs_412825 = {}
    # Getting the type of 'print' (line 399)
    print_412823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'print', False)
    # Calling print(args, kwargs) (line 399)
    print_call_result_412826 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), print_412823, *[str_412824], **kwargs_412825)
    
    
    # Call to print(...): (line 400)
    # Processing the call arguments (line 400)
    # Getting the type of 'head1' (line 400)
    head1_412828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 14), 'head1', False)
    # Getting the type of 'head2' (line 400)
    head2_412829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 21), 'head2', False)
    # Processing the call keyword arguments (line 400)
    kwargs_412830 = {}
    # Getting the type of 'print' (line 400)
    print_412827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'print', False)
    # Calling print(args, kwargs) (line 400)
    print_call_result_412831 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), print_412827, *[head1_412828, head2_412829], **kwargs_412830)
    
    
    # Assigning a Num to a Name (line 401):
    
    # Assigning a Num to a Name (line 401):
    int_412832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 16), 'int')
    # Assigning a type to the variable 'test1' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'test1', int_412832)
    
    # Assigning a BinOp to a Name (line 402):
    
    # Assigning a BinOp to a Name (line 402):
    # Getting the type of 'alfa' (line 402)
    alfa_412833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 16), 'alfa')
    # Getting the type of 'beta' (line 402)
    beta_412834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 23), 'beta')
    # Applying the binary operator 'div' (line 402)
    result_div_412835 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 16), 'div', alfa_412833, beta_412834)
    
    # Assigning a type to the variable 'test2' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'test2', result_div_412835)
    
    # Assigning a BinOp to a Name (line 403):
    
    # Assigning a BinOp to a Name (line 403):
    str_412836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 15), 'str', '%6g %12.5e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 403)
    tuple_412837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 403)
    # Adding element type (line 403)
    # Getting the type of 'itn' (line 403)
    itn_412838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 31), 'itn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 31), tuple_412837, itn_412838)
    # Adding element type (line 403)
    
    # Obtaining the type of the subscript
    int_412839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 38), 'int')
    # Getting the type of 'x' (line 403)
    x_412840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 36), 'x')
    # Obtaining the member '__getitem__' of a type (line 403)
    getitem___412841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 36), x_412840, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 403)
    subscript_call_result_412842 = invoke(stypy.reporting.localization.Localization(__file__, 403, 36), getitem___412841, int_412839)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 31), tuple_412837, subscript_call_result_412842)
    
    # Applying the binary operator '%' (line 403)
    result_mod_412843 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 15), '%', str_412836, tuple_412837)
    
    # Assigning a type to the variable 'str1' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'str1', result_mod_412843)
    
    # Assigning a BinOp to a Name (line 404):
    
    # Assigning a BinOp to a Name (line 404):
    str_412844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 15), 'str', ' %10.3e %10.3e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 404)
    tuple_412845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 404)
    # Adding element type (line 404)
    # Getting the type of 'r1norm' (line 404)
    r1norm_412846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 35), 'r1norm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 35), tuple_412845, r1norm_412846)
    # Adding element type (line 404)
    # Getting the type of 'r2norm' (line 404)
    r2norm_412847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 43), 'r2norm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 35), tuple_412845, r2norm_412847)
    
    # Applying the binary operator '%' (line 404)
    result_mod_412848 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 15), '%', str_412844, tuple_412845)
    
    # Assigning a type to the variable 'str2' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'str2', result_mod_412848)
    
    # Assigning a BinOp to a Name (line 405):
    
    # Assigning a BinOp to a Name (line 405):
    str_412849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 15), 'str', '  %8.1e %8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 405)
    tuple_412850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 405)
    # Adding element type (line 405)
    # Getting the type of 'test1' (line 405)
    test1_412851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 34), 'test1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 34), tuple_412850, test1_412851)
    # Adding element type (line 405)
    # Getting the type of 'test2' (line 405)
    test2_412852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 41), 'test2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 34), tuple_412850, test2_412852)
    
    # Applying the binary operator '%' (line 405)
    result_mod_412853 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 15), '%', str_412849, tuple_412850)
    
    # Assigning a type to the variable 'str3' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'str3', result_mod_412853)
    
    # Call to print(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'str1' (line 406)
    str1_412855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 14), 'str1', False)
    # Getting the type of 'str2' (line 406)
    str2_412856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'str2', False)
    # Getting the type of 'str3' (line 406)
    str3_412857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 26), 'str3', False)
    # Processing the call keyword arguments (line 406)
    kwargs_412858 = {}
    # Getting the type of 'print' (line 406)
    print_412854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'print', False)
    # Calling print(args, kwargs) (line 406)
    print_call_result_412859 = invoke(stypy.reporting.localization.Localization(__file__, 406, 8), print_412854, *[str1_412855, str2_412856, str3_412857], **kwargs_412858)
    
    # SSA join for if statement (line 398)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'itn' (line 409)
    itn_412860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 10), 'itn')
    # Getting the type of 'iter_lim' (line 409)
    iter_lim_412861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 16), 'iter_lim')
    # Applying the binary operator '<' (line 409)
    result_lt_412862 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 10), '<', itn_412860, iter_lim_412861)
    
    # Testing the type of an if condition (line 409)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 409, 4), result_lt_412862)
    # SSA begins for while statement (line 409)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 410):
    
    # Assigning a BinOp to a Name (line 410):
    # Getting the type of 'itn' (line 410)
    itn_412863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 14), 'itn')
    int_412864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 20), 'int')
    # Applying the binary operator '+' (line 410)
    result_add_412865 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 14), '+', itn_412863, int_412864)
    
    # Assigning a type to the variable 'itn' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'itn', result_add_412865)
    str_412866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, (-1)), 'str', "\n        %     Perform the next step of the bidiagonalization to obtain the\n        %     next  beta, u, alfa, v.  These satisfy the relations\n        %                beta*u  =  a*v   -  alfa*u,\n        %                alfa*v  =  A'*u  -  beta*v.\n        ")
    
    # Assigning a BinOp to a Name (line 417):
    
    # Assigning a BinOp to a Name (line 417):
    
    # Call to matvec(...): (line 417)
    # Processing the call arguments (line 417)
    # Getting the type of 'v' (line 417)
    v_412869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 21), 'v', False)
    # Processing the call keyword arguments (line 417)
    kwargs_412870 = {}
    # Getting the type of 'A' (line 417)
    A_412867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'A', False)
    # Obtaining the member 'matvec' of a type (line 417)
    matvec_412868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 12), A_412867, 'matvec')
    # Calling matvec(args, kwargs) (line 417)
    matvec_call_result_412871 = invoke(stypy.reporting.localization.Localization(__file__, 417, 12), matvec_412868, *[v_412869], **kwargs_412870)
    
    # Getting the type of 'alfa' (line 417)
    alfa_412872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 26), 'alfa')
    # Getting the type of 'u' (line 417)
    u_412873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 33), 'u')
    # Applying the binary operator '*' (line 417)
    result_mul_412874 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 26), '*', alfa_412872, u_412873)
    
    # Applying the binary operator '-' (line 417)
    result_sub_412875 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 12), '-', matvec_call_result_412871, result_mul_412874)
    
    # Assigning a type to the variable 'u' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'u', result_sub_412875)
    
    # Assigning a Call to a Name (line 418):
    
    # Assigning a Call to a Name (line 418):
    
    # Call to norm(...): (line 418)
    # Processing the call arguments (line 418)
    # Getting the type of 'u' (line 418)
    u_412879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 30), 'u', False)
    # Processing the call keyword arguments (line 418)
    kwargs_412880 = {}
    # Getting the type of 'np' (line 418)
    np_412876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'np', False)
    # Obtaining the member 'linalg' of a type (line 418)
    linalg_412877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 15), np_412876, 'linalg')
    # Obtaining the member 'norm' of a type (line 418)
    norm_412878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 15), linalg_412877, 'norm')
    # Calling norm(args, kwargs) (line 418)
    norm_call_result_412881 = invoke(stypy.reporting.localization.Localization(__file__, 418, 15), norm_412878, *[u_412879], **kwargs_412880)
    
    # Assigning a type to the variable 'beta' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'beta', norm_call_result_412881)
    
    
    # Getting the type of 'beta' (line 420)
    beta_412882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 11), 'beta')
    int_412883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 18), 'int')
    # Applying the binary operator '>' (line 420)
    result_gt_412884 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 11), '>', beta_412882, int_412883)
    
    # Testing the type of an if condition (line 420)
    if_condition_412885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 8), result_gt_412884)
    # Assigning a type to the variable 'if_condition_412885' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'if_condition_412885', if_condition_412885)
    # SSA begins for if statement (line 420)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 421):
    
    # Assigning a BinOp to a Name (line 421):
    int_412886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 17), 'int')
    # Getting the type of 'beta' (line 421)
    beta_412887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 19), 'beta')
    # Applying the binary operator 'div' (line 421)
    result_div_412888 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 17), 'div', int_412886, beta_412887)
    
    # Getting the type of 'u' (line 421)
    u_412889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 27), 'u')
    # Applying the binary operator '*' (line 421)
    result_mul_412890 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 16), '*', result_div_412888, u_412889)
    
    # Assigning a type to the variable 'u' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'u', result_mul_412890)
    
    # Assigning a Call to a Name (line 422):
    
    # Assigning a Call to a Name (line 422):
    
    # Call to sqrt(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'anorm' (line 422)
    anorm_412892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 25), 'anorm', False)
    int_412893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 32), 'int')
    # Applying the binary operator '**' (line 422)
    result_pow_412894 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 25), '**', anorm_412892, int_412893)
    
    # Getting the type of 'alfa' (line 422)
    alfa_412895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 36), 'alfa', False)
    int_412896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 42), 'int')
    # Applying the binary operator '**' (line 422)
    result_pow_412897 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 36), '**', alfa_412895, int_412896)
    
    # Applying the binary operator '+' (line 422)
    result_add_412898 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 25), '+', result_pow_412894, result_pow_412897)
    
    # Getting the type of 'beta' (line 422)
    beta_412899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 46), 'beta', False)
    int_412900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 52), 'int')
    # Applying the binary operator '**' (line 422)
    result_pow_412901 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 46), '**', beta_412899, int_412900)
    
    # Applying the binary operator '+' (line 422)
    result_add_412902 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 44), '+', result_add_412898, result_pow_412901)
    
    # Getting the type of 'damp' (line 422)
    damp_412903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 56), 'damp', False)
    int_412904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 62), 'int')
    # Applying the binary operator '**' (line 422)
    result_pow_412905 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 56), '**', damp_412903, int_412904)
    
    # Applying the binary operator '+' (line 422)
    result_add_412906 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 54), '+', result_add_412902, result_pow_412905)
    
    # Processing the call keyword arguments (line 422)
    kwargs_412907 = {}
    # Getting the type of 'sqrt' (line 422)
    sqrt_412891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 20), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 422)
    sqrt_call_result_412908 = invoke(stypy.reporting.localization.Localization(__file__, 422, 20), sqrt_412891, *[result_add_412906], **kwargs_412907)
    
    # Assigning a type to the variable 'anorm' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'anorm', sqrt_call_result_412908)
    
    # Assigning a BinOp to a Name (line 423):
    
    # Assigning a BinOp to a Name (line 423):
    
    # Call to rmatvec(...): (line 423)
    # Processing the call arguments (line 423)
    # Getting the type of 'u' (line 423)
    u_412911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 26), 'u', False)
    # Processing the call keyword arguments (line 423)
    kwargs_412912 = {}
    # Getting the type of 'A' (line 423)
    A_412909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'A', False)
    # Obtaining the member 'rmatvec' of a type (line 423)
    rmatvec_412910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 16), A_412909, 'rmatvec')
    # Calling rmatvec(args, kwargs) (line 423)
    rmatvec_call_result_412913 = invoke(stypy.reporting.localization.Localization(__file__, 423, 16), rmatvec_412910, *[u_412911], **kwargs_412912)
    
    # Getting the type of 'beta' (line 423)
    beta_412914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 31), 'beta')
    # Getting the type of 'v' (line 423)
    v_412915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 38), 'v')
    # Applying the binary operator '*' (line 423)
    result_mul_412916 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 31), '*', beta_412914, v_412915)
    
    # Applying the binary operator '-' (line 423)
    result_sub_412917 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 16), '-', rmatvec_call_result_412913, result_mul_412916)
    
    # Assigning a type to the variable 'v' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'v', result_sub_412917)
    
    # Assigning a Call to a Name (line 424):
    
    # Assigning a Call to a Name (line 424):
    
    # Call to norm(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'v' (line 424)
    v_412921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 34), 'v', False)
    # Processing the call keyword arguments (line 424)
    kwargs_412922 = {}
    # Getting the type of 'np' (line 424)
    np_412918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 19), 'np', False)
    # Obtaining the member 'linalg' of a type (line 424)
    linalg_412919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 19), np_412918, 'linalg')
    # Obtaining the member 'norm' of a type (line 424)
    norm_412920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 19), linalg_412919, 'norm')
    # Calling norm(args, kwargs) (line 424)
    norm_call_result_412923 = invoke(stypy.reporting.localization.Localization(__file__, 424, 19), norm_412920, *[v_412921], **kwargs_412922)
    
    # Assigning a type to the variable 'alfa' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'alfa', norm_call_result_412923)
    
    
    # Getting the type of 'alfa' (line 425)
    alfa_412924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 15), 'alfa')
    int_412925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 22), 'int')
    # Applying the binary operator '>' (line 425)
    result_gt_412926 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 15), '>', alfa_412924, int_412925)
    
    # Testing the type of an if condition (line 425)
    if_condition_412927 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 12), result_gt_412926)
    # Assigning a type to the variable 'if_condition_412927' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'if_condition_412927', if_condition_412927)
    # SSA begins for if statement (line 425)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 426):
    
    # Assigning a BinOp to a Name (line 426):
    int_412928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 21), 'int')
    # Getting the type of 'alfa' (line 426)
    alfa_412929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 25), 'alfa')
    # Applying the binary operator 'div' (line 426)
    result_div_412930 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 21), 'div', int_412928, alfa_412929)
    
    # Getting the type of 'v' (line 426)
    v_412931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 33), 'v')
    # Applying the binary operator '*' (line 426)
    result_mul_412932 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 20), '*', result_div_412930, v_412931)
    
    # Assigning a type to the variable 'v' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 16), 'v', result_mul_412932)
    # SSA join for if statement (line 425)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 420)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 430):
    
    # Assigning a Call to a Name (line 430):
    
    # Call to sqrt(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'rhobar' (line 430)
    rhobar_412934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 23), 'rhobar', False)
    int_412935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 31), 'int')
    # Applying the binary operator '**' (line 430)
    result_pow_412936 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 23), '**', rhobar_412934, int_412935)
    
    # Getting the type of 'damp' (line 430)
    damp_412937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 35), 'damp', False)
    int_412938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 41), 'int')
    # Applying the binary operator '**' (line 430)
    result_pow_412939 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 35), '**', damp_412937, int_412938)
    
    # Applying the binary operator '+' (line 430)
    result_add_412940 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 23), '+', result_pow_412936, result_pow_412939)
    
    # Processing the call keyword arguments (line 430)
    kwargs_412941 = {}
    # Getting the type of 'sqrt' (line 430)
    sqrt_412933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 18), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 430)
    sqrt_call_result_412942 = invoke(stypy.reporting.localization.Localization(__file__, 430, 18), sqrt_412933, *[result_add_412940], **kwargs_412941)
    
    # Assigning a type to the variable 'rhobar1' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'rhobar1', sqrt_call_result_412942)
    
    # Assigning a BinOp to a Name (line 431):
    
    # Assigning a BinOp to a Name (line 431):
    # Getting the type of 'rhobar' (line 431)
    rhobar_412943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 14), 'rhobar')
    # Getting the type of 'rhobar1' (line 431)
    rhobar1_412944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 23), 'rhobar1')
    # Applying the binary operator 'div' (line 431)
    result_div_412945 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 14), 'div', rhobar_412943, rhobar1_412944)
    
    # Assigning a type to the variable 'cs1' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'cs1', result_div_412945)
    
    # Assigning a BinOp to a Name (line 432):
    
    # Assigning a BinOp to a Name (line 432):
    # Getting the type of 'damp' (line 432)
    damp_412946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 14), 'damp')
    # Getting the type of 'rhobar1' (line 432)
    rhobar1_412947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 21), 'rhobar1')
    # Applying the binary operator 'div' (line 432)
    result_div_412948 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 14), 'div', damp_412946, rhobar1_412947)
    
    # Assigning a type to the variable 'sn1' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'sn1', result_div_412948)
    
    # Assigning a BinOp to a Name (line 433):
    
    # Assigning a BinOp to a Name (line 433):
    # Getting the type of 'sn1' (line 433)
    sn1_412949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 14), 'sn1')
    # Getting the type of 'phibar' (line 433)
    phibar_412950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'phibar')
    # Applying the binary operator '*' (line 433)
    result_mul_412951 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 14), '*', sn1_412949, phibar_412950)
    
    # Assigning a type to the variable 'psi' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'psi', result_mul_412951)
    
    # Assigning a BinOp to a Name (line 434):
    
    # Assigning a BinOp to a Name (line 434):
    # Getting the type of 'cs1' (line 434)
    cs1_412952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 17), 'cs1')
    # Getting the type of 'phibar' (line 434)
    phibar_412953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 23), 'phibar')
    # Applying the binary operator '*' (line 434)
    result_mul_412954 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 17), '*', cs1_412952, phibar_412953)
    
    # Assigning a type to the variable 'phibar' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'phibar', result_mul_412954)
    
    # Assigning a Call to a Tuple (line 438):
    
    # Assigning a Subscript to a Name (line 438):
    
    # Obtaining the type of the subscript
    int_412955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 8), 'int')
    
    # Call to _sym_ortho(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'rhobar1' (line 438)
    rhobar1_412957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 33), 'rhobar1', False)
    # Getting the type of 'beta' (line 438)
    beta_412958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 42), 'beta', False)
    # Processing the call keyword arguments (line 438)
    kwargs_412959 = {}
    # Getting the type of '_sym_ortho' (line 438)
    _sym_ortho_412956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 22), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 438)
    _sym_ortho_call_result_412960 = invoke(stypy.reporting.localization.Localization(__file__, 438, 22), _sym_ortho_412956, *[rhobar1_412957, beta_412958], **kwargs_412959)
    
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___412961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), _sym_ortho_call_result_412960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_412962 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), getitem___412961, int_412955)
    
    # Assigning a type to the variable 'tuple_var_assignment_412475' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'tuple_var_assignment_412475', subscript_call_result_412962)
    
    # Assigning a Subscript to a Name (line 438):
    
    # Obtaining the type of the subscript
    int_412963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 8), 'int')
    
    # Call to _sym_ortho(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'rhobar1' (line 438)
    rhobar1_412965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 33), 'rhobar1', False)
    # Getting the type of 'beta' (line 438)
    beta_412966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 42), 'beta', False)
    # Processing the call keyword arguments (line 438)
    kwargs_412967 = {}
    # Getting the type of '_sym_ortho' (line 438)
    _sym_ortho_412964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 22), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 438)
    _sym_ortho_call_result_412968 = invoke(stypy.reporting.localization.Localization(__file__, 438, 22), _sym_ortho_412964, *[rhobar1_412965, beta_412966], **kwargs_412967)
    
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___412969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), _sym_ortho_call_result_412968, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_412970 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), getitem___412969, int_412963)
    
    # Assigning a type to the variable 'tuple_var_assignment_412476' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'tuple_var_assignment_412476', subscript_call_result_412970)
    
    # Assigning a Subscript to a Name (line 438):
    
    # Obtaining the type of the subscript
    int_412971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 8), 'int')
    
    # Call to _sym_ortho(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'rhobar1' (line 438)
    rhobar1_412973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 33), 'rhobar1', False)
    # Getting the type of 'beta' (line 438)
    beta_412974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 42), 'beta', False)
    # Processing the call keyword arguments (line 438)
    kwargs_412975 = {}
    # Getting the type of '_sym_ortho' (line 438)
    _sym_ortho_412972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 22), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 438)
    _sym_ortho_call_result_412976 = invoke(stypy.reporting.localization.Localization(__file__, 438, 22), _sym_ortho_412972, *[rhobar1_412973, beta_412974], **kwargs_412975)
    
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___412977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), _sym_ortho_call_result_412976, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_412978 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), getitem___412977, int_412971)
    
    # Assigning a type to the variable 'tuple_var_assignment_412477' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'tuple_var_assignment_412477', subscript_call_result_412978)
    
    # Assigning a Name to a Name (line 438):
    # Getting the type of 'tuple_var_assignment_412475' (line 438)
    tuple_var_assignment_412475_412979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'tuple_var_assignment_412475')
    # Assigning a type to the variable 'cs' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'cs', tuple_var_assignment_412475_412979)
    
    # Assigning a Name to a Name (line 438):
    # Getting the type of 'tuple_var_assignment_412476' (line 438)
    tuple_var_assignment_412476_412980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'tuple_var_assignment_412476')
    # Assigning a type to the variable 'sn' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'sn', tuple_var_assignment_412476_412980)
    
    # Assigning a Name to a Name (line 438):
    # Getting the type of 'tuple_var_assignment_412477' (line 438)
    tuple_var_assignment_412477_412981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'tuple_var_assignment_412477')
    # Assigning a type to the variable 'rho' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'rho', tuple_var_assignment_412477_412981)
    
    # Assigning a BinOp to a Name (line 440):
    
    # Assigning a BinOp to a Name (line 440):
    # Getting the type of 'sn' (line 440)
    sn_412982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'sn')
    # Getting the type of 'alfa' (line 440)
    alfa_412983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 21), 'alfa')
    # Applying the binary operator '*' (line 440)
    result_mul_412984 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 16), '*', sn_412982, alfa_412983)
    
    # Assigning a type to the variable 'theta' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'theta', result_mul_412984)
    
    # Assigning a BinOp to a Name (line 441):
    
    # Assigning a BinOp to a Name (line 441):
    
    # Getting the type of 'cs' (line 441)
    cs_412985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 18), 'cs')
    # Applying the 'usub' unary operator (line 441)
    result___neg___412986 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 17), 'usub', cs_412985)
    
    # Getting the type of 'alfa' (line 441)
    alfa_412987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 23), 'alfa')
    # Applying the binary operator '*' (line 441)
    result_mul_412988 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 17), '*', result___neg___412986, alfa_412987)
    
    # Assigning a type to the variable 'rhobar' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'rhobar', result_mul_412988)
    
    # Assigning a BinOp to a Name (line 442):
    
    # Assigning a BinOp to a Name (line 442):
    # Getting the type of 'cs' (line 442)
    cs_412989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 14), 'cs')
    # Getting the type of 'phibar' (line 442)
    phibar_412990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 19), 'phibar')
    # Applying the binary operator '*' (line 442)
    result_mul_412991 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 14), '*', cs_412989, phibar_412990)
    
    # Assigning a type to the variable 'phi' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'phi', result_mul_412991)
    
    # Assigning a BinOp to a Name (line 443):
    
    # Assigning a BinOp to a Name (line 443):
    # Getting the type of 'sn' (line 443)
    sn_412992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 17), 'sn')
    # Getting the type of 'phibar' (line 443)
    phibar_412993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 22), 'phibar')
    # Applying the binary operator '*' (line 443)
    result_mul_412994 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 17), '*', sn_412992, phibar_412993)
    
    # Assigning a type to the variable 'phibar' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'phibar', result_mul_412994)
    
    # Assigning a BinOp to a Name (line 444):
    
    # Assigning a BinOp to a Name (line 444):
    # Getting the type of 'sn' (line 444)
    sn_412995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 14), 'sn')
    # Getting the type of 'phi' (line 444)
    phi_412996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 19), 'phi')
    # Applying the binary operator '*' (line 444)
    result_mul_412997 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 14), '*', sn_412995, phi_412996)
    
    # Assigning a type to the variable 'tau' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'tau', result_mul_412997)
    
    # Assigning a BinOp to a Name (line 447):
    
    # Assigning a BinOp to a Name (line 447):
    # Getting the type of 'phi' (line 447)
    phi_412998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 13), 'phi')
    # Getting the type of 'rho' (line 447)
    rho_412999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 19), 'rho')
    # Applying the binary operator 'div' (line 447)
    result_div_413000 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 13), 'div', phi_412998, rho_412999)
    
    # Assigning a type to the variable 't1' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 't1', result_div_413000)
    
    # Assigning a BinOp to a Name (line 448):
    
    # Assigning a BinOp to a Name (line 448):
    
    # Getting the type of 'theta' (line 448)
    theta_413001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 14), 'theta')
    # Applying the 'usub' unary operator (line 448)
    result___neg___413002 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 13), 'usub', theta_413001)
    
    # Getting the type of 'rho' (line 448)
    rho_413003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 22), 'rho')
    # Applying the binary operator 'div' (line 448)
    result_div_413004 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 13), 'div', result___neg___413002, rho_413003)
    
    # Assigning a type to the variable 't2' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 't2', result_div_413004)
    
    # Assigning a BinOp to a Name (line 449):
    
    # Assigning a BinOp to a Name (line 449):
    int_413005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 14), 'int')
    # Getting the type of 'rho' (line 449)
    rho_413006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 18), 'rho')
    # Applying the binary operator 'div' (line 449)
    result_div_413007 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 14), 'div', int_413005, rho_413006)
    
    # Getting the type of 'w' (line 449)
    w_413008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 25), 'w')
    # Applying the binary operator '*' (line 449)
    result_mul_413009 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 13), '*', result_div_413007, w_413008)
    
    # Assigning a type to the variable 'dk' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'dk', result_mul_413009)
    
    # Assigning a BinOp to a Name (line 451):
    
    # Assigning a BinOp to a Name (line 451):
    # Getting the type of 'x' (line 451)
    x_413010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'x')
    # Getting the type of 't1' (line 451)
    t1_413011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 16), 't1')
    # Getting the type of 'w' (line 451)
    w_413012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 21), 'w')
    # Applying the binary operator '*' (line 451)
    result_mul_413013 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 16), '*', t1_413011, w_413012)
    
    # Applying the binary operator '+' (line 451)
    result_add_413014 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 12), '+', x_413010, result_mul_413013)
    
    # Assigning a type to the variable 'x' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'x', result_add_413014)
    
    # Assigning a BinOp to a Name (line 452):
    
    # Assigning a BinOp to a Name (line 452):
    # Getting the type of 'v' (line 452)
    v_413015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'v')
    # Getting the type of 't2' (line 452)
    t2_413016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 16), 't2')
    # Getting the type of 'w' (line 452)
    w_413017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 21), 'w')
    # Applying the binary operator '*' (line 452)
    result_mul_413018 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 16), '*', t2_413016, w_413017)
    
    # Applying the binary operator '+' (line 452)
    result_add_413019 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 12), '+', v_413015, result_mul_413018)
    
    # Assigning a type to the variable 'w' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'w', result_add_413019)
    
    # Assigning a BinOp to a Name (line 453):
    
    # Assigning a BinOp to a Name (line 453):
    # Getting the type of 'ddnorm' (line 453)
    ddnorm_413020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 17), 'ddnorm')
    
    # Call to norm(...): (line 453)
    # Processing the call arguments (line 453)
    # Getting the type of 'dk' (line 453)
    dk_413024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 41), 'dk', False)
    # Processing the call keyword arguments (line 453)
    kwargs_413025 = {}
    # Getting the type of 'np' (line 453)
    np_413021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 26), 'np', False)
    # Obtaining the member 'linalg' of a type (line 453)
    linalg_413022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 26), np_413021, 'linalg')
    # Obtaining the member 'norm' of a type (line 453)
    norm_413023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 26), linalg_413022, 'norm')
    # Calling norm(args, kwargs) (line 453)
    norm_call_result_413026 = invoke(stypy.reporting.localization.Localization(__file__, 453, 26), norm_413023, *[dk_413024], **kwargs_413025)
    
    int_413027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 46), 'int')
    # Applying the binary operator '**' (line 453)
    result_pow_413028 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 26), '**', norm_call_result_413026, int_413027)
    
    # Applying the binary operator '+' (line 453)
    result_add_413029 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 17), '+', ddnorm_413020, result_pow_413028)
    
    # Assigning a type to the variable 'ddnorm' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'ddnorm', result_add_413029)
    
    # Getting the type of 'calc_var' (line 455)
    calc_var_413030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 11), 'calc_var')
    # Testing the type of an if condition (line 455)
    if_condition_413031 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 8), calc_var_413030)
    # Assigning a type to the variable 'if_condition_413031' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'if_condition_413031', if_condition_413031)
    # SSA begins for if statement (line 455)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 456):
    
    # Assigning a BinOp to a Name (line 456):
    # Getting the type of 'var' (line 456)
    var_413032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 18), 'var')
    # Getting the type of 'dk' (line 456)
    dk_413033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 24), 'dk')
    int_413034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 28), 'int')
    # Applying the binary operator '**' (line 456)
    result_pow_413035 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 24), '**', dk_413033, int_413034)
    
    # Applying the binary operator '+' (line 456)
    result_add_413036 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 18), '+', var_413032, result_pow_413035)
    
    # Assigning a type to the variable 'var' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'var', result_add_413036)
    # SSA join for if statement (line 455)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 461):
    
    # Assigning a BinOp to a Name (line 461):
    # Getting the type of 'sn2' (line 461)
    sn2_413037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 16), 'sn2')
    # Getting the type of 'rho' (line 461)
    rho_413038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 22), 'rho')
    # Applying the binary operator '*' (line 461)
    result_mul_413039 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 16), '*', sn2_413037, rho_413038)
    
    # Assigning a type to the variable 'delta' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'delta', result_mul_413039)
    
    # Assigning a BinOp to a Name (line 462):
    
    # Assigning a BinOp to a Name (line 462):
    
    # Getting the type of 'cs2' (line 462)
    cs2_413040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 18), 'cs2')
    # Applying the 'usub' unary operator (line 462)
    result___neg___413041 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 17), 'usub', cs2_413040)
    
    # Getting the type of 'rho' (line 462)
    rho_413042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 24), 'rho')
    # Applying the binary operator '*' (line 462)
    result_mul_413043 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 17), '*', result___neg___413041, rho_413042)
    
    # Assigning a type to the variable 'gambar' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'gambar', result_mul_413043)
    
    # Assigning a BinOp to a Name (line 463):
    
    # Assigning a BinOp to a Name (line 463):
    # Getting the type of 'phi' (line 463)
    phi_413044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 14), 'phi')
    # Getting the type of 'delta' (line 463)
    delta_413045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 20), 'delta')
    # Getting the type of 'z' (line 463)
    z_413046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 28), 'z')
    # Applying the binary operator '*' (line 463)
    result_mul_413047 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 20), '*', delta_413045, z_413046)
    
    # Applying the binary operator '-' (line 463)
    result_sub_413048 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 14), '-', phi_413044, result_mul_413047)
    
    # Assigning a type to the variable 'rhs' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'rhs', result_sub_413048)
    
    # Assigning a BinOp to a Name (line 464):
    
    # Assigning a BinOp to a Name (line 464):
    # Getting the type of 'rhs' (line 464)
    rhs_413049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 15), 'rhs')
    # Getting the type of 'gambar' (line 464)
    gambar_413050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 21), 'gambar')
    # Applying the binary operator 'div' (line 464)
    result_div_413051 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 15), 'div', rhs_413049, gambar_413050)
    
    # Assigning a type to the variable 'zbar' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'zbar', result_div_413051)
    
    # Assigning a Call to a Name (line 465):
    
    # Assigning a Call to a Name (line 465):
    
    # Call to sqrt(...): (line 465)
    # Processing the call arguments (line 465)
    # Getting the type of 'xxnorm' (line 465)
    xxnorm_413053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 21), 'xxnorm', False)
    # Getting the type of 'zbar' (line 465)
    zbar_413054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 30), 'zbar', False)
    int_413055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 36), 'int')
    # Applying the binary operator '**' (line 465)
    result_pow_413056 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 30), '**', zbar_413054, int_413055)
    
    # Applying the binary operator '+' (line 465)
    result_add_413057 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 21), '+', xxnorm_413053, result_pow_413056)
    
    # Processing the call keyword arguments (line 465)
    kwargs_413058 = {}
    # Getting the type of 'sqrt' (line 465)
    sqrt_413052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 16), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 465)
    sqrt_call_result_413059 = invoke(stypy.reporting.localization.Localization(__file__, 465, 16), sqrt_413052, *[result_add_413057], **kwargs_413058)
    
    # Assigning a type to the variable 'xnorm' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'xnorm', sqrt_call_result_413059)
    
    # Assigning a Call to a Name (line 466):
    
    # Assigning a Call to a Name (line 466):
    
    # Call to sqrt(...): (line 466)
    # Processing the call arguments (line 466)
    # Getting the type of 'gambar' (line 466)
    gambar_413061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 21), 'gambar', False)
    int_413062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 29), 'int')
    # Applying the binary operator '**' (line 466)
    result_pow_413063 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 21), '**', gambar_413061, int_413062)
    
    # Getting the type of 'theta' (line 466)
    theta_413064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 33), 'theta', False)
    int_413065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 40), 'int')
    # Applying the binary operator '**' (line 466)
    result_pow_413066 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 33), '**', theta_413064, int_413065)
    
    # Applying the binary operator '+' (line 466)
    result_add_413067 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 21), '+', result_pow_413063, result_pow_413066)
    
    # Processing the call keyword arguments (line 466)
    kwargs_413068 = {}
    # Getting the type of 'sqrt' (line 466)
    sqrt_413060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 16), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 466)
    sqrt_call_result_413069 = invoke(stypy.reporting.localization.Localization(__file__, 466, 16), sqrt_413060, *[result_add_413067], **kwargs_413068)
    
    # Assigning a type to the variable 'gamma' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'gamma', sqrt_call_result_413069)
    
    # Assigning a BinOp to a Name (line 467):
    
    # Assigning a BinOp to a Name (line 467):
    # Getting the type of 'gambar' (line 467)
    gambar_413070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 14), 'gambar')
    # Getting the type of 'gamma' (line 467)
    gamma_413071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 23), 'gamma')
    # Applying the binary operator 'div' (line 467)
    result_div_413072 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 14), 'div', gambar_413070, gamma_413071)
    
    # Assigning a type to the variable 'cs2' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'cs2', result_div_413072)
    
    # Assigning a BinOp to a Name (line 468):
    
    # Assigning a BinOp to a Name (line 468):
    # Getting the type of 'theta' (line 468)
    theta_413073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 14), 'theta')
    # Getting the type of 'gamma' (line 468)
    gamma_413074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 22), 'gamma')
    # Applying the binary operator 'div' (line 468)
    result_div_413075 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 14), 'div', theta_413073, gamma_413074)
    
    # Assigning a type to the variable 'sn2' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'sn2', result_div_413075)
    
    # Assigning a BinOp to a Name (line 469):
    
    # Assigning a BinOp to a Name (line 469):
    # Getting the type of 'rhs' (line 469)
    rhs_413076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'rhs')
    # Getting the type of 'gamma' (line 469)
    gamma_413077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 18), 'gamma')
    # Applying the binary operator 'div' (line 469)
    result_div_413078 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 12), 'div', rhs_413076, gamma_413077)
    
    # Assigning a type to the variable 'z' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'z', result_div_413078)
    
    # Assigning a BinOp to a Name (line 470):
    
    # Assigning a BinOp to a Name (line 470):
    # Getting the type of 'xxnorm' (line 470)
    xxnorm_413079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 17), 'xxnorm')
    # Getting the type of 'z' (line 470)
    z_413080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 26), 'z')
    int_413081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 29), 'int')
    # Applying the binary operator '**' (line 470)
    result_pow_413082 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 26), '**', z_413080, int_413081)
    
    # Applying the binary operator '+' (line 470)
    result_add_413083 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 17), '+', xxnorm_413079, result_pow_413082)
    
    # Assigning a type to the variable 'xxnorm' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'xxnorm', result_add_413083)
    
    # Assigning a BinOp to a Name (line 475):
    
    # Assigning a BinOp to a Name (line 475):
    # Getting the type of 'anorm' (line 475)
    anorm_413084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 16), 'anorm')
    
    # Call to sqrt(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'ddnorm' (line 475)
    ddnorm_413086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 29), 'ddnorm', False)
    # Processing the call keyword arguments (line 475)
    kwargs_413087 = {}
    # Getting the type of 'sqrt' (line 475)
    sqrt_413085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 24), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 475)
    sqrt_call_result_413088 = invoke(stypy.reporting.localization.Localization(__file__, 475, 24), sqrt_413085, *[ddnorm_413086], **kwargs_413087)
    
    # Applying the binary operator '*' (line 475)
    result_mul_413089 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 16), '*', anorm_413084, sqrt_call_result_413088)
    
    # Assigning a type to the variable 'acond' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'acond', result_mul_413089)
    
    # Assigning a BinOp to a Name (line 476):
    
    # Assigning a BinOp to a Name (line 476):
    # Getting the type of 'phibar' (line 476)
    phibar_413090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 15), 'phibar')
    int_413091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 23), 'int')
    # Applying the binary operator '**' (line 476)
    result_pow_413092 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 15), '**', phibar_413090, int_413091)
    
    # Assigning a type to the variable 'res1' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'res1', result_pow_413092)
    
    # Assigning a BinOp to a Name (line 477):
    
    # Assigning a BinOp to a Name (line 477):
    # Getting the type of 'res2' (line 477)
    res2_413093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 15), 'res2')
    # Getting the type of 'psi' (line 477)
    psi_413094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 22), 'psi')
    int_413095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 27), 'int')
    # Applying the binary operator '**' (line 477)
    result_pow_413096 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 22), '**', psi_413094, int_413095)
    
    # Applying the binary operator '+' (line 477)
    result_add_413097 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 15), '+', res2_413093, result_pow_413096)
    
    # Assigning a type to the variable 'res2' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'res2', result_add_413097)
    
    # Assigning a Call to a Name (line 478):
    
    # Assigning a Call to a Name (line 478):
    
    # Call to sqrt(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'res1' (line 478)
    res1_413099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 21), 'res1', False)
    # Getting the type of 'res2' (line 478)
    res2_413100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 28), 'res2', False)
    # Applying the binary operator '+' (line 478)
    result_add_413101 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 21), '+', res1_413099, res2_413100)
    
    # Processing the call keyword arguments (line 478)
    kwargs_413102 = {}
    # Getting the type of 'sqrt' (line 478)
    sqrt_413098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 16), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 478)
    sqrt_call_result_413103 = invoke(stypy.reporting.localization.Localization(__file__, 478, 16), sqrt_413098, *[result_add_413101], **kwargs_413102)
    
    # Assigning a type to the variable 'rnorm' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'rnorm', sqrt_call_result_413103)
    
    # Assigning a BinOp to a Name (line 479):
    
    # Assigning a BinOp to a Name (line 479):
    # Getting the type of 'alfa' (line 479)
    alfa_413104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 17), 'alfa')
    
    # Call to abs(...): (line 479)
    # Processing the call arguments (line 479)
    # Getting the type of 'tau' (line 479)
    tau_413106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 28), 'tau', False)
    # Processing the call keyword arguments (line 479)
    kwargs_413107 = {}
    # Getting the type of 'abs' (line 479)
    abs_413105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 24), 'abs', False)
    # Calling abs(args, kwargs) (line 479)
    abs_call_result_413108 = invoke(stypy.reporting.localization.Localization(__file__, 479, 24), abs_413105, *[tau_413106], **kwargs_413107)
    
    # Applying the binary operator '*' (line 479)
    result_mul_413109 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 17), '*', alfa_413104, abs_call_result_413108)
    
    # Assigning a type to the variable 'arnorm' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'arnorm', result_mul_413109)
    
    # Assigning a BinOp to a Name (line 488):
    
    # Assigning a BinOp to a Name (line 488):
    # Getting the type of 'rnorm' (line 488)
    rnorm_413110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 15), 'rnorm')
    int_413111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 22), 'int')
    # Applying the binary operator '**' (line 488)
    result_pow_413112 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 15), '**', rnorm_413110, int_413111)
    
    # Getting the type of 'dampsq' (line 488)
    dampsq_413113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 26), 'dampsq')
    # Getting the type of 'xxnorm' (line 488)
    xxnorm_413114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 35), 'xxnorm')
    # Applying the binary operator '*' (line 488)
    result_mul_413115 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 26), '*', dampsq_413113, xxnorm_413114)
    
    # Applying the binary operator '-' (line 488)
    result_sub_413116 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 15), '-', result_pow_413112, result_mul_413115)
    
    # Assigning a type to the variable 'r1sq' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'r1sq', result_sub_413116)
    
    # Assigning a Call to a Name (line 489):
    
    # Assigning a Call to a Name (line 489):
    
    # Call to sqrt(...): (line 489)
    # Processing the call arguments (line 489)
    
    # Call to abs(...): (line 489)
    # Processing the call arguments (line 489)
    # Getting the type of 'r1sq' (line 489)
    r1sq_413119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 26), 'r1sq', False)
    # Processing the call keyword arguments (line 489)
    kwargs_413120 = {}
    # Getting the type of 'abs' (line 489)
    abs_413118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 22), 'abs', False)
    # Calling abs(args, kwargs) (line 489)
    abs_call_result_413121 = invoke(stypy.reporting.localization.Localization(__file__, 489, 22), abs_413118, *[r1sq_413119], **kwargs_413120)
    
    # Processing the call keyword arguments (line 489)
    kwargs_413122 = {}
    # Getting the type of 'sqrt' (line 489)
    sqrt_413117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 17), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 489)
    sqrt_call_result_413123 = invoke(stypy.reporting.localization.Localization(__file__, 489, 17), sqrt_413117, *[abs_call_result_413121], **kwargs_413122)
    
    # Assigning a type to the variable 'r1norm' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'r1norm', sqrt_call_result_413123)
    
    
    # Getting the type of 'r1sq' (line 490)
    r1sq_413124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 11), 'r1sq')
    int_413125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 18), 'int')
    # Applying the binary operator '<' (line 490)
    result_lt_413126 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 11), '<', r1sq_413124, int_413125)
    
    # Testing the type of an if condition (line 490)
    if_condition_413127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 490, 8), result_lt_413126)
    # Assigning a type to the variable 'if_condition_413127' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'if_condition_413127', if_condition_413127)
    # SSA begins for if statement (line 490)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a UnaryOp to a Name (line 491):
    
    # Assigning a UnaryOp to a Name (line 491):
    
    # Getting the type of 'r1norm' (line 491)
    r1norm_413128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 22), 'r1norm')
    # Applying the 'usub' unary operator (line 491)
    result___neg___413129 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 21), 'usub', r1norm_413128)
    
    # Assigning a type to the variable 'r1norm' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'r1norm', result___neg___413129)
    # SSA join for if statement (line 490)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 492):
    
    # Assigning a Name to a Name (line 492):
    # Getting the type of 'rnorm' (line 492)
    rnorm_413130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 17), 'rnorm')
    # Assigning a type to the variable 'r2norm' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'r2norm', rnorm_413130)
    
    # Assigning a BinOp to a Name (line 496):
    
    # Assigning a BinOp to a Name (line 496):
    # Getting the type of 'rnorm' (line 496)
    rnorm_413131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 16), 'rnorm')
    # Getting the type of 'bnorm' (line 496)
    bnorm_413132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 24), 'bnorm')
    # Applying the binary operator 'div' (line 496)
    result_div_413133 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 16), 'div', rnorm_413131, bnorm_413132)
    
    # Assigning a type to the variable 'test1' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'test1', result_div_413133)
    
    # Assigning a BinOp to a Name (line 497):
    
    # Assigning a BinOp to a Name (line 497):
    # Getting the type of 'arnorm' (line 497)
    arnorm_413134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 16), 'arnorm')
    # Getting the type of 'anorm' (line 497)
    anorm_413135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 26), 'anorm')
    # Getting the type of 'rnorm' (line 497)
    rnorm_413136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 34), 'rnorm')
    # Applying the binary operator '*' (line 497)
    result_mul_413137 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 26), '*', anorm_413135, rnorm_413136)
    
    # Getting the type of 'eps' (line 497)
    eps_413138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 42), 'eps')
    # Applying the binary operator '+' (line 497)
    result_add_413139 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 26), '+', result_mul_413137, eps_413138)
    
    # Applying the binary operator 'div' (line 497)
    result_div_413140 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 16), 'div', arnorm_413134, result_add_413139)
    
    # Assigning a type to the variable 'test2' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'test2', result_div_413140)
    
    # Assigning a BinOp to a Name (line 498):
    
    # Assigning a BinOp to a Name (line 498):
    int_413141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 16), 'int')
    # Getting the type of 'acond' (line 498)
    acond_413142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 21), 'acond')
    # Getting the type of 'eps' (line 498)
    eps_413143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 29), 'eps')
    # Applying the binary operator '+' (line 498)
    result_add_413144 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 21), '+', acond_413142, eps_413143)
    
    # Applying the binary operator 'div' (line 498)
    result_div_413145 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 16), 'div', int_413141, result_add_413144)
    
    # Assigning a type to the variable 'test3' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'test3', result_div_413145)
    
    # Assigning a BinOp to a Name (line 499):
    
    # Assigning a BinOp to a Name (line 499):
    # Getting the type of 'test1' (line 499)
    test1_413146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 13), 'test1')
    int_413147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 22), 'int')
    # Getting the type of 'anorm' (line 499)
    anorm_413148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 26), 'anorm')
    # Getting the type of 'xnorm' (line 499)
    xnorm_413149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 34), 'xnorm')
    # Applying the binary operator '*' (line 499)
    result_mul_413150 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 26), '*', anorm_413148, xnorm_413149)
    
    # Getting the type of 'bnorm' (line 499)
    bnorm_413151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 42), 'bnorm')
    # Applying the binary operator 'div' (line 499)
    result_div_413152 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 40), 'div', result_mul_413150, bnorm_413151)
    
    # Applying the binary operator '+' (line 499)
    result_add_413153 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 22), '+', int_413147, result_div_413152)
    
    # Applying the binary operator 'div' (line 499)
    result_div_413154 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 13), 'div', test1_413146, result_add_413153)
    
    # Assigning a type to the variable 't1' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 't1', result_div_413154)
    
    # Assigning a BinOp to a Name (line 500):
    
    # Assigning a BinOp to a Name (line 500):
    # Getting the type of 'btol' (line 500)
    btol_413155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), 'btol')
    # Getting the type of 'atol' (line 500)
    atol_413156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 22), 'atol')
    # Getting the type of 'anorm' (line 500)
    anorm_413157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 29), 'anorm')
    # Applying the binary operator '*' (line 500)
    result_mul_413158 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 22), '*', atol_413156, anorm_413157)
    
    # Getting the type of 'xnorm' (line 500)
    xnorm_413159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 37), 'xnorm')
    # Applying the binary operator '*' (line 500)
    result_mul_413160 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 35), '*', result_mul_413158, xnorm_413159)
    
    # Getting the type of 'bnorm' (line 500)
    bnorm_413161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 45), 'bnorm')
    # Applying the binary operator 'div' (line 500)
    result_div_413162 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 43), 'div', result_mul_413160, bnorm_413161)
    
    # Applying the binary operator '+' (line 500)
    result_add_413163 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 15), '+', btol_413155, result_div_413162)
    
    # Assigning a type to the variable 'rtol' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'rtol', result_add_413163)
    
    
    # Getting the type of 'itn' (line 507)
    itn_413164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 11), 'itn')
    # Getting the type of 'iter_lim' (line 507)
    iter_lim_413165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 18), 'iter_lim')
    # Applying the binary operator '>=' (line 507)
    result_ge_413166 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 11), '>=', itn_413164, iter_lim_413165)
    
    # Testing the type of an if condition (line 507)
    if_condition_413167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 507, 8), result_ge_413166)
    # Assigning a type to the variable 'if_condition_413167' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'if_condition_413167', if_condition_413167)
    # SSA begins for if statement (line 507)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 508):
    
    # Assigning a Num to a Name (line 508):
    int_413168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 20), 'int')
    # Assigning a type to the variable 'istop' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'istop', int_413168)
    # SSA join for if statement (line 507)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    int_413169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 11), 'int')
    # Getting the type of 'test3' (line 509)
    test3_413170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 15), 'test3')
    # Applying the binary operator '+' (line 509)
    result_add_413171 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 11), '+', int_413169, test3_413170)
    
    int_413172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 24), 'int')
    # Applying the binary operator '<=' (line 509)
    result_le_413173 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 11), '<=', result_add_413171, int_413172)
    
    # Testing the type of an if condition (line 509)
    if_condition_413174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 509, 8), result_le_413173)
    # Assigning a type to the variable 'if_condition_413174' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'if_condition_413174', if_condition_413174)
    # SSA begins for if statement (line 509)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 510):
    
    # Assigning a Num to a Name (line 510):
    int_413175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 20), 'int')
    # Assigning a type to the variable 'istop' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'istop', int_413175)
    # SSA join for if statement (line 509)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    int_413176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 11), 'int')
    # Getting the type of 'test2' (line 511)
    test2_413177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 15), 'test2')
    # Applying the binary operator '+' (line 511)
    result_add_413178 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 11), '+', int_413176, test2_413177)
    
    int_413179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 24), 'int')
    # Applying the binary operator '<=' (line 511)
    result_le_413180 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 11), '<=', result_add_413178, int_413179)
    
    # Testing the type of an if condition (line 511)
    if_condition_413181 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 511, 8), result_le_413180)
    # Assigning a type to the variable 'if_condition_413181' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'if_condition_413181', if_condition_413181)
    # SSA begins for if statement (line 511)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 512):
    
    # Assigning a Num to a Name (line 512):
    int_413182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 20), 'int')
    # Assigning a type to the variable 'istop' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'istop', int_413182)
    # SSA join for if statement (line 511)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    int_413183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 11), 'int')
    # Getting the type of 't1' (line 513)
    t1_413184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 15), 't1')
    # Applying the binary operator '+' (line 513)
    result_add_413185 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 11), '+', int_413183, t1_413184)
    
    int_413186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 21), 'int')
    # Applying the binary operator '<=' (line 513)
    result_le_413187 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 11), '<=', result_add_413185, int_413186)
    
    # Testing the type of an if condition (line 513)
    if_condition_413188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 513, 8), result_le_413187)
    # Assigning a type to the variable 'if_condition_413188' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'if_condition_413188', if_condition_413188)
    # SSA begins for if statement (line 513)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 514):
    
    # Assigning a Num to a Name (line 514):
    int_413189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 20), 'int')
    # Assigning a type to the variable 'istop' (line 514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 12), 'istop', int_413189)
    # SSA join for if statement (line 513)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'test3' (line 517)
    test3_413190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 11), 'test3')
    # Getting the type of 'ctol' (line 517)
    ctol_413191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 20), 'ctol')
    # Applying the binary operator '<=' (line 517)
    result_le_413192 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 11), '<=', test3_413190, ctol_413191)
    
    # Testing the type of an if condition (line 517)
    if_condition_413193 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 8), result_le_413192)
    # Assigning a type to the variable 'if_condition_413193' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'if_condition_413193', if_condition_413193)
    # SSA begins for if statement (line 517)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 518):
    
    # Assigning a Num to a Name (line 518):
    int_413194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 20), 'int')
    # Assigning a type to the variable 'istop' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'istop', int_413194)
    # SSA join for if statement (line 517)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'test2' (line 519)
    test2_413195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 11), 'test2')
    # Getting the type of 'atol' (line 519)
    atol_413196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 20), 'atol')
    # Applying the binary operator '<=' (line 519)
    result_le_413197 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 11), '<=', test2_413195, atol_413196)
    
    # Testing the type of an if condition (line 519)
    if_condition_413198 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 519, 8), result_le_413197)
    # Assigning a type to the variable 'if_condition_413198' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'if_condition_413198', if_condition_413198)
    # SSA begins for if statement (line 519)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 520):
    
    # Assigning a Num to a Name (line 520):
    int_413199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 20), 'int')
    # Assigning a type to the variable 'istop' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'istop', int_413199)
    # SSA join for if statement (line 519)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'test1' (line 521)
    test1_413200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 11), 'test1')
    # Getting the type of 'rtol' (line 521)
    rtol_413201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 20), 'rtol')
    # Applying the binary operator '<=' (line 521)
    result_le_413202 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 11), '<=', test1_413200, rtol_413201)
    
    # Testing the type of an if condition (line 521)
    if_condition_413203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 521, 8), result_le_413202)
    # Assigning a type to the variable 'if_condition_413203' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'if_condition_413203', if_condition_413203)
    # SSA begins for if statement (line 521)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 522):
    
    # Assigning a Num to a Name (line 522):
    int_413204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 20), 'int')
    # Assigning a type to the variable 'istop' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'istop', int_413204)
    # SSA join for if statement (line 521)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 525):
    
    # Assigning a Name to a Name (line 525):
    # Getting the type of 'False' (line 525)
    False_413205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 15), 'False')
    # Assigning a type to the variable 'prnt' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'prnt', False_413205)
    
    
    # Getting the type of 'n' (line 526)
    n_413206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 11), 'n')
    int_413207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 16), 'int')
    # Applying the binary operator '<=' (line 526)
    result_le_413208 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 11), '<=', n_413206, int_413207)
    
    # Testing the type of an if condition (line 526)
    if_condition_413209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 526, 8), result_le_413208)
    # Assigning a type to the variable 'if_condition_413209' (line 526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'if_condition_413209', if_condition_413209)
    # SSA begins for if statement (line 526)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 527):
    
    # Assigning a Name to a Name (line 527):
    # Getting the type of 'True' (line 527)
    True_413210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'prnt', True_413210)
    # SSA join for if statement (line 526)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'itn' (line 528)
    itn_413211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 11), 'itn')
    int_413212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 18), 'int')
    # Applying the binary operator '<=' (line 528)
    result_le_413213 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 11), '<=', itn_413211, int_413212)
    
    # Testing the type of an if condition (line 528)
    if_condition_413214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 528, 8), result_le_413213)
    # Assigning a type to the variable 'if_condition_413214' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'if_condition_413214', if_condition_413214)
    # SSA begins for if statement (line 528)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 529):
    
    # Assigning a Name to a Name (line 529):
    # Getting the type of 'True' (line 529)
    True_413215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'prnt', True_413215)
    # SSA join for if statement (line 528)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'itn' (line 530)
    itn_413216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 11), 'itn')
    # Getting the type of 'iter_lim' (line 530)
    iter_lim_413217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 18), 'iter_lim')
    int_413218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 27), 'int')
    # Applying the binary operator '-' (line 530)
    result_sub_413219 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 18), '-', iter_lim_413217, int_413218)
    
    # Applying the binary operator '>=' (line 530)
    result_ge_413220 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 11), '>=', itn_413216, result_sub_413219)
    
    # Testing the type of an if condition (line 530)
    if_condition_413221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 530, 8), result_ge_413220)
    # Assigning a type to the variable 'if_condition_413221' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'if_condition_413221', if_condition_413221)
    # SSA begins for if statement (line 530)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 531):
    
    # Assigning a Name to a Name (line 531):
    # Getting the type of 'True' (line 531)
    True_413222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 531)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'prnt', True_413222)
    # SSA join for if statement (line 530)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'test3' (line 533)
    test3_413223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 11), 'test3')
    int_413224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 20), 'int')
    # Getting the type of 'ctol' (line 533)
    ctol_413225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 22), 'ctol')
    # Applying the binary operator '*' (line 533)
    result_mul_413226 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 20), '*', int_413224, ctol_413225)
    
    # Applying the binary operator '<=' (line 533)
    result_le_413227 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 11), '<=', test3_413223, result_mul_413226)
    
    # Testing the type of an if condition (line 533)
    if_condition_413228 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 533, 8), result_le_413227)
    # Assigning a type to the variable 'if_condition_413228' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'if_condition_413228', if_condition_413228)
    # SSA begins for if statement (line 533)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 534):
    
    # Assigning a Name to a Name (line 534):
    # Getting the type of 'True' (line 534)
    True_413229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'prnt', True_413229)
    # SSA join for if statement (line 533)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'test2' (line 535)
    test2_413230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 11), 'test2')
    int_413231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 20), 'int')
    # Getting the type of 'atol' (line 535)
    atol_413232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 23), 'atol')
    # Applying the binary operator '*' (line 535)
    result_mul_413233 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 20), '*', int_413231, atol_413232)
    
    # Applying the binary operator '<=' (line 535)
    result_le_413234 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 11), '<=', test2_413230, result_mul_413233)
    
    # Testing the type of an if condition (line 535)
    if_condition_413235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 535, 8), result_le_413234)
    # Assigning a type to the variable 'if_condition_413235' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'if_condition_413235', if_condition_413235)
    # SSA begins for if statement (line 535)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 536):
    
    # Assigning a Name to a Name (line 536):
    # Getting the type of 'True' (line 536)
    True_413236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 12), 'prnt', True_413236)
    # SSA join for if statement (line 535)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'test1' (line 537)
    test1_413237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 11), 'test1')
    int_413238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 20), 'int')
    # Getting the type of 'rtol' (line 537)
    rtol_413239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 23), 'rtol')
    # Applying the binary operator '*' (line 537)
    result_mul_413240 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 20), '*', int_413238, rtol_413239)
    
    # Applying the binary operator '<=' (line 537)
    result_le_413241 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 11), '<=', test1_413237, result_mul_413240)
    
    # Testing the type of an if condition (line 537)
    if_condition_413242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 537, 8), result_le_413241)
    # Assigning a type to the variable 'if_condition_413242' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'if_condition_413242', if_condition_413242)
    # SSA begins for if statement (line 537)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 538):
    
    # Assigning a Name to a Name (line 538):
    # Getting the type of 'True' (line 538)
    True_413243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'prnt', True_413243)
    # SSA join for if statement (line 537)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'istop' (line 539)
    istop_413244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 11), 'istop')
    int_413245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 20), 'int')
    # Applying the binary operator '!=' (line 539)
    result_ne_413246 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 11), '!=', istop_413244, int_413245)
    
    # Testing the type of an if condition (line 539)
    if_condition_413247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 539, 8), result_ne_413246)
    # Assigning a type to the variable 'if_condition_413247' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'if_condition_413247', if_condition_413247)
    # SSA begins for if statement (line 539)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 540):
    
    # Assigning a Name to a Name (line 540):
    # Getting the type of 'True' (line 540)
    True_413248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 12), 'prnt', True_413248)
    # SSA join for if statement (line 539)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'prnt' (line 542)
    prnt_413249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 11), 'prnt')
    # Testing the type of an if condition (line 542)
    if_condition_413250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 542, 8), prnt_413249)
    # Assigning a type to the variable 'if_condition_413250' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'if_condition_413250', if_condition_413250)
    # SSA begins for if statement (line 542)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'show' (line 543)
    show_413251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 15), 'show')
    # Testing the type of an if condition (line 543)
    if_condition_413252 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 543, 12), show_413251)
    # Assigning a type to the variable 'if_condition_413252' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'if_condition_413252', if_condition_413252)
    # SSA begins for if statement (line 543)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 544):
    
    # Assigning a BinOp to a Name (line 544):
    str_413253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 23), 'str', '%6g %12.5e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 544)
    tuple_413254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 544)
    # Adding element type (line 544)
    # Getting the type of 'itn' (line 544)
    itn_413255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 39), 'itn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 39), tuple_413254, itn_413255)
    # Adding element type (line 544)
    
    # Obtaining the type of the subscript
    int_413256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 46), 'int')
    # Getting the type of 'x' (line 544)
    x_413257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 44), 'x')
    # Obtaining the member '__getitem__' of a type (line 544)
    getitem___413258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 44), x_413257, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 544)
    subscript_call_result_413259 = invoke(stypy.reporting.localization.Localization(__file__, 544, 44), getitem___413258, int_413256)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 39), tuple_413254, subscript_call_result_413259)
    
    # Applying the binary operator '%' (line 544)
    result_mod_413260 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 23), '%', str_413253, tuple_413254)
    
    # Assigning a type to the variable 'str1' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'str1', result_mod_413260)
    
    # Assigning a BinOp to a Name (line 545):
    
    # Assigning a BinOp to a Name (line 545):
    str_413261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 23), 'str', ' %10.3e %10.3e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 545)
    tuple_413262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 545)
    # Adding element type (line 545)
    # Getting the type of 'r1norm' (line 545)
    r1norm_413263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 43), 'r1norm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 43), tuple_413262, r1norm_413263)
    # Adding element type (line 545)
    # Getting the type of 'r2norm' (line 545)
    r2norm_413264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 51), 'r2norm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 43), tuple_413262, r2norm_413264)
    
    # Applying the binary operator '%' (line 545)
    result_mod_413265 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 23), '%', str_413261, tuple_413262)
    
    # Assigning a type to the variable 'str2' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'str2', result_mod_413265)
    
    # Assigning a BinOp to a Name (line 546):
    
    # Assigning a BinOp to a Name (line 546):
    str_413266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 23), 'str', '  %8.1e %8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 546)
    tuple_413267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 546)
    # Adding element type (line 546)
    # Getting the type of 'test1' (line 546)
    test1_413268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 42), 'test1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 42), tuple_413267, test1_413268)
    # Adding element type (line 546)
    # Getting the type of 'test2' (line 546)
    test2_413269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 49), 'test2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 42), tuple_413267, test2_413269)
    
    # Applying the binary operator '%' (line 546)
    result_mod_413270 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 23), '%', str_413266, tuple_413267)
    
    # Assigning a type to the variable 'str3' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'str3', result_mod_413270)
    
    # Assigning a BinOp to a Name (line 547):
    
    # Assigning a BinOp to a Name (line 547):
    str_413271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 23), 'str', ' %8.1e %8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 547)
    tuple_413272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 547)
    # Adding element type (line 547)
    # Getting the type of 'anorm' (line 547)
    anorm_413273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 41), 'anorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 41), tuple_413272, anorm_413273)
    # Adding element type (line 547)
    # Getting the type of 'acond' (line 547)
    acond_413274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 48), 'acond')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 41), tuple_413272, acond_413274)
    
    # Applying the binary operator '%' (line 547)
    result_mod_413275 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 23), '%', str_413271, tuple_413272)
    
    # Assigning a type to the variable 'str4' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 16), 'str4', result_mod_413275)
    
    # Call to print(...): (line 548)
    # Processing the call arguments (line 548)
    # Getting the type of 'str1' (line 548)
    str1_413277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 22), 'str1', False)
    # Getting the type of 'str2' (line 548)
    str2_413278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 28), 'str2', False)
    # Getting the type of 'str3' (line 548)
    str3_413279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 34), 'str3', False)
    # Getting the type of 'str4' (line 548)
    str4_413280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 40), 'str4', False)
    # Processing the call keyword arguments (line 548)
    kwargs_413281 = {}
    # Getting the type of 'print' (line 548)
    print_413276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 16), 'print', False)
    # Calling print(args, kwargs) (line 548)
    print_call_result_413282 = invoke(stypy.reporting.localization.Localization(__file__, 548, 16), print_413276, *[str1_413277, str2_413278, str3_413279, str4_413280], **kwargs_413281)
    
    # SSA join for if statement (line 543)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 542)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'istop' (line 550)
    istop_413283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 11), 'istop')
    int_413284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 20), 'int')
    # Applying the binary operator '!=' (line 550)
    result_ne_413285 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 11), '!=', istop_413283, int_413284)
    
    # Testing the type of an if condition (line 550)
    if_condition_413286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 550, 8), result_ne_413285)
    # Assigning a type to the variable 'if_condition_413286' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'if_condition_413286', if_condition_413286)
    # SSA begins for if statement (line 550)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 550)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 409)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'show' (line 555)
    show_413287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 7), 'show')
    # Testing the type of an if condition (line 555)
    if_condition_413288 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 555, 4), show_413287)
    # Assigning a type to the variable 'if_condition_413288' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'if_condition_413288', if_condition_413288)
    # SSA begins for if statement (line 555)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 556)
    # Processing the call arguments (line 556)
    str_413290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 14), 'str', ' ')
    # Processing the call keyword arguments (line 556)
    kwargs_413291 = {}
    # Getting the type of 'print' (line 556)
    print_413289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'print', False)
    # Calling print(args, kwargs) (line 556)
    print_call_result_413292 = invoke(stypy.reporting.localization.Localization(__file__, 556, 8), print_413289, *[str_413290], **kwargs_413291)
    
    
    # Call to print(...): (line 557)
    # Processing the call arguments (line 557)
    str_413294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 14), 'str', 'LSQR finished')
    # Processing the call keyword arguments (line 557)
    kwargs_413295 = {}
    # Getting the type of 'print' (line 557)
    print_413293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'print', False)
    # Calling print(args, kwargs) (line 557)
    print_call_result_413296 = invoke(stypy.reporting.localization.Localization(__file__, 557, 8), print_413293, *[str_413294], **kwargs_413295)
    
    
    # Call to print(...): (line 558)
    # Processing the call arguments (line 558)
    
    # Obtaining the type of the subscript
    # Getting the type of 'istop' (line 558)
    istop_413298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 18), 'istop', False)
    # Getting the type of 'msg' (line 558)
    msg_413299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 14), 'msg', False)
    # Obtaining the member '__getitem__' of a type (line 558)
    getitem___413300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 14), msg_413299, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 558)
    subscript_call_result_413301 = invoke(stypy.reporting.localization.Localization(__file__, 558, 14), getitem___413300, istop_413298)
    
    # Processing the call keyword arguments (line 558)
    kwargs_413302 = {}
    # Getting the type of 'print' (line 558)
    print_413297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'print', False)
    # Calling print(args, kwargs) (line 558)
    print_call_result_413303 = invoke(stypy.reporting.localization.Localization(__file__, 558, 8), print_413297, *[subscript_call_result_413301], **kwargs_413302)
    
    
    # Call to print(...): (line 559)
    # Processing the call arguments (line 559)
    str_413305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 14), 'str', ' ')
    # Processing the call keyword arguments (line 559)
    kwargs_413306 = {}
    # Getting the type of 'print' (line 559)
    print_413304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'print', False)
    # Calling print(args, kwargs) (line 559)
    print_call_result_413307 = invoke(stypy.reporting.localization.Localization(__file__, 559, 8), print_413304, *[str_413305], **kwargs_413306)
    
    
    # Assigning a BinOp to a Name (line 560):
    
    # Assigning a BinOp to a Name (line 560):
    str_413308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 15), 'str', 'istop =%8g   r1norm =%8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 560)
    tuple_413309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 560)
    # Adding element type (line 560)
    # Getting the type of 'istop' (line 560)
    istop_413310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 47), 'istop')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 47), tuple_413309, istop_413310)
    # Adding element type (line 560)
    # Getting the type of 'r1norm' (line 560)
    r1norm_413311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 54), 'r1norm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 47), tuple_413309, r1norm_413311)
    
    # Applying the binary operator '%' (line 560)
    result_mod_413312 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 15), '%', str_413308, tuple_413309)
    
    # Assigning a type to the variable 'str1' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'str1', result_mod_413312)
    
    # Assigning a BinOp to a Name (line 561):
    
    # Assigning a BinOp to a Name (line 561):
    str_413313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 15), 'str', 'anorm =%8.1e   arnorm =%8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 561)
    tuple_413314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 561)
    # Adding element type (line 561)
    # Getting the type of 'anorm' (line 561)
    anorm_413315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 49), 'anorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 49), tuple_413314, anorm_413315)
    # Adding element type (line 561)
    # Getting the type of 'arnorm' (line 561)
    arnorm_413316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 56), 'arnorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 49), tuple_413314, arnorm_413316)
    
    # Applying the binary operator '%' (line 561)
    result_mod_413317 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 15), '%', str_413313, tuple_413314)
    
    # Assigning a type to the variable 'str2' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'str2', result_mod_413317)
    
    # Assigning a BinOp to a Name (line 562):
    
    # Assigning a BinOp to a Name (line 562):
    str_413318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 15), 'str', 'itn   =%8g   r2norm =%8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 562)
    tuple_413319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 562)
    # Adding element type (line 562)
    # Getting the type of 'itn' (line 562)
    itn_413320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 47), 'itn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 47), tuple_413319, itn_413320)
    # Adding element type (line 562)
    # Getting the type of 'r2norm' (line 562)
    r2norm_413321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 52), 'r2norm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 47), tuple_413319, r2norm_413321)
    
    # Applying the binary operator '%' (line 562)
    result_mod_413322 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 15), '%', str_413318, tuple_413319)
    
    # Assigning a type to the variable 'str3' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'str3', result_mod_413322)
    
    # Assigning a BinOp to a Name (line 563):
    
    # Assigning a BinOp to a Name (line 563):
    str_413323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 15), 'str', 'acond =%8.1e   xnorm  =%8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 563)
    tuple_413324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 563)
    # Adding element type (line 563)
    # Getting the type of 'acond' (line 563)
    acond_413325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 49), 'acond')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 49), tuple_413324, acond_413325)
    # Adding element type (line 563)
    # Getting the type of 'xnorm' (line 563)
    xnorm_413326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 56), 'xnorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 49), tuple_413324, xnorm_413326)
    
    # Applying the binary operator '%' (line 563)
    result_mod_413327 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 15), '%', str_413323, tuple_413324)
    
    # Assigning a type to the variable 'str4' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'str4', result_mod_413327)
    
    # Call to print(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'str1' (line 564)
    str1_413329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 14), 'str1', False)
    str_413330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 21), 'str', '   ')
    # Applying the binary operator '+' (line 564)
    result_add_413331 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 14), '+', str1_413329, str_413330)
    
    # Getting the type of 'str2' (line 564)
    str2_413332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 29), 'str2', False)
    # Applying the binary operator '+' (line 564)
    result_add_413333 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 27), '+', result_add_413331, str2_413332)
    
    # Processing the call keyword arguments (line 564)
    kwargs_413334 = {}
    # Getting the type of 'print' (line 564)
    print_413328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'print', False)
    # Calling print(args, kwargs) (line 564)
    print_call_result_413335 = invoke(stypy.reporting.localization.Localization(__file__, 564, 8), print_413328, *[result_add_413333], **kwargs_413334)
    
    
    # Call to print(...): (line 565)
    # Processing the call arguments (line 565)
    # Getting the type of 'str3' (line 565)
    str3_413337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 14), 'str3', False)
    str_413338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 21), 'str', '   ')
    # Applying the binary operator '+' (line 565)
    result_add_413339 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 14), '+', str3_413337, str_413338)
    
    # Getting the type of 'str4' (line 565)
    str4_413340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 29), 'str4', False)
    # Applying the binary operator '+' (line 565)
    result_add_413341 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 27), '+', result_add_413339, str4_413340)
    
    # Processing the call keyword arguments (line 565)
    kwargs_413342 = {}
    # Getting the type of 'print' (line 565)
    print_413336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'print', False)
    # Calling print(args, kwargs) (line 565)
    print_call_result_413343 = invoke(stypy.reporting.localization.Localization(__file__, 565, 8), print_413336, *[result_add_413341], **kwargs_413342)
    
    
    # Call to print(...): (line 566)
    # Processing the call arguments (line 566)
    str_413345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 14), 'str', ' ')
    # Processing the call keyword arguments (line 566)
    kwargs_413346 = {}
    # Getting the type of 'print' (line 566)
    print_413344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'print', False)
    # Calling print(args, kwargs) (line 566)
    print_call_result_413347 = invoke(stypy.reporting.localization.Localization(__file__, 566, 8), print_413344, *[str_413345], **kwargs_413346)
    
    # SSA join for if statement (line 555)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 568)
    tuple_413348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 568)
    # Adding element type (line 568)
    # Getting the type of 'x' (line 568)
    x_413349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 11), tuple_413348, x_413349)
    # Adding element type (line 568)
    # Getting the type of 'istop' (line 568)
    istop_413350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 14), 'istop')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 11), tuple_413348, istop_413350)
    # Adding element type (line 568)
    # Getting the type of 'itn' (line 568)
    itn_413351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 21), 'itn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 11), tuple_413348, itn_413351)
    # Adding element type (line 568)
    # Getting the type of 'r1norm' (line 568)
    r1norm_413352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 26), 'r1norm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 11), tuple_413348, r1norm_413352)
    # Adding element type (line 568)
    # Getting the type of 'r2norm' (line 568)
    r2norm_413353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 34), 'r2norm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 11), tuple_413348, r2norm_413353)
    # Adding element type (line 568)
    # Getting the type of 'anorm' (line 568)
    anorm_413354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 42), 'anorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 11), tuple_413348, anorm_413354)
    # Adding element type (line 568)
    # Getting the type of 'acond' (line 568)
    acond_413355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 49), 'acond')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 11), tuple_413348, acond_413355)
    # Adding element type (line 568)
    # Getting the type of 'arnorm' (line 568)
    arnorm_413356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 56), 'arnorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 11), tuple_413348, arnorm_413356)
    # Adding element type (line 568)
    # Getting the type of 'xnorm' (line 568)
    xnorm_413357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 64), 'xnorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 11), tuple_413348, xnorm_413357)
    # Adding element type (line 568)
    # Getting the type of 'var' (line 568)
    var_413358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 71), 'var')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 11), tuple_413348, var_413358)
    
    # Assigning a type to the variable 'stypy_return_type' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'stypy_return_type', tuple_413348)
    
    # ################# End of 'lsqr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lsqr' in the type store
    # Getting the type of 'stypy_return_type' (line 98)
    stypy_return_type_413359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_413359)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lsqr'
    return stypy_return_type_413359

# Assigning a type to the variable 'lsqr' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'lsqr', lsqr)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
