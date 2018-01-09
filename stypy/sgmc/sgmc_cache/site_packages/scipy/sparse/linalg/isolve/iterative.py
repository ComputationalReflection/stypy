
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Iterative methods for solving linear systems'''
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: __all__ = ['bicg','bicgstab','cg','cgs','gmres','qmr']
6: 
7: from . import _iterative
8: 
9: from scipy.sparse.linalg.interface import LinearOperator
10: from scipy._lib.decorator import decorator
11: from .utils import make_system
12: from scipy._lib._util import _aligned_zeros
13: from scipy._lib._threadsafety import non_reentrant
14: 
15: _type_conv = {'f':'s', 'd':'d', 'F':'c', 'D':'z'}
16: 
17: 
18: # Part of the docstring common to all iterative solvers
19: common_doc1 = \
20: '''
21: Parameters
22: ----------
23: A : {sparse matrix, dense matrix, LinearOperator}'''
24: 
25: common_doc2 = \
26: '''b : {array, matrix}
27:     Right hand side of the linear system. Has shape (N,) or (N,1).
28: 
29: Returns
30: -------
31: x : {array, matrix}
32:     The converged solution.
33: info : integer
34:     Provides convergence information:
35:         0  : successful exit
36:         >0 : convergence to tolerance not achieved, number of iterations
37:         <0 : illegal input or breakdown
38: 
39: Other Parameters
40: ----------------
41: x0  : {array, matrix}
42:     Starting guess for the solution.
43: tol : float
44:     Tolerance to achieve. The algorithm terminates when either the relative
45:     or the absolute residual is below `tol`.
46: maxiter : integer
47:     Maximum number of iterations.  Iteration will stop after maxiter
48:     steps even if the specified tolerance has not been achieved.
49: M : {sparse matrix, dense matrix, LinearOperator}
50:     Preconditioner for A.  The preconditioner should approximate the
51:     inverse of A.  Effective preconditioning dramatically improves the
52:     rate of convergence, which implies that fewer iterations are needed
53:     to reach a given error tolerance.
54: callback : function
55:     User-supplied function to call after each iteration.  It is called
56:     as callback(xk), where xk is the current solution vector.
57: 
58: '''
59: 
60: 
61: def set_docstring(header, Ainfo, footer=''):
62:     def combine(fn):
63:         fn.__doc__ = '\n'.join((header, common_doc1,
64:                                '    ' + Ainfo.replace('\n', '\n    '),
65:                                common_doc2, footer))
66:         return fn
67:     return combine
68: 
69: 
70: @set_docstring('Use BIConjugate Gradient iteration to solve ``Ax = b``.',
71:                'The real or complex N-by-N matrix of the linear system.\n'
72:                'It is required that the linear operator can produce\n'
73:                '``Ax`` and ``A^T x``.')
74: @non_reentrant()
75: def bicg(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None):
76:     A,M,x,b,postprocess = make_system(A, M, x0, b)
77: 
78:     n = len(b)
79:     if maxiter is None:
80:         maxiter = n*10
81: 
82:     matvec, rmatvec = A.matvec, A.rmatvec
83:     psolve, rpsolve = M.matvec, M.rmatvec
84:     ltr = _type_conv[x.dtype.char]
85:     revcom = getattr(_iterative, ltr + 'bicgrevcom')
86:     stoptest = getattr(_iterative, ltr + 'stoptest2')
87: 
88:     resid = tol
89:     ndx1 = 1
90:     ndx2 = -1
91:     # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
92:     work = _aligned_zeros(6*n,dtype=x.dtype)
93:     ijob = 1
94:     info = 0
95:     ftflag = True
96:     bnrm2 = -1.0
97:     iter_ = maxiter
98:     while True:
99:         olditer = iter_
100:         x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
101:            revcom(b, x, work, iter_, resid, info, ndx1, ndx2, ijob)
102:         if callback is not None and iter_ > olditer:
103:             callback(x)
104:         slice1 = slice(ndx1-1, ndx1-1+n)
105:         slice2 = slice(ndx2-1, ndx2-1+n)
106:         if (ijob == -1):
107:             if callback is not None:
108:                 callback(x)
109:             break
110:         elif (ijob == 1):
111:             work[slice2] *= sclr2
112:             work[slice2] += sclr1*matvec(work[slice1])
113:         elif (ijob == 2):
114:             work[slice2] *= sclr2
115:             work[slice2] += sclr1*rmatvec(work[slice1])
116:         elif (ijob == 3):
117:             work[slice1] = psolve(work[slice2])
118:         elif (ijob == 4):
119:             work[slice1] = rpsolve(work[slice2])
120:         elif (ijob == 5):
121:             work[slice2] *= sclr2
122:             work[slice2] += sclr1*matvec(x)
123:         elif (ijob == 6):
124:             if ftflag:
125:                 info = -1
126:                 ftflag = False
127:             bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)
128:         ijob = 2
129: 
130:     if info > 0 and iter_ == maxiter and resid > tol:
131:         # info isn't set appropriately otherwise
132:         info = iter_
133: 
134:     return postprocess(x), info
135: 
136: 
137: @set_docstring('Use BIConjugate Gradient STABilized iteration to solve '
138:                '``Ax = b``.',
139:                'The real or complex N-by-N matrix of the linear system.')
140: @non_reentrant()
141: def bicgstab(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None):
142:     A, M, x, b, postprocess = make_system(A, M, x0, b)
143: 
144:     n = len(b)
145:     if maxiter is None:
146:         maxiter = n*10
147: 
148:     matvec = A.matvec
149:     psolve = M.matvec
150:     ltr = _type_conv[x.dtype.char]
151:     revcom = getattr(_iterative, ltr + 'bicgstabrevcom')
152:     stoptest = getattr(_iterative, ltr + 'stoptest2')
153: 
154:     resid = tol
155:     ndx1 = 1
156:     ndx2 = -1
157:     # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
158:     work = _aligned_zeros(7*n,dtype=x.dtype)
159:     ijob = 1
160:     info = 0
161:     ftflag = True
162:     bnrm2 = -1.0
163:     iter_ = maxiter
164:     while True:
165:         olditer = iter_
166:         x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
167:            revcom(b, x, work, iter_, resid, info, ndx1, ndx2, ijob)
168:         if callback is not None and iter_ > olditer:
169:             callback(x)
170:         slice1 = slice(ndx1-1, ndx1-1+n)
171:         slice2 = slice(ndx2-1, ndx2-1+n)
172:         if (ijob == -1):
173:             if callback is not None:
174:                 callback(x)
175:             break
176:         elif (ijob == 1):
177:             work[slice2] *= sclr2
178:             work[slice2] += sclr1*matvec(work[slice1])
179:         elif (ijob == 2):
180:             work[slice1] = psolve(work[slice2])
181:         elif (ijob == 3):
182:             work[slice2] *= sclr2
183:             work[slice2] += sclr1*matvec(x)
184:         elif (ijob == 4):
185:             if ftflag:
186:                 info = -1
187:                 ftflag = False
188:             bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)
189:         ijob = 2
190: 
191:     if info > 0 and iter_ == maxiter and resid > tol:
192:         # info isn't set appropriately otherwise
193:         info = iter_
194: 
195:     return postprocess(x), info
196: 
197: 
198: @set_docstring('Use Conjugate Gradient iteration to solve ``Ax = b``.',
199:                'The real or complex N-by-N matrix of the linear system.\n'
200:                '``A`` must represent a hermitian, positive definite matrix.')
201: @non_reentrant()
202: def cg(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None):
203:     A, M, x, b, postprocess = make_system(A, M, x0, b)
204: 
205:     n = len(b)
206:     if maxiter is None:
207:         maxiter = n*10
208: 
209:     matvec = A.matvec
210:     psolve = M.matvec
211:     ltr = _type_conv[x.dtype.char]
212:     revcom = getattr(_iterative, ltr + 'cgrevcom')
213:     stoptest = getattr(_iterative, ltr + 'stoptest2')
214: 
215:     resid = tol
216:     ndx1 = 1
217:     ndx2 = -1
218:     # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
219:     work = _aligned_zeros(4*n,dtype=x.dtype)
220:     ijob = 1
221:     info = 0
222:     ftflag = True
223:     bnrm2 = -1.0
224:     iter_ = maxiter
225:     while True:
226:         olditer = iter_
227:         x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
228:            revcom(b, x, work, iter_, resid, info, ndx1, ndx2, ijob)
229:         if callback is not None and iter_ > olditer:
230:             callback(x)
231:         slice1 = slice(ndx1-1, ndx1-1+n)
232:         slice2 = slice(ndx2-1, ndx2-1+n)
233:         if (ijob == -1):
234:             if callback is not None:
235:                 callback(x)
236:             break
237:         elif (ijob == 1):
238:             work[slice2] *= sclr2
239:             work[slice2] += sclr1*matvec(work[slice1])
240:         elif (ijob == 2):
241:             work[slice1] = psolve(work[slice2])
242:         elif (ijob == 3):
243:             work[slice2] *= sclr2
244:             work[slice2] += sclr1*matvec(x)
245:         elif (ijob == 4):
246:             if ftflag:
247:                 info = -1
248:                 ftflag = False
249:             bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)
250:         ijob = 2
251: 
252:     if info > 0 and iter_ == maxiter and resid > tol:
253:         # info isn't set appropriately otherwise
254:         info = iter_
255: 
256:     return postprocess(x), info
257: 
258: 
259: @set_docstring('Use Conjugate Gradient Squared iteration to solve ``Ax = b``.',
260:                'The real-valued N-by-N matrix of the linear system.')
261: @non_reentrant()
262: def cgs(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None):
263:     A, M, x, b, postprocess = make_system(A, M, x0, b)
264: 
265:     n = len(b)
266:     if maxiter is None:
267:         maxiter = n*10
268: 
269:     matvec = A.matvec
270:     psolve = M.matvec
271:     ltr = _type_conv[x.dtype.char]
272:     revcom = getattr(_iterative, ltr + 'cgsrevcom')
273:     stoptest = getattr(_iterative, ltr + 'stoptest2')
274: 
275:     resid = tol
276:     ndx1 = 1
277:     ndx2 = -1
278:     # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
279:     work = _aligned_zeros(7*n,dtype=x.dtype)
280:     ijob = 1
281:     info = 0
282:     ftflag = True
283:     bnrm2 = -1.0
284:     iter_ = maxiter
285:     while True:
286:         olditer = iter_
287:         x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
288:            revcom(b, x, work, iter_, resid, info, ndx1, ndx2, ijob)
289:         if callback is not None and iter_ > olditer:
290:             callback(x)
291:         slice1 = slice(ndx1-1, ndx1-1+n)
292:         slice2 = slice(ndx2-1, ndx2-1+n)
293:         if (ijob == -1):
294:             if callback is not None:
295:                 callback(x)
296:             break
297:         elif (ijob == 1):
298:             work[slice2] *= sclr2
299:             work[slice2] += sclr1*matvec(work[slice1])
300:         elif (ijob == 2):
301:             work[slice1] = psolve(work[slice2])
302:         elif (ijob == 3):
303:             work[slice2] *= sclr2
304:             work[slice2] += sclr1*matvec(x)
305:         elif (ijob == 4):
306:             if ftflag:
307:                 info = -1
308:                 ftflag = False
309:             bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)
310:         ijob = 2
311: 
312:     if info > 0 and iter_ == maxiter and resid > tol:
313:         # info isn't set appropriately otherwise
314:         info = iter_
315: 
316:     return postprocess(x), info
317: 
318: 
319: @non_reentrant()
320: def gmres(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, M=None, callback=None, restrt=None):
321:     '''
322:     Use Generalized Minimal RESidual iteration to solve ``Ax = b``.
323: 
324:     Parameters
325:     ----------
326:     A : {sparse matrix, dense matrix, LinearOperator}
327:         The real or complex N-by-N matrix of the linear system.
328:     b : {array, matrix}
329:         Right hand side of the linear system. Has shape (N,) or (N,1).
330: 
331:     Returns
332:     -------
333:     x : {array, matrix}
334:         The converged solution.
335:     info : int
336:         Provides convergence information:
337:           * 0  : successful exit
338:           * >0 : convergence to tolerance not achieved, number of iterations
339:           * <0 : illegal input or breakdown
340: 
341:     Other parameters
342:     ----------------
343:     x0 : {array, matrix}
344:         Starting guess for the solution (a vector of zeros by default).
345:     tol : float
346:         Tolerance to achieve. The algorithm terminates when either the relative
347:         or the absolute residual is below `tol`.
348:     restart : int, optional
349:         Number of iterations between restarts. Larger values increase
350:         iteration cost, but may be necessary for convergence.
351:         Default is 20.
352:     maxiter : int, optional
353:         Maximum number of iterations (restart cycles).  Iteration will stop
354:         after maxiter steps even if the specified tolerance has not been
355:         achieved.
356:     M : {sparse matrix, dense matrix, LinearOperator}
357:         Inverse of the preconditioner of A.  M should approximate the
358:         inverse of A and be easy to solve for (see Notes).  Effective
359:         preconditioning dramatically improves the rate of convergence,
360:         which implies that fewer iterations are needed to reach a given
361:         error tolerance.  By default, no preconditioner is used.
362:     callback : function
363:         User-supplied function to call after each iteration.  It is called
364:         as callback(rk), where rk is the current residual vector.
365:     restrt : int, optional
366:         DEPRECATED - use `restart` instead.
367: 
368:     See Also
369:     --------
370:     LinearOperator
371: 
372:     Notes
373:     -----
374:     A preconditioner, P, is chosen such that P is close to A but easy to solve
375:     for. The preconditioner parameter required by this routine is
376:     ``M = P^-1``. The inverse should preferably not be calculated
377:     explicitly.  Rather, use the following template to produce M::
378: 
379:       # Construct a linear operator that computes P^-1 * x.
380:       import scipy.sparse.linalg as spla
381:       M_x = lambda x: spla.spsolve(P, x)
382:       M = spla.LinearOperator((n, n), M_x)
383: 
384:     Examples
385:     --------
386:     >>> from scipy.sparse import csc_matrix
387:     >>> from scipy.sparse.linalg import gmres
388:     >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
389:     >>> b = np.array([2, 4, -1], dtype=float)
390:     >>> x, exitCode = gmres(A, b)
391:     >>> print(exitCode)            # 0 indicates successful convergence
392:     0
393:     >>> np.allclose(A.dot(x), b)
394:     True
395:     '''
396: 
397:     # Change 'restrt' keyword to 'restart'
398:     if restrt is None:
399:         restrt = restart
400:     elif restart is not None:
401:         raise ValueError("Cannot specify both restart and restrt keywords. "
402:                          "Preferably use 'restart' only.")
403: 
404:     A, M, x, b,postprocess = make_system(A, M, x0, b)
405: 
406:     n = len(b)
407:     if maxiter is None:
408:         maxiter = n*10
409: 
410:     if restrt is None:
411:         restrt = 20
412:     restrt = min(restrt, n)
413: 
414:     matvec = A.matvec
415:     psolve = M.matvec
416:     ltr = _type_conv[x.dtype.char]
417:     revcom = getattr(_iterative, ltr + 'gmresrevcom')
418:     stoptest = getattr(_iterative, ltr + 'stoptest2')
419: 
420:     resid = tol
421:     ndx1 = 1
422:     ndx2 = -1
423:     # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
424:     work = _aligned_zeros((6+restrt)*n,dtype=x.dtype)
425:     work2 = _aligned_zeros((restrt+1)*(2*restrt+2),dtype=x.dtype)
426:     ijob = 1
427:     info = 0
428:     ftflag = True
429:     bnrm2 = -1.0
430:     iter_ = maxiter
431:     old_ijob = ijob
432:     first_pass = True
433:     resid_ready = False
434:     iter_num = 1
435:     while True:
436:         olditer = iter_
437:         x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
438:            revcom(b, x, restrt, work, work2, iter_, resid, info, ndx1, ndx2, ijob)
439:         # if callback is not None and iter_ > olditer:
440:         #    callback(x)
441:         slice1 = slice(ndx1-1, ndx1-1+n)
442:         slice2 = slice(ndx2-1, ndx2-1+n)
443:         if (ijob == -1):  # gmres success, update last residual
444:             if resid_ready and callback is not None:
445:                 callback(resid)
446:                 resid_ready = False
447: 
448:             break
449:         elif (ijob == 1):
450:             work[slice2] *= sclr2
451:             work[slice2] += sclr1*matvec(x)
452:         elif (ijob == 2):
453:             work[slice1] = psolve(work[slice2])
454:             if not first_pass and old_ijob == 3:
455:                 resid_ready = True
456: 
457:             first_pass = False
458:         elif (ijob == 3):
459:             work[slice2] *= sclr2
460:             work[slice2] += sclr1*matvec(work[slice1])
461:             if resid_ready and callback is not None:
462:                 callback(resid)
463:                 resid_ready = False
464:                 iter_num = iter_num+1
465: 
466:         elif (ijob == 4):
467:             if ftflag:
468:                 info = -1
469:                 ftflag = False
470:             bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)
471: 
472:         old_ijob = ijob
473:         ijob = 2
474: 
475:         if iter_num > maxiter:
476:             break
477: 
478:     if info >= 0 and resid > tol:
479:         # info isn't set appropriately otherwise
480:         info = maxiter
481: 
482:     return postprocess(x), info
483: 
484: 
485: @non_reentrant()
486: def qmr(A, b, x0=None, tol=1e-5, maxiter=None, M1=None, M2=None, callback=None):
487:     '''Use Quasi-Minimal Residual iteration to solve ``Ax = b``.
488: 
489:     Parameters
490:     ----------
491:     A : {sparse matrix, dense matrix, LinearOperator}
492:         The real-valued N-by-N matrix of the linear system.
493:         It is required that the linear operator can produce
494:         ``Ax`` and ``A^T x``.
495:     b : {array, matrix}
496:         Right hand side of the linear system. Has shape (N,) or (N,1).
497: 
498:     Returns
499:     -------
500:     x : {array, matrix}
501:         The converged solution.
502:     info : integer
503:         Provides convergence information:
504:             0  : successful exit
505:             >0 : convergence to tolerance not achieved, number of iterations
506:             <0 : illegal input or breakdown
507: 
508:     Other Parameters
509:     ----------------
510:     x0  : {array, matrix}
511:         Starting guess for the solution.
512:     tol : float
513:         Tolerance to achieve. The algorithm terminates when either the relative
514:         or the absolute residual is below `tol`.
515:     maxiter : integer
516:         Maximum number of iterations.  Iteration will stop after maxiter
517:         steps even if the specified tolerance has not been achieved.
518:     M1 : {sparse matrix, dense matrix, LinearOperator}
519:         Left preconditioner for A.
520:     M2 : {sparse matrix, dense matrix, LinearOperator}
521:         Right preconditioner for A. Used together with the left
522:         preconditioner M1.  The matrix M1*A*M2 should have better
523:         conditioned than A alone.
524:     callback : function
525:         User-supplied function to call after each iteration.  It is called
526:         as callback(xk), where xk is the current solution vector.
527: 
528:     See Also
529:     --------
530:     LinearOperator
531: 
532:     Examples
533:     --------
534:     >>> from scipy.sparse import csc_matrix
535:     >>> from scipy.sparse.linalg import qmr
536:     >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
537:     >>> b = np.array([2, 4, -1], dtype=float)
538:     >>> x, exitCode = qmr(A, b)
539:     >>> print(exitCode)            # 0 indicates successful convergence
540:     0
541:     >>> np.allclose(A.dot(x), b)
542:     True
543:     '''
544:     A_ = A
545:     A, M, x, b, postprocess = make_system(A, None, x0, b)
546: 
547:     if M1 is None and M2 is None:
548:         if hasattr(A_,'psolve'):
549:             def left_psolve(b):
550:                 return A_.psolve(b,'left')
551: 
552:             def right_psolve(b):
553:                 return A_.psolve(b,'right')
554: 
555:             def left_rpsolve(b):
556:                 return A_.rpsolve(b,'left')
557: 
558:             def right_rpsolve(b):
559:                 return A_.rpsolve(b,'right')
560:             M1 = LinearOperator(A.shape, matvec=left_psolve, rmatvec=left_rpsolve)
561:             M2 = LinearOperator(A.shape, matvec=right_psolve, rmatvec=right_rpsolve)
562:         else:
563:             def id(b):
564:                 return b
565:             M1 = LinearOperator(A.shape, matvec=id, rmatvec=id)
566:             M2 = LinearOperator(A.shape, matvec=id, rmatvec=id)
567: 
568:     n = len(b)
569:     if maxiter is None:
570:         maxiter = n*10
571: 
572:     ltr = _type_conv[x.dtype.char]
573:     revcom = getattr(_iterative, ltr + 'qmrrevcom')
574:     stoptest = getattr(_iterative, ltr + 'stoptest2')
575: 
576:     resid = tol
577:     ndx1 = 1
578:     ndx2 = -1
579:     # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
580:     work = _aligned_zeros(11*n,x.dtype)
581:     ijob = 1
582:     info = 0
583:     ftflag = True
584:     bnrm2 = -1.0
585:     iter_ = maxiter
586:     while True:
587:         olditer = iter_
588:         x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
589:            revcom(b, x, work, iter_, resid, info, ndx1, ndx2, ijob)
590:         if callback is not None and iter_ > olditer:
591:             callback(x)
592:         slice1 = slice(ndx1-1, ndx1-1+n)
593:         slice2 = slice(ndx2-1, ndx2-1+n)
594:         if (ijob == -1):
595:             if callback is not None:
596:                 callback(x)
597:             break
598:         elif (ijob == 1):
599:             work[slice2] *= sclr2
600:             work[slice2] += sclr1*A.matvec(work[slice1])
601:         elif (ijob == 2):
602:             work[slice2] *= sclr2
603:             work[slice2] += sclr1*A.rmatvec(work[slice1])
604:         elif (ijob == 3):
605:             work[slice1] = M1.matvec(work[slice2])
606:         elif (ijob == 4):
607:             work[slice1] = M2.matvec(work[slice2])
608:         elif (ijob == 5):
609:             work[slice1] = M1.rmatvec(work[slice2])
610:         elif (ijob == 6):
611:             work[slice1] = M2.rmatvec(work[slice2])
612:         elif (ijob == 7):
613:             work[slice2] *= sclr2
614:             work[slice2] += sclr1*A.matvec(x)
615:         elif (ijob == 8):
616:             if ftflag:
617:                 info = -1
618:                 ftflag = False
619:             bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)
620:         ijob = 2
621: 
622:     if info > 0 and iter_ == maxiter and resid > tol:
623:         # info isn't set appropriately otherwise
624:         info = iter_
625: 
626:     return postprocess(x), info
627: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_407993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Iterative methods for solving linear systems')

# Assigning a List to a Name (line 5):

# Assigning a List to a Name (line 5):
__all__ = ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'qmr']
module_type_store.set_exportable_members(['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'qmr'])

# Obtaining an instance of the builtin type 'list' (line 5)
list_407994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
str_407995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 11), 'str', 'bicg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 10), list_407994, str_407995)
# Adding element type (line 5)
str_407996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'str', 'bicgstab')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 10), list_407994, str_407996)
# Adding element type (line 5)
str_407997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 29), 'str', 'cg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 10), list_407994, str_407997)
# Adding element type (line 5)
str_407998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 34), 'str', 'cgs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 10), list_407994, str_407998)
# Adding element type (line 5)
str_407999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 40), 'str', 'gmres')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 10), list_407994, str_407999)
# Adding element type (line 5)
str_408000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 48), 'str', 'qmr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 10), list_407994, str_408000)

# Assigning a type to the variable '__all__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__all__', list_407994)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.sparse.linalg.isolve import _iterative' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_408001 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg.isolve')

if (type(import_408001) is not StypyTypeError):

    if (import_408001 != 'pyd_module'):
        __import__(import_408001)
        sys_modules_408002 = sys.modules[import_408001]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg.isolve', sys_modules_408002.module_type_store, module_type_store, ['_iterative'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_408002, sys_modules_408002.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve import _iterative

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg.isolve', None, module_type_store, ['_iterative'], [_iterative])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg.isolve', import_408001)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.sparse.linalg.interface import LinearOperator' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_408003 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg.interface')

if (type(import_408003) is not StypyTypeError):

    if (import_408003 != 'pyd_module'):
        __import__(import_408003)
        sys_modules_408004 = sys.modules[import_408003]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg.interface', sys_modules_408004.module_type_store, module_type_store, ['LinearOperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_408004, sys_modules_408004.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.interface import LinearOperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg.interface', None, module_type_store, ['LinearOperator'], [LinearOperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.interface' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg.interface', import_408003)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib.decorator import decorator' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_408005 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.decorator')

if (type(import_408005) is not StypyTypeError):

    if (import_408005 != 'pyd_module'):
        __import__(import_408005)
        sys_modules_408006 = sys.modules[import_408005]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.decorator', sys_modules_408006.module_type_store, module_type_store, ['decorator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_408006, sys_modules_408006.module_type_store, module_type_store)
    else:
        from scipy._lib.decorator import decorator

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.decorator', None, module_type_store, ['decorator'], [decorator])

else:
    # Assigning a type to the variable 'scipy._lib.decorator' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.decorator', import_408005)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.sparse.linalg.isolve.utils import make_system' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_408007 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.isolve.utils')

if (type(import_408007) is not StypyTypeError):

    if (import_408007 != 'pyd_module'):
        __import__(import_408007)
        sys_modules_408008 = sys.modules[import_408007]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.isolve.utils', sys_modules_408008.module_type_store, module_type_store, ['make_system'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_408008, sys_modules_408008.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve.utils import make_system

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.isolve.utils', None, module_type_store, ['make_system'], [make_system])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve.utils' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.isolve.utils', import_408007)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy._lib._util import _aligned_zeros' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_408009 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib._util')

if (type(import_408009) is not StypyTypeError):

    if (import_408009 != 'pyd_module'):
        __import__(import_408009)
        sys_modules_408010 = sys.modules[import_408009]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib._util', sys_modules_408010.module_type_store, module_type_store, ['_aligned_zeros'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_408010, sys_modules_408010.module_type_store, module_type_store)
    else:
        from scipy._lib._util import _aligned_zeros

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib._util', None, module_type_store, ['_aligned_zeros'], [_aligned_zeros])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib._util', import_408009)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy._lib._threadsafety import non_reentrant' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_408011 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._threadsafety')

if (type(import_408011) is not StypyTypeError):

    if (import_408011 != 'pyd_module'):
        __import__(import_408011)
        sys_modules_408012 = sys.modules[import_408011]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._threadsafety', sys_modules_408012.module_type_store, module_type_store, ['non_reentrant'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_408012, sys_modules_408012.module_type_store, module_type_store)
    else:
        from scipy._lib._threadsafety import non_reentrant

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._threadsafety', None, module_type_store, ['non_reentrant'], [non_reentrant])

else:
    # Assigning a type to the variable 'scipy._lib._threadsafety' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._threadsafety', import_408011)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')


# Assigning a Dict to a Name (line 15):

# Assigning a Dict to a Name (line 15):

# Obtaining an instance of the builtin type 'dict' (line 15)
dict_408013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 15)
# Adding element type (key, value) (line 15)
str_408014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 14), 'str', 'f')
str_408015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 18), 'str', 's')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), dict_408013, (str_408014, str_408015))
# Adding element type (key, value) (line 15)
str_408016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'str', 'd')
str_408017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 27), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), dict_408013, (str_408016, str_408017))
# Adding element type (key, value) (line 15)
str_408018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'str', 'F')
str_408019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 36), 'str', 'c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), dict_408013, (str_408018, str_408019))
# Adding element type (key, value) (line 15)
str_408020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 41), 'str', 'D')
str_408021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 45), 'str', 'z')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), dict_408013, (str_408020, str_408021))

# Assigning a type to the variable '_type_conv' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '_type_conv', dict_408013)

# Assigning a Str to a Name (line 19):

# Assigning a Str to a Name (line 19):
str_408022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'str', '\nParameters\n----------\nA : {sparse matrix, dense matrix, LinearOperator}')
# Assigning a type to the variable 'common_doc1' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'common_doc1', str_408022)

# Assigning a Str to a Name (line 25):

# Assigning a Str to a Name (line 25):
str_408023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'str', 'b : {array, matrix}\n    Right hand side of the linear system. Has shape (N,) or (N,1).\n\nReturns\n-------\nx : {array, matrix}\n    The converged solution.\ninfo : integer\n    Provides convergence information:\n        0  : successful exit\n        >0 : convergence to tolerance not achieved, number of iterations\n        <0 : illegal input or breakdown\n\nOther Parameters\n----------------\nx0  : {array, matrix}\n    Starting guess for the solution.\ntol : float\n    Tolerance to achieve. The algorithm terminates when either the relative\n    or the absolute residual is below `tol`.\nmaxiter : integer\n    Maximum number of iterations.  Iteration will stop after maxiter\n    steps even if the specified tolerance has not been achieved.\nM : {sparse matrix, dense matrix, LinearOperator}\n    Preconditioner for A.  The preconditioner should approximate the\n    inverse of A.  Effective preconditioning dramatically improves the\n    rate of convergence, which implies that fewer iterations are needed\n    to reach a given error tolerance.\ncallback : function\n    User-supplied function to call after each iteration.  It is called\n    as callback(xk), where xk is the current solution vector.\n\n')
# Assigning a type to the variable 'common_doc2' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'common_doc2', str_408023)

@norecursion
def set_docstring(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_408024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 40), 'str', '')
    defaults = [str_408024]
    # Create a new context for function 'set_docstring'
    module_type_store = module_type_store.open_function_context('set_docstring', 61, 0, False)
    
    # Passed parameters checking function
    set_docstring.stypy_localization = localization
    set_docstring.stypy_type_of_self = None
    set_docstring.stypy_type_store = module_type_store
    set_docstring.stypy_function_name = 'set_docstring'
    set_docstring.stypy_param_names_list = ['header', 'Ainfo', 'footer']
    set_docstring.stypy_varargs_param_name = None
    set_docstring.stypy_kwargs_param_name = None
    set_docstring.stypy_call_defaults = defaults
    set_docstring.stypy_call_varargs = varargs
    set_docstring.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'set_docstring', ['header', 'Ainfo', 'footer'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'set_docstring', localization, ['header', 'Ainfo', 'footer'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'set_docstring(...)' code ##################


    @norecursion
    def combine(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'combine'
        module_type_store = module_type_store.open_function_context('combine', 62, 4, False)
        
        # Passed parameters checking function
        combine.stypy_localization = localization
        combine.stypy_type_of_self = None
        combine.stypy_type_store = module_type_store
        combine.stypy_function_name = 'combine'
        combine.stypy_param_names_list = ['fn']
        combine.stypy_varargs_param_name = None
        combine.stypy_kwargs_param_name = None
        combine.stypy_call_defaults = defaults
        combine.stypy_call_varargs = varargs
        combine.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'combine', ['fn'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'combine', localization, ['fn'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'combine(...)' code ##################

        
        # Assigning a Call to a Attribute (line 63):
        
        # Assigning a Call to a Attribute (line 63):
        
        # Call to join(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_408027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        # Getting the type of 'header' (line 63)
        header_408028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'header', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 32), tuple_408027, header_408028)
        # Adding element type (line 63)
        # Getting the type of 'common_doc1' (line 63)
        common_doc1_408029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 40), 'common_doc1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 32), tuple_408027, common_doc1_408029)
        # Adding element type (line 63)
        str_408030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 31), 'str', '    ')
        
        # Call to replace(...): (line 64)
        # Processing the call arguments (line 64)
        str_408033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 54), 'str', '\n')
        str_408034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 60), 'str', '\n    ')
        # Processing the call keyword arguments (line 64)
        kwargs_408035 = {}
        # Getting the type of 'Ainfo' (line 64)
        Ainfo_408031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 40), 'Ainfo', False)
        # Obtaining the member 'replace' of a type (line 64)
        replace_408032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 40), Ainfo_408031, 'replace')
        # Calling replace(args, kwargs) (line 64)
        replace_call_result_408036 = invoke(stypy.reporting.localization.Localization(__file__, 64, 40), replace_408032, *[str_408033, str_408034], **kwargs_408035)
        
        # Applying the binary operator '+' (line 64)
        result_add_408037 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 31), '+', str_408030, replace_call_result_408036)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 32), tuple_408027, result_add_408037)
        # Adding element type (line 63)
        # Getting the type of 'common_doc2' (line 65)
        common_doc2_408038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 31), 'common_doc2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 32), tuple_408027, common_doc2_408038)
        # Adding element type (line 63)
        # Getting the type of 'footer' (line 65)
        footer_408039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 44), 'footer', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 32), tuple_408027, footer_408039)
        
        # Processing the call keyword arguments (line 63)
        kwargs_408040 = {}
        str_408025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'str', '\n')
        # Obtaining the member 'join' of a type (line 63)
        join_408026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 21), str_408025, 'join')
        # Calling join(args, kwargs) (line 63)
        join_call_result_408041 = invoke(stypy.reporting.localization.Localization(__file__, 63, 21), join_408026, *[tuple_408027], **kwargs_408040)
        
        # Getting the type of 'fn' (line 63)
        fn_408042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'fn')
        # Setting the type of the member '__doc__' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), fn_408042, '__doc__', join_call_result_408041)
        # Getting the type of 'fn' (line 66)
        fn_408043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 15), 'fn')
        # Assigning a type to the variable 'stypy_return_type' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'stypy_return_type', fn_408043)
        
        # ################# End of 'combine(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'combine' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_408044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_408044)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'combine'
        return stypy_return_type_408044

    # Assigning a type to the variable 'combine' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'combine', combine)
    # Getting the type of 'combine' (line 67)
    combine_408045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'combine')
    # Assigning a type to the variable 'stypy_return_type' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type', combine_408045)
    
    # ################# End of 'set_docstring(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set_docstring' in the type store
    # Getting the type of 'stypy_return_type' (line 61)
    stypy_return_type_408046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_408046)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set_docstring'
    return stypy_return_type_408046

# Assigning a type to the variable 'set_docstring' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'set_docstring', set_docstring)

@norecursion
def bicg(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 75)
    None_408047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'None')
    float_408048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 28), 'float')
    # Getting the type of 'None' (line 75)
    None_408049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 42), 'None')
    # Getting the type of 'None' (line 75)
    None_408050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 50), 'None')
    # Getting the type of 'None' (line 75)
    None_408051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 65), 'None')
    defaults = [None_408047, float_408048, None_408049, None_408050, None_408051]
    # Create a new context for function 'bicg'
    module_type_store = module_type_store.open_function_context('bicg', 70, 0, False)
    
    # Passed parameters checking function
    bicg.stypy_localization = localization
    bicg.stypy_type_of_self = None
    bicg.stypy_type_store = module_type_store
    bicg.stypy_function_name = 'bicg'
    bicg.stypy_param_names_list = ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback']
    bicg.stypy_varargs_param_name = None
    bicg.stypy_kwargs_param_name = None
    bicg.stypy_call_defaults = defaults
    bicg.stypy_call_varargs = varargs
    bicg.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bicg', ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bicg', localization, ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bicg(...)' code ##################

    
    # Assigning a Call to a Tuple (line 76):
    
    # Assigning a Subscript to a Name (line 76):
    
    # Obtaining the type of the subscript
    int_408052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'int')
    
    # Call to make_system(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'A' (line 76)
    A_408054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 38), 'A', False)
    # Getting the type of 'M' (line 76)
    M_408055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 41), 'M', False)
    # Getting the type of 'x0' (line 76)
    x0_408056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'x0', False)
    # Getting the type of 'b' (line 76)
    b_408057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 48), 'b', False)
    # Processing the call keyword arguments (line 76)
    kwargs_408058 = {}
    # Getting the type of 'make_system' (line 76)
    make_system_408053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 76)
    make_system_call_result_408059 = invoke(stypy.reporting.localization.Localization(__file__, 76, 26), make_system_408053, *[A_408054, M_408055, x0_408056, b_408057], **kwargs_408058)
    
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___408060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 4), make_system_call_result_408059, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_408061 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), getitem___408060, int_408052)
    
    # Assigning a type to the variable 'tuple_var_assignment_407887' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_407887', subscript_call_result_408061)
    
    # Assigning a Subscript to a Name (line 76):
    
    # Obtaining the type of the subscript
    int_408062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'int')
    
    # Call to make_system(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'A' (line 76)
    A_408064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 38), 'A', False)
    # Getting the type of 'M' (line 76)
    M_408065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 41), 'M', False)
    # Getting the type of 'x0' (line 76)
    x0_408066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'x0', False)
    # Getting the type of 'b' (line 76)
    b_408067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 48), 'b', False)
    # Processing the call keyword arguments (line 76)
    kwargs_408068 = {}
    # Getting the type of 'make_system' (line 76)
    make_system_408063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 76)
    make_system_call_result_408069 = invoke(stypy.reporting.localization.Localization(__file__, 76, 26), make_system_408063, *[A_408064, M_408065, x0_408066, b_408067], **kwargs_408068)
    
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___408070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 4), make_system_call_result_408069, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_408071 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), getitem___408070, int_408062)
    
    # Assigning a type to the variable 'tuple_var_assignment_407888' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_407888', subscript_call_result_408071)
    
    # Assigning a Subscript to a Name (line 76):
    
    # Obtaining the type of the subscript
    int_408072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'int')
    
    # Call to make_system(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'A' (line 76)
    A_408074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 38), 'A', False)
    # Getting the type of 'M' (line 76)
    M_408075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 41), 'M', False)
    # Getting the type of 'x0' (line 76)
    x0_408076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'x0', False)
    # Getting the type of 'b' (line 76)
    b_408077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 48), 'b', False)
    # Processing the call keyword arguments (line 76)
    kwargs_408078 = {}
    # Getting the type of 'make_system' (line 76)
    make_system_408073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 76)
    make_system_call_result_408079 = invoke(stypy.reporting.localization.Localization(__file__, 76, 26), make_system_408073, *[A_408074, M_408075, x0_408076, b_408077], **kwargs_408078)
    
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___408080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 4), make_system_call_result_408079, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_408081 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), getitem___408080, int_408072)
    
    # Assigning a type to the variable 'tuple_var_assignment_407889' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_407889', subscript_call_result_408081)
    
    # Assigning a Subscript to a Name (line 76):
    
    # Obtaining the type of the subscript
    int_408082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'int')
    
    # Call to make_system(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'A' (line 76)
    A_408084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 38), 'A', False)
    # Getting the type of 'M' (line 76)
    M_408085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 41), 'M', False)
    # Getting the type of 'x0' (line 76)
    x0_408086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'x0', False)
    # Getting the type of 'b' (line 76)
    b_408087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 48), 'b', False)
    # Processing the call keyword arguments (line 76)
    kwargs_408088 = {}
    # Getting the type of 'make_system' (line 76)
    make_system_408083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 76)
    make_system_call_result_408089 = invoke(stypy.reporting.localization.Localization(__file__, 76, 26), make_system_408083, *[A_408084, M_408085, x0_408086, b_408087], **kwargs_408088)
    
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___408090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 4), make_system_call_result_408089, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_408091 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), getitem___408090, int_408082)
    
    # Assigning a type to the variable 'tuple_var_assignment_407890' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_407890', subscript_call_result_408091)
    
    # Assigning a Subscript to a Name (line 76):
    
    # Obtaining the type of the subscript
    int_408092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'int')
    
    # Call to make_system(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'A' (line 76)
    A_408094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 38), 'A', False)
    # Getting the type of 'M' (line 76)
    M_408095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 41), 'M', False)
    # Getting the type of 'x0' (line 76)
    x0_408096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'x0', False)
    # Getting the type of 'b' (line 76)
    b_408097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 48), 'b', False)
    # Processing the call keyword arguments (line 76)
    kwargs_408098 = {}
    # Getting the type of 'make_system' (line 76)
    make_system_408093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 76)
    make_system_call_result_408099 = invoke(stypy.reporting.localization.Localization(__file__, 76, 26), make_system_408093, *[A_408094, M_408095, x0_408096, b_408097], **kwargs_408098)
    
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___408100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 4), make_system_call_result_408099, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_408101 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), getitem___408100, int_408092)
    
    # Assigning a type to the variable 'tuple_var_assignment_407891' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_407891', subscript_call_result_408101)
    
    # Assigning a Name to a Name (line 76):
    # Getting the type of 'tuple_var_assignment_407887' (line 76)
    tuple_var_assignment_407887_408102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_407887')
    # Assigning a type to the variable 'A' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'A', tuple_var_assignment_407887_408102)
    
    # Assigning a Name to a Name (line 76):
    # Getting the type of 'tuple_var_assignment_407888' (line 76)
    tuple_var_assignment_407888_408103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_407888')
    # Assigning a type to the variable 'M' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 6), 'M', tuple_var_assignment_407888_408103)
    
    # Assigning a Name to a Name (line 76):
    # Getting the type of 'tuple_var_assignment_407889' (line 76)
    tuple_var_assignment_407889_408104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_407889')
    # Assigning a type to the variable 'x' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'x', tuple_var_assignment_407889_408104)
    
    # Assigning a Name to a Name (line 76):
    # Getting the type of 'tuple_var_assignment_407890' (line 76)
    tuple_var_assignment_407890_408105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_407890')
    # Assigning a type to the variable 'b' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 10), 'b', tuple_var_assignment_407890_408105)
    
    # Assigning a Name to a Name (line 76):
    # Getting the type of 'tuple_var_assignment_407891' (line 76)
    tuple_var_assignment_407891_408106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_407891')
    # Assigning a type to the variable 'postprocess' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'postprocess', tuple_var_assignment_407891_408106)
    
    # Assigning a Call to a Name (line 78):
    
    # Assigning a Call to a Name (line 78):
    
    # Call to len(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'b' (line 78)
    b_408108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'b', False)
    # Processing the call keyword arguments (line 78)
    kwargs_408109 = {}
    # Getting the type of 'len' (line 78)
    len_408107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'len', False)
    # Calling len(args, kwargs) (line 78)
    len_call_result_408110 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), len_408107, *[b_408108], **kwargs_408109)
    
    # Assigning a type to the variable 'n' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'n', len_call_result_408110)
    
    # Type idiom detected: calculating its left and rigth part (line 79)
    # Getting the type of 'maxiter' (line 79)
    maxiter_408111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 7), 'maxiter')
    # Getting the type of 'None' (line 79)
    None_408112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'None')
    
    (may_be_408113, more_types_in_union_408114) = may_be_none(maxiter_408111, None_408112)

    if may_be_408113:

        if more_types_in_union_408114:
            # Runtime conditional SSA (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 80):
        
        # Assigning a BinOp to a Name (line 80):
        # Getting the type of 'n' (line 80)
        n_408115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'n')
        int_408116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'int')
        # Applying the binary operator '*' (line 80)
        result_mul_408117 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 18), '*', n_408115, int_408116)
        
        # Assigning a type to the variable 'maxiter' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'maxiter', result_mul_408117)

        if more_types_in_union_408114:
            # SSA join for if statement (line 79)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Tuple to a Tuple (line 82):
    
    # Assigning a Attribute to a Name (line 82):
    # Getting the type of 'A' (line 82)
    A_408118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'A')
    # Obtaining the member 'matvec' of a type (line 82)
    matvec_408119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 22), A_408118, 'matvec')
    # Assigning a type to the variable 'tuple_assignment_407892' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'tuple_assignment_407892', matvec_408119)
    
    # Assigning a Attribute to a Name (line 82):
    # Getting the type of 'A' (line 82)
    A_408120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 32), 'A')
    # Obtaining the member 'rmatvec' of a type (line 82)
    rmatvec_408121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 32), A_408120, 'rmatvec')
    # Assigning a type to the variable 'tuple_assignment_407893' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'tuple_assignment_407893', rmatvec_408121)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'tuple_assignment_407892' (line 82)
    tuple_assignment_407892_408122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'tuple_assignment_407892')
    # Assigning a type to the variable 'matvec' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'matvec', tuple_assignment_407892_408122)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'tuple_assignment_407893' (line 82)
    tuple_assignment_407893_408123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'tuple_assignment_407893')
    # Assigning a type to the variable 'rmatvec' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'rmatvec', tuple_assignment_407893_408123)
    
    # Assigning a Tuple to a Tuple (line 83):
    
    # Assigning a Attribute to a Name (line 83):
    # Getting the type of 'M' (line 83)
    M_408124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'M')
    # Obtaining the member 'matvec' of a type (line 83)
    matvec_408125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 22), M_408124, 'matvec')
    # Assigning a type to the variable 'tuple_assignment_407894' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'tuple_assignment_407894', matvec_408125)
    
    # Assigning a Attribute to a Name (line 83):
    # Getting the type of 'M' (line 83)
    M_408126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 32), 'M')
    # Obtaining the member 'rmatvec' of a type (line 83)
    rmatvec_408127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 32), M_408126, 'rmatvec')
    # Assigning a type to the variable 'tuple_assignment_407895' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'tuple_assignment_407895', rmatvec_408127)
    
    # Assigning a Name to a Name (line 83):
    # Getting the type of 'tuple_assignment_407894' (line 83)
    tuple_assignment_407894_408128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'tuple_assignment_407894')
    # Assigning a type to the variable 'psolve' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'psolve', tuple_assignment_407894_408128)
    
    # Assigning a Name to a Name (line 83):
    # Getting the type of 'tuple_assignment_407895' (line 83)
    tuple_assignment_407895_408129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'tuple_assignment_407895')
    # Assigning a type to the variable 'rpsolve' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'rpsolve', tuple_assignment_407895_408129)
    
    # Assigning a Subscript to a Name (line 84):
    
    # Assigning a Subscript to a Name (line 84):
    
    # Obtaining the type of the subscript
    # Getting the type of 'x' (line 84)
    x_408130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'x')
    # Obtaining the member 'dtype' of a type (line 84)
    dtype_408131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), x_408130, 'dtype')
    # Obtaining the member 'char' of a type (line 84)
    char_408132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), dtype_408131, 'char')
    # Getting the type of '_type_conv' (line 84)
    _type_conv_408133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 10), '_type_conv')
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___408134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 10), _type_conv_408133, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_408135 = invoke(stypy.reporting.localization.Localization(__file__, 84, 10), getitem___408134, char_408132)
    
    # Assigning a type to the variable 'ltr' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'ltr', subscript_call_result_408135)
    
    # Assigning a Call to a Name (line 85):
    
    # Assigning a Call to a Name (line 85):
    
    # Call to getattr(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of '_iterative' (line 85)
    _iterative_408137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), '_iterative', False)
    # Getting the type of 'ltr' (line 85)
    ltr_408138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 33), 'ltr', False)
    str_408139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 39), 'str', 'bicgrevcom')
    # Applying the binary operator '+' (line 85)
    result_add_408140 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 33), '+', ltr_408138, str_408139)
    
    # Processing the call keyword arguments (line 85)
    kwargs_408141 = {}
    # Getting the type of 'getattr' (line 85)
    getattr_408136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'getattr', False)
    # Calling getattr(args, kwargs) (line 85)
    getattr_call_result_408142 = invoke(stypy.reporting.localization.Localization(__file__, 85, 13), getattr_408136, *[_iterative_408137, result_add_408140], **kwargs_408141)
    
    # Assigning a type to the variable 'revcom' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'revcom', getattr_call_result_408142)
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to getattr(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of '_iterative' (line 86)
    _iterative_408144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), '_iterative', False)
    # Getting the type of 'ltr' (line 86)
    ltr_408145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 35), 'ltr', False)
    str_408146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 41), 'str', 'stoptest2')
    # Applying the binary operator '+' (line 86)
    result_add_408147 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 35), '+', ltr_408145, str_408146)
    
    # Processing the call keyword arguments (line 86)
    kwargs_408148 = {}
    # Getting the type of 'getattr' (line 86)
    getattr_408143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'getattr', False)
    # Calling getattr(args, kwargs) (line 86)
    getattr_call_result_408149 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), getattr_408143, *[_iterative_408144, result_add_408147], **kwargs_408148)
    
    # Assigning a type to the variable 'stoptest' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stoptest', getattr_call_result_408149)
    
    # Assigning a Name to a Name (line 88):
    
    # Assigning a Name to a Name (line 88):
    # Getting the type of 'tol' (line 88)
    tol_408150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'tol')
    # Assigning a type to the variable 'resid' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'resid', tol_408150)
    
    # Assigning a Num to a Name (line 89):
    
    # Assigning a Num to a Name (line 89):
    int_408151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 11), 'int')
    # Assigning a type to the variable 'ndx1' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'ndx1', int_408151)
    
    # Assigning a Num to a Name (line 90):
    
    # Assigning a Num to a Name (line 90):
    int_408152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 11), 'int')
    # Assigning a type to the variable 'ndx2' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'ndx2', int_408152)
    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to _aligned_zeros(...): (line 92)
    # Processing the call arguments (line 92)
    int_408154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 26), 'int')
    # Getting the type of 'n' (line 92)
    n_408155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'n', False)
    # Applying the binary operator '*' (line 92)
    result_mul_408156 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 26), '*', int_408154, n_408155)
    
    # Processing the call keyword arguments (line 92)
    # Getting the type of 'x' (line 92)
    x_408157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 36), 'x', False)
    # Obtaining the member 'dtype' of a type (line 92)
    dtype_408158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 36), x_408157, 'dtype')
    keyword_408159 = dtype_408158
    kwargs_408160 = {'dtype': keyword_408159}
    # Getting the type of '_aligned_zeros' (line 92)
    _aligned_zeros_408153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), '_aligned_zeros', False)
    # Calling _aligned_zeros(args, kwargs) (line 92)
    _aligned_zeros_call_result_408161 = invoke(stypy.reporting.localization.Localization(__file__, 92, 11), _aligned_zeros_408153, *[result_mul_408156], **kwargs_408160)
    
    # Assigning a type to the variable 'work' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'work', _aligned_zeros_call_result_408161)
    
    # Assigning a Num to a Name (line 93):
    
    # Assigning a Num to a Name (line 93):
    int_408162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 11), 'int')
    # Assigning a type to the variable 'ijob' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'ijob', int_408162)
    
    # Assigning a Num to a Name (line 94):
    
    # Assigning a Num to a Name (line 94):
    int_408163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 11), 'int')
    # Assigning a type to the variable 'info' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'info', int_408163)
    
    # Assigning a Name to a Name (line 95):
    
    # Assigning a Name to a Name (line 95):
    # Getting the type of 'True' (line 95)
    True_408164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'True')
    # Assigning a type to the variable 'ftflag' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'ftflag', True_408164)
    
    # Assigning a Num to a Name (line 96):
    
    # Assigning a Num to a Name (line 96):
    float_408165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'float')
    # Assigning a type to the variable 'bnrm2' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'bnrm2', float_408165)
    
    # Assigning a Name to a Name (line 97):
    
    # Assigning a Name to a Name (line 97):
    # Getting the type of 'maxiter' (line 97)
    maxiter_408166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'maxiter')
    # Assigning a type to the variable 'iter_' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'iter_', maxiter_408166)
    
    # Getting the type of 'True' (line 98)
    True_408167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 10), 'True')
    # Testing the type of an if condition (line 98)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 4), True_408167)
    # SSA begins for while statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Name to a Name (line 99):
    
    # Assigning a Name to a Name (line 99):
    # Getting the type of 'iter_' (line 99)
    iter__408168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'iter_')
    # Assigning a type to the variable 'olditer' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'olditer', iter__408168)
    
    # Assigning a Call to a Tuple (line 100):
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_408169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'int')
    
    # Call to revcom(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'b' (line 101)
    b_408171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'b', False)
    # Getting the type of 'x' (line 101)
    x_408172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'x', False)
    # Getting the type of 'work' (line 101)
    work_408173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'work', False)
    # Getting the type of 'iter_' (line 101)
    iter__408174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'iter_', False)
    # Getting the type of 'resid' (line 101)
    resid_408175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 37), 'resid', False)
    # Getting the type of 'info' (line 101)
    info_408176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'info', False)
    # Getting the type of 'ndx1' (line 101)
    ndx1_408177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 101)
    ndx2_408178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 101)
    ijob_408179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 62), 'ijob', False)
    # Processing the call keyword arguments (line 101)
    kwargs_408180 = {}
    # Getting the type of 'revcom' (line 101)
    revcom_408170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 101)
    revcom_call_result_408181 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), revcom_408170, *[b_408171, x_408172, work_408173, iter__408174, resid_408175, info_408176, ndx1_408177, ndx2_408178, ijob_408179], **kwargs_408180)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___408182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), revcom_call_result_408181, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_408183 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), getitem___408182, int_408169)
    
    # Assigning a type to the variable 'tuple_var_assignment_407896' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407896', subscript_call_result_408183)
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_408184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'int')
    
    # Call to revcom(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'b' (line 101)
    b_408186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'b', False)
    # Getting the type of 'x' (line 101)
    x_408187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'x', False)
    # Getting the type of 'work' (line 101)
    work_408188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'work', False)
    # Getting the type of 'iter_' (line 101)
    iter__408189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'iter_', False)
    # Getting the type of 'resid' (line 101)
    resid_408190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 37), 'resid', False)
    # Getting the type of 'info' (line 101)
    info_408191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'info', False)
    # Getting the type of 'ndx1' (line 101)
    ndx1_408192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 101)
    ndx2_408193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 101)
    ijob_408194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 62), 'ijob', False)
    # Processing the call keyword arguments (line 101)
    kwargs_408195 = {}
    # Getting the type of 'revcom' (line 101)
    revcom_408185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 101)
    revcom_call_result_408196 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), revcom_408185, *[b_408186, x_408187, work_408188, iter__408189, resid_408190, info_408191, ndx1_408192, ndx2_408193, ijob_408194], **kwargs_408195)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___408197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), revcom_call_result_408196, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_408198 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), getitem___408197, int_408184)
    
    # Assigning a type to the variable 'tuple_var_assignment_407897' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407897', subscript_call_result_408198)
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_408199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'int')
    
    # Call to revcom(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'b' (line 101)
    b_408201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'b', False)
    # Getting the type of 'x' (line 101)
    x_408202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'x', False)
    # Getting the type of 'work' (line 101)
    work_408203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'work', False)
    # Getting the type of 'iter_' (line 101)
    iter__408204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'iter_', False)
    # Getting the type of 'resid' (line 101)
    resid_408205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 37), 'resid', False)
    # Getting the type of 'info' (line 101)
    info_408206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'info', False)
    # Getting the type of 'ndx1' (line 101)
    ndx1_408207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 101)
    ndx2_408208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 101)
    ijob_408209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 62), 'ijob', False)
    # Processing the call keyword arguments (line 101)
    kwargs_408210 = {}
    # Getting the type of 'revcom' (line 101)
    revcom_408200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 101)
    revcom_call_result_408211 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), revcom_408200, *[b_408201, x_408202, work_408203, iter__408204, resid_408205, info_408206, ndx1_408207, ndx2_408208, ijob_408209], **kwargs_408210)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___408212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), revcom_call_result_408211, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_408213 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), getitem___408212, int_408199)
    
    # Assigning a type to the variable 'tuple_var_assignment_407898' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407898', subscript_call_result_408213)
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_408214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'int')
    
    # Call to revcom(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'b' (line 101)
    b_408216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'b', False)
    # Getting the type of 'x' (line 101)
    x_408217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'x', False)
    # Getting the type of 'work' (line 101)
    work_408218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'work', False)
    # Getting the type of 'iter_' (line 101)
    iter__408219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'iter_', False)
    # Getting the type of 'resid' (line 101)
    resid_408220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 37), 'resid', False)
    # Getting the type of 'info' (line 101)
    info_408221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'info', False)
    # Getting the type of 'ndx1' (line 101)
    ndx1_408222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 101)
    ndx2_408223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 101)
    ijob_408224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 62), 'ijob', False)
    # Processing the call keyword arguments (line 101)
    kwargs_408225 = {}
    # Getting the type of 'revcom' (line 101)
    revcom_408215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 101)
    revcom_call_result_408226 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), revcom_408215, *[b_408216, x_408217, work_408218, iter__408219, resid_408220, info_408221, ndx1_408222, ndx2_408223, ijob_408224], **kwargs_408225)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___408227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), revcom_call_result_408226, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_408228 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), getitem___408227, int_408214)
    
    # Assigning a type to the variable 'tuple_var_assignment_407899' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407899', subscript_call_result_408228)
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_408229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'int')
    
    # Call to revcom(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'b' (line 101)
    b_408231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'b', False)
    # Getting the type of 'x' (line 101)
    x_408232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'x', False)
    # Getting the type of 'work' (line 101)
    work_408233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'work', False)
    # Getting the type of 'iter_' (line 101)
    iter__408234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'iter_', False)
    # Getting the type of 'resid' (line 101)
    resid_408235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 37), 'resid', False)
    # Getting the type of 'info' (line 101)
    info_408236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'info', False)
    # Getting the type of 'ndx1' (line 101)
    ndx1_408237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 101)
    ndx2_408238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 101)
    ijob_408239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 62), 'ijob', False)
    # Processing the call keyword arguments (line 101)
    kwargs_408240 = {}
    # Getting the type of 'revcom' (line 101)
    revcom_408230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 101)
    revcom_call_result_408241 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), revcom_408230, *[b_408231, x_408232, work_408233, iter__408234, resid_408235, info_408236, ndx1_408237, ndx2_408238, ijob_408239], **kwargs_408240)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___408242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), revcom_call_result_408241, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_408243 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), getitem___408242, int_408229)
    
    # Assigning a type to the variable 'tuple_var_assignment_407900' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407900', subscript_call_result_408243)
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_408244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'int')
    
    # Call to revcom(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'b' (line 101)
    b_408246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'b', False)
    # Getting the type of 'x' (line 101)
    x_408247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'x', False)
    # Getting the type of 'work' (line 101)
    work_408248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'work', False)
    # Getting the type of 'iter_' (line 101)
    iter__408249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'iter_', False)
    # Getting the type of 'resid' (line 101)
    resid_408250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 37), 'resid', False)
    # Getting the type of 'info' (line 101)
    info_408251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'info', False)
    # Getting the type of 'ndx1' (line 101)
    ndx1_408252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 101)
    ndx2_408253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 101)
    ijob_408254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 62), 'ijob', False)
    # Processing the call keyword arguments (line 101)
    kwargs_408255 = {}
    # Getting the type of 'revcom' (line 101)
    revcom_408245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 101)
    revcom_call_result_408256 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), revcom_408245, *[b_408246, x_408247, work_408248, iter__408249, resid_408250, info_408251, ndx1_408252, ndx2_408253, ijob_408254], **kwargs_408255)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___408257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), revcom_call_result_408256, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_408258 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), getitem___408257, int_408244)
    
    # Assigning a type to the variable 'tuple_var_assignment_407901' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407901', subscript_call_result_408258)
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_408259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'int')
    
    # Call to revcom(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'b' (line 101)
    b_408261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'b', False)
    # Getting the type of 'x' (line 101)
    x_408262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'x', False)
    # Getting the type of 'work' (line 101)
    work_408263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'work', False)
    # Getting the type of 'iter_' (line 101)
    iter__408264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'iter_', False)
    # Getting the type of 'resid' (line 101)
    resid_408265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 37), 'resid', False)
    # Getting the type of 'info' (line 101)
    info_408266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'info', False)
    # Getting the type of 'ndx1' (line 101)
    ndx1_408267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 101)
    ndx2_408268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 101)
    ijob_408269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 62), 'ijob', False)
    # Processing the call keyword arguments (line 101)
    kwargs_408270 = {}
    # Getting the type of 'revcom' (line 101)
    revcom_408260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 101)
    revcom_call_result_408271 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), revcom_408260, *[b_408261, x_408262, work_408263, iter__408264, resid_408265, info_408266, ndx1_408267, ndx2_408268, ijob_408269], **kwargs_408270)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___408272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), revcom_call_result_408271, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_408273 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), getitem___408272, int_408259)
    
    # Assigning a type to the variable 'tuple_var_assignment_407902' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407902', subscript_call_result_408273)
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_408274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'int')
    
    # Call to revcom(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'b' (line 101)
    b_408276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'b', False)
    # Getting the type of 'x' (line 101)
    x_408277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'x', False)
    # Getting the type of 'work' (line 101)
    work_408278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'work', False)
    # Getting the type of 'iter_' (line 101)
    iter__408279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'iter_', False)
    # Getting the type of 'resid' (line 101)
    resid_408280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 37), 'resid', False)
    # Getting the type of 'info' (line 101)
    info_408281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'info', False)
    # Getting the type of 'ndx1' (line 101)
    ndx1_408282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 101)
    ndx2_408283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 101)
    ijob_408284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 62), 'ijob', False)
    # Processing the call keyword arguments (line 101)
    kwargs_408285 = {}
    # Getting the type of 'revcom' (line 101)
    revcom_408275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 101)
    revcom_call_result_408286 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), revcom_408275, *[b_408276, x_408277, work_408278, iter__408279, resid_408280, info_408281, ndx1_408282, ndx2_408283, ijob_408284], **kwargs_408285)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___408287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), revcom_call_result_408286, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_408288 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), getitem___408287, int_408274)
    
    # Assigning a type to the variable 'tuple_var_assignment_407903' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407903', subscript_call_result_408288)
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_408289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'int')
    
    # Call to revcom(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'b' (line 101)
    b_408291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'b', False)
    # Getting the type of 'x' (line 101)
    x_408292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'x', False)
    # Getting the type of 'work' (line 101)
    work_408293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'work', False)
    # Getting the type of 'iter_' (line 101)
    iter__408294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'iter_', False)
    # Getting the type of 'resid' (line 101)
    resid_408295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 37), 'resid', False)
    # Getting the type of 'info' (line 101)
    info_408296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'info', False)
    # Getting the type of 'ndx1' (line 101)
    ndx1_408297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 101)
    ndx2_408298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 101)
    ijob_408299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 62), 'ijob', False)
    # Processing the call keyword arguments (line 101)
    kwargs_408300 = {}
    # Getting the type of 'revcom' (line 101)
    revcom_408290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 101)
    revcom_call_result_408301 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), revcom_408290, *[b_408291, x_408292, work_408293, iter__408294, resid_408295, info_408296, ndx1_408297, ndx2_408298, ijob_408299], **kwargs_408300)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___408302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), revcom_call_result_408301, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_408303 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), getitem___408302, int_408289)
    
    # Assigning a type to the variable 'tuple_var_assignment_407904' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407904', subscript_call_result_408303)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_407896' (line 100)
    tuple_var_assignment_407896_408304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407896')
    # Assigning a type to the variable 'x' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'x', tuple_var_assignment_407896_408304)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_407897' (line 100)
    tuple_var_assignment_407897_408305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407897')
    # Assigning a type to the variable 'iter_' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'iter_', tuple_var_assignment_407897_408305)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_407898' (line 100)
    tuple_var_assignment_407898_408306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407898')
    # Assigning a type to the variable 'resid' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'resid', tuple_var_assignment_407898_408306)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_407899' (line 100)
    tuple_var_assignment_407899_408307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407899')
    # Assigning a type to the variable 'info' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'info', tuple_var_assignment_407899_408307)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_407900' (line 100)
    tuple_var_assignment_407900_408308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407900')
    # Assigning a type to the variable 'ndx1' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 31), 'ndx1', tuple_var_assignment_407900_408308)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_407901' (line 100)
    tuple_var_assignment_407901_408309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407901')
    # Assigning a type to the variable 'ndx2' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 37), 'ndx2', tuple_var_assignment_407901_408309)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_407902' (line 100)
    tuple_var_assignment_407902_408310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407902')
    # Assigning a type to the variable 'sclr1' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 43), 'sclr1', tuple_var_assignment_407902_408310)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_407903' (line 100)
    tuple_var_assignment_407903_408311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407903')
    # Assigning a type to the variable 'sclr2' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 50), 'sclr2', tuple_var_assignment_407903_408311)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_407904' (line 100)
    tuple_var_assignment_407904_408312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tuple_var_assignment_407904')
    # Assigning a type to the variable 'ijob' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 57), 'ijob', tuple_var_assignment_407904_408312)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'callback' (line 102)
    callback_408313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'callback')
    # Getting the type of 'None' (line 102)
    None_408314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'None')
    # Applying the binary operator 'isnot' (line 102)
    result_is_not_408315 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), 'isnot', callback_408313, None_408314)
    
    
    # Getting the type of 'iter_' (line 102)
    iter__408316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 36), 'iter_')
    # Getting the type of 'olditer' (line 102)
    olditer_408317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 44), 'olditer')
    # Applying the binary operator '>' (line 102)
    result_gt_408318 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 36), '>', iter__408316, olditer_408317)
    
    # Applying the binary operator 'and' (line 102)
    result_and_keyword_408319 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), 'and', result_is_not_408315, result_gt_408318)
    
    # Testing the type of an if condition (line 102)
    if_condition_408320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), result_and_keyword_408319)
    # Assigning a type to the variable 'if_condition_408320' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'if_condition_408320', if_condition_408320)
    # SSA begins for if statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to callback(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'x' (line 103)
    x_408322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 21), 'x', False)
    # Processing the call keyword arguments (line 103)
    kwargs_408323 = {}
    # Getting the type of 'callback' (line 103)
    callback_408321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'callback', False)
    # Calling callback(args, kwargs) (line 103)
    callback_call_result_408324 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), callback_408321, *[x_408322], **kwargs_408323)
    
    # SSA join for if statement (line 102)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 104):
    
    # Assigning a Call to a Name (line 104):
    
    # Call to slice(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'ndx1' (line 104)
    ndx1_408326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 23), 'ndx1', False)
    int_408327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 28), 'int')
    # Applying the binary operator '-' (line 104)
    result_sub_408328 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 23), '-', ndx1_408326, int_408327)
    
    # Getting the type of 'ndx1' (line 104)
    ndx1_408329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 31), 'ndx1', False)
    int_408330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 36), 'int')
    # Applying the binary operator '-' (line 104)
    result_sub_408331 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 31), '-', ndx1_408329, int_408330)
    
    # Getting the type of 'n' (line 104)
    n_408332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 38), 'n', False)
    # Applying the binary operator '+' (line 104)
    result_add_408333 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 37), '+', result_sub_408331, n_408332)
    
    # Processing the call keyword arguments (line 104)
    kwargs_408334 = {}
    # Getting the type of 'slice' (line 104)
    slice_408325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 104)
    slice_call_result_408335 = invoke(stypy.reporting.localization.Localization(__file__, 104, 17), slice_408325, *[result_sub_408328, result_add_408333], **kwargs_408334)
    
    # Assigning a type to the variable 'slice1' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'slice1', slice_call_result_408335)
    
    # Assigning a Call to a Name (line 105):
    
    # Assigning a Call to a Name (line 105):
    
    # Call to slice(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'ndx2' (line 105)
    ndx2_408337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 23), 'ndx2', False)
    int_408338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 28), 'int')
    # Applying the binary operator '-' (line 105)
    result_sub_408339 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 23), '-', ndx2_408337, int_408338)
    
    # Getting the type of 'ndx2' (line 105)
    ndx2_408340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 31), 'ndx2', False)
    int_408341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 36), 'int')
    # Applying the binary operator '-' (line 105)
    result_sub_408342 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 31), '-', ndx2_408340, int_408341)
    
    # Getting the type of 'n' (line 105)
    n_408343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 38), 'n', False)
    # Applying the binary operator '+' (line 105)
    result_add_408344 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 37), '+', result_sub_408342, n_408343)
    
    # Processing the call keyword arguments (line 105)
    kwargs_408345 = {}
    # Getting the type of 'slice' (line 105)
    slice_408336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 105)
    slice_call_result_408346 = invoke(stypy.reporting.localization.Localization(__file__, 105, 17), slice_408336, *[result_sub_408339, result_add_408344], **kwargs_408345)
    
    # Assigning a type to the variable 'slice2' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'slice2', slice_call_result_408346)
    
    
    # Getting the type of 'ijob' (line 106)
    ijob_408347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'ijob')
    int_408348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 20), 'int')
    # Applying the binary operator '==' (line 106)
    result_eq_408349 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 12), '==', ijob_408347, int_408348)
    
    # Testing the type of an if condition (line 106)
    if_condition_408350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), result_eq_408349)
    # Assigning a type to the variable 'if_condition_408350' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_408350', if_condition_408350)
    # SSA begins for if statement (line 106)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 107)
    # Getting the type of 'callback' (line 107)
    callback_408351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'callback')
    # Getting the type of 'None' (line 107)
    None_408352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 31), 'None')
    
    (may_be_408353, more_types_in_union_408354) = may_not_be_none(callback_408351, None_408352)

    if may_be_408353:

        if more_types_in_union_408354:
            # Runtime conditional SSA (line 107)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to callback(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'x' (line 108)
        x_408356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'x', False)
        # Processing the call keyword arguments (line 108)
        kwargs_408357 = {}
        # Getting the type of 'callback' (line 108)
        callback_408355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'callback', False)
        # Calling callback(args, kwargs) (line 108)
        callback_call_result_408358 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), callback_408355, *[x_408356], **kwargs_408357)
        

        if more_types_in_union_408354:
            # SSA join for if statement (line 107)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 106)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 110)
    ijob_408359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'ijob')
    int_408360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 22), 'int')
    # Applying the binary operator '==' (line 110)
    result_eq_408361 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 14), '==', ijob_408359, int_408360)
    
    # Testing the type of an if condition (line 110)
    if_condition_408362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 13), result_eq_408361)
    # Assigning a type to the variable 'if_condition_408362' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 13), 'if_condition_408362', if_condition_408362)
    # SSA begins for if statement (line 110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 111)
    work_408363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 111)
    slice2_408364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 17), 'slice2')
    # Getting the type of 'work' (line 111)
    work_408365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___408366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), work_408365, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_408367 = invoke(stypy.reporting.localization.Localization(__file__, 111, 12), getitem___408366, slice2_408364)
    
    # Getting the type of 'sclr2' (line 111)
    sclr2_408368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 28), 'sclr2')
    # Applying the binary operator '*=' (line 111)
    result_imul_408369 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 12), '*=', subscript_call_result_408367, sclr2_408368)
    # Getting the type of 'work' (line 111)
    work_408370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'work')
    # Getting the type of 'slice2' (line 111)
    slice2_408371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 17), 'slice2')
    # Storing an element on a container (line 111)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 12), work_408370, (slice2_408371, result_imul_408369))
    
    
    # Getting the type of 'work' (line 112)
    work_408372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 112)
    slice2_408373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'slice2')
    # Getting the type of 'work' (line 112)
    work_408374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___408375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), work_408374, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_408376 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), getitem___408375, slice2_408373)
    
    # Getting the type of 'sclr1' (line 112)
    sclr1_408377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'sclr1')
    
    # Call to matvec(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 112)
    slice1_408379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 46), 'slice1', False)
    # Getting the type of 'work' (line 112)
    work_408380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 41), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___408381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 41), work_408380, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_408382 = invoke(stypy.reporting.localization.Localization(__file__, 112, 41), getitem___408381, slice1_408379)
    
    # Processing the call keyword arguments (line 112)
    kwargs_408383 = {}
    # Getting the type of 'matvec' (line 112)
    matvec_408378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 34), 'matvec', False)
    # Calling matvec(args, kwargs) (line 112)
    matvec_call_result_408384 = invoke(stypy.reporting.localization.Localization(__file__, 112, 34), matvec_408378, *[subscript_call_result_408382], **kwargs_408383)
    
    # Applying the binary operator '*' (line 112)
    result_mul_408385 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 28), '*', sclr1_408377, matvec_call_result_408384)
    
    # Applying the binary operator '+=' (line 112)
    result_iadd_408386 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 12), '+=', subscript_call_result_408376, result_mul_408385)
    # Getting the type of 'work' (line 112)
    work_408387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'work')
    # Getting the type of 'slice2' (line 112)
    slice2_408388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'slice2')
    # Storing an element on a container (line 112)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 12), work_408387, (slice2_408388, result_iadd_408386))
    
    # SSA branch for the else part of an if statement (line 110)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 113)
    ijob_408389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 14), 'ijob')
    int_408390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 22), 'int')
    # Applying the binary operator '==' (line 113)
    result_eq_408391 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 14), '==', ijob_408389, int_408390)
    
    # Testing the type of an if condition (line 113)
    if_condition_408392 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 13), result_eq_408391)
    # Assigning a type to the variable 'if_condition_408392' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'if_condition_408392', if_condition_408392)
    # SSA begins for if statement (line 113)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 114)
    work_408393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 114)
    slice2_408394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'slice2')
    # Getting the type of 'work' (line 114)
    work_408395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 114)
    getitem___408396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), work_408395, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 114)
    subscript_call_result_408397 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), getitem___408396, slice2_408394)
    
    # Getting the type of 'sclr2' (line 114)
    sclr2_408398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'sclr2')
    # Applying the binary operator '*=' (line 114)
    result_imul_408399 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 12), '*=', subscript_call_result_408397, sclr2_408398)
    # Getting the type of 'work' (line 114)
    work_408400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'work')
    # Getting the type of 'slice2' (line 114)
    slice2_408401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'slice2')
    # Storing an element on a container (line 114)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 12), work_408400, (slice2_408401, result_imul_408399))
    
    
    # Getting the type of 'work' (line 115)
    work_408402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 115)
    slice2_408403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'slice2')
    # Getting the type of 'work' (line 115)
    work_408404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___408405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), work_408404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_408406 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), getitem___408405, slice2_408403)
    
    # Getting the type of 'sclr1' (line 115)
    sclr1_408407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'sclr1')
    
    # Call to rmatvec(...): (line 115)
    # Processing the call arguments (line 115)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 115)
    slice1_408409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 47), 'slice1', False)
    # Getting the type of 'work' (line 115)
    work_408410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___408411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 42), work_408410, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_408412 = invoke(stypy.reporting.localization.Localization(__file__, 115, 42), getitem___408411, slice1_408409)
    
    # Processing the call keyword arguments (line 115)
    kwargs_408413 = {}
    # Getting the type of 'rmatvec' (line 115)
    rmatvec_408408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 34), 'rmatvec', False)
    # Calling rmatvec(args, kwargs) (line 115)
    rmatvec_call_result_408414 = invoke(stypy.reporting.localization.Localization(__file__, 115, 34), rmatvec_408408, *[subscript_call_result_408412], **kwargs_408413)
    
    # Applying the binary operator '*' (line 115)
    result_mul_408415 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 28), '*', sclr1_408407, rmatvec_call_result_408414)
    
    # Applying the binary operator '+=' (line 115)
    result_iadd_408416 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 12), '+=', subscript_call_result_408406, result_mul_408415)
    # Getting the type of 'work' (line 115)
    work_408417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'work')
    # Getting the type of 'slice2' (line 115)
    slice2_408418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'slice2')
    # Storing an element on a container (line 115)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 12), work_408417, (slice2_408418, result_iadd_408416))
    
    # SSA branch for the else part of an if statement (line 113)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 116)
    ijob_408419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 14), 'ijob')
    int_408420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 22), 'int')
    # Applying the binary operator '==' (line 116)
    result_eq_408421 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 14), '==', ijob_408419, int_408420)
    
    # Testing the type of an if condition (line 116)
    if_condition_408422 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 13), result_eq_408421)
    # Assigning a type to the variable 'if_condition_408422' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 13), 'if_condition_408422', if_condition_408422)
    # SSA begins for if statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 117):
    
    # Assigning a Call to a Subscript (line 117):
    
    # Call to psolve(...): (line 117)
    # Processing the call arguments (line 117)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 117)
    slice2_408424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 39), 'slice2', False)
    # Getting the type of 'work' (line 117)
    work_408425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 34), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___408426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 34), work_408425, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_408427 = invoke(stypy.reporting.localization.Localization(__file__, 117, 34), getitem___408426, slice2_408424)
    
    # Processing the call keyword arguments (line 117)
    kwargs_408428 = {}
    # Getting the type of 'psolve' (line 117)
    psolve_408423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'psolve', False)
    # Calling psolve(args, kwargs) (line 117)
    psolve_call_result_408429 = invoke(stypy.reporting.localization.Localization(__file__, 117, 27), psolve_408423, *[subscript_call_result_408427], **kwargs_408428)
    
    # Getting the type of 'work' (line 117)
    work_408430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'work')
    # Getting the type of 'slice1' (line 117)
    slice1_408431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'slice1')
    # Storing an element on a container (line 117)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 12), work_408430, (slice1_408431, psolve_call_result_408429))
    # SSA branch for the else part of an if statement (line 116)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 118)
    ijob_408432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'ijob')
    int_408433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 22), 'int')
    # Applying the binary operator '==' (line 118)
    result_eq_408434 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 14), '==', ijob_408432, int_408433)
    
    # Testing the type of an if condition (line 118)
    if_condition_408435 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 13), result_eq_408434)
    # Assigning a type to the variable 'if_condition_408435' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 13), 'if_condition_408435', if_condition_408435)
    # SSA begins for if statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 119):
    
    # Assigning a Call to a Subscript (line 119):
    
    # Call to rpsolve(...): (line 119)
    # Processing the call arguments (line 119)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 119)
    slice2_408437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 40), 'slice2', False)
    # Getting the type of 'work' (line 119)
    work_408438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___408439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 35), work_408438, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_408440 = invoke(stypy.reporting.localization.Localization(__file__, 119, 35), getitem___408439, slice2_408437)
    
    # Processing the call keyword arguments (line 119)
    kwargs_408441 = {}
    # Getting the type of 'rpsolve' (line 119)
    rpsolve_408436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'rpsolve', False)
    # Calling rpsolve(args, kwargs) (line 119)
    rpsolve_call_result_408442 = invoke(stypy.reporting.localization.Localization(__file__, 119, 27), rpsolve_408436, *[subscript_call_result_408440], **kwargs_408441)
    
    # Getting the type of 'work' (line 119)
    work_408443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'work')
    # Getting the type of 'slice1' (line 119)
    slice1_408444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 17), 'slice1')
    # Storing an element on a container (line 119)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 12), work_408443, (slice1_408444, rpsolve_call_result_408442))
    # SSA branch for the else part of an if statement (line 118)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 120)
    ijob_408445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), 'ijob')
    int_408446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 22), 'int')
    # Applying the binary operator '==' (line 120)
    result_eq_408447 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 14), '==', ijob_408445, int_408446)
    
    # Testing the type of an if condition (line 120)
    if_condition_408448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 13), result_eq_408447)
    # Assigning a type to the variable 'if_condition_408448' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 13), 'if_condition_408448', if_condition_408448)
    # SSA begins for if statement (line 120)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 121)
    work_408449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 121)
    slice2_408450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'slice2')
    # Getting the type of 'work' (line 121)
    work_408451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___408452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), work_408451, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_408453 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), getitem___408452, slice2_408450)
    
    # Getting the type of 'sclr2' (line 121)
    sclr2_408454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'sclr2')
    # Applying the binary operator '*=' (line 121)
    result_imul_408455 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 12), '*=', subscript_call_result_408453, sclr2_408454)
    # Getting the type of 'work' (line 121)
    work_408456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'work')
    # Getting the type of 'slice2' (line 121)
    slice2_408457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'slice2')
    # Storing an element on a container (line 121)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 12), work_408456, (slice2_408457, result_imul_408455))
    
    
    # Getting the type of 'work' (line 122)
    work_408458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 122)
    slice2_408459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'slice2')
    # Getting the type of 'work' (line 122)
    work_408460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 122)
    getitem___408461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), work_408460, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 122)
    subscript_call_result_408462 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), getitem___408461, slice2_408459)
    
    # Getting the type of 'sclr1' (line 122)
    sclr1_408463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'sclr1')
    
    # Call to matvec(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'x' (line 122)
    x_408465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 41), 'x', False)
    # Processing the call keyword arguments (line 122)
    kwargs_408466 = {}
    # Getting the type of 'matvec' (line 122)
    matvec_408464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 34), 'matvec', False)
    # Calling matvec(args, kwargs) (line 122)
    matvec_call_result_408467 = invoke(stypy.reporting.localization.Localization(__file__, 122, 34), matvec_408464, *[x_408465], **kwargs_408466)
    
    # Applying the binary operator '*' (line 122)
    result_mul_408468 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 28), '*', sclr1_408463, matvec_call_result_408467)
    
    # Applying the binary operator '+=' (line 122)
    result_iadd_408469 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 12), '+=', subscript_call_result_408462, result_mul_408468)
    # Getting the type of 'work' (line 122)
    work_408470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'work')
    # Getting the type of 'slice2' (line 122)
    slice2_408471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'slice2')
    # Storing an element on a container (line 122)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 12), work_408470, (slice2_408471, result_iadd_408469))
    
    # SSA branch for the else part of an if statement (line 120)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 123)
    ijob_408472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), 'ijob')
    int_408473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 22), 'int')
    # Applying the binary operator '==' (line 123)
    result_eq_408474 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 14), '==', ijob_408472, int_408473)
    
    # Testing the type of an if condition (line 123)
    if_condition_408475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 13), result_eq_408474)
    # Assigning a type to the variable 'if_condition_408475' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 13), 'if_condition_408475', if_condition_408475)
    # SSA begins for if statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ftflag' (line 124)
    ftflag_408476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'ftflag')
    # Testing the type of an if condition (line 124)
    if_condition_408477 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 12), ftflag_408476)
    # Assigning a type to the variable 'if_condition_408477' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'if_condition_408477', if_condition_408477)
    # SSA begins for if statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 125):
    
    # Assigning a Num to a Name (line 125):
    int_408478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 23), 'int')
    # Assigning a type to the variable 'info' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'info', int_408478)
    
    # Assigning a Name to a Name (line 126):
    
    # Assigning a Name to a Name (line 126):
    # Getting the type of 'False' (line 126)
    False_408479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 25), 'False')
    # Assigning a type to the variable 'ftflag' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'ftflag', False_408479)
    # SSA join for if statement (line 124)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 127):
    
    # Assigning a Subscript to a Name (line 127):
    
    # Obtaining the type of the subscript
    int_408480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 12), 'int')
    
    # Call to stoptest(...): (line 127)
    # Processing the call arguments (line 127)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 127)
    slice1_408482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 47), 'slice1', False)
    # Getting the type of 'work' (line 127)
    work_408483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___408484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 42), work_408483, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_408485 = invoke(stypy.reporting.localization.Localization(__file__, 127, 42), getitem___408484, slice1_408482)
    
    # Getting the type of 'b' (line 127)
    b_408486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 127)
    bnrm2_408487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 127)
    tol_408488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 66), 'tol', False)
    # Getting the type of 'info' (line 127)
    info_408489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 71), 'info', False)
    # Processing the call keyword arguments (line 127)
    kwargs_408490 = {}
    # Getting the type of 'stoptest' (line 127)
    stoptest_408481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 127)
    stoptest_call_result_408491 = invoke(stypy.reporting.localization.Localization(__file__, 127, 33), stoptest_408481, *[subscript_call_result_408485, b_408486, bnrm2_408487, tol_408488, info_408489], **kwargs_408490)
    
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___408492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), stoptest_call_result_408491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_408493 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), getitem___408492, int_408480)
    
    # Assigning a type to the variable 'tuple_var_assignment_407905' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'tuple_var_assignment_407905', subscript_call_result_408493)
    
    # Assigning a Subscript to a Name (line 127):
    
    # Obtaining the type of the subscript
    int_408494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 12), 'int')
    
    # Call to stoptest(...): (line 127)
    # Processing the call arguments (line 127)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 127)
    slice1_408496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 47), 'slice1', False)
    # Getting the type of 'work' (line 127)
    work_408497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___408498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 42), work_408497, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_408499 = invoke(stypy.reporting.localization.Localization(__file__, 127, 42), getitem___408498, slice1_408496)
    
    # Getting the type of 'b' (line 127)
    b_408500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 127)
    bnrm2_408501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 127)
    tol_408502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 66), 'tol', False)
    # Getting the type of 'info' (line 127)
    info_408503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 71), 'info', False)
    # Processing the call keyword arguments (line 127)
    kwargs_408504 = {}
    # Getting the type of 'stoptest' (line 127)
    stoptest_408495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 127)
    stoptest_call_result_408505 = invoke(stypy.reporting.localization.Localization(__file__, 127, 33), stoptest_408495, *[subscript_call_result_408499, b_408500, bnrm2_408501, tol_408502, info_408503], **kwargs_408504)
    
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___408506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), stoptest_call_result_408505, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_408507 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), getitem___408506, int_408494)
    
    # Assigning a type to the variable 'tuple_var_assignment_407906' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'tuple_var_assignment_407906', subscript_call_result_408507)
    
    # Assigning a Subscript to a Name (line 127):
    
    # Obtaining the type of the subscript
    int_408508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 12), 'int')
    
    # Call to stoptest(...): (line 127)
    # Processing the call arguments (line 127)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 127)
    slice1_408510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 47), 'slice1', False)
    # Getting the type of 'work' (line 127)
    work_408511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___408512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 42), work_408511, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_408513 = invoke(stypy.reporting.localization.Localization(__file__, 127, 42), getitem___408512, slice1_408510)
    
    # Getting the type of 'b' (line 127)
    b_408514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 127)
    bnrm2_408515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 127)
    tol_408516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 66), 'tol', False)
    # Getting the type of 'info' (line 127)
    info_408517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 71), 'info', False)
    # Processing the call keyword arguments (line 127)
    kwargs_408518 = {}
    # Getting the type of 'stoptest' (line 127)
    stoptest_408509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 127)
    stoptest_call_result_408519 = invoke(stypy.reporting.localization.Localization(__file__, 127, 33), stoptest_408509, *[subscript_call_result_408513, b_408514, bnrm2_408515, tol_408516, info_408517], **kwargs_408518)
    
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___408520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), stoptest_call_result_408519, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_408521 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), getitem___408520, int_408508)
    
    # Assigning a type to the variable 'tuple_var_assignment_407907' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'tuple_var_assignment_407907', subscript_call_result_408521)
    
    # Assigning a Name to a Name (line 127):
    # Getting the type of 'tuple_var_assignment_407905' (line 127)
    tuple_var_assignment_407905_408522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'tuple_var_assignment_407905')
    # Assigning a type to the variable 'bnrm2' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'bnrm2', tuple_var_assignment_407905_408522)
    
    # Assigning a Name to a Name (line 127):
    # Getting the type of 'tuple_var_assignment_407906' (line 127)
    tuple_var_assignment_407906_408523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'tuple_var_assignment_407906')
    # Assigning a type to the variable 'resid' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'resid', tuple_var_assignment_407906_408523)
    
    # Assigning a Name to a Name (line 127):
    # Getting the type of 'tuple_var_assignment_407907' (line 127)
    tuple_var_assignment_407907_408524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'tuple_var_assignment_407907')
    # Assigning a type to the variable 'info' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'info', tuple_var_assignment_407907_408524)
    # SSA join for if statement (line 123)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 120)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 118)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 116)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 113)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 110)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 106)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 128):
    
    # Assigning a Num to a Name (line 128):
    int_408525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 15), 'int')
    # Assigning a type to the variable 'ijob' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'ijob', int_408525)
    # SSA join for while statement (line 98)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'info' (line 130)
    info_408526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 7), 'info')
    int_408527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 14), 'int')
    # Applying the binary operator '>' (line 130)
    result_gt_408528 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 7), '>', info_408526, int_408527)
    
    
    # Getting the type of 'iter_' (line 130)
    iter__408529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'iter_')
    # Getting the type of 'maxiter' (line 130)
    maxiter_408530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 29), 'maxiter')
    # Applying the binary operator '==' (line 130)
    result_eq_408531 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 20), '==', iter__408529, maxiter_408530)
    
    # Applying the binary operator 'and' (line 130)
    result_and_keyword_408532 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 7), 'and', result_gt_408528, result_eq_408531)
    
    # Getting the type of 'resid' (line 130)
    resid_408533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 41), 'resid')
    # Getting the type of 'tol' (line 130)
    tol_408534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 49), 'tol')
    # Applying the binary operator '>' (line 130)
    result_gt_408535 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 41), '>', resid_408533, tol_408534)
    
    # Applying the binary operator 'and' (line 130)
    result_and_keyword_408536 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 7), 'and', result_and_keyword_408532, result_gt_408535)
    
    # Testing the type of an if condition (line 130)
    if_condition_408537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 4), result_and_keyword_408536)
    # Assigning a type to the variable 'if_condition_408537' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'if_condition_408537', if_condition_408537)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 132):
    
    # Assigning a Name to a Name (line 132):
    # Getting the type of 'iter_' (line 132)
    iter__408538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'iter_')
    # Assigning a type to the variable 'info' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'info', iter__408538)
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 134)
    tuple_408539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 134)
    # Adding element type (line 134)
    
    # Call to postprocess(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'x' (line 134)
    x_408541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'x', False)
    # Processing the call keyword arguments (line 134)
    kwargs_408542 = {}
    # Getting the type of 'postprocess' (line 134)
    postprocess_408540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'postprocess', False)
    # Calling postprocess(args, kwargs) (line 134)
    postprocess_call_result_408543 = invoke(stypy.reporting.localization.Localization(__file__, 134, 11), postprocess_408540, *[x_408541], **kwargs_408542)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 11), tuple_408539, postprocess_call_result_408543)
    # Adding element type (line 134)
    # Getting the type of 'info' (line 134)
    info_408544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 27), 'info')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 11), tuple_408539, info_408544)
    
    # Assigning a type to the variable 'stypy_return_type' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type', tuple_408539)
    
    # ################# End of 'bicg(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bicg' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_408545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_408545)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bicg'
    return stypy_return_type_408545

# Assigning a type to the variable 'bicg' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'bicg', bicg)

@norecursion
def bicgstab(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 141)
    None_408546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 22), 'None')
    float_408547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 32), 'float')
    # Getting the type of 'None' (line 141)
    None_408548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 46), 'None')
    # Getting the type of 'None' (line 141)
    None_408549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 54), 'None')
    # Getting the type of 'None' (line 141)
    None_408550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 69), 'None')
    defaults = [None_408546, float_408547, None_408548, None_408549, None_408550]
    # Create a new context for function 'bicgstab'
    module_type_store = module_type_store.open_function_context('bicgstab', 137, 0, False)
    
    # Passed parameters checking function
    bicgstab.stypy_localization = localization
    bicgstab.stypy_type_of_self = None
    bicgstab.stypy_type_store = module_type_store
    bicgstab.stypy_function_name = 'bicgstab'
    bicgstab.stypy_param_names_list = ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback']
    bicgstab.stypy_varargs_param_name = None
    bicgstab.stypy_kwargs_param_name = None
    bicgstab.stypy_call_defaults = defaults
    bicgstab.stypy_call_varargs = varargs
    bicgstab.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bicgstab', ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bicgstab', localization, ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bicgstab(...)' code ##################

    
    # Assigning a Call to a Tuple (line 142):
    
    # Assigning a Subscript to a Name (line 142):
    
    # Obtaining the type of the subscript
    int_408551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 4), 'int')
    
    # Call to make_system(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'A' (line 142)
    A_408553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'A', False)
    # Getting the type of 'M' (line 142)
    M_408554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 45), 'M', False)
    # Getting the type of 'x0' (line 142)
    x0_408555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 48), 'x0', False)
    # Getting the type of 'b' (line 142)
    b_408556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 52), 'b', False)
    # Processing the call keyword arguments (line 142)
    kwargs_408557 = {}
    # Getting the type of 'make_system' (line 142)
    make_system_408552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 142)
    make_system_call_result_408558 = invoke(stypy.reporting.localization.Localization(__file__, 142, 30), make_system_408552, *[A_408553, M_408554, x0_408555, b_408556], **kwargs_408557)
    
    # Obtaining the member '__getitem__' of a type (line 142)
    getitem___408559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 4), make_system_call_result_408558, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 142)
    subscript_call_result_408560 = invoke(stypy.reporting.localization.Localization(__file__, 142, 4), getitem___408559, int_408551)
    
    # Assigning a type to the variable 'tuple_var_assignment_407908' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_407908', subscript_call_result_408560)
    
    # Assigning a Subscript to a Name (line 142):
    
    # Obtaining the type of the subscript
    int_408561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 4), 'int')
    
    # Call to make_system(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'A' (line 142)
    A_408563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'A', False)
    # Getting the type of 'M' (line 142)
    M_408564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 45), 'M', False)
    # Getting the type of 'x0' (line 142)
    x0_408565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 48), 'x0', False)
    # Getting the type of 'b' (line 142)
    b_408566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 52), 'b', False)
    # Processing the call keyword arguments (line 142)
    kwargs_408567 = {}
    # Getting the type of 'make_system' (line 142)
    make_system_408562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 142)
    make_system_call_result_408568 = invoke(stypy.reporting.localization.Localization(__file__, 142, 30), make_system_408562, *[A_408563, M_408564, x0_408565, b_408566], **kwargs_408567)
    
    # Obtaining the member '__getitem__' of a type (line 142)
    getitem___408569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 4), make_system_call_result_408568, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 142)
    subscript_call_result_408570 = invoke(stypy.reporting.localization.Localization(__file__, 142, 4), getitem___408569, int_408561)
    
    # Assigning a type to the variable 'tuple_var_assignment_407909' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_407909', subscript_call_result_408570)
    
    # Assigning a Subscript to a Name (line 142):
    
    # Obtaining the type of the subscript
    int_408571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 4), 'int')
    
    # Call to make_system(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'A' (line 142)
    A_408573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'A', False)
    # Getting the type of 'M' (line 142)
    M_408574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 45), 'M', False)
    # Getting the type of 'x0' (line 142)
    x0_408575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 48), 'x0', False)
    # Getting the type of 'b' (line 142)
    b_408576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 52), 'b', False)
    # Processing the call keyword arguments (line 142)
    kwargs_408577 = {}
    # Getting the type of 'make_system' (line 142)
    make_system_408572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 142)
    make_system_call_result_408578 = invoke(stypy.reporting.localization.Localization(__file__, 142, 30), make_system_408572, *[A_408573, M_408574, x0_408575, b_408576], **kwargs_408577)
    
    # Obtaining the member '__getitem__' of a type (line 142)
    getitem___408579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 4), make_system_call_result_408578, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 142)
    subscript_call_result_408580 = invoke(stypy.reporting.localization.Localization(__file__, 142, 4), getitem___408579, int_408571)
    
    # Assigning a type to the variable 'tuple_var_assignment_407910' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_407910', subscript_call_result_408580)
    
    # Assigning a Subscript to a Name (line 142):
    
    # Obtaining the type of the subscript
    int_408581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 4), 'int')
    
    # Call to make_system(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'A' (line 142)
    A_408583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'A', False)
    # Getting the type of 'M' (line 142)
    M_408584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 45), 'M', False)
    # Getting the type of 'x0' (line 142)
    x0_408585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 48), 'x0', False)
    # Getting the type of 'b' (line 142)
    b_408586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 52), 'b', False)
    # Processing the call keyword arguments (line 142)
    kwargs_408587 = {}
    # Getting the type of 'make_system' (line 142)
    make_system_408582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 142)
    make_system_call_result_408588 = invoke(stypy.reporting.localization.Localization(__file__, 142, 30), make_system_408582, *[A_408583, M_408584, x0_408585, b_408586], **kwargs_408587)
    
    # Obtaining the member '__getitem__' of a type (line 142)
    getitem___408589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 4), make_system_call_result_408588, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 142)
    subscript_call_result_408590 = invoke(stypy.reporting.localization.Localization(__file__, 142, 4), getitem___408589, int_408581)
    
    # Assigning a type to the variable 'tuple_var_assignment_407911' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_407911', subscript_call_result_408590)
    
    # Assigning a Subscript to a Name (line 142):
    
    # Obtaining the type of the subscript
    int_408591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 4), 'int')
    
    # Call to make_system(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'A' (line 142)
    A_408593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'A', False)
    # Getting the type of 'M' (line 142)
    M_408594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 45), 'M', False)
    # Getting the type of 'x0' (line 142)
    x0_408595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 48), 'x0', False)
    # Getting the type of 'b' (line 142)
    b_408596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 52), 'b', False)
    # Processing the call keyword arguments (line 142)
    kwargs_408597 = {}
    # Getting the type of 'make_system' (line 142)
    make_system_408592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 142)
    make_system_call_result_408598 = invoke(stypy.reporting.localization.Localization(__file__, 142, 30), make_system_408592, *[A_408593, M_408594, x0_408595, b_408596], **kwargs_408597)
    
    # Obtaining the member '__getitem__' of a type (line 142)
    getitem___408599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 4), make_system_call_result_408598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 142)
    subscript_call_result_408600 = invoke(stypy.reporting.localization.Localization(__file__, 142, 4), getitem___408599, int_408591)
    
    # Assigning a type to the variable 'tuple_var_assignment_407912' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_407912', subscript_call_result_408600)
    
    # Assigning a Name to a Name (line 142):
    # Getting the type of 'tuple_var_assignment_407908' (line 142)
    tuple_var_assignment_407908_408601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_407908')
    # Assigning a type to the variable 'A' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'A', tuple_var_assignment_407908_408601)
    
    # Assigning a Name to a Name (line 142):
    # Getting the type of 'tuple_var_assignment_407909' (line 142)
    tuple_var_assignment_407909_408602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_407909')
    # Assigning a type to the variable 'M' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 7), 'M', tuple_var_assignment_407909_408602)
    
    # Assigning a Name to a Name (line 142):
    # Getting the type of 'tuple_var_assignment_407910' (line 142)
    tuple_var_assignment_407910_408603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_407910')
    # Assigning a type to the variable 'x' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 10), 'x', tuple_var_assignment_407910_408603)
    
    # Assigning a Name to a Name (line 142):
    # Getting the type of 'tuple_var_assignment_407911' (line 142)
    tuple_var_assignment_407911_408604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_407911')
    # Assigning a type to the variable 'b' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 13), 'b', tuple_var_assignment_407911_408604)
    
    # Assigning a Name to a Name (line 142):
    # Getting the type of 'tuple_var_assignment_407912' (line 142)
    tuple_var_assignment_407912_408605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_407912')
    # Assigning a type to the variable 'postprocess' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'postprocess', tuple_var_assignment_407912_408605)
    
    # Assigning a Call to a Name (line 144):
    
    # Assigning a Call to a Name (line 144):
    
    # Call to len(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'b' (line 144)
    b_408607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'b', False)
    # Processing the call keyword arguments (line 144)
    kwargs_408608 = {}
    # Getting the type of 'len' (line 144)
    len_408606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'len', False)
    # Calling len(args, kwargs) (line 144)
    len_call_result_408609 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), len_408606, *[b_408607], **kwargs_408608)
    
    # Assigning a type to the variable 'n' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'n', len_call_result_408609)
    
    # Type idiom detected: calculating its left and rigth part (line 145)
    # Getting the type of 'maxiter' (line 145)
    maxiter_408610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 7), 'maxiter')
    # Getting the type of 'None' (line 145)
    None_408611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 18), 'None')
    
    (may_be_408612, more_types_in_union_408613) = may_be_none(maxiter_408610, None_408611)

    if may_be_408612:

        if more_types_in_union_408613:
            # Runtime conditional SSA (line 145)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 146):
        
        # Assigning a BinOp to a Name (line 146):
        # Getting the type of 'n' (line 146)
        n_408614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'n')
        int_408615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 20), 'int')
        # Applying the binary operator '*' (line 146)
        result_mul_408616 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 18), '*', n_408614, int_408615)
        
        # Assigning a type to the variable 'maxiter' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'maxiter', result_mul_408616)

        if more_types_in_union_408613:
            # SSA join for if statement (line 145)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 148):
    
    # Assigning a Attribute to a Name (line 148):
    # Getting the type of 'A' (line 148)
    A_408617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 13), 'A')
    # Obtaining the member 'matvec' of a type (line 148)
    matvec_408618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 13), A_408617, 'matvec')
    # Assigning a type to the variable 'matvec' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'matvec', matvec_408618)
    
    # Assigning a Attribute to a Name (line 149):
    
    # Assigning a Attribute to a Name (line 149):
    # Getting the type of 'M' (line 149)
    M_408619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 13), 'M')
    # Obtaining the member 'matvec' of a type (line 149)
    matvec_408620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 13), M_408619, 'matvec')
    # Assigning a type to the variable 'psolve' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'psolve', matvec_408620)
    
    # Assigning a Subscript to a Name (line 150):
    
    # Assigning a Subscript to a Name (line 150):
    
    # Obtaining the type of the subscript
    # Getting the type of 'x' (line 150)
    x_408621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'x')
    # Obtaining the member 'dtype' of a type (line 150)
    dtype_408622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 21), x_408621, 'dtype')
    # Obtaining the member 'char' of a type (line 150)
    char_408623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 21), dtype_408622, 'char')
    # Getting the type of '_type_conv' (line 150)
    _type_conv_408624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 10), '_type_conv')
    # Obtaining the member '__getitem__' of a type (line 150)
    getitem___408625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 10), _type_conv_408624, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 150)
    subscript_call_result_408626 = invoke(stypy.reporting.localization.Localization(__file__, 150, 10), getitem___408625, char_408623)
    
    # Assigning a type to the variable 'ltr' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'ltr', subscript_call_result_408626)
    
    # Assigning a Call to a Name (line 151):
    
    # Assigning a Call to a Name (line 151):
    
    # Call to getattr(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of '_iterative' (line 151)
    _iterative_408628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), '_iterative', False)
    # Getting the type of 'ltr' (line 151)
    ltr_408629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 33), 'ltr', False)
    str_408630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 39), 'str', 'bicgstabrevcom')
    # Applying the binary operator '+' (line 151)
    result_add_408631 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 33), '+', ltr_408629, str_408630)
    
    # Processing the call keyword arguments (line 151)
    kwargs_408632 = {}
    # Getting the type of 'getattr' (line 151)
    getattr_408627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'getattr', False)
    # Calling getattr(args, kwargs) (line 151)
    getattr_call_result_408633 = invoke(stypy.reporting.localization.Localization(__file__, 151, 13), getattr_408627, *[_iterative_408628, result_add_408631], **kwargs_408632)
    
    # Assigning a type to the variable 'revcom' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'revcom', getattr_call_result_408633)
    
    # Assigning a Call to a Name (line 152):
    
    # Assigning a Call to a Name (line 152):
    
    # Call to getattr(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of '_iterative' (line 152)
    _iterative_408635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), '_iterative', False)
    # Getting the type of 'ltr' (line 152)
    ltr_408636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 35), 'ltr', False)
    str_408637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 41), 'str', 'stoptest2')
    # Applying the binary operator '+' (line 152)
    result_add_408638 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 35), '+', ltr_408636, str_408637)
    
    # Processing the call keyword arguments (line 152)
    kwargs_408639 = {}
    # Getting the type of 'getattr' (line 152)
    getattr_408634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'getattr', False)
    # Calling getattr(args, kwargs) (line 152)
    getattr_call_result_408640 = invoke(stypy.reporting.localization.Localization(__file__, 152, 15), getattr_408634, *[_iterative_408635, result_add_408638], **kwargs_408639)
    
    # Assigning a type to the variable 'stoptest' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stoptest', getattr_call_result_408640)
    
    # Assigning a Name to a Name (line 154):
    
    # Assigning a Name to a Name (line 154):
    # Getting the type of 'tol' (line 154)
    tol_408641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'tol')
    # Assigning a type to the variable 'resid' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'resid', tol_408641)
    
    # Assigning a Num to a Name (line 155):
    
    # Assigning a Num to a Name (line 155):
    int_408642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 11), 'int')
    # Assigning a type to the variable 'ndx1' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'ndx1', int_408642)
    
    # Assigning a Num to a Name (line 156):
    
    # Assigning a Num to a Name (line 156):
    int_408643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 11), 'int')
    # Assigning a type to the variable 'ndx2' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'ndx2', int_408643)
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to _aligned_zeros(...): (line 158)
    # Processing the call arguments (line 158)
    int_408645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 26), 'int')
    # Getting the type of 'n' (line 158)
    n_408646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'n', False)
    # Applying the binary operator '*' (line 158)
    result_mul_408647 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 26), '*', int_408645, n_408646)
    
    # Processing the call keyword arguments (line 158)
    # Getting the type of 'x' (line 158)
    x_408648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 36), 'x', False)
    # Obtaining the member 'dtype' of a type (line 158)
    dtype_408649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 36), x_408648, 'dtype')
    keyword_408650 = dtype_408649
    kwargs_408651 = {'dtype': keyword_408650}
    # Getting the type of '_aligned_zeros' (line 158)
    _aligned_zeros_408644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), '_aligned_zeros', False)
    # Calling _aligned_zeros(args, kwargs) (line 158)
    _aligned_zeros_call_result_408652 = invoke(stypy.reporting.localization.Localization(__file__, 158, 11), _aligned_zeros_408644, *[result_mul_408647], **kwargs_408651)
    
    # Assigning a type to the variable 'work' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'work', _aligned_zeros_call_result_408652)
    
    # Assigning a Num to a Name (line 159):
    
    # Assigning a Num to a Name (line 159):
    int_408653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 11), 'int')
    # Assigning a type to the variable 'ijob' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'ijob', int_408653)
    
    # Assigning a Num to a Name (line 160):
    
    # Assigning a Num to a Name (line 160):
    int_408654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 11), 'int')
    # Assigning a type to the variable 'info' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'info', int_408654)
    
    # Assigning a Name to a Name (line 161):
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'True' (line 161)
    True_408655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), 'True')
    # Assigning a type to the variable 'ftflag' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'ftflag', True_408655)
    
    # Assigning a Num to a Name (line 162):
    
    # Assigning a Num to a Name (line 162):
    float_408656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 12), 'float')
    # Assigning a type to the variable 'bnrm2' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'bnrm2', float_408656)
    
    # Assigning a Name to a Name (line 163):
    
    # Assigning a Name to a Name (line 163):
    # Getting the type of 'maxiter' (line 163)
    maxiter_408657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'maxiter')
    # Assigning a type to the variable 'iter_' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'iter_', maxiter_408657)
    
    # Getting the type of 'True' (line 164)
    True_408658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 10), 'True')
    # Testing the type of an if condition (line 164)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 4), True_408658)
    # SSA begins for while statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Name to a Name (line 165):
    
    # Assigning a Name to a Name (line 165):
    # Getting the type of 'iter_' (line 165)
    iter__408659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), 'iter_')
    # Assigning a type to the variable 'olditer' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'olditer', iter__408659)
    
    # Assigning a Call to a Tuple (line 166):
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    int_408660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
    
    # Call to revcom(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'b' (line 167)
    b_408662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'b', False)
    # Getting the type of 'x' (line 167)
    x_408663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'x', False)
    # Getting the type of 'work' (line 167)
    work_408664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'work', False)
    # Getting the type of 'iter_' (line 167)
    iter__408665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'iter_', False)
    # Getting the type of 'resid' (line 167)
    resid_408666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 37), 'resid', False)
    # Getting the type of 'info' (line 167)
    info_408667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'info', False)
    # Getting the type of 'ndx1' (line 167)
    ndx1_408668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 167)
    ndx2_408669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 167)
    ijob_408670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 62), 'ijob', False)
    # Processing the call keyword arguments (line 167)
    kwargs_408671 = {}
    # Getting the type of 'revcom' (line 167)
    revcom_408661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 167)
    revcom_call_result_408672 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), revcom_408661, *[b_408662, x_408663, work_408664, iter__408665, resid_408666, info_408667, ndx1_408668, ndx2_408669, ijob_408670], **kwargs_408671)
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___408673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), revcom_call_result_408672, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_408674 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___408673, int_408660)
    
    # Assigning a type to the variable 'tuple_var_assignment_407913' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407913', subscript_call_result_408674)
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    int_408675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
    
    # Call to revcom(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'b' (line 167)
    b_408677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'b', False)
    # Getting the type of 'x' (line 167)
    x_408678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'x', False)
    # Getting the type of 'work' (line 167)
    work_408679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'work', False)
    # Getting the type of 'iter_' (line 167)
    iter__408680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'iter_', False)
    # Getting the type of 'resid' (line 167)
    resid_408681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 37), 'resid', False)
    # Getting the type of 'info' (line 167)
    info_408682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'info', False)
    # Getting the type of 'ndx1' (line 167)
    ndx1_408683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 167)
    ndx2_408684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 167)
    ijob_408685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 62), 'ijob', False)
    # Processing the call keyword arguments (line 167)
    kwargs_408686 = {}
    # Getting the type of 'revcom' (line 167)
    revcom_408676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 167)
    revcom_call_result_408687 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), revcom_408676, *[b_408677, x_408678, work_408679, iter__408680, resid_408681, info_408682, ndx1_408683, ndx2_408684, ijob_408685], **kwargs_408686)
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___408688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), revcom_call_result_408687, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_408689 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___408688, int_408675)
    
    # Assigning a type to the variable 'tuple_var_assignment_407914' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407914', subscript_call_result_408689)
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    int_408690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
    
    # Call to revcom(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'b' (line 167)
    b_408692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'b', False)
    # Getting the type of 'x' (line 167)
    x_408693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'x', False)
    # Getting the type of 'work' (line 167)
    work_408694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'work', False)
    # Getting the type of 'iter_' (line 167)
    iter__408695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'iter_', False)
    # Getting the type of 'resid' (line 167)
    resid_408696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 37), 'resid', False)
    # Getting the type of 'info' (line 167)
    info_408697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'info', False)
    # Getting the type of 'ndx1' (line 167)
    ndx1_408698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 167)
    ndx2_408699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 167)
    ijob_408700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 62), 'ijob', False)
    # Processing the call keyword arguments (line 167)
    kwargs_408701 = {}
    # Getting the type of 'revcom' (line 167)
    revcom_408691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 167)
    revcom_call_result_408702 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), revcom_408691, *[b_408692, x_408693, work_408694, iter__408695, resid_408696, info_408697, ndx1_408698, ndx2_408699, ijob_408700], **kwargs_408701)
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___408703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), revcom_call_result_408702, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_408704 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___408703, int_408690)
    
    # Assigning a type to the variable 'tuple_var_assignment_407915' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407915', subscript_call_result_408704)
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    int_408705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
    
    # Call to revcom(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'b' (line 167)
    b_408707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'b', False)
    # Getting the type of 'x' (line 167)
    x_408708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'x', False)
    # Getting the type of 'work' (line 167)
    work_408709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'work', False)
    # Getting the type of 'iter_' (line 167)
    iter__408710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'iter_', False)
    # Getting the type of 'resid' (line 167)
    resid_408711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 37), 'resid', False)
    # Getting the type of 'info' (line 167)
    info_408712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'info', False)
    # Getting the type of 'ndx1' (line 167)
    ndx1_408713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 167)
    ndx2_408714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 167)
    ijob_408715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 62), 'ijob', False)
    # Processing the call keyword arguments (line 167)
    kwargs_408716 = {}
    # Getting the type of 'revcom' (line 167)
    revcom_408706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 167)
    revcom_call_result_408717 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), revcom_408706, *[b_408707, x_408708, work_408709, iter__408710, resid_408711, info_408712, ndx1_408713, ndx2_408714, ijob_408715], **kwargs_408716)
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___408718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), revcom_call_result_408717, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_408719 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___408718, int_408705)
    
    # Assigning a type to the variable 'tuple_var_assignment_407916' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407916', subscript_call_result_408719)
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    int_408720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
    
    # Call to revcom(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'b' (line 167)
    b_408722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'b', False)
    # Getting the type of 'x' (line 167)
    x_408723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'x', False)
    # Getting the type of 'work' (line 167)
    work_408724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'work', False)
    # Getting the type of 'iter_' (line 167)
    iter__408725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'iter_', False)
    # Getting the type of 'resid' (line 167)
    resid_408726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 37), 'resid', False)
    # Getting the type of 'info' (line 167)
    info_408727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'info', False)
    # Getting the type of 'ndx1' (line 167)
    ndx1_408728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 167)
    ndx2_408729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 167)
    ijob_408730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 62), 'ijob', False)
    # Processing the call keyword arguments (line 167)
    kwargs_408731 = {}
    # Getting the type of 'revcom' (line 167)
    revcom_408721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 167)
    revcom_call_result_408732 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), revcom_408721, *[b_408722, x_408723, work_408724, iter__408725, resid_408726, info_408727, ndx1_408728, ndx2_408729, ijob_408730], **kwargs_408731)
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___408733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), revcom_call_result_408732, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_408734 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___408733, int_408720)
    
    # Assigning a type to the variable 'tuple_var_assignment_407917' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407917', subscript_call_result_408734)
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    int_408735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
    
    # Call to revcom(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'b' (line 167)
    b_408737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'b', False)
    # Getting the type of 'x' (line 167)
    x_408738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'x', False)
    # Getting the type of 'work' (line 167)
    work_408739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'work', False)
    # Getting the type of 'iter_' (line 167)
    iter__408740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'iter_', False)
    # Getting the type of 'resid' (line 167)
    resid_408741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 37), 'resid', False)
    # Getting the type of 'info' (line 167)
    info_408742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'info', False)
    # Getting the type of 'ndx1' (line 167)
    ndx1_408743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 167)
    ndx2_408744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 167)
    ijob_408745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 62), 'ijob', False)
    # Processing the call keyword arguments (line 167)
    kwargs_408746 = {}
    # Getting the type of 'revcom' (line 167)
    revcom_408736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 167)
    revcom_call_result_408747 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), revcom_408736, *[b_408737, x_408738, work_408739, iter__408740, resid_408741, info_408742, ndx1_408743, ndx2_408744, ijob_408745], **kwargs_408746)
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___408748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), revcom_call_result_408747, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_408749 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___408748, int_408735)
    
    # Assigning a type to the variable 'tuple_var_assignment_407918' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407918', subscript_call_result_408749)
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    int_408750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
    
    # Call to revcom(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'b' (line 167)
    b_408752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'b', False)
    # Getting the type of 'x' (line 167)
    x_408753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'x', False)
    # Getting the type of 'work' (line 167)
    work_408754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'work', False)
    # Getting the type of 'iter_' (line 167)
    iter__408755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'iter_', False)
    # Getting the type of 'resid' (line 167)
    resid_408756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 37), 'resid', False)
    # Getting the type of 'info' (line 167)
    info_408757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'info', False)
    # Getting the type of 'ndx1' (line 167)
    ndx1_408758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 167)
    ndx2_408759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 167)
    ijob_408760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 62), 'ijob', False)
    # Processing the call keyword arguments (line 167)
    kwargs_408761 = {}
    # Getting the type of 'revcom' (line 167)
    revcom_408751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 167)
    revcom_call_result_408762 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), revcom_408751, *[b_408752, x_408753, work_408754, iter__408755, resid_408756, info_408757, ndx1_408758, ndx2_408759, ijob_408760], **kwargs_408761)
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___408763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), revcom_call_result_408762, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_408764 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___408763, int_408750)
    
    # Assigning a type to the variable 'tuple_var_assignment_407919' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407919', subscript_call_result_408764)
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    int_408765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
    
    # Call to revcom(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'b' (line 167)
    b_408767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'b', False)
    # Getting the type of 'x' (line 167)
    x_408768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'x', False)
    # Getting the type of 'work' (line 167)
    work_408769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'work', False)
    # Getting the type of 'iter_' (line 167)
    iter__408770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'iter_', False)
    # Getting the type of 'resid' (line 167)
    resid_408771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 37), 'resid', False)
    # Getting the type of 'info' (line 167)
    info_408772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'info', False)
    # Getting the type of 'ndx1' (line 167)
    ndx1_408773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 167)
    ndx2_408774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 167)
    ijob_408775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 62), 'ijob', False)
    # Processing the call keyword arguments (line 167)
    kwargs_408776 = {}
    # Getting the type of 'revcom' (line 167)
    revcom_408766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 167)
    revcom_call_result_408777 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), revcom_408766, *[b_408767, x_408768, work_408769, iter__408770, resid_408771, info_408772, ndx1_408773, ndx2_408774, ijob_408775], **kwargs_408776)
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___408778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), revcom_call_result_408777, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_408779 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___408778, int_408765)
    
    # Assigning a type to the variable 'tuple_var_assignment_407920' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407920', subscript_call_result_408779)
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    int_408780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
    
    # Call to revcom(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'b' (line 167)
    b_408782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'b', False)
    # Getting the type of 'x' (line 167)
    x_408783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'x', False)
    # Getting the type of 'work' (line 167)
    work_408784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'work', False)
    # Getting the type of 'iter_' (line 167)
    iter__408785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'iter_', False)
    # Getting the type of 'resid' (line 167)
    resid_408786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 37), 'resid', False)
    # Getting the type of 'info' (line 167)
    info_408787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'info', False)
    # Getting the type of 'ndx1' (line 167)
    ndx1_408788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 167)
    ndx2_408789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 167)
    ijob_408790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 62), 'ijob', False)
    # Processing the call keyword arguments (line 167)
    kwargs_408791 = {}
    # Getting the type of 'revcom' (line 167)
    revcom_408781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 167)
    revcom_call_result_408792 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), revcom_408781, *[b_408782, x_408783, work_408784, iter__408785, resid_408786, info_408787, ndx1_408788, ndx2_408789, ijob_408790], **kwargs_408791)
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___408793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), revcom_call_result_408792, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_408794 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___408793, int_408780)
    
    # Assigning a type to the variable 'tuple_var_assignment_407921' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407921', subscript_call_result_408794)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'tuple_var_assignment_407913' (line 166)
    tuple_var_assignment_407913_408795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407913')
    # Assigning a type to the variable 'x' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'x', tuple_var_assignment_407913_408795)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'tuple_var_assignment_407914' (line 166)
    tuple_var_assignment_407914_408796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407914')
    # Assigning a type to the variable 'iter_' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), 'iter_', tuple_var_assignment_407914_408796)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'tuple_var_assignment_407915' (line 166)
    tuple_var_assignment_407915_408797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407915')
    # Assigning a type to the variable 'resid' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 18), 'resid', tuple_var_assignment_407915_408797)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'tuple_var_assignment_407916' (line 166)
    tuple_var_assignment_407916_408798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407916')
    # Assigning a type to the variable 'info' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 25), 'info', tuple_var_assignment_407916_408798)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'tuple_var_assignment_407917' (line 166)
    tuple_var_assignment_407917_408799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407917')
    # Assigning a type to the variable 'ndx1' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 31), 'ndx1', tuple_var_assignment_407917_408799)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'tuple_var_assignment_407918' (line 166)
    tuple_var_assignment_407918_408800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407918')
    # Assigning a type to the variable 'ndx2' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 37), 'ndx2', tuple_var_assignment_407918_408800)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'tuple_var_assignment_407919' (line 166)
    tuple_var_assignment_407919_408801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407919')
    # Assigning a type to the variable 'sclr1' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 43), 'sclr1', tuple_var_assignment_407919_408801)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'tuple_var_assignment_407920' (line 166)
    tuple_var_assignment_407920_408802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407920')
    # Assigning a type to the variable 'sclr2' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 50), 'sclr2', tuple_var_assignment_407920_408802)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'tuple_var_assignment_407921' (line 166)
    tuple_var_assignment_407921_408803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_407921')
    # Assigning a type to the variable 'ijob' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 57), 'ijob', tuple_var_assignment_407921_408803)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'callback' (line 168)
    callback_408804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'callback')
    # Getting the type of 'None' (line 168)
    None_408805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'None')
    # Applying the binary operator 'isnot' (line 168)
    result_is_not_408806 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 11), 'isnot', callback_408804, None_408805)
    
    
    # Getting the type of 'iter_' (line 168)
    iter__408807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 36), 'iter_')
    # Getting the type of 'olditer' (line 168)
    olditer_408808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 44), 'olditer')
    # Applying the binary operator '>' (line 168)
    result_gt_408809 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 36), '>', iter__408807, olditer_408808)
    
    # Applying the binary operator 'and' (line 168)
    result_and_keyword_408810 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 11), 'and', result_is_not_408806, result_gt_408809)
    
    # Testing the type of an if condition (line 168)
    if_condition_408811 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 8), result_and_keyword_408810)
    # Assigning a type to the variable 'if_condition_408811' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'if_condition_408811', if_condition_408811)
    # SSA begins for if statement (line 168)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to callback(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'x' (line 169)
    x_408813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'x', False)
    # Processing the call keyword arguments (line 169)
    kwargs_408814 = {}
    # Getting the type of 'callback' (line 169)
    callback_408812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'callback', False)
    # Calling callback(args, kwargs) (line 169)
    callback_call_result_408815 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), callback_408812, *[x_408813], **kwargs_408814)
    
    # SSA join for if statement (line 168)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 170):
    
    # Assigning a Call to a Name (line 170):
    
    # Call to slice(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'ndx1' (line 170)
    ndx1_408817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 23), 'ndx1', False)
    int_408818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 28), 'int')
    # Applying the binary operator '-' (line 170)
    result_sub_408819 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 23), '-', ndx1_408817, int_408818)
    
    # Getting the type of 'ndx1' (line 170)
    ndx1_408820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 31), 'ndx1', False)
    int_408821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 36), 'int')
    # Applying the binary operator '-' (line 170)
    result_sub_408822 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 31), '-', ndx1_408820, int_408821)
    
    # Getting the type of 'n' (line 170)
    n_408823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 38), 'n', False)
    # Applying the binary operator '+' (line 170)
    result_add_408824 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 37), '+', result_sub_408822, n_408823)
    
    # Processing the call keyword arguments (line 170)
    kwargs_408825 = {}
    # Getting the type of 'slice' (line 170)
    slice_408816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 170)
    slice_call_result_408826 = invoke(stypy.reporting.localization.Localization(__file__, 170, 17), slice_408816, *[result_sub_408819, result_add_408824], **kwargs_408825)
    
    # Assigning a type to the variable 'slice1' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'slice1', slice_call_result_408826)
    
    # Assigning a Call to a Name (line 171):
    
    # Assigning a Call to a Name (line 171):
    
    # Call to slice(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'ndx2' (line 171)
    ndx2_408828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 23), 'ndx2', False)
    int_408829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 28), 'int')
    # Applying the binary operator '-' (line 171)
    result_sub_408830 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 23), '-', ndx2_408828, int_408829)
    
    # Getting the type of 'ndx2' (line 171)
    ndx2_408831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 31), 'ndx2', False)
    int_408832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 36), 'int')
    # Applying the binary operator '-' (line 171)
    result_sub_408833 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 31), '-', ndx2_408831, int_408832)
    
    # Getting the type of 'n' (line 171)
    n_408834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 38), 'n', False)
    # Applying the binary operator '+' (line 171)
    result_add_408835 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 37), '+', result_sub_408833, n_408834)
    
    # Processing the call keyword arguments (line 171)
    kwargs_408836 = {}
    # Getting the type of 'slice' (line 171)
    slice_408827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 171)
    slice_call_result_408837 = invoke(stypy.reporting.localization.Localization(__file__, 171, 17), slice_408827, *[result_sub_408830, result_add_408835], **kwargs_408836)
    
    # Assigning a type to the variable 'slice2' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'slice2', slice_call_result_408837)
    
    
    # Getting the type of 'ijob' (line 172)
    ijob_408838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'ijob')
    int_408839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 20), 'int')
    # Applying the binary operator '==' (line 172)
    result_eq_408840 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 12), '==', ijob_408838, int_408839)
    
    # Testing the type of an if condition (line 172)
    if_condition_408841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), result_eq_408840)
    # Assigning a type to the variable 'if_condition_408841' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_408841', if_condition_408841)
    # SSA begins for if statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 173)
    # Getting the type of 'callback' (line 173)
    callback_408842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'callback')
    # Getting the type of 'None' (line 173)
    None_408843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 31), 'None')
    
    (may_be_408844, more_types_in_union_408845) = may_not_be_none(callback_408842, None_408843)

    if may_be_408844:

        if more_types_in_union_408845:
            # Runtime conditional SSA (line 173)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to callback(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'x' (line 174)
        x_408847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'x', False)
        # Processing the call keyword arguments (line 174)
        kwargs_408848 = {}
        # Getting the type of 'callback' (line 174)
        callback_408846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'callback', False)
        # Calling callback(args, kwargs) (line 174)
        callback_call_result_408849 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), callback_408846, *[x_408847], **kwargs_408848)
        

        if more_types_in_union_408845:
            # SSA join for if statement (line 173)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 172)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 176)
    ijob_408850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 14), 'ijob')
    int_408851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 22), 'int')
    # Applying the binary operator '==' (line 176)
    result_eq_408852 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 14), '==', ijob_408850, int_408851)
    
    # Testing the type of an if condition (line 176)
    if_condition_408853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 13), result_eq_408852)
    # Assigning a type to the variable 'if_condition_408853' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 13), 'if_condition_408853', if_condition_408853)
    # SSA begins for if statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 177)
    work_408854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 177)
    slice2_408855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'slice2')
    # Getting the type of 'work' (line 177)
    work_408856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 177)
    getitem___408857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), work_408856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 177)
    subscript_call_result_408858 = invoke(stypy.reporting.localization.Localization(__file__, 177, 12), getitem___408857, slice2_408855)
    
    # Getting the type of 'sclr2' (line 177)
    sclr2_408859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 28), 'sclr2')
    # Applying the binary operator '*=' (line 177)
    result_imul_408860 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 12), '*=', subscript_call_result_408858, sclr2_408859)
    # Getting the type of 'work' (line 177)
    work_408861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'work')
    # Getting the type of 'slice2' (line 177)
    slice2_408862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'slice2')
    # Storing an element on a container (line 177)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 12), work_408861, (slice2_408862, result_imul_408860))
    
    
    # Getting the type of 'work' (line 178)
    work_408863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 178)
    slice2_408864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'slice2')
    # Getting the type of 'work' (line 178)
    work_408865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___408866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), work_408865, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_408867 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), getitem___408866, slice2_408864)
    
    # Getting the type of 'sclr1' (line 178)
    sclr1_408868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'sclr1')
    
    # Call to matvec(...): (line 178)
    # Processing the call arguments (line 178)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 178)
    slice1_408870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 46), 'slice1', False)
    # Getting the type of 'work' (line 178)
    work_408871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 41), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___408872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 41), work_408871, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_408873 = invoke(stypy.reporting.localization.Localization(__file__, 178, 41), getitem___408872, slice1_408870)
    
    # Processing the call keyword arguments (line 178)
    kwargs_408874 = {}
    # Getting the type of 'matvec' (line 178)
    matvec_408869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 34), 'matvec', False)
    # Calling matvec(args, kwargs) (line 178)
    matvec_call_result_408875 = invoke(stypy.reporting.localization.Localization(__file__, 178, 34), matvec_408869, *[subscript_call_result_408873], **kwargs_408874)
    
    # Applying the binary operator '*' (line 178)
    result_mul_408876 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 28), '*', sclr1_408868, matvec_call_result_408875)
    
    # Applying the binary operator '+=' (line 178)
    result_iadd_408877 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 12), '+=', subscript_call_result_408867, result_mul_408876)
    # Getting the type of 'work' (line 178)
    work_408878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'work')
    # Getting the type of 'slice2' (line 178)
    slice2_408879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'slice2')
    # Storing an element on a container (line 178)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 12), work_408878, (slice2_408879, result_iadd_408877))
    
    # SSA branch for the else part of an if statement (line 176)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 179)
    ijob_408880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 14), 'ijob')
    int_408881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 22), 'int')
    # Applying the binary operator '==' (line 179)
    result_eq_408882 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 14), '==', ijob_408880, int_408881)
    
    # Testing the type of an if condition (line 179)
    if_condition_408883 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 13), result_eq_408882)
    # Assigning a type to the variable 'if_condition_408883' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), 'if_condition_408883', if_condition_408883)
    # SSA begins for if statement (line 179)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 180):
    
    # Assigning a Call to a Subscript (line 180):
    
    # Call to psolve(...): (line 180)
    # Processing the call arguments (line 180)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 180)
    slice2_408885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 39), 'slice2', False)
    # Getting the type of 'work' (line 180)
    work_408886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 34), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___408887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 34), work_408886, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_408888 = invoke(stypy.reporting.localization.Localization(__file__, 180, 34), getitem___408887, slice2_408885)
    
    # Processing the call keyword arguments (line 180)
    kwargs_408889 = {}
    # Getting the type of 'psolve' (line 180)
    psolve_408884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 27), 'psolve', False)
    # Calling psolve(args, kwargs) (line 180)
    psolve_call_result_408890 = invoke(stypy.reporting.localization.Localization(__file__, 180, 27), psolve_408884, *[subscript_call_result_408888], **kwargs_408889)
    
    # Getting the type of 'work' (line 180)
    work_408891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'work')
    # Getting the type of 'slice1' (line 180)
    slice1_408892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'slice1')
    # Storing an element on a container (line 180)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 12), work_408891, (slice1_408892, psolve_call_result_408890))
    # SSA branch for the else part of an if statement (line 179)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 181)
    ijob_408893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 14), 'ijob')
    int_408894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 22), 'int')
    # Applying the binary operator '==' (line 181)
    result_eq_408895 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 14), '==', ijob_408893, int_408894)
    
    # Testing the type of an if condition (line 181)
    if_condition_408896 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 13), result_eq_408895)
    # Assigning a type to the variable 'if_condition_408896' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 13), 'if_condition_408896', if_condition_408896)
    # SSA begins for if statement (line 181)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 182)
    work_408897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 182)
    slice2_408898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'slice2')
    # Getting the type of 'work' (line 182)
    work_408899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___408900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 12), work_408899, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_408901 = invoke(stypy.reporting.localization.Localization(__file__, 182, 12), getitem___408900, slice2_408898)
    
    # Getting the type of 'sclr2' (line 182)
    sclr2_408902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 28), 'sclr2')
    # Applying the binary operator '*=' (line 182)
    result_imul_408903 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 12), '*=', subscript_call_result_408901, sclr2_408902)
    # Getting the type of 'work' (line 182)
    work_408904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'work')
    # Getting the type of 'slice2' (line 182)
    slice2_408905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'slice2')
    # Storing an element on a container (line 182)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 12), work_408904, (slice2_408905, result_imul_408903))
    
    
    # Getting the type of 'work' (line 183)
    work_408906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 183)
    slice2_408907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), 'slice2')
    # Getting the type of 'work' (line 183)
    work_408908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___408909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), work_408908, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_408910 = invoke(stypy.reporting.localization.Localization(__file__, 183, 12), getitem___408909, slice2_408907)
    
    # Getting the type of 'sclr1' (line 183)
    sclr1_408911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'sclr1')
    
    # Call to matvec(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'x' (line 183)
    x_408913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 41), 'x', False)
    # Processing the call keyword arguments (line 183)
    kwargs_408914 = {}
    # Getting the type of 'matvec' (line 183)
    matvec_408912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 34), 'matvec', False)
    # Calling matvec(args, kwargs) (line 183)
    matvec_call_result_408915 = invoke(stypy.reporting.localization.Localization(__file__, 183, 34), matvec_408912, *[x_408913], **kwargs_408914)
    
    # Applying the binary operator '*' (line 183)
    result_mul_408916 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 28), '*', sclr1_408911, matvec_call_result_408915)
    
    # Applying the binary operator '+=' (line 183)
    result_iadd_408917 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 12), '+=', subscript_call_result_408910, result_mul_408916)
    # Getting the type of 'work' (line 183)
    work_408918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'work')
    # Getting the type of 'slice2' (line 183)
    slice2_408919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), 'slice2')
    # Storing an element on a container (line 183)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 12), work_408918, (slice2_408919, result_iadd_408917))
    
    # SSA branch for the else part of an if statement (line 181)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 184)
    ijob_408920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 14), 'ijob')
    int_408921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 22), 'int')
    # Applying the binary operator '==' (line 184)
    result_eq_408922 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 14), '==', ijob_408920, int_408921)
    
    # Testing the type of an if condition (line 184)
    if_condition_408923 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 13), result_eq_408922)
    # Assigning a type to the variable 'if_condition_408923' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 13), 'if_condition_408923', if_condition_408923)
    # SSA begins for if statement (line 184)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ftflag' (line 185)
    ftflag_408924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'ftflag')
    # Testing the type of an if condition (line 185)
    if_condition_408925 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 12), ftflag_408924)
    # Assigning a type to the variable 'if_condition_408925' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'if_condition_408925', if_condition_408925)
    # SSA begins for if statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 186):
    
    # Assigning a Num to a Name (line 186):
    int_408926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 23), 'int')
    # Assigning a type to the variable 'info' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'info', int_408926)
    
    # Assigning a Name to a Name (line 187):
    
    # Assigning a Name to a Name (line 187):
    # Getting the type of 'False' (line 187)
    False_408927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 25), 'False')
    # Assigning a type to the variable 'ftflag' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'ftflag', False_408927)
    # SSA join for if statement (line 185)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 188):
    
    # Assigning a Subscript to a Name (line 188):
    
    # Obtaining the type of the subscript
    int_408928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 12), 'int')
    
    # Call to stoptest(...): (line 188)
    # Processing the call arguments (line 188)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 188)
    slice1_408930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 47), 'slice1', False)
    # Getting the type of 'work' (line 188)
    work_408931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___408932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 42), work_408931, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_408933 = invoke(stypy.reporting.localization.Localization(__file__, 188, 42), getitem___408932, slice1_408930)
    
    # Getting the type of 'b' (line 188)
    b_408934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 188)
    bnrm2_408935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 188)
    tol_408936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 66), 'tol', False)
    # Getting the type of 'info' (line 188)
    info_408937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 71), 'info', False)
    # Processing the call keyword arguments (line 188)
    kwargs_408938 = {}
    # Getting the type of 'stoptest' (line 188)
    stoptest_408929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 188)
    stoptest_call_result_408939 = invoke(stypy.reporting.localization.Localization(__file__, 188, 33), stoptest_408929, *[subscript_call_result_408933, b_408934, bnrm2_408935, tol_408936, info_408937], **kwargs_408938)
    
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___408940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), stoptest_call_result_408939, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_408941 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), getitem___408940, int_408928)
    
    # Assigning a type to the variable 'tuple_var_assignment_407922' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'tuple_var_assignment_407922', subscript_call_result_408941)
    
    # Assigning a Subscript to a Name (line 188):
    
    # Obtaining the type of the subscript
    int_408942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 12), 'int')
    
    # Call to stoptest(...): (line 188)
    # Processing the call arguments (line 188)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 188)
    slice1_408944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 47), 'slice1', False)
    # Getting the type of 'work' (line 188)
    work_408945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___408946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 42), work_408945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_408947 = invoke(stypy.reporting.localization.Localization(__file__, 188, 42), getitem___408946, slice1_408944)
    
    # Getting the type of 'b' (line 188)
    b_408948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 188)
    bnrm2_408949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 188)
    tol_408950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 66), 'tol', False)
    # Getting the type of 'info' (line 188)
    info_408951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 71), 'info', False)
    # Processing the call keyword arguments (line 188)
    kwargs_408952 = {}
    # Getting the type of 'stoptest' (line 188)
    stoptest_408943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 188)
    stoptest_call_result_408953 = invoke(stypy.reporting.localization.Localization(__file__, 188, 33), stoptest_408943, *[subscript_call_result_408947, b_408948, bnrm2_408949, tol_408950, info_408951], **kwargs_408952)
    
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___408954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), stoptest_call_result_408953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_408955 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), getitem___408954, int_408942)
    
    # Assigning a type to the variable 'tuple_var_assignment_407923' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'tuple_var_assignment_407923', subscript_call_result_408955)
    
    # Assigning a Subscript to a Name (line 188):
    
    # Obtaining the type of the subscript
    int_408956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 12), 'int')
    
    # Call to stoptest(...): (line 188)
    # Processing the call arguments (line 188)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 188)
    slice1_408958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 47), 'slice1', False)
    # Getting the type of 'work' (line 188)
    work_408959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___408960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 42), work_408959, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_408961 = invoke(stypy.reporting.localization.Localization(__file__, 188, 42), getitem___408960, slice1_408958)
    
    # Getting the type of 'b' (line 188)
    b_408962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 188)
    bnrm2_408963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 188)
    tol_408964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 66), 'tol', False)
    # Getting the type of 'info' (line 188)
    info_408965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 71), 'info', False)
    # Processing the call keyword arguments (line 188)
    kwargs_408966 = {}
    # Getting the type of 'stoptest' (line 188)
    stoptest_408957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 188)
    stoptest_call_result_408967 = invoke(stypy.reporting.localization.Localization(__file__, 188, 33), stoptest_408957, *[subscript_call_result_408961, b_408962, bnrm2_408963, tol_408964, info_408965], **kwargs_408966)
    
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___408968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), stoptest_call_result_408967, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_408969 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), getitem___408968, int_408956)
    
    # Assigning a type to the variable 'tuple_var_assignment_407924' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'tuple_var_assignment_407924', subscript_call_result_408969)
    
    # Assigning a Name to a Name (line 188):
    # Getting the type of 'tuple_var_assignment_407922' (line 188)
    tuple_var_assignment_407922_408970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'tuple_var_assignment_407922')
    # Assigning a type to the variable 'bnrm2' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'bnrm2', tuple_var_assignment_407922_408970)
    
    # Assigning a Name to a Name (line 188):
    # Getting the type of 'tuple_var_assignment_407923' (line 188)
    tuple_var_assignment_407923_408971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'tuple_var_assignment_407923')
    # Assigning a type to the variable 'resid' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 19), 'resid', tuple_var_assignment_407923_408971)
    
    # Assigning a Name to a Name (line 188):
    # Getting the type of 'tuple_var_assignment_407924' (line 188)
    tuple_var_assignment_407924_408972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'tuple_var_assignment_407924')
    # Assigning a type to the variable 'info' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 26), 'info', tuple_var_assignment_407924_408972)
    # SSA join for if statement (line 184)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 181)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 179)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 176)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 189):
    
    # Assigning a Num to a Name (line 189):
    int_408973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 15), 'int')
    # Assigning a type to the variable 'ijob' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'ijob', int_408973)
    # SSA join for while statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'info' (line 191)
    info_408974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 7), 'info')
    int_408975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 14), 'int')
    # Applying the binary operator '>' (line 191)
    result_gt_408976 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 7), '>', info_408974, int_408975)
    
    
    # Getting the type of 'iter_' (line 191)
    iter__408977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), 'iter_')
    # Getting the type of 'maxiter' (line 191)
    maxiter_408978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 29), 'maxiter')
    # Applying the binary operator '==' (line 191)
    result_eq_408979 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 20), '==', iter__408977, maxiter_408978)
    
    # Applying the binary operator 'and' (line 191)
    result_and_keyword_408980 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 7), 'and', result_gt_408976, result_eq_408979)
    
    # Getting the type of 'resid' (line 191)
    resid_408981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 41), 'resid')
    # Getting the type of 'tol' (line 191)
    tol_408982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 49), 'tol')
    # Applying the binary operator '>' (line 191)
    result_gt_408983 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 41), '>', resid_408981, tol_408982)
    
    # Applying the binary operator 'and' (line 191)
    result_and_keyword_408984 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 7), 'and', result_and_keyword_408980, result_gt_408983)
    
    # Testing the type of an if condition (line 191)
    if_condition_408985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 4), result_and_keyword_408984)
    # Assigning a type to the variable 'if_condition_408985' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'if_condition_408985', if_condition_408985)
    # SSA begins for if statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 193):
    
    # Assigning a Name to a Name (line 193):
    # Getting the type of 'iter_' (line 193)
    iter__408986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'iter_')
    # Assigning a type to the variable 'info' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'info', iter__408986)
    # SSA join for if statement (line 191)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 195)
    tuple_408987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 195)
    # Adding element type (line 195)
    
    # Call to postprocess(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'x' (line 195)
    x_408989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 23), 'x', False)
    # Processing the call keyword arguments (line 195)
    kwargs_408990 = {}
    # Getting the type of 'postprocess' (line 195)
    postprocess_408988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 11), 'postprocess', False)
    # Calling postprocess(args, kwargs) (line 195)
    postprocess_call_result_408991 = invoke(stypy.reporting.localization.Localization(__file__, 195, 11), postprocess_408988, *[x_408989], **kwargs_408990)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 11), tuple_408987, postprocess_call_result_408991)
    # Adding element type (line 195)
    # Getting the type of 'info' (line 195)
    info_408992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 27), 'info')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 11), tuple_408987, info_408992)
    
    # Assigning a type to the variable 'stypy_return_type' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type', tuple_408987)
    
    # ################# End of 'bicgstab(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bicgstab' in the type store
    # Getting the type of 'stypy_return_type' (line 137)
    stypy_return_type_408993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_408993)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bicgstab'
    return stypy_return_type_408993

# Assigning a type to the variable 'bicgstab' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'bicgstab', bicgstab)

@norecursion
def cg(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 202)
    None_408994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'None')
    float_408995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 26), 'float')
    # Getting the type of 'None' (line 202)
    None_408996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 40), 'None')
    # Getting the type of 'None' (line 202)
    None_408997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 48), 'None')
    # Getting the type of 'None' (line 202)
    None_408998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 63), 'None')
    defaults = [None_408994, float_408995, None_408996, None_408997, None_408998]
    # Create a new context for function 'cg'
    module_type_store = module_type_store.open_function_context('cg', 198, 0, False)
    
    # Passed parameters checking function
    cg.stypy_localization = localization
    cg.stypy_type_of_self = None
    cg.stypy_type_store = module_type_store
    cg.stypy_function_name = 'cg'
    cg.stypy_param_names_list = ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback']
    cg.stypy_varargs_param_name = None
    cg.stypy_kwargs_param_name = None
    cg.stypy_call_defaults = defaults
    cg.stypy_call_varargs = varargs
    cg.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cg', ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cg', localization, ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cg(...)' code ##################

    
    # Assigning a Call to a Tuple (line 203):
    
    # Assigning a Subscript to a Name (line 203):
    
    # Obtaining the type of the subscript
    int_408999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 4), 'int')
    
    # Call to make_system(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'A' (line 203)
    A_409001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), 'A', False)
    # Getting the type of 'M' (line 203)
    M_409002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 45), 'M', False)
    # Getting the type of 'x0' (line 203)
    x0_409003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 48), 'x0', False)
    # Getting the type of 'b' (line 203)
    b_409004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 52), 'b', False)
    # Processing the call keyword arguments (line 203)
    kwargs_409005 = {}
    # Getting the type of 'make_system' (line 203)
    make_system_409000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 203)
    make_system_call_result_409006 = invoke(stypy.reporting.localization.Localization(__file__, 203, 30), make_system_409000, *[A_409001, M_409002, x0_409003, b_409004], **kwargs_409005)
    
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___409007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 4), make_system_call_result_409006, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_409008 = invoke(stypy.reporting.localization.Localization(__file__, 203, 4), getitem___409007, int_408999)
    
    # Assigning a type to the variable 'tuple_var_assignment_407925' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_407925', subscript_call_result_409008)
    
    # Assigning a Subscript to a Name (line 203):
    
    # Obtaining the type of the subscript
    int_409009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 4), 'int')
    
    # Call to make_system(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'A' (line 203)
    A_409011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), 'A', False)
    # Getting the type of 'M' (line 203)
    M_409012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 45), 'M', False)
    # Getting the type of 'x0' (line 203)
    x0_409013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 48), 'x0', False)
    # Getting the type of 'b' (line 203)
    b_409014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 52), 'b', False)
    # Processing the call keyword arguments (line 203)
    kwargs_409015 = {}
    # Getting the type of 'make_system' (line 203)
    make_system_409010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 203)
    make_system_call_result_409016 = invoke(stypy.reporting.localization.Localization(__file__, 203, 30), make_system_409010, *[A_409011, M_409012, x0_409013, b_409014], **kwargs_409015)
    
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___409017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 4), make_system_call_result_409016, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_409018 = invoke(stypy.reporting.localization.Localization(__file__, 203, 4), getitem___409017, int_409009)
    
    # Assigning a type to the variable 'tuple_var_assignment_407926' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_407926', subscript_call_result_409018)
    
    # Assigning a Subscript to a Name (line 203):
    
    # Obtaining the type of the subscript
    int_409019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 4), 'int')
    
    # Call to make_system(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'A' (line 203)
    A_409021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), 'A', False)
    # Getting the type of 'M' (line 203)
    M_409022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 45), 'M', False)
    # Getting the type of 'x0' (line 203)
    x0_409023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 48), 'x0', False)
    # Getting the type of 'b' (line 203)
    b_409024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 52), 'b', False)
    # Processing the call keyword arguments (line 203)
    kwargs_409025 = {}
    # Getting the type of 'make_system' (line 203)
    make_system_409020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 203)
    make_system_call_result_409026 = invoke(stypy.reporting.localization.Localization(__file__, 203, 30), make_system_409020, *[A_409021, M_409022, x0_409023, b_409024], **kwargs_409025)
    
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___409027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 4), make_system_call_result_409026, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_409028 = invoke(stypy.reporting.localization.Localization(__file__, 203, 4), getitem___409027, int_409019)
    
    # Assigning a type to the variable 'tuple_var_assignment_407927' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_407927', subscript_call_result_409028)
    
    # Assigning a Subscript to a Name (line 203):
    
    # Obtaining the type of the subscript
    int_409029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 4), 'int')
    
    # Call to make_system(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'A' (line 203)
    A_409031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), 'A', False)
    # Getting the type of 'M' (line 203)
    M_409032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 45), 'M', False)
    # Getting the type of 'x0' (line 203)
    x0_409033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 48), 'x0', False)
    # Getting the type of 'b' (line 203)
    b_409034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 52), 'b', False)
    # Processing the call keyword arguments (line 203)
    kwargs_409035 = {}
    # Getting the type of 'make_system' (line 203)
    make_system_409030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 203)
    make_system_call_result_409036 = invoke(stypy.reporting.localization.Localization(__file__, 203, 30), make_system_409030, *[A_409031, M_409032, x0_409033, b_409034], **kwargs_409035)
    
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___409037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 4), make_system_call_result_409036, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_409038 = invoke(stypy.reporting.localization.Localization(__file__, 203, 4), getitem___409037, int_409029)
    
    # Assigning a type to the variable 'tuple_var_assignment_407928' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_407928', subscript_call_result_409038)
    
    # Assigning a Subscript to a Name (line 203):
    
    # Obtaining the type of the subscript
    int_409039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 4), 'int')
    
    # Call to make_system(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'A' (line 203)
    A_409041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), 'A', False)
    # Getting the type of 'M' (line 203)
    M_409042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 45), 'M', False)
    # Getting the type of 'x0' (line 203)
    x0_409043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 48), 'x0', False)
    # Getting the type of 'b' (line 203)
    b_409044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 52), 'b', False)
    # Processing the call keyword arguments (line 203)
    kwargs_409045 = {}
    # Getting the type of 'make_system' (line 203)
    make_system_409040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 203)
    make_system_call_result_409046 = invoke(stypy.reporting.localization.Localization(__file__, 203, 30), make_system_409040, *[A_409041, M_409042, x0_409043, b_409044], **kwargs_409045)
    
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___409047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 4), make_system_call_result_409046, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_409048 = invoke(stypy.reporting.localization.Localization(__file__, 203, 4), getitem___409047, int_409039)
    
    # Assigning a type to the variable 'tuple_var_assignment_407929' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_407929', subscript_call_result_409048)
    
    # Assigning a Name to a Name (line 203):
    # Getting the type of 'tuple_var_assignment_407925' (line 203)
    tuple_var_assignment_407925_409049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_407925')
    # Assigning a type to the variable 'A' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'A', tuple_var_assignment_407925_409049)
    
    # Assigning a Name to a Name (line 203):
    # Getting the type of 'tuple_var_assignment_407926' (line 203)
    tuple_var_assignment_407926_409050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_407926')
    # Assigning a type to the variable 'M' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 7), 'M', tuple_var_assignment_407926_409050)
    
    # Assigning a Name to a Name (line 203):
    # Getting the type of 'tuple_var_assignment_407927' (line 203)
    tuple_var_assignment_407927_409051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_407927')
    # Assigning a type to the variable 'x' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 10), 'x', tuple_var_assignment_407927_409051)
    
    # Assigning a Name to a Name (line 203):
    # Getting the type of 'tuple_var_assignment_407928' (line 203)
    tuple_var_assignment_407928_409052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_407928')
    # Assigning a type to the variable 'b' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 13), 'b', tuple_var_assignment_407928_409052)
    
    # Assigning a Name to a Name (line 203):
    # Getting the type of 'tuple_var_assignment_407929' (line 203)
    tuple_var_assignment_407929_409053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_407929')
    # Assigning a type to the variable 'postprocess' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'postprocess', tuple_var_assignment_407929_409053)
    
    # Assigning a Call to a Name (line 205):
    
    # Assigning a Call to a Name (line 205):
    
    # Call to len(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'b' (line 205)
    b_409055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'b', False)
    # Processing the call keyword arguments (line 205)
    kwargs_409056 = {}
    # Getting the type of 'len' (line 205)
    len_409054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'len', False)
    # Calling len(args, kwargs) (line 205)
    len_call_result_409057 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), len_409054, *[b_409055], **kwargs_409056)
    
    # Assigning a type to the variable 'n' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'n', len_call_result_409057)
    
    # Type idiom detected: calculating its left and rigth part (line 206)
    # Getting the type of 'maxiter' (line 206)
    maxiter_409058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 7), 'maxiter')
    # Getting the type of 'None' (line 206)
    None_409059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 18), 'None')
    
    (may_be_409060, more_types_in_union_409061) = may_be_none(maxiter_409058, None_409059)

    if may_be_409060:

        if more_types_in_union_409061:
            # Runtime conditional SSA (line 206)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 207):
        
        # Assigning a BinOp to a Name (line 207):
        # Getting the type of 'n' (line 207)
        n_409062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 18), 'n')
        int_409063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 20), 'int')
        # Applying the binary operator '*' (line 207)
        result_mul_409064 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 18), '*', n_409062, int_409063)
        
        # Assigning a type to the variable 'maxiter' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'maxiter', result_mul_409064)

        if more_types_in_union_409061:
            # SSA join for if statement (line 206)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 209):
    
    # Assigning a Attribute to a Name (line 209):
    # Getting the type of 'A' (line 209)
    A_409065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), 'A')
    # Obtaining the member 'matvec' of a type (line 209)
    matvec_409066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 13), A_409065, 'matvec')
    # Assigning a type to the variable 'matvec' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'matvec', matvec_409066)
    
    # Assigning a Attribute to a Name (line 210):
    
    # Assigning a Attribute to a Name (line 210):
    # Getting the type of 'M' (line 210)
    M_409067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 13), 'M')
    # Obtaining the member 'matvec' of a type (line 210)
    matvec_409068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 13), M_409067, 'matvec')
    # Assigning a type to the variable 'psolve' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'psolve', matvec_409068)
    
    # Assigning a Subscript to a Name (line 211):
    
    # Assigning a Subscript to a Name (line 211):
    
    # Obtaining the type of the subscript
    # Getting the type of 'x' (line 211)
    x_409069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'x')
    # Obtaining the member 'dtype' of a type (line 211)
    dtype_409070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 21), x_409069, 'dtype')
    # Obtaining the member 'char' of a type (line 211)
    char_409071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 21), dtype_409070, 'char')
    # Getting the type of '_type_conv' (line 211)
    _type_conv_409072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 10), '_type_conv')
    # Obtaining the member '__getitem__' of a type (line 211)
    getitem___409073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 10), _type_conv_409072, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 211)
    subscript_call_result_409074 = invoke(stypy.reporting.localization.Localization(__file__, 211, 10), getitem___409073, char_409071)
    
    # Assigning a type to the variable 'ltr' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'ltr', subscript_call_result_409074)
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Call to getattr(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of '_iterative' (line 212)
    _iterative_409076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 21), '_iterative', False)
    # Getting the type of 'ltr' (line 212)
    ltr_409077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 33), 'ltr', False)
    str_409078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 39), 'str', 'cgrevcom')
    # Applying the binary operator '+' (line 212)
    result_add_409079 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 33), '+', ltr_409077, str_409078)
    
    # Processing the call keyword arguments (line 212)
    kwargs_409080 = {}
    # Getting the type of 'getattr' (line 212)
    getattr_409075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 13), 'getattr', False)
    # Calling getattr(args, kwargs) (line 212)
    getattr_call_result_409081 = invoke(stypy.reporting.localization.Localization(__file__, 212, 13), getattr_409075, *[_iterative_409076, result_add_409079], **kwargs_409080)
    
    # Assigning a type to the variable 'revcom' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'revcom', getattr_call_result_409081)
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to getattr(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of '_iterative' (line 213)
    _iterative_409083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 23), '_iterative', False)
    # Getting the type of 'ltr' (line 213)
    ltr_409084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 35), 'ltr', False)
    str_409085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 41), 'str', 'stoptest2')
    # Applying the binary operator '+' (line 213)
    result_add_409086 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 35), '+', ltr_409084, str_409085)
    
    # Processing the call keyword arguments (line 213)
    kwargs_409087 = {}
    # Getting the type of 'getattr' (line 213)
    getattr_409082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'getattr', False)
    # Calling getattr(args, kwargs) (line 213)
    getattr_call_result_409088 = invoke(stypy.reporting.localization.Localization(__file__, 213, 15), getattr_409082, *[_iterative_409083, result_add_409086], **kwargs_409087)
    
    # Assigning a type to the variable 'stoptest' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stoptest', getattr_call_result_409088)
    
    # Assigning a Name to a Name (line 215):
    
    # Assigning a Name to a Name (line 215):
    # Getting the type of 'tol' (line 215)
    tol_409089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'tol')
    # Assigning a type to the variable 'resid' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'resid', tol_409089)
    
    # Assigning a Num to a Name (line 216):
    
    # Assigning a Num to a Name (line 216):
    int_409090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 11), 'int')
    # Assigning a type to the variable 'ndx1' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'ndx1', int_409090)
    
    # Assigning a Num to a Name (line 217):
    
    # Assigning a Num to a Name (line 217):
    int_409091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 11), 'int')
    # Assigning a type to the variable 'ndx2' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'ndx2', int_409091)
    
    # Assigning a Call to a Name (line 219):
    
    # Assigning a Call to a Name (line 219):
    
    # Call to _aligned_zeros(...): (line 219)
    # Processing the call arguments (line 219)
    int_409093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 26), 'int')
    # Getting the type of 'n' (line 219)
    n_409094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'n', False)
    # Applying the binary operator '*' (line 219)
    result_mul_409095 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 26), '*', int_409093, n_409094)
    
    # Processing the call keyword arguments (line 219)
    # Getting the type of 'x' (line 219)
    x_409096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 36), 'x', False)
    # Obtaining the member 'dtype' of a type (line 219)
    dtype_409097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 36), x_409096, 'dtype')
    keyword_409098 = dtype_409097
    kwargs_409099 = {'dtype': keyword_409098}
    # Getting the type of '_aligned_zeros' (line 219)
    _aligned_zeros_409092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 11), '_aligned_zeros', False)
    # Calling _aligned_zeros(args, kwargs) (line 219)
    _aligned_zeros_call_result_409100 = invoke(stypy.reporting.localization.Localization(__file__, 219, 11), _aligned_zeros_409092, *[result_mul_409095], **kwargs_409099)
    
    # Assigning a type to the variable 'work' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'work', _aligned_zeros_call_result_409100)
    
    # Assigning a Num to a Name (line 220):
    
    # Assigning a Num to a Name (line 220):
    int_409101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 11), 'int')
    # Assigning a type to the variable 'ijob' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'ijob', int_409101)
    
    # Assigning a Num to a Name (line 221):
    
    # Assigning a Num to a Name (line 221):
    int_409102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 11), 'int')
    # Assigning a type to the variable 'info' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'info', int_409102)
    
    # Assigning a Name to a Name (line 222):
    
    # Assigning a Name to a Name (line 222):
    # Getting the type of 'True' (line 222)
    True_409103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 13), 'True')
    # Assigning a type to the variable 'ftflag' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'ftflag', True_409103)
    
    # Assigning a Num to a Name (line 223):
    
    # Assigning a Num to a Name (line 223):
    float_409104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 12), 'float')
    # Assigning a type to the variable 'bnrm2' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'bnrm2', float_409104)
    
    # Assigning a Name to a Name (line 224):
    
    # Assigning a Name to a Name (line 224):
    # Getting the type of 'maxiter' (line 224)
    maxiter_409105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'maxiter')
    # Assigning a type to the variable 'iter_' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'iter_', maxiter_409105)
    
    # Getting the type of 'True' (line 225)
    True_409106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 10), 'True')
    # Testing the type of an if condition (line 225)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 4), True_409106)
    # SSA begins for while statement (line 225)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Name to a Name (line 226):
    
    # Assigning a Name to a Name (line 226):
    # Getting the type of 'iter_' (line 226)
    iter__409107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 18), 'iter_')
    # Assigning a type to the variable 'olditer' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'olditer', iter__409107)
    
    # Assigning a Call to a Tuple (line 227):
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_409108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
    
    # Call to revcom(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'b' (line 228)
    b_409110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'b', False)
    # Getting the type of 'x' (line 228)
    x_409111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'x', False)
    # Getting the type of 'work' (line 228)
    work_409112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'work', False)
    # Getting the type of 'iter_' (line 228)
    iter__409113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'iter_', False)
    # Getting the type of 'resid' (line 228)
    resid_409114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 37), 'resid', False)
    # Getting the type of 'info' (line 228)
    info_409115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 44), 'info', False)
    # Getting the type of 'ndx1' (line 228)
    ndx1_409116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 228)
    ndx2_409117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 228)
    ijob_409118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 62), 'ijob', False)
    # Processing the call keyword arguments (line 228)
    kwargs_409119 = {}
    # Getting the type of 'revcom' (line 228)
    revcom_409109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 228)
    revcom_call_result_409120 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), revcom_409109, *[b_409110, x_409111, work_409112, iter__409113, resid_409114, info_409115, ndx1_409116, ndx2_409117, ijob_409118], **kwargs_409119)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___409121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), revcom_call_result_409120, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_409122 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___409121, int_409108)
    
    # Assigning a type to the variable 'tuple_var_assignment_407930' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407930', subscript_call_result_409122)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_409123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
    
    # Call to revcom(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'b' (line 228)
    b_409125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'b', False)
    # Getting the type of 'x' (line 228)
    x_409126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'x', False)
    # Getting the type of 'work' (line 228)
    work_409127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'work', False)
    # Getting the type of 'iter_' (line 228)
    iter__409128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'iter_', False)
    # Getting the type of 'resid' (line 228)
    resid_409129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 37), 'resid', False)
    # Getting the type of 'info' (line 228)
    info_409130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 44), 'info', False)
    # Getting the type of 'ndx1' (line 228)
    ndx1_409131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 228)
    ndx2_409132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 228)
    ijob_409133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 62), 'ijob', False)
    # Processing the call keyword arguments (line 228)
    kwargs_409134 = {}
    # Getting the type of 'revcom' (line 228)
    revcom_409124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 228)
    revcom_call_result_409135 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), revcom_409124, *[b_409125, x_409126, work_409127, iter__409128, resid_409129, info_409130, ndx1_409131, ndx2_409132, ijob_409133], **kwargs_409134)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___409136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), revcom_call_result_409135, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_409137 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___409136, int_409123)
    
    # Assigning a type to the variable 'tuple_var_assignment_407931' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407931', subscript_call_result_409137)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_409138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
    
    # Call to revcom(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'b' (line 228)
    b_409140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'b', False)
    # Getting the type of 'x' (line 228)
    x_409141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'x', False)
    # Getting the type of 'work' (line 228)
    work_409142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'work', False)
    # Getting the type of 'iter_' (line 228)
    iter__409143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'iter_', False)
    # Getting the type of 'resid' (line 228)
    resid_409144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 37), 'resid', False)
    # Getting the type of 'info' (line 228)
    info_409145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 44), 'info', False)
    # Getting the type of 'ndx1' (line 228)
    ndx1_409146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 228)
    ndx2_409147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 228)
    ijob_409148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 62), 'ijob', False)
    # Processing the call keyword arguments (line 228)
    kwargs_409149 = {}
    # Getting the type of 'revcom' (line 228)
    revcom_409139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 228)
    revcom_call_result_409150 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), revcom_409139, *[b_409140, x_409141, work_409142, iter__409143, resid_409144, info_409145, ndx1_409146, ndx2_409147, ijob_409148], **kwargs_409149)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___409151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), revcom_call_result_409150, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_409152 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___409151, int_409138)
    
    # Assigning a type to the variable 'tuple_var_assignment_407932' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407932', subscript_call_result_409152)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_409153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
    
    # Call to revcom(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'b' (line 228)
    b_409155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'b', False)
    # Getting the type of 'x' (line 228)
    x_409156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'x', False)
    # Getting the type of 'work' (line 228)
    work_409157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'work', False)
    # Getting the type of 'iter_' (line 228)
    iter__409158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'iter_', False)
    # Getting the type of 'resid' (line 228)
    resid_409159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 37), 'resid', False)
    # Getting the type of 'info' (line 228)
    info_409160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 44), 'info', False)
    # Getting the type of 'ndx1' (line 228)
    ndx1_409161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 228)
    ndx2_409162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 228)
    ijob_409163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 62), 'ijob', False)
    # Processing the call keyword arguments (line 228)
    kwargs_409164 = {}
    # Getting the type of 'revcom' (line 228)
    revcom_409154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 228)
    revcom_call_result_409165 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), revcom_409154, *[b_409155, x_409156, work_409157, iter__409158, resid_409159, info_409160, ndx1_409161, ndx2_409162, ijob_409163], **kwargs_409164)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___409166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), revcom_call_result_409165, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_409167 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___409166, int_409153)
    
    # Assigning a type to the variable 'tuple_var_assignment_407933' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407933', subscript_call_result_409167)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_409168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
    
    # Call to revcom(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'b' (line 228)
    b_409170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'b', False)
    # Getting the type of 'x' (line 228)
    x_409171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'x', False)
    # Getting the type of 'work' (line 228)
    work_409172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'work', False)
    # Getting the type of 'iter_' (line 228)
    iter__409173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'iter_', False)
    # Getting the type of 'resid' (line 228)
    resid_409174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 37), 'resid', False)
    # Getting the type of 'info' (line 228)
    info_409175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 44), 'info', False)
    # Getting the type of 'ndx1' (line 228)
    ndx1_409176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 228)
    ndx2_409177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 228)
    ijob_409178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 62), 'ijob', False)
    # Processing the call keyword arguments (line 228)
    kwargs_409179 = {}
    # Getting the type of 'revcom' (line 228)
    revcom_409169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 228)
    revcom_call_result_409180 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), revcom_409169, *[b_409170, x_409171, work_409172, iter__409173, resid_409174, info_409175, ndx1_409176, ndx2_409177, ijob_409178], **kwargs_409179)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___409181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), revcom_call_result_409180, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_409182 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___409181, int_409168)
    
    # Assigning a type to the variable 'tuple_var_assignment_407934' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407934', subscript_call_result_409182)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_409183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
    
    # Call to revcom(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'b' (line 228)
    b_409185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'b', False)
    # Getting the type of 'x' (line 228)
    x_409186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'x', False)
    # Getting the type of 'work' (line 228)
    work_409187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'work', False)
    # Getting the type of 'iter_' (line 228)
    iter__409188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'iter_', False)
    # Getting the type of 'resid' (line 228)
    resid_409189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 37), 'resid', False)
    # Getting the type of 'info' (line 228)
    info_409190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 44), 'info', False)
    # Getting the type of 'ndx1' (line 228)
    ndx1_409191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 228)
    ndx2_409192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 228)
    ijob_409193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 62), 'ijob', False)
    # Processing the call keyword arguments (line 228)
    kwargs_409194 = {}
    # Getting the type of 'revcom' (line 228)
    revcom_409184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 228)
    revcom_call_result_409195 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), revcom_409184, *[b_409185, x_409186, work_409187, iter__409188, resid_409189, info_409190, ndx1_409191, ndx2_409192, ijob_409193], **kwargs_409194)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___409196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), revcom_call_result_409195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_409197 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___409196, int_409183)
    
    # Assigning a type to the variable 'tuple_var_assignment_407935' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407935', subscript_call_result_409197)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_409198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
    
    # Call to revcom(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'b' (line 228)
    b_409200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'b', False)
    # Getting the type of 'x' (line 228)
    x_409201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'x', False)
    # Getting the type of 'work' (line 228)
    work_409202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'work', False)
    # Getting the type of 'iter_' (line 228)
    iter__409203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'iter_', False)
    # Getting the type of 'resid' (line 228)
    resid_409204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 37), 'resid', False)
    # Getting the type of 'info' (line 228)
    info_409205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 44), 'info', False)
    # Getting the type of 'ndx1' (line 228)
    ndx1_409206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 228)
    ndx2_409207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 228)
    ijob_409208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 62), 'ijob', False)
    # Processing the call keyword arguments (line 228)
    kwargs_409209 = {}
    # Getting the type of 'revcom' (line 228)
    revcom_409199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 228)
    revcom_call_result_409210 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), revcom_409199, *[b_409200, x_409201, work_409202, iter__409203, resid_409204, info_409205, ndx1_409206, ndx2_409207, ijob_409208], **kwargs_409209)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___409211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), revcom_call_result_409210, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_409212 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___409211, int_409198)
    
    # Assigning a type to the variable 'tuple_var_assignment_407936' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407936', subscript_call_result_409212)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_409213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
    
    # Call to revcom(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'b' (line 228)
    b_409215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'b', False)
    # Getting the type of 'x' (line 228)
    x_409216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'x', False)
    # Getting the type of 'work' (line 228)
    work_409217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'work', False)
    # Getting the type of 'iter_' (line 228)
    iter__409218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'iter_', False)
    # Getting the type of 'resid' (line 228)
    resid_409219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 37), 'resid', False)
    # Getting the type of 'info' (line 228)
    info_409220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 44), 'info', False)
    # Getting the type of 'ndx1' (line 228)
    ndx1_409221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 228)
    ndx2_409222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 228)
    ijob_409223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 62), 'ijob', False)
    # Processing the call keyword arguments (line 228)
    kwargs_409224 = {}
    # Getting the type of 'revcom' (line 228)
    revcom_409214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 228)
    revcom_call_result_409225 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), revcom_409214, *[b_409215, x_409216, work_409217, iter__409218, resid_409219, info_409220, ndx1_409221, ndx2_409222, ijob_409223], **kwargs_409224)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___409226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), revcom_call_result_409225, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_409227 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___409226, int_409213)
    
    # Assigning a type to the variable 'tuple_var_assignment_407937' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407937', subscript_call_result_409227)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_409228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
    
    # Call to revcom(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'b' (line 228)
    b_409230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'b', False)
    # Getting the type of 'x' (line 228)
    x_409231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'x', False)
    # Getting the type of 'work' (line 228)
    work_409232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'work', False)
    # Getting the type of 'iter_' (line 228)
    iter__409233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'iter_', False)
    # Getting the type of 'resid' (line 228)
    resid_409234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 37), 'resid', False)
    # Getting the type of 'info' (line 228)
    info_409235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 44), 'info', False)
    # Getting the type of 'ndx1' (line 228)
    ndx1_409236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 228)
    ndx2_409237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 228)
    ijob_409238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 62), 'ijob', False)
    # Processing the call keyword arguments (line 228)
    kwargs_409239 = {}
    # Getting the type of 'revcom' (line 228)
    revcom_409229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 228)
    revcom_call_result_409240 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), revcom_409229, *[b_409230, x_409231, work_409232, iter__409233, resid_409234, info_409235, ndx1_409236, ndx2_409237, ijob_409238], **kwargs_409239)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___409241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), revcom_call_result_409240, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_409242 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___409241, int_409228)
    
    # Assigning a type to the variable 'tuple_var_assignment_407938' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407938', subscript_call_result_409242)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_407930' (line 227)
    tuple_var_assignment_407930_409243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407930')
    # Assigning a type to the variable 'x' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'x', tuple_var_assignment_407930_409243)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_407931' (line 227)
    tuple_var_assignment_407931_409244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407931')
    # Assigning a type to the variable 'iter_' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'iter_', tuple_var_assignment_407931_409244)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_407932' (line 227)
    tuple_var_assignment_407932_409245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407932')
    # Assigning a type to the variable 'resid' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 18), 'resid', tuple_var_assignment_407932_409245)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_407933' (line 227)
    tuple_var_assignment_407933_409246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407933')
    # Assigning a type to the variable 'info' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'info', tuple_var_assignment_407933_409246)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_407934' (line 227)
    tuple_var_assignment_407934_409247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407934')
    # Assigning a type to the variable 'ndx1' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 31), 'ndx1', tuple_var_assignment_407934_409247)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_407935' (line 227)
    tuple_var_assignment_407935_409248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407935')
    # Assigning a type to the variable 'ndx2' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 37), 'ndx2', tuple_var_assignment_407935_409248)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_407936' (line 227)
    tuple_var_assignment_407936_409249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407936')
    # Assigning a type to the variable 'sclr1' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 43), 'sclr1', tuple_var_assignment_407936_409249)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_407937' (line 227)
    tuple_var_assignment_407937_409250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407937')
    # Assigning a type to the variable 'sclr2' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 50), 'sclr2', tuple_var_assignment_407937_409250)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_407938' (line 227)
    tuple_var_assignment_407938_409251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_407938')
    # Assigning a type to the variable 'ijob' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 57), 'ijob', tuple_var_assignment_407938_409251)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'callback' (line 229)
    callback_409252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 11), 'callback')
    # Getting the type of 'None' (line 229)
    None_409253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'None')
    # Applying the binary operator 'isnot' (line 229)
    result_is_not_409254 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 11), 'isnot', callback_409252, None_409253)
    
    
    # Getting the type of 'iter_' (line 229)
    iter__409255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 36), 'iter_')
    # Getting the type of 'olditer' (line 229)
    olditer_409256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 44), 'olditer')
    # Applying the binary operator '>' (line 229)
    result_gt_409257 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 36), '>', iter__409255, olditer_409256)
    
    # Applying the binary operator 'and' (line 229)
    result_and_keyword_409258 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 11), 'and', result_is_not_409254, result_gt_409257)
    
    # Testing the type of an if condition (line 229)
    if_condition_409259 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 8), result_and_keyword_409258)
    # Assigning a type to the variable 'if_condition_409259' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'if_condition_409259', if_condition_409259)
    # SSA begins for if statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to callback(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'x' (line 230)
    x_409261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 'x', False)
    # Processing the call keyword arguments (line 230)
    kwargs_409262 = {}
    # Getting the type of 'callback' (line 230)
    callback_409260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'callback', False)
    # Calling callback(args, kwargs) (line 230)
    callback_call_result_409263 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), callback_409260, *[x_409261], **kwargs_409262)
    
    # SSA join for if statement (line 229)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 231):
    
    # Assigning a Call to a Name (line 231):
    
    # Call to slice(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'ndx1' (line 231)
    ndx1_409265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 'ndx1', False)
    int_409266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 28), 'int')
    # Applying the binary operator '-' (line 231)
    result_sub_409267 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 23), '-', ndx1_409265, int_409266)
    
    # Getting the type of 'ndx1' (line 231)
    ndx1_409268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 31), 'ndx1', False)
    int_409269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 36), 'int')
    # Applying the binary operator '-' (line 231)
    result_sub_409270 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 31), '-', ndx1_409268, int_409269)
    
    # Getting the type of 'n' (line 231)
    n_409271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 38), 'n', False)
    # Applying the binary operator '+' (line 231)
    result_add_409272 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 37), '+', result_sub_409270, n_409271)
    
    # Processing the call keyword arguments (line 231)
    kwargs_409273 = {}
    # Getting the type of 'slice' (line 231)
    slice_409264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 231)
    slice_call_result_409274 = invoke(stypy.reporting.localization.Localization(__file__, 231, 17), slice_409264, *[result_sub_409267, result_add_409272], **kwargs_409273)
    
    # Assigning a type to the variable 'slice1' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'slice1', slice_call_result_409274)
    
    # Assigning a Call to a Name (line 232):
    
    # Assigning a Call to a Name (line 232):
    
    # Call to slice(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'ndx2' (line 232)
    ndx2_409276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 'ndx2', False)
    int_409277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 28), 'int')
    # Applying the binary operator '-' (line 232)
    result_sub_409278 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 23), '-', ndx2_409276, int_409277)
    
    # Getting the type of 'ndx2' (line 232)
    ndx2_409279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 31), 'ndx2', False)
    int_409280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 36), 'int')
    # Applying the binary operator '-' (line 232)
    result_sub_409281 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 31), '-', ndx2_409279, int_409280)
    
    # Getting the type of 'n' (line 232)
    n_409282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 38), 'n', False)
    # Applying the binary operator '+' (line 232)
    result_add_409283 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 37), '+', result_sub_409281, n_409282)
    
    # Processing the call keyword arguments (line 232)
    kwargs_409284 = {}
    # Getting the type of 'slice' (line 232)
    slice_409275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 232)
    slice_call_result_409285 = invoke(stypy.reporting.localization.Localization(__file__, 232, 17), slice_409275, *[result_sub_409278, result_add_409283], **kwargs_409284)
    
    # Assigning a type to the variable 'slice2' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'slice2', slice_call_result_409285)
    
    
    # Getting the type of 'ijob' (line 233)
    ijob_409286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'ijob')
    int_409287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 20), 'int')
    # Applying the binary operator '==' (line 233)
    result_eq_409288 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 12), '==', ijob_409286, int_409287)
    
    # Testing the type of an if condition (line 233)
    if_condition_409289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 8), result_eq_409288)
    # Assigning a type to the variable 'if_condition_409289' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'if_condition_409289', if_condition_409289)
    # SSA begins for if statement (line 233)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 234)
    # Getting the type of 'callback' (line 234)
    callback_409290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'callback')
    # Getting the type of 'None' (line 234)
    None_409291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 31), 'None')
    
    (may_be_409292, more_types_in_union_409293) = may_not_be_none(callback_409290, None_409291)

    if may_be_409292:

        if more_types_in_union_409293:
            # Runtime conditional SSA (line 234)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to callback(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'x' (line 235)
        x_409295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 25), 'x', False)
        # Processing the call keyword arguments (line 235)
        kwargs_409296 = {}
        # Getting the type of 'callback' (line 235)
        callback_409294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'callback', False)
        # Calling callback(args, kwargs) (line 235)
        callback_call_result_409297 = invoke(stypy.reporting.localization.Localization(__file__, 235, 16), callback_409294, *[x_409295], **kwargs_409296)
        

        if more_types_in_union_409293:
            # SSA join for if statement (line 234)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 233)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 237)
    ijob_409298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 14), 'ijob')
    int_409299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 22), 'int')
    # Applying the binary operator '==' (line 237)
    result_eq_409300 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 14), '==', ijob_409298, int_409299)
    
    # Testing the type of an if condition (line 237)
    if_condition_409301 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 13), result_eq_409300)
    # Assigning a type to the variable 'if_condition_409301' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 13), 'if_condition_409301', if_condition_409301)
    # SSA begins for if statement (line 237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 238)
    work_409302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 238)
    slice2_409303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 17), 'slice2')
    # Getting the type of 'work' (line 238)
    work_409304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 238)
    getitem___409305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), work_409304, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 238)
    subscript_call_result_409306 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), getitem___409305, slice2_409303)
    
    # Getting the type of 'sclr2' (line 238)
    sclr2_409307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 28), 'sclr2')
    # Applying the binary operator '*=' (line 238)
    result_imul_409308 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 12), '*=', subscript_call_result_409306, sclr2_409307)
    # Getting the type of 'work' (line 238)
    work_409309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'work')
    # Getting the type of 'slice2' (line 238)
    slice2_409310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 17), 'slice2')
    # Storing an element on a container (line 238)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 12), work_409309, (slice2_409310, result_imul_409308))
    
    
    # Getting the type of 'work' (line 239)
    work_409311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 239)
    slice2_409312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 17), 'slice2')
    # Getting the type of 'work' (line 239)
    work_409313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___409314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), work_409313, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_409315 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), getitem___409314, slice2_409312)
    
    # Getting the type of 'sclr1' (line 239)
    sclr1_409316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 28), 'sclr1')
    
    # Call to matvec(...): (line 239)
    # Processing the call arguments (line 239)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 239)
    slice1_409318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 46), 'slice1', False)
    # Getting the type of 'work' (line 239)
    work_409319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 41), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___409320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 41), work_409319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_409321 = invoke(stypy.reporting.localization.Localization(__file__, 239, 41), getitem___409320, slice1_409318)
    
    # Processing the call keyword arguments (line 239)
    kwargs_409322 = {}
    # Getting the type of 'matvec' (line 239)
    matvec_409317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 34), 'matvec', False)
    # Calling matvec(args, kwargs) (line 239)
    matvec_call_result_409323 = invoke(stypy.reporting.localization.Localization(__file__, 239, 34), matvec_409317, *[subscript_call_result_409321], **kwargs_409322)
    
    # Applying the binary operator '*' (line 239)
    result_mul_409324 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 28), '*', sclr1_409316, matvec_call_result_409323)
    
    # Applying the binary operator '+=' (line 239)
    result_iadd_409325 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 12), '+=', subscript_call_result_409315, result_mul_409324)
    # Getting the type of 'work' (line 239)
    work_409326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'work')
    # Getting the type of 'slice2' (line 239)
    slice2_409327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 17), 'slice2')
    # Storing an element on a container (line 239)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 12), work_409326, (slice2_409327, result_iadd_409325))
    
    # SSA branch for the else part of an if statement (line 237)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 240)
    ijob_409328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 14), 'ijob')
    int_409329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 22), 'int')
    # Applying the binary operator '==' (line 240)
    result_eq_409330 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 14), '==', ijob_409328, int_409329)
    
    # Testing the type of an if condition (line 240)
    if_condition_409331 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 13), result_eq_409330)
    # Assigning a type to the variable 'if_condition_409331' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 13), 'if_condition_409331', if_condition_409331)
    # SSA begins for if statement (line 240)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 241):
    
    # Assigning a Call to a Subscript (line 241):
    
    # Call to psolve(...): (line 241)
    # Processing the call arguments (line 241)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 241)
    slice2_409333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 39), 'slice2', False)
    # Getting the type of 'work' (line 241)
    work_409334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 34), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___409335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 34), work_409334, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_409336 = invoke(stypy.reporting.localization.Localization(__file__, 241, 34), getitem___409335, slice2_409333)
    
    # Processing the call keyword arguments (line 241)
    kwargs_409337 = {}
    # Getting the type of 'psolve' (line 241)
    psolve_409332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 27), 'psolve', False)
    # Calling psolve(args, kwargs) (line 241)
    psolve_call_result_409338 = invoke(stypy.reporting.localization.Localization(__file__, 241, 27), psolve_409332, *[subscript_call_result_409336], **kwargs_409337)
    
    # Getting the type of 'work' (line 241)
    work_409339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'work')
    # Getting the type of 'slice1' (line 241)
    slice1_409340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'slice1')
    # Storing an element on a container (line 241)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 12), work_409339, (slice1_409340, psolve_call_result_409338))
    # SSA branch for the else part of an if statement (line 240)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 242)
    ijob_409341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 14), 'ijob')
    int_409342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 22), 'int')
    # Applying the binary operator '==' (line 242)
    result_eq_409343 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 14), '==', ijob_409341, int_409342)
    
    # Testing the type of an if condition (line 242)
    if_condition_409344 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 13), result_eq_409343)
    # Assigning a type to the variable 'if_condition_409344' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 13), 'if_condition_409344', if_condition_409344)
    # SSA begins for if statement (line 242)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 243)
    work_409345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 243)
    slice2_409346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 17), 'slice2')
    # Getting the type of 'work' (line 243)
    work_409347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___409348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), work_409347, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_409349 = invoke(stypy.reporting.localization.Localization(__file__, 243, 12), getitem___409348, slice2_409346)
    
    # Getting the type of 'sclr2' (line 243)
    sclr2_409350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 28), 'sclr2')
    # Applying the binary operator '*=' (line 243)
    result_imul_409351 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 12), '*=', subscript_call_result_409349, sclr2_409350)
    # Getting the type of 'work' (line 243)
    work_409352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'work')
    # Getting the type of 'slice2' (line 243)
    slice2_409353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 17), 'slice2')
    # Storing an element on a container (line 243)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 12), work_409352, (slice2_409353, result_imul_409351))
    
    
    # Getting the type of 'work' (line 244)
    work_409354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 244)
    slice2_409355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 17), 'slice2')
    # Getting the type of 'work' (line 244)
    work_409356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 244)
    getitem___409357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), work_409356, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 244)
    subscript_call_result_409358 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), getitem___409357, slice2_409355)
    
    # Getting the type of 'sclr1' (line 244)
    sclr1_409359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 28), 'sclr1')
    
    # Call to matvec(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'x' (line 244)
    x_409361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 41), 'x', False)
    # Processing the call keyword arguments (line 244)
    kwargs_409362 = {}
    # Getting the type of 'matvec' (line 244)
    matvec_409360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 34), 'matvec', False)
    # Calling matvec(args, kwargs) (line 244)
    matvec_call_result_409363 = invoke(stypy.reporting.localization.Localization(__file__, 244, 34), matvec_409360, *[x_409361], **kwargs_409362)
    
    # Applying the binary operator '*' (line 244)
    result_mul_409364 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 28), '*', sclr1_409359, matvec_call_result_409363)
    
    # Applying the binary operator '+=' (line 244)
    result_iadd_409365 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 12), '+=', subscript_call_result_409358, result_mul_409364)
    # Getting the type of 'work' (line 244)
    work_409366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'work')
    # Getting the type of 'slice2' (line 244)
    slice2_409367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 17), 'slice2')
    # Storing an element on a container (line 244)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 12), work_409366, (slice2_409367, result_iadd_409365))
    
    # SSA branch for the else part of an if statement (line 242)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 245)
    ijob_409368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 14), 'ijob')
    int_409369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 22), 'int')
    # Applying the binary operator '==' (line 245)
    result_eq_409370 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 14), '==', ijob_409368, int_409369)
    
    # Testing the type of an if condition (line 245)
    if_condition_409371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 13), result_eq_409370)
    # Assigning a type to the variable 'if_condition_409371' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 13), 'if_condition_409371', if_condition_409371)
    # SSA begins for if statement (line 245)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ftflag' (line 246)
    ftflag_409372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'ftflag')
    # Testing the type of an if condition (line 246)
    if_condition_409373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 12), ftflag_409372)
    # Assigning a type to the variable 'if_condition_409373' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'if_condition_409373', if_condition_409373)
    # SSA begins for if statement (line 246)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 247):
    
    # Assigning a Num to a Name (line 247):
    int_409374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 23), 'int')
    # Assigning a type to the variable 'info' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'info', int_409374)
    
    # Assigning a Name to a Name (line 248):
    
    # Assigning a Name to a Name (line 248):
    # Getting the type of 'False' (line 248)
    False_409375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 25), 'False')
    # Assigning a type to the variable 'ftflag' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'ftflag', False_409375)
    # SSA join for if statement (line 246)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 249):
    
    # Assigning a Subscript to a Name (line 249):
    
    # Obtaining the type of the subscript
    int_409376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 12), 'int')
    
    # Call to stoptest(...): (line 249)
    # Processing the call arguments (line 249)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 249)
    slice1_409378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 47), 'slice1', False)
    # Getting the type of 'work' (line 249)
    work_409379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___409380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 42), work_409379, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_409381 = invoke(stypy.reporting.localization.Localization(__file__, 249, 42), getitem___409380, slice1_409378)
    
    # Getting the type of 'b' (line 249)
    b_409382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 249)
    bnrm2_409383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 249)
    tol_409384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 66), 'tol', False)
    # Getting the type of 'info' (line 249)
    info_409385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 71), 'info', False)
    # Processing the call keyword arguments (line 249)
    kwargs_409386 = {}
    # Getting the type of 'stoptest' (line 249)
    stoptest_409377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 249)
    stoptest_call_result_409387 = invoke(stypy.reporting.localization.Localization(__file__, 249, 33), stoptest_409377, *[subscript_call_result_409381, b_409382, bnrm2_409383, tol_409384, info_409385], **kwargs_409386)
    
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___409388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 12), stoptest_call_result_409387, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_409389 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), getitem___409388, int_409376)
    
    # Assigning a type to the variable 'tuple_var_assignment_407939' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'tuple_var_assignment_407939', subscript_call_result_409389)
    
    # Assigning a Subscript to a Name (line 249):
    
    # Obtaining the type of the subscript
    int_409390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 12), 'int')
    
    # Call to stoptest(...): (line 249)
    # Processing the call arguments (line 249)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 249)
    slice1_409392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 47), 'slice1', False)
    # Getting the type of 'work' (line 249)
    work_409393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___409394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 42), work_409393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_409395 = invoke(stypy.reporting.localization.Localization(__file__, 249, 42), getitem___409394, slice1_409392)
    
    # Getting the type of 'b' (line 249)
    b_409396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 249)
    bnrm2_409397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 249)
    tol_409398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 66), 'tol', False)
    # Getting the type of 'info' (line 249)
    info_409399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 71), 'info', False)
    # Processing the call keyword arguments (line 249)
    kwargs_409400 = {}
    # Getting the type of 'stoptest' (line 249)
    stoptest_409391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 249)
    stoptest_call_result_409401 = invoke(stypy.reporting.localization.Localization(__file__, 249, 33), stoptest_409391, *[subscript_call_result_409395, b_409396, bnrm2_409397, tol_409398, info_409399], **kwargs_409400)
    
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___409402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 12), stoptest_call_result_409401, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_409403 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), getitem___409402, int_409390)
    
    # Assigning a type to the variable 'tuple_var_assignment_407940' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'tuple_var_assignment_407940', subscript_call_result_409403)
    
    # Assigning a Subscript to a Name (line 249):
    
    # Obtaining the type of the subscript
    int_409404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 12), 'int')
    
    # Call to stoptest(...): (line 249)
    # Processing the call arguments (line 249)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 249)
    slice1_409406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 47), 'slice1', False)
    # Getting the type of 'work' (line 249)
    work_409407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___409408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 42), work_409407, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_409409 = invoke(stypy.reporting.localization.Localization(__file__, 249, 42), getitem___409408, slice1_409406)
    
    # Getting the type of 'b' (line 249)
    b_409410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 249)
    bnrm2_409411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 249)
    tol_409412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 66), 'tol', False)
    # Getting the type of 'info' (line 249)
    info_409413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 71), 'info', False)
    # Processing the call keyword arguments (line 249)
    kwargs_409414 = {}
    # Getting the type of 'stoptest' (line 249)
    stoptest_409405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 249)
    stoptest_call_result_409415 = invoke(stypy.reporting.localization.Localization(__file__, 249, 33), stoptest_409405, *[subscript_call_result_409409, b_409410, bnrm2_409411, tol_409412, info_409413], **kwargs_409414)
    
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___409416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 12), stoptest_call_result_409415, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_409417 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), getitem___409416, int_409404)
    
    # Assigning a type to the variable 'tuple_var_assignment_407941' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'tuple_var_assignment_407941', subscript_call_result_409417)
    
    # Assigning a Name to a Name (line 249):
    # Getting the type of 'tuple_var_assignment_407939' (line 249)
    tuple_var_assignment_407939_409418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'tuple_var_assignment_407939')
    # Assigning a type to the variable 'bnrm2' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'bnrm2', tuple_var_assignment_407939_409418)
    
    # Assigning a Name to a Name (line 249):
    # Getting the type of 'tuple_var_assignment_407940' (line 249)
    tuple_var_assignment_407940_409419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'tuple_var_assignment_407940')
    # Assigning a type to the variable 'resid' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'resid', tuple_var_assignment_407940_409419)
    
    # Assigning a Name to a Name (line 249):
    # Getting the type of 'tuple_var_assignment_407941' (line 249)
    tuple_var_assignment_407941_409420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'tuple_var_assignment_407941')
    # Assigning a type to the variable 'info' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 26), 'info', tuple_var_assignment_407941_409420)
    # SSA join for if statement (line 245)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 242)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 240)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 237)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 233)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 250):
    
    # Assigning a Num to a Name (line 250):
    int_409421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 15), 'int')
    # Assigning a type to the variable 'ijob' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'ijob', int_409421)
    # SSA join for while statement (line 225)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'info' (line 252)
    info_409422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 7), 'info')
    int_409423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 14), 'int')
    # Applying the binary operator '>' (line 252)
    result_gt_409424 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 7), '>', info_409422, int_409423)
    
    
    # Getting the type of 'iter_' (line 252)
    iter__409425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 20), 'iter_')
    # Getting the type of 'maxiter' (line 252)
    maxiter_409426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 29), 'maxiter')
    # Applying the binary operator '==' (line 252)
    result_eq_409427 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 20), '==', iter__409425, maxiter_409426)
    
    # Applying the binary operator 'and' (line 252)
    result_and_keyword_409428 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 7), 'and', result_gt_409424, result_eq_409427)
    
    # Getting the type of 'resid' (line 252)
    resid_409429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 41), 'resid')
    # Getting the type of 'tol' (line 252)
    tol_409430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 49), 'tol')
    # Applying the binary operator '>' (line 252)
    result_gt_409431 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 41), '>', resid_409429, tol_409430)
    
    # Applying the binary operator 'and' (line 252)
    result_and_keyword_409432 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 7), 'and', result_and_keyword_409428, result_gt_409431)
    
    # Testing the type of an if condition (line 252)
    if_condition_409433 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 4), result_and_keyword_409432)
    # Assigning a type to the variable 'if_condition_409433' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'if_condition_409433', if_condition_409433)
    # SSA begins for if statement (line 252)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 254):
    
    # Assigning a Name to a Name (line 254):
    # Getting the type of 'iter_' (line 254)
    iter__409434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'iter_')
    # Assigning a type to the variable 'info' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'info', iter__409434)
    # SSA join for if statement (line 252)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 256)
    tuple_409435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 256)
    # Adding element type (line 256)
    
    # Call to postprocess(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'x' (line 256)
    x_409437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 23), 'x', False)
    # Processing the call keyword arguments (line 256)
    kwargs_409438 = {}
    # Getting the type of 'postprocess' (line 256)
    postprocess_409436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'postprocess', False)
    # Calling postprocess(args, kwargs) (line 256)
    postprocess_call_result_409439 = invoke(stypy.reporting.localization.Localization(__file__, 256, 11), postprocess_409436, *[x_409437], **kwargs_409438)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 11), tuple_409435, postprocess_call_result_409439)
    # Adding element type (line 256)
    # Getting the type of 'info' (line 256)
    info_409440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 27), 'info')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 11), tuple_409435, info_409440)
    
    # Assigning a type to the variable 'stypy_return_type' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type', tuple_409435)
    
    # ################# End of 'cg(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cg' in the type store
    # Getting the type of 'stypy_return_type' (line 198)
    stypy_return_type_409441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_409441)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cg'
    return stypy_return_type_409441

# Assigning a type to the variable 'cg' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'cg', cg)

@norecursion
def cgs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 262)
    None_409442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 17), 'None')
    float_409443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 27), 'float')
    # Getting the type of 'None' (line 262)
    None_409444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 41), 'None')
    # Getting the type of 'None' (line 262)
    None_409445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 49), 'None')
    # Getting the type of 'None' (line 262)
    None_409446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 64), 'None')
    defaults = [None_409442, float_409443, None_409444, None_409445, None_409446]
    # Create a new context for function 'cgs'
    module_type_store = module_type_store.open_function_context('cgs', 259, 0, False)
    
    # Passed parameters checking function
    cgs.stypy_localization = localization
    cgs.stypy_type_of_self = None
    cgs.stypy_type_store = module_type_store
    cgs.stypy_function_name = 'cgs'
    cgs.stypy_param_names_list = ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback']
    cgs.stypy_varargs_param_name = None
    cgs.stypy_kwargs_param_name = None
    cgs.stypy_call_defaults = defaults
    cgs.stypy_call_varargs = varargs
    cgs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cgs', ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cgs', localization, ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cgs(...)' code ##################

    
    # Assigning a Call to a Tuple (line 263):
    
    # Assigning a Subscript to a Name (line 263):
    
    # Obtaining the type of the subscript
    int_409447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 4), 'int')
    
    # Call to make_system(...): (line 263)
    # Processing the call arguments (line 263)
    # Getting the type of 'A' (line 263)
    A_409449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 42), 'A', False)
    # Getting the type of 'M' (line 263)
    M_409450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 45), 'M', False)
    # Getting the type of 'x0' (line 263)
    x0_409451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 48), 'x0', False)
    # Getting the type of 'b' (line 263)
    b_409452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 52), 'b', False)
    # Processing the call keyword arguments (line 263)
    kwargs_409453 = {}
    # Getting the type of 'make_system' (line 263)
    make_system_409448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 263)
    make_system_call_result_409454 = invoke(stypy.reporting.localization.Localization(__file__, 263, 30), make_system_409448, *[A_409449, M_409450, x0_409451, b_409452], **kwargs_409453)
    
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___409455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 4), make_system_call_result_409454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_409456 = invoke(stypy.reporting.localization.Localization(__file__, 263, 4), getitem___409455, int_409447)
    
    # Assigning a type to the variable 'tuple_var_assignment_407942' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'tuple_var_assignment_407942', subscript_call_result_409456)
    
    # Assigning a Subscript to a Name (line 263):
    
    # Obtaining the type of the subscript
    int_409457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 4), 'int')
    
    # Call to make_system(...): (line 263)
    # Processing the call arguments (line 263)
    # Getting the type of 'A' (line 263)
    A_409459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 42), 'A', False)
    # Getting the type of 'M' (line 263)
    M_409460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 45), 'M', False)
    # Getting the type of 'x0' (line 263)
    x0_409461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 48), 'x0', False)
    # Getting the type of 'b' (line 263)
    b_409462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 52), 'b', False)
    # Processing the call keyword arguments (line 263)
    kwargs_409463 = {}
    # Getting the type of 'make_system' (line 263)
    make_system_409458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 263)
    make_system_call_result_409464 = invoke(stypy.reporting.localization.Localization(__file__, 263, 30), make_system_409458, *[A_409459, M_409460, x0_409461, b_409462], **kwargs_409463)
    
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___409465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 4), make_system_call_result_409464, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_409466 = invoke(stypy.reporting.localization.Localization(__file__, 263, 4), getitem___409465, int_409457)
    
    # Assigning a type to the variable 'tuple_var_assignment_407943' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'tuple_var_assignment_407943', subscript_call_result_409466)
    
    # Assigning a Subscript to a Name (line 263):
    
    # Obtaining the type of the subscript
    int_409467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 4), 'int')
    
    # Call to make_system(...): (line 263)
    # Processing the call arguments (line 263)
    # Getting the type of 'A' (line 263)
    A_409469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 42), 'A', False)
    # Getting the type of 'M' (line 263)
    M_409470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 45), 'M', False)
    # Getting the type of 'x0' (line 263)
    x0_409471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 48), 'x0', False)
    # Getting the type of 'b' (line 263)
    b_409472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 52), 'b', False)
    # Processing the call keyword arguments (line 263)
    kwargs_409473 = {}
    # Getting the type of 'make_system' (line 263)
    make_system_409468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 263)
    make_system_call_result_409474 = invoke(stypy.reporting.localization.Localization(__file__, 263, 30), make_system_409468, *[A_409469, M_409470, x0_409471, b_409472], **kwargs_409473)
    
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___409475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 4), make_system_call_result_409474, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_409476 = invoke(stypy.reporting.localization.Localization(__file__, 263, 4), getitem___409475, int_409467)
    
    # Assigning a type to the variable 'tuple_var_assignment_407944' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'tuple_var_assignment_407944', subscript_call_result_409476)
    
    # Assigning a Subscript to a Name (line 263):
    
    # Obtaining the type of the subscript
    int_409477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 4), 'int')
    
    # Call to make_system(...): (line 263)
    # Processing the call arguments (line 263)
    # Getting the type of 'A' (line 263)
    A_409479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 42), 'A', False)
    # Getting the type of 'M' (line 263)
    M_409480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 45), 'M', False)
    # Getting the type of 'x0' (line 263)
    x0_409481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 48), 'x0', False)
    # Getting the type of 'b' (line 263)
    b_409482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 52), 'b', False)
    # Processing the call keyword arguments (line 263)
    kwargs_409483 = {}
    # Getting the type of 'make_system' (line 263)
    make_system_409478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 263)
    make_system_call_result_409484 = invoke(stypy.reporting.localization.Localization(__file__, 263, 30), make_system_409478, *[A_409479, M_409480, x0_409481, b_409482], **kwargs_409483)
    
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___409485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 4), make_system_call_result_409484, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_409486 = invoke(stypy.reporting.localization.Localization(__file__, 263, 4), getitem___409485, int_409477)
    
    # Assigning a type to the variable 'tuple_var_assignment_407945' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'tuple_var_assignment_407945', subscript_call_result_409486)
    
    # Assigning a Subscript to a Name (line 263):
    
    # Obtaining the type of the subscript
    int_409487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 4), 'int')
    
    # Call to make_system(...): (line 263)
    # Processing the call arguments (line 263)
    # Getting the type of 'A' (line 263)
    A_409489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 42), 'A', False)
    # Getting the type of 'M' (line 263)
    M_409490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 45), 'M', False)
    # Getting the type of 'x0' (line 263)
    x0_409491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 48), 'x0', False)
    # Getting the type of 'b' (line 263)
    b_409492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 52), 'b', False)
    # Processing the call keyword arguments (line 263)
    kwargs_409493 = {}
    # Getting the type of 'make_system' (line 263)
    make_system_409488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 263)
    make_system_call_result_409494 = invoke(stypy.reporting.localization.Localization(__file__, 263, 30), make_system_409488, *[A_409489, M_409490, x0_409491, b_409492], **kwargs_409493)
    
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___409495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 4), make_system_call_result_409494, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_409496 = invoke(stypy.reporting.localization.Localization(__file__, 263, 4), getitem___409495, int_409487)
    
    # Assigning a type to the variable 'tuple_var_assignment_407946' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'tuple_var_assignment_407946', subscript_call_result_409496)
    
    # Assigning a Name to a Name (line 263):
    # Getting the type of 'tuple_var_assignment_407942' (line 263)
    tuple_var_assignment_407942_409497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'tuple_var_assignment_407942')
    # Assigning a type to the variable 'A' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'A', tuple_var_assignment_407942_409497)
    
    # Assigning a Name to a Name (line 263):
    # Getting the type of 'tuple_var_assignment_407943' (line 263)
    tuple_var_assignment_407943_409498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'tuple_var_assignment_407943')
    # Assigning a type to the variable 'M' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 7), 'M', tuple_var_assignment_407943_409498)
    
    # Assigning a Name to a Name (line 263):
    # Getting the type of 'tuple_var_assignment_407944' (line 263)
    tuple_var_assignment_407944_409499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'tuple_var_assignment_407944')
    # Assigning a type to the variable 'x' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 10), 'x', tuple_var_assignment_407944_409499)
    
    # Assigning a Name to a Name (line 263):
    # Getting the type of 'tuple_var_assignment_407945' (line 263)
    tuple_var_assignment_407945_409500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'tuple_var_assignment_407945')
    # Assigning a type to the variable 'b' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 13), 'b', tuple_var_assignment_407945_409500)
    
    # Assigning a Name to a Name (line 263):
    # Getting the type of 'tuple_var_assignment_407946' (line 263)
    tuple_var_assignment_407946_409501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'tuple_var_assignment_407946')
    # Assigning a type to the variable 'postprocess' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'postprocess', tuple_var_assignment_407946_409501)
    
    # Assigning a Call to a Name (line 265):
    
    # Assigning a Call to a Name (line 265):
    
    # Call to len(...): (line 265)
    # Processing the call arguments (line 265)
    # Getting the type of 'b' (line 265)
    b_409503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'b', False)
    # Processing the call keyword arguments (line 265)
    kwargs_409504 = {}
    # Getting the type of 'len' (line 265)
    len_409502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'len', False)
    # Calling len(args, kwargs) (line 265)
    len_call_result_409505 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), len_409502, *[b_409503], **kwargs_409504)
    
    # Assigning a type to the variable 'n' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'n', len_call_result_409505)
    
    # Type idiom detected: calculating its left and rigth part (line 266)
    # Getting the type of 'maxiter' (line 266)
    maxiter_409506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 7), 'maxiter')
    # Getting the type of 'None' (line 266)
    None_409507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 18), 'None')
    
    (may_be_409508, more_types_in_union_409509) = may_be_none(maxiter_409506, None_409507)

    if may_be_409508:

        if more_types_in_union_409509:
            # Runtime conditional SSA (line 266)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 267):
        
        # Assigning a BinOp to a Name (line 267):
        # Getting the type of 'n' (line 267)
        n_409510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 18), 'n')
        int_409511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 20), 'int')
        # Applying the binary operator '*' (line 267)
        result_mul_409512 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 18), '*', n_409510, int_409511)
        
        # Assigning a type to the variable 'maxiter' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'maxiter', result_mul_409512)

        if more_types_in_union_409509:
            # SSA join for if statement (line 266)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 269):
    
    # Assigning a Attribute to a Name (line 269):
    # Getting the type of 'A' (line 269)
    A_409513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 13), 'A')
    # Obtaining the member 'matvec' of a type (line 269)
    matvec_409514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 13), A_409513, 'matvec')
    # Assigning a type to the variable 'matvec' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'matvec', matvec_409514)
    
    # Assigning a Attribute to a Name (line 270):
    
    # Assigning a Attribute to a Name (line 270):
    # Getting the type of 'M' (line 270)
    M_409515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 13), 'M')
    # Obtaining the member 'matvec' of a type (line 270)
    matvec_409516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 13), M_409515, 'matvec')
    # Assigning a type to the variable 'psolve' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'psolve', matvec_409516)
    
    # Assigning a Subscript to a Name (line 271):
    
    # Assigning a Subscript to a Name (line 271):
    
    # Obtaining the type of the subscript
    # Getting the type of 'x' (line 271)
    x_409517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 21), 'x')
    # Obtaining the member 'dtype' of a type (line 271)
    dtype_409518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 21), x_409517, 'dtype')
    # Obtaining the member 'char' of a type (line 271)
    char_409519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 21), dtype_409518, 'char')
    # Getting the type of '_type_conv' (line 271)
    _type_conv_409520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 10), '_type_conv')
    # Obtaining the member '__getitem__' of a type (line 271)
    getitem___409521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 10), _type_conv_409520, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 271)
    subscript_call_result_409522 = invoke(stypy.reporting.localization.Localization(__file__, 271, 10), getitem___409521, char_409519)
    
    # Assigning a type to the variable 'ltr' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'ltr', subscript_call_result_409522)
    
    # Assigning a Call to a Name (line 272):
    
    # Assigning a Call to a Name (line 272):
    
    # Call to getattr(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of '_iterative' (line 272)
    _iterative_409524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 21), '_iterative', False)
    # Getting the type of 'ltr' (line 272)
    ltr_409525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 33), 'ltr', False)
    str_409526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 39), 'str', 'cgsrevcom')
    # Applying the binary operator '+' (line 272)
    result_add_409527 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 33), '+', ltr_409525, str_409526)
    
    # Processing the call keyword arguments (line 272)
    kwargs_409528 = {}
    # Getting the type of 'getattr' (line 272)
    getattr_409523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 13), 'getattr', False)
    # Calling getattr(args, kwargs) (line 272)
    getattr_call_result_409529 = invoke(stypy.reporting.localization.Localization(__file__, 272, 13), getattr_409523, *[_iterative_409524, result_add_409527], **kwargs_409528)
    
    # Assigning a type to the variable 'revcom' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'revcom', getattr_call_result_409529)
    
    # Assigning a Call to a Name (line 273):
    
    # Assigning a Call to a Name (line 273):
    
    # Call to getattr(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of '_iterative' (line 273)
    _iterative_409531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 23), '_iterative', False)
    # Getting the type of 'ltr' (line 273)
    ltr_409532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 35), 'ltr', False)
    str_409533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 41), 'str', 'stoptest2')
    # Applying the binary operator '+' (line 273)
    result_add_409534 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 35), '+', ltr_409532, str_409533)
    
    # Processing the call keyword arguments (line 273)
    kwargs_409535 = {}
    # Getting the type of 'getattr' (line 273)
    getattr_409530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 15), 'getattr', False)
    # Calling getattr(args, kwargs) (line 273)
    getattr_call_result_409536 = invoke(stypy.reporting.localization.Localization(__file__, 273, 15), getattr_409530, *[_iterative_409531, result_add_409534], **kwargs_409535)
    
    # Assigning a type to the variable 'stoptest' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'stoptest', getattr_call_result_409536)
    
    # Assigning a Name to a Name (line 275):
    
    # Assigning a Name to a Name (line 275):
    # Getting the type of 'tol' (line 275)
    tol_409537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'tol')
    # Assigning a type to the variable 'resid' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'resid', tol_409537)
    
    # Assigning a Num to a Name (line 276):
    
    # Assigning a Num to a Name (line 276):
    int_409538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 11), 'int')
    # Assigning a type to the variable 'ndx1' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'ndx1', int_409538)
    
    # Assigning a Num to a Name (line 277):
    
    # Assigning a Num to a Name (line 277):
    int_409539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 11), 'int')
    # Assigning a type to the variable 'ndx2' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'ndx2', int_409539)
    
    # Assigning a Call to a Name (line 279):
    
    # Assigning a Call to a Name (line 279):
    
    # Call to _aligned_zeros(...): (line 279)
    # Processing the call arguments (line 279)
    int_409541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 26), 'int')
    # Getting the type of 'n' (line 279)
    n_409542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 28), 'n', False)
    # Applying the binary operator '*' (line 279)
    result_mul_409543 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 26), '*', int_409541, n_409542)
    
    # Processing the call keyword arguments (line 279)
    # Getting the type of 'x' (line 279)
    x_409544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 36), 'x', False)
    # Obtaining the member 'dtype' of a type (line 279)
    dtype_409545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 36), x_409544, 'dtype')
    keyword_409546 = dtype_409545
    kwargs_409547 = {'dtype': keyword_409546}
    # Getting the type of '_aligned_zeros' (line 279)
    _aligned_zeros_409540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), '_aligned_zeros', False)
    # Calling _aligned_zeros(args, kwargs) (line 279)
    _aligned_zeros_call_result_409548 = invoke(stypy.reporting.localization.Localization(__file__, 279, 11), _aligned_zeros_409540, *[result_mul_409543], **kwargs_409547)
    
    # Assigning a type to the variable 'work' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'work', _aligned_zeros_call_result_409548)
    
    # Assigning a Num to a Name (line 280):
    
    # Assigning a Num to a Name (line 280):
    int_409549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 11), 'int')
    # Assigning a type to the variable 'ijob' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'ijob', int_409549)
    
    # Assigning a Num to a Name (line 281):
    
    # Assigning a Num to a Name (line 281):
    int_409550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 11), 'int')
    # Assigning a type to the variable 'info' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'info', int_409550)
    
    # Assigning a Name to a Name (line 282):
    
    # Assigning a Name to a Name (line 282):
    # Getting the type of 'True' (line 282)
    True_409551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 13), 'True')
    # Assigning a type to the variable 'ftflag' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'ftflag', True_409551)
    
    # Assigning a Num to a Name (line 283):
    
    # Assigning a Num to a Name (line 283):
    float_409552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 12), 'float')
    # Assigning a type to the variable 'bnrm2' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'bnrm2', float_409552)
    
    # Assigning a Name to a Name (line 284):
    
    # Assigning a Name to a Name (line 284):
    # Getting the type of 'maxiter' (line 284)
    maxiter_409553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'maxiter')
    # Assigning a type to the variable 'iter_' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'iter_', maxiter_409553)
    
    # Getting the type of 'True' (line 285)
    True_409554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 10), 'True')
    # Testing the type of an if condition (line 285)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 4), True_409554)
    # SSA begins for while statement (line 285)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Name to a Name (line 286):
    
    # Assigning a Name to a Name (line 286):
    # Getting the type of 'iter_' (line 286)
    iter__409555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 18), 'iter_')
    # Assigning a type to the variable 'olditer' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'olditer', iter__409555)
    
    # Assigning a Call to a Tuple (line 287):
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_409556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
    
    # Call to revcom(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'b' (line 288)
    b_409558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'b', False)
    # Getting the type of 'x' (line 288)
    x_409559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'x', False)
    # Getting the type of 'work' (line 288)
    work_409560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'work', False)
    # Getting the type of 'iter_' (line 288)
    iter__409561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 30), 'iter_', False)
    # Getting the type of 'resid' (line 288)
    resid_409562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 'resid', False)
    # Getting the type of 'info' (line 288)
    info_409563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 44), 'info', False)
    # Getting the type of 'ndx1' (line 288)
    ndx1_409564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 288)
    ndx2_409565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 288)
    ijob_409566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 62), 'ijob', False)
    # Processing the call keyword arguments (line 288)
    kwargs_409567 = {}
    # Getting the type of 'revcom' (line 288)
    revcom_409557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 288)
    revcom_call_result_409568 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), revcom_409557, *[b_409558, x_409559, work_409560, iter__409561, resid_409562, info_409563, ndx1_409564, ndx2_409565, ijob_409566], **kwargs_409567)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___409569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), revcom_call_result_409568, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_409570 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___409569, int_409556)
    
    # Assigning a type to the variable 'tuple_var_assignment_407947' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407947', subscript_call_result_409570)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_409571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
    
    # Call to revcom(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'b' (line 288)
    b_409573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'b', False)
    # Getting the type of 'x' (line 288)
    x_409574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'x', False)
    # Getting the type of 'work' (line 288)
    work_409575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'work', False)
    # Getting the type of 'iter_' (line 288)
    iter__409576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 30), 'iter_', False)
    # Getting the type of 'resid' (line 288)
    resid_409577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 'resid', False)
    # Getting the type of 'info' (line 288)
    info_409578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 44), 'info', False)
    # Getting the type of 'ndx1' (line 288)
    ndx1_409579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 288)
    ndx2_409580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 288)
    ijob_409581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 62), 'ijob', False)
    # Processing the call keyword arguments (line 288)
    kwargs_409582 = {}
    # Getting the type of 'revcom' (line 288)
    revcom_409572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 288)
    revcom_call_result_409583 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), revcom_409572, *[b_409573, x_409574, work_409575, iter__409576, resid_409577, info_409578, ndx1_409579, ndx2_409580, ijob_409581], **kwargs_409582)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___409584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), revcom_call_result_409583, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_409585 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___409584, int_409571)
    
    # Assigning a type to the variable 'tuple_var_assignment_407948' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407948', subscript_call_result_409585)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_409586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
    
    # Call to revcom(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'b' (line 288)
    b_409588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'b', False)
    # Getting the type of 'x' (line 288)
    x_409589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'x', False)
    # Getting the type of 'work' (line 288)
    work_409590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'work', False)
    # Getting the type of 'iter_' (line 288)
    iter__409591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 30), 'iter_', False)
    # Getting the type of 'resid' (line 288)
    resid_409592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 'resid', False)
    # Getting the type of 'info' (line 288)
    info_409593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 44), 'info', False)
    # Getting the type of 'ndx1' (line 288)
    ndx1_409594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 288)
    ndx2_409595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 288)
    ijob_409596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 62), 'ijob', False)
    # Processing the call keyword arguments (line 288)
    kwargs_409597 = {}
    # Getting the type of 'revcom' (line 288)
    revcom_409587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 288)
    revcom_call_result_409598 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), revcom_409587, *[b_409588, x_409589, work_409590, iter__409591, resid_409592, info_409593, ndx1_409594, ndx2_409595, ijob_409596], **kwargs_409597)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___409599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), revcom_call_result_409598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_409600 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___409599, int_409586)
    
    # Assigning a type to the variable 'tuple_var_assignment_407949' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407949', subscript_call_result_409600)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_409601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
    
    # Call to revcom(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'b' (line 288)
    b_409603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'b', False)
    # Getting the type of 'x' (line 288)
    x_409604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'x', False)
    # Getting the type of 'work' (line 288)
    work_409605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'work', False)
    # Getting the type of 'iter_' (line 288)
    iter__409606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 30), 'iter_', False)
    # Getting the type of 'resid' (line 288)
    resid_409607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 'resid', False)
    # Getting the type of 'info' (line 288)
    info_409608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 44), 'info', False)
    # Getting the type of 'ndx1' (line 288)
    ndx1_409609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 288)
    ndx2_409610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 288)
    ijob_409611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 62), 'ijob', False)
    # Processing the call keyword arguments (line 288)
    kwargs_409612 = {}
    # Getting the type of 'revcom' (line 288)
    revcom_409602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 288)
    revcom_call_result_409613 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), revcom_409602, *[b_409603, x_409604, work_409605, iter__409606, resid_409607, info_409608, ndx1_409609, ndx2_409610, ijob_409611], **kwargs_409612)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___409614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), revcom_call_result_409613, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_409615 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___409614, int_409601)
    
    # Assigning a type to the variable 'tuple_var_assignment_407950' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407950', subscript_call_result_409615)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_409616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
    
    # Call to revcom(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'b' (line 288)
    b_409618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'b', False)
    # Getting the type of 'x' (line 288)
    x_409619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'x', False)
    # Getting the type of 'work' (line 288)
    work_409620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'work', False)
    # Getting the type of 'iter_' (line 288)
    iter__409621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 30), 'iter_', False)
    # Getting the type of 'resid' (line 288)
    resid_409622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 'resid', False)
    # Getting the type of 'info' (line 288)
    info_409623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 44), 'info', False)
    # Getting the type of 'ndx1' (line 288)
    ndx1_409624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 288)
    ndx2_409625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 288)
    ijob_409626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 62), 'ijob', False)
    # Processing the call keyword arguments (line 288)
    kwargs_409627 = {}
    # Getting the type of 'revcom' (line 288)
    revcom_409617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 288)
    revcom_call_result_409628 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), revcom_409617, *[b_409618, x_409619, work_409620, iter__409621, resid_409622, info_409623, ndx1_409624, ndx2_409625, ijob_409626], **kwargs_409627)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___409629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), revcom_call_result_409628, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_409630 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___409629, int_409616)
    
    # Assigning a type to the variable 'tuple_var_assignment_407951' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407951', subscript_call_result_409630)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_409631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
    
    # Call to revcom(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'b' (line 288)
    b_409633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'b', False)
    # Getting the type of 'x' (line 288)
    x_409634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'x', False)
    # Getting the type of 'work' (line 288)
    work_409635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'work', False)
    # Getting the type of 'iter_' (line 288)
    iter__409636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 30), 'iter_', False)
    # Getting the type of 'resid' (line 288)
    resid_409637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 'resid', False)
    # Getting the type of 'info' (line 288)
    info_409638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 44), 'info', False)
    # Getting the type of 'ndx1' (line 288)
    ndx1_409639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 288)
    ndx2_409640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 288)
    ijob_409641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 62), 'ijob', False)
    # Processing the call keyword arguments (line 288)
    kwargs_409642 = {}
    # Getting the type of 'revcom' (line 288)
    revcom_409632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 288)
    revcom_call_result_409643 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), revcom_409632, *[b_409633, x_409634, work_409635, iter__409636, resid_409637, info_409638, ndx1_409639, ndx2_409640, ijob_409641], **kwargs_409642)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___409644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), revcom_call_result_409643, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_409645 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___409644, int_409631)
    
    # Assigning a type to the variable 'tuple_var_assignment_407952' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407952', subscript_call_result_409645)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_409646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
    
    # Call to revcom(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'b' (line 288)
    b_409648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'b', False)
    # Getting the type of 'x' (line 288)
    x_409649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'x', False)
    # Getting the type of 'work' (line 288)
    work_409650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'work', False)
    # Getting the type of 'iter_' (line 288)
    iter__409651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 30), 'iter_', False)
    # Getting the type of 'resid' (line 288)
    resid_409652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 'resid', False)
    # Getting the type of 'info' (line 288)
    info_409653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 44), 'info', False)
    # Getting the type of 'ndx1' (line 288)
    ndx1_409654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 288)
    ndx2_409655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 288)
    ijob_409656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 62), 'ijob', False)
    # Processing the call keyword arguments (line 288)
    kwargs_409657 = {}
    # Getting the type of 'revcom' (line 288)
    revcom_409647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 288)
    revcom_call_result_409658 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), revcom_409647, *[b_409648, x_409649, work_409650, iter__409651, resid_409652, info_409653, ndx1_409654, ndx2_409655, ijob_409656], **kwargs_409657)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___409659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), revcom_call_result_409658, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_409660 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___409659, int_409646)
    
    # Assigning a type to the variable 'tuple_var_assignment_407953' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407953', subscript_call_result_409660)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_409661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
    
    # Call to revcom(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'b' (line 288)
    b_409663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'b', False)
    # Getting the type of 'x' (line 288)
    x_409664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'x', False)
    # Getting the type of 'work' (line 288)
    work_409665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'work', False)
    # Getting the type of 'iter_' (line 288)
    iter__409666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 30), 'iter_', False)
    # Getting the type of 'resid' (line 288)
    resid_409667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 'resid', False)
    # Getting the type of 'info' (line 288)
    info_409668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 44), 'info', False)
    # Getting the type of 'ndx1' (line 288)
    ndx1_409669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 288)
    ndx2_409670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 288)
    ijob_409671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 62), 'ijob', False)
    # Processing the call keyword arguments (line 288)
    kwargs_409672 = {}
    # Getting the type of 'revcom' (line 288)
    revcom_409662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 288)
    revcom_call_result_409673 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), revcom_409662, *[b_409663, x_409664, work_409665, iter__409666, resid_409667, info_409668, ndx1_409669, ndx2_409670, ijob_409671], **kwargs_409672)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___409674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), revcom_call_result_409673, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_409675 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___409674, int_409661)
    
    # Assigning a type to the variable 'tuple_var_assignment_407954' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407954', subscript_call_result_409675)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_409676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
    
    # Call to revcom(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'b' (line 288)
    b_409678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'b', False)
    # Getting the type of 'x' (line 288)
    x_409679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'x', False)
    # Getting the type of 'work' (line 288)
    work_409680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'work', False)
    # Getting the type of 'iter_' (line 288)
    iter__409681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 30), 'iter_', False)
    # Getting the type of 'resid' (line 288)
    resid_409682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 'resid', False)
    # Getting the type of 'info' (line 288)
    info_409683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 44), 'info', False)
    # Getting the type of 'ndx1' (line 288)
    ndx1_409684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 288)
    ndx2_409685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 288)
    ijob_409686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 62), 'ijob', False)
    # Processing the call keyword arguments (line 288)
    kwargs_409687 = {}
    # Getting the type of 'revcom' (line 288)
    revcom_409677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 288)
    revcom_call_result_409688 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), revcom_409677, *[b_409678, x_409679, work_409680, iter__409681, resid_409682, info_409683, ndx1_409684, ndx2_409685, ijob_409686], **kwargs_409687)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___409689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), revcom_call_result_409688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_409690 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___409689, int_409676)
    
    # Assigning a type to the variable 'tuple_var_assignment_407955' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407955', subscript_call_result_409690)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_407947' (line 287)
    tuple_var_assignment_407947_409691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407947')
    # Assigning a type to the variable 'x' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'x', tuple_var_assignment_407947_409691)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_407948' (line 287)
    tuple_var_assignment_407948_409692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407948')
    # Assigning a type to the variable 'iter_' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 11), 'iter_', tuple_var_assignment_407948_409692)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_407949' (line 287)
    tuple_var_assignment_407949_409693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407949')
    # Assigning a type to the variable 'resid' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 18), 'resid', tuple_var_assignment_407949_409693)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_407950' (line 287)
    tuple_var_assignment_407950_409694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407950')
    # Assigning a type to the variable 'info' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 25), 'info', tuple_var_assignment_407950_409694)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_407951' (line 287)
    tuple_var_assignment_407951_409695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407951')
    # Assigning a type to the variable 'ndx1' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'ndx1', tuple_var_assignment_407951_409695)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_407952' (line 287)
    tuple_var_assignment_407952_409696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407952')
    # Assigning a type to the variable 'ndx2' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 37), 'ndx2', tuple_var_assignment_407952_409696)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_407953' (line 287)
    tuple_var_assignment_407953_409697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407953')
    # Assigning a type to the variable 'sclr1' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 43), 'sclr1', tuple_var_assignment_407953_409697)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_407954' (line 287)
    tuple_var_assignment_407954_409698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407954')
    # Assigning a type to the variable 'sclr2' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 50), 'sclr2', tuple_var_assignment_407954_409698)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_407955' (line 287)
    tuple_var_assignment_407955_409699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_407955')
    # Assigning a type to the variable 'ijob' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 57), 'ijob', tuple_var_assignment_407955_409699)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'callback' (line 289)
    callback_409700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 11), 'callback')
    # Getting the type of 'None' (line 289)
    None_409701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 27), 'None')
    # Applying the binary operator 'isnot' (line 289)
    result_is_not_409702 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 11), 'isnot', callback_409700, None_409701)
    
    
    # Getting the type of 'iter_' (line 289)
    iter__409703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 36), 'iter_')
    # Getting the type of 'olditer' (line 289)
    olditer_409704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 44), 'olditer')
    # Applying the binary operator '>' (line 289)
    result_gt_409705 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 36), '>', iter__409703, olditer_409704)
    
    # Applying the binary operator 'and' (line 289)
    result_and_keyword_409706 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 11), 'and', result_is_not_409702, result_gt_409705)
    
    # Testing the type of an if condition (line 289)
    if_condition_409707 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 8), result_and_keyword_409706)
    # Assigning a type to the variable 'if_condition_409707' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'if_condition_409707', if_condition_409707)
    # SSA begins for if statement (line 289)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to callback(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'x' (line 290)
    x_409709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 21), 'x', False)
    # Processing the call keyword arguments (line 290)
    kwargs_409710 = {}
    # Getting the type of 'callback' (line 290)
    callback_409708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'callback', False)
    # Calling callback(args, kwargs) (line 290)
    callback_call_result_409711 = invoke(stypy.reporting.localization.Localization(__file__, 290, 12), callback_409708, *[x_409709], **kwargs_409710)
    
    # SSA join for if statement (line 289)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 291):
    
    # Assigning a Call to a Name (line 291):
    
    # Call to slice(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'ndx1' (line 291)
    ndx1_409713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 23), 'ndx1', False)
    int_409714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 28), 'int')
    # Applying the binary operator '-' (line 291)
    result_sub_409715 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 23), '-', ndx1_409713, int_409714)
    
    # Getting the type of 'ndx1' (line 291)
    ndx1_409716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 31), 'ndx1', False)
    int_409717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 36), 'int')
    # Applying the binary operator '-' (line 291)
    result_sub_409718 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 31), '-', ndx1_409716, int_409717)
    
    # Getting the type of 'n' (line 291)
    n_409719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 38), 'n', False)
    # Applying the binary operator '+' (line 291)
    result_add_409720 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 37), '+', result_sub_409718, n_409719)
    
    # Processing the call keyword arguments (line 291)
    kwargs_409721 = {}
    # Getting the type of 'slice' (line 291)
    slice_409712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 291)
    slice_call_result_409722 = invoke(stypy.reporting.localization.Localization(__file__, 291, 17), slice_409712, *[result_sub_409715, result_add_409720], **kwargs_409721)
    
    # Assigning a type to the variable 'slice1' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'slice1', slice_call_result_409722)
    
    # Assigning a Call to a Name (line 292):
    
    # Assigning a Call to a Name (line 292):
    
    # Call to slice(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'ndx2' (line 292)
    ndx2_409724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 23), 'ndx2', False)
    int_409725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 28), 'int')
    # Applying the binary operator '-' (line 292)
    result_sub_409726 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 23), '-', ndx2_409724, int_409725)
    
    # Getting the type of 'ndx2' (line 292)
    ndx2_409727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 31), 'ndx2', False)
    int_409728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 36), 'int')
    # Applying the binary operator '-' (line 292)
    result_sub_409729 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 31), '-', ndx2_409727, int_409728)
    
    # Getting the type of 'n' (line 292)
    n_409730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 38), 'n', False)
    # Applying the binary operator '+' (line 292)
    result_add_409731 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 37), '+', result_sub_409729, n_409730)
    
    # Processing the call keyword arguments (line 292)
    kwargs_409732 = {}
    # Getting the type of 'slice' (line 292)
    slice_409723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 292)
    slice_call_result_409733 = invoke(stypy.reporting.localization.Localization(__file__, 292, 17), slice_409723, *[result_sub_409726, result_add_409731], **kwargs_409732)
    
    # Assigning a type to the variable 'slice2' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'slice2', slice_call_result_409733)
    
    
    # Getting the type of 'ijob' (line 293)
    ijob_409734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'ijob')
    int_409735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 20), 'int')
    # Applying the binary operator '==' (line 293)
    result_eq_409736 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 12), '==', ijob_409734, int_409735)
    
    # Testing the type of an if condition (line 293)
    if_condition_409737 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 8), result_eq_409736)
    # Assigning a type to the variable 'if_condition_409737' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'if_condition_409737', if_condition_409737)
    # SSA begins for if statement (line 293)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 294)
    # Getting the type of 'callback' (line 294)
    callback_409738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'callback')
    # Getting the type of 'None' (line 294)
    None_409739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 31), 'None')
    
    (may_be_409740, more_types_in_union_409741) = may_not_be_none(callback_409738, None_409739)

    if may_be_409740:

        if more_types_in_union_409741:
            # Runtime conditional SSA (line 294)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to callback(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'x' (line 295)
        x_409743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 25), 'x', False)
        # Processing the call keyword arguments (line 295)
        kwargs_409744 = {}
        # Getting the type of 'callback' (line 295)
        callback_409742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'callback', False)
        # Calling callback(args, kwargs) (line 295)
        callback_call_result_409745 = invoke(stypy.reporting.localization.Localization(__file__, 295, 16), callback_409742, *[x_409743], **kwargs_409744)
        

        if more_types_in_union_409741:
            # SSA join for if statement (line 294)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 293)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 297)
    ijob_409746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 14), 'ijob')
    int_409747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 22), 'int')
    # Applying the binary operator '==' (line 297)
    result_eq_409748 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 14), '==', ijob_409746, int_409747)
    
    # Testing the type of an if condition (line 297)
    if_condition_409749 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 13), result_eq_409748)
    # Assigning a type to the variable 'if_condition_409749' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 13), 'if_condition_409749', if_condition_409749)
    # SSA begins for if statement (line 297)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 298)
    work_409750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 298)
    slice2_409751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 17), 'slice2')
    # Getting the type of 'work' (line 298)
    work_409752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___409753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 12), work_409752, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_409754 = invoke(stypy.reporting.localization.Localization(__file__, 298, 12), getitem___409753, slice2_409751)
    
    # Getting the type of 'sclr2' (line 298)
    sclr2_409755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 28), 'sclr2')
    # Applying the binary operator '*=' (line 298)
    result_imul_409756 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 12), '*=', subscript_call_result_409754, sclr2_409755)
    # Getting the type of 'work' (line 298)
    work_409757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'work')
    # Getting the type of 'slice2' (line 298)
    slice2_409758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 17), 'slice2')
    # Storing an element on a container (line 298)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 12), work_409757, (slice2_409758, result_imul_409756))
    
    
    # Getting the type of 'work' (line 299)
    work_409759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 299)
    slice2_409760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'slice2')
    # Getting the type of 'work' (line 299)
    work_409761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___409762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), work_409761, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_409763 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), getitem___409762, slice2_409760)
    
    # Getting the type of 'sclr1' (line 299)
    sclr1_409764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 28), 'sclr1')
    
    # Call to matvec(...): (line 299)
    # Processing the call arguments (line 299)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 299)
    slice1_409766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 46), 'slice1', False)
    # Getting the type of 'work' (line 299)
    work_409767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 41), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___409768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 41), work_409767, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_409769 = invoke(stypy.reporting.localization.Localization(__file__, 299, 41), getitem___409768, slice1_409766)
    
    # Processing the call keyword arguments (line 299)
    kwargs_409770 = {}
    # Getting the type of 'matvec' (line 299)
    matvec_409765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 34), 'matvec', False)
    # Calling matvec(args, kwargs) (line 299)
    matvec_call_result_409771 = invoke(stypy.reporting.localization.Localization(__file__, 299, 34), matvec_409765, *[subscript_call_result_409769], **kwargs_409770)
    
    # Applying the binary operator '*' (line 299)
    result_mul_409772 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 28), '*', sclr1_409764, matvec_call_result_409771)
    
    # Applying the binary operator '+=' (line 299)
    result_iadd_409773 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 12), '+=', subscript_call_result_409763, result_mul_409772)
    # Getting the type of 'work' (line 299)
    work_409774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'work')
    # Getting the type of 'slice2' (line 299)
    slice2_409775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'slice2')
    # Storing an element on a container (line 299)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 12), work_409774, (slice2_409775, result_iadd_409773))
    
    # SSA branch for the else part of an if statement (line 297)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 300)
    ijob_409776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 14), 'ijob')
    int_409777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 22), 'int')
    # Applying the binary operator '==' (line 300)
    result_eq_409778 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 14), '==', ijob_409776, int_409777)
    
    # Testing the type of an if condition (line 300)
    if_condition_409779 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 13), result_eq_409778)
    # Assigning a type to the variable 'if_condition_409779' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 13), 'if_condition_409779', if_condition_409779)
    # SSA begins for if statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 301):
    
    # Assigning a Call to a Subscript (line 301):
    
    # Call to psolve(...): (line 301)
    # Processing the call arguments (line 301)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 301)
    slice2_409781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 39), 'slice2', False)
    # Getting the type of 'work' (line 301)
    work_409782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 34), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 301)
    getitem___409783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 34), work_409782, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 301)
    subscript_call_result_409784 = invoke(stypy.reporting.localization.Localization(__file__, 301, 34), getitem___409783, slice2_409781)
    
    # Processing the call keyword arguments (line 301)
    kwargs_409785 = {}
    # Getting the type of 'psolve' (line 301)
    psolve_409780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 27), 'psolve', False)
    # Calling psolve(args, kwargs) (line 301)
    psolve_call_result_409786 = invoke(stypy.reporting.localization.Localization(__file__, 301, 27), psolve_409780, *[subscript_call_result_409784], **kwargs_409785)
    
    # Getting the type of 'work' (line 301)
    work_409787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'work')
    # Getting the type of 'slice1' (line 301)
    slice1_409788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 17), 'slice1')
    # Storing an element on a container (line 301)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), work_409787, (slice1_409788, psolve_call_result_409786))
    # SSA branch for the else part of an if statement (line 300)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 302)
    ijob_409789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 14), 'ijob')
    int_409790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 22), 'int')
    # Applying the binary operator '==' (line 302)
    result_eq_409791 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 14), '==', ijob_409789, int_409790)
    
    # Testing the type of an if condition (line 302)
    if_condition_409792 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 13), result_eq_409791)
    # Assigning a type to the variable 'if_condition_409792' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 13), 'if_condition_409792', if_condition_409792)
    # SSA begins for if statement (line 302)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 303)
    work_409793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 303)
    slice2_409794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 17), 'slice2')
    # Getting the type of 'work' (line 303)
    work_409795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___409796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 12), work_409795, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_409797 = invoke(stypy.reporting.localization.Localization(__file__, 303, 12), getitem___409796, slice2_409794)
    
    # Getting the type of 'sclr2' (line 303)
    sclr2_409798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 28), 'sclr2')
    # Applying the binary operator '*=' (line 303)
    result_imul_409799 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 12), '*=', subscript_call_result_409797, sclr2_409798)
    # Getting the type of 'work' (line 303)
    work_409800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'work')
    # Getting the type of 'slice2' (line 303)
    slice2_409801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 17), 'slice2')
    # Storing an element on a container (line 303)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 12), work_409800, (slice2_409801, result_imul_409799))
    
    
    # Getting the type of 'work' (line 304)
    work_409802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 304)
    slice2_409803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 17), 'slice2')
    # Getting the type of 'work' (line 304)
    work_409804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___409805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), work_409804, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_409806 = invoke(stypy.reporting.localization.Localization(__file__, 304, 12), getitem___409805, slice2_409803)
    
    # Getting the type of 'sclr1' (line 304)
    sclr1_409807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 28), 'sclr1')
    
    # Call to matvec(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'x' (line 304)
    x_409809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 41), 'x', False)
    # Processing the call keyword arguments (line 304)
    kwargs_409810 = {}
    # Getting the type of 'matvec' (line 304)
    matvec_409808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 34), 'matvec', False)
    # Calling matvec(args, kwargs) (line 304)
    matvec_call_result_409811 = invoke(stypy.reporting.localization.Localization(__file__, 304, 34), matvec_409808, *[x_409809], **kwargs_409810)
    
    # Applying the binary operator '*' (line 304)
    result_mul_409812 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 28), '*', sclr1_409807, matvec_call_result_409811)
    
    # Applying the binary operator '+=' (line 304)
    result_iadd_409813 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 12), '+=', subscript_call_result_409806, result_mul_409812)
    # Getting the type of 'work' (line 304)
    work_409814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'work')
    # Getting the type of 'slice2' (line 304)
    slice2_409815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 17), 'slice2')
    # Storing an element on a container (line 304)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), work_409814, (slice2_409815, result_iadd_409813))
    
    # SSA branch for the else part of an if statement (line 302)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 305)
    ijob_409816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 14), 'ijob')
    int_409817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 22), 'int')
    # Applying the binary operator '==' (line 305)
    result_eq_409818 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 14), '==', ijob_409816, int_409817)
    
    # Testing the type of an if condition (line 305)
    if_condition_409819 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 13), result_eq_409818)
    # Assigning a type to the variable 'if_condition_409819' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 13), 'if_condition_409819', if_condition_409819)
    # SSA begins for if statement (line 305)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ftflag' (line 306)
    ftflag_409820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), 'ftflag')
    # Testing the type of an if condition (line 306)
    if_condition_409821 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 12), ftflag_409820)
    # Assigning a type to the variable 'if_condition_409821' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'if_condition_409821', if_condition_409821)
    # SSA begins for if statement (line 306)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 307):
    
    # Assigning a Num to a Name (line 307):
    int_409822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 23), 'int')
    # Assigning a type to the variable 'info' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'info', int_409822)
    
    # Assigning a Name to a Name (line 308):
    
    # Assigning a Name to a Name (line 308):
    # Getting the type of 'False' (line 308)
    False_409823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 25), 'False')
    # Assigning a type to the variable 'ftflag' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'ftflag', False_409823)
    # SSA join for if statement (line 306)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 309):
    
    # Assigning a Subscript to a Name (line 309):
    
    # Obtaining the type of the subscript
    int_409824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 12), 'int')
    
    # Call to stoptest(...): (line 309)
    # Processing the call arguments (line 309)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 309)
    slice1_409826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 47), 'slice1', False)
    # Getting the type of 'work' (line 309)
    work_409827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 309)
    getitem___409828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 42), work_409827, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 309)
    subscript_call_result_409829 = invoke(stypy.reporting.localization.Localization(__file__, 309, 42), getitem___409828, slice1_409826)
    
    # Getting the type of 'b' (line 309)
    b_409830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 309)
    bnrm2_409831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 309)
    tol_409832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 66), 'tol', False)
    # Getting the type of 'info' (line 309)
    info_409833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 71), 'info', False)
    # Processing the call keyword arguments (line 309)
    kwargs_409834 = {}
    # Getting the type of 'stoptest' (line 309)
    stoptest_409825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 309)
    stoptest_call_result_409835 = invoke(stypy.reporting.localization.Localization(__file__, 309, 33), stoptest_409825, *[subscript_call_result_409829, b_409830, bnrm2_409831, tol_409832, info_409833], **kwargs_409834)
    
    # Obtaining the member '__getitem__' of a type (line 309)
    getitem___409836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 12), stoptest_call_result_409835, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 309)
    subscript_call_result_409837 = invoke(stypy.reporting.localization.Localization(__file__, 309, 12), getitem___409836, int_409824)
    
    # Assigning a type to the variable 'tuple_var_assignment_407956' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'tuple_var_assignment_407956', subscript_call_result_409837)
    
    # Assigning a Subscript to a Name (line 309):
    
    # Obtaining the type of the subscript
    int_409838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 12), 'int')
    
    # Call to stoptest(...): (line 309)
    # Processing the call arguments (line 309)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 309)
    slice1_409840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 47), 'slice1', False)
    # Getting the type of 'work' (line 309)
    work_409841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 309)
    getitem___409842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 42), work_409841, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 309)
    subscript_call_result_409843 = invoke(stypy.reporting.localization.Localization(__file__, 309, 42), getitem___409842, slice1_409840)
    
    # Getting the type of 'b' (line 309)
    b_409844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 309)
    bnrm2_409845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 309)
    tol_409846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 66), 'tol', False)
    # Getting the type of 'info' (line 309)
    info_409847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 71), 'info', False)
    # Processing the call keyword arguments (line 309)
    kwargs_409848 = {}
    # Getting the type of 'stoptest' (line 309)
    stoptest_409839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 309)
    stoptest_call_result_409849 = invoke(stypy.reporting.localization.Localization(__file__, 309, 33), stoptest_409839, *[subscript_call_result_409843, b_409844, bnrm2_409845, tol_409846, info_409847], **kwargs_409848)
    
    # Obtaining the member '__getitem__' of a type (line 309)
    getitem___409850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 12), stoptest_call_result_409849, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 309)
    subscript_call_result_409851 = invoke(stypy.reporting.localization.Localization(__file__, 309, 12), getitem___409850, int_409838)
    
    # Assigning a type to the variable 'tuple_var_assignment_407957' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'tuple_var_assignment_407957', subscript_call_result_409851)
    
    # Assigning a Subscript to a Name (line 309):
    
    # Obtaining the type of the subscript
    int_409852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 12), 'int')
    
    # Call to stoptest(...): (line 309)
    # Processing the call arguments (line 309)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 309)
    slice1_409854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 47), 'slice1', False)
    # Getting the type of 'work' (line 309)
    work_409855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 309)
    getitem___409856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 42), work_409855, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 309)
    subscript_call_result_409857 = invoke(stypy.reporting.localization.Localization(__file__, 309, 42), getitem___409856, slice1_409854)
    
    # Getting the type of 'b' (line 309)
    b_409858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 309)
    bnrm2_409859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 309)
    tol_409860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 66), 'tol', False)
    # Getting the type of 'info' (line 309)
    info_409861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 71), 'info', False)
    # Processing the call keyword arguments (line 309)
    kwargs_409862 = {}
    # Getting the type of 'stoptest' (line 309)
    stoptest_409853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 309)
    stoptest_call_result_409863 = invoke(stypy.reporting.localization.Localization(__file__, 309, 33), stoptest_409853, *[subscript_call_result_409857, b_409858, bnrm2_409859, tol_409860, info_409861], **kwargs_409862)
    
    # Obtaining the member '__getitem__' of a type (line 309)
    getitem___409864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 12), stoptest_call_result_409863, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 309)
    subscript_call_result_409865 = invoke(stypy.reporting.localization.Localization(__file__, 309, 12), getitem___409864, int_409852)
    
    # Assigning a type to the variable 'tuple_var_assignment_407958' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'tuple_var_assignment_407958', subscript_call_result_409865)
    
    # Assigning a Name to a Name (line 309):
    # Getting the type of 'tuple_var_assignment_407956' (line 309)
    tuple_var_assignment_407956_409866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'tuple_var_assignment_407956')
    # Assigning a type to the variable 'bnrm2' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'bnrm2', tuple_var_assignment_407956_409866)
    
    # Assigning a Name to a Name (line 309):
    # Getting the type of 'tuple_var_assignment_407957' (line 309)
    tuple_var_assignment_407957_409867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'tuple_var_assignment_407957')
    # Assigning a type to the variable 'resid' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'resid', tuple_var_assignment_407957_409867)
    
    # Assigning a Name to a Name (line 309):
    # Getting the type of 'tuple_var_assignment_407958' (line 309)
    tuple_var_assignment_407958_409868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'tuple_var_assignment_407958')
    # Assigning a type to the variable 'info' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 26), 'info', tuple_var_assignment_407958_409868)
    # SSA join for if statement (line 305)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 302)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 300)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 297)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 293)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 310):
    
    # Assigning a Num to a Name (line 310):
    int_409869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 15), 'int')
    # Assigning a type to the variable 'ijob' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'ijob', int_409869)
    # SSA join for while statement (line 285)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'info' (line 312)
    info_409870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 7), 'info')
    int_409871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 14), 'int')
    # Applying the binary operator '>' (line 312)
    result_gt_409872 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 7), '>', info_409870, int_409871)
    
    
    # Getting the type of 'iter_' (line 312)
    iter__409873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 20), 'iter_')
    # Getting the type of 'maxiter' (line 312)
    maxiter_409874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 29), 'maxiter')
    # Applying the binary operator '==' (line 312)
    result_eq_409875 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 20), '==', iter__409873, maxiter_409874)
    
    # Applying the binary operator 'and' (line 312)
    result_and_keyword_409876 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 7), 'and', result_gt_409872, result_eq_409875)
    
    # Getting the type of 'resid' (line 312)
    resid_409877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 41), 'resid')
    # Getting the type of 'tol' (line 312)
    tol_409878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 49), 'tol')
    # Applying the binary operator '>' (line 312)
    result_gt_409879 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 41), '>', resid_409877, tol_409878)
    
    # Applying the binary operator 'and' (line 312)
    result_and_keyword_409880 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 7), 'and', result_and_keyword_409876, result_gt_409879)
    
    # Testing the type of an if condition (line 312)
    if_condition_409881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 4), result_and_keyword_409880)
    # Assigning a type to the variable 'if_condition_409881' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'if_condition_409881', if_condition_409881)
    # SSA begins for if statement (line 312)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 314):
    
    # Assigning a Name to a Name (line 314):
    # Getting the type of 'iter_' (line 314)
    iter__409882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 15), 'iter_')
    # Assigning a type to the variable 'info' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'info', iter__409882)
    # SSA join for if statement (line 312)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 316)
    tuple_409883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 316)
    # Adding element type (line 316)
    
    # Call to postprocess(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'x' (line 316)
    x_409885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 23), 'x', False)
    # Processing the call keyword arguments (line 316)
    kwargs_409886 = {}
    # Getting the type of 'postprocess' (line 316)
    postprocess_409884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 11), 'postprocess', False)
    # Calling postprocess(args, kwargs) (line 316)
    postprocess_call_result_409887 = invoke(stypy.reporting.localization.Localization(__file__, 316, 11), postprocess_409884, *[x_409885], **kwargs_409886)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 11), tuple_409883, postprocess_call_result_409887)
    # Adding element type (line 316)
    # Getting the type of 'info' (line 316)
    info_409888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 27), 'info')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 11), tuple_409883, info_409888)
    
    # Assigning a type to the variable 'stypy_return_type' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'stypy_return_type', tuple_409883)
    
    # ################# End of 'cgs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cgs' in the type store
    # Getting the type of 'stypy_return_type' (line 259)
    stypy_return_type_409889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_409889)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cgs'
    return stypy_return_type_409889

# Assigning a type to the variable 'cgs' (line 259)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), 'cgs', cgs)

@norecursion
def gmres(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 320)
    None_409890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 19), 'None')
    float_409891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 29), 'float')
    # Getting the type of 'None' (line 320)
    None_409892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 43), 'None')
    # Getting the type of 'None' (line 320)
    None_409893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 57), 'None')
    # Getting the type of 'None' (line 320)
    None_409894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 65), 'None')
    # Getting the type of 'None' (line 320)
    None_409895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 80), 'None')
    # Getting the type of 'None' (line 320)
    None_409896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 93), 'None')
    defaults = [None_409890, float_409891, None_409892, None_409893, None_409894, None_409895, None_409896]
    # Create a new context for function 'gmres'
    module_type_store = module_type_store.open_function_context('gmres', 319, 0, False)
    
    # Passed parameters checking function
    gmres.stypy_localization = localization
    gmres.stypy_type_of_self = None
    gmres.stypy_type_store = module_type_store
    gmres.stypy_function_name = 'gmres'
    gmres.stypy_param_names_list = ['A', 'b', 'x0', 'tol', 'restart', 'maxiter', 'M', 'callback', 'restrt']
    gmres.stypy_varargs_param_name = None
    gmres.stypy_kwargs_param_name = None
    gmres.stypy_call_defaults = defaults
    gmres.stypy_call_varargs = varargs
    gmres.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gmres', ['A', 'b', 'x0', 'tol', 'restart', 'maxiter', 'M', 'callback', 'restrt'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gmres', localization, ['A', 'b', 'x0', 'tol', 'restart', 'maxiter', 'M', 'callback', 'restrt'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gmres(...)' code ##################

    str_409897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, (-1)), 'str', '\n    Use Generalized Minimal RESidual iteration to solve ``Ax = b``.\n\n    Parameters\n    ----------\n    A : {sparse matrix, dense matrix, LinearOperator}\n        The real or complex N-by-N matrix of the linear system.\n    b : {array, matrix}\n        Right hand side of the linear system. Has shape (N,) or (N,1).\n\n    Returns\n    -------\n    x : {array, matrix}\n        The converged solution.\n    info : int\n        Provides convergence information:\n          * 0  : successful exit\n          * >0 : convergence to tolerance not achieved, number of iterations\n          * <0 : illegal input or breakdown\n\n    Other parameters\n    ----------------\n    x0 : {array, matrix}\n        Starting guess for the solution (a vector of zeros by default).\n    tol : float\n        Tolerance to achieve. The algorithm terminates when either the relative\n        or the absolute residual is below `tol`.\n    restart : int, optional\n        Number of iterations between restarts. Larger values increase\n        iteration cost, but may be necessary for convergence.\n        Default is 20.\n    maxiter : int, optional\n        Maximum number of iterations (restart cycles).  Iteration will stop\n        after maxiter steps even if the specified tolerance has not been\n        achieved.\n    M : {sparse matrix, dense matrix, LinearOperator}\n        Inverse of the preconditioner of A.  M should approximate the\n        inverse of A and be easy to solve for (see Notes).  Effective\n        preconditioning dramatically improves the rate of convergence,\n        which implies that fewer iterations are needed to reach a given\n        error tolerance.  By default, no preconditioner is used.\n    callback : function\n        User-supplied function to call after each iteration.  It is called\n        as callback(rk), where rk is the current residual vector.\n    restrt : int, optional\n        DEPRECATED - use `restart` instead.\n\n    See Also\n    --------\n    LinearOperator\n\n    Notes\n    -----\n    A preconditioner, P, is chosen such that P is close to A but easy to solve\n    for. The preconditioner parameter required by this routine is\n    ``M = P^-1``. The inverse should preferably not be calculated\n    explicitly.  Rather, use the following template to produce M::\n\n      # Construct a linear operator that computes P^-1 * x.\n      import scipy.sparse.linalg as spla\n      M_x = lambda x: spla.spsolve(P, x)\n      M = spla.LinearOperator((n, n), M_x)\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import gmres\n    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)\n    >>> b = np.array([2, 4, -1], dtype=float)\n    >>> x, exitCode = gmres(A, b)\n    >>> print(exitCode)            # 0 indicates successful convergence\n    0\n    >>> np.allclose(A.dot(x), b)\n    True\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 398)
    # Getting the type of 'restrt' (line 398)
    restrt_409898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 7), 'restrt')
    # Getting the type of 'None' (line 398)
    None_409899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 17), 'None')
    
    (may_be_409900, more_types_in_union_409901) = may_be_none(restrt_409898, None_409899)

    if may_be_409900:

        if more_types_in_union_409901:
            # Runtime conditional SSA (line 398)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 399):
        
        # Assigning a Name to a Name (line 399):
        # Getting the type of 'restart' (line 399)
        restart_409902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 17), 'restart')
        # Assigning a type to the variable 'restrt' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'restrt', restart_409902)

        if more_types_in_union_409901:
            # Runtime conditional SSA for else branch (line 398)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_409900) or more_types_in_union_409901):
        
        # Type idiom detected: calculating its left and rigth part (line 400)
        # Getting the type of 'restart' (line 400)
        restart_409903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 9), 'restart')
        # Getting the type of 'None' (line 400)
        None_409904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 24), 'None')
        
        (may_be_409905, more_types_in_union_409906) = may_not_be_none(restart_409903, None_409904)

        if may_be_409905:

            if more_types_in_union_409906:
                # Runtime conditional SSA (line 400)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 401)
            # Processing the call arguments (line 401)
            str_409908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 25), 'str', "Cannot specify both restart and restrt keywords. Preferably use 'restart' only.")
            # Processing the call keyword arguments (line 401)
            kwargs_409909 = {}
            # Getting the type of 'ValueError' (line 401)
            ValueError_409907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 14), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 401)
            ValueError_call_result_409910 = invoke(stypy.reporting.localization.Localization(__file__, 401, 14), ValueError_409907, *[str_409908], **kwargs_409909)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 401, 8), ValueError_call_result_409910, 'raise parameter', BaseException)

            if more_types_in_union_409906:
                # SSA join for if statement (line 400)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_409900 and more_types_in_union_409901):
            # SSA join for if statement (line 398)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 404):
    
    # Assigning a Subscript to a Name (line 404):
    
    # Obtaining the type of the subscript
    int_409911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 4), 'int')
    
    # Call to make_system(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'A' (line 404)
    A_409913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 41), 'A', False)
    # Getting the type of 'M' (line 404)
    M_409914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 44), 'M', False)
    # Getting the type of 'x0' (line 404)
    x0_409915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 47), 'x0', False)
    # Getting the type of 'b' (line 404)
    b_409916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 51), 'b', False)
    # Processing the call keyword arguments (line 404)
    kwargs_409917 = {}
    # Getting the type of 'make_system' (line 404)
    make_system_409912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 29), 'make_system', False)
    # Calling make_system(args, kwargs) (line 404)
    make_system_call_result_409918 = invoke(stypy.reporting.localization.Localization(__file__, 404, 29), make_system_409912, *[A_409913, M_409914, x0_409915, b_409916], **kwargs_409917)
    
    # Obtaining the member '__getitem__' of a type (line 404)
    getitem___409919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 4), make_system_call_result_409918, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 404)
    subscript_call_result_409920 = invoke(stypy.reporting.localization.Localization(__file__, 404, 4), getitem___409919, int_409911)
    
    # Assigning a type to the variable 'tuple_var_assignment_407959' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_var_assignment_407959', subscript_call_result_409920)
    
    # Assigning a Subscript to a Name (line 404):
    
    # Obtaining the type of the subscript
    int_409921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 4), 'int')
    
    # Call to make_system(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'A' (line 404)
    A_409923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 41), 'A', False)
    # Getting the type of 'M' (line 404)
    M_409924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 44), 'M', False)
    # Getting the type of 'x0' (line 404)
    x0_409925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 47), 'x0', False)
    # Getting the type of 'b' (line 404)
    b_409926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 51), 'b', False)
    # Processing the call keyword arguments (line 404)
    kwargs_409927 = {}
    # Getting the type of 'make_system' (line 404)
    make_system_409922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 29), 'make_system', False)
    # Calling make_system(args, kwargs) (line 404)
    make_system_call_result_409928 = invoke(stypy.reporting.localization.Localization(__file__, 404, 29), make_system_409922, *[A_409923, M_409924, x0_409925, b_409926], **kwargs_409927)
    
    # Obtaining the member '__getitem__' of a type (line 404)
    getitem___409929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 4), make_system_call_result_409928, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 404)
    subscript_call_result_409930 = invoke(stypy.reporting.localization.Localization(__file__, 404, 4), getitem___409929, int_409921)
    
    # Assigning a type to the variable 'tuple_var_assignment_407960' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_var_assignment_407960', subscript_call_result_409930)
    
    # Assigning a Subscript to a Name (line 404):
    
    # Obtaining the type of the subscript
    int_409931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 4), 'int')
    
    # Call to make_system(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'A' (line 404)
    A_409933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 41), 'A', False)
    # Getting the type of 'M' (line 404)
    M_409934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 44), 'M', False)
    # Getting the type of 'x0' (line 404)
    x0_409935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 47), 'x0', False)
    # Getting the type of 'b' (line 404)
    b_409936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 51), 'b', False)
    # Processing the call keyword arguments (line 404)
    kwargs_409937 = {}
    # Getting the type of 'make_system' (line 404)
    make_system_409932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 29), 'make_system', False)
    # Calling make_system(args, kwargs) (line 404)
    make_system_call_result_409938 = invoke(stypy.reporting.localization.Localization(__file__, 404, 29), make_system_409932, *[A_409933, M_409934, x0_409935, b_409936], **kwargs_409937)
    
    # Obtaining the member '__getitem__' of a type (line 404)
    getitem___409939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 4), make_system_call_result_409938, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 404)
    subscript_call_result_409940 = invoke(stypy.reporting.localization.Localization(__file__, 404, 4), getitem___409939, int_409931)
    
    # Assigning a type to the variable 'tuple_var_assignment_407961' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_var_assignment_407961', subscript_call_result_409940)
    
    # Assigning a Subscript to a Name (line 404):
    
    # Obtaining the type of the subscript
    int_409941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 4), 'int')
    
    # Call to make_system(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'A' (line 404)
    A_409943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 41), 'A', False)
    # Getting the type of 'M' (line 404)
    M_409944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 44), 'M', False)
    # Getting the type of 'x0' (line 404)
    x0_409945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 47), 'x0', False)
    # Getting the type of 'b' (line 404)
    b_409946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 51), 'b', False)
    # Processing the call keyword arguments (line 404)
    kwargs_409947 = {}
    # Getting the type of 'make_system' (line 404)
    make_system_409942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 29), 'make_system', False)
    # Calling make_system(args, kwargs) (line 404)
    make_system_call_result_409948 = invoke(stypy.reporting.localization.Localization(__file__, 404, 29), make_system_409942, *[A_409943, M_409944, x0_409945, b_409946], **kwargs_409947)
    
    # Obtaining the member '__getitem__' of a type (line 404)
    getitem___409949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 4), make_system_call_result_409948, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 404)
    subscript_call_result_409950 = invoke(stypy.reporting.localization.Localization(__file__, 404, 4), getitem___409949, int_409941)
    
    # Assigning a type to the variable 'tuple_var_assignment_407962' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_var_assignment_407962', subscript_call_result_409950)
    
    # Assigning a Subscript to a Name (line 404):
    
    # Obtaining the type of the subscript
    int_409951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 4), 'int')
    
    # Call to make_system(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'A' (line 404)
    A_409953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 41), 'A', False)
    # Getting the type of 'M' (line 404)
    M_409954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 44), 'M', False)
    # Getting the type of 'x0' (line 404)
    x0_409955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 47), 'x0', False)
    # Getting the type of 'b' (line 404)
    b_409956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 51), 'b', False)
    # Processing the call keyword arguments (line 404)
    kwargs_409957 = {}
    # Getting the type of 'make_system' (line 404)
    make_system_409952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 29), 'make_system', False)
    # Calling make_system(args, kwargs) (line 404)
    make_system_call_result_409958 = invoke(stypy.reporting.localization.Localization(__file__, 404, 29), make_system_409952, *[A_409953, M_409954, x0_409955, b_409956], **kwargs_409957)
    
    # Obtaining the member '__getitem__' of a type (line 404)
    getitem___409959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 4), make_system_call_result_409958, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 404)
    subscript_call_result_409960 = invoke(stypy.reporting.localization.Localization(__file__, 404, 4), getitem___409959, int_409951)
    
    # Assigning a type to the variable 'tuple_var_assignment_407963' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_var_assignment_407963', subscript_call_result_409960)
    
    # Assigning a Name to a Name (line 404):
    # Getting the type of 'tuple_var_assignment_407959' (line 404)
    tuple_var_assignment_407959_409961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_var_assignment_407959')
    # Assigning a type to the variable 'A' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'A', tuple_var_assignment_407959_409961)
    
    # Assigning a Name to a Name (line 404):
    # Getting the type of 'tuple_var_assignment_407960' (line 404)
    tuple_var_assignment_407960_409962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_var_assignment_407960')
    # Assigning a type to the variable 'M' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 7), 'M', tuple_var_assignment_407960_409962)
    
    # Assigning a Name to a Name (line 404):
    # Getting the type of 'tuple_var_assignment_407961' (line 404)
    tuple_var_assignment_407961_409963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_var_assignment_407961')
    # Assigning a type to the variable 'x' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 10), 'x', tuple_var_assignment_407961_409963)
    
    # Assigning a Name to a Name (line 404):
    # Getting the type of 'tuple_var_assignment_407962' (line 404)
    tuple_var_assignment_407962_409964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_var_assignment_407962')
    # Assigning a type to the variable 'b' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 13), 'b', tuple_var_assignment_407962_409964)
    
    # Assigning a Name to a Name (line 404):
    # Getting the type of 'tuple_var_assignment_407963' (line 404)
    tuple_var_assignment_407963_409965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_var_assignment_407963')
    # Assigning a type to the variable 'postprocess' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'postprocess', tuple_var_assignment_407963_409965)
    
    # Assigning a Call to a Name (line 406):
    
    # Assigning a Call to a Name (line 406):
    
    # Call to len(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'b' (line 406)
    b_409967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'b', False)
    # Processing the call keyword arguments (line 406)
    kwargs_409968 = {}
    # Getting the type of 'len' (line 406)
    len_409966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'len', False)
    # Calling len(args, kwargs) (line 406)
    len_call_result_409969 = invoke(stypy.reporting.localization.Localization(__file__, 406, 8), len_409966, *[b_409967], **kwargs_409968)
    
    # Assigning a type to the variable 'n' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'n', len_call_result_409969)
    
    # Type idiom detected: calculating its left and rigth part (line 407)
    # Getting the type of 'maxiter' (line 407)
    maxiter_409970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 7), 'maxiter')
    # Getting the type of 'None' (line 407)
    None_409971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 18), 'None')
    
    (may_be_409972, more_types_in_union_409973) = may_be_none(maxiter_409970, None_409971)

    if may_be_409972:

        if more_types_in_union_409973:
            # Runtime conditional SSA (line 407)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 408):
        
        # Assigning a BinOp to a Name (line 408):
        # Getting the type of 'n' (line 408)
        n_409974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 18), 'n')
        int_409975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 20), 'int')
        # Applying the binary operator '*' (line 408)
        result_mul_409976 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 18), '*', n_409974, int_409975)
        
        # Assigning a type to the variable 'maxiter' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'maxiter', result_mul_409976)

        if more_types_in_union_409973:
            # SSA join for if statement (line 407)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 410)
    # Getting the type of 'restrt' (line 410)
    restrt_409977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 7), 'restrt')
    # Getting the type of 'None' (line 410)
    None_409978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 17), 'None')
    
    (may_be_409979, more_types_in_union_409980) = may_be_none(restrt_409977, None_409978)

    if may_be_409979:

        if more_types_in_union_409980:
            # Runtime conditional SSA (line 410)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 411):
        
        # Assigning a Num to a Name (line 411):
        int_409981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 17), 'int')
        # Assigning a type to the variable 'restrt' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'restrt', int_409981)

        if more_types_in_union_409980:
            # SSA join for if statement (line 410)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 412):
    
    # Assigning a Call to a Name (line 412):
    
    # Call to min(...): (line 412)
    # Processing the call arguments (line 412)
    # Getting the type of 'restrt' (line 412)
    restrt_409983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 17), 'restrt', False)
    # Getting the type of 'n' (line 412)
    n_409984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 25), 'n', False)
    # Processing the call keyword arguments (line 412)
    kwargs_409985 = {}
    # Getting the type of 'min' (line 412)
    min_409982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 13), 'min', False)
    # Calling min(args, kwargs) (line 412)
    min_call_result_409986 = invoke(stypy.reporting.localization.Localization(__file__, 412, 13), min_409982, *[restrt_409983, n_409984], **kwargs_409985)
    
    # Assigning a type to the variable 'restrt' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'restrt', min_call_result_409986)
    
    # Assigning a Attribute to a Name (line 414):
    
    # Assigning a Attribute to a Name (line 414):
    # Getting the type of 'A' (line 414)
    A_409987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 13), 'A')
    # Obtaining the member 'matvec' of a type (line 414)
    matvec_409988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 13), A_409987, 'matvec')
    # Assigning a type to the variable 'matvec' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'matvec', matvec_409988)
    
    # Assigning a Attribute to a Name (line 415):
    
    # Assigning a Attribute to a Name (line 415):
    # Getting the type of 'M' (line 415)
    M_409989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 13), 'M')
    # Obtaining the member 'matvec' of a type (line 415)
    matvec_409990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 13), M_409989, 'matvec')
    # Assigning a type to the variable 'psolve' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'psolve', matvec_409990)
    
    # Assigning a Subscript to a Name (line 416):
    
    # Assigning a Subscript to a Name (line 416):
    
    # Obtaining the type of the subscript
    # Getting the type of 'x' (line 416)
    x_409991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 21), 'x')
    # Obtaining the member 'dtype' of a type (line 416)
    dtype_409992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 21), x_409991, 'dtype')
    # Obtaining the member 'char' of a type (line 416)
    char_409993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 21), dtype_409992, 'char')
    # Getting the type of '_type_conv' (line 416)
    _type_conv_409994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 10), '_type_conv')
    # Obtaining the member '__getitem__' of a type (line 416)
    getitem___409995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 10), _type_conv_409994, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 416)
    subscript_call_result_409996 = invoke(stypy.reporting.localization.Localization(__file__, 416, 10), getitem___409995, char_409993)
    
    # Assigning a type to the variable 'ltr' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'ltr', subscript_call_result_409996)
    
    # Assigning a Call to a Name (line 417):
    
    # Assigning a Call to a Name (line 417):
    
    # Call to getattr(...): (line 417)
    # Processing the call arguments (line 417)
    # Getting the type of '_iterative' (line 417)
    _iterative_409998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 21), '_iterative', False)
    # Getting the type of 'ltr' (line 417)
    ltr_409999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 33), 'ltr', False)
    str_410000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 39), 'str', 'gmresrevcom')
    # Applying the binary operator '+' (line 417)
    result_add_410001 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 33), '+', ltr_409999, str_410000)
    
    # Processing the call keyword arguments (line 417)
    kwargs_410002 = {}
    # Getting the type of 'getattr' (line 417)
    getattr_409997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 13), 'getattr', False)
    # Calling getattr(args, kwargs) (line 417)
    getattr_call_result_410003 = invoke(stypy.reporting.localization.Localization(__file__, 417, 13), getattr_409997, *[_iterative_409998, result_add_410001], **kwargs_410002)
    
    # Assigning a type to the variable 'revcom' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'revcom', getattr_call_result_410003)
    
    # Assigning a Call to a Name (line 418):
    
    # Assigning a Call to a Name (line 418):
    
    # Call to getattr(...): (line 418)
    # Processing the call arguments (line 418)
    # Getting the type of '_iterative' (line 418)
    _iterative_410005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 23), '_iterative', False)
    # Getting the type of 'ltr' (line 418)
    ltr_410006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 35), 'ltr', False)
    str_410007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 41), 'str', 'stoptest2')
    # Applying the binary operator '+' (line 418)
    result_add_410008 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 35), '+', ltr_410006, str_410007)
    
    # Processing the call keyword arguments (line 418)
    kwargs_410009 = {}
    # Getting the type of 'getattr' (line 418)
    getattr_410004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'getattr', False)
    # Calling getattr(args, kwargs) (line 418)
    getattr_call_result_410010 = invoke(stypy.reporting.localization.Localization(__file__, 418, 15), getattr_410004, *[_iterative_410005, result_add_410008], **kwargs_410009)
    
    # Assigning a type to the variable 'stoptest' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'stoptest', getattr_call_result_410010)
    
    # Assigning a Name to a Name (line 420):
    
    # Assigning a Name to a Name (line 420):
    # Getting the type of 'tol' (line 420)
    tol_410011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'tol')
    # Assigning a type to the variable 'resid' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'resid', tol_410011)
    
    # Assigning a Num to a Name (line 421):
    
    # Assigning a Num to a Name (line 421):
    int_410012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 11), 'int')
    # Assigning a type to the variable 'ndx1' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'ndx1', int_410012)
    
    # Assigning a Num to a Name (line 422):
    
    # Assigning a Num to a Name (line 422):
    int_410013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 11), 'int')
    # Assigning a type to the variable 'ndx2' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'ndx2', int_410013)
    
    # Assigning a Call to a Name (line 424):
    
    # Assigning a Call to a Name (line 424):
    
    # Call to _aligned_zeros(...): (line 424)
    # Processing the call arguments (line 424)
    int_410015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 27), 'int')
    # Getting the type of 'restrt' (line 424)
    restrt_410016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 29), 'restrt', False)
    # Applying the binary operator '+' (line 424)
    result_add_410017 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 27), '+', int_410015, restrt_410016)
    
    # Getting the type of 'n' (line 424)
    n_410018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 37), 'n', False)
    # Applying the binary operator '*' (line 424)
    result_mul_410019 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 26), '*', result_add_410017, n_410018)
    
    # Processing the call keyword arguments (line 424)
    # Getting the type of 'x' (line 424)
    x_410020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 45), 'x', False)
    # Obtaining the member 'dtype' of a type (line 424)
    dtype_410021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 45), x_410020, 'dtype')
    keyword_410022 = dtype_410021
    kwargs_410023 = {'dtype': keyword_410022}
    # Getting the type of '_aligned_zeros' (line 424)
    _aligned_zeros_410014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 11), '_aligned_zeros', False)
    # Calling _aligned_zeros(args, kwargs) (line 424)
    _aligned_zeros_call_result_410024 = invoke(stypy.reporting.localization.Localization(__file__, 424, 11), _aligned_zeros_410014, *[result_mul_410019], **kwargs_410023)
    
    # Assigning a type to the variable 'work' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'work', _aligned_zeros_call_result_410024)
    
    # Assigning a Call to a Name (line 425):
    
    # Assigning a Call to a Name (line 425):
    
    # Call to _aligned_zeros(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'restrt' (line 425)
    restrt_410026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 28), 'restrt', False)
    int_410027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 35), 'int')
    # Applying the binary operator '+' (line 425)
    result_add_410028 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 28), '+', restrt_410026, int_410027)
    
    int_410029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 39), 'int')
    # Getting the type of 'restrt' (line 425)
    restrt_410030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 41), 'restrt', False)
    # Applying the binary operator '*' (line 425)
    result_mul_410031 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 39), '*', int_410029, restrt_410030)
    
    int_410032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 48), 'int')
    # Applying the binary operator '+' (line 425)
    result_add_410033 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 39), '+', result_mul_410031, int_410032)
    
    # Applying the binary operator '*' (line 425)
    result_mul_410034 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 27), '*', result_add_410028, result_add_410033)
    
    # Processing the call keyword arguments (line 425)
    # Getting the type of 'x' (line 425)
    x_410035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 57), 'x', False)
    # Obtaining the member 'dtype' of a type (line 425)
    dtype_410036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 57), x_410035, 'dtype')
    keyword_410037 = dtype_410036
    kwargs_410038 = {'dtype': keyword_410037}
    # Getting the type of '_aligned_zeros' (line 425)
    _aligned_zeros_410025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), '_aligned_zeros', False)
    # Calling _aligned_zeros(args, kwargs) (line 425)
    _aligned_zeros_call_result_410039 = invoke(stypy.reporting.localization.Localization(__file__, 425, 12), _aligned_zeros_410025, *[result_mul_410034], **kwargs_410038)
    
    # Assigning a type to the variable 'work2' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'work2', _aligned_zeros_call_result_410039)
    
    # Assigning a Num to a Name (line 426):
    
    # Assigning a Num to a Name (line 426):
    int_410040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 11), 'int')
    # Assigning a type to the variable 'ijob' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'ijob', int_410040)
    
    # Assigning a Num to a Name (line 427):
    
    # Assigning a Num to a Name (line 427):
    int_410041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 11), 'int')
    # Assigning a type to the variable 'info' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'info', int_410041)
    
    # Assigning a Name to a Name (line 428):
    
    # Assigning a Name to a Name (line 428):
    # Getting the type of 'True' (line 428)
    True_410042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 13), 'True')
    # Assigning a type to the variable 'ftflag' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'ftflag', True_410042)
    
    # Assigning a Num to a Name (line 429):
    
    # Assigning a Num to a Name (line 429):
    float_410043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 12), 'float')
    # Assigning a type to the variable 'bnrm2' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'bnrm2', float_410043)
    
    # Assigning a Name to a Name (line 430):
    
    # Assigning a Name to a Name (line 430):
    # Getting the type of 'maxiter' (line 430)
    maxiter_410044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'maxiter')
    # Assigning a type to the variable 'iter_' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'iter_', maxiter_410044)
    
    # Assigning a Name to a Name (line 431):
    
    # Assigning a Name to a Name (line 431):
    # Getting the type of 'ijob' (line 431)
    ijob_410045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 15), 'ijob')
    # Assigning a type to the variable 'old_ijob' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'old_ijob', ijob_410045)
    
    # Assigning a Name to a Name (line 432):
    
    # Assigning a Name to a Name (line 432):
    # Getting the type of 'True' (line 432)
    True_410046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 17), 'True')
    # Assigning a type to the variable 'first_pass' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'first_pass', True_410046)
    
    # Assigning a Name to a Name (line 433):
    
    # Assigning a Name to a Name (line 433):
    # Getting the type of 'False' (line 433)
    False_410047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 18), 'False')
    # Assigning a type to the variable 'resid_ready' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'resid_ready', False_410047)
    
    # Assigning a Num to a Name (line 434):
    
    # Assigning a Num to a Name (line 434):
    int_410048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 15), 'int')
    # Assigning a type to the variable 'iter_num' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'iter_num', int_410048)
    
    # Getting the type of 'True' (line 435)
    True_410049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 10), 'True')
    # Testing the type of an if condition (line 435)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 4), True_410049)
    # SSA begins for while statement (line 435)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Name to a Name (line 436):
    
    # Assigning a Name to a Name (line 436):
    # Getting the type of 'iter_' (line 436)
    iter__410050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 18), 'iter_')
    # Assigning a type to the variable 'olditer' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'olditer', iter__410050)
    
    # Assigning a Call to a Tuple (line 437):
    
    # Assigning a Subscript to a Name (line 437):
    
    # Obtaining the type of the subscript
    int_410051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
    
    # Call to revcom(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'b' (line 438)
    b_410053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'b', False)
    # Getting the type of 'x' (line 438)
    x_410054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 21), 'x', False)
    # Getting the type of 'restrt' (line 438)
    restrt_410055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 24), 'restrt', False)
    # Getting the type of 'work' (line 438)
    work_410056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 32), 'work', False)
    # Getting the type of 'work2' (line 438)
    work2_410057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 38), 'work2', False)
    # Getting the type of 'iter_' (line 438)
    iter__410058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 45), 'iter_', False)
    # Getting the type of 'resid' (line 438)
    resid_410059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 52), 'resid', False)
    # Getting the type of 'info' (line 438)
    info_410060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 59), 'info', False)
    # Getting the type of 'ndx1' (line 438)
    ndx1_410061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 65), 'ndx1', False)
    # Getting the type of 'ndx2' (line 438)
    ndx2_410062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 71), 'ndx2', False)
    # Getting the type of 'ijob' (line 438)
    ijob_410063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 77), 'ijob', False)
    # Processing the call keyword arguments (line 438)
    kwargs_410064 = {}
    # Getting the type of 'revcom' (line 438)
    revcom_410052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 438)
    revcom_call_result_410065 = invoke(stypy.reporting.localization.Localization(__file__, 438, 11), revcom_410052, *[b_410053, x_410054, restrt_410055, work_410056, work2_410057, iter__410058, resid_410059, info_410060, ndx1_410061, ndx2_410062, ijob_410063], **kwargs_410064)
    
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___410066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), revcom_call_result_410065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_410067 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___410066, int_410051)
    
    # Assigning a type to the variable 'tuple_var_assignment_407964' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407964', subscript_call_result_410067)
    
    # Assigning a Subscript to a Name (line 437):
    
    # Obtaining the type of the subscript
    int_410068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
    
    # Call to revcom(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'b' (line 438)
    b_410070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'b', False)
    # Getting the type of 'x' (line 438)
    x_410071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 21), 'x', False)
    # Getting the type of 'restrt' (line 438)
    restrt_410072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 24), 'restrt', False)
    # Getting the type of 'work' (line 438)
    work_410073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 32), 'work', False)
    # Getting the type of 'work2' (line 438)
    work2_410074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 38), 'work2', False)
    # Getting the type of 'iter_' (line 438)
    iter__410075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 45), 'iter_', False)
    # Getting the type of 'resid' (line 438)
    resid_410076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 52), 'resid', False)
    # Getting the type of 'info' (line 438)
    info_410077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 59), 'info', False)
    # Getting the type of 'ndx1' (line 438)
    ndx1_410078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 65), 'ndx1', False)
    # Getting the type of 'ndx2' (line 438)
    ndx2_410079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 71), 'ndx2', False)
    # Getting the type of 'ijob' (line 438)
    ijob_410080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 77), 'ijob', False)
    # Processing the call keyword arguments (line 438)
    kwargs_410081 = {}
    # Getting the type of 'revcom' (line 438)
    revcom_410069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 438)
    revcom_call_result_410082 = invoke(stypy.reporting.localization.Localization(__file__, 438, 11), revcom_410069, *[b_410070, x_410071, restrt_410072, work_410073, work2_410074, iter__410075, resid_410076, info_410077, ndx1_410078, ndx2_410079, ijob_410080], **kwargs_410081)
    
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___410083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), revcom_call_result_410082, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_410084 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___410083, int_410068)
    
    # Assigning a type to the variable 'tuple_var_assignment_407965' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407965', subscript_call_result_410084)
    
    # Assigning a Subscript to a Name (line 437):
    
    # Obtaining the type of the subscript
    int_410085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
    
    # Call to revcom(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'b' (line 438)
    b_410087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'b', False)
    # Getting the type of 'x' (line 438)
    x_410088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 21), 'x', False)
    # Getting the type of 'restrt' (line 438)
    restrt_410089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 24), 'restrt', False)
    # Getting the type of 'work' (line 438)
    work_410090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 32), 'work', False)
    # Getting the type of 'work2' (line 438)
    work2_410091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 38), 'work2', False)
    # Getting the type of 'iter_' (line 438)
    iter__410092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 45), 'iter_', False)
    # Getting the type of 'resid' (line 438)
    resid_410093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 52), 'resid', False)
    # Getting the type of 'info' (line 438)
    info_410094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 59), 'info', False)
    # Getting the type of 'ndx1' (line 438)
    ndx1_410095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 65), 'ndx1', False)
    # Getting the type of 'ndx2' (line 438)
    ndx2_410096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 71), 'ndx2', False)
    # Getting the type of 'ijob' (line 438)
    ijob_410097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 77), 'ijob', False)
    # Processing the call keyword arguments (line 438)
    kwargs_410098 = {}
    # Getting the type of 'revcom' (line 438)
    revcom_410086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 438)
    revcom_call_result_410099 = invoke(stypy.reporting.localization.Localization(__file__, 438, 11), revcom_410086, *[b_410087, x_410088, restrt_410089, work_410090, work2_410091, iter__410092, resid_410093, info_410094, ndx1_410095, ndx2_410096, ijob_410097], **kwargs_410098)
    
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___410100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), revcom_call_result_410099, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_410101 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___410100, int_410085)
    
    # Assigning a type to the variable 'tuple_var_assignment_407966' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407966', subscript_call_result_410101)
    
    # Assigning a Subscript to a Name (line 437):
    
    # Obtaining the type of the subscript
    int_410102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
    
    # Call to revcom(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'b' (line 438)
    b_410104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'b', False)
    # Getting the type of 'x' (line 438)
    x_410105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 21), 'x', False)
    # Getting the type of 'restrt' (line 438)
    restrt_410106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 24), 'restrt', False)
    # Getting the type of 'work' (line 438)
    work_410107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 32), 'work', False)
    # Getting the type of 'work2' (line 438)
    work2_410108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 38), 'work2', False)
    # Getting the type of 'iter_' (line 438)
    iter__410109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 45), 'iter_', False)
    # Getting the type of 'resid' (line 438)
    resid_410110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 52), 'resid', False)
    # Getting the type of 'info' (line 438)
    info_410111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 59), 'info', False)
    # Getting the type of 'ndx1' (line 438)
    ndx1_410112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 65), 'ndx1', False)
    # Getting the type of 'ndx2' (line 438)
    ndx2_410113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 71), 'ndx2', False)
    # Getting the type of 'ijob' (line 438)
    ijob_410114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 77), 'ijob', False)
    # Processing the call keyword arguments (line 438)
    kwargs_410115 = {}
    # Getting the type of 'revcom' (line 438)
    revcom_410103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 438)
    revcom_call_result_410116 = invoke(stypy.reporting.localization.Localization(__file__, 438, 11), revcom_410103, *[b_410104, x_410105, restrt_410106, work_410107, work2_410108, iter__410109, resid_410110, info_410111, ndx1_410112, ndx2_410113, ijob_410114], **kwargs_410115)
    
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___410117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), revcom_call_result_410116, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_410118 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___410117, int_410102)
    
    # Assigning a type to the variable 'tuple_var_assignment_407967' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407967', subscript_call_result_410118)
    
    # Assigning a Subscript to a Name (line 437):
    
    # Obtaining the type of the subscript
    int_410119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
    
    # Call to revcom(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'b' (line 438)
    b_410121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'b', False)
    # Getting the type of 'x' (line 438)
    x_410122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 21), 'x', False)
    # Getting the type of 'restrt' (line 438)
    restrt_410123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 24), 'restrt', False)
    # Getting the type of 'work' (line 438)
    work_410124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 32), 'work', False)
    # Getting the type of 'work2' (line 438)
    work2_410125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 38), 'work2', False)
    # Getting the type of 'iter_' (line 438)
    iter__410126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 45), 'iter_', False)
    # Getting the type of 'resid' (line 438)
    resid_410127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 52), 'resid', False)
    # Getting the type of 'info' (line 438)
    info_410128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 59), 'info', False)
    # Getting the type of 'ndx1' (line 438)
    ndx1_410129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 65), 'ndx1', False)
    # Getting the type of 'ndx2' (line 438)
    ndx2_410130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 71), 'ndx2', False)
    # Getting the type of 'ijob' (line 438)
    ijob_410131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 77), 'ijob', False)
    # Processing the call keyword arguments (line 438)
    kwargs_410132 = {}
    # Getting the type of 'revcom' (line 438)
    revcom_410120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 438)
    revcom_call_result_410133 = invoke(stypy.reporting.localization.Localization(__file__, 438, 11), revcom_410120, *[b_410121, x_410122, restrt_410123, work_410124, work2_410125, iter__410126, resid_410127, info_410128, ndx1_410129, ndx2_410130, ijob_410131], **kwargs_410132)
    
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___410134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), revcom_call_result_410133, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_410135 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___410134, int_410119)
    
    # Assigning a type to the variable 'tuple_var_assignment_407968' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407968', subscript_call_result_410135)
    
    # Assigning a Subscript to a Name (line 437):
    
    # Obtaining the type of the subscript
    int_410136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
    
    # Call to revcom(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'b' (line 438)
    b_410138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'b', False)
    # Getting the type of 'x' (line 438)
    x_410139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 21), 'x', False)
    # Getting the type of 'restrt' (line 438)
    restrt_410140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 24), 'restrt', False)
    # Getting the type of 'work' (line 438)
    work_410141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 32), 'work', False)
    # Getting the type of 'work2' (line 438)
    work2_410142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 38), 'work2', False)
    # Getting the type of 'iter_' (line 438)
    iter__410143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 45), 'iter_', False)
    # Getting the type of 'resid' (line 438)
    resid_410144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 52), 'resid', False)
    # Getting the type of 'info' (line 438)
    info_410145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 59), 'info', False)
    # Getting the type of 'ndx1' (line 438)
    ndx1_410146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 65), 'ndx1', False)
    # Getting the type of 'ndx2' (line 438)
    ndx2_410147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 71), 'ndx2', False)
    # Getting the type of 'ijob' (line 438)
    ijob_410148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 77), 'ijob', False)
    # Processing the call keyword arguments (line 438)
    kwargs_410149 = {}
    # Getting the type of 'revcom' (line 438)
    revcom_410137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 438)
    revcom_call_result_410150 = invoke(stypy.reporting.localization.Localization(__file__, 438, 11), revcom_410137, *[b_410138, x_410139, restrt_410140, work_410141, work2_410142, iter__410143, resid_410144, info_410145, ndx1_410146, ndx2_410147, ijob_410148], **kwargs_410149)
    
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___410151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), revcom_call_result_410150, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_410152 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___410151, int_410136)
    
    # Assigning a type to the variable 'tuple_var_assignment_407969' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407969', subscript_call_result_410152)
    
    # Assigning a Subscript to a Name (line 437):
    
    # Obtaining the type of the subscript
    int_410153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
    
    # Call to revcom(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'b' (line 438)
    b_410155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'b', False)
    # Getting the type of 'x' (line 438)
    x_410156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 21), 'x', False)
    # Getting the type of 'restrt' (line 438)
    restrt_410157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 24), 'restrt', False)
    # Getting the type of 'work' (line 438)
    work_410158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 32), 'work', False)
    # Getting the type of 'work2' (line 438)
    work2_410159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 38), 'work2', False)
    # Getting the type of 'iter_' (line 438)
    iter__410160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 45), 'iter_', False)
    # Getting the type of 'resid' (line 438)
    resid_410161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 52), 'resid', False)
    # Getting the type of 'info' (line 438)
    info_410162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 59), 'info', False)
    # Getting the type of 'ndx1' (line 438)
    ndx1_410163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 65), 'ndx1', False)
    # Getting the type of 'ndx2' (line 438)
    ndx2_410164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 71), 'ndx2', False)
    # Getting the type of 'ijob' (line 438)
    ijob_410165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 77), 'ijob', False)
    # Processing the call keyword arguments (line 438)
    kwargs_410166 = {}
    # Getting the type of 'revcom' (line 438)
    revcom_410154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 438)
    revcom_call_result_410167 = invoke(stypy.reporting.localization.Localization(__file__, 438, 11), revcom_410154, *[b_410155, x_410156, restrt_410157, work_410158, work2_410159, iter__410160, resid_410161, info_410162, ndx1_410163, ndx2_410164, ijob_410165], **kwargs_410166)
    
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___410168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), revcom_call_result_410167, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_410169 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___410168, int_410153)
    
    # Assigning a type to the variable 'tuple_var_assignment_407970' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407970', subscript_call_result_410169)
    
    # Assigning a Subscript to a Name (line 437):
    
    # Obtaining the type of the subscript
    int_410170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
    
    # Call to revcom(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'b' (line 438)
    b_410172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'b', False)
    # Getting the type of 'x' (line 438)
    x_410173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 21), 'x', False)
    # Getting the type of 'restrt' (line 438)
    restrt_410174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 24), 'restrt', False)
    # Getting the type of 'work' (line 438)
    work_410175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 32), 'work', False)
    # Getting the type of 'work2' (line 438)
    work2_410176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 38), 'work2', False)
    # Getting the type of 'iter_' (line 438)
    iter__410177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 45), 'iter_', False)
    # Getting the type of 'resid' (line 438)
    resid_410178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 52), 'resid', False)
    # Getting the type of 'info' (line 438)
    info_410179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 59), 'info', False)
    # Getting the type of 'ndx1' (line 438)
    ndx1_410180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 65), 'ndx1', False)
    # Getting the type of 'ndx2' (line 438)
    ndx2_410181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 71), 'ndx2', False)
    # Getting the type of 'ijob' (line 438)
    ijob_410182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 77), 'ijob', False)
    # Processing the call keyword arguments (line 438)
    kwargs_410183 = {}
    # Getting the type of 'revcom' (line 438)
    revcom_410171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 438)
    revcom_call_result_410184 = invoke(stypy.reporting.localization.Localization(__file__, 438, 11), revcom_410171, *[b_410172, x_410173, restrt_410174, work_410175, work2_410176, iter__410177, resid_410178, info_410179, ndx1_410180, ndx2_410181, ijob_410182], **kwargs_410183)
    
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___410185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), revcom_call_result_410184, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_410186 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___410185, int_410170)
    
    # Assigning a type to the variable 'tuple_var_assignment_407971' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407971', subscript_call_result_410186)
    
    # Assigning a Subscript to a Name (line 437):
    
    # Obtaining the type of the subscript
    int_410187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
    
    # Call to revcom(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'b' (line 438)
    b_410189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'b', False)
    # Getting the type of 'x' (line 438)
    x_410190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 21), 'x', False)
    # Getting the type of 'restrt' (line 438)
    restrt_410191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 24), 'restrt', False)
    # Getting the type of 'work' (line 438)
    work_410192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 32), 'work', False)
    # Getting the type of 'work2' (line 438)
    work2_410193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 38), 'work2', False)
    # Getting the type of 'iter_' (line 438)
    iter__410194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 45), 'iter_', False)
    # Getting the type of 'resid' (line 438)
    resid_410195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 52), 'resid', False)
    # Getting the type of 'info' (line 438)
    info_410196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 59), 'info', False)
    # Getting the type of 'ndx1' (line 438)
    ndx1_410197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 65), 'ndx1', False)
    # Getting the type of 'ndx2' (line 438)
    ndx2_410198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 71), 'ndx2', False)
    # Getting the type of 'ijob' (line 438)
    ijob_410199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 77), 'ijob', False)
    # Processing the call keyword arguments (line 438)
    kwargs_410200 = {}
    # Getting the type of 'revcom' (line 438)
    revcom_410188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 438)
    revcom_call_result_410201 = invoke(stypy.reporting.localization.Localization(__file__, 438, 11), revcom_410188, *[b_410189, x_410190, restrt_410191, work_410192, work2_410193, iter__410194, resid_410195, info_410196, ndx1_410197, ndx2_410198, ijob_410199], **kwargs_410200)
    
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___410202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), revcom_call_result_410201, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_410203 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___410202, int_410187)
    
    # Assigning a type to the variable 'tuple_var_assignment_407972' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407972', subscript_call_result_410203)
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'tuple_var_assignment_407964' (line 437)
    tuple_var_assignment_407964_410204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407964')
    # Assigning a type to the variable 'x' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'x', tuple_var_assignment_407964_410204)
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'tuple_var_assignment_407965' (line 437)
    tuple_var_assignment_407965_410205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407965')
    # Assigning a type to the variable 'iter_' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 11), 'iter_', tuple_var_assignment_407965_410205)
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'tuple_var_assignment_407966' (line 437)
    tuple_var_assignment_407966_410206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407966')
    # Assigning a type to the variable 'resid' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 18), 'resid', tuple_var_assignment_407966_410206)
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'tuple_var_assignment_407967' (line 437)
    tuple_var_assignment_407967_410207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407967')
    # Assigning a type to the variable 'info' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 25), 'info', tuple_var_assignment_407967_410207)
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'tuple_var_assignment_407968' (line 437)
    tuple_var_assignment_407968_410208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407968')
    # Assigning a type to the variable 'ndx1' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 31), 'ndx1', tuple_var_assignment_407968_410208)
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'tuple_var_assignment_407969' (line 437)
    tuple_var_assignment_407969_410209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407969')
    # Assigning a type to the variable 'ndx2' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 37), 'ndx2', tuple_var_assignment_407969_410209)
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'tuple_var_assignment_407970' (line 437)
    tuple_var_assignment_407970_410210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407970')
    # Assigning a type to the variable 'sclr1' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 43), 'sclr1', tuple_var_assignment_407970_410210)
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'tuple_var_assignment_407971' (line 437)
    tuple_var_assignment_407971_410211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407971')
    # Assigning a type to the variable 'sclr2' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 50), 'sclr2', tuple_var_assignment_407971_410211)
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'tuple_var_assignment_407972' (line 437)
    tuple_var_assignment_407972_410212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_407972')
    # Assigning a type to the variable 'ijob' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 57), 'ijob', tuple_var_assignment_407972_410212)
    
    # Assigning a Call to a Name (line 441):
    
    # Assigning a Call to a Name (line 441):
    
    # Call to slice(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'ndx1' (line 441)
    ndx1_410214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 23), 'ndx1', False)
    int_410215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 28), 'int')
    # Applying the binary operator '-' (line 441)
    result_sub_410216 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 23), '-', ndx1_410214, int_410215)
    
    # Getting the type of 'ndx1' (line 441)
    ndx1_410217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 31), 'ndx1', False)
    int_410218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 36), 'int')
    # Applying the binary operator '-' (line 441)
    result_sub_410219 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 31), '-', ndx1_410217, int_410218)
    
    # Getting the type of 'n' (line 441)
    n_410220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 38), 'n', False)
    # Applying the binary operator '+' (line 441)
    result_add_410221 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 37), '+', result_sub_410219, n_410220)
    
    # Processing the call keyword arguments (line 441)
    kwargs_410222 = {}
    # Getting the type of 'slice' (line 441)
    slice_410213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 441)
    slice_call_result_410223 = invoke(stypy.reporting.localization.Localization(__file__, 441, 17), slice_410213, *[result_sub_410216, result_add_410221], **kwargs_410222)
    
    # Assigning a type to the variable 'slice1' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'slice1', slice_call_result_410223)
    
    # Assigning a Call to a Name (line 442):
    
    # Assigning a Call to a Name (line 442):
    
    # Call to slice(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'ndx2' (line 442)
    ndx2_410225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 23), 'ndx2', False)
    int_410226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 28), 'int')
    # Applying the binary operator '-' (line 442)
    result_sub_410227 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 23), '-', ndx2_410225, int_410226)
    
    # Getting the type of 'ndx2' (line 442)
    ndx2_410228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 31), 'ndx2', False)
    int_410229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 36), 'int')
    # Applying the binary operator '-' (line 442)
    result_sub_410230 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 31), '-', ndx2_410228, int_410229)
    
    # Getting the type of 'n' (line 442)
    n_410231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 38), 'n', False)
    # Applying the binary operator '+' (line 442)
    result_add_410232 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 37), '+', result_sub_410230, n_410231)
    
    # Processing the call keyword arguments (line 442)
    kwargs_410233 = {}
    # Getting the type of 'slice' (line 442)
    slice_410224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 442)
    slice_call_result_410234 = invoke(stypy.reporting.localization.Localization(__file__, 442, 17), slice_410224, *[result_sub_410227, result_add_410232], **kwargs_410233)
    
    # Assigning a type to the variable 'slice2' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'slice2', slice_call_result_410234)
    
    
    # Getting the type of 'ijob' (line 443)
    ijob_410235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 12), 'ijob')
    int_410236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 20), 'int')
    # Applying the binary operator '==' (line 443)
    result_eq_410237 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 12), '==', ijob_410235, int_410236)
    
    # Testing the type of an if condition (line 443)
    if_condition_410238 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 8), result_eq_410237)
    # Assigning a type to the variable 'if_condition_410238' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'if_condition_410238', if_condition_410238)
    # SSA begins for if statement (line 443)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'resid_ready' (line 444)
    resid_ready_410239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 15), 'resid_ready')
    
    # Getting the type of 'callback' (line 444)
    callback_410240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 31), 'callback')
    # Getting the type of 'None' (line 444)
    None_410241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 47), 'None')
    # Applying the binary operator 'isnot' (line 444)
    result_is_not_410242 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 31), 'isnot', callback_410240, None_410241)
    
    # Applying the binary operator 'and' (line 444)
    result_and_keyword_410243 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 15), 'and', resid_ready_410239, result_is_not_410242)
    
    # Testing the type of an if condition (line 444)
    if_condition_410244 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 444, 12), result_and_keyword_410243)
    # Assigning a type to the variable 'if_condition_410244' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'if_condition_410244', if_condition_410244)
    # SSA begins for if statement (line 444)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to callback(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'resid' (line 445)
    resid_410246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 25), 'resid', False)
    # Processing the call keyword arguments (line 445)
    kwargs_410247 = {}
    # Getting the type of 'callback' (line 445)
    callback_410245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 16), 'callback', False)
    # Calling callback(args, kwargs) (line 445)
    callback_call_result_410248 = invoke(stypy.reporting.localization.Localization(__file__, 445, 16), callback_410245, *[resid_410246], **kwargs_410247)
    
    
    # Assigning a Name to a Name (line 446):
    
    # Assigning a Name to a Name (line 446):
    # Getting the type of 'False' (line 446)
    False_410249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 30), 'False')
    # Assigning a type to the variable 'resid_ready' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 16), 'resid_ready', False_410249)
    # SSA join for if statement (line 444)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 443)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 449)
    ijob_410250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 14), 'ijob')
    int_410251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 22), 'int')
    # Applying the binary operator '==' (line 449)
    result_eq_410252 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 14), '==', ijob_410250, int_410251)
    
    # Testing the type of an if condition (line 449)
    if_condition_410253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 449, 13), result_eq_410252)
    # Assigning a type to the variable 'if_condition_410253' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 13), 'if_condition_410253', if_condition_410253)
    # SSA begins for if statement (line 449)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 450)
    work_410254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 450)
    slice2_410255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 17), 'slice2')
    # Getting the type of 'work' (line 450)
    work_410256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 450)
    getitem___410257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 12), work_410256, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 450)
    subscript_call_result_410258 = invoke(stypy.reporting.localization.Localization(__file__, 450, 12), getitem___410257, slice2_410255)
    
    # Getting the type of 'sclr2' (line 450)
    sclr2_410259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 28), 'sclr2')
    # Applying the binary operator '*=' (line 450)
    result_imul_410260 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 12), '*=', subscript_call_result_410258, sclr2_410259)
    # Getting the type of 'work' (line 450)
    work_410261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'work')
    # Getting the type of 'slice2' (line 450)
    slice2_410262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 17), 'slice2')
    # Storing an element on a container (line 450)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 12), work_410261, (slice2_410262, result_imul_410260))
    
    
    # Getting the type of 'work' (line 451)
    work_410263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 451)
    slice2_410264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 17), 'slice2')
    # Getting the type of 'work' (line 451)
    work_410265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 451)
    getitem___410266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 12), work_410265, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 451)
    subscript_call_result_410267 = invoke(stypy.reporting.localization.Localization(__file__, 451, 12), getitem___410266, slice2_410264)
    
    # Getting the type of 'sclr1' (line 451)
    sclr1_410268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 28), 'sclr1')
    
    # Call to matvec(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'x' (line 451)
    x_410270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 41), 'x', False)
    # Processing the call keyword arguments (line 451)
    kwargs_410271 = {}
    # Getting the type of 'matvec' (line 451)
    matvec_410269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 34), 'matvec', False)
    # Calling matvec(args, kwargs) (line 451)
    matvec_call_result_410272 = invoke(stypy.reporting.localization.Localization(__file__, 451, 34), matvec_410269, *[x_410270], **kwargs_410271)
    
    # Applying the binary operator '*' (line 451)
    result_mul_410273 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 28), '*', sclr1_410268, matvec_call_result_410272)
    
    # Applying the binary operator '+=' (line 451)
    result_iadd_410274 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 12), '+=', subscript_call_result_410267, result_mul_410273)
    # Getting the type of 'work' (line 451)
    work_410275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'work')
    # Getting the type of 'slice2' (line 451)
    slice2_410276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 17), 'slice2')
    # Storing an element on a container (line 451)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 12), work_410275, (slice2_410276, result_iadd_410274))
    
    # SSA branch for the else part of an if statement (line 449)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 452)
    ijob_410277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 14), 'ijob')
    int_410278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 22), 'int')
    # Applying the binary operator '==' (line 452)
    result_eq_410279 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 14), '==', ijob_410277, int_410278)
    
    # Testing the type of an if condition (line 452)
    if_condition_410280 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 452, 13), result_eq_410279)
    # Assigning a type to the variable 'if_condition_410280' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 13), 'if_condition_410280', if_condition_410280)
    # SSA begins for if statement (line 452)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 453):
    
    # Assigning a Call to a Subscript (line 453):
    
    # Call to psolve(...): (line 453)
    # Processing the call arguments (line 453)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 453)
    slice2_410282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 39), 'slice2', False)
    # Getting the type of 'work' (line 453)
    work_410283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 34), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 453)
    getitem___410284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 34), work_410283, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 453)
    subscript_call_result_410285 = invoke(stypy.reporting.localization.Localization(__file__, 453, 34), getitem___410284, slice2_410282)
    
    # Processing the call keyword arguments (line 453)
    kwargs_410286 = {}
    # Getting the type of 'psolve' (line 453)
    psolve_410281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 27), 'psolve', False)
    # Calling psolve(args, kwargs) (line 453)
    psolve_call_result_410287 = invoke(stypy.reporting.localization.Localization(__file__, 453, 27), psolve_410281, *[subscript_call_result_410285], **kwargs_410286)
    
    # Getting the type of 'work' (line 453)
    work_410288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'work')
    # Getting the type of 'slice1' (line 453)
    slice1_410289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 17), 'slice1')
    # Storing an element on a container (line 453)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 12), work_410288, (slice1_410289, psolve_call_result_410287))
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'first_pass' (line 454)
    first_pass_410290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 19), 'first_pass')
    # Applying the 'not' unary operator (line 454)
    result_not__410291 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 15), 'not', first_pass_410290)
    
    
    # Getting the type of 'old_ijob' (line 454)
    old_ijob_410292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 34), 'old_ijob')
    int_410293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 46), 'int')
    # Applying the binary operator '==' (line 454)
    result_eq_410294 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 34), '==', old_ijob_410292, int_410293)
    
    # Applying the binary operator 'and' (line 454)
    result_and_keyword_410295 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 15), 'and', result_not__410291, result_eq_410294)
    
    # Testing the type of an if condition (line 454)
    if_condition_410296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 12), result_and_keyword_410295)
    # Assigning a type to the variable 'if_condition_410296' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'if_condition_410296', if_condition_410296)
    # SSA begins for if statement (line 454)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 455):
    
    # Assigning a Name to a Name (line 455):
    # Getting the type of 'True' (line 455)
    True_410297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 30), 'True')
    # Assigning a type to the variable 'resid_ready' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'resid_ready', True_410297)
    # SSA join for if statement (line 454)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 457):
    
    # Assigning a Name to a Name (line 457):
    # Getting the type of 'False' (line 457)
    False_410298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 25), 'False')
    # Assigning a type to the variable 'first_pass' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'first_pass', False_410298)
    # SSA branch for the else part of an if statement (line 452)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 458)
    ijob_410299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 14), 'ijob')
    int_410300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 22), 'int')
    # Applying the binary operator '==' (line 458)
    result_eq_410301 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 14), '==', ijob_410299, int_410300)
    
    # Testing the type of an if condition (line 458)
    if_condition_410302 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 458, 13), result_eq_410301)
    # Assigning a type to the variable 'if_condition_410302' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 13), 'if_condition_410302', if_condition_410302)
    # SSA begins for if statement (line 458)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 459)
    work_410303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 459)
    slice2_410304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 17), 'slice2')
    # Getting the type of 'work' (line 459)
    work_410305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 459)
    getitem___410306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 12), work_410305, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 459)
    subscript_call_result_410307 = invoke(stypy.reporting.localization.Localization(__file__, 459, 12), getitem___410306, slice2_410304)
    
    # Getting the type of 'sclr2' (line 459)
    sclr2_410308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 28), 'sclr2')
    # Applying the binary operator '*=' (line 459)
    result_imul_410309 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 12), '*=', subscript_call_result_410307, sclr2_410308)
    # Getting the type of 'work' (line 459)
    work_410310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'work')
    # Getting the type of 'slice2' (line 459)
    slice2_410311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 17), 'slice2')
    # Storing an element on a container (line 459)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 12), work_410310, (slice2_410311, result_imul_410309))
    
    
    # Getting the type of 'work' (line 460)
    work_410312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 460)
    slice2_410313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 17), 'slice2')
    # Getting the type of 'work' (line 460)
    work_410314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 460)
    getitem___410315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 12), work_410314, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 460)
    subscript_call_result_410316 = invoke(stypy.reporting.localization.Localization(__file__, 460, 12), getitem___410315, slice2_410313)
    
    # Getting the type of 'sclr1' (line 460)
    sclr1_410317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 28), 'sclr1')
    
    # Call to matvec(...): (line 460)
    # Processing the call arguments (line 460)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 460)
    slice1_410319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 46), 'slice1', False)
    # Getting the type of 'work' (line 460)
    work_410320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 41), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 460)
    getitem___410321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 41), work_410320, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 460)
    subscript_call_result_410322 = invoke(stypy.reporting.localization.Localization(__file__, 460, 41), getitem___410321, slice1_410319)
    
    # Processing the call keyword arguments (line 460)
    kwargs_410323 = {}
    # Getting the type of 'matvec' (line 460)
    matvec_410318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'matvec', False)
    # Calling matvec(args, kwargs) (line 460)
    matvec_call_result_410324 = invoke(stypy.reporting.localization.Localization(__file__, 460, 34), matvec_410318, *[subscript_call_result_410322], **kwargs_410323)
    
    # Applying the binary operator '*' (line 460)
    result_mul_410325 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 28), '*', sclr1_410317, matvec_call_result_410324)
    
    # Applying the binary operator '+=' (line 460)
    result_iadd_410326 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 12), '+=', subscript_call_result_410316, result_mul_410325)
    # Getting the type of 'work' (line 460)
    work_410327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'work')
    # Getting the type of 'slice2' (line 460)
    slice2_410328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 17), 'slice2')
    # Storing an element on a container (line 460)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 12), work_410327, (slice2_410328, result_iadd_410326))
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'resid_ready' (line 461)
    resid_ready_410329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 15), 'resid_ready')
    
    # Getting the type of 'callback' (line 461)
    callback_410330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 31), 'callback')
    # Getting the type of 'None' (line 461)
    None_410331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 47), 'None')
    # Applying the binary operator 'isnot' (line 461)
    result_is_not_410332 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 31), 'isnot', callback_410330, None_410331)
    
    # Applying the binary operator 'and' (line 461)
    result_and_keyword_410333 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 15), 'and', resid_ready_410329, result_is_not_410332)
    
    # Testing the type of an if condition (line 461)
    if_condition_410334 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 461, 12), result_and_keyword_410333)
    # Assigning a type to the variable 'if_condition_410334' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'if_condition_410334', if_condition_410334)
    # SSA begins for if statement (line 461)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to callback(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'resid' (line 462)
    resid_410336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 25), 'resid', False)
    # Processing the call keyword arguments (line 462)
    kwargs_410337 = {}
    # Getting the type of 'callback' (line 462)
    callback_410335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 16), 'callback', False)
    # Calling callback(args, kwargs) (line 462)
    callback_call_result_410338 = invoke(stypy.reporting.localization.Localization(__file__, 462, 16), callback_410335, *[resid_410336], **kwargs_410337)
    
    
    # Assigning a Name to a Name (line 463):
    
    # Assigning a Name to a Name (line 463):
    # Getting the type of 'False' (line 463)
    False_410339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 30), 'False')
    # Assigning a type to the variable 'resid_ready' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 16), 'resid_ready', False_410339)
    
    # Assigning a BinOp to a Name (line 464):
    
    # Assigning a BinOp to a Name (line 464):
    # Getting the type of 'iter_num' (line 464)
    iter_num_410340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 27), 'iter_num')
    int_410341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 36), 'int')
    # Applying the binary operator '+' (line 464)
    result_add_410342 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 27), '+', iter_num_410340, int_410341)
    
    # Assigning a type to the variable 'iter_num' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 16), 'iter_num', result_add_410342)
    # SSA join for if statement (line 461)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 458)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 466)
    ijob_410343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 14), 'ijob')
    int_410344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 22), 'int')
    # Applying the binary operator '==' (line 466)
    result_eq_410345 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 14), '==', ijob_410343, int_410344)
    
    # Testing the type of an if condition (line 466)
    if_condition_410346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 466, 13), result_eq_410345)
    # Assigning a type to the variable 'if_condition_410346' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 13), 'if_condition_410346', if_condition_410346)
    # SSA begins for if statement (line 466)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ftflag' (line 467)
    ftflag_410347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 15), 'ftflag')
    # Testing the type of an if condition (line 467)
    if_condition_410348 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 467, 12), ftflag_410347)
    # Assigning a type to the variable 'if_condition_410348' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'if_condition_410348', if_condition_410348)
    # SSA begins for if statement (line 467)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 468):
    
    # Assigning a Num to a Name (line 468):
    int_410349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 23), 'int')
    # Assigning a type to the variable 'info' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 16), 'info', int_410349)
    
    # Assigning a Name to a Name (line 469):
    
    # Assigning a Name to a Name (line 469):
    # Getting the type of 'False' (line 469)
    False_410350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 25), 'False')
    # Assigning a type to the variable 'ftflag' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 16), 'ftflag', False_410350)
    # SSA join for if statement (line 467)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 470):
    
    # Assigning a Subscript to a Name (line 470):
    
    # Obtaining the type of the subscript
    int_410351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 12), 'int')
    
    # Call to stoptest(...): (line 470)
    # Processing the call arguments (line 470)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 470)
    slice1_410353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 47), 'slice1', False)
    # Getting the type of 'work' (line 470)
    work_410354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___410355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 42), work_410354, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_410356 = invoke(stypy.reporting.localization.Localization(__file__, 470, 42), getitem___410355, slice1_410353)
    
    # Getting the type of 'b' (line 470)
    b_410357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 470)
    bnrm2_410358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 470)
    tol_410359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 66), 'tol', False)
    # Getting the type of 'info' (line 470)
    info_410360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 71), 'info', False)
    # Processing the call keyword arguments (line 470)
    kwargs_410361 = {}
    # Getting the type of 'stoptest' (line 470)
    stoptest_410352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 470)
    stoptest_call_result_410362 = invoke(stypy.reporting.localization.Localization(__file__, 470, 33), stoptest_410352, *[subscript_call_result_410356, b_410357, bnrm2_410358, tol_410359, info_410360], **kwargs_410361)
    
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___410363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 12), stoptest_call_result_410362, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_410364 = invoke(stypy.reporting.localization.Localization(__file__, 470, 12), getitem___410363, int_410351)
    
    # Assigning a type to the variable 'tuple_var_assignment_407973' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'tuple_var_assignment_407973', subscript_call_result_410364)
    
    # Assigning a Subscript to a Name (line 470):
    
    # Obtaining the type of the subscript
    int_410365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 12), 'int')
    
    # Call to stoptest(...): (line 470)
    # Processing the call arguments (line 470)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 470)
    slice1_410367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 47), 'slice1', False)
    # Getting the type of 'work' (line 470)
    work_410368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___410369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 42), work_410368, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_410370 = invoke(stypy.reporting.localization.Localization(__file__, 470, 42), getitem___410369, slice1_410367)
    
    # Getting the type of 'b' (line 470)
    b_410371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 470)
    bnrm2_410372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 470)
    tol_410373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 66), 'tol', False)
    # Getting the type of 'info' (line 470)
    info_410374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 71), 'info', False)
    # Processing the call keyword arguments (line 470)
    kwargs_410375 = {}
    # Getting the type of 'stoptest' (line 470)
    stoptest_410366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 470)
    stoptest_call_result_410376 = invoke(stypy.reporting.localization.Localization(__file__, 470, 33), stoptest_410366, *[subscript_call_result_410370, b_410371, bnrm2_410372, tol_410373, info_410374], **kwargs_410375)
    
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___410377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 12), stoptest_call_result_410376, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_410378 = invoke(stypy.reporting.localization.Localization(__file__, 470, 12), getitem___410377, int_410365)
    
    # Assigning a type to the variable 'tuple_var_assignment_407974' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'tuple_var_assignment_407974', subscript_call_result_410378)
    
    # Assigning a Subscript to a Name (line 470):
    
    # Obtaining the type of the subscript
    int_410379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 12), 'int')
    
    # Call to stoptest(...): (line 470)
    # Processing the call arguments (line 470)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 470)
    slice1_410381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 47), 'slice1', False)
    # Getting the type of 'work' (line 470)
    work_410382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___410383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 42), work_410382, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_410384 = invoke(stypy.reporting.localization.Localization(__file__, 470, 42), getitem___410383, slice1_410381)
    
    # Getting the type of 'b' (line 470)
    b_410385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 470)
    bnrm2_410386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 470)
    tol_410387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 66), 'tol', False)
    # Getting the type of 'info' (line 470)
    info_410388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 71), 'info', False)
    # Processing the call keyword arguments (line 470)
    kwargs_410389 = {}
    # Getting the type of 'stoptest' (line 470)
    stoptest_410380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 470)
    stoptest_call_result_410390 = invoke(stypy.reporting.localization.Localization(__file__, 470, 33), stoptest_410380, *[subscript_call_result_410384, b_410385, bnrm2_410386, tol_410387, info_410388], **kwargs_410389)
    
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___410391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 12), stoptest_call_result_410390, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_410392 = invoke(stypy.reporting.localization.Localization(__file__, 470, 12), getitem___410391, int_410379)
    
    # Assigning a type to the variable 'tuple_var_assignment_407975' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'tuple_var_assignment_407975', subscript_call_result_410392)
    
    # Assigning a Name to a Name (line 470):
    # Getting the type of 'tuple_var_assignment_407973' (line 470)
    tuple_var_assignment_407973_410393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'tuple_var_assignment_407973')
    # Assigning a type to the variable 'bnrm2' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'bnrm2', tuple_var_assignment_407973_410393)
    
    # Assigning a Name to a Name (line 470):
    # Getting the type of 'tuple_var_assignment_407974' (line 470)
    tuple_var_assignment_407974_410394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'tuple_var_assignment_407974')
    # Assigning a type to the variable 'resid' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 19), 'resid', tuple_var_assignment_407974_410394)
    
    # Assigning a Name to a Name (line 470):
    # Getting the type of 'tuple_var_assignment_407975' (line 470)
    tuple_var_assignment_407975_410395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'tuple_var_assignment_407975')
    # Assigning a type to the variable 'info' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 26), 'info', tuple_var_assignment_407975_410395)
    # SSA join for if statement (line 466)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 458)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 452)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 449)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 443)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 472):
    
    # Assigning a Name to a Name (line 472):
    # Getting the type of 'ijob' (line 472)
    ijob_410396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 19), 'ijob')
    # Assigning a type to the variable 'old_ijob' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'old_ijob', ijob_410396)
    
    # Assigning a Num to a Name (line 473):
    
    # Assigning a Num to a Name (line 473):
    int_410397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 15), 'int')
    # Assigning a type to the variable 'ijob' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'ijob', int_410397)
    
    
    # Getting the type of 'iter_num' (line 475)
    iter_num_410398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 11), 'iter_num')
    # Getting the type of 'maxiter' (line 475)
    maxiter_410399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 22), 'maxiter')
    # Applying the binary operator '>' (line 475)
    result_gt_410400 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 11), '>', iter_num_410398, maxiter_410399)
    
    # Testing the type of an if condition (line 475)
    if_condition_410401 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 475, 8), result_gt_410400)
    # Assigning a type to the variable 'if_condition_410401' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'if_condition_410401', if_condition_410401)
    # SSA begins for if statement (line 475)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 475)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 435)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'info' (line 478)
    info_410402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 7), 'info')
    int_410403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 15), 'int')
    # Applying the binary operator '>=' (line 478)
    result_ge_410404 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 7), '>=', info_410402, int_410403)
    
    
    # Getting the type of 'resid' (line 478)
    resid_410405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 21), 'resid')
    # Getting the type of 'tol' (line 478)
    tol_410406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 29), 'tol')
    # Applying the binary operator '>' (line 478)
    result_gt_410407 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 21), '>', resid_410405, tol_410406)
    
    # Applying the binary operator 'and' (line 478)
    result_and_keyword_410408 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 7), 'and', result_ge_410404, result_gt_410407)
    
    # Testing the type of an if condition (line 478)
    if_condition_410409 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 478, 4), result_and_keyword_410408)
    # Assigning a type to the variable 'if_condition_410409' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'if_condition_410409', if_condition_410409)
    # SSA begins for if statement (line 478)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 480):
    
    # Assigning a Name to a Name (line 480):
    # Getting the type of 'maxiter' (line 480)
    maxiter_410410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 15), 'maxiter')
    # Assigning a type to the variable 'info' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'info', maxiter_410410)
    # SSA join for if statement (line 478)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 482)
    tuple_410411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 482)
    # Adding element type (line 482)
    
    # Call to postprocess(...): (line 482)
    # Processing the call arguments (line 482)
    # Getting the type of 'x' (line 482)
    x_410413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 23), 'x', False)
    # Processing the call keyword arguments (line 482)
    kwargs_410414 = {}
    # Getting the type of 'postprocess' (line 482)
    postprocess_410412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 11), 'postprocess', False)
    # Calling postprocess(args, kwargs) (line 482)
    postprocess_call_result_410415 = invoke(stypy.reporting.localization.Localization(__file__, 482, 11), postprocess_410412, *[x_410413], **kwargs_410414)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 11), tuple_410411, postprocess_call_result_410415)
    # Adding element type (line 482)
    # Getting the type of 'info' (line 482)
    info_410416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 27), 'info')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 11), tuple_410411, info_410416)
    
    # Assigning a type to the variable 'stypy_return_type' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'stypy_return_type', tuple_410411)
    
    # ################# End of 'gmres(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gmres' in the type store
    # Getting the type of 'stypy_return_type' (line 319)
    stypy_return_type_410417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_410417)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gmres'
    return stypy_return_type_410417

# Assigning a type to the variable 'gmres' (line 319)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 0), 'gmres', gmres)

@norecursion
def qmr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 486)
    None_410418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 17), 'None')
    float_410419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 27), 'float')
    # Getting the type of 'None' (line 486)
    None_410420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 41), 'None')
    # Getting the type of 'None' (line 486)
    None_410421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 50), 'None')
    # Getting the type of 'None' (line 486)
    None_410422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 59), 'None')
    # Getting the type of 'None' (line 486)
    None_410423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 74), 'None')
    defaults = [None_410418, float_410419, None_410420, None_410421, None_410422, None_410423]
    # Create a new context for function 'qmr'
    module_type_store = module_type_store.open_function_context('qmr', 485, 0, False)
    
    # Passed parameters checking function
    qmr.stypy_localization = localization
    qmr.stypy_type_of_self = None
    qmr.stypy_type_store = module_type_store
    qmr.stypy_function_name = 'qmr'
    qmr.stypy_param_names_list = ['A', 'b', 'x0', 'tol', 'maxiter', 'M1', 'M2', 'callback']
    qmr.stypy_varargs_param_name = None
    qmr.stypy_kwargs_param_name = None
    qmr.stypy_call_defaults = defaults
    qmr.stypy_call_varargs = varargs
    qmr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'qmr', ['A', 'b', 'x0', 'tol', 'maxiter', 'M1', 'M2', 'callback'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'qmr', localization, ['A', 'b', 'x0', 'tol', 'maxiter', 'M1', 'M2', 'callback'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'qmr(...)' code ##################

    str_410424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, (-1)), 'str', 'Use Quasi-Minimal Residual iteration to solve ``Ax = b``.\n\n    Parameters\n    ----------\n    A : {sparse matrix, dense matrix, LinearOperator}\n        The real-valued N-by-N matrix of the linear system.\n        It is required that the linear operator can produce\n        ``Ax`` and ``A^T x``.\n    b : {array, matrix}\n        Right hand side of the linear system. Has shape (N,) or (N,1).\n\n    Returns\n    -------\n    x : {array, matrix}\n        The converged solution.\n    info : integer\n        Provides convergence information:\n            0  : successful exit\n            >0 : convergence to tolerance not achieved, number of iterations\n            <0 : illegal input or breakdown\n\n    Other Parameters\n    ----------------\n    x0  : {array, matrix}\n        Starting guess for the solution.\n    tol : float\n        Tolerance to achieve. The algorithm terminates when either the relative\n        or the absolute residual is below `tol`.\n    maxiter : integer\n        Maximum number of iterations.  Iteration will stop after maxiter\n        steps even if the specified tolerance has not been achieved.\n    M1 : {sparse matrix, dense matrix, LinearOperator}\n        Left preconditioner for A.\n    M2 : {sparse matrix, dense matrix, LinearOperator}\n        Right preconditioner for A. Used together with the left\n        preconditioner M1.  The matrix M1*A*M2 should have better\n        conditioned than A alone.\n    callback : function\n        User-supplied function to call after each iteration.  It is called\n        as callback(xk), where xk is the current solution vector.\n\n    See Also\n    --------\n    LinearOperator\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import qmr\n    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)\n    >>> b = np.array([2, 4, -1], dtype=float)\n    >>> x, exitCode = qmr(A, b)\n    >>> print(exitCode)            # 0 indicates successful convergence\n    0\n    >>> np.allclose(A.dot(x), b)\n    True\n    ')
    
    # Assigning a Name to a Name (line 544):
    
    # Assigning a Name to a Name (line 544):
    # Getting the type of 'A' (line 544)
    A_410425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 9), 'A')
    # Assigning a type to the variable 'A_' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'A_', A_410425)
    
    # Assigning a Call to a Tuple (line 545):
    
    # Assigning a Subscript to a Name (line 545):
    
    # Obtaining the type of the subscript
    int_410426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 4), 'int')
    
    # Call to make_system(...): (line 545)
    # Processing the call arguments (line 545)
    # Getting the type of 'A' (line 545)
    A_410428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 42), 'A', False)
    # Getting the type of 'None' (line 545)
    None_410429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 45), 'None', False)
    # Getting the type of 'x0' (line 545)
    x0_410430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 51), 'x0', False)
    # Getting the type of 'b' (line 545)
    b_410431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 55), 'b', False)
    # Processing the call keyword arguments (line 545)
    kwargs_410432 = {}
    # Getting the type of 'make_system' (line 545)
    make_system_410427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 545)
    make_system_call_result_410433 = invoke(stypy.reporting.localization.Localization(__file__, 545, 30), make_system_410427, *[A_410428, None_410429, x0_410430, b_410431], **kwargs_410432)
    
    # Obtaining the member '__getitem__' of a type (line 545)
    getitem___410434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 4), make_system_call_result_410433, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 545)
    subscript_call_result_410435 = invoke(stypy.reporting.localization.Localization(__file__, 545, 4), getitem___410434, int_410426)
    
    # Assigning a type to the variable 'tuple_var_assignment_407976' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'tuple_var_assignment_407976', subscript_call_result_410435)
    
    # Assigning a Subscript to a Name (line 545):
    
    # Obtaining the type of the subscript
    int_410436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 4), 'int')
    
    # Call to make_system(...): (line 545)
    # Processing the call arguments (line 545)
    # Getting the type of 'A' (line 545)
    A_410438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 42), 'A', False)
    # Getting the type of 'None' (line 545)
    None_410439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 45), 'None', False)
    # Getting the type of 'x0' (line 545)
    x0_410440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 51), 'x0', False)
    # Getting the type of 'b' (line 545)
    b_410441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 55), 'b', False)
    # Processing the call keyword arguments (line 545)
    kwargs_410442 = {}
    # Getting the type of 'make_system' (line 545)
    make_system_410437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 545)
    make_system_call_result_410443 = invoke(stypy.reporting.localization.Localization(__file__, 545, 30), make_system_410437, *[A_410438, None_410439, x0_410440, b_410441], **kwargs_410442)
    
    # Obtaining the member '__getitem__' of a type (line 545)
    getitem___410444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 4), make_system_call_result_410443, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 545)
    subscript_call_result_410445 = invoke(stypy.reporting.localization.Localization(__file__, 545, 4), getitem___410444, int_410436)
    
    # Assigning a type to the variable 'tuple_var_assignment_407977' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'tuple_var_assignment_407977', subscript_call_result_410445)
    
    # Assigning a Subscript to a Name (line 545):
    
    # Obtaining the type of the subscript
    int_410446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 4), 'int')
    
    # Call to make_system(...): (line 545)
    # Processing the call arguments (line 545)
    # Getting the type of 'A' (line 545)
    A_410448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 42), 'A', False)
    # Getting the type of 'None' (line 545)
    None_410449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 45), 'None', False)
    # Getting the type of 'x0' (line 545)
    x0_410450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 51), 'x0', False)
    # Getting the type of 'b' (line 545)
    b_410451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 55), 'b', False)
    # Processing the call keyword arguments (line 545)
    kwargs_410452 = {}
    # Getting the type of 'make_system' (line 545)
    make_system_410447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 545)
    make_system_call_result_410453 = invoke(stypy.reporting.localization.Localization(__file__, 545, 30), make_system_410447, *[A_410448, None_410449, x0_410450, b_410451], **kwargs_410452)
    
    # Obtaining the member '__getitem__' of a type (line 545)
    getitem___410454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 4), make_system_call_result_410453, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 545)
    subscript_call_result_410455 = invoke(stypy.reporting.localization.Localization(__file__, 545, 4), getitem___410454, int_410446)
    
    # Assigning a type to the variable 'tuple_var_assignment_407978' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'tuple_var_assignment_407978', subscript_call_result_410455)
    
    # Assigning a Subscript to a Name (line 545):
    
    # Obtaining the type of the subscript
    int_410456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 4), 'int')
    
    # Call to make_system(...): (line 545)
    # Processing the call arguments (line 545)
    # Getting the type of 'A' (line 545)
    A_410458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 42), 'A', False)
    # Getting the type of 'None' (line 545)
    None_410459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 45), 'None', False)
    # Getting the type of 'x0' (line 545)
    x0_410460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 51), 'x0', False)
    # Getting the type of 'b' (line 545)
    b_410461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 55), 'b', False)
    # Processing the call keyword arguments (line 545)
    kwargs_410462 = {}
    # Getting the type of 'make_system' (line 545)
    make_system_410457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 545)
    make_system_call_result_410463 = invoke(stypy.reporting.localization.Localization(__file__, 545, 30), make_system_410457, *[A_410458, None_410459, x0_410460, b_410461], **kwargs_410462)
    
    # Obtaining the member '__getitem__' of a type (line 545)
    getitem___410464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 4), make_system_call_result_410463, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 545)
    subscript_call_result_410465 = invoke(stypy.reporting.localization.Localization(__file__, 545, 4), getitem___410464, int_410456)
    
    # Assigning a type to the variable 'tuple_var_assignment_407979' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'tuple_var_assignment_407979', subscript_call_result_410465)
    
    # Assigning a Subscript to a Name (line 545):
    
    # Obtaining the type of the subscript
    int_410466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 4), 'int')
    
    # Call to make_system(...): (line 545)
    # Processing the call arguments (line 545)
    # Getting the type of 'A' (line 545)
    A_410468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 42), 'A', False)
    # Getting the type of 'None' (line 545)
    None_410469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 45), 'None', False)
    # Getting the type of 'x0' (line 545)
    x0_410470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 51), 'x0', False)
    # Getting the type of 'b' (line 545)
    b_410471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 55), 'b', False)
    # Processing the call keyword arguments (line 545)
    kwargs_410472 = {}
    # Getting the type of 'make_system' (line 545)
    make_system_410467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 545)
    make_system_call_result_410473 = invoke(stypy.reporting.localization.Localization(__file__, 545, 30), make_system_410467, *[A_410468, None_410469, x0_410470, b_410471], **kwargs_410472)
    
    # Obtaining the member '__getitem__' of a type (line 545)
    getitem___410474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 4), make_system_call_result_410473, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 545)
    subscript_call_result_410475 = invoke(stypy.reporting.localization.Localization(__file__, 545, 4), getitem___410474, int_410466)
    
    # Assigning a type to the variable 'tuple_var_assignment_407980' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'tuple_var_assignment_407980', subscript_call_result_410475)
    
    # Assigning a Name to a Name (line 545):
    # Getting the type of 'tuple_var_assignment_407976' (line 545)
    tuple_var_assignment_407976_410476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'tuple_var_assignment_407976')
    # Assigning a type to the variable 'A' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'A', tuple_var_assignment_407976_410476)
    
    # Assigning a Name to a Name (line 545):
    # Getting the type of 'tuple_var_assignment_407977' (line 545)
    tuple_var_assignment_407977_410477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'tuple_var_assignment_407977')
    # Assigning a type to the variable 'M' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 7), 'M', tuple_var_assignment_407977_410477)
    
    # Assigning a Name to a Name (line 545):
    # Getting the type of 'tuple_var_assignment_407978' (line 545)
    tuple_var_assignment_407978_410478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'tuple_var_assignment_407978')
    # Assigning a type to the variable 'x' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 10), 'x', tuple_var_assignment_407978_410478)
    
    # Assigning a Name to a Name (line 545):
    # Getting the type of 'tuple_var_assignment_407979' (line 545)
    tuple_var_assignment_407979_410479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'tuple_var_assignment_407979')
    # Assigning a type to the variable 'b' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 13), 'b', tuple_var_assignment_407979_410479)
    
    # Assigning a Name to a Name (line 545):
    # Getting the type of 'tuple_var_assignment_407980' (line 545)
    tuple_var_assignment_407980_410480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'tuple_var_assignment_407980')
    # Assigning a type to the variable 'postprocess' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'postprocess', tuple_var_assignment_407980_410480)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'M1' (line 547)
    M1_410481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 7), 'M1')
    # Getting the type of 'None' (line 547)
    None_410482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 13), 'None')
    # Applying the binary operator 'is' (line 547)
    result_is__410483 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 7), 'is', M1_410481, None_410482)
    
    
    # Getting the type of 'M2' (line 547)
    M2_410484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 22), 'M2')
    # Getting the type of 'None' (line 547)
    None_410485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 28), 'None')
    # Applying the binary operator 'is' (line 547)
    result_is__410486 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 22), 'is', M2_410484, None_410485)
    
    # Applying the binary operator 'and' (line 547)
    result_and_keyword_410487 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 7), 'and', result_is__410483, result_is__410486)
    
    # Testing the type of an if condition (line 547)
    if_condition_410488 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 547, 4), result_and_keyword_410487)
    # Assigning a type to the variable 'if_condition_410488' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'if_condition_410488', if_condition_410488)
    # SSA begins for if statement (line 547)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 548)
    str_410489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 22), 'str', 'psolve')
    # Getting the type of 'A_' (line 548)
    A__410490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 19), 'A_')
    
    (may_be_410491, more_types_in_union_410492) = may_provide_member(str_410489, A__410490)

    if may_be_410491:

        if more_types_in_union_410492:
            # Runtime conditional SSA (line 548)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'A_' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'A_', remove_not_member_provider_from_union(A__410490, 'psolve'))

        @norecursion
        def left_psolve(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'left_psolve'
            module_type_store = module_type_store.open_function_context('left_psolve', 549, 12, False)
            
            # Passed parameters checking function
            left_psolve.stypy_localization = localization
            left_psolve.stypy_type_of_self = None
            left_psolve.stypy_type_store = module_type_store
            left_psolve.stypy_function_name = 'left_psolve'
            left_psolve.stypy_param_names_list = ['b']
            left_psolve.stypy_varargs_param_name = None
            left_psolve.stypy_kwargs_param_name = None
            left_psolve.stypy_call_defaults = defaults
            left_psolve.stypy_call_varargs = varargs
            left_psolve.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'left_psolve', ['b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'left_psolve', localization, ['b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'left_psolve(...)' code ##################

            
            # Call to psolve(...): (line 550)
            # Processing the call arguments (line 550)
            # Getting the type of 'b' (line 550)
            b_410495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 33), 'b', False)
            str_410496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 35), 'str', 'left')
            # Processing the call keyword arguments (line 550)
            kwargs_410497 = {}
            # Getting the type of 'A_' (line 550)
            A__410493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 23), 'A_', False)
            # Obtaining the member 'psolve' of a type (line 550)
            psolve_410494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 23), A__410493, 'psolve')
            # Calling psolve(args, kwargs) (line 550)
            psolve_call_result_410498 = invoke(stypy.reporting.localization.Localization(__file__, 550, 23), psolve_410494, *[b_410495, str_410496], **kwargs_410497)
            
            # Assigning a type to the variable 'stypy_return_type' (line 550)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 16), 'stypy_return_type', psolve_call_result_410498)
            
            # ################# End of 'left_psolve(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'left_psolve' in the type store
            # Getting the type of 'stypy_return_type' (line 549)
            stypy_return_type_410499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_410499)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'left_psolve'
            return stypy_return_type_410499

        # Assigning a type to the variable 'left_psolve' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'left_psolve', left_psolve)

        @norecursion
        def right_psolve(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'right_psolve'
            module_type_store = module_type_store.open_function_context('right_psolve', 552, 12, False)
            
            # Passed parameters checking function
            right_psolve.stypy_localization = localization
            right_psolve.stypy_type_of_self = None
            right_psolve.stypy_type_store = module_type_store
            right_psolve.stypy_function_name = 'right_psolve'
            right_psolve.stypy_param_names_list = ['b']
            right_psolve.stypy_varargs_param_name = None
            right_psolve.stypy_kwargs_param_name = None
            right_psolve.stypy_call_defaults = defaults
            right_psolve.stypy_call_varargs = varargs
            right_psolve.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'right_psolve', ['b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'right_psolve', localization, ['b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'right_psolve(...)' code ##################

            
            # Call to psolve(...): (line 553)
            # Processing the call arguments (line 553)
            # Getting the type of 'b' (line 553)
            b_410502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 33), 'b', False)
            str_410503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 35), 'str', 'right')
            # Processing the call keyword arguments (line 553)
            kwargs_410504 = {}
            # Getting the type of 'A_' (line 553)
            A__410500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 23), 'A_', False)
            # Obtaining the member 'psolve' of a type (line 553)
            psolve_410501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 23), A__410500, 'psolve')
            # Calling psolve(args, kwargs) (line 553)
            psolve_call_result_410505 = invoke(stypy.reporting.localization.Localization(__file__, 553, 23), psolve_410501, *[b_410502, str_410503], **kwargs_410504)
            
            # Assigning a type to the variable 'stypy_return_type' (line 553)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 16), 'stypy_return_type', psolve_call_result_410505)
            
            # ################# End of 'right_psolve(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'right_psolve' in the type store
            # Getting the type of 'stypy_return_type' (line 552)
            stypy_return_type_410506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_410506)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'right_psolve'
            return stypy_return_type_410506

        # Assigning a type to the variable 'right_psolve' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'right_psolve', right_psolve)

        @norecursion
        def left_rpsolve(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'left_rpsolve'
            module_type_store = module_type_store.open_function_context('left_rpsolve', 555, 12, False)
            
            # Passed parameters checking function
            left_rpsolve.stypy_localization = localization
            left_rpsolve.stypy_type_of_self = None
            left_rpsolve.stypy_type_store = module_type_store
            left_rpsolve.stypy_function_name = 'left_rpsolve'
            left_rpsolve.stypy_param_names_list = ['b']
            left_rpsolve.stypy_varargs_param_name = None
            left_rpsolve.stypy_kwargs_param_name = None
            left_rpsolve.stypy_call_defaults = defaults
            left_rpsolve.stypy_call_varargs = varargs
            left_rpsolve.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'left_rpsolve', ['b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'left_rpsolve', localization, ['b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'left_rpsolve(...)' code ##################

            
            # Call to rpsolve(...): (line 556)
            # Processing the call arguments (line 556)
            # Getting the type of 'b' (line 556)
            b_410509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 34), 'b', False)
            str_410510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 36), 'str', 'left')
            # Processing the call keyword arguments (line 556)
            kwargs_410511 = {}
            # Getting the type of 'A_' (line 556)
            A__410507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 23), 'A_', False)
            # Obtaining the member 'rpsolve' of a type (line 556)
            rpsolve_410508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 23), A__410507, 'rpsolve')
            # Calling rpsolve(args, kwargs) (line 556)
            rpsolve_call_result_410512 = invoke(stypy.reporting.localization.Localization(__file__, 556, 23), rpsolve_410508, *[b_410509, str_410510], **kwargs_410511)
            
            # Assigning a type to the variable 'stypy_return_type' (line 556)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), 'stypy_return_type', rpsolve_call_result_410512)
            
            # ################# End of 'left_rpsolve(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'left_rpsolve' in the type store
            # Getting the type of 'stypy_return_type' (line 555)
            stypy_return_type_410513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_410513)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'left_rpsolve'
            return stypy_return_type_410513

        # Assigning a type to the variable 'left_rpsolve' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'left_rpsolve', left_rpsolve)

        @norecursion
        def right_rpsolve(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'right_rpsolve'
            module_type_store = module_type_store.open_function_context('right_rpsolve', 558, 12, False)
            
            # Passed parameters checking function
            right_rpsolve.stypy_localization = localization
            right_rpsolve.stypy_type_of_self = None
            right_rpsolve.stypy_type_store = module_type_store
            right_rpsolve.stypy_function_name = 'right_rpsolve'
            right_rpsolve.stypy_param_names_list = ['b']
            right_rpsolve.stypy_varargs_param_name = None
            right_rpsolve.stypy_kwargs_param_name = None
            right_rpsolve.stypy_call_defaults = defaults
            right_rpsolve.stypy_call_varargs = varargs
            right_rpsolve.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'right_rpsolve', ['b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'right_rpsolve', localization, ['b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'right_rpsolve(...)' code ##################

            
            # Call to rpsolve(...): (line 559)
            # Processing the call arguments (line 559)
            # Getting the type of 'b' (line 559)
            b_410516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 34), 'b', False)
            str_410517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 36), 'str', 'right')
            # Processing the call keyword arguments (line 559)
            kwargs_410518 = {}
            # Getting the type of 'A_' (line 559)
            A__410514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 23), 'A_', False)
            # Obtaining the member 'rpsolve' of a type (line 559)
            rpsolve_410515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 23), A__410514, 'rpsolve')
            # Calling rpsolve(args, kwargs) (line 559)
            rpsolve_call_result_410519 = invoke(stypy.reporting.localization.Localization(__file__, 559, 23), rpsolve_410515, *[b_410516, str_410517], **kwargs_410518)
            
            # Assigning a type to the variable 'stypy_return_type' (line 559)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 16), 'stypy_return_type', rpsolve_call_result_410519)
            
            # ################# End of 'right_rpsolve(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'right_rpsolve' in the type store
            # Getting the type of 'stypy_return_type' (line 558)
            stypy_return_type_410520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_410520)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'right_rpsolve'
            return stypy_return_type_410520

        # Assigning a type to the variable 'right_rpsolve' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'right_rpsolve', right_rpsolve)
        
        # Assigning a Call to a Name (line 560):
        
        # Assigning a Call to a Name (line 560):
        
        # Call to LinearOperator(...): (line 560)
        # Processing the call arguments (line 560)
        # Getting the type of 'A' (line 560)
        A_410522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 32), 'A', False)
        # Obtaining the member 'shape' of a type (line 560)
        shape_410523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 32), A_410522, 'shape')
        # Processing the call keyword arguments (line 560)
        # Getting the type of 'left_psolve' (line 560)
        left_psolve_410524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 48), 'left_psolve', False)
        keyword_410525 = left_psolve_410524
        # Getting the type of 'left_rpsolve' (line 560)
        left_rpsolve_410526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 69), 'left_rpsolve', False)
        keyword_410527 = left_rpsolve_410526
        kwargs_410528 = {'rmatvec': keyword_410527, 'matvec': keyword_410525}
        # Getting the type of 'LinearOperator' (line 560)
        LinearOperator_410521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 17), 'LinearOperator', False)
        # Calling LinearOperator(args, kwargs) (line 560)
        LinearOperator_call_result_410529 = invoke(stypy.reporting.localization.Localization(__file__, 560, 17), LinearOperator_410521, *[shape_410523], **kwargs_410528)
        
        # Assigning a type to the variable 'M1' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 12), 'M1', LinearOperator_call_result_410529)
        
        # Assigning a Call to a Name (line 561):
        
        # Assigning a Call to a Name (line 561):
        
        # Call to LinearOperator(...): (line 561)
        # Processing the call arguments (line 561)
        # Getting the type of 'A' (line 561)
        A_410531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 32), 'A', False)
        # Obtaining the member 'shape' of a type (line 561)
        shape_410532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 32), A_410531, 'shape')
        # Processing the call keyword arguments (line 561)
        # Getting the type of 'right_psolve' (line 561)
        right_psolve_410533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 48), 'right_psolve', False)
        keyword_410534 = right_psolve_410533
        # Getting the type of 'right_rpsolve' (line 561)
        right_rpsolve_410535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 70), 'right_rpsolve', False)
        keyword_410536 = right_rpsolve_410535
        kwargs_410537 = {'rmatvec': keyword_410536, 'matvec': keyword_410534}
        # Getting the type of 'LinearOperator' (line 561)
        LinearOperator_410530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 17), 'LinearOperator', False)
        # Calling LinearOperator(args, kwargs) (line 561)
        LinearOperator_call_result_410538 = invoke(stypy.reporting.localization.Localization(__file__, 561, 17), LinearOperator_410530, *[shape_410532], **kwargs_410537)
        
        # Assigning a type to the variable 'M2' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'M2', LinearOperator_call_result_410538)

        if more_types_in_union_410492:
            # Runtime conditional SSA for else branch (line 548)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_410491) or more_types_in_union_410492):
        # Assigning a type to the variable 'A_' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'A_', remove_member_provider_from_union(A__410490, 'psolve'))

        @norecursion
        def id(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'id'
            module_type_store = module_type_store.open_function_context('id', 563, 12, False)
            
            # Passed parameters checking function
            id.stypy_localization = localization
            id.stypy_type_of_self = None
            id.stypy_type_store = module_type_store
            id.stypy_function_name = 'id'
            id.stypy_param_names_list = ['b']
            id.stypy_varargs_param_name = None
            id.stypy_kwargs_param_name = None
            id.stypy_call_defaults = defaults
            id.stypy_call_varargs = varargs
            id.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'id', ['b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'id', localization, ['b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'id(...)' code ##################

            # Getting the type of 'b' (line 564)
            b_410539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 23), 'b')
            # Assigning a type to the variable 'stypy_return_type' (line 564)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 16), 'stypy_return_type', b_410539)
            
            # ################# End of 'id(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'id' in the type store
            # Getting the type of 'stypy_return_type' (line 563)
            stypy_return_type_410540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_410540)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'id'
            return stypy_return_type_410540

        # Assigning a type to the variable 'id' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'id', id)
        
        # Assigning a Call to a Name (line 565):
        
        # Assigning a Call to a Name (line 565):
        
        # Call to LinearOperator(...): (line 565)
        # Processing the call arguments (line 565)
        # Getting the type of 'A' (line 565)
        A_410542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 32), 'A', False)
        # Obtaining the member 'shape' of a type (line 565)
        shape_410543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 32), A_410542, 'shape')
        # Processing the call keyword arguments (line 565)
        # Getting the type of 'id' (line 565)
        id_410544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 48), 'id', False)
        keyword_410545 = id_410544
        # Getting the type of 'id' (line 565)
        id_410546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 60), 'id', False)
        keyword_410547 = id_410546
        kwargs_410548 = {'rmatvec': keyword_410547, 'matvec': keyword_410545}
        # Getting the type of 'LinearOperator' (line 565)
        LinearOperator_410541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 17), 'LinearOperator', False)
        # Calling LinearOperator(args, kwargs) (line 565)
        LinearOperator_call_result_410549 = invoke(stypy.reporting.localization.Localization(__file__, 565, 17), LinearOperator_410541, *[shape_410543], **kwargs_410548)
        
        # Assigning a type to the variable 'M1' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'M1', LinearOperator_call_result_410549)
        
        # Assigning a Call to a Name (line 566):
        
        # Assigning a Call to a Name (line 566):
        
        # Call to LinearOperator(...): (line 566)
        # Processing the call arguments (line 566)
        # Getting the type of 'A' (line 566)
        A_410551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 32), 'A', False)
        # Obtaining the member 'shape' of a type (line 566)
        shape_410552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 32), A_410551, 'shape')
        # Processing the call keyword arguments (line 566)
        # Getting the type of 'id' (line 566)
        id_410553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 48), 'id', False)
        keyword_410554 = id_410553
        # Getting the type of 'id' (line 566)
        id_410555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 60), 'id', False)
        keyword_410556 = id_410555
        kwargs_410557 = {'rmatvec': keyword_410556, 'matvec': keyword_410554}
        # Getting the type of 'LinearOperator' (line 566)
        LinearOperator_410550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 17), 'LinearOperator', False)
        # Calling LinearOperator(args, kwargs) (line 566)
        LinearOperator_call_result_410558 = invoke(stypy.reporting.localization.Localization(__file__, 566, 17), LinearOperator_410550, *[shape_410552], **kwargs_410557)
        
        # Assigning a type to the variable 'M2' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'M2', LinearOperator_call_result_410558)

        if (may_be_410491 and more_types_in_union_410492):
            # SSA join for if statement (line 548)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 547)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 568):
    
    # Assigning a Call to a Name (line 568):
    
    # Call to len(...): (line 568)
    # Processing the call arguments (line 568)
    # Getting the type of 'b' (line 568)
    b_410560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'b', False)
    # Processing the call keyword arguments (line 568)
    kwargs_410561 = {}
    # Getting the type of 'len' (line 568)
    len_410559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'len', False)
    # Calling len(args, kwargs) (line 568)
    len_call_result_410562 = invoke(stypy.reporting.localization.Localization(__file__, 568, 8), len_410559, *[b_410560], **kwargs_410561)
    
    # Assigning a type to the variable 'n' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'n', len_call_result_410562)
    
    # Type idiom detected: calculating its left and rigth part (line 569)
    # Getting the type of 'maxiter' (line 569)
    maxiter_410563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 7), 'maxiter')
    # Getting the type of 'None' (line 569)
    None_410564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 18), 'None')
    
    (may_be_410565, more_types_in_union_410566) = may_be_none(maxiter_410563, None_410564)

    if may_be_410565:

        if more_types_in_union_410566:
            # Runtime conditional SSA (line 569)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 570):
        
        # Assigning a BinOp to a Name (line 570):
        # Getting the type of 'n' (line 570)
        n_410567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 18), 'n')
        int_410568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 20), 'int')
        # Applying the binary operator '*' (line 570)
        result_mul_410569 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 18), '*', n_410567, int_410568)
        
        # Assigning a type to the variable 'maxiter' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'maxiter', result_mul_410569)

        if more_types_in_union_410566:
            # SSA join for if statement (line 569)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Subscript to a Name (line 572):
    
    # Assigning a Subscript to a Name (line 572):
    
    # Obtaining the type of the subscript
    # Getting the type of 'x' (line 572)
    x_410570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 21), 'x')
    # Obtaining the member 'dtype' of a type (line 572)
    dtype_410571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 21), x_410570, 'dtype')
    # Obtaining the member 'char' of a type (line 572)
    char_410572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 21), dtype_410571, 'char')
    # Getting the type of '_type_conv' (line 572)
    _type_conv_410573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 10), '_type_conv')
    # Obtaining the member '__getitem__' of a type (line 572)
    getitem___410574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 10), _type_conv_410573, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 572)
    subscript_call_result_410575 = invoke(stypy.reporting.localization.Localization(__file__, 572, 10), getitem___410574, char_410572)
    
    # Assigning a type to the variable 'ltr' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'ltr', subscript_call_result_410575)
    
    # Assigning a Call to a Name (line 573):
    
    # Assigning a Call to a Name (line 573):
    
    # Call to getattr(...): (line 573)
    # Processing the call arguments (line 573)
    # Getting the type of '_iterative' (line 573)
    _iterative_410577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 21), '_iterative', False)
    # Getting the type of 'ltr' (line 573)
    ltr_410578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 33), 'ltr', False)
    str_410579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 39), 'str', 'qmrrevcom')
    # Applying the binary operator '+' (line 573)
    result_add_410580 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 33), '+', ltr_410578, str_410579)
    
    # Processing the call keyword arguments (line 573)
    kwargs_410581 = {}
    # Getting the type of 'getattr' (line 573)
    getattr_410576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 13), 'getattr', False)
    # Calling getattr(args, kwargs) (line 573)
    getattr_call_result_410582 = invoke(stypy.reporting.localization.Localization(__file__, 573, 13), getattr_410576, *[_iterative_410577, result_add_410580], **kwargs_410581)
    
    # Assigning a type to the variable 'revcom' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'revcom', getattr_call_result_410582)
    
    # Assigning a Call to a Name (line 574):
    
    # Assigning a Call to a Name (line 574):
    
    # Call to getattr(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of '_iterative' (line 574)
    _iterative_410584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 23), '_iterative', False)
    # Getting the type of 'ltr' (line 574)
    ltr_410585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 35), 'ltr', False)
    str_410586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 41), 'str', 'stoptest2')
    # Applying the binary operator '+' (line 574)
    result_add_410587 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 35), '+', ltr_410585, str_410586)
    
    # Processing the call keyword arguments (line 574)
    kwargs_410588 = {}
    # Getting the type of 'getattr' (line 574)
    getattr_410583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 15), 'getattr', False)
    # Calling getattr(args, kwargs) (line 574)
    getattr_call_result_410589 = invoke(stypy.reporting.localization.Localization(__file__, 574, 15), getattr_410583, *[_iterative_410584, result_add_410587], **kwargs_410588)
    
    # Assigning a type to the variable 'stoptest' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'stoptest', getattr_call_result_410589)
    
    # Assigning a Name to a Name (line 576):
    
    # Assigning a Name to a Name (line 576):
    # Getting the type of 'tol' (line 576)
    tol_410590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'tol')
    # Assigning a type to the variable 'resid' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'resid', tol_410590)
    
    # Assigning a Num to a Name (line 577):
    
    # Assigning a Num to a Name (line 577):
    int_410591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 11), 'int')
    # Assigning a type to the variable 'ndx1' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'ndx1', int_410591)
    
    # Assigning a Num to a Name (line 578):
    
    # Assigning a Num to a Name (line 578):
    int_410592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 11), 'int')
    # Assigning a type to the variable 'ndx2' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'ndx2', int_410592)
    
    # Assigning a Call to a Name (line 580):
    
    # Assigning a Call to a Name (line 580):
    
    # Call to _aligned_zeros(...): (line 580)
    # Processing the call arguments (line 580)
    int_410594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 26), 'int')
    # Getting the type of 'n' (line 580)
    n_410595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 29), 'n', False)
    # Applying the binary operator '*' (line 580)
    result_mul_410596 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 26), '*', int_410594, n_410595)
    
    # Getting the type of 'x' (line 580)
    x_410597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 31), 'x', False)
    # Obtaining the member 'dtype' of a type (line 580)
    dtype_410598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 31), x_410597, 'dtype')
    # Processing the call keyword arguments (line 580)
    kwargs_410599 = {}
    # Getting the type of '_aligned_zeros' (line 580)
    _aligned_zeros_410593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 11), '_aligned_zeros', False)
    # Calling _aligned_zeros(args, kwargs) (line 580)
    _aligned_zeros_call_result_410600 = invoke(stypy.reporting.localization.Localization(__file__, 580, 11), _aligned_zeros_410593, *[result_mul_410596, dtype_410598], **kwargs_410599)
    
    # Assigning a type to the variable 'work' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'work', _aligned_zeros_call_result_410600)
    
    # Assigning a Num to a Name (line 581):
    
    # Assigning a Num to a Name (line 581):
    int_410601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 11), 'int')
    # Assigning a type to the variable 'ijob' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'ijob', int_410601)
    
    # Assigning a Num to a Name (line 582):
    
    # Assigning a Num to a Name (line 582):
    int_410602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 11), 'int')
    # Assigning a type to the variable 'info' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'info', int_410602)
    
    # Assigning a Name to a Name (line 583):
    
    # Assigning a Name to a Name (line 583):
    # Getting the type of 'True' (line 583)
    True_410603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 13), 'True')
    # Assigning a type to the variable 'ftflag' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'ftflag', True_410603)
    
    # Assigning a Num to a Name (line 584):
    
    # Assigning a Num to a Name (line 584):
    float_410604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 12), 'float')
    # Assigning a type to the variable 'bnrm2' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'bnrm2', float_410604)
    
    # Assigning a Name to a Name (line 585):
    
    # Assigning a Name to a Name (line 585):
    # Getting the type of 'maxiter' (line 585)
    maxiter_410605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'maxiter')
    # Assigning a type to the variable 'iter_' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'iter_', maxiter_410605)
    
    # Getting the type of 'True' (line 586)
    True_410606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 10), 'True')
    # Testing the type of an if condition (line 586)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 586, 4), True_410606)
    # SSA begins for while statement (line 586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Name to a Name (line 587):
    
    # Assigning a Name to a Name (line 587):
    # Getting the type of 'iter_' (line 587)
    iter__410607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 18), 'iter_')
    # Assigning a type to the variable 'olditer' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'olditer', iter__410607)
    
    # Assigning a Call to a Tuple (line 588):
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    int_410608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 8), 'int')
    
    # Call to revcom(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'b' (line 589)
    b_410610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 18), 'b', False)
    # Getting the type of 'x' (line 589)
    x_410611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'x', False)
    # Getting the type of 'work' (line 589)
    work_410612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 24), 'work', False)
    # Getting the type of 'iter_' (line 589)
    iter__410613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 30), 'iter_', False)
    # Getting the type of 'resid' (line 589)
    resid_410614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 37), 'resid', False)
    # Getting the type of 'info' (line 589)
    info_410615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 44), 'info', False)
    # Getting the type of 'ndx1' (line 589)
    ndx1_410616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 589)
    ndx2_410617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 589)
    ijob_410618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 62), 'ijob', False)
    # Processing the call keyword arguments (line 589)
    kwargs_410619 = {}
    # Getting the type of 'revcom' (line 589)
    revcom_410609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 589)
    revcom_call_result_410620 = invoke(stypy.reporting.localization.Localization(__file__, 589, 11), revcom_410609, *[b_410610, x_410611, work_410612, iter__410613, resid_410614, info_410615, ndx1_410616, ndx2_410617, ijob_410618], **kwargs_410619)
    
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___410621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), revcom_call_result_410620, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_410622 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), getitem___410621, int_410608)
    
    # Assigning a type to the variable 'tuple_var_assignment_407981' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407981', subscript_call_result_410622)
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    int_410623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 8), 'int')
    
    # Call to revcom(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'b' (line 589)
    b_410625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 18), 'b', False)
    # Getting the type of 'x' (line 589)
    x_410626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'x', False)
    # Getting the type of 'work' (line 589)
    work_410627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 24), 'work', False)
    # Getting the type of 'iter_' (line 589)
    iter__410628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 30), 'iter_', False)
    # Getting the type of 'resid' (line 589)
    resid_410629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 37), 'resid', False)
    # Getting the type of 'info' (line 589)
    info_410630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 44), 'info', False)
    # Getting the type of 'ndx1' (line 589)
    ndx1_410631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 589)
    ndx2_410632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 589)
    ijob_410633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 62), 'ijob', False)
    # Processing the call keyword arguments (line 589)
    kwargs_410634 = {}
    # Getting the type of 'revcom' (line 589)
    revcom_410624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 589)
    revcom_call_result_410635 = invoke(stypy.reporting.localization.Localization(__file__, 589, 11), revcom_410624, *[b_410625, x_410626, work_410627, iter__410628, resid_410629, info_410630, ndx1_410631, ndx2_410632, ijob_410633], **kwargs_410634)
    
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___410636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), revcom_call_result_410635, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_410637 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), getitem___410636, int_410623)
    
    # Assigning a type to the variable 'tuple_var_assignment_407982' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407982', subscript_call_result_410637)
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    int_410638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 8), 'int')
    
    # Call to revcom(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'b' (line 589)
    b_410640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 18), 'b', False)
    # Getting the type of 'x' (line 589)
    x_410641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'x', False)
    # Getting the type of 'work' (line 589)
    work_410642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 24), 'work', False)
    # Getting the type of 'iter_' (line 589)
    iter__410643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 30), 'iter_', False)
    # Getting the type of 'resid' (line 589)
    resid_410644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 37), 'resid', False)
    # Getting the type of 'info' (line 589)
    info_410645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 44), 'info', False)
    # Getting the type of 'ndx1' (line 589)
    ndx1_410646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 589)
    ndx2_410647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 589)
    ijob_410648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 62), 'ijob', False)
    # Processing the call keyword arguments (line 589)
    kwargs_410649 = {}
    # Getting the type of 'revcom' (line 589)
    revcom_410639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 589)
    revcom_call_result_410650 = invoke(stypy.reporting.localization.Localization(__file__, 589, 11), revcom_410639, *[b_410640, x_410641, work_410642, iter__410643, resid_410644, info_410645, ndx1_410646, ndx2_410647, ijob_410648], **kwargs_410649)
    
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___410651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), revcom_call_result_410650, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_410652 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), getitem___410651, int_410638)
    
    # Assigning a type to the variable 'tuple_var_assignment_407983' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407983', subscript_call_result_410652)
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    int_410653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 8), 'int')
    
    # Call to revcom(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'b' (line 589)
    b_410655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 18), 'b', False)
    # Getting the type of 'x' (line 589)
    x_410656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'x', False)
    # Getting the type of 'work' (line 589)
    work_410657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 24), 'work', False)
    # Getting the type of 'iter_' (line 589)
    iter__410658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 30), 'iter_', False)
    # Getting the type of 'resid' (line 589)
    resid_410659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 37), 'resid', False)
    # Getting the type of 'info' (line 589)
    info_410660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 44), 'info', False)
    # Getting the type of 'ndx1' (line 589)
    ndx1_410661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 589)
    ndx2_410662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 589)
    ijob_410663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 62), 'ijob', False)
    # Processing the call keyword arguments (line 589)
    kwargs_410664 = {}
    # Getting the type of 'revcom' (line 589)
    revcom_410654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 589)
    revcom_call_result_410665 = invoke(stypy.reporting.localization.Localization(__file__, 589, 11), revcom_410654, *[b_410655, x_410656, work_410657, iter__410658, resid_410659, info_410660, ndx1_410661, ndx2_410662, ijob_410663], **kwargs_410664)
    
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___410666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), revcom_call_result_410665, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_410667 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), getitem___410666, int_410653)
    
    # Assigning a type to the variable 'tuple_var_assignment_407984' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407984', subscript_call_result_410667)
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    int_410668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 8), 'int')
    
    # Call to revcom(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'b' (line 589)
    b_410670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 18), 'b', False)
    # Getting the type of 'x' (line 589)
    x_410671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'x', False)
    # Getting the type of 'work' (line 589)
    work_410672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 24), 'work', False)
    # Getting the type of 'iter_' (line 589)
    iter__410673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 30), 'iter_', False)
    # Getting the type of 'resid' (line 589)
    resid_410674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 37), 'resid', False)
    # Getting the type of 'info' (line 589)
    info_410675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 44), 'info', False)
    # Getting the type of 'ndx1' (line 589)
    ndx1_410676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 589)
    ndx2_410677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 589)
    ijob_410678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 62), 'ijob', False)
    # Processing the call keyword arguments (line 589)
    kwargs_410679 = {}
    # Getting the type of 'revcom' (line 589)
    revcom_410669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 589)
    revcom_call_result_410680 = invoke(stypy.reporting.localization.Localization(__file__, 589, 11), revcom_410669, *[b_410670, x_410671, work_410672, iter__410673, resid_410674, info_410675, ndx1_410676, ndx2_410677, ijob_410678], **kwargs_410679)
    
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___410681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), revcom_call_result_410680, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_410682 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), getitem___410681, int_410668)
    
    # Assigning a type to the variable 'tuple_var_assignment_407985' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407985', subscript_call_result_410682)
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    int_410683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 8), 'int')
    
    # Call to revcom(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'b' (line 589)
    b_410685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 18), 'b', False)
    # Getting the type of 'x' (line 589)
    x_410686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'x', False)
    # Getting the type of 'work' (line 589)
    work_410687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 24), 'work', False)
    # Getting the type of 'iter_' (line 589)
    iter__410688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 30), 'iter_', False)
    # Getting the type of 'resid' (line 589)
    resid_410689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 37), 'resid', False)
    # Getting the type of 'info' (line 589)
    info_410690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 44), 'info', False)
    # Getting the type of 'ndx1' (line 589)
    ndx1_410691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 589)
    ndx2_410692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 589)
    ijob_410693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 62), 'ijob', False)
    # Processing the call keyword arguments (line 589)
    kwargs_410694 = {}
    # Getting the type of 'revcom' (line 589)
    revcom_410684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 589)
    revcom_call_result_410695 = invoke(stypy.reporting.localization.Localization(__file__, 589, 11), revcom_410684, *[b_410685, x_410686, work_410687, iter__410688, resid_410689, info_410690, ndx1_410691, ndx2_410692, ijob_410693], **kwargs_410694)
    
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___410696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), revcom_call_result_410695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_410697 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), getitem___410696, int_410683)
    
    # Assigning a type to the variable 'tuple_var_assignment_407986' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407986', subscript_call_result_410697)
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    int_410698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 8), 'int')
    
    # Call to revcom(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'b' (line 589)
    b_410700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 18), 'b', False)
    # Getting the type of 'x' (line 589)
    x_410701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'x', False)
    # Getting the type of 'work' (line 589)
    work_410702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 24), 'work', False)
    # Getting the type of 'iter_' (line 589)
    iter__410703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 30), 'iter_', False)
    # Getting the type of 'resid' (line 589)
    resid_410704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 37), 'resid', False)
    # Getting the type of 'info' (line 589)
    info_410705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 44), 'info', False)
    # Getting the type of 'ndx1' (line 589)
    ndx1_410706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 589)
    ndx2_410707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 589)
    ijob_410708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 62), 'ijob', False)
    # Processing the call keyword arguments (line 589)
    kwargs_410709 = {}
    # Getting the type of 'revcom' (line 589)
    revcom_410699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 589)
    revcom_call_result_410710 = invoke(stypy.reporting.localization.Localization(__file__, 589, 11), revcom_410699, *[b_410700, x_410701, work_410702, iter__410703, resid_410704, info_410705, ndx1_410706, ndx2_410707, ijob_410708], **kwargs_410709)
    
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___410711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), revcom_call_result_410710, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_410712 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), getitem___410711, int_410698)
    
    # Assigning a type to the variable 'tuple_var_assignment_407987' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407987', subscript_call_result_410712)
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    int_410713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 8), 'int')
    
    # Call to revcom(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'b' (line 589)
    b_410715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 18), 'b', False)
    # Getting the type of 'x' (line 589)
    x_410716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'x', False)
    # Getting the type of 'work' (line 589)
    work_410717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 24), 'work', False)
    # Getting the type of 'iter_' (line 589)
    iter__410718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 30), 'iter_', False)
    # Getting the type of 'resid' (line 589)
    resid_410719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 37), 'resid', False)
    # Getting the type of 'info' (line 589)
    info_410720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 44), 'info', False)
    # Getting the type of 'ndx1' (line 589)
    ndx1_410721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 589)
    ndx2_410722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 589)
    ijob_410723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 62), 'ijob', False)
    # Processing the call keyword arguments (line 589)
    kwargs_410724 = {}
    # Getting the type of 'revcom' (line 589)
    revcom_410714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 589)
    revcom_call_result_410725 = invoke(stypy.reporting.localization.Localization(__file__, 589, 11), revcom_410714, *[b_410715, x_410716, work_410717, iter__410718, resid_410719, info_410720, ndx1_410721, ndx2_410722, ijob_410723], **kwargs_410724)
    
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___410726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), revcom_call_result_410725, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_410727 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), getitem___410726, int_410713)
    
    # Assigning a type to the variable 'tuple_var_assignment_407988' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407988', subscript_call_result_410727)
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    int_410728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 8), 'int')
    
    # Call to revcom(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'b' (line 589)
    b_410730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 18), 'b', False)
    # Getting the type of 'x' (line 589)
    x_410731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'x', False)
    # Getting the type of 'work' (line 589)
    work_410732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 24), 'work', False)
    # Getting the type of 'iter_' (line 589)
    iter__410733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 30), 'iter_', False)
    # Getting the type of 'resid' (line 589)
    resid_410734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 37), 'resid', False)
    # Getting the type of 'info' (line 589)
    info_410735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 44), 'info', False)
    # Getting the type of 'ndx1' (line 589)
    ndx1_410736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 50), 'ndx1', False)
    # Getting the type of 'ndx2' (line 589)
    ndx2_410737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 56), 'ndx2', False)
    # Getting the type of 'ijob' (line 589)
    ijob_410738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 62), 'ijob', False)
    # Processing the call keyword arguments (line 589)
    kwargs_410739 = {}
    # Getting the type of 'revcom' (line 589)
    revcom_410729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 11), 'revcom', False)
    # Calling revcom(args, kwargs) (line 589)
    revcom_call_result_410740 = invoke(stypy.reporting.localization.Localization(__file__, 589, 11), revcom_410729, *[b_410730, x_410731, work_410732, iter__410733, resid_410734, info_410735, ndx1_410736, ndx2_410737, ijob_410738], **kwargs_410739)
    
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___410741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), revcom_call_result_410740, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_410742 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), getitem___410741, int_410728)
    
    # Assigning a type to the variable 'tuple_var_assignment_407989' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407989', subscript_call_result_410742)
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'tuple_var_assignment_407981' (line 588)
    tuple_var_assignment_407981_410743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407981')
    # Assigning a type to the variable 'x' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'x', tuple_var_assignment_407981_410743)
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'tuple_var_assignment_407982' (line 588)
    tuple_var_assignment_407982_410744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407982')
    # Assigning a type to the variable 'iter_' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 11), 'iter_', tuple_var_assignment_407982_410744)
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'tuple_var_assignment_407983' (line 588)
    tuple_var_assignment_407983_410745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407983')
    # Assigning a type to the variable 'resid' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 18), 'resid', tuple_var_assignment_407983_410745)
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'tuple_var_assignment_407984' (line 588)
    tuple_var_assignment_407984_410746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407984')
    # Assigning a type to the variable 'info' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 25), 'info', tuple_var_assignment_407984_410746)
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'tuple_var_assignment_407985' (line 588)
    tuple_var_assignment_407985_410747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407985')
    # Assigning a type to the variable 'ndx1' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 31), 'ndx1', tuple_var_assignment_407985_410747)
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'tuple_var_assignment_407986' (line 588)
    tuple_var_assignment_407986_410748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407986')
    # Assigning a type to the variable 'ndx2' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 37), 'ndx2', tuple_var_assignment_407986_410748)
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'tuple_var_assignment_407987' (line 588)
    tuple_var_assignment_407987_410749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407987')
    # Assigning a type to the variable 'sclr1' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 43), 'sclr1', tuple_var_assignment_407987_410749)
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'tuple_var_assignment_407988' (line 588)
    tuple_var_assignment_407988_410750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407988')
    # Assigning a type to the variable 'sclr2' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 50), 'sclr2', tuple_var_assignment_407988_410750)
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'tuple_var_assignment_407989' (line 588)
    tuple_var_assignment_407989_410751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_407989')
    # Assigning a type to the variable 'ijob' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 57), 'ijob', tuple_var_assignment_407989_410751)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'callback' (line 590)
    callback_410752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 11), 'callback')
    # Getting the type of 'None' (line 590)
    None_410753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 27), 'None')
    # Applying the binary operator 'isnot' (line 590)
    result_is_not_410754 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 11), 'isnot', callback_410752, None_410753)
    
    
    # Getting the type of 'iter_' (line 590)
    iter__410755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 36), 'iter_')
    # Getting the type of 'olditer' (line 590)
    olditer_410756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 44), 'olditer')
    # Applying the binary operator '>' (line 590)
    result_gt_410757 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 36), '>', iter__410755, olditer_410756)
    
    # Applying the binary operator 'and' (line 590)
    result_and_keyword_410758 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 11), 'and', result_is_not_410754, result_gt_410757)
    
    # Testing the type of an if condition (line 590)
    if_condition_410759 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 590, 8), result_and_keyword_410758)
    # Assigning a type to the variable 'if_condition_410759' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'if_condition_410759', if_condition_410759)
    # SSA begins for if statement (line 590)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to callback(...): (line 591)
    # Processing the call arguments (line 591)
    # Getting the type of 'x' (line 591)
    x_410761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 21), 'x', False)
    # Processing the call keyword arguments (line 591)
    kwargs_410762 = {}
    # Getting the type of 'callback' (line 591)
    callback_410760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 12), 'callback', False)
    # Calling callback(args, kwargs) (line 591)
    callback_call_result_410763 = invoke(stypy.reporting.localization.Localization(__file__, 591, 12), callback_410760, *[x_410761], **kwargs_410762)
    
    # SSA join for if statement (line 590)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 592):
    
    # Assigning a Call to a Name (line 592):
    
    # Call to slice(...): (line 592)
    # Processing the call arguments (line 592)
    # Getting the type of 'ndx1' (line 592)
    ndx1_410765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 23), 'ndx1', False)
    int_410766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 28), 'int')
    # Applying the binary operator '-' (line 592)
    result_sub_410767 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 23), '-', ndx1_410765, int_410766)
    
    # Getting the type of 'ndx1' (line 592)
    ndx1_410768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 31), 'ndx1', False)
    int_410769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 36), 'int')
    # Applying the binary operator '-' (line 592)
    result_sub_410770 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 31), '-', ndx1_410768, int_410769)
    
    # Getting the type of 'n' (line 592)
    n_410771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 38), 'n', False)
    # Applying the binary operator '+' (line 592)
    result_add_410772 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 37), '+', result_sub_410770, n_410771)
    
    # Processing the call keyword arguments (line 592)
    kwargs_410773 = {}
    # Getting the type of 'slice' (line 592)
    slice_410764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 592)
    slice_call_result_410774 = invoke(stypy.reporting.localization.Localization(__file__, 592, 17), slice_410764, *[result_sub_410767, result_add_410772], **kwargs_410773)
    
    # Assigning a type to the variable 'slice1' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'slice1', slice_call_result_410774)
    
    # Assigning a Call to a Name (line 593):
    
    # Assigning a Call to a Name (line 593):
    
    # Call to slice(...): (line 593)
    # Processing the call arguments (line 593)
    # Getting the type of 'ndx2' (line 593)
    ndx2_410776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 23), 'ndx2', False)
    int_410777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 28), 'int')
    # Applying the binary operator '-' (line 593)
    result_sub_410778 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 23), '-', ndx2_410776, int_410777)
    
    # Getting the type of 'ndx2' (line 593)
    ndx2_410779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 31), 'ndx2', False)
    int_410780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 36), 'int')
    # Applying the binary operator '-' (line 593)
    result_sub_410781 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 31), '-', ndx2_410779, int_410780)
    
    # Getting the type of 'n' (line 593)
    n_410782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 38), 'n', False)
    # Applying the binary operator '+' (line 593)
    result_add_410783 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 37), '+', result_sub_410781, n_410782)
    
    # Processing the call keyword arguments (line 593)
    kwargs_410784 = {}
    # Getting the type of 'slice' (line 593)
    slice_410775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 593)
    slice_call_result_410785 = invoke(stypy.reporting.localization.Localization(__file__, 593, 17), slice_410775, *[result_sub_410778, result_add_410783], **kwargs_410784)
    
    # Assigning a type to the variable 'slice2' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'slice2', slice_call_result_410785)
    
    
    # Getting the type of 'ijob' (line 594)
    ijob_410786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'ijob')
    int_410787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 20), 'int')
    # Applying the binary operator '==' (line 594)
    result_eq_410788 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 12), '==', ijob_410786, int_410787)
    
    # Testing the type of an if condition (line 594)
    if_condition_410789 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 594, 8), result_eq_410788)
    # Assigning a type to the variable 'if_condition_410789' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'if_condition_410789', if_condition_410789)
    # SSA begins for if statement (line 594)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 595)
    # Getting the type of 'callback' (line 595)
    callback_410790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'callback')
    # Getting the type of 'None' (line 595)
    None_410791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 31), 'None')
    
    (may_be_410792, more_types_in_union_410793) = may_not_be_none(callback_410790, None_410791)

    if may_be_410792:

        if more_types_in_union_410793:
            # Runtime conditional SSA (line 595)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to callback(...): (line 596)
        # Processing the call arguments (line 596)
        # Getting the type of 'x' (line 596)
        x_410795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 25), 'x', False)
        # Processing the call keyword arguments (line 596)
        kwargs_410796 = {}
        # Getting the type of 'callback' (line 596)
        callback_410794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 16), 'callback', False)
        # Calling callback(args, kwargs) (line 596)
        callback_call_result_410797 = invoke(stypy.reporting.localization.Localization(__file__, 596, 16), callback_410794, *[x_410795], **kwargs_410796)
        

        if more_types_in_union_410793:
            # SSA join for if statement (line 595)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 594)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 598)
    ijob_410798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 14), 'ijob')
    int_410799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 22), 'int')
    # Applying the binary operator '==' (line 598)
    result_eq_410800 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 14), '==', ijob_410798, int_410799)
    
    # Testing the type of an if condition (line 598)
    if_condition_410801 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 598, 13), result_eq_410800)
    # Assigning a type to the variable 'if_condition_410801' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 13), 'if_condition_410801', if_condition_410801)
    # SSA begins for if statement (line 598)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 599)
    work_410802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 599)
    slice2_410803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 17), 'slice2')
    # Getting the type of 'work' (line 599)
    work_410804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 599)
    getitem___410805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 12), work_410804, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 599)
    subscript_call_result_410806 = invoke(stypy.reporting.localization.Localization(__file__, 599, 12), getitem___410805, slice2_410803)
    
    # Getting the type of 'sclr2' (line 599)
    sclr2_410807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 28), 'sclr2')
    # Applying the binary operator '*=' (line 599)
    result_imul_410808 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 12), '*=', subscript_call_result_410806, sclr2_410807)
    # Getting the type of 'work' (line 599)
    work_410809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 12), 'work')
    # Getting the type of 'slice2' (line 599)
    slice2_410810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 17), 'slice2')
    # Storing an element on a container (line 599)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 12), work_410809, (slice2_410810, result_imul_410808))
    
    
    # Getting the type of 'work' (line 600)
    work_410811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 600)
    slice2_410812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 17), 'slice2')
    # Getting the type of 'work' (line 600)
    work_410813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 600)
    getitem___410814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 12), work_410813, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 600)
    subscript_call_result_410815 = invoke(stypy.reporting.localization.Localization(__file__, 600, 12), getitem___410814, slice2_410812)
    
    # Getting the type of 'sclr1' (line 600)
    sclr1_410816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 28), 'sclr1')
    
    # Call to matvec(...): (line 600)
    # Processing the call arguments (line 600)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 600)
    slice1_410819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 48), 'slice1', False)
    # Getting the type of 'work' (line 600)
    work_410820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 43), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 600)
    getitem___410821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 43), work_410820, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 600)
    subscript_call_result_410822 = invoke(stypy.reporting.localization.Localization(__file__, 600, 43), getitem___410821, slice1_410819)
    
    # Processing the call keyword arguments (line 600)
    kwargs_410823 = {}
    # Getting the type of 'A' (line 600)
    A_410817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 34), 'A', False)
    # Obtaining the member 'matvec' of a type (line 600)
    matvec_410818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 34), A_410817, 'matvec')
    # Calling matvec(args, kwargs) (line 600)
    matvec_call_result_410824 = invoke(stypy.reporting.localization.Localization(__file__, 600, 34), matvec_410818, *[subscript_call_result_410822], **kwargs_410823)
    
    # Applying the binary operator '*' (line 600)
    result_mul_410825 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 28), '*', sclr1_410816, matvec_call_result_410824)
    
    # Applying the binary operator '+=' (line 600)
    result_iadd_410826 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 12), '+=', subscript_call_result_410815, result_mul_410825)
    # Getting the type of 'work' (line 600)
    work_410827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'work')
    # Getting the type of 'slice2' (line 600)
    slice2_410828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 17), 'slice2')
    # Storing an element on a container (line 600)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 12), work_410827, (slice2_410828, result_iadd_410826))
    
    # SSA branch for the else part of an if statement (line 598)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 601)
    ijob_410829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 14), 'ijob')
    int_410830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 22), 'int')
    # Applying the binary operator '==' (line 601)
    result_eq_410831 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 14), '==', ijob_410829, int_410830)
    
    # Testing the type of an if condition (line 601)
    if_condition_410832 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 601, 13), result_eq_410831)
    # Assigning a type to the variable 'if_condition_410832' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 13), 'if_condition_410832', if_condition_410832)
    # SSA begins for if statement (line 601)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 602)
    work_410833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 602)
    slice2_410834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 17), 'slice2')
    # Getting the type of 'work' (line 602)
    work_410835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 602)
    getitem___410836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 12), work_410835, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 602)
    subscript_call_result_410837 = invoke(stypy.reporting.localization.Localization(__file__, 602, 12), getitem___410836, slice2_410834)
    
    # Getting the type of 'sclr2' (line 602)
    sclr2_410838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 28), 'sclr2')
    # Applying the binary operator '*=' (line 602)
    result_imul_410839 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 12), '*=', subscript_call_result_410837, sclr2_410838)
    # Getting the type of 'work' (line 602)
    work_410840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 12), 'work')
    # Getting the type of 'slice2' (line 602)
    slice2_410841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 17), 'slice2')
    # Storing an element on a container (line 602)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 602, 12), work_410840, (slice2_410841, result_imul_410839))
    
    
    # Getting the type of 'work' (line 603)
    work_410842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 603)
    slice2_410843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 17), 'slice2')
    # Getting the type of 'work' (line 603)
    work_410844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 603)
    getitem___410845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 12), work_410844, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 603)
    subscript_call_result_410846 = invoke(stypy.reporting.localization.Localization(__file__, 603, 12), getitem___410845, slice2_410843)
    
    # Getting the type of 'sclr1' (line 603)
    sclr1_410847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 28), 'sclr1')
    
    # Call to rmatvec(...): (line 603)
    # Processing the call arguments (line 603)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 603)
    slice1_410850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 49), 'slice1', False)
    # Getting the type of 'work' (line 603)
    work_410851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 44), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 603)
    getitem___410852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 44), work_410851, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 603)
    subscript_call_result_410853 = invoke(stypy.reporting.localization.Localization(__file__, 603, 44), getitem___410852, slice1_410850)
    
    # Processing the call keyword arguments (line 603)
    kwargs_410854 = {}
    # Getting the type of 'A' (line 603)
    A_410848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 34), 'A', False)
    # Obtaining the member 'rmatvec' of a type (line 603)
    rmatvec_410849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 34), A_410848, 'rmatvec')
    # Calling rmatvec(args, kwargs) (line 603)
    rmatvec_call_result_410855 = invoke(stypy.reporting.localization.Localization(__file__, 603, 34), rmatvec_410849, *[subscript_call_result_410853], **kwargs_410854)
    
    # Applying the binary operator '*' (line 603)
    result_mul_410856 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 28), '*', sclr1_410847, rmatvec_call_result_410855)
    
    # Applying the binary operator '+=' (line 603)
    result_iadd_410857 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 12), '+=', subscript_call_result_410846, result_mul_410856)
    # Getting the type of 'work' (line 603)
    work_410858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), 'work')
    # Getting the type of 'slice2' (line 603)
    slice2_410859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 17), 'slice2')
    # Storing an element on a container (line 603)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 12), work_410858, (slice2_410859, result_iadd_410857))
    
    # SSA branch for the else part of an if statement (line 601)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 604)
    ijob_410860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 14), 'ijob')
    int_410861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 22), 'int')
    # Applying the binary operator '==' (line 604)
    result_eq_410862 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 14), '==', ijob_410860, int_410861)
    
    # Testing the type of an if condition (line 604)
    if_condition_410863 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 604, 13), result_eq_410862)
    # Assigning a type to the variable 'if_condition_410863' (line 604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 13), 'if_condition_410863', if_condition_410863)
    # SSA begins for if statement (line 604)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 605):
    
    # Assigning a Call to a Subscript (line 605):
    
    # Call to matvec(...): (line 605)
    # Processing the call arguments (line 605)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 605)
    slice2_410866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 42), 'slice2', False)
    # Getting the type of 'work' (line 605)
    work_410867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 37), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 605)
    getitem___410868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 37), work_410867, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 605)
    subscript_call_result_410869 = invoke(stypy.reporting.localization.Localization(__file__, 605, 37), getitem___410868, slice2_410866)
    
    # Processing the call keyword arguments (line 605)
    kwargs_410870 = {}
    # Getting the type of 'M1' (line 605)
    M1_410864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 27), 'M1', False)
    # Obtaining the member 'matvec' of a type (line 605)
    matvec_410865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 27), M1_410864, 'matvec')
    # Calling matvec(args, kwargs) (line 605)
    matvec_call_result_410871 = invoke(stypy.reporting.localization.Localization(__file__, 605, 27), matvec_410865, *[subscript_call_result_410869], **kwargs_410870)
    
    # Getting the type of 'work' (line 605)
    work_410872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'work')
    # Getting the type of 'slice1' (line 605)
    slice1_410873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 17), 'slice1')
    # Storing an element on a container (line 605)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 12), work_410872, (slice1_410873, matvec_call_result_410871))
    # SSA branch for the else part of an if statement (line 604)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 606)
    ijob_410874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 14), 'ijob')
    int_410875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 22), 'int')
    # Applying the binary operator '==' (line 606)
    result_eq_410876 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 14), '==', ijob_410874, int_410875)
    
    # Testing the type of an if condition (line 606)
    if_condition_410877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 606, 13), result_eq_410876)
    # Assigning a type to the variable 'if_condition_410877' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 13), 'if_condition_410877', if_condition_410877)
    # SSA begins for if statement (line 606)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 607):
    
    # Assigning a Call to a Subscript (line 607):
    
    # Call to matvec(...): (line 607)
    # Processing the call arguments (line 607)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 607)
    slice2_410880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 42), 'slice2', False)
    # Getting the type of 'work' (line 607)
    work_410881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 37), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 607)
    getitem___410882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 37), work_410881, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 607)
    subscript_call_result_410883 = invoke(stypy.reporting.localization.Localization(__file__, 607, 37), getitem___410882, slice2_410880)
    
    # Processing the call keyword arguments (line 607)
    kwargs_410884 = {}
    # Getting the type of 'M2' (line 607)
    M2_410878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 27), 'M2', False)
    # Obtaining the member 'matvec' of a type (line 607)
    matvec_410879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 27), M2_410878, 'matvec')
    # Calling matvec(args, kwargs) (line 607)
    matvec_call_result_410885 = invoke(stypy.reporting.localization.Localization(__file__, 607, 27), matvec_410879, *[subscript_call_result_410883], **kwargs_410884)
    
    # Getting the type of 'work' (line 607)
    work_410886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'work')
    # Getting the type of 'slice1' (line 607)
    slice1_410887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 17), 'slice1')
    # Storing an element on a container (line 607)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 12), work_410886, (slice1_410887, matvec_call_result_410885))
    # SSA branch for the else part of an if statement (line 606)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 608)
    ijob_410888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 14), 'ijob')
    int_410889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 22), 'int')
    # Applying the binary operator '==' (line 608)
    result_eq_410890 = python_operator(stypy.reporting.localization.Localization(__file__, 608, 14), '==', ijob_410888, int_410889)
    
    # Testing the type of an if condition (line 608)
    if_condition_410891 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 608, 13), result_eq_410890)
    # Assigning a type to the variable 'if_condition_410891' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 13), 'if_condition_410891', if_condition_410891)
    # SSA begins for if statement (line 608)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 609):
    
    # Assigning a Call to a Subscript (line 609):
    
    # Call to rmatvec(...): (line 609)
    # Processing the call arguments (line 609)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 609)
    slice2_410894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 43), 'slice2', False)
    # Getting the type of 'work' (line 609)
    work_410895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 38), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 609)
    getitem___410896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 38), work_410895, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 609)
    subscript_call_result_410897 = invoke(stypy.reporting.localization.Localization(__file__, 609, 38), getitem___410896, slice2_410894)
    
    # Processing the call keyword arguments (line 609)
    kwargs_410898 = {}
    # Getting the type of 'M1' (line 609)
    M1_410892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 27), 'M1', False)
    # Obtaining the member 'rmatvec' of a type (line 609)
    rmatvec_410893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 27), M1_410892, 'rmatvec')
    # Calling rmatvec(args, kwargs) (line 609)
    rmatvec_call_result_410899 = invoke(stypy.reporting.localization.Localization(__file__, 609, 27), rmatvec_410893, *[subscript_call_result_410897], **kwargs_410898)
    
    # Getting the type of 'work' (line 609)
    work_410900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'work')
    # Getting the type of 'slice1' (line 609)
    slice1_410901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 17), 'slice1')
    # Storing an element on a container (line 609)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 609, 12), work_410900, (slice1_410901, rmatvec_call_result_410899))
    # SSA branch for the else part of an if statement (line 608)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 610)
    ijob_410902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 14), 'ijob')
    int_410903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 22), 'int')
    # Applying the binary operator '==' (line 610)
    result_eq_410904 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 14), '==', ijob_410902, int_410903)
    
    # Testing the type of an if condition (line 610)
    if_condition_410905 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 610, 13), result_eq_410904)
    # Assigning a type to the variable 'if_condition_410905' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 13), 'if_condition_410905', if_condition_410905)
    # SSA begins for if statement (line 610)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 611):
    
    # Assigning a Call to a Subscript (line 611):
    
    # Call to rmatvec(...): (line 611)
    # Processing the call arguments (line 611)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 611)
    slice2_410908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 43), 'slice2', False)
    # Getting the type of 'work' (line 611)
    work_410909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 38), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 611)
    getitem___410910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 38), work_410909, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 611)
    subscript_call_result_410911 = invoke(stypy.reporting.localization.Localization(__file__, 611, 38), getitem___410910, slice2_410908)
    
    # Processing the call keyword arguments (line 611)
    kwargs_410912 = {}
    # Getting the type of 'M2' (line 611)
    M2_410906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 27), 'M2', False)
    # Obtaining the member 'rmatvec' of a type (line 611)
    rmatvec_410907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 27), M2_410906, 'rmatvec')
    # Calling rmatvec(args, kwargs) (line 611)
    rmatvec_call_result_410913 = invoke(stypy.reporting.localization.Localization(__file__, 611, 27), rmatvec_410907, *[subscript_call_result_410911], **kwargs_410912)
    
    # Getting the type of 'work' (line 611)
    work_410914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'work')
    # Getting the type of 'slice1' (line 611)
    slice1_410915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 17), 'slice1')
    # Storing an element on a container (line 611)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 12), work_410914, (slice1_410915, rmatvec_call_result_410913))
    # SSA branch for the else part of an if statement (line 610)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 612)
    ijob_410916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 14), 'ijob')
    int_410917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 22), 'int')
    # Applying the binary operator '==' (line 612)
    result_eq_410918 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 14), '==', ijob_410916, int_410917)
    
    # Testing the type of an if condition (line 612)
    if_condition_410919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 612, 13), result_eq_410918)
    # Assigning a type to the variable 'if_condition_410919' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 13), 'if_condition_410919', if_condition_410919)
    # SSA begins for if statement (line 612)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'work' (line 613)
    work_410920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 613)
    slice2_410921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 17), 'slice2')
    # Getting the type of 'work' (line 613)
    work_410922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 613)
    getitem___410923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 12), work_410922, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 613)
    subscript_call_result_410924 = invoke(stypy.reporting.localization.Localization(__file__, 613, 12), getitem___410923, slice2_410921)
    
    # Getting the type of 'sclr2' (line 613)
    sclr2_410925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 28), 'sclr2')
    # Applying the binary operator '*=' (line 613)
    result_imul_410926 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 12), '*=', subscript_call_result_410924, sclr2_410925)
    # Getting the type of 'work' (line 613)
    work_410927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'work')
    # Getting the type of 'slice2' (line 613)
    slice2_410928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 17), 'slice2')
    # Storing an element on a container (line 613)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 12), work_410927, (slice2_410928, result_imul_410926))
    
    
    # Getting the type of 'work' (line 614)
    work_410929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'work')
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 614)
    slice2_410930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 17), 'slice2')
    # Getting the type of 'work' (line 614)
    work_410931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'work')
    # Obtaining the member '__getitem__' of a type (line 614)
    getitem___410932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 12), work_410931, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 614)
    subscript_call_result_410933 = invoke(stypy.reporting.localization.Localization(__file__, 614, 12), getitem___410932, slice2_410930)
    
    # Getting the type of 'sclr1' (line 614)
    sclr1_410934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 28), 'sclr1')
    
    # Call to matvec(...): (line 614)
    # Processing the call arguments (line 614)
    # Getting the type of 'x' (line 614)
    x_410937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 43), 'x', False)
    # Processing the call keyword arguments (line 614)
    kwargs_410938 = {}
    # Getting the type of 'A' (line 614)
    A_410935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 34), 'A', False)
    # Obtaining the member 'matvec' of a type (line 614)
    matvec_410936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 34), A_410935, 'matvec')
    # Calling matvec(args, kwargs) (line 614)
    matvec_call_result_410939 = invoke(stypy.reporting.localization.Localization(__file__, 614, 34), matvec_410936, *[x_410937], **kwargs_410938)
    
    # Applying the binary operator '*' (line 614)
    result_mul_410940 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 28), '*', sclr1_410934, matvec_call_result_410939)
    
    # Applying the binary operator '+=' (line 614)
    result_iadd_410941 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 12), '+=', subscript_call_result_410933, result_mul_410940)
    # Getting the type of 'work' (line 614)
    work_410942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'work')
    # Getting the type of 'slice2' (line 614)
    slice2_410943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 17), 'slice2')
    # Storing an element on a container (line 614)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 12), work_410942, (slice2_410943, result_iadd_410941))
    
    # SSA branch for the else part of an if statement (line 612)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ijob' (line 615)
    ijob_410944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 14), 'ijob')
    int_410945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 22), 'int')
    # Applying the binary operator '==' (line 615)
    result_eq_410946 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 14), '==', ijob_410944, int_410945)
    
    # Testing the type of an if condition (line 615)
    if_condition_410947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 615, 13), result_eq_410946)
    # Assigning a type to the variable 'if_condition_410947' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 13), 'if_condition_410947', if_condition_410947)
    # SSA begins for if statement (line 615)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ftflag' (line 616)
    ftflag_410948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 15), 'ftflag')
    # Testing the type of an if condition (line 616)
    if_condition_410949 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 616, 12), ftflag_410948)
    # Assigning a type to the variable 'if_condition_410949' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'if_condition_410949', if_condition_410949)
    # SSA begins for if statement (line 616)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 617):
    
    # Assigning a Num to a Name (line 617):
    int_410950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 23), 'int')
    # Assigning a type to the variable 'info' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 16), 'info', int_410950)
    
    # Assigning a Name to a Name (line 618):
    
    # Assigning a Name to a Name (line 618):
    # Getting the type of 'False' (line 618)
    False_410951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 25), 'False')
    # Assigning a type to the variable 'ftflag' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 16), 'ftflag', False_410951)
    # SSA join for if statement (line 616)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 619):
    
    # Assigning a Subscript to a Name (line 619):
    
    # Obtaining the type of the subscript
    int_410952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 12), 'int')
    
    # Call to stoptest(...): (line 619)
    # Processing the call arguments (line 619)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 619)
    slice1_410954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 47), 'slice1', False)
    # Getting the type of 'work' (line 619)
    work_410955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 619)
    getitem___410956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 42), work_410955, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 619)
    subscript_call_result_410957 = invoke(stypy.reporting.localization.Localization(__file__, 619, 42), getitem___410956, slice1_410954)
    
    # Getting the type of 'b' (line 619)
    b_410958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 619)
    bnrm2_410959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 619)
    tol_410960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 66), 'tol', False)
    # Getting the type of 'info' (line 619)
    info_410961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 71), 'info', False)
    # Processing the call keyword arguments (line 619)
    kwargs_410962 = {}
    # Getting the type of 'stoptest' (line 619)
    stoptest_410953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 619)
    stoptest_call_result_410963 = invoke(stypy.reporting.localization.Localization(__file__, 619, 33), stoptest_410953, *[subscript_call_result_410957, b_410958, bnrm2_410959, tol_410960, info_410961], **kwargs_410962)
    
    # Obtaining the member '__getitem__' of a type (line 619)
    getitem___410964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 12), stoptest_call_result_410963, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 619)
    subscript_call_result_410965 = invoke(stypy.reporting.localization.Localization(__file__, 619, 12), getitem___410964, int_410952)
    
    # Assigning a type to the variable 'tuple_var_assignment_407990' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'tuple_var_assignment_407990', subscript_call_result_410965)
    
    # Assigning a Subscript to a Name (line 619):
    
    # Obtaining the type of the subscript
    int_410966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 12), 'int')
    
    # Call to stoptest(...): (line 619)
    # Processing the call arguments (line 619)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 619)
    slice1_410968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 47), 'slice1', False)
    # Getting the type of 'work' (line 619)
    work_410969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 619)
    getitem___410970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 42), work_410969, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 619)
    subscript_call_result_410971 = invoke(stypy.reporting.localization.Localization(__file__, 619, 42), getitem___410970, slice1_410968)
    
    # Getting the type of 'b' (line 619)
    b_410972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 619)
    bnrm2_410973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 619)
    tol_410974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 66), 'tol', False)
    # Getting the type of 'info' (line 619)
    info_410975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 71), 'info', False)
    # Processing the call keyword arguments (line 619)
    kwargs_410976 = {}
    # Getting the type of 'stoptest' (line 619)
    stoptest_410967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 619)
    stoptest_call_result_410977 = invoke(stypy.reporting.localization.Localization(__file__, 619, 33), stoptest_410967, *[subscript_call_result_410971, b_410972, bnrm2_410973, tol_410974, info_410975], **kwargs_410976)
    
    # Obtaining the member '__getitem__' of a type (line 619)
    getitem___410978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 12), stoptest_call_result_410977, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 619)
    subscript_call_result_410979 = invoke(stypy.reporting.localization.Localization(__file__, 619, 12), getitem___410978, int_410966)
    
    # Assigning a type to the variable 'tuple_var_assignment_407991' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'tuple_var_assignment_407991', subscript_call_result_410979)
    
    # Assigning a Subscript to a Name (line 619):
    
    # Obtaining the type of the subscript
    int_410980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 12), 'int')
    
    # Call to stoptest(...): (line 619)
    # Processing the call arguments (line 619)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 619)
    slice1_410982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 47), 'slice1', False)
    # Getting the type of 'work' (line 619)
    work_410983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 42), 'work', False)
    # Obtaining the member '__getitem__' of a type (line 619)
    getitem___410984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 42), work_410983, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 619)
    subscript_call_result_410985 = invoke(stypy.reporting.localization.Localization(__file__, 619, 42), getitem___410984, slice1_410982)
    
    # Getting the type of 'b' (line 619)
    b_410986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 56), 'b', False)
    # Getting the type of 'bnrm2' (line 619)
    bnrm2_410987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 59), 'bnrm2', False)
    # Getting the type of 'tol' (line 619)
    tol_410988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 66), 'tol', False)
    # Getting the type of 'info' (line 619)
    info_410989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 71), 'info', False)
    # Processing the call keyword arguments (line 619)
    kwargs_410990 = {}
    # Getting the type of 'stoptest' (line 619)
    stoptest_410981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 33), 'stoptest', False)
    # Calling stoptest(args, kwargs) (line 619)
    stoptest_call_result_410991 = invoke(stypy.reporting.localization.Localization(__file__, 619, 33), stoptest_410981, *[subscript_call_result_410985, b_410986, bnrm2_410987, tol_410988, info_410989], **kwargs_410990)
    
    # Obtaining the member '__getitem__' of a type (line 619)
    getitem___410992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 12), stoptest_call_result_410991, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 619)
    subscript_call_result_410993 = invoke(stypy.reporting.localization.Localization(__file__, 619, 12), getitem___410992, int_410980)
    
    # Assigning a type to the variable 'tuple_var_assignment_407992' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'tuple_var_assignment_407992', subscript_call_result_410993)
    
    # Assigning a Name to a Name (line 619):
    # Getting the type of 'tuple_var_assignment_407990' (line 619)
    tuple_var_assignment_407990_410994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'tuple_var_assignment_407990')
    # Assigning a type to the variable 'bnrm2' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'bnrm2', tuple_var_assignment_407990_410994)
    
    # Assigning a Name to a Name (line 619):
    # Getting the type of 'tuple_var_assignment_407991' (line 619)
    tuple_var_assignment_407991_410995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'tuple_var_assignment_407991')
    # Assigning a type to the variable 'resid' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 19), 'resid', tuple_var_assignment_407991_410995)
    
    # Assigning a Name to a Name (line 619):
    # Getting the type of 'tuple_var_assignment_407992' (line 619)
    tuple_var_assignment_407992_410996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'tuple_var_assignment_407992')
    # Assigning a type to the variable 'info' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 26), 'info', tuple_var_assignment_407992_410996)
    # SSA join for if statement (line 615)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 612)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 610)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 608)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 606)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 604)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 601)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 598)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 594)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 620):
    
    # Assigning a Num to a Name (line 620):
    int_410997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 15), 'int')
    # Assigning a type to the variable 'ijob' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'ijob', int_410997)
    # SSA join for while statement (line 586)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'info' (line 622)
    info_410998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 7), 'info')
    int_410999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 14), 'int')
    # Applying the binary operator '>' (line 622)
    result_gt_411000 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 7), '>', info_410998, int_410999)
    
    
    # Getting the type of 'iter_' (line 622)
    iter__411001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 20), 'iter_')
    # Getting the type of 'maxiter' (line 622)
    maxiter_411002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 29), 'maxiter')
    # Applying the binary operator '==' (line 622)
    result_eq_411003 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 20), '==', iter__411001, maxiter_411002)
    
    # Applying the binary operator 'and' (line 622)
    result_and_keyword_411004 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 7), 'and', result_gt_411000, result_eq_411003)
    
    # Getting the type of 'resid' (line 622)
    resid_411005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 41), 'resid')
    # Getting the type of 'tol' (line 622)
    tol_411006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 49), 'tol')
    # Applying the binary operator '>' (line 622)
    result_gt_411007 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 41), '>', resid_411005, tol_411006)
    
    # Applying the binary operator 'and' (line 622)
    result_and_keyword_411008 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 7), 'and', result_and_keyword_411004, result_gt_411007)
    
    # Testing the type of an if condition (line 622)
    if_condition_411009 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 622, 4), result_and_keyword_411008)
    # Assigning a type to the variable 'if_condition_411009' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'if_condition_411009', if_condition_411009)
    # SSA begins for if statement (line 622)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 624):
    
    # Assigning a Name to a Name (line 624):
    # Getting the type of 'iter_' (line 624)
    iter__411010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 15), 'iter_')
    # Assigning a type to the variable 'info' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'info', iter__411010)
    # SSA join for if statement (line 622)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 626)
    tuple_411011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 626)
    # Adding element type (line 626)
    
    # Call to postprocess(...): (line 626)
    # Processing the call arguments (line 626)
    # Getting the type of 'x' (line 626)
    x_411013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 23), 'x', False)
    # Processing the call keyword arguments (line 626)
    kwargs_411014 = {}
    # Getting the type of 'postprocess' (line 626)
    postprocess_411012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 11), 'postprocess', False)
    # Calling postprocess(args, kwargs) (line 626)
    postprocess_call_result_411015 = invoke(stypy.reporting.localization.Localization(__file__, 626, 11), postprocess_411012, *[x_411013], **kwargs_411014)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 626, 11), tuple_411011, postprocess_call_result_411015)
    # Adding element type (line 626)
    # Getting the type of 'info' (line 626)
    info_411016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 27), 'info')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 626, 11), tuple_411011, info_411016)
    
    # Assigning a type to the variable 'stypy_return_type' (line 626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 4), 'stypy_return_type', tuple_411011)
    
    # ################# End of 'qmr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'qmr' in the type store
    # Getting the type of 'stypy_return_type' (line 485)
    stypy_return_type_411017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_411017)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'qmr'
    return stypy_return_type_411017

# Assigning a type to the variable 'qmr' (line 485)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 0), 'qmr', qmr)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
