
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Pure SciPy implementation of Locally Optimal Block Preconditioned Conjugate
3: Gradient Method (LOBPCG), see
4: https://bitbucket.org/joseroman/blopex
5: 
6: License: BSD
7: 
8: Authors: Robert Cimrman, Andrew Knyazev
9: 
10: Examples in tests directory contributed by Nils Wagner.
11: '''
12: 
13: from __future__ import division, print_function, absolute_import
14: 
15: import sys
16: 
17: import numpy as np
18: from numpy.testing import assert_allclose
19: from scipy._lib.six import xrange
20: from scipy.linalg import inv, eigh, cho_factor, cho_solve, cholesky
21: from scipy.sparse.linalg import aslinearoperator, LinearOperator
22: 
23: __all__ = ['lobpcg']
24: 
25: 
26: def pause():
27:     # Used only when verbosity level > 10.
28:     input()
29: 
30: 
31: def save(ar, fileName):
32:     # Used only when verbosity level > 10.
33:     from numpy import savetxt
34:     savetxt(fileName, ar, precision=8)
35: 
36: 
37: def _assert_symmetric(M, rtol=1e-5, atol=1e-8):
38:     assert_allclose(M.T, M, rtol=rtol, atol=atol)
39: 
40: 
41: ##
42: # 21.05.2007, c
43: 
44: 
45: def as2d(ar):
46:     '''
47:     If the input array is 2D return it, if it is 1D, append a dimension,
48:     making it a column vector.
49:     '''
50:     if ar.ndim == 2:
51:         return ar
52:     else:  # Assume 1!
53:         aux = np.array(ar, copy=False)
54:         aux.shape = (ar.shape[0], 1)
55:         return aux
56: 
57: 
58: def _makeOperator(operatorInput, expectedShape):
59:     '''Takes a dense numpy array or a sparse matrix or
60:     a function and makes an operator performing matrix * blockvector
61:     products.
62: 
63:     Examples
64:     --------
65:     >>> A = _makeOperator( arrayA, (n, n) )
66:     >>> vectorB = A( vectorX )
67: 
68:     '''
69:     if operatorInput is None:
70:         def ident(x):
71:             return x
72:         operator = LinearOperator(expectedShape, ident, matmat=ident)
73:     else:
74:         operator = aslinearoperator(operatorInput)
75: 
76:     if operator.shape != expectedShape:
77:         raise ValueError('operator has invalid shape')
78: 
79:     return operator
80: 
81: 
82: def _applyConstraints(blockVectorV, factYBY, blockVectorBY, blockVectorY):
83:     '''Changes blockVectorV in place.'''
84:     gramYBV = np.dot(blockVectorBY.T, blockVectorV)
85:     tmp = cho_solve(factYBY, gramYBV)
86:     blockVectorV -= np.dot(blockVectorY, tmp)
87: 
88: 
89: def _b_orthonormalize(B, blockVectorV, blockVectorBV=None, retInvR=False):
90:     if blockVectorBV is None:
91:         if B is not None:
92:             blockVectorBV = B(blockVectorV)
93:         else:
94:             blockVectorBV = blockVectorV  # Shared data!!!
95:     gramVBV = np.dot(blockVectorV.T, blockVectorBV)
96:     gramVBV = cholesky(gramVBV)
97:     gramVBV = inv(gramVBV, overwrite_a=True)
98:     # gramVBV is now R^{-1}.
99:     blockVectorV = np.dot(blockVectorV, gramVBV)
100:     if B is not None:
101:         blockVectorBV = np.dot(blockVectorBV, gramVBV)
102: 
103:     if retInvR:
104:         return blockVectorV, blockVectorBV, gramVBV
105:     else:
106:         return blockVectorV, blockVectorBV
107: 
108: 
109: def lobpcg(A, X,
110:             B=None, M=None, Y=None,
111:             tol=None, maxiter=20,
112:             largest=True, verbosityLevel=0,
113:             retLambdaHistory=False, retResidualNormsHistory=False):
114:     '''Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)
115: 
116:     LOBPCG is a preconditioned eigensolver for large symmetric positive
117:     definite (SPD) generalized eigenproblems.
118: 
119:     Parameters
120:     ----------
121:     A : {sparse matrix, dense matrix, LinearOperator}
122:         The symmetric linear operator of the problem, usually a
123:         sparse matrix.  Often called the "stiffness matrix".
124:     X : array_like
125:         Initial approximation to the k eigenvectors. If A has
126:         shape=(n,n) then X should have shape shape=(n,k).
127:     B : {dense matrix, sparse matrix, LinearOperator}, optional
128:         the right hand side operator in a generalized eigenproblem.
129:         by default, B = Identity
130:         often called the "mass matrix"
131:     M : {dense matrix, sparse matrix, LinearOperator}, optional
132:         preconditioner to A; by default M = Identity
133:         M should approximate the inverse of A
134:     Y : array_like, optional
135:         n-by-sizeY matrix of constraints, sizeY < n
136:         The iterations will be performed in the B-orthogonal complement
137:         of the column-space of Y. Y must be full rank.
138: 
139:     Returns
140:     -------
141:     w : array
142:         Array of k eigenvalues
143:     v : array
144:         An array of k eigenvectors.  V has the same shape as X.
145: 
146:     Other Parameters
147:     ----------------
148:     tol : scalar, optional
149:         Solver tolerance (stopping criterion)
150:         by default: tol=n*sqrt(eps)
151:     maxiter : integer, optional
152:         maximum number of iterations
153:         by default: maxiter=min(n,20)
154:     largest : bool, optional
155:         when True, solve for the largest eigenvalues, otherwise the smallest
156:     verbosityLevel : integer, optional
157:         controls solver output.  default: verbosityLevel = 0.
158:     retLambdaHistory : boolean, optional
159:         whether to return eigenvalue history
160:     retResidualNormsHistory : boolean, optional
161:         whether to return history of residual norms
162: 
163:     Examples
164:     --------
165: 
166:     Solve A x = lambda B x with constraints and preconditioning.
167: 
168:     >>> from scipy.sparse import spdiags, issparse
169:     >>> from scipy.sparse.linalg import lobpcg, LinearOperator
170:     >>> n = 100
171:     >>> vals = [np.arange(n, dtype=np.float64) + 1]
172:     >>> A = spdiags(vals, 0, n, n)
173:     >>> A.toarray()
174:     array([[   1.,    0.,    0., ...,    0.,    0.,    0.],
175:            [   0.,    2.,    0., ...,    0.,    0.,    0.],
176:            [   0.,    0.,    3., ...,    0.,    0.,    0.],
177:            ...,
178:            [   0.,    0.,    0., ...,   98.,    0.,    0.],
179:            [   0.,    0.,    0., ...,    0.,   99.,    0.],
180:            [   0.,    0.,    0., ...,    0.,    0.,  100.]])
181: 
182:     Constraints.
183: 
184:     >>> Y = np.eye(n, 3)
185: 
186:     Initial guess for eigenvectors, should have linearly independent
187:     columns. Column dimension = number of requested eigenvalues.
188: 
189:     >>> X = np.random.rand(n, 3)
190: 
191:     Preconditioner -- inverse of A (as an abstract linear operator).
192: 
193:     >>> invA = spdiags([1./vals[0]], 0, n, n)
194:     >>> def precond( x ):
195:     ...     return invA  * x
196:     >>> M = LinearOperator(matvec=precond, shape=(n, n), dtype=float)
197: 
198:     Here, ``invA`` could of course have been used directly as a preconditioner.
199:     Let us then solve the problem:
200: 
201:     >>> eigs, vecs = lobpcg(A, X, Y=Y, M=M, tol=1e-4, maxiter=40, largest=False)
202:     >>> eigs
203:     array([ 4.,  5.,  6.])
204: 
205:     Note that the vectors passed in Y are the eigenvectors of the 3 smallest
206:     eigenvalues. The results returned are orthogonal to those.
207: 
208:     Notes
209:     -----
210:     If both retLambdaHistory and retResidualNormsHistory are True,
211:     the return tuple has the following format
212:     (lambda, V, lambda history, residual norms history).
213: 
214:     In the following ``n`` denotes the matrix size and ``m`` the number
215:     of required eigenvalues (smallest or largest).
216: 
217:     The LOBPCG code internally solves eigenproblems of the size 3``m`` on every
218:     iteration by calling the "standard" dense eigensolver, so if ``m`` is not
219:     small enough compared to ``n``, it does not make sense to call the LOBPCG
220:     code, but rather one should use the "standard" eigensolver,
221:     e.g. numpy or scipy function in this case.
222:     If one calls the LOBPCG algorithm for 5``m``>``n``,
223:     it will most likely break internally, so the code tries to call the standard
224:     function instead.
225: 
226:     It is not that n should be large for the LOBPCG to work, but rather the
227:     ratio ``n``/``m`` should be large. It you call the LOBPCG code with ``m``=1
228:     and ``n``=10, it should work, though ``n`` is small. The method is intended
229:     for extremely large ``n``/``m``, see e.g., reference [28] in
230:     http://arxiv.org/abs/0705.2626
231: 
232:     The convergence speed depends basically on two factors:
233: 
234:     1.  How well relatively separated the seeking eigenvalues are
235:         from the rest of the eigenvalues.
236:         One can try to vary ``m`` to make this better.
237: 
238:     2.  How well conditioned the problem is. This can be changed by using proper
239:         preconditioning. For example, a rod vibration test problem (under tests
240:         directory) is ill-conditioned for large ``n``, so convergence will be
241:         slow, unless efficient preconditioning is used.
242:         For this specific problem, a good simple preconditioner function would
243:         be a linear solve for A, which is easy to code since A is tridiagonal.
244: 
245:     *Acknowledgements*
246: 
247:     lobpcg.py code was written by Robert Cimrman.
248:     Many thanks belong to Andrew Knyazev, the author of the algorithm,
249:     for lots of advice and support.
250: 
251:     References
252:     ----------
253:     .. [1] A. V. Knyazev (2001),
254:            Toward the Optimal Preconditioned Eigensolver: Locally Optimal
255:            Block Preconditioned Conjugate Gradient Method.
256:            SIAM Journal on Scientific Computing 23, no. 2,
257:            pp. 517-541. http://dx.doi.org/10.1137/S1064827500366124
258: 
259:     .. [2] A. V. Knyazev, I. Lashuk, M. E. Argentati, and E. Ovchinnikov (2007),
260:            Block Locally Optimal Preconditioned Eigenvalue Xolvers (BLOPEX)
261:            in hypre and PETSc.  http://arxiv.org/abs/0705.2626
262: 
263:     .. [3] A. V. Knyazev's C and MATLAB implementations:
264:            https://bitbucket.org/joseroman/blopex
265: 
266:     '''
267:     blockVectorX = X
268:     blockVectorY = Y
269:     residualTolerance = tol
270:     maxIterations = maxiter
271: 
272:     if blockVectorY is not None:
273:         sizeY = blockVectorY.shape[1]
274:     else:
275:         sizeY = 0
276: 
277:     # Block size.
278:     if len(blockVectorX.shape) != 2:
279:         raise ValueError('expected rank-2 array for argument X')
280: 
281:     n, sizeX = blockVectorX.shape
282:     if sizeX > n:
283:         raise ValueError('X column dimension exceeds the row dimension')
284: 
285:     A = _makeOperator(A, (n,n))
286:     B = _makeOperator(B, (n,n))
287:     M = _makeOperator(M, (n,n))
288: 
289:     if (n - sizeY) < (5 * sizeX):
290:         # warn('The problem size is small compared to the block size.' \
291:         #        ' Using dense eigensolver instead of LOBPCG.')
292: 
293:         if blockVectorY is not None:
294:             raise NotImplementedError('The dense eigensolver '
295:                     'does not support constraints.')
296: 
297:         # Define the closed range of indices of eigenvalues to return.
298:         if largest:
299:             eigvals = (n - sizeX, n-1)
300:         else:
301:             eigvals = (0, sizeX-1)
302: 
303:         A_dense = A(np.eye(n))
304:         B_dense = None if B is None else B(np.eye(n))
305:         return eigh(A_dense, B_dense, eigvals=eigvals, check_finite=False)
306: 
307:     if residualTolerance is None:
308:         residualTolerance = np.sqrt(1e-15) * n
309: 
310:     maxIterations = min(n, maxIterations)
311: 
312:     if verbosityLevel:
313:         aux = "Solving "
314:         if B is None:
315:             aux += "standard"
316:         else:
317:             aux += "generalized"
318:         aux += " eigenvalue problem with"
319:         if M is None:
320:             aux += "out"
321:         aux += " preconditioning\n\n"
322:         aux += "matrix size %d\n" % n
323:         aux += "block size %d\n\n" % sizeX
324:         if blockVectorY is None:
325:             aux += "No constraints\n\n"
326:         else:
327:             if sizeY > 1:
328:                 aux += "%d constraints\n\n" % sizeY
329:             else:
330:                 aux += "%d constraint\n\n" % sizeY
331:         print(aux)
332: 
333:     ##
334:     # Apply constraints to X.
335:     if blockVectorY is not None:
336: 
337:         if B is not None:
338:             blockVectorBY = B(blockVectorY)
339:         else:
340:             blockVectorBY = blockVectorY
341: 
342:         # gramYBY is a dense array.
343:         gramYBY = np.dot(blockVectorY.T, blockVectorBY)
344:         try:
345:             # gramYBY is a Cholesky factor from now on...
346:             gramYBY = cho_factor(gramYBY)
347:         except:
348:             raise ValueError('cannot handle linearly dependent constraints')
349: 
350:         _applyConstraints(blockVectorX, gramYBY, blockVectorBY, blockVectorY)
351: 
352:     ##
353:     # B-orthonormalize X.
354:     blockVectorX, blockVectorBX = _b_orthonormalize(B, blockVectorX)
355: 
356:     ##
357:     # Compute the initial Ritz vectors: solve the eigenproblem.
358:     blockVectorAX = A(blockVectorX)
359:     gramXAX = np.dot(blockVectorX.T, blockVectorAX)
360: 
361:     _lambda, eigBlockVector = eigh(gramXAX, check_finite=False)
362:     ii = np.argsort(_lambda)[:sizeX]
363:     if largest:
364:         ii = ii[::-1]
365:     _lambda = _lambda[ii]
366: 
367:     eigBlockVector = np.asarray(eigBlockVector[:,ii])
368:     blockVectorX = np.dot(blockVectorX, eigBlockVector)
369:     blockVectorAX = np.dot(blockVectorAX, eigBlockVector)
370:     if B is not None:
371:         blockVectorBX = np.dot(blockVectorBX, eigBlockVector)
372: 
373:     ##
374:     # Active index set.
375:     activeMask = np.ones((sizeX,), dtype=bool)
376: 
377:     lambdaHistory = [_lambda]
378:     residualNormsHistory = []
379: 
380:     previousBlockSize = sizeX
381:     ident = np.eye(sizeX, dtype=A.dtype)
382:     ident0 = np.eye(sizeX, dtype=A.dtype)
383: 
384:     ##
385:     # Main iteration loop.
386: 
387:     blockVectorP = None  # set during iteration
388:     blockVectorAP = None
389:     blockVectorBP = None
390: 
391:     for iterationNumber in xrange(maxIterations):
392:         if verbosityLevel > 0:
393:             print('iteration %d' % iterationNumber)
394: 
395:         aux = blockVectorBX * _lambda[np.newaxis,:]
396:         blockVectorR = blockVectorAX - aux
397: 
398:         aux = np.sum(blockVectorR.conjugate() * blockVectorR, 0)
399:         residualNorms = np.sqrt(aux)
400: 
401:         residualNormsHistory.append(residualNorms)
402: 
403:         ii = np.where(residualNorms > residualTolerance, True, False)
404:         activeMask = activeMask & ii
405:         if verbosityLevel > 2:
406:             print(activeMask)
407: 
408:         currentBlockSize = activeMask.sum()
409:         if currentBlockSize != previousBlockSize:
410:             previousBlockSize = currentBlockSize
411:             ident = np.eye(currentBlockSize, dtype=A.dtype)
412: 
413:         if currentBlockSize == 0:
414:             break
415: 
416:         if verbosityLevel > 0:
417:             print('current block size:', currentBlockSize)
418:             print('eigenvalue:', _lambda)
419:             print('residual norms:', residualNorms)
420:         if verbosityLevel > 10:
421:             print(eigBlockVector)
422: 
423:         activeBlockVectorR = as2d(blockVectorR[:,activeMask])
424: 
425:         if iterationNumber > 0:
426:             activeBlockVectorP = as2d(blockVectorP[:,activeMask])
427:             activeBlockVectorAP = as2d(blockVectorAP[:,activeMask])
428:             activeBlockVectorBP = as2d(blockVectorBP[:,activeMask])
429: 
430:         if M is not None:
431:             # Apply preconditioner T to the active residuals.
432:             activeBlockVectorR = M(activeBlockVectorR)
433: 
434:         ##
435:         # Apply constraints to the preconditioned residuals.
436:         if blockVectorY is not None:
437:             _applyConstraints(activeBlockVectorR,
438:                               gramYBY, blockVectorBY, blockVectorY)
439: 
440:         ##
441:         # B-orthonormalize the preconditioned residuals.
442: 
443:         aux = _b_orthonormalize(B, activeBlockVectorR)
444:         activeBlockVectorR, activeBlockVectorBR = aux
445: 
446:         activeBlockVectorAR = A(activeBlockVectorR)
447: 
448:         if iterationNumber > 0:
449:             aux = _b_orthonormalize(B, activeBlockVectorP,
450:                                     activeBlockVectorBP, retInvR=True)
451:             activeBlockVectorP, activeBlockVectorBP, invR = aux
452:             activeBlockVectorAP = np.dot(activeBlockVectorAP, invR)
453: 
454:         ##
455:         # Perform the Rayleigh Ritz Procedure:
456:         # Compute symmetric Gram matrices:
457: 
458:         xaw = np.dot(blockVectorX.T, activeBlockVectorAR)
459:         waw = np.dot(activeBlockVectorR.T, activeBlockVectorAR)
460:         xbw = np.dot(blockVectorX.T, activeBlockVectorBR)
461: 
462:         if iterationNumber > 0:
463:             xap = np.dot(blockVectorX.T, activeBlockVectorAP)
464:             wap = np.dot(activeBlockVectorR.T, activeBlockVectorAP)
465:             pap = np.dot(activeBlockVectorP.T, activeBlockVectorAP)
466:             xbp = np.dot(blockVectorX.T, activeBlockVectorBP)
467:             wbp = np.dot(activeBlockVectorR.T, activeBlockVectorBP)
468: 
469:             gramA = np.bmat([[np.diag(_lambda), xaw, xap],
470:                               [xaw.T, waw, wap],
471:                               [xap.T, wap.T, pap]])
472: 
473:             gramB = np.bmat([[ident0, xbw, xbp],
474:                               [xbw.T, ident, wbp],
475:                               [xbp.T, wbp.T, ident]])
476:         else:
477:             gramA = np.bmat([[np.diag(_lambda), xaw],
478:                               [xaw.T, waw]])
479:             gramB = np.bmat([[ident0, xbw],
480:                               [xbw.T, ident]])
481: 
482:         _assert_symmetric(gramA)
483:         _assert_symmetric(gramB)
484: 
485:         if verbosityLevel > 10:
486:             save(gramA, 'gramA')
487:             save(gramB, 'gramB')
488: 
489:         # Solve the generalized eigenvalue problem.
490:         _lambda, eigBlockVector = eigh(gramA, gramB, check_finite=False)
491:         ii = np.argsort(_lambda)[:sizeX]
492:         if largest:
493:             ii = ii[::-1]
494:         if verbosityLevel > 10:
495:             print(ii)
496: 
497:         _lambda = _lambda[ii].astype(np.float64)
498:         eigBlockVector = np.asarray(eigBlockVector[:,ii].astype(np.float64))
499: 
500:         lambdaHistory.append(_lambda)
501: 
502:         if verbosityLevel > 10:
503:             print('lambda:', _lambda)
504: ##         # Normalize eigenvectors!
505: ##         aux = np.sum( eigBlockVector.conjugate() * eigBlockVector, 0 )
506: ##         eigVecNorms = np.sqrt( aux )
507: ##         eigBlockVector = eigBlockVector / eigVecNorms[np.newaxis,:]
508: #        eigBlockVector, aux = _b_orthonormalize( B, eigBlockVector )
509: 
510:         if verbosityLevel > 10:
511:             print(eigBlockVector)
512:             pause()
513: 
514:         ##
515:         # Compute Ritz vectors.
516:         if iterationNumber > 0:
517:             eigBlockVectorX = eigBlockVector[:sizeX]
518:             eigBlockVectorR = eigBlockVector[sizeX:sizeX+currentBlockSize]
519:             eigBlockVectorP = eigBlockVector[sizeX+currentBlockSize:]
520: 
521:             pp = np.dot(activeBlockVectorR, eigBlockVectorR)
522:             pp += np.dot(activeBlockVectorP, eigBlockVectorP)
523: 
524:             app = np.dot(activeBlockVectorAR, eigBlockVectorR)
525:             app += np.dot(activeBlockVectorAP, eigBlockVectorP)
526: 
527:             bpp = np.dot(activeBlockVectorBR, eigBlockVectorR)
528:             bpp += np.dot(activeBlockVectorBP, eigBlockVectorP)
529:         else:
530:             eigBlockVectorX = eigBlockVector[:sizeX]
531:             eigBlockVectorR = eigBlockVector[sizeX:]
532: 
533:             pp = np.dot(activeBlockVectorR, eigBlockVectorR)
534:             app = np.dot(activeBlockVectorAR, eigBlockVectorR)
535:             bpp = np.dot(activeBlockVectorBR, eigBlockVectorR)
536: 
537:         if verbosityLevel > 10:
538:             print(pp)
539:             print(app)
540:             print(bpp)
541:             pause()
542: 
543:         blockVectorX = np.dot(blockVectorX, eigBlockVectorX) + pp
544:         blockVectorAX = np.dot(blockVectorAX, eigBlockVectorX) + app
545:         blockVectorBX = np.dot(blockVectorBX, eigBlockVectorX) + bpp
546: 
547:         blockVectorP, blockVectorAP, blockVectorBP = pp, app, bpp
548: 
549:     aux = blockVectorBX * _lambda[np.newaxis,:]
550:     blockVectorR = blockVectorAX - aux
551: 
552:     aux = np.sum(blockVectorR.conjugate() * blockVectorR, 0)
553:     residualNorms = np.sqrt(aux)
554: 
555:     if verbosityLevel > 0:
556:         print('final eigenvalue:', _lambda)
557:         print('final residual norms:', residualNorms)
558: 
559:     if retLambdaHistory:
560:         if retResidualNormsHistory:
561:             return _lambda, blockVectorX, lambdaHistory, residualNormsHistory
562:         else:
563:             return _lambda, blockVectorX, lambdaHistory
564:     else:
565:         if retResidualNormsHistory:
566:             return _lambda, blockVectorX, residualNormsHistory
567:         else:
568:             return _lambda, blockVectorX
569: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_405649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\nPure SciPy implementation of Locally Optimal Block Preconditioned Conjugate\nGradient Method (LOBPCG), see\nhttps://bitbucket.org/joseroman/blopex\n\nLicense: BSD\n\nAuthors: Robert Cimrman, Andrew Knyazev\n\nExamples in tests directory contributed by Nils Wagner.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import sys' statement (line 15)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import numpy' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')
import_405650 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy')

if (type(import_405650) is not StypyTypeError):

    if (import_405650 != 'pyd_module'):
        __import__(import_405650)
        sys_modules_405651 = sys.modules[import_405650]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'np', sys_modules_405651.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy', import_405650)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy.testing import assert_allclose' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')
import_405652 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.testing')

if (type(import_405652) is not StypyTypeError):

    if (import_405652 != 'pyd_module'):
        __import__(import_405652)
        sys_modules_405653 = sys.modules[import_405652]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.testing', sys_modules_405653.module_type_store, module_type_store, ['assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_405653, sys_modules_405653.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.testing', None, module_type_store, ['assert_allclose'], [assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.testing', import_405652)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy._lib.six import xrange' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')
import_405654 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib.six')

if (type(import_405654) is not StypyTypeError):

    if (import_405654 != 'pyd_module'):
        __import__(import_405654)
        sys_modules_405655 = sys.modules[import_405654]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib.six', sys_modules_405655.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_405655, sys_modules_405655.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib.six', import_405654)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy.linalg import inv, eigh, cho_factor, cho_solve, cholesky' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')
import_405656 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg')

if (type(import_405656) is not StypyTypeError):

    if (import_405656 != 'pyd_module'):
        __import__(import_405656)
        sys_modules_405657 = sys.modules[import_405656]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg', sys_modules_405657.module_type_store, module_type_store, ['inv', 'eigh', 'cho_factor', 'cho_solve', 'cholesky'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_405657, sys_modules_405657.module_type_store, module_type_store)
    else:
        from scipy.linalg import inv, eigh, cho_factor, cho_solve, cholesky

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg', None, module_type_store, ['inv', 'eigh', 'cho_factor', 'cho_solve', 'cholesky'], [inv, eigh, cho_factor, cho_solve, cholesky])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg', import_405656)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from scipy.sparse.linalg import aslinearoperator, LinearOperator' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')
import_405658 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.sparse.linalg')

if (type(import_405658) is not StypyTypeError):

    if (import_405658 != 'pyd_module'):
        __import__(import_405658)
        sys_modules_405659 = sys.modules[import_405658]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.sparse.linalg', sys_modules_405659.module_type_store, module_type_store, ['aslinearoperator', 'LinearOperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_405659, sys_modules_405659.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import aslinearoperator, LinearOperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.sparse.linalg', None, module_type_store, ['aslinearoperator', 'LinearOperator'], [aslinearoperator, LinearOperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.sparse.linalg', import_405658)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')


# Assigning a List to a Name (line 23):

# Assigning a List to a Name (line 23):
__all__ = ['lobpcg']
module_type_store.set_exportable_members(['lobpcg'])

# Obtaining an instance of the builtin type 'list' (line 23)
list_405660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
str_405661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 11), 'str', 'lobpcg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 10), list_405660, str_405661)

# Assigning a type to the variable '__all__' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), '__all__', list_405660)

@norecursion
def pause(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pause'
    module_type_store = module_type_store.open_function_context('pause', 26, 0, False)
    
    # Passed parameters checking function
    pause.stypy_localization = localization
    pause.stypy_type_of_self = None
    pause.stypy_type_store = module_type_store
    pause.stypy_function_name = 'pause'
    pause.stypy_param_names_list = []
    pause.stypy_varargs_param_name = None
    pause.stypy_kwargs_param_name = None
    pause.stypy_call_defaults = defaults
    pause.stypy_call_varargs = varargs
    pause.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pause', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pause', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pause(...)' code ##################

    
    # Call to input(...): (line 28)
    # Processing the call keyword arguments (line 28)
    kwargs_405663 = {}
    # Getting the type of 'input' (line 28)
    input_405662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'input', False)
    # Calling input(args, kwargs) (line 28)
    input_call_result_405664 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), input_405662, *[], **kwargs_405663)
    
    
    # ################# End of 'pause(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pause' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_405665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_405665)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pause'
    return stypy_return_type_405665

# Assigning a type to the variable 'pause' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'pause', pause)

@norecursion
def save(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'save'
    module_type_store = module_type_store.open_function_context('save', 31, 0, False)
    
    # Passed parameters checking function
    save.stypy_localization = localization
    save.stypy_type_of_self = None
    save.stypy_type_store = module_type_store
    save.stypy_function_name = 'save'
    save.stypy_param_names_list = ['ar', 'fileName']
    save.stypy_varargs_param_name = None
    save.stypy_kwargs_param_name = None
    save.stypy_call_defaults = defaults
    save.stypy_call_varargs = varargs
    save.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'save', ['ar', 'fileName'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'save', localization, ['ar', 'fileName'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'save(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 4))
    
    # 'from numpy import savetxt' statement (line 33)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')
    import_405666 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 4), 'numpy')

    if (type(import_405666) is not StypyTypeError):

        if (import_405666 != 'pyd_module'):
            __import__(import_405666)
            sys_modules_405667 = sys.modules[import_405666]
            import_from_module(stypy.reporting.localization.Localization(__file__, 33, 4), 'numpy', sys_modules_405667.module_type_store, module_type_store, ['savetxt'])
            nest_module(stypy.reporting.localization.Localization(__file__, 33, 4), __file__, sys_modules_405667, sys_modules_405667.module_type_store, module_type_store)
        else:
            from numpy import savetxt

            import_from_module(stypy.reporting.localization.Localization(__file__, 33, 4), 'numpy', None, module_type_store, ['savetxt'], [savetxt])

    else:
        # Assigning a type to the variable 'numpy' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'numpy', import_405666)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')
    
    
    # Call to savetxt(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'fileName' (line 34)
    fileName_405669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'fileName', False)
    # Getting the type of 'ar' (line 34)
    ar_405670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 22), 'ar', False)
    # Processing the call keyword arguments (line 34)
    int_405671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 36), 'int')
    keyword_405672 = int_405671
    kwargs_405673 = {'precision': keyword_405672}
    # Getting the type of 'savetxt' (line 34)
    savetxt_405668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'savetxt', False)
    # Calling savetxt(args, kwargs) (line 34)
    savetxt_call_result_405674 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), savetxt_405668, *[fileName_405669, ar_405670], **kwargs_405673)
    
    
    # ################# End of 'save(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'save' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_405675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_405675)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'save'
    return stypy_return_type_405675

# Assigning a type to the variable 'save' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'save', save)

@norecursion
def _assert_symmetric(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_405676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 30), 'float')
    float_405677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 41), 'float')
    defaults = [float_405676, float_405677]
    # Create a new context for function '_assert_symmetric'
    module_type_store = module_type_store.open_function_context('_assert_symmetric', 37, 0, False)
    
    # Passed parameters checking function
    _assert_symmetric.stypy_localization = localization
    _assert_symmetric.stypy_type_of_self = None
    _assert_symmetric.stypy_type_store = module_type_store
    _assert_symmetric.stypy_function_name = '_assert_symmetric'
    _assert_symmetric.stypy_param_names_list = ['M', 'rtol', 'atol']
    _assert_symmetric.stypy_varargs_param_name = None
    _assert_symmetric.stypy_kwargs_param_name = None
    _assert_symmetric.stypy_call_defaults = defaults
    _assert_symmetric.stypy_call_varargs = varargs
    _assert_symmetric.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_assert_symmetric', ['M', 'rtol', 'atol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_assert_symmetric', localization, ['M', 'rtol', 'atol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_assert_symmetric(...)' code ##################

    
    # Call to assert_allclose(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'M' (line 38)
    M_405679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'M', False)
    # Obtaining the member 'T' of a type (line 38)
    T_405680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 20), M_405679, 'T')
    # Getting the type of 'M' (line 38)
    M_405681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'M', False)
    # Processing the call keyword arguments (line 38)
    # Getting the type of 'rtol' (line 38)
    rtol_405682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'rtol', False)
    keyword_405683 = rtol_405682
    # Getting the type of 'atol' (line 38)
    atol_405684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 44), 'atol', False)
    keyword_405685 = atol_405684
    kwargs_405686 = {'rtol': keyword_405683, 'atol': keyword_405685}
    # Getting the type of 'assert_allclose' (line 38)
    assert_allclose_405678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 38)
    assert_allclose_call_result_405687 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), assert_allclose_405678, *[T_405680, M_405681], **kwargs_405686)
    
    
    # ################# End of '_assert_symmetric(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_assert_symmetric' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_405688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_405688)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_assert_symmetric'
    return stypy_return_type_405688

# Assigning a type to the variable '_assert_symmetric' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), '_assert_symmetric', _assert_symmetric)

@norecursion
def as2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'as2d'
    module_type_store = module_type_store.open_function_context('as2d', 45, 0, False)
    
    # Passed parameters checking function
    as2d.stypy_localization = localization
    as2d.stypy_type_of_self = None
    as2d.stypy_type_store = module_type_store
    as2d.stypy_function_name = 'as2d'
    as2d.stypy_param_names_list = ['ar']
    as2d.stypy_varargs_param_name = None
    as2d.stypy_kwargs_param_name = None
    as2d.stypy_call_defaults = defaults
    as2d.stypy_call_varargs = varargs
    as2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'as2d', ['ar'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'as2d', localization, ['ar'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'as2d(...)' code ##################

    str_405689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, (-1)), 'str', '\n    If the input array is 2D return it, if it is 1D, append a dimension,\n    making it a column vector.\n    ')
    
    
    # Getting the type of 'ar' (line 50)
    ar_405690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'ar')
    # Obtaining the member 'ndim' of a type (line 50)
    ndim_405691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 7), ar_405690, 'ndim')
    int_405692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 18), 'int')
    # Applying the binary operator '==' (line 50)
    result_eq_405693 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 7), '==', ndim_405691, int_405692)
    
    # Testing the type of an if condition (line 50)
    if_condition_405694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 4), result_eq_405693)
    # Assigning a type to the variable 'if_condition_405694' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'if_condition_405694', if_condition_405694)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'ar' (line 51)
    ar_405695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'ar')
    # Assigning a type to the variable 'stypy_return_type' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', ar_405695)
    # SSA branch for the else part of an if statement (line 50)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 53):
    
    # Assigning a Call to a Name (line 53):
    
    # Call to array(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'ar' (line 53)
    ar_405698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 23), 'ar', False)
    # Processing the call keyword arguments (line 53)
    # Getting the type of 'False' (line 53)
    False_405699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 32), 'False', False)
    keyword_405700 = False_405699
    kwargs_405701 = {'copy': keyword_405700}
    # Getting the type of 'np' (line 53)
    np_405696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 53)
    array_405697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 14), np_405696, 'array')
    # Calling array(args, kwargs) (line 53)
    array_call_result_405702 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), array_405697, *[ar_405698], **kwargs_405701)
    
    # Assigning a type to the variable 'aux' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'aux', array_call_result_405702)
    
    # Assigning a Tuple to a Attribute (line 54):
    
    # Assigning a Tuple to a Attribute (line 54):
    
    # Obtaining an instance of the builtin type 'tuple' (line 54)
    tuple_405703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 54)
    # Adding element type (line 54)
    
    # Obtaining the type of the subscript
    int_405704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 30), 'int')
    # Getting the type of 'ar' (line 54)
    ar_405705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'ar')
    # Obtaining the member 'shape' of a type (line 54)
    shape_405706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 21), ar_405705, 'shape')
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___405707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 21), shape_405706, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_405708 = invoke(stypy.reporting.localization.Localization(__file__, 54, 21), getitem___405707, int_405704)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 21), tuple_405703, subscript_call_result_405708)
    # Adding element type (line 54)
    int_405709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 21), tuple_405703, int_405709)
    
    # Getting the type of 'aux' (line 54)
    aux_405710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'aux')
    # Setting the type of the member 'shape' of a type (line 54)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), aux_405710, 'shape', tuple_405703)
    # Getting the type of 'aux' (line 55)
    aux_405711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 15), 'aux')
    # Assigning a type to the variable 'stypy_return_type' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'stypy_return_type', aux_405711)
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'as2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'as2d' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_405712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_405712)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'as2d'
    return stypy_return_type_405712

# Assigning a type to the variable 'as2d' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'as2d', as2d)

@norecursion
def _makeOperator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_makeOperator'
    module_type_store = module_type_store.open_function_context('_makeOperator', 58, 0, False)
    
    # Passed parameters checking function
    _makeOperator.stypy_localization = localization
    _makeOperator.stypy_type_of_self = None
    _makeOperator.stypy_type_store = module_type_store
    _makeOperator.stypy_function_name = '_makeOperator'
    _makeOperator.stypy_param_names_list = ['operatorInput', 'expectedShape']
    _makeOperator.stypy_varargs_param_name = None
    _makeOperator.stypy_kwargs_param_name = None
    _makeOperator.stypy_call_defaults = defaults
    _makeOperator.stypy_call_varargs = varargs
    _makeOperator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_makeOperator', ['operatorInput', 'expectedShape'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_makeOperator', localization, ['operatorInput', 'expectedShape'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_makeOperator(...)' code ##################

    str_405713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', 'Takes a dense numpy array or a sparse matrix or\n    a function and makes an operator performing matrix * blockvector\n    products.\n\n    Examples\n    --------\n    >>> A = _makeOperator( arrayA, (n, n) )\n    >>> vectorB = A( vectorX )\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 69)
    # Getting the type of 'operatorInput' (line 69)
    operatorInput_405714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 7), 'operatorInput')
    # Getting the type of 'None' (line 69)
    None_405715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'None')
    
    (may_be_405716, more_types_in_union_405717) = may_be_none(operatorInput_405714, None_405715)

    if may_be_405716:

        if more_types_in_union_405717:
            # Runtime conditional SSA (line 69)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def ident(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'ident'
            module_type_store = module_type_store.open_function_context('ident', 70, 8, False)
            
            # Passed parameters checking function
            ident.stypy_localization = localization
            ident.stypy_type_of_self = None
            ident.stypy_type_store = module_type_store
            ident.stypy_function_name = 'ident'
            ident.stypy_param_names_list = ['x']
            ident.stypy_varargs_param_name = None
            ident.stypy_kwargs_param_name = None
            ident.stypy_call_defaults = defaults
            ident.stypy_call_varargs = varargs
            ident.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'ident', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'ident', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'ident(...)' code ##################

            # Getting the type of 'x' (line 71)
            x_405718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'stypy_return_type', x_405718)
            
            # ################# End of 'ident(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'ident' in the type store
            # Getting the type of 'stypy_return_type' (line 70)
            stypy_return_type_405719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_405719)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'ident'
            return stypy_return_type_405719

        # Assigning a type to the variable 'ident' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'ident', ident)
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to LinearOperator(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'expectedShape' (line 72)
        expectedShape_405721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 34), 'expectedShape', False)
        # Getting the type of 'ident' (line 72)
        ident_405722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 49), 'ident', False)
        # Processing the call keyword arguments (line 72)
        # Getting the type of 'ident' (line 72)
        ident_405723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 63), 'ident', False)
        keyword_405724 = ident_405723
        kwargs_405725 = {'matmat': keyword_405724}
        # Getting the type of 'LinearOperator' (line 72)
        LinearOperator_405720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'LinearOperator', False)
        # Calling LinearOperator(args, kwargs) (line 72)
        LinearOperator_call_result_405726 = invoke(stypy.reporting.localization.Localization(__file__, 72, 19), LinearOperator_405720, *[expectedShape_405721, ident_405722], **kwargs_405725)
        
        # Assigning a type to the variable 'operator' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'operator', LinearOperator_call_result_405726)

        if more_types_in_union_405717:
            # Runtime conditional SSA for else branch (line 69)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_405716) or more_types_in_union_405717):
        
        # Assigning a Call to a Name (line 74):
        
        # Assigning a Call to a Name (line 74):
        
        # Call to aslinearoperator(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'operatorInput' (line 74)
        operatorInput_405728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 36), 'operatorInput', False)
        # Processing the call keyword arguments (line 74)
        kwargs_405729 = {}
        # Getting the type of 'aslinearoperator' (line 74)
        aslinearoperator_405727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'aslinearoperator', False)
        # Calling aslinearoperator(args, kwargs) (line 74)
        aslinearoperator_call_result_405730 = invoke(stypy.reporting.localization.Localization(__file__, 74, 19), aslinearoperator_405727, *[operatorInput_405728], **kwargs_405729)
        
        # Assigning a type to the variable 'operator' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'operator', aslinearoperator_call_result_405730)

        if (may_be_405716 and more_types_in_union_405717):
            # SSA join for if statement (line 69)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'operator' (line 76)
    operator_405731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 7), 'operator')
    # Obtaining the member 'shape' of a type (line 76)
    shape_405732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 7), operator_405731, 'shape')
    # Getting the type of 'expectedShape' (line 76)
    expectedShape_405733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'expectedShape')
    # Applying the binary operator '!=' (line 76)
    result_ne_405734 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 7), '!=', shape_405732, expectedShape_405733)
    
    # Testing the type of an if condition (line 76)
    if_condition_405735 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 4), result_ne_405734)
    # Assigning a type to the variable 'if_condition_405735' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'if_condition_405735', if_condition_405735)
    # SSA begins for if statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 77)
    # Processing the call arguments (line 77)
    str_405737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'str', 'operator has invalid shape')
    # Processing the call keyword arguments (line 77)
    kwargs_405738 = {}
    # Getting the type of 'ValueError' (line 77)
    ValueError_405736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 77)
    ValueError_call_result_405739 = invoke(stypy.reporting.localization.Localization(__file__, 77, 14), ValueError_405736, *[str_405737], **kwargs_405738)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 77, 8), ValueError_call_result_405739, 'raise parameter', BaseException)
    # SSA join for if statement (line 76)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'operator' (line 79)
    operator_405740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'operator')
    # Assigning a type to the variable 'stypy_return_type' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type', operator_405740)
    
    # ################# End of '_makeOperator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_makeOperator' in the type store
    # Getting the type of 'stypy_return_type' (line 58)
    stypy_return_type_405741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_405741)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_makeOperator'
    return stypy_return_type_405741

# Assigning a type to the variable '_makeOperator' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), '_makeOperator', _makeOperator)

@norecursion
def _applyConstraints(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_applyConstraints'
    module_type_store = module_type_store.open_function_context('_applyConstraints', 82, 0, False)
    
    # Passed parameters checking function
    _applyConstraints.stypy_localization = localization
    _applyConstraints.stypy_type_of_self = None
    _applyConstraints.stypy_type_store = module_type_store
    _applyConstraints.stypy_function_name = '_applyConstraints'
    _applyConstraints.stypy_param_names_list = ['blockVectorV', 'factYBY', 'blockVectorBY', 'blockVectorY']
    _applyConstraints.stypy_varargs_param_name = None
    _applyConstraints.stypy_kwargs_param_name = None
    _applyConstraints.stypy_call_defaults = defaults
    _applyConstraints.stypy_call_varargs = varargs
    _applyConstraints.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_applyConstraints', ['blockVectorV', 'factYBY', 'blockVectorBY', 'blockVectorY'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_applyConstraints', localization, ['blockVectorV', 'factYBY', 'blockVectorBY', 'blockVectorY'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_applyConstraints(...)' code ##################

    str_405742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 4), 'str', 'Changes blockVectorV in place.')
    
    # Assigning a Call to a Name (line 84):
    
    # Assigning a Call to a Name (line 84):
    
    # Call to dot(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'blockVectorBY' (line 84)
    blockVectorBY_405745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'blockVectorBY', False)
    # Obtaining the member 'T' of a type (line 84)
    T_405746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), blockVectorBY_405745, 'T')
    # Getting the type of 'blockVectorV' (line 84)
    blockVectorV_405747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 38), 'blockVectorV', False)
    # Processing the call keyword arguments (line 84)
    kwargs_405748 = {}
    # Getting the type of 'np' (line 84)
    np_405743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'np', False)
    # Obtaining the member 'dot' of a type (line 84)
    dot_405744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 14), np_405743, 'dot')
    # Calling dot(args, kwargs) (line 84)
    dot_call_result_405749 = invoke(stypy.reporting.localization.Localization(__file__, 84, 14), dot_405744, *[T_405746, blockVectorV_405747], **kwargs_405748)
    
    # Assigning a type to the variable 'gramYBV' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'gramYBV', dot_call_result_405749)
    
    # Assigning a Call to a Name (line 85):
    
    # Assigning a Call to a Name (line 85):
    
    # Call to cho_solve(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'factYBY' (line 85)
    factYBY_405751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'factYBY', False)
    # Getting the type of 'gramYBV' (line 85)
    gramYBV_405752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'gramYBV', False)
    # Processing the call keyword arguments (line 85)
    kwargs_405753 = {}
    # Getting the type of 'cho_solve' (line 85)
    cho_solve_405750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 10), 'cho_solve', False)
    # Calling cho_solve(args, kwargs) (line 85)
    cho_solve_call_result_405754 = invoke(stypy.reporting.localization.Localization(__file__, 85, 10), cho_solve_405750, *[factYBY_405751, gramYBV_405752], **kwargs_405753)
    
    # Assigning a type to the variable 'tmp' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tmp', cho_solve_call_result_405754)
    
    # Getting the type of 'blockVectorV' (line 86)
    blockVectorV_405755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'blockVectorV')
    
    # Call to dot(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'blockVectorY' (line 86)
    blockVectorY_405758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 27), 'blockVectorY', False)
    # Getting the type of 'tmp' (line 86)
    tmp_405759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 41), 'tmp', False)
    # Processing the call keyword arguments (line 86)
    kwargs_405760 = {}
    # Getting the type of 'np' (line 86)
    np_405756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'np', False)
    # Obtaining the member 'dot' of a type (line 86)
    dot_405757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 20), np_405756, 'dot')
    # Calling dot(args, kwargs) (line 86)
    dot_call_result_405761 = invoke(stypy.reporting.localization.Localization(__file__, 86, 20), dot_405757, *[blockVectorY_405758, tmp_405759], **kwargs_405760)
    
    # Applying the binary operator '-=' (line 86)
    result_isub_405762 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 4), '-=', blockVectorV_405755, dot_call_result_405761)
    # Assigning a type to the variable 'blockVectorV' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'blockVectorV', result_isub_405762)
    
    
    # ################# End of '_applyConstraints(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_applyConstraints' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_405763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_405763)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_applyConstraints'
    return stypy_return_type_405763

# Assigning a type to the variable '_applyConstraints' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), '_applyConstraints', _applyConstraints)

@norecursion
def _b_orthonormalize(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 89)
    None_405764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 53), 'None')
    # Getting the type of 'False' (line 89)
    False_405765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 67), 'False')
    defaults = [None_405764, False_405765]
    # Create a new context for function '_b_orthonormalize'
    module_type_store = module_type_store.open_function_context('_b_orthonormalize', 89, 0, False)
    
    # Passed parameters checking function
    _b_orthonormalize.stypy_localization = localization
    _b_orthonormalize.stypy_type_of_self = None
    _b_orthonormalize.stypy_type_store = module_type_store
    _b_orthonormalize.stypy_function_name = '_b_orthonormalize'
    _b_orthonormalize.stypy_param_names_list = ['B', 'blockVectorV', 'blockVectorBV', 'retInvR']
    _b_orthonormalize.stypy_varargs_param_name = None
    _b_orthonormalize.stypy_kwargs_param_name = None
    _b_orthonormalize.stypy_call_defaults = defaults
    _b_orthonormalize.stypy_call_varargs = varargs
    _b_orthonormalize.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_b_orthonormalize', ['B', 'blockVectorV', 'blockVectorBV', 'retInvR'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_b_orthonormalize', localization, ['B', 'blockVectorV', 'blockVectorBV', 'retInvR'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_b_orthonormalize(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 90)
    # Getting the type of 'blockVectorBV' (line 90)
    blockVectorBV_405766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 7), 'blockVectorBV')
    # Getting the type of 'None' (line 90)
    None_405767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'None')
    
    (may_be_405768, more_types_in_union_405769) = may_be_none(blockVectorBV_405766, None_405767)

    if may_be_405768:

        if more_types_in_union_405769:
            # Runtime conditional SSA (line 90)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 91)
        # Getting the type of 'B' (line 91)
        B_405770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'B')
        # Getting the type of 'None' (line 91)
        None_405771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'None')
        
        (may_be_405772, more_types_in_union_405773) = may_not_be_none(B_405770, None_405771)

        if may_be_405772:

            if more_types_in_union_405773:
                # Runtime conditional SSA (line 91)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 92):
            
            # Assigning a Call to a Name (line 92):
            
            # Call to B(...): (line 92)
            # Processing the call arguments (line 92)
            # Getting the type of 'blockVectorV' (line 92)
            blockVectorV_405775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 30), 'blockVectorV', False)
            # Processing the call keyword arguments (line 92)
            kwargs_405776 = {}
            # Getting the type of 'B' (line 92)
            B_405774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'B', False)
            # Calling B(args, kwargs) (line 92)
            B_call_result_405777 = invoke(stypy.reporting.localization.Localization(__file__, 92, 28), B_405774, *[blockVectorV_405775], **kwargs_405776)
            
            # Assigning a type to the variable 'blockVectorBV' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'blockVectorBV', B_call_result_405777)

            if more_types_in_union_405773:
                # Runtime conditional SSA for else branch (line 91)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_405772) or more_types_in_union_405773):
            
            # Assigning a Name to a Name (line 94):
            
            # Assigning a Name to a Name (line 94):
            # Getting the type of 'blockVectorV' (line 94)
            blockVectorV_405778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'blockVectorV')
            # Assigning a type to the variable 'blockVectorBV' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'blockVectorBV', blockVectorV_405778)

            if (may_be_405772 and more_types_in_union_405773):
                # SSA join for if statement (line 91)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_405769:
            # SSA join for if statement (line 90)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 95):
    
    # Assigning a Call to a Name (line 95):
    
    # Call to dot(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'blockVectorV' (line 95)
    blockVectorV_405781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'blockVectorV', False)
    # Obtaining the member 'T' of a type (line 95)
    T_405782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 21), blockVectorV_405781, 'T')
    # Getting the type of 'blockVectorBV' (line 95)
    blockVectorBV_405783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 37), 'blockVectorBV', False)
    # Processing the call keyword arguments (line 95)
    kwargs_405784 = {}
    # Getting the type of 'np' (line 95)
    np_405779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 14), 'np', False)
    # Obtaining the member 'dot' of a type (line 95)
    dot_405780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 14), np_405779, 'dot')
    # Calling dot(args, kwargs) (line 95)
    dot_call_result_405785 = invoke(stypy.reporting.localization.Localization(__file__, 95, 14), dot_405780, *[T_405782, blockVectorBV_405783], **kwargs_405784)
    
    # Assigning a type to the variable 'gramVBV' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'gramVBV', dot_call_result_405785)
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to cholesky(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'gramVBV' (line 96)
    gramVBV_405787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'gramVBV', False)
    # Processing the call keyword arguments (line 96)
    kwargs_405788 = {}
    # Getting the type of 'cholesky' (line 96)
    cholesky_405786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 14), 'cholesky', False)
    # Calling cholesky(args, kwargs) (line 96)
    cholesky_call_result_405789 = invoke(stypy.reporting.localization.Localization(__file__, 96, 14), cholesky_405786, *[gramVBV_405787], **kwargs_405788)
    
    # Assigning a type to the variable 'gramVBV' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'gramVBV', cholesky_call_result_405789)
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to inv(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'gramVBV' (line 97)
    gramVBV_405791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'gramVBV', False)
    # Processing the call keyword arguments (line 97)
    # Getting the type of 'True' (line 97)
    True_405792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 39), 'True', False)
    keyword_405793 = True_405792
    kwargs_405794 = {'overwrite_a': keyword_405793}
    # Getting the type of 'inv' (line 97)
    inv_405790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'inv', False)
    # Calling inv(args, kwargs) (line 97)
    inv_call_result_405795 = invoke(stypy.reporting.localization.Localization(__file__, 97, 14), inv_405790, *[gramVBV_405791], **kwargs_405794)
    
    # Assigning a type to the variable 'gramVBV' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'gramVBV', inv_call_result_405795)
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to dot(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'blockVectorV' (line 99)
    blockVectorV_405798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'blockVectorV', False)
    # Getting the type of 'gramVBV' (line 99)
    gramVBV_405799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 40), 'gramVBV', False)
    # Processing the call keyword arguments (line 99)
    kwargs_405800 = {}
    # Getting the type of 'np' (line 99)
    np_405796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'np', False)
    # Obtaining the member 'dot' of a type (line 99)
    dot_405797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 19), np_405796, 'dot')
    # Calling dot(args, kwargs) (line 99)
    dot_call_result_405801 = invoke(stypy.reporting.localization.Localization(__file__, 99, 19), dot_405797, *[blockVectorV_405798, gramVBV_405799], **kwargs_405800)
    
    # Assigning a type to the variable 'blockVectorV' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'blockVectorV', dot_call_result_405801)
    
    # Type idiom detected: calculating its left and rigth part (line 100)
    # Getting the type of 'B' (line 100)
    B_405802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'B')
    # Getting the type of 'None' (line 100)
    None_405803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'None')
    
    (may_be_405804, more_types_in_union_405805) = may_not_be_none(B_405802, None_405803)

    if may_be_405804:

        if more_types_in_union_405805:
            # Runtime conditional SSA (line 100)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to dot(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'blockVectorBV' (line 101)
        blockVectorBV_405808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'blockVectorBV', False)
        # Getting the type of 'gramVBV' (line 101)
        gramVBV_405809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 46), 'gramVBV', False)
        # Processing the call keyword arguments (line 101)
        kwargs_405810 = {}
        # Getting the type of 'np' (line 101)
        np_405806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'np', False)
        # Obtaining the member 'dot' of a type (line 101)
        dot_405807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 24), np_405806, 'dot')
        # Calling dot(args, kwargs) (line 101)
        dot_call_result_405811 = invoke(stypy.reporting.localization.Localization(__file__, 101, 24), dot_405807, *[blockVectorBV_405808, gramVBV_405809], **kwargs_405810)
        
        # Assigning a type to the variable 'blockVectorBV' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'blockVectorBV', dot_call_result_405811)

        if more_types_in_union_405805:
            # SSA join for if statement (line 100)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'retInvR' (line 103)
    retInvR_405812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 7), 'retInvR')
    # Testing the type of an if condition (line 103)
    if_condition_405813 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 4), retInvR_405812)
    # Assigning a type to the variable 'if_condition_405813' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'if_condition_405813', if_condition_405813)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_405814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    # Getting the type of 'blockVectorV' (line 104)
    blockVectorV_405815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'blockVectorV')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), tuple_405814, blockVectorV_405815)
    # Adding element type (line 104)
    # Getting the type of 'blockVectorBV' (line 104)
    blockVectorBV_405816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'blockVectorBV')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), tuple_405814, blockVectorBV_405816)
    # Adding element type (line 104)
    # Getting the type of 'gramVBV' (line 104)
    gramVBV_405817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 44), 'gramVBV')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), tuple_405814, gramVBV_405817)
    
    # Assigning a type to the variable 'stypy_return_type' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'stypy_return_type', tuple_405814)
    # SSA branch for the else part of an if statement (line 103)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 106)
    tuple_405818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 106)
    # Adding element type (line 106)
    # Getting the type of 'blockVectorV' (line 106)
    blockVectorV_405819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'blockVectorV')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 15), tuple_405818, blockVectorV_405819)
    # Adding element type (line 106)
    # Getting the type of 'blockVectorBV' (line 106)
    blockVectorBV_405820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'blockVectorBV')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 15), tuple_405818, blockVectorBV_405820)
    
    # Assigning a type to the variable 'stypy_return_type' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'stypy_return_type', tuple_405818)
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_b_orthonormalize(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_b_orthonormalize' in the type store
    # Getting the type of 'stypy_return_type' (line 89)
    stypy_return_type_405821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_405821)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_b_orthonormalize'
    return stypy_return_type_405821

# Assigning a type to the variable '_b_orthonormalize' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), '_b_orthonormalize', _b_orthonormalize)

@norecursion
def lobpcg(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 110)
    None_405822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'None')
    # Getting the type of 'None' (line 110)
    None_405823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'None')
    # Getting the type of 'None' (line 110)
    None_405824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 30), 'None')
    # Getting the type of 'None' (line 111)
    None_405825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'None')
    int_405826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 30), 'int')
    # Getting the type of 'True' (line 112)
    True_405827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'True')
    int_405828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 41), 'int')
    # Getting the type of 'False' (line 113)
    False_405829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 29), 'False')
    # Getting the type of 'False' (line 113)
    False_405830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 60), 'False')
    defaults = [None_405822, None_405823, None_405824, None_405825, int_405826, True_405827, int_405828, False_405829, False_405830]
    # Create a new context for function 'lobpcg'
    module_type_store = module_type_store.open_function_context('lobpcg', 109, 0, False)
    
    # Passed parameters checking function
    lobpcg.stypy_localization = localization
    lobpcg.stypy_type_of_self = None
    lobpcg.stypy_type_store = module_type_store
    lobpcg.stypy_function_name = 'lobpcg'
    lobpcg.stypy_param_names_list = ['A', 'X', 'B', 'M', 'Y', 'tol', 'maxiter', 'largest', 'verbosityLevel', 'retLambdaHistory', 'retResidualNormsHistory']
    lobpcg.stypy_varargs_param_name = None
    lobpcg.stypy_kwargs_param_name = None
    lobpcg.stypy_call_defaults = defaults
    lobpcg.stypy_call_varargs = varargs
    lobpcg.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lobpcg', ['A', 'X', 'B', 'M', 'Y', 'tol', 'maxiter', 'largest', 'verbosityLevel', 'retLambdaHistory', 'retResidualNormsHistory'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lobpcg', localization, ['A', 'X', 'B', 'M', 'Y', 'tol', 'maxiter', 'largest', 'verbosityLevel', 'retLambdaHistory', 'retResidualNormsHistory'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lobpcg(...)' code ##################

    str_405831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, (-1)), 'str', 'Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)\n\n    LOBPCG is a preconditioned eigensolver for large symmetric positive\n    definite (SPD) generalized eigenproblems.\n\n    Parameters\n    ----------\n    A : {sparse matrix, dense matrix, LinearOperator}\n        The symmetric linear operator of the problem, usually a\n        sparse matrix.  Often called the "stiffness matrix".\n    X : array_like\n        Initial approximation to the k eigenvectors. If A has\n        shape=(n,n) then X should have shape shape=(n,k).\n    B : {dense matrix, sparse matrix, LinearOperator}, optional\n        the right hand side operator in a generalized eigenproblem.\n        by default, B = Identity\n        often called the "mass matrix"\n    M : {dense matrix, sparse matrix, LinearOperator}, optional\n        preconditioner to A; by default M = Identity\n        M should approximate the inverse of A\n    Y : array_like, optional\n        n-by-sizeY matrix of constraints, sizeY < n\n        The iterations will be performed in the B-orthogonal complement\n        of the column-space of Y. Y must be full rank.\n\n    Returns\n    -------\n    w : array\n        Array of k eigenvalues\n    v : array\n        An array of k eigenvectors.  V has the same shape as X.\n\n    Other Parameters\n    ----------------\n    tol : scalar, optional\n        Solver tolerance (stopping criterion)\n        by default: tol=n*sqrt(eps)\n    maxiter : integer, optional\n        maximum number of iterations\n        by default: maxiter=min(n,20)\n    largest : bool, optional\n        when True, solve for the largest eigenvalues, otherwise the smallest\n    verbosityLevel : integer, optional\n        controls solver output.  default: verbosityLevel = 0.\n    retLambdaHistory : boolean, optional\n        whether to return eigenvalue history\n    retResidualNormsHistory : boolean, optional\n        whether to return history of residual norms\n\n    Examples\n    --------\n\n    Solve A x = lambda B x with constraints and preconditioning.\n\n    >>> from scipy.sparse import spdiags, issparse\n    >>> from scipy.sparse.linalg import lobpcg, LinearOperator\n    >>> n = 100\n    >>> vals = [np.arange(n, dtype=np.float64) + 1]\n    >>> A = spdiags(vals, 0, n, n)\n    >>> A.toarray()\n    array([[   1.,    0.,    0., ...,    0.,    0.,    0.],\n           [   0.,    2.,    0., ...,    0.,    0.,    0.],\n           [   0.,    0.,    3., ...,    0.,    0.,    0.],\n           ...,\n           [   0.,    0.,    0., ...,   98.,    0.,    0.],\n           [   0.,    0.,    0., ...,    0.,   99.,    0.],\n           [   0.,    0.,    0., ...,    0.,    0.,  100.]])\n\n    Constraints.\n\n    >>> Y = np.eye(n, 3)\n\n    Initial guess for eigenvectors, should have linearly independent\n    columns. Column dimension = number of requested eigenvalues.\n\n    >>> X = np.random.rand(n, 3)\n\n    Preconditioner -- inverse of A (as an abstract linear operator).\n\n    >>> invA = spdiags([1./vals[0]], 0, n, n)\n    >>> def precond( x ):\n    ...     return invA  * x\n    >>> M = LinearOperator(matvec=precond, shape=(n, n), dtype=float)\n\n    Here, ``invA`` could of course have been used directly as a preconditioner.\n    Let us then solve the problem:\n\n    >>> eigs, vecs = lobpcg(A, X, Y=Y, M=M, tol=1e-4, maxiter=40, largest=False)\n    >>> eigs\n    array([ 4.,  5.,  6.])\n\n    Note that the vectors passed in Y are the eigenvectors of the 3 smallest\n    eigenvalues. The results returned are orthogonal to those.\n\n    Notes\n    -----\n    If both retLambdaHistory and retResidualNormsHistory are True,\n    the return tuple has the following format\n    (lambda, V, lambda history, residual norms history).\n\n    In the following ``n`` denotes the matrix size and ``m`` the number\n    of required eigenvalues (smallest or largest).\n\n    The LOBPCG code internally solves eigenproblems of the size 3``m`` on every\n    iteration by calling the "standard" dense eigensolver, so if ``m`` is not\n    small enough compared to ``n``, it does not make sense to call the LOBPCG\n    code, but rather one should use the "standard" eigensolver,\n    e.g. numpy or scipy function in this case.\n    If one calls the LOBPCG algorithm for 5``m``>``n``,\n    it will most likely break internally, so the code tries to call the standard\n    function instead.\n\n    It is not that n should be large for the LOBPCG to work, but rather the\n    ratio ``n``/``m`` should be large. It you call the LOBPCG code with ``m``=1\n    and ``n``=10, it should work, though ``n`` is small. The method is intended\n    for extremely large ``n``/``m``, see e.g., reference [28] in\n    http://arxiv.org/abs/0705.2626\n\n    The convergence speed depends basically on two factors:\n\n    1.  How well relatively separated the seeking eigenvalues are\n        from the rest of the eigenvalues.\n        One can try to vary ``m`` to make this better.\n\n    2.  How well conditioned the problem is. This can be changed by using proper\n        preconditioning. For example, a rod vibration test problem (under tests\n        directory) is ill-conditioned for large ``n``, so convergence will be\n        slow, unless efficient preconditioning is used.\n        For this specific problem, a good simple preconditioner function would\n        be a linear solve for A, which is easy to code since A is tridiagonal.\n\n    *Acknowledgements*\n\n    lobpcg.py code was written by Robert Cimrman.\n    Many thanks belong to Andrew Knyazev, the author of the algorithm,\n    for lots of advice and support.\n\n    References\n    ----------\n    .. [1] A. V. Knyazev (2001),\n           Toward the Optimal Preconditioned Eigensolver: Locally Optimal\n           Block Preconditioned Conjugate Gradient Method.\n           SIAM Journal on Scientific Computing 23, no. 2,\n           pp. 517-541. http://dx.doi.org/10.1137/S1064827500366124\n\n    .. [2] A. V. Knyazev, I. Lashuk, M. E. Argentati, and E. Ovchinnikov (2007),\n           Block Locally Optimal Preconditioned Eigenvalue Xolvers (BLOPEX)\n           in hypre and PETSc.  http://arxiv.org/abs/0705.2626\n\n    .. [3] A. V. Knyazev\'s C and MATLAB implementations:\n           https://bitbucket.org/joseroman/blopex\n\n    ')
    
    # Assigning a Name to a Name (line 267):
    
    # Assigning a Name to a Name (line 267):
    # Getting the type of 'X' (line 267)
    X_405832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 19), 'X')
    # Assigning a type to the variable 'blockVectorX' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'blockVectorX', X_405832)
    
    # Assigning a Name to a Name (line 268):
    
    # Assigning a Name to a Name (line 268):
    # Getting the type of 'Y' (line 268)
    Y_405833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'Y')
    # Assigning a type to the variable 'blockVectorY' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'blockVectorY', Y_405833)
    
    # Assigning a Name to a Name (line 269):
    
    # Assigning a Name to a Name (line 269):
    # Getting the type of 'tol' (line 269)
    tol_405834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 24), 'tol')
    # Assigning a type to the variable 'residualTolerance' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'residualTolerance', tol_405834)
    
    # Assigning a Name to a Name (line 270):
    
    # Assigning a Name to a Name (line 270):
    # Getting the type of 'maxiter' (line 270)
    maxiter_405835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'maxiter')
    # Assigning a type to the variable 'maxIterations' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'maxIterations', maxiter_405835)
    
    # Type idiom detected: calculating its left and rigth part (line 272)
    # Getting the type of 'blockVectorY' (line 272)
    blockVectorY_405836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'blockVectorY')
    # Getting the type of 'None' (line 272)
    None_405837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 27), 'None')
    
    (may_be_405838, more_types_in_union_405839) = may_not_be_none(blockVectorY_405836, None_405837)

    if may_be_405838:

        if more_types_in_union_405839:
            # Runtime conditional SSA (line 272)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 273):
        
        # Assigning a Subscript to a Name (line 273):
        
        # Obtaining the type of the subscript
        int_405840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 35), 'int')
        # Getting the type of 'blockVectorY' (line 273)
        blockVectorY_405841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'blockVectorY')
        # Obtaining the member 'shape' of a type (line 273)
        shape_405842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), blockVectorY_405841, 'shape')
        # Obtaining the member '__getitem__' of a type (line 273)
        getitem___405843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), shape_405842, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 273)
        subscript_call_result_405844 = invoke(stypy.reporting.localization.Localization(__file__, 273, 16), getitem___405843, int_405840)
        
        # Assigning a type to the variable 'sizeY' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'sizeY', subscript_call_result_405844)

        if more_types_in_union_405839:
            # Runtime conditional SSA for else branch (line 272)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_405838) or more_types_in_union_405839):
        
        # Assigning a Num to a Name (line 275):
        
        # Assigning a Num to a Name (line 275):
        int_405845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 16), 'int')
        # Assigning a type to the variable 'sizeY' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'sizeY', int_405845)

        if (may_be_405838 and more_types_in_union_405839):
            # SSA join for if statement (line 272)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to len(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'blockVectorX' (line 278)
    blockVectorX_405847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), 'blockVectorX', False)
    # Obtaining the member 'shape' of a type (line 278)
    shape_405848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 11), blockVectorX_405847, 'shape')
    # Processing the call keyword arguments (line 278)
    kwargs_405849 = {}
    # Getting the type of 'len' (line 278)
    len_405846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 7), 'len', False)
    # Calling len(args, kwargs) (line 278)
    len_call_result_405850 = invoke(stypy.reporting.localization.Localization(__file__, 278, 7), len_405846, *[shape_405848], **kwargs_405849)
    
    int_405851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 34), 'int')
    # Applying the binary operator '!=' (line 278)
    result_ne_405852 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 7), '!=', len_call_result_405850, int_405851)
    
    # Testing the type of an if condition (line 278)
    if_condition_405853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 4), result_ne_405852)
    # Assigning a type to the variable 'if_condition_405853' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'if_condition_405853', if_condition_405853)
    # SSA begins for if statement (line 278)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 279)
    # Processing the call arguments (line 279)
    str_405855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 25), 'str', 'expected rank-2 array for argument X')
    # Processing the call keyword arguments (line 279)
    kwargs_405856 = {}
    # Getting the type of 'ValueError' (line 279)
    ValueError_405854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 279)
    ValueError_call_result_405857 = invoke(stypy.reporting.localization.Localization(__file__, 279, 14), ValueError_405854, *[str_405855], **kwargs_405856)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 279, 8), ValueError_call_result_405857, 'raise parameter', BaseException)
    # SSA join for if statement (line 278)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 281):
    
    # Assigning a Subscript to a Name (line 281):
    
    # Obtaining the type of the subscript
    int_405858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 4), 'int')
    # Getting the type of 'blockVectorX' (line 281)
    blockVectorX_405859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 15), 'blockVectorX')
    # Obtaining the member 'shape' of a type (line 281)
    shape_405860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 15), blockVectorX_405859, 'shape')
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___405861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 4), shape_405860, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_405862 = invoke(stypy.reporting.localization.Localization(__file__, 281, 4), getitem___405861, int_405858)
    
    # Assigning a type to the variable 'tuple_var_assignment_405633' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'tuple_var_assignment_405633', subscript_call_result_405862)
    
    # Assigning a Subscript to a Name (line 281):
    
    # Obtaining the type of the subscript
    int_405863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 4), 'int')
    # Getting the type of 'blockVectorX' (line 281)
    blockVectorX_405864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 15), 'blockVectorX')
    # Obtaining the member 'shape' of a type (line 281)
    shape_405865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 15), blockVectorX_405864, 'shape')
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___405866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 4), shape_405865, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_405867 = invoke(stypy.reporting.localization.Localization(__file__, 281, 4), getitem___405866, int_405863)
    
    # Assigning a type to the variable 'tuple_var_assignment_405634' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'tuple_var_assignment_405634', subscript_call_result_405867)
    
    # Assigning a Name to a Name (line 281):
    # Getting the type of 'tuple_var_assignment_405633' (line 281)
    tuple_var_assignment_405633_405868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'tuple_var_assignment_405633')
    # Assigning a type to the variable 'n' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'n', tuple_var_assignment_405633_405868)
    
    # Assigning a Name to a Name (line 281):
    # Getting the type of 'tuple_var_assignment_405634' (line 281)
    tuple_var_assignment_405634_405869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'tuple_var_assignment_405634')
    # Assigning a type to the variable 'sizeX' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 7), 'sizeX', tuple_var_assignment_405634_405869)
    
    
    # Getting the type of 'sizeX' (line 282)
    sizeX_405870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 7), 'sizeX')
    # Getting the type of 'n' (line 282)
    n_405871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'n')
    # Applying the binary operator '>' (line 282)
    result_gt_405872 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 7), '>', sizeX_405870, n_405871)
    
    # Testing the type of an if condition (line 282)
    if_condition_405873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 4), result_gt_405872)
    # Assigning a type to the variable 'if_condition_405873' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'if_condition_405873', if_condition_405873)
    # SSA begins for if statement (line 282)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 283)
    # Processing the call arguments (line 283)
    str_405875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 25), 'str', 'X column dimension exceeds the row dimension')
    # Processing the call keyword arguments (line 283)
    kwargs_405876 = {}
    # Getting the type of 'ValueError' (line 283)
    ValueError_405874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 283)
    ValueError_call_result_405877 = invoke(stypy.reporting.localization.Localization(__file__, 283, 14), ValueError_405874, *[str_405875], **kwargs_405876)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 283, 8), ValueError_call_result_405877, 'raise parameter', BaseException)
    # SSA join for if statement (line 282)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 285):
    
    # Assigning a Call to a Name (line 285):
    
    # Call to _makeOperator(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'A' (line 285)
    A_405879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 22), 'A', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 285)
    tuple_405880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 285)
    # Adding element type (line 285)
    # Getting the type of 'n' (line 285)
    n_405881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 26), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 26), tuple_405880, n_405881)
    # Adding element type (line 285)
    # Getting the type of 'n' (line 285)
    n_405882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 26), tuple_405880, n_405882)
    
    # Processing the call keyword arguments (line 285)
    kwargs_405883 = {}
    # Getting the type of '_makeOperator' (line 285)
    _makeOperator_405878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), '_makeOperator', False)
    # Calling _makeOperator(args, kwargs) (line 285)
    _makeOperator_call_result_405884 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), _makeOperator_405878, *[A_405879, tuple_405880], **kwargs_405883)
    
    # Assigning a type to the variable 'A' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'A', _makeOperator_call_result_405884)
    
    # Assigning a Call to a Name (line 286):
    
    # Assigning a Call to a Name (line 286):
    
    # Call to _makeOperator(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'B' (line 286)
    B_405886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'B', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 286)
    tuple_405887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 286)
    # Adding element type (line 286)
    # Getting the type of 'n' (line 286)
    n_405888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 26), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 26), tuple_405887, n_405888)
    # Adding element type (line 286)
    # Getting the type of 'n' (line 286)
    n_405889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 26), tuple_405887, n_405889)
    
    # Processing the call keyword arguments (line 286)
    kwargs_405890 = {}
    # Getting the type of '_makeOperator' (line 286)
    _makeOperator_405885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), '_makeOperator', False)
    # Calling _makeOperator(args, kwargs) (line 286)
    _makeOperator_call_result_405891 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), _makeOperator_405885, *[B_405886, tuple_405887], **kwargs_405890)
    
    # Assigning a type to the variable 'B' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'B', _makeOperator_call_result_405891)
    
    # Assigning a Call to a Name (line 287):
    
    # Assigning a Call to a Name (line 287):
    
    # Call to _makeOperator(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'M' (line 287)
    M_405893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 22), 'M', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 287)
    tuple_405894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 287)
    # Adding element type (line 287)
    # Getting the type of 'n' (line 287)
    n_405895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 26), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 26), tuple_405894, n_405895)
    # Adding element type (line 287)
    # Getting the type of 'n' (line 287)
    n_405896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 26), tuple_405894, n_405896)
    
    # Processing the call keyword arguments (line 287)
    kwargs_405897 = {}
    # Getting the type of '_makeOperator' (line 287)
    _makeOperator_405892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), '_makeOperator', False)
    # Calling _makeOperator(args, kwargs) (line 287)
    _makeOperator_call_result_405898 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), _makeOperator_405892, *[M_405893, tuple_405894], **kwargs_405897)
    
    # Assigning a type to the variable 'M' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'M', _makeOperator_call_result_405898)
    
    
    # Getting the type of 'n' (line 289)
    n_405899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'n')
    # Getting the type of 'sizeY' (line 289)
    sizeY_405900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'sizeY')
    # Applying the binary operator '-' (line 289)
    result_sub_405901 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 8), '-', n_405899, sizeY_405900)
    
    int_405902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 22), 'int')
    # Getting the type of 'sizeX' (line 289)
    sizeX_405903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'sizeX')
    # Applying the binary operator '*' (line 289)
    result_mul_405904 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 22), '*', int_405902, sizeX_405903)
    
    # Applying the binary operator '<' (line 289)
    result_lt_405905 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 7), '<', result_sub_405901, result_mul_405904)
    
    # Testing the type of an if condition (line 289)
    if_condition_405906 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 4), result_lt_405905)
    # Assigning a type to the variable 'if_condition_405906' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'if_condition_405906', if_condition_405906)
    # SSA begins for if statement (line 289)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 293)
    # Getting the type of 'blockVectorY' (line 293)
    blockVectorY_405907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'blockVectorY')
    # Getting the type of 'None' (line 293)
    None_405908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 31), 'None')
    
    (may_be_405909, more_types_in_union_405910) = may_not_be_none(blockVectorY_405907, None_405908)

    if may_be_405909:

        if more_types_in_union_405910:
            # Runtime conditional SSA (line 293)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to NotImplementedError(...): (line 294)
        # Processing the call arguments (line 294)
        str_405912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 38), 'str', 'The dense eigensolver does not support constraints.')
        # Processing the call keyword arguments (line 294)
        kwargs_405913 = {}
        # Getting the type of 'NotImplementedError' (line 294)
        NotImplementedError_405911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 18), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 294)
        NotImplementedError_call_result_405914 = invoke(stypy.reporting.localization.Localization(__file__, 294, 18), NotImplementedError_405911, *[str_405912], **kwargs_405913)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 294, 12), NotImplementedError_call_result_405914, 'raise parameter', BaseException)

        if more_types_in_union_405910:
            # SSA join for if statement (line 293)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'largest' (line 298)
    largest_405915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 11), 'largest')
    # Testing the type of an if condition (line 298)
    if_condition_405916 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 8), largest_405915)
    # Assigning a type to the variable 'if_condition_405916' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'if_condition_405916', if_condition_405916)
    # SSA begins for if statement (line 298)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 299):
    
    # Assigning a Tuple to a Name (line 299):
    
    # Obtaining an instance of the builtin type 'tuple' (line 299)
    tuple_405917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 299)
    # Adding element type (line 299)
    # Getting the type of 'n' (line 299)
    n_405918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 23), 'n')
    # Getting the type of 'sizeX' (line 299)
    sizeX_405919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 27), 'sizeX')
    # Applying the binary operator '-' (line 299)
    result_sub_405920 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 23), '-', n_405918, sizeX_405919)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 23), tuple_405917, result_sub_405920)
    # Adding element type (line 299)
    # Getting the type of 'n' (line 299)
    n_405921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 34), 'n')
    int_405922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 36), 'int')
    # Applying the binary operator '-' (line 299)
    result_sub_405923 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 34), '-', n_405921, int_405922)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 23), tuple_405917, result_sub_405923)
    
    # Assigning a type to the variable 'eigvals' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'eigvals', tuple_405917)
    # SSA branch for the else part of an if statement (line 298)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Name (line 301):
    
    # Assigning a Tuple to a Name (line 301):
    
    # Obtaining an instance of the builtin type 'tuple' (line 301)
    tuple_405924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 301)
    # Adding element type (line 301)
    int_405925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 23), tuple_405924, int_405925)
    # Adding element type (line 301)
    # Getting the type of 'sizeX' (line 301)
    sizeX_405926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 26), 'sizeX')
    int_405927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 32), 'int')
    # Applying the binary operator '-' (line 301)
    result_sub_405928 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 26), '-', sizeX_405926, int_405927)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 23), tuple_405924, result_sub_405928)
    
    # Assigning a type to the variable 'eigvals' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'eigvals', tuple_405924)
    # SSA join for if statement (line 298)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 303):
    
    # Assigning a Call to a Name (line 303):
    
    # Call to A(...): (line 303)
    # Processing the call arguments (line 303)
    
    # Call to eye(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'n' (line 303)
    n_405932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 27), 'n', False)
    # Processing the call keyword arguments (line 303)
    kwargs_405933 = {}
    # Getting the type of 'np' (line 303)
    np_405930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'np', False)
    # Obtaining the member 'eye' of a type (line 303)
    eye_405931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 20), np_405930, 'eye')
    # Calling eye(args, kwargs) (line 303)
    eye_call_result_405934 = invoke(stypy.reporting.localization.Localization(__file__, 303, 20), eye_405931, *[n_405932], **kwargs_405933)
    
    # Processing the call keyword arguments (line 303)
    kwargs_405935 = {}
    # Getting the type of 'A' (line 303)
    A_405929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 18), 'A', False)
    # Calling A(args, kwargs) (line 303)
    A_call_result_405936 = invoke(stypy.reporting.localization.Localization(__file__, 303, 18), A_405929, *[eye_call_result_405934], **kwargs_405935)
    
    # Assigning a type to the variable 'A_dense' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'A_dense', A_call_result_405936)
    
    # Assigning a IfExp to a Name (line 304):
    
    # Assigning a IfExp to a Name (line 304):
    
    
    # Getting the type of 'B' (line 304)
    B_405937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 26), 'B')
    # Getting the type of 'None' (line 304)
    None_405938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 31), 'None')
    # Applying the binary operator 'is' (line 304)
    result_is__405939 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 26), 'is', B_405937, None_405938)
    
    # Testing the type of an if expression (line 304)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 18), result_is__405939)
    # SSA begins for if expression (line 304)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'None' (line 304)
    None_405940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 18), 'None')
    # SSA branch for the else part of an if expression (line 304)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to B(...): (line 304)
    # Processing the call arguments (line 304)
    
    # Call to eye(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'n' (line 304)
    n_405944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 50), 'n', False)
    # Processing the call keyword arguments (line 304)
    kwargs_405945 = {}
    # Getting the type of 'np' (line 304)
    np_405942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 43), 'np', False)
    # Obtaining the member 'eye' of a type (line 304)
    eye_405943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 43), np_405942, 'eye')
    # Calling eye(args, kwargs) (line 304)
    eye_call_result_405946 = invoke(stypy.reporting.localization.Localization(__file__, 304, 43), eye_405943, *[n_405944], **kwargs_405945)
    
    # Processing the call keyword arguments (line 304)
    kwargs_405947 = {}
    # Getting the type of 'B' (line 304)
    B_405941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 41), 'B', False)
    # Calling B(args, kwargs) (line 304)
    B_call_result_405948 = invoke(stypy.reporting.localization.Localization(__file__, 304, 41), B_405941, *[eye_call_result_405946], **kwargs_405947)
    
    # SSA join for if expression (line 304)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_405949 = union_type.UnionType.add(None_405940, B_call_result_405948)
    
    # Assigning a type to the variable 'B_dense' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'B_dense', if_exp_405949)
    
    # Call to eigh(...): (line 305)
    # Processing the call arguments (line 305)
    # Getting the type of 'A_dense' (line 305)
    A_dense_405951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 20), 'A_dense', False)
    # Getting the type of 'B_dense' (line 305)
    B_dense_405952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 29), 'B_dense', False)
    # Processing the call keyword arguments (line 305)
    # Getting the type of 'eigvals' (line 305)
    eigvals_405953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 46), 'eigvals', False)
    keyword_405954 = eigvals_405953
    # Getting the type of 'False' (line 305)
    False_405955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 68), 'False', False)
    keyword_405956 = False_405955
    kwargs_405957 = {'eigvals': keyword_405954, 'check_finite': keyword_405956}
    # Getting the type of 'eigh' (line 305)
    eigh_405950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 15), 'eigh', False)
    # Calling eigh(args, kwargs) (line 305)
    eigh_call_result_405958 = invoke(stypy.reporting.localization.Localization(__file__, 305, 15), eigh_405950, *[A_dense_405951, B_dense_405952], **kwargs_405957)
    
    # Assigning a type to the variable 'stypy_return_type' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'stypy_return_type', eigh_call_result_405958)
    # SSA join for if statement (line 289)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 307)
    # Getting the type of 'residualTolerance' (line 307)
    residualTolerance_405959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 7), 'residualTolerance')
    # Getting the type of 'None' (line 307)
    None_405960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 28), 'None')
    
    (may_be_405961, more_types_in_union_405962) = may_be_none(residualTolerance_405959, None_405960)

    if may_be_405961:

        if more_types_in_union_405962:
            # Runtime conditional SSA (line 307)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 308):
        
        # Assigning a BinOp to a Name (line 308):
        
        # Call to sqrt(...): (line 308)
        # Processing the call arguments (line 308)
        float_405965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 36), 'float')
        # Processing the call keyword arguments (line 308)
        kwargs_405966 = {}
        # Getting the type of 'np' (line 308)
        np_405963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 28), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 308)
        sqrt_405964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 28), np_405963, 'sqrt')
        # Calling sqrt(args, kwargs) (line 308)
        sqrt_call_result_405967 = invoke(stypy.reporting.localization.Localization(__file__, 308, 28), sqrt_405964, *[float_405965], **kwargs_405966)
        
        # Getting the type of 'n' (line 308)
        n_405968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 45), 'n')
        # Applying the binary operator '*' (line 308)
        result_mul_405969 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 28), '*', sqrt_call_result_405967, n_405968)
        
        # Assigning a type to the variable 'residualTolerance' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'residualTolerance', result_mul_405969)

        if more_types_in_union_405962:
            # SSA join for if statement (line 307)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 310):
    
    # Assigning a Call to a Name (line 310):
    
    # Call to min(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'n' (line 310)
    n_405971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 24), 'n', False)
    # Getting the type of 'maxIterations' (line 310)
    maxIterations_405972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 27), 'maxIterations', False)
    # Processing the call keyword arguments (line 310)
    kwargs_405973 = {}
    # Getting the type of 'min' (line 310)
    min_405970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 20), 'min', False)
    # Calling min(args, kwargs) (line 310)
    min_call_result_405974 = invoke(stypy.reporting.localization.Localization(__file__, 310, 20), min_405970, *[n_405971, maxIterations_405972], **kwargs_405973)
    
    # Assigning a type to the variable 'maxIterations' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'maxIterations', min_call_result_405974)
    
    # Getting the type of 'verbosityLevel' (line 312)
    verbosityLevel_405975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 7), 'verbosityLevel')
    # Testing the type of an if condition (line 312)
    if_condition_405976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 4), verbosityLevel_405975)
    # Assigning a type to the variable 'if_condition_405976' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'if_condition_405976', if_condition_405976)
    # SSA begins for if statement (line 312)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 313):
    
    # Assigning a Str to a Name (line 313):
    str_405977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 14), 'str', 'Solving ')
    # Assigning a type to the variable 'aux' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'aux', str_405977)
    
    # Type idiom detected: calculating its left and rigth part (line 314)
    # Getting the type of 'B' (line 314)
    B_405978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'B')
    # Getting the type of 'None' (line 314)
    None_405979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'None')
    
    (may_be_405980, more_types_in_union_405981) = may_be_none(B_405978, None_405979)

    if may_be_405980:

        if more_types_in_union_405981:
            # Runtime conditional SSA (line 314)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'aux' (line 315)
        aux_405982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'aux')
        str_405983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 19), 'str', 'standard')
        # Applying the binary operator '+=' (line 315)
        result_iadd_405984 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 12), '+=', aux_405982, str_405983)
        # Assigning a type to the variable 'aux' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'aux', result_iadd_405984)
        

        if more_types_in_union_405981:
            # Runtime conditional SSA for else branch (line 314)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_405980) or more_types_in_union_405981):
        
        # Getting the type of 'aux' (line 317)
        aux_405985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'aux')
        str_405986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 19), 'str', 'generalized')
        # Applying the binary operator '+=' (line 317)
        result_iadd_405987 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 12), '+=', aux_405985, str_405986)
        # Assigning a type to the variable 'aux' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'aux', result_iadd_405987)
        

        if (may_be_405980 and more_types_in_union_405981):
            # SSA join for if statement (line 314)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'aux' (line 318)
    aux_405988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'aux')
    str_405989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 15), 'str', ' eigenvalue problem with')
    # Applying the binary operator '+=' (line 318)
    result_iadd_405990 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 8), '+=', aux_405988, str_405989)
    # Assigning a type to the variable 'aux' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'aux', result_iadd_405990)
    
    
    # Type idiom detected: calculating its left and rigth part (line 319)
    # Getting the type of 'M' (line 319)
    M_405991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 11), 'M')
    # Getting the type of 'None' (line 319)
    None_405992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 16), 'None')
    
    (may_be_405993, more_types_in_union_405994) = may_be_none(M_405991, None_405992)

    if may_be_405993:

        if more_types_in_union_405994:
            # Runtime conditional SSA (line 319)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'aux' (line 320)
        aux_405995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'aux')
        str_405996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 19), 'str', 'out')
        # Applying the binary operator '+=' (line 320)
        result_iadd_405997 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 12), '+=', aux_405995, str_405996)
        # Assigning a type to the variable 'aux' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'aux', result_iadd_405997)
        

        if more_types_in_union_405994:
            # SSA join for if statement (line 319)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'aux' (line 321)
    aux_405998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'aux')
    str_405999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 15), 'str', ' preconditioning\n\n')
    # Applying the binary operator '+=' (line 321)
    result_iadd_406000 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 8), '+=', aux_405998, str_405999)
    # Assigning a type to the variable 'aux' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'aux', result_iadd_406000)
    
    
    # Getting the type of 'aux' (line 322)
    aux_406001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'aux')
    str_406002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 15), 'str', 'matrix size %d\n')
    # Getting the type of 'n' (line 322)
    n_406003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 36), 'n')
    # Applying the binary operator '%' (line 322)
    result_mod_406004 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 15), '%', str_406002, n_406003)
    
    # Applying the binary operator '+=' (line 322)
    result_iadd_406005 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 8), '+=', aux_406001, result_mod_406004)
    # Assigning a type to the variable 'aux' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'aux', result_iadd_406005)
    
    
    # Getting the type of 'aux' (line 323)
    aux_406006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'aux')
    str_406007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 15), 'str', 'block size %d\n\n')
    # Getting the type of 'sizeX' (line 323)
    sizeX_406008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 37), 'sizeX')
    # Applying the binary operator '%' (line 323)
    result_mod_406009 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 15), '%', str_406007, sizeX_406008)
    
    # Applying the binary operator '+=' (line 323)
    result_iadd_406010 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 8), '+=', aux_406006, result_mod_406009)
    # Assigning a type to the variable 'aux' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'aux', result_iadd_406010)
    
    
    # Type idiom detected: calculating its left and rigth part (line 324)
    # Getting the type of 'blockVectorY' (line 324)
    blockVectorY_406011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 11), 'blockVectorY')
    # Getting the type of 'None' (line 324)
    None_406012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 27), 'None')
    
    (may_be_406013, more_types_in_union_406014) = may_be_none(blockVectorY_406011, None_406012)

    if may_be_406013:

        if more_types_in_union_406014:
            # Runtime conditional SSA (line 324)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'aux' (line 325)
        aux_406015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'aux')
        str_406016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 19), 'str', 'No constraints\n\n')
        # Applying the binary operator '+=' (line 325)
        result_iadd_406017 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 12), '+=', aux_406015, str_406016)
        # Assigning a type to the variable 'aux' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'aux', result_iadd_406017)
        

        if more_types_in_union_406014:
            # Runtime conditional SSA for else branch (line 324)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_406013) or more_types_in_union_406014):
        
        
        # Getting the type of 'sizeY' (line 327)
        sizeY_406018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'sizeY')
        int_406019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 23), 'int')
        # Applying the binary operator '>' (line 327)
        result_gt_406020 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 15), '>', sizeY_406018, int_406019)
        
        # Testing the type of an if condition (line 327)
        if_condition_406021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 12), result_gt_406020)
        # Assigning a type to the variable 'if_condition_406021' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'if_condition_406021', if_condition_406021)
        # SSA begins for if statement (line 327)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'aux' (line 328)
        aux_406022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 'aux')
        str_406023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 23), 'str', '%d constraints\n\n')
        # Getting the type of 'sizeY' (line 328)
        sizeY_406024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 46), 'sizeY')
        # Applying the binary operator '%' (line 328)
        result_mod_406025 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 23), '%', str_406023, sizeY_406024)
        
        # Applying the binary operator '+=' (line 328)
        result_iadd_406026 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 16), '+=', aux_406022, result_mod_406025)
        # Assigning a type to the variable 'aux' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 'aux', result_iadd_406026)
        
        # SSA branch for the else part of an if statement (line 327)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'aux' (line 330)
        aux_406027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'aux')
        str_406028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 23), 'str', '%d constraint\n\n')
        # Getting the type of 'sizeY' (line 330)
        sizeY_406029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 45), 'sizeY')
        # Applying the binary operator '%' (line 330)
        result_mod_406030 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 23), '%', str_406028, sizeY_406029)
        
        # Applying the binary operator '+=' (line 330)
        result_iadd_406031 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 16), '+=', aux_406027, result_mod_406030)
        # Assigning a type to the variable 'aux' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'aux', result_iadd_406031)
        
        # SSA join for if statement (line 327)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_406013 and more_types_in_union_406014):
            # SSA join for if statement (line 324)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to print(...): (line 331)
    # Processing the call arguments (line 331)
    # Getting the type of 'aux' (line 331)
    aux_406033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 14), 'aux', False)
    # Processing the call keyword arguments (line 331)
    kwargs_406034 = {}
    # Getting the type of 'print' (line 331)
    print_406032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'print', False)
    # Calling print(args, kwargs) (line 331)
    print_call_result_406035 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), print_406032, *[aux_406033], **kwargs_406034)
    
    # SSA join for if statement (line 312)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 335)
    # Getting the type of 'blockVectorY' (line 335)
    blockVectorY_406036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'blockVectorY')
    # Getting the type of 'None' (line 335)
    None_406037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 27), 'None')
    
    (may_be_406038, more_types_in_union_406039) = may_not_be_none(blockVectorY_406036, None_406037)

    if may_be_406038:

        if more_types_in_union_406039:
            # Runtime conditional SSA (line 335)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 337)
        # Getting the type of 'B' (line 337)
        B_406040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'B')
        # Getting the type of 'None' (line 337)
        None_406041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'None')
        
        (may_be_406042, more_types_in_union_406043) = may_not_be_none(B_406040, None_406041)

        if may_be_406042:

            if more_types_in_union_406043:
                # Runtime conditional SSA (line 337)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 338):
            
            # Assigning a Call to a Name (line 338):
            
            # Call to B(...): (line 338)
            # Processing the call arguments (line 338)
            # Getting the type of 'blockVectorY' (line 338)
            blockVectorY_406045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 30), 'blockVectorY', False)
            # Processing the call keyword arguments (line 338)
            kwargs_406046 = {}
            # Getting the type of 'B' (line 338)
            B_406044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 28), 'B', False)
            # Calling B(args, kwargs) (line 338)
            B_call_result_406047 = invoke(stypy.reporting.localization.Localization(__file__, 338, 28), B_406044, *[blockVectorY_406045], **kwargs_406046)
            
            # Assigning a type to the variable 'blockVectorBY' (line 338)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'blockVectorBY', B_call_result_406047)

            if more_types_in_union_406043:
                # Runtime conditional SSA for else branch (line 337)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_406042) or more_types_in_union_406043):
            
            # Assigning a Name to a Name (line 340):
            
            # Assigning a Name to a Name (line 340):
            # Getting the type of 'blockVectorY' (line 340)
            blockVectorY_406048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 28), 'blockVectorY')
            # Assigning a type to the variable 'blockVectorBY' (line 340)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'blockVectorBY', blockVectorY_406048)

            if (may_be_406042 and more_types_in_union_406043):
                # SSA join for if statement (line 337)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to dot(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'blockVectorY' (line 343)
        blockVectorY_406051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 25), 'blockVectorY', False)
        # Obtaining the member 'T' of a type (line 343)
        T_406052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 25), blockVectorY_406051, 'T')
        # Getting the type of 'blockVectorBY' (line 343)
        blockVectorBY_406053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 41), 'blockVectorBY', False)
        # Processing the call keyword arguments (line 343)
        kwargs_406054 = {}
        # Getting the type of 'np' (line 343)
        np_406049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 18), 'np', False)
        # Obtaining the member 'dot' of a type (line 343)
        dot_406050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 18), np_406049, 'dot')
        # Calling dot(args, kwargs) (line 343)
        dot_call_result_406055 = invoke(stypy.reporting.localization.Localization(__file__, 343, 18), dot_406050, *[T_406052, blockVectorBY_406053], **kwargs_406054)
        
        # Assigning a type to the variable 'gramYBY' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'gramYBY', dot_call_result_406055)
        
        
        # SSA begins for try-except statement (line 344)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 346):
        
        # Assigning a Call to a Name (line 346):
        
        # Call to cho_factor(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'gramYBY' (line 346)
        gramYBY_406057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 33), 'gramYBY', False)
        # Processing the call keyword arguments (line 346)
        kwargs_406058 = {}
        # Getting the type of 'cho_factor' (line 346)
        cho_factor_406056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 22), 'cho_factor', False)
        # Calling cho_factor(args, kwargs) (line 346)
        cho_factor_call_result_406059 = invoke(stypy.reporting.localization.Localization(__file__, 346, 22), cho_factor_406056, *[gramYBY_406057], **kwargs_406058)
        
        # Assigning a type to the variable 'gramYBY' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'gramYBY', cho_factor_call_result_406059)
        # SSA branch for the except part of a try statement (line 344)
        # SSA branch for the except '<any exception>' branch of a try statement (line 344)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 348)
        # Processing the call arguments (line 348)
        str_406061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 29), 'str', 'cannot handle linearly dependent constraints')
        # Processing the call keyword arguments (line 348)
        kwargs_406062 = {}
        # Getting the type of 'ValueError' (line 348)
        ValueError_406060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 348)
        ValueError_call_result_406063 = invoke(stypy.reporting.localization.Localization(__file__, 348, 18), ValueError_406060, *[str_406061], **kwargs_406062)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 348, 12), ValueError_call_result_406063, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 344)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _applyConstraints(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'blockVectorX' (line 350)
        blockVectorX_406065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 26), 'blockVectorX', False)
        # Getting the type of 'gramYBY' (line 350)
        gramYBY_406066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 40), 'gramYBY', False)
        # Getting the type of 'blockVectorBY' (line 350)
        blockVectorBY_406067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 49), 'blockVectorBY', False)
        # Getting the type of 'blockVectorY' (line 350)
        blockVectorY_406068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 64), 'blockVectorY', False)
        # Processing the call keyword arguments (line 350)
        kwargs_406069 = {}
        # Getting the type of '_applyConstraints' (line 350)
        _applyConstraints_406064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), '_applyConstraints', False)
        # Calling _applyConstraints(args, kwargs) (line 350)
        _applyConstraints_call_result_406070 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), _applyConstraints_406064, *[blockVectorX_406065, gramYBY_406066, blockVectorBY_406067, blockVectorY_406068], **kwargs_406069)
        

        if more_types_in_union_406039:
            # SSA join for if statement (line 335)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 354):
    
    # Assigning a Subscript to a Name (line 354):
    
    # Obtaining the type of the subscript
    int_406071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 4), 'int')
    
    # Call to _b_orthonormalize(...): (line 354)
    # Processing the call arguments (line 354)
    # Getting the type of 'B' (line 354)
    B_406073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 52), 'B', False)
    # Getting the type of 'blockVectorX' (line 354)
    blockVectorX_406074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 55), 'blockVectorX', False)
    # Processing the call keyword arguments (line 354)
    kwargs_406075 = {}
    # Getting the type of '_b_orthonormalize' (line 354)
    _b_orthonormalize_406072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 34), '_b_orthonormalize', False)
    # Calling _b_orthonormalize(args, kwargs) (line 354)
    _b_orthonormalize_call_result_406076 = invoke(stypy.reporting.localization.Localization(__file__, 354, 34), _b_orthonormalize_406072, *[B_406073, blockVectorX_406074], **kwargs_406075)
    
    # Obtaining the member '__getitem__' of a type (line 354)
    getitem___406077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 4), _b_orthonormalize_call_result_406076, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 354)
    subscript_call_result_406078 = invoke(stypy.reporting.localization.Localization(__file__, 354, 4), getitem___406077, int_406071)
    
    # Assigning a type to the variable 'tuple_var_assignment_405635' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'tuple_var_assignment_405635', subscript_call_result_406078)
    
    # Assigning a Subscript to a Name (line 354):
    
    # Obtaining the type of the subscript
    int_406079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 4), 'int')
    
    # Call to _b_orthonormalize(...): (line 354)
    # Processing the call arguments (line 354)
    # Getting the type of 'B' (line 354)
    B_406081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 52), 'B', False)
    # Getting the type of 'blockVectorX' (line 354)
    blockVectorX_406082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 55), 'blockVectorX', False)
    # Processing the call keyword arguments (line 354)
    kwargs_406083 = {}
    # Getting the type of '_b_orthonormalize' (line 354)
    _b_orthonormalize_406080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 34), '_b_orthonormalize', False)
    # Calling _b_orthonormalize(args, kwargs) (line 354)
    _b_orthonormalize_call_result_406084 = invoke(stypy.reporting.localization.Localization(__file__, 354, 34), _b_orthonormalize_406080, *[B_406081, blockVectorX_406082], **kwargs_406083)
    
    # Obtaining the member '__getitem__' of a type (line 354)
    getitem___406085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 4), _b_orthonormalize_call_result_406084, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 354)
    subscript_call_result_406086 = invoke(stypy.reporting.localization.Localization(__file__, 354, 4), getitem___406085, int_406079)
    
    # Assigning a type to the variable 'tuple_var_assignment_405636' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'tuple_var_assignment_405636', subscript_call_result_406086)
    
    # Assigning a Name to a Name (line 354):
    # Getting the type of 'tuple_var_assignment_405635' (line 354)
    tuple_var_assignment_405635_406087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'tuple_var_assignment_405635')
    # Assigning a type to the variable 'blockVectorX' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'blockVectorX', tuple_var_assignment_405635_406087)
    
    # Assigning a Name to a Name (line 354):
    # Getting the type of 'tuple_var_assignment_405636' (line 354)
    tuple_var_assignment_405636_406088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'tuple_var_assignment_405636')
    # Assigning a type to the variable 'blockVectorBX' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 18), 'blockVectorBX', tuple_var_assignment_405636_406088)
    
    # Assigning a Call to a Name (line 358):
    
    # Assigning a Call to a Name (line 358):
    
    # Call to A(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'blockVectorX' (line 358)
    blockVectorX_406090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 22), 'blockVectorX', False)
    # Processing the call keyword arguments (line 358)
    kwargs_406091 = {}
    # Getting the type of 'A' (line 358)
    A_406089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 20), 'A', False)
    # Calling A(args, kwargs) (line 358)
    A_call_result_406092 = invoke(stypy.reporting.localization.Localization(__file__, 358, 20), A_406089, *[blockVectorX_406090], **kwargs_406091)
    
    # Assigning a type to the variable 'blockVectorAX' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'blockVectorAX', A_call_result_406092)
    
    # Assigning a Call to a Name (line 359):
    
    # Assigning a Call to a Name (line 359):
    
    # Call to dot(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of 'blockVectorX' (line 359)
    blockVectorX_406095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 21), 'blockVectorX', False)
    # Obtaining the member 'T' of a type (line 359)
    T_406096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 21), blockVectorX_406095, 'T')
    # Getting the type of 'blockVectorAX' (line 359)
    blockVectorAX_406097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 37), 'blockVectorAX', False)
    # Processing the call keyword arguments (line 359)
    kwargs_406098 = {}
    # Getting the type of 'np' (line 359)
    np_406093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 14), 'np', False)
    # Obtaining the member 'dot' of a type (line 359)
    dot_406094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 14), np_406093, 'dot')
    # Calling dot(args, kwargs) (line 359)
    dot_call_result_406099 = invoke(stypy.reporting.localization.Localization(__file__, 359, 14), dot_406094, *[T_406096, blockVectorAX_406097], **kwargs_406098)
    
    # Assigning a type to the variable 'gramXAX' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'gramXAX', dot_call_result_406099)
    
    # Assigning a Call to a Tuple (line 361):
    
    # Assigning a Subscript to a Name (line 361):
    
    # Obtaining the type of the subscript
    int_406100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 4), 'int')
    
    # Call to eigh(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'gramXAX' (line 361)
    gramXAX_406102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 35), 'gramXAX', False)
    # Processing the call keyword arguments (line 361)
    # Getting the type of 'False' (line 361)
    False_406103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 57), 'False', False)
    keyword_406104 = False_406103
    kwargs_406105 = {'check_finite': keyword_406104}
    # Getting the type of 'eigh' (line 361)
    eigh_406101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 30), 'eigh', False)
    # Calling eigh(args, kwargs) (line 361)
    eigh_call_result_406106 = invoke(stypy.reporting.localization.Localization(__file__, 361, 30), eigh_406101, *[gramXAX_406102], **kwargs_406105)
    
    # Obtaining the member '__getitem__' of a type (line 361)
    getitem___406107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 4), eigh_call_result_406106, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 361)
    subscript_call_result_406108 = invoke(stypy.reporting.localization.Localization(__file__, 361, 4), getitem___406107, int_406100)
    
    # Assigning a type to the variable 'tuple_var_assignment_405637' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'tuple_var_assignment_405637', subscript_call_result_406108)
    
    # Assigning a Subscript to a Name (line 361):
    
    # Obtaining the type of the subscript
    int_406109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 4), 'int')
    
    # Call to eigh(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'gramXAX' (line 361)
    gramXAX_406111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 35), 'gramXAX', False)
    # Processing the call keyword arguments (line 361)
    # Getting the type of 'False' (line 361)
    False_406112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 57), 'False', False)
    keyword_406113 = False_406112
    kwargs_406114 = {'check_finite': keyword_406113}
    # Getting the type of 'eigh' (line 361)
    eigh_406110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 30), 'eigh', False)
    # Calling eigh(args, kwargs) (line 361)
    eigh_call_result_406115 = invoke(stypy.reporting.localization.Localization(__file__, 361, 30), eigh_406110, *[gramXAX_406111], **kwargs_406114)
    
    # Obtaining the member '__getitem__' of a type (line 361)
    getitem___406116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 4), eigh_call_result_406115, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 361)
    subscript_call_result_406117 = invoke(stypy.reporting.localization.Localization(__file__, 361, 4), getitem___406116, int_406109)
    
    # Assigning a type to the variable 'tuple_var_assignment_405638' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'tuple_var_assignment_405638', subscript_call_result_406117)
    
    # Assigning a Name to a Name (line 361):
    # Getting the type of 'tuple_var_assignment_405637' (line 361)
    tuple_var_assignment_405637_406118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'tuple_var_assignment_405637')
    # Assigning a type to the variable '_lambda' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), '_lambda', tuple_var_assignment_405637_406118)
    
    # Assigning a Name to a Name (line 361):
    # Getting the type of 'tuple_var_assignment_405638' (line 361)
    tuple_var_assignment_405638_406119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'tuple_var_assignment_405638')
    # Assigning a type to the variable 'eigBlockVector' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 13), 'eigBlockVector', tuple_var_assignment_405638_406119)
    
    # Assigning a Subscript to a Name (line 362):
    
    # Assigning a Subscript to a Name (line 362):
    
    # Obtaining the type of the subscript
    # Getting the type of 'sizeX' (line 362)
    sizeX_406120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 30), 'sizeX')
    slice_406121 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 362, 9), None, sizeX_406120, None)
    
    # Call to argsort(...): (line 362)
    # Processing the call arguments (line 362)
    # Getting the type of '_lambda' (line 362)
    _lambda_406124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 20), '_lambda', False)
    # Processing the call keyword arguments (line 362)
    kwargs_406125 = {}
    # Getting the type of 'np' (line 362)
    np_406122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 9), 'np', False)
    # Obtaining the member 'argsort' of a type (line 362)
    argsort_406123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 9), np_406122, 'argsort')
    # Calling argsort(args, kwargs) (line 362)
    argsort_call_result_406126 = invoke(stypy.reporting.localization.Localization(__file__, 362, 9), argsort_406123, *[_lambda_406124], **kwargs_406125)
    
    # Obtaining the member '__getitem__' of a type (line 362)
    getitem___406127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 9), argsort_call_result_406126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 362)
    subscript_call_result_406128 = invoke(stypy.reporting.localization.Localization(__file__, 362, 9), getitem___406127, slice_406121)
    
    # Assigning a type to the variable 'ii' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'ii', subscript_call_result_406128)
    
    # Getting the type of 'largest' (line 363)
    largest_406129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 7), 'largest')
    # Testing the type of an if condition (line 363)
    if_condition_406130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 363, 4), largest_406129)
    # Assigning a type to the variable 'if_condition_406130' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'if_condition_406130', if_condition_406130)
    # SSA begins for if statement (line 363)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 364):
    
    # Assigning a Subscript to a Name (line 364):
    
    # Obtaining the type of the subscript
    int_406131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 18), 'int')
    slice_406132 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 364, 13), None, None, int_406131)
    # Getting the type of 'ii' (line 364)
    ii_406133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 13), 'ii')
    # Obtaining the member '__getitem__' of a type (line 364)
    getitem___406134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 13), ii_406133, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 364)
    subscript_call_result_406135 = invoke(stypy.reporting.localization.Localization(__file__, 364, 13), getitem___406134, slice_406132)
    
    # Assigning a type to the variable 'ii' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'ii', subscript_call_result_406135)
    # SSA join for if statement (line 363)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 365):
    
    # Assigning a Subscript to a Name (line 365):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 365)
    ii_406136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 22), 'ii')
    # Getting the type of '_lambda' (line 365)
    _lambda_406137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 14), '_lambda')
    # Obtaining the member '__getitem__' of a type (line 365)
    getitem___406138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 14), _lambda_406137, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 365)
    subscript_call_result_406139 = invoke(stypy.reporting.localization.Localization(__file__, 365, 14), getitem___406138, ii_406136)
    
    # Assigning a type to the variable '_lambda' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), '_lambda', subscript_call_result_406139)
    
    # Assigning a Call to a Name (line 367):
    
    # Assigning a Call to a Name (line 367):
    
    # Call to asarray(...): (line 367)
    # Processing the call arguments (line 367)
    
    # Obtaining the type of the subscript
    slice_406142 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 367, 32), None, None, None)
    # Getting the type of 'ii' (line 367)
    ii_406143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 49), 'ii', False)
    # Getting the type of 'eigBlockVector' (line 367)
    eigBlockVector_406144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 32), 'eigBlockVector', False)
    # Obtaining the member '__getitem__' of a type (line 367)
    getitem___406145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 32), eigBlockVector_406144, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 367)
    subscript_call_result_406146 = invoke(stypy.reporting.localization.Localization(__file__, 367, 32), getitem___406145, (slice_406142, ii_406143))
    
    # Processing the call keyword arguments (line 367)
    kwargs_406147 = {}
    # Getting the type of 'np' (line 367)
    np_406140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 21), 'np', False)
    # Obtaining the member 'asarray' of a type (line 367)
    asarray_406141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 21), np_406140, 'asarray')
    # Calling asarray(args, kwargs) (line 367)
    asarray_call_result_406148 = invoke(stypy.reporting.localization.Localization(__file__, 367, 21), asarray_406141, *[subscript_call_result_406146], **kwargs_406147)
    
    # Assigning a type to the variable 'eigBlockVector' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'eigBlockVector', asarray_call_result_406148)
    
    # Assigning a Call to a Name (line 368):
    
    # Assigning a Call to a Name (line 368):
    
    # Call to dot(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'blockVectorX' (line 368)
    blockVectorX_406151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 26), 'blockVectorX', False)
    # Getting the type of 'eigBlockVector' (line 368)
    eigBlockVector_406152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 40), 'eigBlockVector', False)
    # Processing the call keyword arguments (line 368)
    kwargs_406153 = {}
    # Getting the type of 'np' (line 368)
    np_406149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 19), 'np', False)
    # Obtaining the member 'dot' of a type (line 368)
    dot_406150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 19), np_406149, 'dot')
    # Calling dot(args, kwargs) (line 368)
    dot_call_result_406154 = invoke(stypy.reporting.localization.Localization(__file__, 368, 19), dot_406150, *[blockVectorX_406151, eigBlockVector_406152], **kwargs_406153)
    
    # Assigning a type to the variable 'blockVectorX' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'blockVectorX', dot_call_result_406154)
    
    # Assigning a Call to a Name (line 369):
    
    # Assigning a Call to a Name (line 369):
    
    # Call to dot(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'blockVectorAX' (line 369)
    blockVectorAX_406157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 27), 'blockVectorAX', False)
    # Getting the type of 'eigBlockVector' (line 369)
    eigBlockVector_406158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 42), 'eigBlockVector', False)
    # Processing the call keyword arguments (line 369)
    kwargs_406159 = {}
    # Getting the type of 'np' (line 369)
    np_406155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 20), 'np', False)
    # Obtaining the member 'dot' of a type (line 369)
    dot_406156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 20), np_406155, 'dot')
    # Calling dot(args, kwargs) (line 369)
    dot_call_result_406160 = invoke(stypy.reporting.localization.Localization(__file__, 369, 20), dot_406156, *[blockVectorAX_406157, eigBlockVector_406158], **kwargs_406159)
    
    # Assigning a type to the variable 'blockVectorAX' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'blockVectorAX', dot_call_result_406160)
    
    # Type idiom detected: calculating its left and rigth part (line 370)
    # Getting the type of 'B' (line 370)
    B_406161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'B')
    # Getting the type of 'None' (line 370)
    None_406162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'None')
    
    (may_be_406163, more_types_in_union_406164) = may_not_be_none(B_406161, None_406162)

    if may_be_406163:

        if more_types_in_union_406164:
            # Runtime conditional SSA (line 370)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 371):
        
        # Assigning a Call to a Name (line 371):
        
        # Call to dot(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'blockVectorBX' (line 371)
        blockVectorBX_406167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 31), 'blockVectorBX', False)
        # Getting the type of 'eigBlockVector' (line 371)
        eigBlockVector_406168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 46), 'eigBlockVector', False)
        # Processing the call keyword arguments (line 371)
        kwargs_406169 = {}
        # Getting the type of 'np' (line 371)
        np_406165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'np', False)
        # Obtaining the member 'dot' of a type (line 371)
        dot_406166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 24), np_406165, 'dot')
        # Calling dot(args, kwargs) (line 371)
        dot_call_result_406170 = invoke(stypy.reporting.localization.Localization(__file__, 371, 24), dot_406166, *[blockVectorBX_406167, eigBlockVector_406168], **kwargs_406169)
        
        # Assigning a type to the variable 'blockVectorBX' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'blockVectorBX', dot_call_result_406170)

        if more_types_in_union_406164:
            # SSA join for if statement (line 370)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 375):
    
    # Assigning a Call to a Name (line 375):
    
    # Call to ones(...): (line 375)
    # Processing the call arguments (line 375)
    
    # Obtaining an instance of the builtin type 'tuple' (line 375)
    tuple_406173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 375)
    # Adding element type (line 375)
    # Getting the type of 'sizeX' (line 375)
    sizeX_406174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 26), 'sizeX', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 26), tuple_406173, sizeX_406174)
    
    # Processing the call keyword arguments (line 375)
    # Getting the type of 'bool' (line 375)
    bool_406175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 41), 'bool', False)
    keyword_406176 = bool_406175
    kwargs_406177 = {'dtype': keyword_406176}
    # Getting the type of 'np' (line 375)
    np_406171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 17), 'np', False)
    # Obtaining the member 'ones' of a type (line 375)
    ones_406172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 17), np_406171, 'ones')
    # Calling ones(args, kwargs) (line 375)
    ones_call_result_406178 = invoke(stypy.reporting.localization.Localization(__file__, 375, 17), ones_406172, *[tuple_406173], **kwargs_406177)
    
    # Assigning a type to the variable 'activeMask' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'activeMask', ones_call_result_406178)
    
    # Assigning a List to a Name (line 377):
    
    # Assigning a List to a Name (line 377):
    
    # Obtaining an instance of the builtin type 'list' (line 377)
    list_406179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 377)
    # Adding element type (line 377)
    # Getting the type of '_lambda' (line 377)
    _lambda_406180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 21), '_lambda')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 20), list_406179, _lambda_406180)
    
    # Assigning a type to the variable 'lambdaHistory' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'lambdaHistory', list_406179)
    
    # Assigning a List to a Name (line 378):
    
    # Assigning a List to a Name (line 378):
    
    # Obtaining an instance of the builtin type 'list' (line 378)
    list_406181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 378)
    
    # Assigning a type to the variable 'residualNormsHistory' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'residualNormsHistory', list_406181)
    
    # Assigning a Name to a Name (line 380):
    
    # Assigning a Name to a Name (line 380):
    # Getting the type of 'sizeX' (line 380)
    sizeX_406182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 24), 'sizeX')
    # Assigning a type to the variable 'previousBlockSize' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'previousBlockSize', sizeX_406182)
    
    # Assigning a Call to a Name (line 381):
    
    # Assigning a Call to a Name (line 381):
    
    # Call to eye(...): (line 381)
    # Processing the call arguments (line 381)
    # Getting the type of 'sizeX' (line 381)
    sizeX_406185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 19), 'sizeX', False)
    # Processing the call keyword arguments (line 381)
    # Getting the type of 'A' (line 381)
    A_406186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 32), 'A', False)
    # Obtaining the member 'dtype' of a type (line 381)
    dtype_406187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 32), A_406186, 'dtype')
    keyword_406188 = dtype_406187
    kwargs_406189 = {'dtype': keyword_406188}
    # Getting the type of 'np' (line 381)
    np_406183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'np', False)
    # Obtaining the member 'eye' of a type (line 381)
    eye_406184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 12), np_406183, 'eye')
    # Calling eye(args, kwargs) (line 381)
    eye_call_result_406190 = invoke(stypy.reporting.localization.Localization(__file__, 381, 12), eye_406184, *[sizeX_406185], **kwargs_406189)
    
    # Assigning a type to the variable 'ident' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'ident', eye_call_result_406190)
    
    # Assigning a Call to a Name (line 382):
    
    # Assigning a Call to a Name (line 382):
    
    # Call to eye(...): (line 382)
    # Processing the call arguments (line 382)
    # Getting the type of 'sizeX' (line 382)
    sizeX_406193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 20), 'sizeX', False)
    # Processing the call keyword arguments (line 382)
    # Getting the type of 'A' (line 382)
    A_406194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 33), 'A', False)
    # Obtaining the member 'dtype' of a type (line 382)
    dtype_406195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 33), A_406194, 'dtype')
    keyword_406196 = dtype_406195
    kwargs_406197 = {'dtype': keyword_406196}
    # Getting the type of 'np' (line 382)
    np_406191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 13), 'np', False)
    # Obtaining the member 'eye' of a type (line 382)
    eye_406192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 13), np_406191, 'eye')
    # Calling eye(args, kwargs) (line 382)
    eye_call_result_406198 = invoke(stypy.reporting.localization.Localization(__file__, 382, 13), eye_406192, *[sizeX_406193], **kwargs_406197)
    
    # Assigning a type to the variable 'ident0' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'ident0', eye_call_result_406198)
    
    # Assigning a Name to a Name (line 387):
    
    # Assigning a Name to a Name (line 387):
    # Getting the type of 'None' (line 387)
    None_406199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 19), 'None')
    # Assigning a type to the variable 'blockVectorP' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'blockVectorP', None_406199)
    
    # Assigning a Name to a Name (line 388):
    
    # Assigning a Name to a Name (line 388):
    # Getting the type of 'None' (line 388)
    None_406200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 20), 'None')
    # Assigning a type to the variable 'blockVectorAP' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'blockVectorAP', None_406200)
    
    # Assigning a Name to a Name (line 389):
    
    # Assigning a Name to a Name (line 389):
    # Getting the type of 'None' (line 389)
    None_406201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 20), 'None')
    # Assigning a type to the variable 'blockVectorBP' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'blockVectorBP', None_406201)
    
    
    # Call to xrange(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'maxIterations' (line 391)
    maxIterations_406203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 34), 'maxIterations', False)
    # Processing the call keyword arguments (line 391)
    kwargs_406204 = {}
    # Getting the type of 'xrange' (line 391)
    xrange_406202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 27), 'xrange', False)
    # Calling xrange(args, kwargs) (line 391)
    xrange_call_result_406205 = invoke(stypy.reporting.localization.Localization(__file__, 391, 27), xrange_406202, *[maxIterations_406203], **kwargs_406204)
    
    # Testing the type of a for loop iterable (line 391)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 391, 4), xrange_call_result_406205)
    # Getting the type of the for loop variable (line 391)
    for_loop_var_406206 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 391, 4), xrange_call_result_406205)
    # Assigning a type to the variable 'iterationNumber' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'iterationNumber', for_loop_var_406206)
    # SSA begins for a for statement (line 391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'verbosityLevel' (line 392)
    verbosityLevel_406207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 11), 'verbosityLevel')
    int_406208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 28), 'int')
    # Applying the binary operator '>' (line 392)
    result_gt_406209 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 11), '>', verbosityLevel_406207, int_406208)
    
    # Testing the type of an if condition (line 392)
    if_condition_406210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 392, 8), result_gt_406209)
    # Assigning a type to the variable 'if_condition_406210' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'if_condition_406210', if_condition_406210)
    # SSA begins for if statement (line 392)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 393)
    # Processing the call arguments (line 393)
    str_406212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 18), 'str', 'iteration %d')
    # Getting the type of 'iterationNumber' (line 393)
    iterationNumber_406213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 35), 'iterationNumber', False)
    # Applying the binary operator '%' (line 393)
    result_mod_406214 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 18), '%', str_406212, iterationNumber_406213)
    
    # Processing the call keyword arguments (line 393)
    kwargs_406215 = {}
    # Getting the type of 'print' (line 393)
    print_406211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'print', False)
    # Calling print(args, kwargs) (line 393)
    print_call_result_406216 = invoke(stypy.reporting.localization.Localization(__file__, 393, 12), print_406211, *[result_mod_406214], **kwargs_406215)
    
    # SSA join for if statement (line 392)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 395):
    
    # Assigning a BinOp to a Name (line 395):
    # Getting the type of 'blockVectorBX' (line 395)
    blockVectorBX_406217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 14), 'blockVectorBX')
    
    # Obtaining the type of the subscript
    # Getting the type of 'np' (line 395)
    np_406218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 38), 'np')
    # Obtaining the member 'newaxis' of a type (line 395)
    newaxis_406219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 38), np_406218, 'newaxis')
    slice_406220 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 395, 30), None, None, None)
    # Getting the type of '_lambda' (line 395)
    _lambda_406221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 30), '_lambda')
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___406222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 30), _lambda_406221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 395)
    subscript_call_result_406223 = invoke(stypy.reporting.localization.Localization(__file__, 395, 30), getitem___406222, (newaxis_406219, slice_406220))
    
    # Applying the binary operator '*' (line 395)
    result_mul_406224 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 14), '*', blockVectorBX_406217, subscript_call_result_406223)
    
    # Assigning a type to the variable 'aux' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'aux', result_mul_406224)
    
    # Assigning a BinOp to a Name (line 396):
    
    # Assigning a BinOp to a Name (line 396):
    # Getting the type of 'blockVectorAX' (line 396)
    blockVectorAX_406225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 23), 'blockVectorAX')
    # Getting the type of 'aux' (line 396)
    aux_406226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 39), 'aux')
    # Applying the binary operator '-' (line 396)
    result_sub_406227 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 23), '-', blockVectorAX_406225, aux_406226)
    
    # Assigning a type to the variable 'blockVectorR' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'blockVectorR', result_sub_406227)
    
    # Assigning a Call to a Name (line 398):
    
    # Assigning a Call to a Name (line 398):
    
    # Call to sum(...): (line 398)
    # Processing the call arguments (line 398)
    
    # Call to conjugate(...): (line 398)
    # Processing the call keyword arguments (line 398)
    kwargs_406232 = {}
    # Getting the type of 'blockVectorR' (line 398)
    blockVectorR_406230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 21), 'blockVectorR', False)
    # Obtaining the member 'conjugate' of a type (line 398)
    conjugate_406231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 21), blockVectorR_406230, 'conjugate')
    # Calling conjugate(args, kwargs) (line 398)
    conjugate_call_result_406233 = invoke(stypy.reporting.localization.Localization(__file__, 398, 21), conjugate_406231, *[], **kwargs_406232)
    
    # Getting the type of 'blockVectorR' (line 398)
    blockVectorR_406234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 48), 'blockVectorR', False)
    # Applying the binary operator '*' (line 398)
    result_mul_406235 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 21), '*', conjugate_call_result_406233, blockVectorR_406234)
    
    int_406236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 62), 'int')
    # Processing the call keyword arguments (line 398)
    kwargs_406237 = {}
    # Getting the type of 'np' (line 398)
    np_406228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 14), 'np', False)
    # Obtaining the member 'sum' of a type (line 398)
    sum_406229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 14), np_406228, 'sum')
    # Calling sum(args, kwargs) (line 398)
    sum_call_result_406238 = invoke(stypy.reporting.localization.Localization(__file__, 398, 14), sum_406229, *[result_mul_406235, int_406236], **kwargs_406237)
    
    # Assigning a type to the variable 'aux' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'aux', sum_call_result_406238)
    
    # Assigning a Call to a Name (line 399):
    
    # Assigning a Call to a Name (line 399):
    
    # Call to sqrt(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'aux' (line 399)
    aux_406241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 32), 'aux', False)
    # Processing the call keyword arguments (line 399)
    kwargs_406242 = {}
    # Getting the type of 'np' (line 399)
    np_406239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 24), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 399)
    sqrt_406240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 24), np_406239, 'sqrt')
    # Calling sqrt(args, kwargs) (line 399)
    sqrt_call_result_406243 = invoke(stypy.reporting.localization.Localization(__file__, 399, 24), sqrt_406240, *[aux_406241], **kwargs_406242)
    
    # Assigning a type to the variable 'residualNorms' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'residualNorms', sqrt_call_result_406243)
    
    # Call to append(...): (line 401)
    # Processing the call arguments (line 401)
    # Getting the type of 'residualNorms' (line 401)
    residualNorms_406246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 36), 'residualNorms', False)
    # Processing the call keyword arguments (line 401)
    kwargs_406247 = {}
    # Getting the type of 'residualNormsHistory' (line 401)
    residualNormsHistory_406244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'residualNormsHistory', False)
    # Obtaining the member 'append' of a type (line 401)
    append_406245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), residualNormsHistory_406244, 'append')
    # Calling append(args, kwargs) (line 401)
    append_call_result_406248 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), append_406245, *[residualNorms_406246], **kwargs_406247)
    
    
    # Assigning a Call to a Name (line 403):
    
    # Assigning a Call to a Name (line 403):
    
    # Call to where(...): (line 403)
    # Processing the call arguments (line 403)
    
    # Getting the type of 'residualNorms' (line 403)
    residualNorms_406251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 22), 'residualNorms', False)
    # Getting the type of 'residualTolerance' (line 403)
    residualTolerance_406252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 38), 'residualTolerance', False)
    # Applying the binary operator '>' (line 403)
    result_gt_406253 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 22), '>', residualNorms_406251, residualTolerance_406252)
    
    # Getting the type of 'True' (line 403)
    True_406254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 57), 'True', False)
    # Getting the type of 'False' (line 403)
    False_406255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 63), 'False', False)
    # Processing the call keyword arguments (line 403)
    kwargs_406256 = {}
    # Getting the type of 'np' (line 403)
    np_406249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 13), 'np', False)
    # Obtaining the member 'where' of a type (line 403)
    where_406250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 13), np_406249, 'where')
    # Calling where(args, kwargs) (line 403)
    where_call_result_406257 = invoke(stypy.reporting.localization.Localization(__file__, 403, 13), where_406250, *[result_gt_406253, True_406254, False_406255], **kwargs_406256)
    
    # Assigning a type to the variable 'ii' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'ii', where_call_result_406257)
    
    # Assigning a BinOp to a Name (line 404):
    
    # Assigning a BinOp to a Name (line 404):
    # Getting the type of 'activeMask' (line 404)
    activeMask_406258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 21), 'activeMask')
    # Getting the type of 'ii' (line 404)
    ii_406259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 34), 'ii')
    # Applying the binary operator '&' (line 404)
    result_and__406260 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 21), '&', activeMask_406258, ii_406259)
    
    # Assigning a type to the variable 'activeMask' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'activeMask', result_and__406260)
    
    
    # Getting the type of 'verbosityLevel' (line 405)
    verbosityLevel_406261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 11), 'verbosityLevel')
    int_406262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 28), 'int')
    # Applying the binary operator '>' (line 405)
    result_gt_406263 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 11), '>', verbosityLevel_406261, int_406262)
    
    # Testing the type of an if condition (line 405)
    if_condition_406264 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 405, 8), result_gt_406263)
    # Assigning a type to the variable 'if_condition_406264' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'if_condition_406264', if_condition_406264)
    # SSA begins for if statement (line 405)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'activeMask' (line 406)
    activeMask_406266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 18), 'activeMask', False)
    # Processing the call keyword arguments (line 406)
    kwargs_406267 = {}
    # Getting the type of 'print' (line 406)
    print_406265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'print', False)
    # Calling print(args, kwargs) (line 406)
    print_call_result_406268 = invoke(stypy.reporting.localization.Localization(__file__, 406, 12), print_406265, *[activeMask_406266], **kwargs_406267)
    
    # SSA join for if statement (line 405)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 408):
    
    # Assigning a Call to a Name (line 408):
    
    # Call to sum(...): (line 408)
    # Processing the call keyword arguments (line 408)
    kwargs_406271 = {}
    # Getting the type of 'activeMask' (line 408)
    activeMask_406269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 27), 'activeMask', False)
    # Obtaining the member 'sum' of a type (line 408)
    sum_406270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 27), activeMask_406269, 'sum')
    # Calling sum(args, kwargs) (line 408)
    sum_call_result_406272 = invoke(stypy.reporting.localization.Localization(__file__, 408, 27), sum_406270, *[], **kwargs_406271)
    
    # Assigning a type to the variable 'currentBlockSize' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'currentBlockSize', sum_call_result_406272)
    
    
    # Getting the type of 'currentBlockSize' (line 409)
    currentBlockSize_406273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 11), 'currentBlockSize')
    # Getting the type of 'previousBlockSize' (line 409)
    previousBlockSize_406274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 31), 'previousBlockSize')
    # Applying the binary operator '!=' (line 409)
    result_ne_406275 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 11), '!=', currentBlockSize_406273, previousBlockSize_406274)
    
    # Testing the type of an if condition (line 409)
    if_condition_406276 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 409, 8), result_ne_406275)
    # Assigning a type to the variable 'if_condition_406276' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'if_condition_406276', if_condition_406276)
    # SSA begins for if statement (line 409)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 410):
    
    # Assigning a Name to a Name (line 410):
    # Getting the type of 'currentBlockSize' (line 410)
    currentBlockSize_406277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 32), 'currentBlockSize')
    # Assigning a type to the variable 'previousBlockSize' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'previousBlockSize', currentBlockSize_406277)
    
    # Assigning a Call to a Name (line 411):
    
    # Assigning a Call to a Name (line 411):
    
    # Call to eye(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'currentBlockSize' (line 411)
    currentBlockSize_406280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 27), 'currentBlockSize', False)
    # Processing the call keyword arguments (line 411)
    # Getting the type of 'A' (line 411)
    A_406281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 51), 'A', False)
    # Obtaining the member 'dtype' of a type (line 411)
    dtype_406282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 51), A_406281, 'dtype')
    keyword_406283 = dtype_406282
    kwargs_406284 = {'dtype': keyword_406283}
    # Getting the type of 'np' (line 411)
    np_406278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 20), 'np', False)
    # Obtaining the member 'eye' of a type (line 411)
    eye_406279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 20), np_406278, 'eye')
    # Calling eye(args, kwargs) (line 411)
    eye_call_result_406285 = invoke(stypy.reporting.localization.Localization(__file__, 411, 20), eye_406279, *[currentBlockSize_406280], **kwargs_406284)
    
    # Assigning a type to the variable 'ident' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'ident', eye_call_result_406285)
    # SSA join for if statement (line 409)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'currentBlockSize' (line 413)
    currentBlockSize_406286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 11), 'currentBlockSize')
    int_406287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 31), 'int')
    # Applying the binary operator '==' (line 413)
    result_eq_406288 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 11), '==', currentBlockSize_406286, int_406287)
    
    # Testing the type of an if condition (line 413)
    if_condition_406289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 8), result_eq_406288)
    # Assigning a type to the variable 'if_condition_406289' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'if_condition_406289', if_condition_406289)
    # SSA begins for if statement (line 413)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 413)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbosityLevel' (line 416)
    verbosityLevel_406290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 11), 'verbosityLevel')
    int_406291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 28), 'int')
    # Applying the binary operator '>' (line 416)
    result_gt_406292 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 11), '>', verbosityLevel_406290, int_406291)
    
    # Testing the type of an if condition (line 416)
    if_condition_406293 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 416, 8), result_gt_406292)
    # Assigning a type to the variable 'if_condition_406293' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'if_condition_406293', if_condition_406293)
    # SSA begins for if statement (line 416)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 417)
    # Processing the call arguments (line 417)
    str_406295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 18), 'str', 'current block size:')
    # Getting the type of 'currentBlockSize' (line 417)
    currentBlockSize_406296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 41), 'currentBlockSize', False)
    # Processing the call keyword arguments (line 417)
    kwargs_406297 = {}
    # Getting the type of 'print' (line 417)
    print_406294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'print', False)
    # Calling print(args, kwargs) (line 417)
    print_call_result_406298 = invoke(stypy.reporting.localization.Localization(__file__, 417, 12), print_406294, *[str_406295, currentBlockSize_406296], **kwargs_406297)
    
    
    # Call to print(...): (line 418)
    # Processing the call arguments (line 418)
    str_406300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 18), 'str', 'eigenvalue:')
    # Getting the type of '_lambda' (line 418)
    _lambda_406301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 33), '_lambda', False)
    # Processing the call keyword arguments (line 418)
    kwargs_406302 = {}
    # Getting the type of 'print' (line 418)
    print_406299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'print', False)
    # Calling print(args, kwargs) (line 418)
    print_call_result_406303 = invoke(stypy.reporting.localization.Localization(__file__, 418, 12), print_406299, *[str_406300, _lambda_406301], **kwargs_406302)
    
    
    # Call to print(...): (line 419)
    # Processing the call arguments (line 419)
    str_406305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 18), 'str', 'residual norms:')
    # Getting the type of 'residualNorms' (line 419)
    residualNorms_406306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 37), 'residualNorms', False)
    # Processing the call keyword arguments (line 419)
    kwargs_406307 = {}
    # Getting the type of 'print' (line 419)
    print_406304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'print', False)
    # Calling print(args, kwargs) (line 419)
    print_call_result_406308 = invoke(stypy.reporting.localization.Localization(__file__, 419, 12), print_406304, *[str_406305, residualNorms_406306], **kwargs_406307)
    
    # SSA join for if statement (line 416)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbosityLevel' (line 420)
    verbosityLevel_406309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 11), 'verbosityLevel')
    int_406310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 28), 'int')
    # Applying the binary operator '>' (line 420)
    result_gt_406311 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 11), '>', verbosityLevel_406309, int_406310)
    
    # Testing the type of an if condition (line 420)
    if_condition_406312 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 8), result_gt_406311)
    # Assigning a type to the variable 'if_condition_406312' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'if_condition_406312', if_condition_406312)
    # SSA begins for if statement (line 420)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 421)
    # Processing the call arguments (line 421)
    # Getting the type of 'eigBlockVector' (line 421)
    eigBlockVector_406314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 18), 'eigBlockVector', False)
    # Processing the call keyword arguments (line 421)
    kwargs_406315 = {}
    # Getting the type of 'print' (line 421)
    print_406313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'print', False)
    # Calling print(args, kwargs) (line 421)
    print_call_result_406316 = invoke(stypy.reporting.localization.Localization(__file__, 421, 12), print_406313, *[eigBlockVector_406314], **kwargs_406315)
    
    # SSA join for if statement (line 420)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 423):
    
    # Assigning a Call to a Name (line 423):
    
    # Call to as2d(...): (line 423)
    # Processing the call arguments (line 423)
    
    # Obtaining the type of the subscript
    slice_406318 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 423, 34), None, None, None)
    # Getting the type of 'activeMask' (line 423)
    activeMask_406319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 49), 'activeMask', False)
    # Getting the type of 'blockVectorR' (line 423)
    blockVectorR_406320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 34), 'blockVectorR', False)
    # Obtaining the member '__getitem__' of a type (line 423)
    getitem___406321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 34), blockVectorR_406320, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 423)
    subscript_call_result_406322 = invoke(stypy.reporting.localization.Localization(__file__, 423, 34), getitem___406321, (slice_406318, activeMask_406319))
    
    # Processing the call keyword arguments (line 423)
    kwargs_406323 = {}
    # Getting the type of 'as2d' (line 423)
    as2d_406317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 29), 'as2d', False)
    # Calling as2d(args, kwargs) (line 423)
    as2d_call_result_406324 = invoke(stypy.reporting.localization.Localization(__file__, 423, 29), as2d_406317, *[subscript_call_result_406322], **kwargs_406323)
    
    # Assigning a type to the variable 'activeBlockVectorR' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'activeBlockVectorR', as2d_call_result_406324)
    
    
    # Getting the type of 'iterationNumber' (line 425)
    iterationNumber_406325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 11), 'iterationNumber')
    int_406326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 29), 'int')
    # Applying the binary operator '>' (line 425)
    result_gt_406327 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 11), '>', iterationNumber_406325, int_406326)
    
    # Testing the type of an if condition (line 425)
    if_condition_406328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 8), result_gt_406327)
    # Assigning a type to the variable 'if_condition_406328' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'if_condition_406328', if_condition_406328)
    # SSA begins for if statement (line 425)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 426):
    
    # Assigning a Call to a Name (line 426):
    
    # Call to as2d(...): (line 426)
    # Processing the call arguments (line 426)
    
    # Obtaining the type of the subscript
    slice_406330 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 426, 38), None, None, None)
    # Getting the type of 'activeMask' (line 426)
    activeMask_406331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 53), 'activeMask', False)
    # Getting the type of 'blockVectorP' (line 426)
    blockVectorP_406332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 38), 'blockVectorP', False)
    # Obtaining the member '__getitem__' of a type (line 426)
    getitem___406333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 38), blockVectorP_406332, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 426)
    subscript_call_result_406334 = invoke(stypy.reporting.localization.Localization(__file__, 426, 38), getitem___406333, (slice_406330, activeMask_406331))
    
    # Processing the call keyword arguments (line 426)
    kwargs_406335 = {}
    # Getting the type of 'as2d' (line 426)
    as2d_406329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 33), 'as2d', False)
    # Calling as2d(args, kwargs) (line 426)
    as2d_call_result_406336 = invoke(stypy.reporting.localization.Localization(__file__, 426, 33), as2d_406329, *[subscript_call_result_406334], **kwargs_406335)
    
    # Assigning a type to the variable 'activeBlockVectorP' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'activeBlockVectorP', as2d_call_result_406336)
    
    # Assigning a Call to a Name (line 427):
    
    # Assigning a Call to a Name (line 427):
    
    # Call to as2d(...): (line 427)
    # Processing the call arguments (line 427)
    
    # Obtaining the type of the subscript
    slice_406338 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 427, 39), None, None, None)
    # Getting the type of 'activeMask' (line 427)
    activeMask_406339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 55), 'activeMask', False)
    # Getting the type of 'blockVectorAP' (line 427)
    blockVectorAP_406340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 39), 'blockVectorAP', False)
    # Obtaining the member '__getitem__' of a type (line 427)
    getitem___406341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 39), blockVectorAP_406340, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 427)
    subscript_call_result_406342 = invoke(stypy.reporting.localization.Localization(__file__, 427, 39), getitem___406341, (slice_406338, activeMask_406339))
    
    # Processing the call keyword arguments (line 427)
    kwargs_406343 = {}
    # Getting the type of 'as2d' (line 427)
    as2d_406337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 34), 'as2d', False)
    # Calling as2d(args, kwargs) (line 427)
    as2d_call_result_406344 = invoke(stypy.reporting.localization.Localization(__file__, 427, 34), as2d_406337, *[subscript_call_result_406342], **kwargs_406343)
    
    # Assigning a type to the variable 'activeBlockVectorAP' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'activeBlockVectorAP', as2d_call_result_406344)
    
    # Assigning a Call to a Name (line 428):
    
    # Assigning a Call to a Name (line 428):
    
    # Call to as2d(...): (line 428)
    # Processing the call arguments (line 428)
    
    # Obtaining the type of the subscript
    slice_406346 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 428, 39), None, None, None)
    # Getting the type of 'activeMask' (line 428)
    activeMask_406347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 55), 'activeMask', False)
    # Getting the type of 'blockVectorBP' (line 428)
    blockVectorBP_406348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 39), 'blockVectorBP', False)
    # Obtaining the member '__getitem__' of a type (line 428)
    getitem___406349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 39), blockVectorBP_406348, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 428)
    subscript_call_result_406350 = invoke(stypy.reporting.localization.Localization(__file__, 428, 39), getitem___406349, (slice_406346, activeMask_406347))
    
    # Processing the call keyword arguments (line 428)
    kwargs_406351 = {}
    # Getting the type of 'as2d' (line 428)
    as2d_406345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 34), 'as2d', False)
    # Calling as2d(args, kwargs) (line 428)
    as2d_call_result_406352 = invoke(stypy.reporting.localization.Localization(__file__, 428, 34), as2d_406345, *[subscript_call_result_406350], **kwargs_406351)
    
    # Assigning a type to the variable 'activeBlockVectorBP' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'activeBlockVectorBP', as2d_call_result_406352)
    # SSA join for if statement (line 425)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 430)
    # Getting the type of 'M' (line 430)
    M_406353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'M')
    # Getting the type of 'None' (line 430)
    None_406354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 20), 'None')
    
    (may_be_406355, more_types_in_union_406356) = may_not_be_none(M_406353, None_406354)

    if may_be_406355:

        if more_types_in_union_406356:
            # Runtime conditional SSA (line 430)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 432):
        
        # Assigning a Call to a Name (line 432):
        
        # Call to M(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'activeBlockVectorR' (line 432)
        activeBlockVectorR_406358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 35), 'activeBlockVectorR', False)
        # Processing the call keyword arguments (line 432)
        kwargs_406359 = {}
        # Getting the type of 'M' (line 432)
        M_406357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 33), 'M', False)
        # Calling M(args, kwargs) (line 432)
        M_call_result_406360 = invoke(stypy.reporting.localization.Localization(__file__, 432, 33), M_406357, *[activeBlockVectorR_406358], **kwargs_406359)
        
        # Assigning a type to the variable 'activeBlockVectorR' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'activeBlockVectorR', M_call_result_406360)

        if more_types_in_union_406356:
            # SSA join for if statement (line 430)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 436)
    # Getting the type of 'blockVectorY' (line 436)
    blockVectorY_406361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'blockVectorY')
    # Getting the type of 'None' (line 436)
    None_406362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 31), 'None')
    
    (may_be_406363, more_types_in_union_406364) = may_not_be_none(blockVectorY_406361, None_406362)

    if may_be_406363:

        if more_types_in_union_406364:
            # Runtime conditional SSA (line 436)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to _applyConstraints(...): (line 437)
        # Processing the call arguments (line 437)
        # Getting the type of 'activeBlockVectorR' (line 437)
        activeBlockVectorR_406366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 30), 'activeBlockVectorR', False)
        # Getting the type of 'gramYBY' (line 438)
        gramYBY_406367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 30), 'gramYBY', False)
        # Getting the type of 'blockVectorBY' (line 438)
        blockVectorBY_406368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 39), 'blockVectorBY', False)
        # Getting the type of 'blockVectorY' (line 438)
        blockVectorY_406369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 54), 'blockVectorY', False)
        # Processing the call keyword arguments (line 437)
        kwargs_406370 = {}
        # Getting the type of '_applyConstraints' (line 437)
        _applyConstraints_406365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), '_applyConstraints', False)
        # Calling _applyConstraints(args, kwargs) (line 437)
        _applyConstraints_call_result_406371 = invoke(stypy.reporting.localization.Localization(__file__, 437, 12), _applyConstraints_406365, *[activeBlockVectorR_406366, gramYBY_406367, blockVectorBY_406368, blockVectorY_406369], **kwargs_406370)
        

        if more_types_in_union_406364:
            # SSA join for if statement (line 436)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 443):
    
    # Assigning a Call to a Name (line 443):
    
    # Call to _b_orthonormalize(...): (line 443)
    # Processing the call arguments (line 443)
    # Getting the type of 'B' (line 443)
    B_406373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 32), 'B', False)
    # Getting the type of 'activeBlockVectorR' (line 443)
    activeBlockVectorR_406374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 35), 'activeBlockVectorR', False)
    # Processing the call keyword arguments (line 443)
    kwargs_406375 = {}
    # Getting the type of '_b_orthonormalize' (line 443)
    _b_orthonormalize_406372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 14), '_b_orthonormalize', False)
    # Calling _b_orthonormalize(args, kwargs) (line 443)
    _b_orthonormalize_call_result_406376 = invoke(stypy.reporting.localization.Localization(__file__, 443, 14), _b_orthonormalize_406372, *[B_406373, activeBlockVectorR_406374], **kwargs_406375)
    
    # Assigning a type to the variable 'aux' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'aux', _b_orthonormalize_call_result_406376)
    
    # Assigning a Name to a Tuple (line 444):
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_406377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 8), 'int')
    # Getting the type of 'aux' (line 444)
    aux_406378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 50), 'aux')
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___406379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), aux_406378, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_406380 = invoke(stypy.reporting.localization.Localization(__file__, 444, 8), getitem___406379, int_406377)
    
    # Assigning a type to the variable 'tuple_var_assignment_405639' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'tuple_var_assignment_405639', subscript_call_result_406380)
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_406381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 8), 'int')
    # Getting the type of 'aux' (line 444)
    aux_406382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 50), 'aux')
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___406383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), aux_406382, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_406384 = invoke(stypy.reporting.localization.Localization(__file__, 444, 8), getitem___406383, int_406381)
    
    # Assigning a type to the variable 'tuple_var_assignment_405640' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'tuple_var_assignment_405640', subscript_call_result_406384)
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_405639' (line 444)
    tuple_var_assignment_405639_406385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'tuple_var_assignment_405639')
    # Assigning a type to the variable 'activeBlockVectorR' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'activeBlockVectorR', tuple_var_assignment_405639_406385)
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_405640' (line 444)
    tuple_var_assignment_405640_406386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'tuple_var_assignment_405640')
    # Assigning a type to the variable 'activeBlockVectorBR' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 28), 'activeBlockVectorBR', tuple_var_assignment_405640_406386)
    
    # Assigning a Call to a Name (line 446):
    
    # Assigning a Call to a Name (line 446):
    
    # Call to A(...): (line 446)
    # Processing the call arguments (line 446)
    # Getting the type of 'activeBlockVectorR' (line 446)
    activeBlockVectorR_406388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 32), 'activeBlockVectorR', False)
    # Processing the call keyword arguments (line 446)
    kwargs_406389 = {}
    # Getting the type of 'A' (line 446)
    A_406387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 30), 'A', False)
    # Calling A(args, kwargs) (line 446)
    A_call_result_406390 = invoke(stypy.reporting.localization.Localization(__file__, 446, 30), A_406387, *[activeBlockVectorR_406388], **kwargs_406389)
    
    # Assigning a type to the variable 'activeBlockVectorAR' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'activeBlockVectorAR', A_call_result_406390)
    
    
    # Getting the type of 'iterationNumber' (line 448)
    iterationNumber_406391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 11), 'iterationNumber')
    int_406392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 29), 'int')
    # Applying the binary operator '>' (line 448)
    result_gt_406393 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 11), '>', iterationNumber_406391, int_406392)
    
    # Testing the type of an if condition (line 448)
    if_condition_406394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 448, 8), result_gt_406393)
    # Assigning a type to the variable 'if_condition_406394' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'if_condition_406394', if_condition_406394)
    # SSA begins for if statement (line 448)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 449):
    
    # Assigning a Call to a Name (line 449):
    
    # Call to _b_orthonormalize(...): (line 449)
    # Processing the call arguments (line 449)
    # Getting the type of 'B' (line 449)
    B_406396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 36), 'B', False)
    # Getting the type of 'activeBlockVectorP' (line 449)
    activeBlockVectorP_406397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 39), 'activeBlockVectorP', False)
    # Getting the type of 'activeBlockVectorBP' (line 450)
    activeBlockVectorBP_406398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 36), 'activeBlockVectorBP', False)
    # Processing the call keyword arguments (line 449)
    # Getting the type of 'True' (line 450)
    True_406399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 65), 'True', False)
    keyword_406400 = True_406399
    kwargs_406401 = {'retInvR': keyword_406400}
    # Getting the type of '_b_orthonormalize' (line 449)
    _b_orthonormalize_406395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 18), '_b_orthonormalize', False)
    # Calling _b_orthonormalize(args, kwargs) (line 449)
    _b_orthonormalize_call_result_406402 = invoke(stypy.reporting.localization.Localization(__file__, 449, 18), _b_orthonormalize_406395, *[B_406396, activeBlockVectorP_406397, activeBlockVectorBP_406398], **kwargs_406401)
    
    # Assigning a type to the variable 'aux' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'aux', _b_orthonormalize_call_result_406402)
    
    # Assigning a Name to a Tuple (line 451):
    
    # Assigning a Subscript to a Name (line 451):
    
    # Obtaining the type of the subscript
    int_406403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 12), 'int')
    # Getting the type of 'aux' (line 451)
    aux_406404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 60), 'aux')
    # Obtaining the member '__getitem__' of a type (line 451)
    getitem___406405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 12), aux_406404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 451)
    subscript_call_result_406406 = invoke(stypy.reporting.localization.Localization(__file__, 451, 12), getitem___406405, int_406403)
    
    # Assigning a type to the variable 'tuple_var_assignment_405641' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'tuple_var_assignment_405641', subscript_call_result_406406)
    
    # Assigning a Subscript to a Name (line 451):
    
    # Obtaining the type of the subscript
    int_406407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 12), 'int')
    # Getting the type of 'aux' (line 451)
    aux_406408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 60), 'aux')
    # Obtaining the member '__getitem__' of a type (line 451)
    getitem___406409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 12), aux_406408, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 451)
    subscript_call_result_406410 = invoke(stypy.reporting.localization.Localization(__file__, 451, 12), getitem___406409, int_406407)
    
    # Assigning a type to the variable 'tuple_var_assignment_405642' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'tuple_var_assignment_405642', subscript_call_result_406410)
    
    # Assigning a Subscript to a Name (line 451):
    
    # Obtaining the type of the subscript
    int_406411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 12), 'int')
    # Getting the type of 'aux' (line 451)
    aux_406412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 60), 'aux')
    # Obtaining the member '__getitem__' of a type (line 451)
    getitem___406413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 12), aux_406412, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 451)
    subscript_call_result_406414 = invoke(stypy.reporting.localization.Localization(__file__, 451, 12), getitem___406413, int_406411)
    
    # Assigning a type to the variable 'tuple_var_assignment_405643' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'tuple_var_assignment_405643', subscript_call_result_406414)
    
    # Assigning a Name to a Name (line 451):
    # Getting the type of 'tuple_var_assignment_405641' (line 451)
    tuple_var_assignment_405641_406415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'tuple_var_assignment_405641')
    # Assigning a type to the variable 'activeBlockVectorP' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'activeBlockVectorP', tuple_var_assignment_405641_406415)
    
    # Assigning a Name to a Name (line 451):
    # Getting the type of 'tuple_var_assignment_405642' (line 451)
    tuple_var_assignment_405642_406416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'tuple_var_assignment_405642')
    # Assigning a type to the variable 'activeBlockVectorBP' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 32), 'activeBlockVectorBP', tuple_var_assignment_405642_406416)
    
    # Assigning a Name to a Name (line 451):
    # Getting the type of 'tuple_var_assignment_405643' (line 451)
    tuple_var_assignment_405643_406417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'tuple_var_assignment_405643')
    # Assigning a type to the variable 'invR' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 53), 'invR', tuple_var_assignment_405643_406417)
    
    # Assigning a Call to a Name (line 452):
    
    # Assigning a Call to a Name (line 452):
    
    # Call to dot(...): (line 452)
    # Processing the call arguments (line 452)
    # Getting the type of 'activeBlockVectorAP' (line 452)
    activeBlockVectorAP_406420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 41), 'activeBlockVectorAP', False)
    # Getting the type of 'invR' (line 452)
    invR_406421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 62), 'invR', False)
    # Processing the call keyword arguments (line 452)
    kwargs_406422 = {}
    # Getting the type of 'np' (line 452)
    np_406418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 34), 'np', False)
    # Obtaining the member 'dot' of a type (line 452)
    dot_406419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 34), np_406418, 'dot')
    # Calling dot(args, kwargs) (line 452)
    dot_call_result_406423 = invoke(stypy.reporting.localization.Localization(__file__, 452, 34), dot_406419, *[activeBlockVectorAP_406420, invR_406421], **kwargs_406422)
    
    # Assigning a type to the variable 'activeBlockVectorAP' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'activeBlockVectorAP', dot_call_result_406423)
    # SSA join for if statement (line 448)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 458):
    
    # Assigning a Call to a Name (line 458):
    
    # Call to dot(...): (line 458)
    # Processing the call arguments (line 458)
    # Getting the type of 'blockVectorX' (line 458)
    blockVectorX_406426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 21), 'blockVectorX', False)
    # Obtaining the member 'T' of a type (line 458)
    T_406427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 21), blockVectorX_406426, 'T')
    # Getting the type of 'activeBlockVectorAR' (line 458)
    activeBlockVectorAR_406428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 37), 'activeBlockVectorAR', False)
    # Processing the call keyword arguments (line 458)
    kwargs_406429 = {}
    # Getting the type of 'np' (line 458)
    np_406424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 14), 'np', False)
    # Obtaining the member 'dot' of a type (line 458)
    dot_406425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 14), np_406424, 'dot')
    # Calling dot(args, kwargs) (line 458)
    dot_call_result_406430 = invoke(stypy.reporting.localization.Localization(__file__, 458, 14), dot_406425, *[T_406427, activeBlockVectorAR_406428], **kwargs_406429)
    
    # Assigning a type to the variable 'xaw' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'xaw', dot_call_result_406430)
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Call to dot(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'activeBlockVectorR' (line 459)
    activeBlockVectorR_406433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 21), 'activeBlockVectorR', False)
    # Obtaining the member 'T' of a type (line 459)
    T_406434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 21), activeBlockVectorR_406433, 'T')
    # Getting the type of 'activeBlockVectorAR' (line 459)
    activeBlockVectorAR_406435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 43), 'activeBlockVectorAR', False)
    # Processing the call keyword arguments (line 459)
    kwargs_406436 = {}
    # Getting the type of 'np' (line 459)
    np_406431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 14), 'np', False)
    # Obtaining the member 'dot' of a type (line 459)
    dot_406432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 14), np_406431, 'dot')
    # Calling dot(args, kwargs) (line 459)
    dot_call_result_406437 = invoke(stypy.reporting.localization.Localization(__file__, 459, 14), dot_406432, *[T_406434, activeBlockVectorAR_406435], **kwargs_406436)
    
    # Assigning a type to the variable 'waw' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'waw', dot_call_result_406437)
    
    # Assigning a Call to a Name (line 460):
    
    # Assigning a Call to a Name (line 460):
    
    # Call to dot(...): (line 460)
    # Processing the call arguments (line 460)
    # Getting the type of 'blockVectorX' (line 460)
    blockVectorX_406440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 21), 'blockVectorX', False)
    # Obtaining the member 'T' of a type (line 460)
    T_406441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 21), blockVectorX_406440, 'T')
    # Getting the type of 'activeBlockVectorBR' (line 460)
    activeBlockVectorBR_406442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 37), 'activeBlockVectorBR', False)
    # Processing the call keyword arguments (line 460)
    kwargs_406443 = {}
    # Getting the type of 'np' (line 460)
    np_406438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 14), 'np', False)
    # Obtaining the member 'dot' of a type (line 460)
    dot_406439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 14), np_406438, 'dot')
    # Calling dot(args, kwargs) (line 460)
    dot_call_result_406444 = invoke(stypy.reporting.localization.Localization(__file__, 460, 14), dot_406439, *[T_406441, activeBlockVectorBR_406442], **kwargs_406443)
    
    # Assigning a type to the variable 'xbw' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'xbw', dot_call_result_406444)
    
    
    # Getting the type of 'iterationNumber' (line 462)
    iterationNumber_406445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 11), 'iterationNumber')
    int_406446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 29), 'int')
    # Applying the binary operator '>' (line 462)
    result_gt_406447 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 11), '>', iterationNumber_406445, int_406446)
    
    # Testing the type of an if condition (line 462)
    if_condition_406448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 8), result_gt_406447)
    # Assigning a type to the variable 'if_condition_406448' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'if_condition_406448', if_condition_406448)
    # SSA begins for if statement (line 462)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 463):
    
    # Assigning a Call to a Name (line 463):
    
    # Call to dot(...): (line 463)
    # Processing the call arguments (line 463)
    # Getting the type of 'blockVectorX' (line 463)
    blockVectorX_406451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 25), 'blockVectorX', False)
    # Obtaining the member 'T' of a type (line 463)
    T_406452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 25), blockVectorX_406451, 'T')
    # Getting the type of 'activeBlockVectorAP' (line 463)
    activeBlockVectorAP_406453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 41), 'activeBlockVectorAP', False)
    # Processing the call keyword arguments (line 463)
    kwargs_406454 = {}
    # Getting the type of 'np' (line 463)
    np_406449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 463)
    dot_406450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 18), np_406449, 'dot')
    # Calling dot(args, kwargs) (line 463)
    dot_call_result_406455 = invoke(stypy.reporting.localization.Localization(__file__, 463, 18), dot_406450, *[T_406452, activeBlockVectorAP_406453], **kwargs_406454)
    
    # Assigning a type to the variable 'xap' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'xap', dot_call_result_406455)
    
    # Assigning a Call to a Name (line 464):
    
    # Assigning a Call to a Name (line 464):
    
    # Call to dot(...): (line 464)
    # Processing the call arguments (line 464)
    # Getting the type of 'activeBlockVectorR' (line 464)
    activeBlockVectorR_406458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 25), 'activeBlockVectorR', False)
    # Obtaining the member 'T' of a type (line 464)
    T_406459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 25), activeBlockVectorR_406458, 'T')
    # Getting the type of 'activeBlockVectorAP' (line 464)
    activeBlockVectorAP_406460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 47), 'activeBlockVectorAP', False)
    # Processing the call keyword arguments (line 464)
    kwargs_406461 = {}
    # Getting the type of 'np' (line 464)
    np_406456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 464)
    dot_406457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 18), np_406456, 'dot')
    # Calling dot(args, kwargs) (line 464)
    dot_call_result_406462 = invoke(stypy.reporting.localization.Localization(__file__, 464, 18), dot_406457, *[T_406459, activeBlockVectorAP_406460], **kwargs_406461)
    
    # Assigning a type to the variable 'wap' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'wap', dot_call_result_406462)
    
    # Assigning a Call to a Name (line 465):
    
    # Assigning a Call to a Name (line 465):
    
    # Call to dot(...): (line 465)
    # Processing the call arguments (line 465)
    # Getting the type of 'activeBlockVectorP' (line 465)
    activeBlockVectorP_406465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 25), 'activeBlockVectorP', False)
    # Obtaining the member 'T' of a type (line 465)
    T_406466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 25), activeBlockVectorP_406465, 'T')
    # Getting the type of 'activeBlockVectorAP' (line 465)
    activeBlockVectorAP_406467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 47), 'activeBlockVectorAP', False)
    # Processing the call keyword arguments (line 465)
    kwargs_406468 = {}
    # Getting the type of 'np' (line 465)
    np_406463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 465)
    dot_406464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 18), np_406463, 'dot')
    # Calling dot(args, kwargs) (line 465)
    dot_call_result_406469 = invoke(stypy.reporting.localization.Localization(__file__, 465, 18), dot_406464, *[T_406466, activeBlockVectorAP_406467], **kwargs_406468)
    
    # Assigning a type to the variable 'pap' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'pap', dot_call_result_406469)
    
    # Assigning a Call to a Name (line 466):
    
    # Assigning a Call to a Name (line 466):
    
    # Call to dot(...): (line 466)
    # Processing the call arguments (line 466)
    # Getting the type of 'blockVectorX' (line 466)
    blockVectorX_406472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 25), 'blockVectorX', False)
    # Obtaining the member 'T' of a type (line 466)
    T_406473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 25), blockVectorX_406472, 'T')
    # Getting the type of 'activeBlockVectorBP' (line 466)
    activeBlockVectorBP_406474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 41), 'activeBlockVectorBP', False)
    # Processing the call keyword arguments (line 466)
    kwargs_406475 = {}
    # Getting the type of 'np' (line 466)
    np_406470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 466)
    dot_406471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 18), np_406470, 'dot')
    # Calling dot(args, kwargs) (line 466)
    dot_call_result_406476 = invoke(stypy.reporting.localization.Localization(__file__, 466, 18), dot_406471, *[T_406473, activeBlockVectorBP_406474], **kwargs_406475)
    
    # Assigning a type to the variable 'xbp' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'xbp', dot_call_result_406476)
    
    # Assigning a Call to a Name (line 467):
    
    # Assigning a Call to a Name (line 467):
    
    # Call to dot(...): (line 467)
    # Processing the call arguments (line 467)
    # Getting the type of 'activeBlockVectorR' (line 467)
    activeBlockVectorR_406479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 25), 'activeBlockVectorR', False)
    # Obtaining the member 'T' of a type (line 467)
    T_406480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 25), activeBlockVectorR_406479, 'T')
    # Getting the type of 'activeBlockVectorBP' (line 467)
    activeBlockVectorBP_406481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 47), 'activeBlockVectorBP', False)
    # Processing the call keyword arguments (line 467)
    kwargs_406482 = {}
    # Getting the type of 'np' (line 467)
    np_406477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 467)
    dot_406478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 18), np_406477, 'dot')
    # Calling dot(args, kwargs) (line 467)
    dot_call_result_406483 = invoke(stypy.reporting.localization.Localization(__file__, 467, 18), dot_406478, *[T_406480, activeBlockVectorBP_406481], **kwargs_406482)
    
    # Assigning a type to the variable 'wbp' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'wbp', dot_call_result_406483)
    
    # Assigning a Call to a Name (line 469):
    
    # Assigning a Call to a Name (line 469):
    
    # Call to bmat(...): (line 469)
    # Processing the call arguments (line 469)
    
    # Obtaining an instance of the builtin type 'list' (line 469)
    list_406486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 469)
    # Adding element type (line 469)
    
    # Obtaining an instance of the builtin type 'list' (line 469)
    list_406487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 469)
    # Adding element type (line 469)
    
    # Call to diag(...): (line 469)
    # Processing the call arguments (line 469)
    # Getting the type of '_lambda' (line 469)
    _lambda_406490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 38), '_lambda', False)
    # Processing the call keyword arguments (line 469)
    kwargs_406491 = {}
    # Getting the type of 'np' (line 469)
    np_406488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 30), 'np', False)
    # Obtaining the member 'diag' of a type (line 469)
    diag_406489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 30), np_406488, 'diag')
    # Calling diag(args, kwargs) (line 469)
    diag_call_result_406492 = invoke(stypy.reporting.localization.Localization(__file__, 469, 30), diag_406489, *[_lambda_406490], **kwargs_406491)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 29), list_406487, diag_call_result_406492)
    # Adding element type (line 469)
    # Getting the type of 'xaw' (line 469)
    xaw_406493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 48), 'xaw', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 29), list_406487, xaw_406493)
    # Adding element type (line 469)
    # Getting the type of 'xap' (line 469)
    xap_406494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 53), 'xap', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 29), list_406487, xap_406494)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 28), list_406486, list_406487)
    # Adding element type (line 469)
    
    # Obtaining an instance of the builtin type 'list' (line 470)
    list_406495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 470)
    # Adding element type (line 470)
    # Getting the type of 'xaw' (line 470)
    xaw_406496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 31), 'xaw', False)
    # Obtaining the member 'T' of a type (line 470)
    T_406497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 31), xaw_406496, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 30), list_406495, T_406497)
    # Adding element type (line 470)
    # Getting the type of 'waw' (line 470)
    waw_406498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 38), 'waw', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 30), list_406495, waw_406498)
    # Adding element type (line 470)
    # Getting the type of 'wap' (line 470)
    wap_406499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 43), 'wap', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 30), list_406495, wap_406499)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 28), list_406486, list_406495)
    # Adding element type (line 469)
    
    # Obtaining an instance of the builtin type 'list' (line 471)
    list_406500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 471)
    # Adding element type (line 471)
    # Getting the type of 'xap' (line 471)
    xap_406501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 31), 'xap', False)
    # Obtaining the member 'T' of a type (line 471)
    T_406502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 31), xap_406501, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 30), list_406500, T_406502)
    # Adding element type (line 471)
    # Getting the type of 'wap' (line 471)
    wap_406503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 38), 'wap', False)
    # Obtaining the member 'T' of a type (line 471)
    T_406504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 38), wap_406503, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 30), list_406500, T_406504)
    # Adding element type (line 471)
    # Getting the type of 'pap' (line 471)
    pap_406505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 45), 'pap', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 30), list_406500, pap_406505)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 28), list_406486, list_406500)
    
    # Processing the call keyword arguments (line 469)
    kwargs_406506 = {}
    # Getting the type of 'np' (line 469)
    np_406484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 20), 'np', False)
    # Obtaining the member 'bmat' of a type (line 469)
    bmat_406485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 20), np_406484, 'bmat')
    # Calling bmat(args, kwargs) (line 469)
    bmat_call_result_406507 = invoke(stypy.reporting.localization.Localization(__file__, 469, 20), bmat_406485, *[list_406486], **kwargs_406506)
    
    # Assigning a type to the variable 'gramA' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'gramA', bmat_call_result_406507)
    
    # Assigning a Call to a Name (line 473):
    
    # Assigning a Call to a Name (line 473):
    
    # Call to bmat(...): (line 473)
    # Processing the call arguments (line 473)
    
    # Obtaining an instance of the builtin type 'list' (line 473)
    list_406510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 473)
    # Adding element type (line 473)
    
    # Obtaining an instance of the builtin type 'list' (line 473)
    list_406511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 473)
    # Adding element type (line 473)
    # Getting the type of 'ident0' (line 473)
    ident0_406512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 30), 'ident0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 29), list_406511, ident0_406512)
    # Adding element type (line 473)
    # Getting the type of 'xbw' (line 473)
    xbw_406513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 38), 'xbw', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 29), list_406511, xbw_406513)
    # Adding element type (line 473)
    # Getting the type of 'xbp' (line 473)
    xbp_406514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 43), 'xbp', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 29), list_406511, xbp_406514)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 28), list_406510, list_406511)
    # Adding element type (line 473)
    
    # Obtaining an instance of the builtin type 'list' (line 474)
    list_406515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 474)
    # Adding element type (line 474)
    # Getting the type of 'xbw' (line 474)
    xbw_406516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 31), 'xbw', False)
    # Obtaining the member 'T' of a type (line 474)
    T_406517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 31), xbw_406516, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 30), list_406515, T_406517)
    # Adding element type (line 474)
    # Getting the type of 'ident' (line 474)
    ident_406518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 38), 'ident', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 30), list_406515, ident_406518)
    # Adding element type (line 474)
    # Getting the type of 'wbp' (line 474)
    wbp_406519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 45), 'wbp', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 30), list_406515, wbp_406519)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 28), list_406510, list_406515)
    # Adding element type (line 473)
    
    # Obtaining an instance of the builtin type 'list' (line 475)
    list_406520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 475)
    # Adding element type (line 475)
    # Getting the type of 'xbp' (line 475)
    xbp_406521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 31), 'xbp', False)
    # Obtaining the member 'T' of a type (line 475)
    T_406522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 31), xbp_406521, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 30), list_406520, T_406522)
    # Adding element type (line 475)
    # Getting the type of 'wbp' (line 475)
    wbp_406523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 38), 'wbp', False)
    # Obtaining the member 'T' of a type (line 475)
    T_406524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 38), wbp_406523, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 30), list_406520, T_406524)
    # Adding element type (line 475)
    # Getting the type of 'ident' (line 475)
    ident_406525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 45), 'ident', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 30), list_406520, ident_406525)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 28), list_406510, list_406520)
    
    # Processing the call keyword arguments (line 473)
    kwargs_406526 = {}
    # Getting the type of 'np' (line 473)
    np_406508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 20), 'np', False)
    # Obtaining the member 'bmat' of a type (line 473)
    bmat_406509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 20), np_406508, 'bmat')
    # Calling bmat(args, kwargs) (line 473)
    bmat_call_result_406527 = invoke(stypy.reporting.localization.Localization(__file__, 473, 20), bmat_406509, *[list_406510], **kwargs_406526)
    
    # Assigning a type to the variable 'gramB' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'gramB', bmat_call_result_406527)
    # SSA branch for the else part of an if statement (line 462)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 477):
    
    # Assigning a Call to a Name (line 477):
    
    # Call to bmat(...): (line 477)
    # Processing the call arguments (line 477)
    
    # Obtaining an instance of the builtin type 'list' (line 477)
    list_406530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 477)
    # Adding element type (line 477)
    
    # Obtaining an instance of the builtin type 'list' (line 477)
    list_406531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 477)
    # Adding element type (line 477)
    
    # Call to diag(...): (line 477)
    # Processing the call arguments (line 477)
    # Getting the type of '_lambda' (line 477)
    _lambda_406534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 38), '_lambda', False)
    # Processing the call keyword arguments (line 477)
    kwargs_406535 = {}
    # Getting the type of 'np' (line 477)
    np_406532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 30), 'np', False)
    # Obtaining the member 'diag' of a type (line 477)
    diag_406533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 30), np_406532, 'diag')
    # Calling diag(args, kwargs) (line 477)
    diag_call_result_406536 = invoke(stypy.reporting.localization.Localization(__file__, 477, 30), diag_406533, *[_lambda_406534], **kwargs_406535)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 29), list_406531, diag_call_result_406536)
    # Adding element type (line 477)
    # Getting the type of 'xaw' (line 477)
    xaw_406537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 48), 'xaw', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 29), list_406531, xaw_406537)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 28), list_406530, list_406531)
    # Adding element type (line 477)
    
    # Obtaining an instance of the builtin type 'list' (line 478)
    list_406538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 478)
    # Adding element type (line 478)
    # Getting the type of 'xaw' (line 478)
    xaw_406539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 31), 'xaw', False)
    # Obtaining the member 'T' of a type (line 478)
    T_406540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 31), xaw_406539, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 30), list_406538, T_406540)
    # Adding element type (line 478)
    # Getting the type of 'waw' (line 478)
    waw_406541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 38), 'waw', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 30), list_406538, waw_406541)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 28), list_406530, list_406538)
    
    # Processing the call keyword arguments (line 477)
    kwargs_406542 = {}
    # Getting the type of 'np' (line 477)
    np_406528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 20), 'np', False)
    # Obtaining the member 'bmat' of a type (line 477)
    bmat_406529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 20), np_406528, 'bmat')
    # Calling bmat(args, kwargs) (line 477)
    bmat_call_result_406543 = invoke(stypy.reporting.localization.Localization(__file__, 477, 20), bmat_406529, *[list_406530], **kwargs_406542)
    
    # Assigning a type to the variable 'gramA' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'gramA', bmat_call_result_406543)
    
    # Assigning a Call to a Name (line 479):
    
    # Assigning a Call to a Name (line 479):
    
    # Call to bmat(...): (line 479)
    # Processing the call arguments (line 479)
    
    # Obtaining an instance of the builtin type 'list' (line 479)
    list_406546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 479)
    # Adding element type (line 479)
    
    # Obtaining an instance of the builtin type 'list' (line 479)
    list_406547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 479)
    # Adding element type (line 479)
    # Getting the type of 'ident0' (line 479)
    ident0_406548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 30), 'ident0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 29), list_406547, ident0_406548)
    # Adding element type (line 479)
    # Getting the type of 'xbw' (line 479)
    xbw_406549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 38), 'xbw', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 29), list_406547, xbw_406549)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 28), list_406546, list_406547)
    # Adding element type (line 479)
    
    # Obtaining an instance of the builtin type 'list' (line 480)
    list_406550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 480)
    # Adding element type (line 480)
    # Getting the type of 'xbw' (line 480)
    xbw_406551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 31), 'xbw', False)
    # Obtaining the member 'T' of a type (line 480)
    T_406552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 31), xbw_406551, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 30), list_406550, T_406552)
    # Adding element type (line 480)
    # Getting the type of 'ident' (line 480)
    ident_406553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 38), 'ident', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 30), list_406550, ident_406553)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 28), list_406546, list_406550)
    
    # Processing the call keyword arguments (line 479)
    kwargs_406554 = {}
    # Getting the type of 'np' (line 479)
    np_406544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 20), 'np', False)
    # Obtaining the member 'bmat' of a type (line 479)
    bmat_406545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 20), np_406544, 'bmat')
    # Calling bmat(args, kwargs) (line 479)
    bmat_call_result_406555 = invoke(stypy.reporting.localization.Localization(__file__, 479, 20), bmat_406545, *[list_406546], **kwargs_406554)
    
    # Assigning a type to the variable 'gramB' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'gramB', bmat_call_result_406555)
    # SSA join for if statement (line 462)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _assert_symmetric(...): (line 482)
    # Processing the call arguments (line 482)
    # Getting the type of 'gramA' (line 482)
    gramA_406557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 26), 'gramA', False)
    # Processing the call keyword arguments (line 482)
    kwargs_406558 = {}
    # Getting the type of '_assert_symmetric' (line 482)
    _assert_symmetric_406556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), '_assert_symmetric', False)
    # Calling _assert_symmetric(args, kwargs) (line 482)
    _assert_symmetric_call_result_406559 = invoke(stypy.reporting.localization.Localization(__file__, 482, 8), _assert_symmetric_406556, *[gramA_406557], **kwargs_406558)
    
    
    # Call to _assert_symmetric(...): (line 483)
    # Processing the call arguments (line 483)
    # Getting the type of 'gramB' (line 483)
    gramB_406561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 26), 'gramB', False)
    # Processing the call keyword arguments (line 483)
    kwargs_406562 = {}
    # Getting the type of '_assert_symmetric' (line 483)
    _assert_symmetric_406560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), '_assert_symmetric', False)
    # Calling _assert_symmetric(args, kwargs) (line 483)
    _assert_symmetric_call_result_406563 = invoke(stypy.reporting.localization.Localization(__file__, 483, 8), _assert_symmetric_406560, *[gramB_406561], **kwargs_406562)
    
    
    
    # Getting the type of 'verbosityLevel' (line 485)
    verbosityLevel_406564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 11), 'verbosityLevel')
    int_406565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 28), 'int')
    # Applying the binary operator '>' (line 485)
    result_gt_406566 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 11), '>', verbosityLevel_406564, int_406565)
    
    # Testing the type of an if condition (line 485)
    if_condition_406567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 485, 8), result_gt_406566)
    # Assigning a type to the variable 'if_condition_406567' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'if_condition_406567', if_condition_406567)
    # SSA begins for if statement (line 485)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to save(...): (line 486)
    # Processing the call arguments (line 486)
    # Getting the type of 'gramA' (line 486)
    gramA_406569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 17), 'gramA', False)
    str_406570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 24), 'str', 'gramA')
    # Processing the call keyword arguments (line 486)
    kwargs_406571 = {}
    # Getting the type of 'save' (line 486)
    save_406568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'save', False)
    # Calling save(args, kwargs) (line 486)
    save_call_result_406572 = invoke(stypy.reporting.localization.Localization(__file__, 486, 12), save_406568, *[gramA_406569, str_406570], **kwargs_406571)
    
    
    # Call to save(...): (line 487)
    # Processing the call arguments (line 487)
    # Getting the type of 'gramB' (line 487)
    gramB_406574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 17), 'gramB', False)
    str_406575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 24), 'str', 'gramB')
    # Processing the call keyword arguments (line 487)
    kwargs_406576 = {}
    # Getting the type of 'save' (line 487)
    save_406573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'save', False)
    # Calling save(args, kwargs) (line 487)
    save_call_result_406577 = invoke(stypy.reporting.localization.Localization(__file__, 487, 12), save_406573, *[gramB_406574, str_406575], **kwargs_406576)
    
    # SSA join for if statement (line 485)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 490):
    
    # Assigning a Subscript to a Name (line 490):
    
    # Obtaining the type of the subscript
    int_406578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 8), 'int')
    
    # Call to eigh(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'gramA' (line 490)
    gramA_406580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 39), 'gramA', False)
    # Getting the type of 'gramB' (line 490)
    gramB_406581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 46), 'gramB', False)
    # Processing the call keyword arguments (line 490)
    # Getting the type of 'False' (line 490)
    False_406582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 66), 'False', False)
    keyword_406583 = False_406582
    kwargs_406584 = {'check_finite': keyword_406583}
    # Getting the type of 'eigh' (line 490)
    eigh_406579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 34), 'eigh', False)
    # Calling eigh(args, kwargs) (line 490)
    eigh_call_result_406585 = invoke(stypy.reporting.localization.Localization(__file__, 490, 34), eigh_406579, *[gramA_406580, gramB_406581], **kwargs_406584)
    
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___406586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), eigh_call_result_406585, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 490)
    subscript_call_result_406587 = invoke(stypy.reporting.localization.Localization(__file__, 490, 8), getitem___406586, int_406578)
    
    # Assigning a type to the variable 'tuple_var_assignment_405644' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'tuple_var_assignment_405644', subscript_call_result_406587)
    
    # Assigning a Subscript to a Name (line 490):
    
    # Obtaining the type of the subscript
    int_406588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 8), 'int')
    
    # Call to eigh(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'gramA' (line 490)
    gramA_406590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 39), 'gramA', False)
    # Getting the type of 'gramB' (line 490)
    gramB_406591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 46), 'gramB', False)
    # Processing the call keyword arguments (line 490)
    # Getting the type of 'False' (line 490)
    False_406592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 66), 'False', False)
    keyword_406593 = False_406592
    kwargs_406594 = {'check_finite': keyword_406593}
    # Getting the type of 'eigh' (line 490)
    eigh_406589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 34), 'eigh', False)
    # Calling eigh(args, kwargs) (line 490)
    eigh_call_result_406595 = invoke(stypy.reporting.localization.Localization(__file__, 490, 34), eigh_406589, *[gramA_406590, gramB_406591], **kwargs_406594)
    
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___406596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), eigh_call_result_406595, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 490)
    subscript_call_result_406597 = invoke(stypy.reporting.localization.Localization(__file__, 490, 8), getitem___406596, int_406588)
    
    # Assigning a type to the variable 'tuple_var_assignment_405645' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'tuple_var_assignment_405645', subscript_call_result_406597)
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'tuple_var_assignment_405644' (line 490)
    tuple_var_assignment_405644_406598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'tuple_var_assignment_405644')
    # Assigning a type to the variable '_lambda' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), '_lambda', tuple_var_assignment_405644_406598)
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'tuple_var_assignment_405645' (line 490)
    tuple_var_assignment_405645_406599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'tuple_var_assignment_405645')
    # Assigning a type to the variable 'eigBlockVector' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 17), 'eigBlockVector', tuple_var_assignment_405645_406599)
    
    # Assigning a Subscript to a Name (line 491):
    
    # Assigning a Subscript to a Name (line 491):
    
    # Obtaining the type of the subscript
    # Getting the type of 'sizeX' (line 491)
    sizeX_406600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 34), 'sizeX')
    slice_406601 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 491, 13), None, sizeX_406600, None)
    
    # Call to argsort(...): (line 491)
    # Processing the call arguments (line 491)
    # Getting the type of '_lambda' (line 491)
    _lambda_406604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 24), '_lambda', False)
    # Processing the call keyword arguments (line 491)
    kwargs_406605 = {}
    # Getting the type of 'np' (line 491)
    np_406602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 13), 'np', False)
    # Obtaining the member 'argsort' of a type (line 491)
    argsort_406603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 13), np_406602, 'argsort')
    # Calling argsort(args, kwargs) (line 491)
    argsort_call_result_406606 = invoke(stypy.reporting.localization.Localization(__file__, 491, 13), argsort_406603, *[_lambda_406604], **kwargs_406605)
    
    # Obtaining the member '__getitem__' of a type (line 491)
    getitem___406607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 13), argsort_call_result_406606, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 491)
    subscript_call_result_406608 = invoke(stypy.reporting.localization.Localization(__file__, 491, 13), getitem___406607, slice_406601)
    
    # Assigning a type to the variable 'ii' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'ii', subscript_call_result_406608)
    
    # Getting the type of 'largest' (line 492)
    largest_406609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 11), 'largest')
    # Testing the type of an if condition (line 492)
    if_condition_406610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 492, 8), largest_406609)
    # Assigning a type to the variable 'if_condition_406610' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'if_condition_406610', if_condition_406610)
    # SSA begins for if statement (line 492)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 493):
    
    # Assigning a Subscript to a Name (line 493):
    
    # Obtaining the type of the subscript
    int_406611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 22), 'int')
    slice_406612 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 493, 17), None, None, int_406611)
    # Getting the type of 'ii' (line 493)
    ii_406613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 17), 'ii')
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___406614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 17), ii_406613, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 493)
    subscript_call_result_406615 = invoke(stypy.reporting.localization.Localization(__file__, 493, 17), getitem___406614, slice_406612)
    
    # Assigning a type to the variable 'ii' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'ii', subscript_call_result_406615)
    # SSA join for if statement (line 492)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbosityLevel' (line 494)
    verbosityLevel_406616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 11), 'verbosityLevel')
    int_406617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 28), 'int')
    # Applying the binary operator '>' (line 494)
    result_gt_406618 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 11), '>', verbosityLevel_406616, int_406617)
    
    # Testing the type of an if condition (line 494)
    if_condition_406619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 494, 8), result_gt_406618)
    # Assigning a type to the variable 'if_condition_406619' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'if_condition_406619', if_condition_406619)
    # SSA begins for if statement (line 494)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 495)
    # Processing the call arguments (line 495)
    # Getting the type of 'ii' (line 495)
    ii_406621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 18), 'ii', False)
    # Processing the call keyword arguments (line 495)
    kwargs_406622 = {}
    # Getting the type of 'print' (line 495)
    print_406620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'print', False)
    # Calling print(args, kwargs) (line 495)
    print_call_result_406623 = invoke(stypy.reporting.localization.Localization(__file__, 495, 12), print_406620, *[ii_406621], **kwargs_406622)
    
    # SSA join for if statement (line 494)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 497):
    
    # Assigning a Call to a Name (line 497):
    
    # Call to astype(...): (line 497)
    # Processing the call arguments (line 497)
    # Getting the type of 'np' (line 497)
    np_406629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 37), 'np', False)
    # Obtaining the member 'float64' of a type (line 497)
    float64_406630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 37), np_406629, 'float64')
    # Processing the call keyword arguments (line 497)
    kwargs_406631 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 497)
    ii_406624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 26), 'ii', False)
    # Getting the type of '_lambda' (line 497)
    _lambda_406625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 18), '_lambda', False)
    # Obtaining the member '__getitem__' of a type (line 497)
    getitem___406626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 18), _lambda_406625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 497)
    subscript_call_result_406627 = invoke(stypy.reporting.localization.Localization(__file__, 497, 18), getitem___406626, ii_406624)
    
    # Obtaining the member 'astype' of a type (line 497)
    astype_406628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 18), subscript_call_result_406627, 'astype')
    # Calling astype(args, kwargs) (line 497)
    astype_call_result_406632 = invoke(stypy.reporting.localization.Localization(__file__, 497, 18), astype_406628, *[float64_406630], **kwargs_406631)
    
    # Assigning a type to the variable '_lambda' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), '_lambda', astype_call_result_406632)
    
    # Assigning a Call to a Name (line 498):
    
    # Assigning a Call to a Name (line 498):
    
    # Call to asarray(...): (line 498)
    # Processing the call arguments (line 498)
    
    # Call to astype(...): (line 498)
    # Processing the call arguments (line 498)
    # Getting the type of 'np' (line 498)
    np_406641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 64), 'np', False)
    # Obtaining the member 'float64' of a type (line 498)
    float64_406642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 64), np_406641, 'float64')
    # Processing the call keyword arguments (line 498)
    kwargs_406643 = {}
    
    # Obtaining the type of the subscript
    slice_406635 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 498, 36), None, None, None)
    # Getting the type of 'ii' (line 498)
    ii_406636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 53), 'ii', False)
    # Getting the type of 'eigBlockVector' (line 498)
    eigBlockVector_406637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 36), 'eigBlockVector', False)
    # Obtaining the member '__getitem__' of a type (line 498)
    getitem___406638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 36), eigBlockVector_406637, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 498)
    subscript_call_result_406639 = invoke(stypy.reporting.localization.Localization(__file__, 498, 36), getitem___406638, (slice_406635, ii_406636))
    
    # Obtaining the member 'astype' of a type (line 498)
    astype_406640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 36), subscript_call_result_406639, 'astype')
    # Calling astype(args, kwargs) (line 498)
    astype_call_result_406644 = invoke(stypy.reporting.localization.Localization(__file__, 498, 36), astype_406640, *[float64_406642], **kwargs_406643)
    
    # Processing the call keyword arguments (line 498)
    kwargs_406645 = {}
    # Getting the type of 'np' (line 498)
    np_406633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 25), 'np', False)
    # Obtaining the member 'asarray' of a type (line 498)
    asarray_406634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 25), np_406633, 'asarray')
    # Calling asarray(args, kwargs) (line 498)
    asarray_call_result_406646 = invoke(stypy.reporting.localization.Localization(__file__, 498, 25), asarray_406634, *[astype_call_result_406644], **kwargs_406645)
    
    # Assigning a type to the variable 'eigBlockVector' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'eigBlockVector', asarray_call_result_406646)
    
    # Call to append(...): (line 500)
    # Processing the call arguments (line 500)
    # Getting the type of '_lambda' (line 500)
    _lambda_406649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 29), '_lambda', False)
    # Processing the call keyword arguments (line 500)
    kwargs_406650 = {}
    # Getting the type of 'lambdaHistory' (line 500)
    lambdaHistory_406647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'lambdaHistory', False)
    # Obtaining the member 'append' of a type (line 500)
    append_406648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), lambdaHistory_406647, 'append')
    # Calling append(args, kwargs) (line 500)
    append_call_result_406651 = invoke(stypy.reporting.localization.Localization(__file__, 500, 8), append_406648, *[_lambda_406649], **kwargs_406650)
    
    
    
    # Getting the type of 'verbosityLevel' (line 502)
    verbosityLevel_406652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 11), 'verbosityLevel')
    int_406653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 28), 'int')
    # Applying the binary operator '>' (line 502)
    result_gt_406654 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 11), '>', verbosityLevel_406652, int_406653)
    
    # Testing the type of an if condition (line 502)
    if_condition_406655 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 8), result_gt_406654)
    # Assigning a type to the variable 'if_condition_406655' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'if_condition_406655', if_condition_406655)
    # SSA begins for if statement (line 502)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 503)
    # Processing the call arguments (line 503)
    str_406657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 18), 'str', 'lambda:')
    # Getting the type of '_lambda' (line 503)
    _lambda_406658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 29), '_lambda', False)
    # Processing the call keyword arguments (line 503)
    kwargs_406659 = {}
    # Getting the type of 'print' (line 503)
    print_406656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'print', False)
    # Calling print(args, kwargs) (line 503)
    print_call_result_406660 = invoke(stypy.reporting.localization.Localization(__file__, 503, 12), print_406656, *[str_406657, _lambda_406658], **kwargs_406659)
    
    # SSA join for if statement (line 502)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbosityLevel' (line 510)
    verbosityLevel_406661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 11), 'verbosityLevel')
    int_406662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 28), 'int')
    # Applying the binary operator '>' (line 510)
    result_gt_406663 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 11), '>', verbosityLevel_406661, int_406662)
    
    # Testing the type of an if condition (line 510)
    if_condition_406664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 510, 8), result_gt_406663)
    # Assigning a type to the variable 'if_condition_406664' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'if_condition_406664', if_condition_406664)
    # SSA begins for if statement (line 510)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 511)
    # Processing the call arguments (line 511)
    # Getting the type of 'eigBlockVector' (line 511)
    eigBlockVector_406666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 18), 'eigBlockVector', False)
    # Processing the call keyword arguments (line 511)
    kwargs_406667 = {}
    # Getting the type of 'print' (line 511)
    print_406665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'print', False)
    # Calling print(args, kwargs) (line 511)
    print_call_result_406668 = invoke(stypy.reporting.localization.Localization(__file__, 511, 12), print_406665, *[eigBlockVector_406666], **kwargs_406667)
    
    
    # Call to pause(...): (line 512)
    # Processing the call keyword arguments (line 512)
    kwargs_406670 = {}
    # Getting the type of 'pause' (line 512)
    pause_406669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'pause', False)
    # Calling pause(args, kwargs) (line 512)
    pause_call_result_406671 = invoke(stypy.reporting.localization.Localization(__file__, 512, 12), pause_406669, *[], **kwargs_406670)
    
    # SSA join for if statement (line 510)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iterationNumber' (line 516)
    iterationNumber_406672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 11), 'iterationNumber')
    int_406673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 29), 'int')
    # Applying the binary operator '>' (line 516)
    result_gt_406674 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 11), '>', iterationNumber_406672, int_406673)
    
    # Testing the type of an if condition (line 516)
    if_condition_406675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 516, 8), result_gt_406674)
    # Assigning a type to the variable 'if_condition_406675' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'if_condition_406675', if_condition_406675)
    # SSA begins for if statement (line 516)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 517):
    
    # Assigning a Subscript to a Name (line 517):
    
    # Obtaining the type of the subscript
    # Getting the type of 'sizeX' (line 517)
    sizeX_406676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 46), 'sizeX')
    slice_406677 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 517, 30), None, sizeX_406676, None)
    # Getting the type of 'eigBlockVector' (line 517)
    eigBlockVector_406678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 30), 'eigBlockVector')
    # Obtaining the member '__getitem__' of a type (line 517)
    getitem___406679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 30), eigBlockVector_406678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 517)
    subscript_call_result_406680 = invoke(stypy.reporting.localization.Localization(__file__, 517, 30), getitem___406679, slice_406677)
    
    # Assigning a type to the variable 'eigBlockVectorX' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'eigBlockVectorX', subscript_call_result_406680)
    
    # Assigning a Subscript to a Name (line 518):
    
    # Assigning a Subscript to a Name (line 518):
    
    # Obtaining the type of the subscript
    # Getting the type of 'sizeX' (line 518)
    sizeX_406681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 45), 'sizeX')
    # Getting the type of 'sizeX' (line 518)
    sizeX_406682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 51), 'sizeX')
    # Getting the type of 'currentBlockSize' (line 518)
    currentBlockSize_406683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 57), 'currentBlockSize')
    # Applying the binary operator '+' (line 518)
    result_add_406684 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 51), '+', sizeX_406682, currentBlockSize_406683)
    
    slice_406685 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 518, 30), sizeX_406681, result_add_406684, None)
    # Getting the type of 'eigBlockVector' (line 518)
    eigBlockVector_406686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 30), 'eigBlockVector')
    # Obtaining the member '__getitem__' of a type (line 518)
    getitem___406687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 30), eigBlockVector_406686, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 518)
    subscript_call_result_406688 = invoke(stypy.reporting.localization.Localization(__file__, 518, 30), getitem___406687, slice_406685)
    
    # Assigning a type to the variable 'eigBlockVectorR' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'eigBlockVectorR', subscript_call_result_406688)
    
    # Assigning a Subscript to a Name (line 519):
    
    # Assigning a Subscript to a Name (line 519):
    
    # Obtaining the type of the subscript
    # Getting the type of 'sizeX' (line 519)
    sizeX_406689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 45), 'sizeX')
    # Getting the type of 'currentBlockSize' (line 519)
    currentBlockSize_406690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 51), 'currentBlockSize')
    # Applying the binary operator '+' (line 519)
    result_add_406691 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 45), '+', sizeX_406689, currentBlockSize_406690)
    
    slice_406692 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 519, 30), result_add_406691, None, None)
    # Getting the type of 'eigBlockVector' (line 519)
    eigBlockVector_406693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 30), 'eigBlockVector')
    # Obtaining the member '__getitem__' of a type (line 519)
    getitem___406694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 30), eigBlockVector_406693, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 519)
    subscript_call_result_406695 = invoke(stypy.reporting.localization.Localization(__file__, 519, 30), getitem___406694, slice_406692)
    
    # Assigning a type to the variable 'eigBlockVectorP' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'eigBlockVectorP', subscript_call_result_406695)
    
    # Assigning a Call to a Name (line 521):
    
    # Assigning a Call to a Name (line 521):
    
    # Call to dot(...): (line 521)
    # Processing the call arguments (line 521)
    # Getting the type of 'activeBlockVectorR' (line 521)
    activeBlockVectorR_406698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 24), 'activeBlockVectorR', False)
    # Getting the type of 'eigBlockVectorR' (line 521)
    eigBlockVectorR_406699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 44), 'eigBlockVectorR', False)
    # Processing the call keyword arguments (line 521)
    kwargs_406700 = {}
    # Getting the type of 'np' (line 521)
    np_406696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 17), 'np', False)
    # Obtaining the member 'dot' of a type (line 521)
    dot_406697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 17), np_406696, 'dot')
    # Calling dot(args, kwargs) (line 521)
    dot_call_result_406701 = invoke(stypy.reporting.localization.Localization(__file__, 521, 17), dot_406697, *[activeBlockVectorR_406698, eigBlockVectorR_406699], **kwargs_406700)
    
    # Assigning a type to the variable 'pp' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'pp', dot_call_result_406701)
    
    # Getting the type of 'pp' (line 522)
    pp_406702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'pp')
    
    # Call to dot(...): (line 522)
    # Processing the call arguments (line 522)
    # Getting the type of 'activeBlockVectorP' (line 522)
    activeBlockVectorP_406705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 25), 'activeBlockVectorP', False)
    # Getting the type of 'eigBlockVectorP' (line 522)
    eigBlockVectorP_406706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 45), 'eigBlockVectorP', False)
    # Processing the call keyword arguments (line 522)
    kwargs_406707 = {}
    # Getting the type of 'np' (line 522)
    np_406703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 522)
    dot_406704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 18), np_406703, 'dot')
    # Calling dot(args, kwargs) (line 522)
    dot_call_result_406708 = invoke(stypy.reporting.localization.Localization(__file__, 522, 18), dot_406704, *[activeBlockVectorP_406705, eigBlockVectorP_406706], **kwargs_406707)
    
    # Applying the binary operator '+=' (line 522)
    result_iadd_406709 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 12), '+=', pp_406702, dot_call_result_406708)
    # Assigning a type to the variable 'pp' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'pp', result_iadd_406709)
    
    
    # Assigning a Call to a Name (line 524):
    
    # Assigning a Call to a Name (line 524):
    
    # Call to dot(...): (line 524)
    # Processing the call arguments (line 524)
    # Getting the type of 'activeBlockVectorAR' (line 524)
    activeBlockVectorAR_406712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 25), 'activeBlockVectorAR', False)
    # Getting the type of 'eigBlockVectorR' (line 524)
    eigBlockVectorR_406713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 46), 'eigBlockVectorR', False)
    # Processing the call keyword arguments (line 524)
    kwargs_406714 = {}
    # Getting the type of 'np' (line 524)
    np_406710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 524)
    dot_406711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 18), np_406710, 'dot')
    # Calling dot(args, kwargs) (line 524)
    dot_call_result_406715 = invoke(stypy.reporting.localization.Localization(__file__, 524, 18), dot_406711, *[activeBlockVectorAR_406712, eigBlockVectorR_406713], **kwargs_406714)
    
    # Assigning a type to the variable 'app' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'app', dot_call_result_406715)
    
    # Getting the type of 'app' (line 525)
    app_406716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'app')
    
    # Call to dot(...): (line 525)
    # Processing the call arguments (line 525)
    # Getting the type of 'activeBlockVectorAP' (line 525)
    activeBlockVectorAP_406719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 26), 'activeBlockVectorAP', False)
    # Getting the type of 'eigBlockVectorP' (line 525)
    eigBlockVectorP_406720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 47), 'eigBlockVectorP', False)
    # Processing the call keyword arguments (line 525)
    kwargs_406721 = {}
    # Getting the type of 'np' (line 525)
    np_406717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 19), 'np', False)
    # Obtaining the member 'dot' of a type (line 525)
    dot_406718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 19), np_406717, 'dot')
    # Calling dot(args, kwargs) (line 525)
    dot_call_result_406722 = invoke(stypy.reporting.localization.Localization(__file__, 525, 19), dot_406718, *[activeBlockVectorAP_406719, eigBlockVectorP_406720], **kwargs_406721)
    
    # Applying the binary operator '+=' (line 525)
    result_iadd_406723 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 12), '+=', app_406716, dot_call_result_406722)
    # Assigning a type to the variable 'app' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'app', result_iadd_406723)
    
    
    # Assigning a Call to a Name (line 527):
    
    # Assigning a Call to a Name (line 527):
    
    # Call to dot(...): (line 527)
    # Processing the call arguments (line 527)
    # Getting the type of 'activeBlockVectorBR' (line 527)
    activeBlockVectorBR_406726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 25), 'activeBlockVectorBR', False)
    # Getting the type of 'eigBlockVectorR' (line 527)
    eigBlockVectorR_406727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 46), 'eigBlockVectorR', False)
    # Processing the call keyword arguments (line 527)
    kwargs_406728 = {}
    # Getting the type of 'np' (line 527)
    np_406724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 527)
    dot_406725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 18), np_406724, 'dot')
    # Calling dot(args, kwargs) (line 527)
    dot_call_result_406729 = invoke(stypy.reporting.localization.Localization(__file__, 527, 18), dot_406725, *[activeBlockVectorBR_406726, eigBlockVectorR_406727], **kwargs_406728)
    
    # Assigning a type to the variable 'bpp' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'bpp', dot_call_result_406729)
    
    # Getting the type of 'bpp' (line 528)
    bpp_406730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'bpp')
    
    # Call to dot(...): (line 528)
    # Processing the call arguments (line 528)
    # Getting the type of 'activeBlockVectorBP' (line 528)
    activeBlockVectorBP_406733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 26), 'activeBlockVectorBP', False)
    # Getting the type of 'eigBlockVectorP' (line 528)
    eigBlockVectorP_406734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 47), 'eigBlockVectorP', False)
    # Processing the call keyword arguments (line 528)
    kwargs_406735 = {}
    # Getting the type of 'np' (line 528)
    np_406731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 19), 'np', False)
    # Obtaining the member 'dot' of a type (line 528)
    dot_406732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 19), np_406731, 'dot')
    # Calling dot(args, kwargs) (line 528)
    dot_call_result_406736 = invoke(stypy.reporting.localization.Localization(__file__, 528, 19), dot_406732, *[activeBlockVectorBP_406733, eigBlockVectorP_406734], **kwargs_406735)
    
    # Applying the binary operator '+=' (line 528)
    result_iadd_406737 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 12), '+=', bpp_406730, dot_call_result_406736)
    # Assigning a type to the variable 'bpp' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'bpp', result_iadd_406737)
    
    # SSA branch for the else part of an if statement (line 516)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 530):
    
    # Assigning a Subscript to a Name (line 530):
    
    # Obtaining the type of the subscript
    # Getting the type of 'sizeX' (line 530)
    sizeX_406738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 46), 'sizeX')
    slice_406739 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 530, 30), None, sizeX_406738, None)
    # Getting the type of 'eigBlockVector' (line 530)
    eigBlockVector_406740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 30), 'eigBlockVector')
    # Obtaining the member '__getitem__' of a type (line 530)
    getitem___406741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 30), eigBlockVector_406740, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 530)
    subscript_call_result_406742 = invoke(stypy.reporting.localization.Localization(__file__, 530, 30), getitem___406741, slice_406739)
    
    # Assigning a type to the variable 'eigBlockVectorX' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'eigBlockVectorX', subscript_call_result_406742)
    
    # Assigning a Subscript to a Name (line 531):
    
    # Assigning a Subscript to a Name (line 531):
    
    # Obtaining the type of the subscript
    # Getting the type of 'sizeX' (line 531)
    sizeX_406743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 45), 'sizeX')
    slice_406744 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 531, 30), sizeX_406743, None, None)
    # Getting the type of 'eigBlockVector' (line 531)
    eigBlockVector_406745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 30), 'eigBlockVector')
    # Obtaining the member '__getitem__' of a type (line 531)
    getitem___406746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 30), eigBlockVector_406745, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 531)
    subscript_call_result_406747 = invoke(stypy.reporting.localization.Localization(__file__, 531, 30), getitem___406746, slice_406744)
    
    # Assigning a type to the variable 'eigBlockVectorR' (line 531)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'eigBlockVectorR', subscript_call_result_406747)
    
    # Assigning a Call to a Name (line 533):
    
    # Assigning a Call to a Name (line 533):
    
    # Call to dot(...): (line 533)
    # Processing the call arguments (line 533)
    # Getting the type of 'activeBlockVectorR' (line 533)
    activeBlockVectorR_406750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 24), 'activeBlockVectorR', False)
    # Getting the type of 'eigBlockVectorR' (line 533)
    eigBlockVectorR_406751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 44), 'eigBlockVectorR', False)
    # Processing the call keyword arguments (line 533)
    kwargs_406752 = {}
    # Getting the type of 'np' (line 533)
    np_406748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 17), 'np', False)
    # Obtaining the member 'dot' of a type (line 533)
    dot_406749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 17), np_406748, 'dot')
    # Calling dot(args, kwargs) (line 533)
    dot_call_result_406753 = invoke(stypy.reporting.localization.Localization(__file__, 533, 17), dot_406749, *[activeBlockVectorR_406750, eigBlockVectorR_406751], **kwargs_406752)
    
    # Assigning a type to the variable 'pp' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'pp', dot_call_result_406753)
    
    # Assigning a Call to a Name (line 534):
    
    # Assigning a Call to a Name (line 534):
    
    # Call to dot(...): (line 534)
    # Processing the call arguments (line 534)
    # Getting the type of 'activeBlockVectorAR' (line 534)
    activeBlockVectorAR_406756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 25), 'activeBlockVectorAR', False)
    # Getting the type of 'eigBlockVectorR' (line 534)
    eigBlockVectorR_406757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 46), 'eigBlockVectorR', False)
    # Processing the call keyword arguments (line 534)
    kwargs_406758 = {}
    # Getting the type of 'np' (line 534)
    np_406754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 534)
    dot_406755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 18), np_406754, 'dot')
    # Calling dot(args, kwargs) (line 534)
    dot_call_result_406759 = invoke(stypy.reporting.localization.Localization(__file__, 534, 18), dot_406755, *[activeBlockVectorAR_406756, eigBlockVectorR_406757], **kwargs_406758)
    
    # Assigning a type to the variable 'app' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'app', dot_call_result_406759)
    
    # Assigning a Call to a Name (line 535):
    
    # Assigning a Call to a Name (line 535):
    
    # Call to dot(...): (line 535)
    # Processing the call arguments (line 535)
    # Getting the type of 'activeBlockVectorBR' (line 535)
    activeBlockVectorBR_406762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 25), 'activeBlockVectorBR', False)
    # Getting the type of 'eigBlockVectorR' (line 535)
    eigBlockVectorR_406763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 46), 'eigBlockVectorR', False)
    # Processing the call keyword arguments (line 535)
    kwargs_406764 = {}
    # Getting the type of 'np' (line 535)
    np_406760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 535)
    dot_406761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 18), np_406760, 'dot')
    # Calling dot(args, kwargs) (line 535)
    dot_call_result_406765 = invoke(stypy.reporting.localization.Localization(__file__, 535, 18), dot_406761, *[activeBlockVectorBR_406762, eigBlockVectorR_406763], **kwargs_406764)
    
    # Assigning a type to the variable 'bpp' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'bpp', dot_call_result_406765)
    # SSA join for if statement (line 516)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbosityLevel' (line 537)
    verbosityLevel_406766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 11), 'verbosityLevel')
    int_406767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 28), 'int')
    # Applying the binary operator '>' (line 537)
    result_gt_406768 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 11), '>', verbosityLevel_406766, int_406767)
    
    # Testing the type of an if condition (line 537)
    if_condition_406769 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 537, 8), result_gt_406768)
    # Assigning a type to the variable 'if_condition_406769' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'if_condition_406769', if_condition_406769)
    # SSA begins for if statement (line 537)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 538)
    # Processing the call arguments (line 538)
    # Getting the type of 'pp' (line 538)
    pp_406771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 18), 'pp', False)
    # Processing the call keyword arguments (line 538)
    kwargs_406772 = {}
    # Getting the type of 'print' (line 538)
    print_406770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'print', False)
    # Calling print(args, kwargs) (line 538)
    print_call_result_406773 = invoke(stypy.reporting.localization.Localization(__file__, 538, 12), print_406770, *[pp_406771], **kwargs_406772)
    
    
    # Call to print(...): (line 539)
    # Processing the call arguments (line 539)
    # Getting the type of 'app' (line 539)
    app_406775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 18), 'app', False)
    # Processing the call keyword arguments (line 539)
    kwargs_406776 = {}
    # Getting the type of 'print' (line 539)
    print_406774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'print', False)
    # Calling print(args, kwargs) (line 539)
    print_call_result_406777 = invoke(stypy.reporting.localization.Localization(__file__, 539, 12), print_406774, *[app_406775], **kwargs_406776)
    
    
    # Call to print(...): (line 540)
    # Processing the call arguments (line 540)
    # Getting the type of 'bpp' (line 540)
    bpp_406779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 18), 'bpp', False)
    # Processing the call keyword arguments (line 540)
    kwargs_406780 = {}
    # Getting the type of 'print' (line 540)
    print_406778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 12), 'print', False)
    # Calling print(args, kwargs) (line 540)
    print_call_result_406781 = invoke(stypy.reporting.localization.Localization(__file__, 540, 12), print_406778, *[bpp_406779], **kwargs_406780)
    
    
    # Call to pause(...): (line 541)
    # Processing the call keyword arguments (line 541)
    kwargs_406783 = {}
    # Getting the type of 'pause' (line 541)
    pause_406782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'pause', False)
    # Calling pause(args, kwargs) (line 541)
    pause_call_result_406784 = invoke(stypy.reporting.localization.Localization(__file__, 541, 12), pause_406782, *[], **kwargs_406783)
    
    # SSA join for if statement (line 537)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 543):
    
    # Assigning a BinOp to a Name (line 543):
    
    # Call to dot(...): (line 543)
    # Processing the call arguments (line 543)
    # Getting the type of 'blockVectorX' (line 543)
    blockVectorX_406787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 30), 'blockVectorX', False)
    # Getting the type of 'eigBlockVectorX' (line 543)
    eigBlockVectorX_406788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 44), 'eigBlockVectorX', False)
    # Processing the call keyword arguments (line 543)
    kwargs_406789 = {}
    # Getting the type of 'np' (line 543)
    np_406785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 23), 'np', False)
    # Obtaining the member 'dot' of a type (line 543)
    dot_406786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 23), np_406785, 'dot')
    # Calling dot(args, kwargs) (line 543)
    dot_call_result_406790 = invoke(stypy.reporting.localization.Localization(__file__, 543, 23), dot_406786, *[blockVectorX_406787, eigBlockVectorX_406788], **kwargs_406789)
    
    # Getting the type of 'pp' (line 543)
    pp_406791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 63), 'pp')
    # Applying the binary operator '+' (line 543)
    result_add_406792 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 23), '+', dot_call_result_406790, pp_406791)
    
    # Assigning a type to the variable 'blockVectorX' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'blockVectorX', result_add_406792)
    
    # Assigning a BinOp to a Name (line 544):
    
    # Assigning a BinOp to a Name (line 544):
    
    # Call to dot(...): (line 544)
    # Processing the call arguments (line 544)
    # Getting the type of 'blockVectorAX' (line 544)
    blockVectorAX_406795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 31), 'blockVectorAX', False)
    # Getting the type of 'eigBlockVectorX' (line 544)
    eigBlockVectorX_406796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 46), 'eigBlockVectorX', False)
    # Processing the call keyword arguments (line 544)
    kwargs_406797 = {}
    # Getting the type of 'np' (line 544)
    np_406793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 24), 'np', False)
    # Obtaining the member 'dot' of a type (line 544)
    dot_406794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 24), np_406793, 'dot')
    # Calling dot(args, kwargs) (line 544)
    dot_call_result_406798 = invoke(stypy.reporting.localization.Localization(__file__, 544, 24), dot_406794, *[blockVectorAX_406795, eigBlockVectorX_406796], **kwargs_406797)
    
    # Getting the type of 'app' (line 544)
    app_406799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 65), 'app')
    # Applying the binary operator '+' (line 544)
    result_add_406800 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 24), '+', dot_call_result_406798, app_406799)
    
    # Assigning a type to the variable 'blockVectorAX' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'blockVectorAX', result_add_406800)
    
    # Assigning a BinOp to a Name (line 545):
    
    # Assigning a BinOp to a Name (line 545):
    
    # Call to dot(...): (line 545)
    # Processing the call arguments (line 545)
    # Getting the type of 'blockVectorBX' (line 545)
    blockVectorBX_406803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 31), 'blockVectorBX', False)
    # Getting the type of 'eigBlockVectorX' (line 545)
    eigBlockVectorX_406804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 46), 'eigBlockVectorX', False)
    # Processing the call keyword arguments (line 545)
    kwargs_406805 = {}
    # Getting the type of 'np' (line 545)
    np_406801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 24), 'np', False)
    # Obtaining the member 'dot' of a type (line 545)
    dot_406802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 24), np_406801, 'dot')
    # Calling dot(args, kwargs) (line 545)
    dot_call_result_406806 = invoke(stypy.reporting.localization.Localization(__file__, 545, 24), dot_406802, *[blockVectorBX_406803, eigBlockVectorX_406804], **kwargs_406805)
    
    # Getting the type of 'bpp' (line 545)
    bpp_406807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 65), 'bpp')
    # Applying the binary operator '+' (line 545)
    result_add_406808 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 24), '+', dot_call_result_406806, bpp_406807)
    
    # Assigning a type to the variable 'blockVectorBX' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'blockVectorBX', result_add_406808)
    
    # Assigning a Tuple to a Tuple (line 547):
    
    # Assigning a Name to a Name (line 547):
    # Getting the type of 'pp' (line 547)
    pp_406809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 53), 'pp')
    # Assigning a type to the variable 'tuple_assignment_405646' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'tuple_assignment_405646', pp_406809)
    
    # Assigning a Name to a Name (line 547):
    # Getting the type of 'app' (line 547)
    app_406810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 57), 'app')
    # Assigning a type to the variable 'tuple_assignment_405647' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'tuple_assignment_405647', app_406810)
    
    # Assigning a Name to a Name (line 547):
    # Getting the type of 'bpp' (line 547)
    bpp_406811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 62), 'bpp')
    # Assigning a type to the variable 'tuple_assignment_405648' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'tuple_assignment_405648', bpp_406811)
    
    # Assigning a Name to a Name (line 547):
    # Getting the type of 'tuple_assignment_405646' (line 547)
    tuple_assignment_405646_406812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'tuple_assignment_405646')
    # Assigning a type to the variable 'blockVectorP' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'blockVectorP', tuple_assignment_405646_406812)
    
    # Assigning a Name to a Name (line 547):
    # Getting the type of 'tuple_assignment_405647' (line 547)
    tuple_assignment_405647_406813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'tuple_assignment_405647')
    # Assigning a type to the variable 'blockVectorAP' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 22), 'blockVectorAP', tuple_assignment_405647_406813)
    
    # Assigning a Name to a Name (line 547):
    # Getting the type of 'tuple_assignment_405648' (line 547)
    tuple_assignment_405648_406814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'tuple_assignment_405648')
    # Assigning a type to the variable 'blockVectorBP' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 37), 'blockVectorBP', tuple_assignment_405648_406814)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 549):
    
    # Assigning a BinOp to a Name (line 549):
    # Getting the type of 'blockVectorBX' (line 549)
    blockVectorBX_406815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 10), 'blockVectorBX')
    
    # Obtaining the type of the subscript
    # Getting the type of 'np' (line 549)
    np_406816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 34), 'np')
    # Obtaining the member 'newaxis' of a type (line 549)
    newaxis_406817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 34), np_406816, 'newaxis')
    slice_406818 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 549, 26), None, None, None)
    # Getting the type of '_lambda' (line 549)
    _lambda_406819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 26), '_lambda')
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___406820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 26), _lambda_406819, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_406821 = invoke(stypy.reporting.localization.Localization(__file__, 549, 26), getitem___406820, (newaxis_406817, slice_406818))
    
    # Applying the binary operator '*' (line 549)
    result_mul_406822 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 10), '*', blockVectorBX_406815, subscript_call_result_406821)
    
    # Assigning a type to the variable 'aux' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'aux', result_mul_406822)
    
    # Assigning a BinOp to a Name (line 550):
    
    # Assigning a BinOp to a Name (line 550):
    # Getting the type of 'blockVectorAX' (line 550)
    blockVectorAX_406823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 19), 'blockVectorAX')
    # Getting the type of 'aux' (line 550)
    aux_406824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 35), 'aux')
    # Applying the binary operator '-' (line 550)
    result_sub_406825 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 19), '-', blockVectorAX_406823, aux_406824)
    
    # Assigning a type to the variable 'blockVectorR' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'blockVectorR', result_sub_406825)
    
    # Assigning a Call to a Name (line 552):
    
    # Assigning a Call to a Name (line 552):
    
    # Call to sum(...): (line 552)
    # Processing the call arguments (line 552)
    
    # Call to conjugate(...): (line 552)
    # Processing the call keyword arguments (line 552)
    kwargs_406830 = {}
    # Getting the type of 'blockVectorR' (line 552)
    blockVectorR_406828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 17), 'blockVectorR', False)
    # Obtaining the member 'conjugate' of a type (line 552)
    conjugate_406829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 17), blockVectorR_406828, 'conjugate')
    # Calling conjugate(args, kwargs) (line 552)
    conjugate_call_result_406831 = invoke(stypy.reporting.localization.Localization(__file__, 552, 17), conjugate_406829, *[], **kwargs_406830)
    
    # Getting the type of 'blockVectorR' (line 552)
    blockVectorR_406832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 44), 'blockVectorR', False)
    # Applying the binary operator '*' (line 552)
    result_mul_406833 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 17), '*', conjugate_call_result_406831, blockVectorR_406832)
    
    int_406834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 58), 'int')
    # Processing the call keyword arguments (line 552)
    kwargs_406835 = {}
    # Getting the type of 'np' (line 552)
    np_406826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 10), 'np', False)
    # Obtaining the member 'sum' of a type (line 552)
    sum_406827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 10), np_406826, 'sum')
    # Calling sum(args, kwargs) (line 552)
    sum_call_result_406836 = invoke(stypy.reporting.localization.Localization(__file__, 552, 10), sum_406827, *[result_mul_406833, int_406834], **kwargs_406835)
    
    # Assigning a type to the variable 'aux' (line 552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'aux', sum_call_result_406836)
    
    # Assigning a Call to a Name (line 553):
    
    # Assigning a Call to a Name (line 553):
    
    # Call to sqrt(...): (line 553)
    # Processing the call arguments (line 553)
    # Getting the type of 'aux' (line 553)
    aux_406839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 28), 'aux', False)
    # Processing the call keyword arguments (line 553)
    kwargs_406840 = {}
    # Getting the type of 'np' (line 553)
    np_406837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 20), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 553)
    sqrt_406838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 20), np_406837, 'sqrt')
    # Calling sqrt(args, kwargs) (line 553)
    sqrt_call_result_406841 = invoke(stypy.reporting.localization.Localization(__file__, 553, 20), sqrt_406838, *[aux_406839], **kwargs_406840)
    
    # Assigning a type to the variable 'residualNorms' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'residualNorms', sqrt_call_result_406841)
    
    
    # Getting the type of 'verbosityLevel' (line 555)
    verbosityLevel_406842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 7), 'verbosityLevel')
    int_406843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 24), 'int')
    # Applying the binary operator '>' (line 555)
    result_gt_406844 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 7), '>', verbosityLevel_406842, int_406843)
    
    # Testing the type of an if condition (line 555)
    if_condition_406845 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 555, 4), result_gt_406844)
    # Assigning a type to the variable 'if_condition_406845' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'if_condition_406845', if_condition_406845)
    # SSA begins for if statement (line 555)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 556)
    # Processing the call arguments (line 556)
    str_406847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 14), 'str', 'final eigenvalue:')
    # Getting the type of '_lambda' (line 556)
    _lambda_406848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 35), '_lambda', False)
    # Processing the call keyword arguments (line 556)
    kwargs_406849 = {}
    # Getting the type of 'print' (line 556)
    print_406846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'print', False)
    # Calling print(args, kwargs) (line 556)
    print_call_result_406850 = invoke(stypy.reporting.localization.Localization(__file__, 556, 8), print_406846, *[str_406847, _lambda_406848], **kwargs_406849)
    
    
    # Call to print(...): (line 557)
    # Processing the call arguments (line 557)
    str_406852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 14), 'str', 'final residual norms:')
    # Getting the type of 'residualNorms' (line 557)
    residualNorms_406853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 39), 'residualNorms', False)
    # Processing the call keyword arguments (line 557)
    kwargs_406854 = {}
    # Getting the type of 'print' (line 557)
    print_406851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'print', False)
    # Calling print(args, kwargs) (line 557)
    print_call_result_406855 = invoke(stypy.reporting.localization.Localization(__file__, 557, 8), print_406851, *[str_406852, residualNorms_406853], **kwargs_406854)
    
    # SSA join for if statement (line 555)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'retLambdaHistory' (line 559)
    retLambdaHistory_406856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 7), 'retLambdaHistory')
    # Testing the type of an if condition (line 559)
    if_condition_406857 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 559, 4), retLambdaHistory_406856)
    # Assigning a type to the variable 'if_condition_406857' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'if_condition_406857', if_condition_406857)
    # SSA begins for if statement (line 559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'retResidualNormsHistory' (line 560)
    retResidualNormsHistory_406858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 11), 'retResidualNormsHistory')
    # Testing the type of an if condition (line 560)
    if_condition_406859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 560, 8), retResidualNormsHistory_406858)
    # Assigning a type to the variable 'if_condition_406859' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'if_condition_406859', if_condition_406859)
    # SSA begins for if statement (line 560)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 561)
    tuple_406860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 561)
    # Adding element type (line 561)
    # Getting the type of '_lambda' (line 561)
    _lambda_406861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 19), '_lambda')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 19), tuple_406860, _lambda_406861)
    # Adding element type (line 561)
    # Getting the type of 'blockVectorX' (line 561)
    blockVectorX_406862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 28), 'blockVectorX')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 19), tuple_406860, blockVectorX_406862)
    # Adding element type (line 561)
    # Getting the type of 'lambdaHistory' (line 561)
    lambdaHistory_406863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 42), 'lambdaHistory')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 19), tuple_406860, lambdaHistory_406863)
    # Adding element type (line 561)
    # Getting the type of 'residualNormsHistory' (line 561)
    residualNormsHistory_406864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 57), 'residualNormsHistory')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 19), tuple_406860, residualNormsHistory_406864)
    
    # Assigning a type to the variable 'stypy_return_type' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'stypy_return_type', tuple_406860)
    # SSA branch for the else part of an if statement (line 560)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 563)
    tuple_406865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 563)
    # Adding element type (line 563)
    # Getting the type of '_lambda' (line 563)
    _lambda_406866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 19), '_lambda')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 19), tuple_406865, _lambda_406866)
    # Adding element type (line 563)
    # Getting the type of 'blockVectorX' (line 563)
    blockVectorX_406867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 28), 'blockVectorX')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 19), tuple_406865, blockVectorX_406867)
    # Adding element type (line 563)
    # Getting the type of 'lambdaHistory' (line 563)
    lambdaHistory_406868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 42), 'lambdaHistory')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 19), tuple_406865, lambdaHistory_406868)
    
    # Assigning a type to the variable 'stypy_return_type' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'stypy_return_type', tuple_406865)
    # SSA join for if statement (line 560)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 559)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'retResidualNormsHistory' (line 565)
    retResidualNormsHistory_406869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 11), 'retResidualNormsHistory')
    # Testing the type of an if condition (line 565)
    if_condition_406870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 565, 8), retResidualNormsHistory_406869)
    # Assigning a type to the variable 'if_condition_406870' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'if_condition_406870', if_condition_406870)
    # SSA begins for if statement (line 565)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 566)
    tuple_406871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 566)
    # Adding element type (line 566)
    # Getting the type of '_lambda' (line 566)
    _lambda_406872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 19), '_lambda')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 19), tuple_406871, _lambda_406872)
    # Adding element type (line 566)
    # Getting the type of 'blockVectorX' (line 566)
    blockVectorX_406873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 28), 'blockVectorX')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 19), tuple_406871, blockVectorX_406873)
    # Adding element type (line 566)
    # Getting the type of 'residualNormsHistory' (line 566)
    residualNormsHistory_406874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 42), 'residualNormsHistory')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 19), tuple_406871, residualNormsHistory_406874)
    
    # Assigning a type to the variable 'stypy_return_type' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'stypy_return_type', tuple_406871)
    # SSA branch for the else part of an if statement (line 565)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 568)
    tuple_406875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 568)
    # Adding element type (line 568)
    # Getting the type of '_lambda' (line 568)
    _lambda_406876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 19), '_lambda')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 19), tuple_406875, _lambda_406876)
    # Adding element type (line 568)
    # Getting the type of 'blockVectorX' (line 568)
    blockVectorX_406877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 28), 'blockVectorX')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 19), tuple_406875, blockVectorX_406877)
    
    # Assigning a type to the variable 'stypy_return_type' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'stypy_return_type', tuple_406875)
    # SSA join for if statement (line 565)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 559)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'lobpcg(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lobpcg' in the type store
    # Getting the type of 'stypy_return_type' (line 109)
    stypy_return_type_406878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_406878)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lobpcg'
    return stypy_return_type_406878

# Assigning a type to the variable 'lobpcg' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'lobpcg', lobpcg)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
