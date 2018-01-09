
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2015, Pauli Virtanen <pav@iki.fi>
2: # Distributed under the same license as Scipy.
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from numpy.linalg import LinAlgError
8: from scipy._lib.six import xrange
9: from scipy.linalg import (get_blas_funcs, qr, solve, svd, qr_insert, lstsq)
10: from scipy.sparse.linalg.isolve.utils import make_system
11: 
12: 
13: __all__ = ['gcrotmk']
14: 
15: 
16: def _fgmres(matvec, v0, m, atol, lpsolve=None, rpsolve=None, cs=(), outer_v=(),
17:             prepend_outer_v=False):
18:     '''
19:     FGMRES Arnoldi process, with optional projection or augmentation
20: 
21:     Parameters
22:     ----------
23:     matvec : callable
24:         Operation A*x
25:     v0 : ndarray
26:         Initial vector, normalized to nrm2(v0) == 1
27:     m : int
28:         Number of GMRES rounds
29:     atol : float
30:         Absolute tolerance for early exit
31:     lpsolve : callable
32:         Left preconditioner L
33:     rpsolve : callable
34:         Right preconditioner R
35:     CU : list of (ndarray, ndarray)
36:         Columns of matrices C and U in GCROT
37:     outer_v : list of ndarrays
38:         Augmentation vectors in LGMRES
39:     prepend_outer_v : bool, optional
40:         Whether augmentation vectors come before or after 
41:         Krylov iterates
42: 
43:     Raises
44:     ------
45:     LinAlgError
46:         If nans encountered
47: 
48:     Returns
49:     -------
50:     Q, R : ndarray
51:         QR decomposition of the upper Hessenberg H=QR
52:     B : ndarray
53:         Projections corresponding to matrix C
54:     vs : list of ndarray
55:         Columns of matrix V
56:     zs : list of ndarray
57:         Columns of matrix Z
58:     y : ndarray
59:         Solution to ||H y - e_1||_2 = min!
60: 
61:     '''
62: 
63:     if lpsolve is None:
64:         lpsolve = lambda x: x
65:     if rpsolve is None:
66:         rpsolve = lambda x: x
67: 
68:     axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (v0,))
69: 
70:     vs = [v0]
71:     zs = []
72:     y = None
73: 
74:     m = m + len(outer_v)
75: 
76:     # Orthogonal projection coefficients
77:     B = np.zeros((len(cs), m), dtype=v0.dtype)
78: 
79:     # H is stored in QR factorized form
80:     Q = np.ones((1, 1), dtype=v0.dtype)
81:     R = np.zeros((1, 0), dtype=v0.dtype)
82: 
83:     eps = np.finfo(v0.dtype).eps
84: 
85:     breakdown = False
86: 
87:     # FGMRES Arnoldi process
88:     for j in xrange(m):
89:         # L A Z = C B + V H
90: 
91:         if prepend_outer_v and j < len(outer_v):
92:             z, w = outer_v[j]
93:         elif prepend_outer_v and j == len(outer_v):
94:             z = rpsolve(v0)
95:             w = None
96:         elif not prepend_outer_v and j >= m - len(outer_v):
97:             z, w = outer_v[j - (m - len(outer_v))]
98:         else:
99:             z = rpsolve(vs[-1])
100:             w = None
101: 
102:         if w is None:
103:             w = lpsolve(matvec(z))
104:         else:
105:             # w is clobbered below
106:             w = w.copy()
107: 
108:         w_norm = nrm2(w)
109: 
110:         # GCROT projection: L A -> (1 - C C^H) L A
111:         # i.e. orthogonalize against C
112:         for i, c in enumerate(cs):
113:             alpha = dot(c, w)
114:             B[i,j] = alpha
115:             w = axpy(c, w, c.shape[0], -alpha)  # w -= alpha*c
116: 
117:         # Orthogonalize against V
118:         hcur = np.zeros(j+2, dtype=Q.dtype)
119:         for i, v in enumerate(vs):
120:             alpha = dot(v, w)
121:             hcur[i] = alpha
122:             w = axpy(v, w, v.shape[0], -alpha)  # w -= alpha*v
123:         hcur[i+1] = nrm2(w)
124: 
125:         with np.errstate(over='ignore', divide='ignore'):
126:             # Careful with denormals
127:             alpha = 1/hcur[-1]
128: 
129:         if np.isfinite(alpha):
130:             w = scal(alpha, w)
131: 
132:         if not (hcur[-1] > eps * w_norm):
133:             # w essentially in the span of previous vectors,
134:             # or we have nans. Bail out after updating the QR
135:             # solution.
136:             breakdown = True
137: 
138:         vs.append(w)
139:         zs.append(z)
140: 
141:         # Arnoldi LSQ problem
142: 
143:         # Add new column to H=Q*R, padding other columns with zeros
144:         Q2 = np.zeros((j+2, j+2), dtype=Q.dtype, order='F')
145:         Q2[:j+1,:j+1] = Q
146:         Q2[j+1,j+1] = 1
147: 
148:         R2 = np.zeros((j+2, j), dtype=R.dtype, order='F')
149:         R2[:j+1,:] = R
150: 
151:         Q, R = qr_insert(Q2, R2, hcur, j, which='col',
152:                          overwrite_qru=True, check_finite=False)
153: 
154:         # Transformed least squares problem
155:         # || Q R y - inner_res_0 * e_1 ||_2 = min!
156:         # Since R = [R'; 0], solution is y = inner_res_0 (R')^{-1} (Q^H)[:j,0]
157: 
158:         # Residual is immediately known
159:         res = abs(Q[0,-1])
160: 
161:         # Check for termination
162:         if res < atol or breakdown:
163:             break
164: 
165:     if not np.isfinite(R[j,j]):
166:         # nans encountered, bail out
167:         raise LinAlgError()
168: 
169:     # -- Get the LSQ problem solution
170: 
171:     # The problem is triangular, but the condition number may be
172:     # bad (or in case of breakdown the last diagonal entry may be
173:     # zero), so use lstsq instead of trtrs.
174:     y, _, _, _, = lstsq(R[:j+1,:j+1], Q[0,:j+1].conj())
175: 
176:     B = B[:,:j+1]
177: 
178:     return Q, R, B, vs, zs, y
179: 
180: 
181: def gcrotmk(A, b, x0=None, tol=1e-5, maxiter=1000, M=None, callback=None,
182:             m=20, k=None, CU=None, discard_C=False, truncate='oldest'):
183:     '''
184:     Solve a matrix equation using flexible GCROT(m,k) algorithm.
185: 
186:     Parameters
187:     ----------
188:     A : {sparse matrix, dense matrix, LinearOperator}
189:         The real or complex N-by-N matrix of the linear system.
190:     b : {array, matrix}
191:         Right hand side of the linear system. Has shape (N,) or (N,1).
192:     x0  : {array, matrix}
193:         Starting guess for the solution.
194:     tol : float, optional
195:         Tolerance to achieve. The algorithm terminates when either the relative
196:         or the absolute residual is below `tol`.
197:     maxiter : int, optional
198:         Maximum number of iterations.  Iteration will stop after maxiter
199:         steps even if the specified tolerance has not been achieved.
200:     M : {sparse matrix, dense matrix, LinearOperator}, optional
201:         Preconditioner for A.  The preconditioner should approximate the
202:         inverse of A. gcrotmk is a 'flexible' algorithm and the preconditioner
203:         can vary from iteration to iteration. Effective preconditioning
204:         dramatically improves the rate of convergence, which implies that
205:         fewer iterations are needed to reach a given error tolerance.
206:     callback : function, optional
207:         User-supplied function to call after each iteration.  It is called
208:         as callback(xk), where xk is the current solution vector.
209:     m : int, optional
210:         Number of inner FGMRES iterations per each outer iteration.
211:         Default: 20
212:     k : int, optional
213:         Number of vectors to carry between inner FGMRES iterations.
214:         According to [2]_, good values are around m.
215:         Default: m
216:     CU : list of tuples, optional
217:         List of tuples ``(c, u)`` which contain the columns of the matrices
218:         C and U in the GCROT(m,k) algorithm. For details, see [2]_.
219:         The list given and vectors contained in it are modified in-place.
220:         If not given, start from empty matrices. The ``c`` elements in the
221:         tuples can be ``None``, in which case the vectors are recomputed
222:         via ``c = A u`` on start and orthogonalized as described in [3]_.
223:     discard_C : bool, optional
224:         Discard the C-vectors at the end. Useful if recycling Krylov subspaces
225:         for different linear systems.
226:     truncate : {'oldest', 'smallest'}, optional
227:         Truncation scheme to use. Drop: oldest vectors, or vectors with
228:         smallest singular values using the scheme discussed in [1,2].
229:         See [2]_ for detailed comparison.
230:         Default: 'oldest'
231: 
232:     Returns
233:     -------
234:     x : array or matrix
235:         The solution found.
236:     info : int
237:         Provides convergence information:
238: 
239:         * 0  : successful exit
240:         * >0 : convergence to tolerance not achieved, number of iterations
241: 
242:     References
243:     ----------
244:     .. [1] E. de Sturler, ''Truncation strategies for optimal Krylov subspace
245:            methods'', SIAM J. Numer. Anal. 36, 864 (1999).
246:     .. [2] J.E. Hicken and D.W. Zingg, ''A simplified and flexible variant
247:            of GCROT for solving nonsymmetric linear systems'',
248:            SIAM J. Sci. Comput. 32, 172 (2010).
249:     .. [3] M.L. Parks, E. de Sturler, G. Mackey, D.D. Johnson, S. Maiti,
250:            ''Recycling Krylov subspaces for sequences of linear systems'',
251:            SIAM J. Sci. Comput. 28, 1651 (2006).
252: 
253:     '''
254:     A,M,x,b,postprocess = make_system(A,M,x0,b)
255: 
256:     if not np.isfinite(b).all():
257:         raise ValueError("RHS must contain only finite numbers")
258: 
259:     if truncate not in ('oldest', 'smallest'):
260:         raise ValueError("Invalid value for 'truncate': %r" % (truncate,))
261: 
262:     matvec = A.matvec
263:     psolve = M.matvec
264: 
265:     if CU is None:
266:         CU = []
267: 
268:     if k is None:
269:         k = m
270: 
271:     axpy, dot, scal = None, None, None
272: 
273:     r = b - matvec(x)
274: 
275:     axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (x, r))
276: 
277:     b_norm = nrm2(b)
278:     if b_norm == 0:
279:         b_norm = 1
280: 
281:     if discard_C:
282:         CU[:] = [(None, u) for c, u in CU]
283: 
284:     # Reorthogonalize old vectors
285:     if CU:
286:         # Sort already existing vectors to the front
287:         CU.sort(key=lambda cu: cu[0] is not None)
288: 
289:         # Fill-in missing ones
290:         C = np.empty((A.shape[0], len(CU)), dtype=r.dtype, order='F')
291:         us = []
292:         j = 0
293:         while CU:
294:             # More memory-efficient: throw away old vectors as we go
295:             c, u = CU.pop(0)
296:             if c is None:
297:                 c = matvec(u)
298:             C[:,j] = c
299:             j += 1
300:             us.append(u)
301: 
302:         # Orthogonalize
303:         Q, R, P = qr(C, overwrite_a=True, mode='economic', pivoting=True)
304:         del C
305: 
306:         # C := Q
307:         cs = list(Q.T)
308: 
309:         # U := U P R^-1,  back-substitution
310:         new_us = []
311:         for j in xrange(len(cs)):
312:             u = us[P[j]]
313:             for i in xrange(j):
314:                 u = axpy(us[P[i]], u, u.shape[0], -R[i,j])
315:             if abs(R[j,j]) < 1e-12 * abs(R[0,0]):
316:                 # discard rest of the vectors
317:                 break
318:             u = scal(1.0/R[j,j], u)
319:             new_us.append(u)
320: 
321:         # Form the new CU lists
322:         CU[:] = list(zip(cs, new_us))[::-1]
323: 
324:     if CU:
325:         axpy, dot = get_blas_funcs(['axpy', 'dot'], (r,))
326: 
327:         # Solve first the projection operation with respect to the CU
328:         # vectors. This corresponds to modifying the initial guess to
329:         # be
330:         #
331:         #     x' = x + U y
332:         #     y = argmin_y || b - A (x + U y) ||^2
333:         #
334:         # The solution is y = C^H (b - A x)
335:         for c, u in CU:
336:             yc = dot(c, r)
337:             x = axpy(u, x, x.shape[0], yc)
338:             r = axpy(c, r, r.shape[0], -yc)
339: 
340:     # GCROT main iteration
341:     for j_outer in xrange(maxiter):
342:         # -- callback
343:         if callback is not None:
344:             callback(x)
345: 
346:         beta = nrm2(r)
347: 
348:         # -- check stopping condition
349:         if beta <= max(tol, tol * b_norm):
350:             j_outer = -1
351:             break
352: 
353:         ml = m + max(k - len(CU), 0)
354: 
355:         cs = [c for c, u in CU]
356: 
357:         try:
358:             Q, R, B, vs, zs, y = _fgmres(matvec,
359:                                         r/beta,
360:                                         ml,
361:                                         rpsolve=psolve,
362:                                         atol=tol*b_norm/beta,
363:                                         cs=cs)
364:             y *= beta
365:         except LinAlgError:
366:             # Floating point over/underflow, non-finite result from
367:             # matmul etc. -- report failure.
368:             break
369: 
370:         #
371:         # At this point,
372:         #
373:         #     [A U, A Z] = [C, V] G;   G =  [ I  B ]
374:         #                                   [ 0  H ]
375:         #
376:         # where [C, V] has orthonormal columns, and r = beta v_0. Moreover,
377:         #
378:         #     || b - A (x + Z y + U q) ||_2 = || r - C B y - V H y - C q ||_2 = min!
379:         #
380:         # from which y = argmin_y || beta e_1 - H y ||_2, and q = -B y
381:         #
382: 
383:         #
384:         # GCROT(m,k) update
385:         #
386: 
387:         # Define new outer vectors
388: 
389:         # ux := (Z - U B) y
390:         ux = zs[0]*y[0]
391:         for z, yc in zip(zs[1:], y[1:]):
392:             ux = axpy(z, ux, ux.shape[0], yc)  # ux += z*yc
393:         by = B.dot(y)
394:         for cu, byc in zip(CU, by):
395:             c, u = cu
396:             ux = axpy(u, ux, ux.shape[0], -byc)  # ux -= u*byc
397: 
398:         # cx := V H y
399:         hy = Q.dot(R.dot(y))
400:         cx = vs[0] * hy[0]
401:         for v, hyc in zip(vs[1:], hy[1:]):
402:             cx = axpy(v, cx, cx.shape[0], hyc)  # cx += v*hyc
403: 
404:         # Normalize cx, maintaining cx = A ux
405:         # This new cx is orthogonal to the previous C, by construction
406:         try:
407:             alpha = 1/nrm2(cx)
408:             if not np.isfinite(alpha):
409:                 raise FloatingPointError()
410:         except (FloatingPointError, ZeroDivisionError):
411:             # Cannot update, so skip it
412:             continue
413: 
414:         cx = scal(alpha, cx)
415:         ux = scal(alpha, ux)
416: 
417:         # Update residual and solution
418:         gamma = dot(cx, r)
419:         r = axpy(cx, r, r.shape[0], -gamma)  # r -= gamma*cx
420:         x = axpy(ux, x, x.shape[0], gamma)  # x += gamma*ux
421: 
422:         # Truncate CU
423:         if truncate == 'oldest':
424:             while len(CU) >= k and CU:
425:                 del CU[0]
426:         elif truncate == 'smallest':
427:             if len(CU) >= k and CU:
428:                 # cf. [1,2]
429:                 D = solve(R[:-1,:].T, B.T).T
430:                 W, sigma, V = svd(D)
431: 
432:                 # C := C W[:,:k-1],  U := U W[:,:k-1]
433:                 new_CU = []
434:                 for j, w in enumerate(W[:,:k-1].T):
435:                     c, u = CU[0]
436:                     c = c * w[0]
437:                     u = u * w[0]
438:                     for cup, wp in zip(CU[1:], w[1:]):
439:                         cp, up = cup
440:                         c = axpy(cp, c, c.shape[0], wp)
441:                         u = axpy(up, u, u.shape[0], wp)
442: 
443:                     # Reorthogonalize at the same time; not necessary
444:                     # in exact arithmetic, but floating point error
445:                     # tends to accumulate here
446:                     for cp, up in new_CU:
447:                         alpha = dot(cp, c)
448:                         c = axpy(cp, c, c.shape[0], -alpha)
449:                         u = axpy(up, u, u.shape[0], -alpha)
450:                     alpha = nrm2(c)
451:                     c = scal(1.0/alpha, c)
452:                     u = scal(1.0/alpha, u)
453: 
454:                     new_CU.append((c, u))
455:                 CU[:] = new_CU
456: 
457:         # Add new vector to CU
458:         CU.append((cx, ux))
459: 
460:     # Include the solution vector to the span
461:     CU.append((None, x.copy()))
462:     if discard_C:
463:         CU[:] = [(None, uz) for cz, uz in CU]
464: 
465:     return postprocess(x), j_outer + 1
466: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_414659 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_414659) is not StypyTypeError):

    if (import_414659 != 'pyd_module'):
        __import__(import_414659)
        sys_modules_414660 = sys.modules[import_414659]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_414660.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_414659)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.linalg import LinAlgError' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_414661 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg')

if (type(import_414661) is not StypyTypeError):

    if (import_414661 != 'pyd_module'):
        __import__(import_414661)
        sys_modules_414662 = sys.modules[import_414661]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', sys_modules_414662.module_type_store, module_type_store, ['LinAlgError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_414662, sys_modules_414662.module_type_store, module_type_store)
    else:
        from numpy.linalg import LinAlgError

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', None, module_type_store, ['LinAlgError'], [LinAlgError])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', import_414661)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy._lib.six import xrange' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_414663 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six')

if (type(import_414663) is not StypyTypeError):

    if (import_414663 != 'pyd_module'):
        __import__(import_414663)
        sys_modules_414664 = sys.modules[import_414663]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', sys_modules_414664.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_414664, sys_modules_414664.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', import_414663)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.linalg import get_blas_funcs, qr, solve, svd, qr_insert, lstsq' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_414665 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg')

if (type(import_414665) is not StypyTypeError):

    if (import_414665 != 'pyd_module'):
        __import__(import_414665)
        sys_modules_414666 = sys.modules[import_414665]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', sys_modules_414666.module_type_store, module_type_store, ['get_blas_funcs', 'qr', 'solve', 'svd', 'qr_insert', 'lstsq'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_414666, sys_modules_414666.module_type_store, module_type_store)
    else:
        from scipy.linalg import get_blas_funcs, qr, solve, svd, qr_insert, lstsq

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', None, module_type_store, ['get_blas_funcs', 'qr', 'solve', 'svd', 'qr_insert', 'lstsq'], [get_blas_funcs, qr, solve, svd, qr_insert, lstsq])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', import_414665)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.sparse.linalg.isolve.utils import make_system' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_414667 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg.isolve.utils')

if (type(import_414667) is not StypyTypeError):

    if (import_414667 != 'pyd_module'):
        __import__(import_414667)
        sys_modules_414668 = sys.modules[import_414667]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg.isolve.utils', sys_modules_414668.module_type_store, module_type_store, ['make_system'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_414668, sys_modules_414668.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve.utils import make_system

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg.isolve.utils', None, module_type_store, ['make_system'], [make_system])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve.utils' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg.isolve.utils', import_414667)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')


# Assigning a List to a Name (line 13):

# Assigning a List to a Name (line 13):
__all__ = ['gcrotmk']
module_type_store.set_exportable_members(['gcrotmk'])

# Obtaining an instance of the builtin type 'list' (line 13)
list_414669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
str_414670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'gcrotmk')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_414669, str_414670)

# Assigning a type to the variable '__all__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__all__', list_414669)

@norecursion
def _fgmres(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 16)
    None_414671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 41), 'None')
    # Getting the type of 'None' (line 16)
    None_414672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 55), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_414673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 64), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_414674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 76), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    
    # Getting the type of 'False' (line 17)
    False_414675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 28), 'False')
    defaults = [None_414671, None_414672, tuple_414673, tuple_414674, False_414675]
    # Create a new context for function '_fgmres'
    module_type_store = module_type_store.open_function_context('_fgmres', 16, 0, False)
    
    # Passed parameters checking function
    _fgmres.stypy_localization = localization
    _fgmres.stypy_type_of_self = None
    _fgmres.stypy_type_store = module_type_store
    _fgmres.stypy_function_name = '_fgmres'
    _fgmres.stypy_param_names_list = ['matvec', 'v0', 'm', 'atol', 'lpsolve', 'rpsolve', 'cs', 'outer_v', 'prepend_outer_v']
    _fgmres.stypy_varargs_param_name = None
    _fgmres.stypy_kwargs_param_name = None
    _fgmres.stypy_call_defaults = defaults
    _fgmres.stypy_call_varargs = varargs
    _fgmres.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fgmres', ['matvec', 'v0', 'm', 'atol', 'lpsolve', 'rpsolve', 'cs', 'outer_v', 'prepend_outer_v'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fgmres', localization, ['matvec', 'v0', 'm', 'atol', 'lpsolve', 'rpsolve', 'cs', 'outer_v', 'prepend_outer_v'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fgmres(...)' code ##################

    str_414676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'str', '\n    FGMRES Arnoldi process, with optional projection or augmentation\n\n    Parameters\n    ----------\n    matvec : callable\n        Operation A*x\n    v0 : ndarray\n        Initial vector, normalized to nrm2(v0) == 1\n    m : int\n        Number of GMRES rounds\n    atol : float\n        Absolute tolerance for early exit\n    lpsolve : callable\n        Left preconditioner L\n    rpsolve : callable\n        Right preconditioner R\n    CU : list of (ndarray, ndarray)\n        Columns of matrices C and U in GCROT\n    outer_v : list of ndarrays\n        Augmentation vectors in LGMRES\n    prepend_outer_v : bool, optional\n        Whether augmentation vectors come before or after \n        Krylov iterates\n\n    Raises\n    ------\n    LinAlgError\n        If nans encountered\n\n    Returns\n    -------\n    Q, R : ndarray\n        QR decomposition of the upper Hessenberg H=QR\n    B : ndarray\n        Projections corresponding to matrix C\n    vs : list of ndarray\n        Columns of matrix V\n    zs : list of ndarray\n        Columns of matrix Z\n    y : ndarray\n        Solution to ||H y - e_1||_2 = min!\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 63)
    # Getting the type of 'lpsolve' (line 63)
    lpsolve_414677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 7), 'lpsolve')
    # Getting the type of 'None' (line 63)
    None_414678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'None')
    
    (may_be_414679, more_types_in_union_414680) = may_be_none(lpsolve_414677, None_414678)

    if may_be_414679:

        if more_types_in_union_414680:
            # Runtime conditional SSA (line 63)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Lambda to a Name (line 64):
        
        # Assigning a Lambda to a Name (line 64):

        @norecursion
        def _stypy_temp_lambda_222(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_222'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_222', 64, 18, True)
            # Passed parameters checking function
            _stypy_temp_lambda_222.stypy_localization = localization
            _stypy_temp_lambda_222.stypy_type_of_self = None
            _stypy_temp_lambda_222.stypy_type_store = module_type_store
            _stypy_temp_lambda_222.stypy_function_name = '_stypy_temp_lambda_222'
            _stypy_temp_lambda_222.stypy_param_names_list = ['x']
            _stypy_temp_lambda_222.stypy_varargs_param_name = None
            _stypy_temp_lambda_222.stypy_kwargs_param_name = None
            _stypy_temp_lambda_222.stypy_call_defaults = defaults
            _stypy_temp_lambda_222.stypy_call_varargs = varargs
            _stypy_temp_lambda_222.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_222', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_222', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 64)
            x_414681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'x')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'stypy_return_type', x_414681)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_222' in the type store
            # Getting the type of 'stypy_return_type' (line 64)
            stypy_return_type_414682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_414682)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_222'
            return stypy_return_type_414682

        # Assigning a type to the variable '_stypy_temp_lambda_222' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), '_stypy_temp_lambda_222', _stypy_temp_lambda_222)
        # Getting the type of '_stypy_temp_lambda_222' (line 64)
        _stypy_temp_lambda_222_414683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), '_stypy_temp_lambda_222')
        # Assigning a type to the variable 'lpsolve' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'lpsolve', _stypy_temp_lambda_222_414683)

        if more_types_in_union_414680:
            # SSA join for if statement (line 63)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 65)
    # Getting the type of 'rpsolve' (line 65)
    rpsolve_414684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 7), 'rpsolve')
    # Getting the type of 'None' (line 65)
    None_414685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 18), 'None')
    
    (may_be_414686, more_types_in_union_414687) = may_be_none(rpsolve_414684, None_414685)

    if may_be_414686:

        if more_types_in_union_414687:
            # Runtime conditional SSA (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Lambda to a Name (line 66):
        
        # Assigning a Lambda to a Name (line 66):

        @norecursion
        def _stypy_temp_lambda_223(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_223'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_223', 66, 18, True)
            # Passed parameters checking function
            _stypy_temp_lambda_223.stypy_localization = localization
            _stypy_temp_lambda_223.stypy_type_of_self = None
            _stypy_temp_lambda_223.stypy_type_store = module_type_store
            _stypy_temp_lambda_223.stypy_function_name = '_stypy_temp_lambda_223'
            _stypy_temp_lambda_223.stypy_param_names_list = ['x']
            _stypy_temp_lambda_223.stypy_varargs_param_name = None
            _stypy_temp_lambda_223.stypy_kwargs_param_name = None
            _stypy_temp_lambda_223.stypy_call_defaults = defaults
            _stypy_temp_lambda_223.stypy_call_varargs = varargs
            _stypy_temp_lambda_223.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_223', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_223', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 66)
            x_414688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'x')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 18), 'stypy_return_type', x_414688)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_223' in the type store
            # Getting the type of 'stypy_return_type' (line 66)
            stypy_return_type_414689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 18), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_414689)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_223'
            return stypy_return_type_414689

        # Assigning a type to the variable '_stypy_temp_lambda_223' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 18), '_stypy_temp_lambda_223', _stypy_temp_lambda_223)
        # Getting the type of '_stypy_temp_lambda_223' (line 66)
        _stypy_temp_lambda_223_414690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 18), '_stypy_temp_lambda_223')
        # Assigning a type to the variable 'rpsolve' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'rpsolve', _stypy_temp_lambda_223_414690)

        if more_types_in_union_414687:
            # SSA join for if statement (line 65)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 68):
    
    # Assigning a Subscript to a Name (line 68):
    
    # Obtaining the type of the subscript
    int_414691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'int')
    
    # Call to get_blas_funcs(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_414693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    str_414694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 44), 'str', 'axpy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414693, str_414694)
    # Adding element type (line 68)
    str_414695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 52), 'str', 'dot')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414693, str_414695)
    # Adding element type (line 68)
    str_414696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 59), 'str', 'scal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414693, str_414696)
    # Adding element type (line 68)
    str_414697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 67), 'str', 'nrm2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414693, str_414697)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 68)
    tuple_414698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 77), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 68)
    # Adding element type (line 68)
    # Getting the type of 'v0' (line 68)
    v0_414699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 77), 'v0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 77), tuple_414698, v0_414699)
    
    # Processing the call keyword arguments (line 68)
    kwargs_414700 = {}
    # Getting the type of 'get_blas_funcs' (line 68)
    get_blas_funcs_414692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 28), 'get_blas_funcs', False)
    # Calling get_blas_funcs(args, kwargs) (line 68)
    get_blas_funcs_call_result_414701 = invoke(stypy.reporting.localization.Localization(__file__, 68, 28), get_blas_funcs_414692, *[list_414693, tuple_414698], **kwargs_414700)
    
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___414702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), get_blas_funcs_call_result_414701, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_414703 = invoke(stypy.reporting.localization.Localization(__file__, 68, 4), getitem___414702, int_414691)
    
    # Assigning a type to the variable 'tuple_var_assignment_414611' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_414611', subscript_call_result_414703)
    
    # Assigning a Subscript to a Name (line 68):
    
    # Obtaining the type of the subscript
    int_414704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'int')
    
    # Call to get_blas_funcs(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_414706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    str_414707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 44), 'str', 'axpy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414706, str_414707)
    # Adding element type (line 68)
    str_414708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 52), 'str', 'dot')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414706, str_414708)
    # Adding element type (line 68)
    str_414709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 59), 'str', 'scal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414706, str_414709)
    # Adding element type (line 68)
    str_414710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 67), 'str', 'nrm2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414706, str_414710)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 68)
    tuple_414711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 77), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 68)
    # Adding element type (line 68)
    # Getting the type of 'v0' (line 68)
    v0_414712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 77), 'v0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 77), tuple_414711, v0_414712)
    
    # Processing the call keyword arguments (line 68)
    kwargs_414713 = {}
    # Getting the type of 'get_blas_funcs' (line 68)
    get_blas_funcs_414705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 28), 'get_blas_funcs', False)
    # Calling get_blas_funcs(args, kwargs) (line 68)
    get_blas_funcs_call_result_414714 = invoke(stypy.reporting.localization.Localization(__file__, 68, 28), get_blas_funcs_414705, *[list_414706, tuple_414711], **kwargs_414713)
    
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___414715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), get_blas_funcs_call_result_414714, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_414716 = invoke(stypy.reporting.localization.Localization(__file__, 68, 4), getitem___414715, int_414704)
    
    # Assigning a type to the variable 'tuple_var_assignment_414612' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_414612', subscript_call_result_414716)
    
    # Assigning a Subscript to a Name (line 68):
    
    # Obtaining the type of the subscript
    int_414717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'int')
    
    # Call to get_blas_funcs(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_414719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    str_414720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 44), 'str', 'axpy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414719, str_414720)
    # Adding element type (line 68)
    str_414721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 52), 'str', 'dot')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414719, str_414721)
    # Adding element type (line 68)
    str_414722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 59), 'str', 'scal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414719, str_414722)
    # Adding element type (line 68)
    str_414723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 67), 'str', 'nrm2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414719, str_414723)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 68)
    tuple_414724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 77), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 68)
    # Adding element type (line 68)
    # Getting the type of 'v0' (line 68)
    v0_414725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 77), 'v0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 77), tuple_414724, v0_414725)
    
    # Processing the call keyword arguments (line 68)
    kwargs_414726 = {}
    # Getting the type of 'get_blas_funcs' (line 68)
    get_blas_funcs_414718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 28), 'get_blas_funcs', False)
    # Calling get_blas_funcs(args, kwargs) (line 68)
    get_blas_funcs_call_result_414727 = invoke(stypy.reporting.localization.Localization(__file__, 68, 28), get_blas_funcs_414718, *[list_414719, tuple_414724], **kwargs_414726)
    
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___414728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), get_blas_funcs_call_result_414727, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_414729 = invoke(stypy.reporting.localization.Localization(__file__, 68, 4), getitem___414728, int_414717)
    
    # Assigning a type to the variable 'tuple_var_assignment_414613' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_414613', subscript_call_result_414729)
    
    # Assigning a Subscript to a Name (line 68):
    
    # Obtaining the type of the subscript
    int_414730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'int')
    
    # Call to get_blas_funcs(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_414732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    str_414733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 44), 'str', 'axpy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414732, str_414733)
    # Adding element type (line 68)
    str_414734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 52), 'str', 'dot')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414732, str_414734)
    # Adding element type (line 68)
    str_414735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 59), 'str', 'scal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414732, str_414735)
    # Adding element type (line 68)
    str_414736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 67), 'str', 'nrm2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_414732, str_414736)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 68)
    tuple_414737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 77), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 68)
    # Adding element type (line 68)
    # Getting the type of 'v0' (line 68)
    v0_414738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 77), 'v0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 77), tuple_414737, v0_414738)
    
    # Processing the call keyword arguments (line 68)
    kwargs_414739 = {}
    # Getting the type of 'get_blas_funcs' (line 68)
    get_blas_funcs_414731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 28), 'get_blas_funcs', False)
    # Calling get_blas_funcs(args, kwargs) (line 68)
    get_blas_funcs_call_result_414740 = invoke(stypy.reporting.localization.Localization(__file__, 68, 28), get_blas_funcs_414731, *[list_414732, tuple_414737], **kwargs_414739)
    
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___414741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), get_blas_funcs_call_result_414740, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_414742 = invoke(stypy.reporting.localization.Localization(__file__, 68, 4), getitem___414741, int_414730)
    
    # Assigning a type to the variable 'tuple_var_assignment_414614' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_414614', subscript_call_result_414742)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'tuple_var_assignment_414611' (line 68)
    tuple_var_assignment_414611_414743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_414611')
    # Assigning a type to the variable 'axpy' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'axpy', tuple_var_assignment_414611_414743)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'tuple_var_assignment_414612' (line 68)
    tuple_var_assignment_414612_414744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_414612')
    # Assigning a type to the variable 'dot' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 10), 'dot', tuple_var_assignment_414612_414744)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'tuple_var_assignment_414613' (line 68)
    tuple_var_assignment_414613_414745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_414613')
    # Assigning a type to the variable 'scal' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'scal', tuple_var_assignment_414613_414745)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'tuple_var_assignment_414614' (line 68)
    tuple_var_assignment_414614_414746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_414614')
    # Assigning a type to the variable 'nrm2' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'nrm2', tuple_var_assignment_414614_414746)
    
    # Assigning a List to a Name (line 70):
    
    # Assigning a List to a Name (line 70):
    
    # Obtaining an instance of the builtin type 'list' (line 70)
    list_414747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 70)
    # Adding element type (line 70)
    # Getting the type of 'v0' (line 70)
    v0_414748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 10), 'v0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 9), list_414747, v0_414748)
    
    # Assigning a type to the variable 'vs' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'vs', list_414747)
    
    # Assigning a List to a Name (line 71):
    
    # Assigning a List to a Name (line 71):
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_414749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    
    # Assigning a type to the variable 'zs' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'zs', list_414749)
    
    # Assigning a Name to a Name (line 72):
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'None' (line 72)
    None_414750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'None')
    # Assigning a type to the variable 'y' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'y', None_414750)
    
    # Assigning a BinOp to a Name (line 74):
    
    # Assigning a BinOp to a Name (line 74):
    # Getting the type of 'm' (line 74)
    m_414751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'm')
    
    # Call to len(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'outer_v' (line 74)
    outer_v_414753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'outer_v', False)
    # Processing the call keyword arguments (line 74)
    kwargs_414754 = {}
    # Getting the type of 'len' (line 74)
    len_414752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'len', False)
    # Calling len(args, kwargs) (line 74)
    len_call_result_414755 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), len_414752, *[outer_v_414753], **kwargs_414754)
    
    # Applying the binary operator '+' (line 74)
    result_add_414756 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 8), '+', m_414751, len_call_result_414755)
    
    # Assigning a type to the variable 'm' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'm', result_add_414756)
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to zeros(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Obtaining an instance of the builtin type 'tuple' (line 77)
    tuple_414759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 77)
    # Adding element type (line 77)
    
    # Call to len(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'cs' (line 77)
    cs_414761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), 'cs', False)
    # Processing the call keyword arguments (line 77)
    kwargs_414762 = {}
    # Getting the type of 'len' (line 77)
    len_414760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'len', False)
    # Calling len(args, kwargs) (line 77)
    len_call_result_414763 = invoke(stypy.reporting.localization.Localization(__file__, 77, 18), len_414760, *[cs_414761], **kwargs_414762)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 18), tuple_414759, len_call_result_414763)
    # Adding element type (line 77)
    # Getting the type of 'm' (line 77)
    m_414764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 18), tuple_414759, m_414764)
    
    # Processing the call keyword arguments (line 77)
    # Getting the type of 'v0' (line 77)
    v0_414765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 37), 'v0', False)
    # Obtaining the member 'dtype' of a type (line 77)
    dtype_414766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 37), v0_414765, 'dtype')
    keyword_414767 = dtype_414766
    kwargs_414768 = {'dtype': keyword_414767}
    # Getting the type of 'np' (line 77)
    np_414757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 77)
    zeros_414758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), np_414757, 'zeros')
    # Calling zeros(args, kwargs) (line 77)
    zeros_call_result_414769 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), zeros_414758, *[tuple_414759], **kwargs_414768)
    
    # Assigning a type to the variable 'B' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'B', zeros_call_result_414769)
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to ones(...): (line 80)
    # Processing the call arguments (line 80)
    
    # Obtaining an instance of the builtin type 'tuple' (line 80)
    tuple_414772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 80)
    # Adding element type (line 80)
    int_414773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 17), tuple_414772, int_414773)
    # Adding element type (line 80)
    int_414774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 17), tuple_414772, int_414774)
    
    # Processing the call keyword arguments (line 80)
    # Getting the type of 'v0' (line 80)
    v0_414775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'v0', False)
    # Obtaining the member 'dtype' of a type (line 80)
    dtype_414776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 30), v0_414775, 'dtype')
    keyword_414777 = dtype_414776
    kwargs_414778 = {'dtype': keyword_414777}
    # Getting the type of 'np' (line 80)
    np_414770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 80)
    ones_414771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), np_414770, 'ones')
    # Calling ones(args, kwargs) (line 80)
    ones_call_result_414779 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), ones_414771, *[tuple_414772], **kwargs_414778)
    
    # Assigning a type to the variable 'Q' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'Q', ones_call_result_414779)
    
    # Assigning a Call to a Name (line 81):
    
    # Assigning a Call to a Name (line 81):
    
    # Call to zeros(...): (line 81)
    # Processing the call arguments (line 81)
    
    # Obtaining an instance of the builtin type 'tuple' (line 81)
    tuple_414782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 81)
    # Adding element type (line 81)
    int_414783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 18), tuple_414782, int_414783)
    # Adding element type (line 81)
    int_414784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 18), tuple_414782, int_414784)
    
    # Processing the call keyword arguments (line 81)
    # Getting the type of 'v0' (line 81)
    v0_414785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 31), 'v0', False)
    # Obtaining the member 'dtype' of a type (line 81)
    dtype_414786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 31), v0_414785, 'dtype')
    keyword_414787 = dtype_414786
    kwargs_414788 = {'dtype': keyword_414787}
    # Getting the type of 'np' (line 81)
    np_414780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 81)
    zeros_414781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), np_414780, 'zeros')
    # Calling zeros(args, kwargs) (line 81)
    zeros_call_result_414789 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), zeros_414781, *[tuple_414782], **kwargs_414788)
    
    # Assigning a type to the variable 'R' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'R', zeros_call_result_414789)
    
    # Assigning a Attribute to a Name (line 83):
    
    # Assigning a Attribute to a Name (line 83):
    
    # Call to finfo(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'v0' (line 83)
    v0_414792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'v0', False)
    # Obtaining the member 'dtype' of a type (line 83)
    dtype_414793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 19), v0_414792, 'dtype')
    # Processing the call keyword arguments (line 83)
    kwargs_414794 = {}
    # Getting the type of 'np' (line 83)
    np_414790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 10), 'np', False)
    # Obtaining the member 'finfo' of a type (line 83)
    finfo_414791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 10), np_414790, 'finfo')
    # Calling finfo(args, kwargs) (line 83)
    finfo_call_result_414795 = invoke(stypy.reporting.localization.Localization(__file__, 83, 10), finfo_414791, *[dtype_414793], **kwargs_414794)
    
    # Obtaining the member 'eps' of a type (line 83)
    eps_414796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 10), finfo_call_result_414795, 'eps')
    # Assigning a type to the variable 'eps' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'eps', eps_414796)
    
    # Assigning a Name to a Name (line 85):
    
    # Assigning a Name to a Name (line 85):
    # Getting the type of 'False' (line 85)
    False_414797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'False')
    # Assigning a type to the variable 'breakdown' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'breakdown', False_414797)
    
    
    # Call to xrange(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'm' (line 88)
    m_414799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'm', False)
    # Processing the call keyword arguments (line 88)
    kwargs_414800 = {}
    # Getting the type of 'xrange' (line 88)
    xrange_414798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 88)
    xrange_call_result_414801 = invoke(stypy.reporting.localization.Localization(__file__, 88, 13), xrange_414798, *[m_414799], **kwargs_414800)
    
    # Testing the type of a for loop iterable (line 88)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 4), xrange_call_result_414801)
    # Getting the type of the for loop variable (line 88)
    for_loop_var_414802 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 4), xrange_call_result_414801)
    # Assigning a type to the variable 'j' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'j', for_loop_var_414802)
    # SSA begins for a for statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'prepend_outer_v' (line 91)
    prepend_outer_v_414803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'prepend_outer_v')
    
    # Getting the type of 'j' (line 91)
    j_414804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 31), 'j')
    
    # Call to len(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'outer_v' (line 91)
    outer_v_414806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 39), 'outer_v', False)
    # Processing the call keyword arguments (line 91)
    kwargs_414807 = {}
    # Getting the type of 'len' (line 91)
    len_414805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 35), 'len', False)
    # Calling len(args, kwargs) (line 91)
    len_call_result_414808 = invoke(stypy.reporting.localization.Localization(__file__, 91, 35), len_414805, *[outer_v_414806], **kwargs_414807)
    
    # Applying the binary operator '<' (line 91)
    result_lt_414809 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 31), '<', j_414804, len_call_result_414808)
    
    # Applying the binary operator 'and' (line 91)
    result_and_keyword_414810 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 11), 'and', prepend_outer_v_414803, result_lt_414809)
    
    # Testing the type of an if condition (line 91)
    if_condition_414811 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 8), result_and_keyword_414810)
    # Assigning a type to the variable 'if_condition_414811' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'if_condition_414811', if_condition_414811)
    # SSA begins for if statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Tuple (line 92):
    
    # Assigning a Subscript to a Name (line 92):
    
    # Obtaining the type of the subscript
    int_414812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 12), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 92)
    j_414813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'j')
    # Getting the type of 'outer_v' (line 92)
    outer_v_414814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'outer_v')
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___414815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 19), outer_v_414814, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_414816 = invoke(stypy.reporting.localization.Localization(__file__, 92, 19), getitem___414815, j_414813)
    
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___414817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), subscript_call_result_414816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_414818 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), getitem___414817, int_414812)
    
    # Assigning a type to the variable 'tuple_var_assignment_414615' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'tuple_var_assignment_414615', subscript_call_result_414818)
    
    # Assigning a Subscript to a Name (line 92):
    
    # Obtaining the type of the subscript
    int_414819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 12), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 92)
    j_414820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'j')
    # Getting the type of 'outer_v' (line 92)
    outer_v_414821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'outer_v')
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___414822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 19), outer_v_414821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_414823 = invoke(stypy.reporting.localization.Localization(__file__, 92, 19), getitem___414822, j_414820)
    
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___414824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), subscript_call_result_414823, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_414825 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), getitem___414824, int_414819)
    
    # Assigning a type to the variable 'tuple_var_assignment_414616' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'tuple_var_assignment_414616', subscript_call_result_414825)
    
    # Assigning a Name to a Name (line 92):
    # Getting the type of 'tuple_var_assignment_414615' (line 92)
    tuple_var_assignment_414615_414826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'tuple_var_assignment_414615')
    # Assigning a type to the variable 'z' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'z', tuple_var_assignment_414615_414826)
    
    # Assigning a Name to a Name (line 92):
    # Getting the type of 'tuple_var_assignment_414616' (line 92)
    tuple_var_assignment_414616_414827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'tuple_var_assignment_414616')
    # Assigning a type to the variable 'w' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'w', tuple_var_assignment_414616_414827)
    # SSA branch for the else part of an if statement (line 91)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'prepend_outer_v' (line 93)
    prepend_outer_v_414828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 13), 'prepend_outer_v')
    
    # Getting the type of 'j' (line 93)
    j_414829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 33), 'j')
    
    # Call to len(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'outer_v' (line 93)
    outer_v_414831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 42), 'outer_v', False)
    # Processing the call keyword arguments (line 93)
    kwargs_414832 = {}
    # Getting the type of 'len' (line 93)
    len_414830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'len', False)
    # Calling len(args, kwargs) (line 93)
    len_call_result_414833 = invoke(stypy.reporting.localization.Localization(__file__, 93, 38), len_414830, *[outer_v_414831], **kwargs_414832)
    
    # Applying the binary operator '==' (line 93)
    result_eq_414834 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 33), '==', j_414829, len_call_result_414833)
    
    # Applying the binary operator 'and' (line 93)
    result_and_keyword_414835 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 13), 'and', prepend_outer_v_414828, result_eq_414834)
    
    # Testing the type of an if condition (line 93)
    if_condition_414836 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 13), result_and_keyword_414835)
    # Assigning a type to the variable 'if_condition_414836' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 13), 'if_condition_414836', if_condition_414836)
    # SSA begins for if statement (line 93)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 94):
    
    # Assigning a Call to a Name (line 94):
    
    # Call to rpsolve(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'v0' (line 94)
    v0_414838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'v0', False)
    # Processing the call keyword arguments (line 94)
    kwargs_414839 = {}
    # Getting the type of 'rpsolve' (line 94)
    rpsolve_414837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'rpsolve', False)
    # Calling rpsolve(args, kwargs) (line 94)
    rpsolve_call_result_414840 = invoke(stypy.reporting.localization.Localization(__file__, 94, 16), rpsolve_414837, *[v0_414838], **kwargs_414839)
    
    # Assigning a type to the variable 'z' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'z', rpsolve_call_result_414840)
    
    # Assigning a Name to a Name (line 95):
    
    # Assigning a Name to a Name (line 95):
    # Getting the type of 'None' (line 95)
    None_414841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'None')
    # Assigning a type to the variable 'w' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'w', None_414841)
    # SSA branch for the else part of an if statement (line 93)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'prepend_outer_v' (line 96)
    prepend_outer_v_414842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'prepend_outer_v')
    # Applying the 'not' unary operator (line 96)
    result_not__414843 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 13), 'not', prepend_outer_v_414842)
    
    
    # Getting the type of 'j' (line 96)
    j_414844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 37), 'j')
    # Getting the type of 'm' (line 96)
    m_414845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 42), 'm')
    
    # Call to len(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'outer_v' (line 96)
    outer_v_414847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 50), 'outer_v', False)
    # Processing the call keyword arguments (line 96)
    kwargs_414848 = {}
    # Getting the type of 'len' (line 96)
    len_414846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 46), 'len', False)
    # Calling len(args, kwargs) (line 96)
    len_call_result_414849 = invoke(stypy.reporting.localization.Localization(__file__, 96, 46), len_414846, *[outer_v_414847], **kwargs_414848)
    
    # Applying the binary operator '-' (line 96)
    result_sub_414850 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 42), '-', m_414845, len_call_result_414849)
    
    # Applying the binary operator '>=' (line 96)
    result_ge_414851 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 37), '>=', j_414844, result_sub_414850)
    
    # Applying the binary operator 'and' (line 96)
    result_and_keyword_414852 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 13), 'and', result_not__414843, result_ge_414851)
    
    # Testing the type of an if condition (line 96)
    if_condition_414853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 13), result_and_keyword_414852)
    # Assigning a type to the variable 'if_condition_414853' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'if_condition_414853', if_condition_414853)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Tuple (line 97):
    
    # Assigning a Subscript to a Name (line 97):
    
    # Obtaining the type of the subscript
    int_414854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 12), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 97)
    j_414855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 27), 'j')
    # Getting the type of 'm' (line 97)
    m_414856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 32), 'm')
    
    # Call to len(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'outer_v' (line 97)
    outer_v_414858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'outer_v', False)
    # Processing the call keyword arguments (line 97)
    kwargs_414859 = {}
    # Getting the type of 'len' (line 97)
    len_414857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 36), 'len', False)
    # Calling len(args, kwargs) (line 97)
    len_call_result_414860 = invoke(stypy.reporting.localization.Localization(__file__, 97, 36), len_414857, *[outer_v_414858], **kwargs_414859)
    
    # Applying the binary operator '-' (line 97)
    result_sub_414861 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 32), '-', m_414856, len_call_result_414860)
    
    # Applying the binary operator '-' (line 97)
    result_sub_414862 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 27), '-', j_414855, result_sub_414861)
    
    # Getting the type of 'outer_v' (line 97)
    outer_v_414863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'outer_v')
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___414864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 19), outer_v_414863, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_414865 = invoke(stypy.reporting.localization.Localization(__file__, 97, 19), getitem___414864, result_sub_414862)
    
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___414866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), subscript_call_result_414865, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_414867 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), getitem___414866, int_414854)
    
    # Assigning a type to the variable 'tuple_var_assignment_414617' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'tuple_var_assignment_414617', subscript_call_result_414867)
    
    # Assigning a Subscript to a Name (line 97):
    
    # Obtaining the type of the subscript
    int_414868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 12), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 97)
    j_414869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 27), 'j')
    # Getting the type of 'm' (line 97)
    m_414870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 32), 'm')
    
    # Call to len(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'outer_v' (line 97)
    outer_v_414872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'outer_v', False)
    # Processing the call keyword arguments (line 97)
    kwargs_414873 = {}
    # Getting the type of 'len' (line 97)
    len_414871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 36), 'len', False)
    # Calling len(args, kwargs) (line 97)
    len_call_result_414874 = invoke(stypy.reporting.localization.Localization(__file__, 97, 36), len_414871, *[outer_v_414872], **kwargs_414873)
    
    # Applying the binary operator '-' (line 97)
    result_sub_414875 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 32), '-', m_414870, len_call_result_414874)
    
    # Applying the binary operator '-' (line 97)
    result_sub_414876 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 27), '-', j_414869, result_sub_414875)
    
    # Getting the type of 'outer_v' (line 97)
    outer_v_414877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'outer_v')
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___414878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 19), outer_v_414877, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_414879 = invoke(stypy.reporting.localization.Localization(__file__, 97, 19), getitem___414878, result_sub_414876)
    
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___414880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), subscript_call_result_414879, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_414881 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), getitem___414880, int_414868)
    
    # Assigning a type to the variable 'tuple_var_assignment_414618' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'tuple_var_assignment_414618', subscript_call_result_414881)
    
    # Assigning a Name to a Name (line 97):
    # Getting the type of 'tuple_var_assignment_414617' (line 97)
    tuple_var_assignment_414617_414882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'tuple_var_assignment_414617')
    # Assigning a type to the variable 'z' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'z', tuple_var_assignment_414617_414882)
    
    # Assigning a Name to a Name (line 97):
    # Getting the type of 'tuple_var_assignment_414618' (line 97)
    tuple_var_assignment_414618_414883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'tuple_var_assignment_414618')
    # Assigning a type to the variable 'w' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'w', tuple_var_assignment_414618_414883)
    # SSA branch for the else part of an if statement (line 96)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to rpsolve(...): (line 99)
    # Processing the call arguments (line 99)
    
    # Obtaining the type of the subscript
    int_414885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 27), 'int')
    # Getting the type of 'vs' (line 99)
    vs_414886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'vs', False)
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___414887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 24), vs_414886, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_414888 = invoke(stypy.reporting.localization.Localization(__file__, 99, 24), getitem___414887, int_414885)
    
    # Processing the call keyword arguments (line 99)
    kwargs_414889 = {}
    # Getting the type of 'rpsolve' (line 99)
    rpsolve_414884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'rpsolve', False)
    # Calling rpsolve(args, kwargs) (line 99)
    rpsolve_call_result_414890 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), rpsolve_414884, *[subscript_call_result_414888], **kwargs_414889)
    
    # Assigning a type to the variable 'z' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'z', rpsolve_call_result_414890)
    
    # Assigning a Name to a Name (line 100):
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'None' (line 100)
    None_414891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'None')
    # Assigning a type to the variable 'w' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'w', None_414891)
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 93)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 91)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 102)
    # Getting the type of 'w' (line 102)
    w_414892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'w')
    # Getting the type of 'None' (line 102)
    None_414893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'None')
    
    (may_be_414894, more_types_in_union_414895) = may_be_none(w_414892, None_414893)

    if may_be_414894:

        if more_types_in_union_414895:
            # Runtime conditional SSA (line 102)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to lpsolve(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to matvec(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'z' (line 103)
        z_414898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 31), 'z', False)
        # Processing the call keyword arguments (line 103)
        kwargs_414899 = {}
        # Getting the type of 'matvec' (line 103)
        matvec_414897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'matvec', False)
        # Calling matvec(args, kwargs) (line 103)
        matvec_call_result_414900 = invoke(stypy.reporting.localization.Localization(__file__, 103, 24), matvec_414897, *[z_414898], **kwargs_414899)
        
        # Processing the call keyword arguments (line 103)
        kwargs_414901 = {}
        # Getting the type of 'lpsolve' (line 103)
        lpsolve_414896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'lpsolve', False)
        # Calling lpsolve(args, kwargs) (line 103)
        lpsolve_call_result_414902 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), lpsolve_414896, *[matvec_call_result_414900], **kwargs_414901)
        
        # Assigning a type to the variable 'w' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'w', lpsolve_call_result_414902)

        if more_types_in_union_414895:
            # Runtime conditional SSA for else branch (line 102)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_414894) or more_types_in_union_414895):
        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to copy(...): (line 106)
        # Processing the call keyword arguments (line 106)
        kwargs_414905 = {}
        # Getting the type of 'w' (line 106)
        w_414903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'w', False)
        # Obtaining the member 'copy' of a type (line 106)
        copy_414904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), w_414903, 'copy')
        # Calling copy(args, kwargs) (line 106)
        copy_call_result_414906 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), copy_414904, *[], **kwargs_414905)
        
        # Assigning a type to the variable 'w' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'w', copy_call_result_414906)

        if (may_be_414894 and more_types_in_union_414895):
            # SSA join for if statement (line 102)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 108):
    
    # Assigning a Call to a Name (line 108):
    
    # Call to nrm2(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'w' (line 108)
    w_414908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 22), 'w', False)
    # Processing the call keyword arguments (line 108)
    kwargs_414909 = {}
    # Getting the type of 'nrm2' (line 108)
    nrm2_414907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 17), 'nrm2', False)
    # Calling nrm2(args, kwargs) (line 108)
    nrm2_call_result_414910 = invoke(stypy.reporting.localization.Localization(__file__, 108, 17), nrm2_414907, *[w_414908], **kwargs_414909)
    
    # Assigning a type to the variable 'w_norm' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'w_norm', nrm2_call_result_414910)
    
    
    # Call to enumerate(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'cs' (line 112)
    cs_414912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'cs', False)
    # Processing the call keyword arguments (line 112)
    kwargs_414913 = {}
    # Getting the type of 'enumerate' (line 112)
    enumerate_414911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 112)
    enumerate_call_result_414914 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), enumerate_414911, *[cs_414912], **kwargs_414913)
    
    # Testing the type of a for loop iterable (line 112)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 112, 8), enumerate_call_result_414914)
    # Getting the type of the for loop variable (line 112)
    for_loop_var_414915 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 112, 8), enumerate_call_result_414914)
    # Assigning a type to the variable 'i' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 8), for_loop_var_414915))
    # Assigning a type to the variable 'c' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 8), for_loop_var_414915))
    # SSA begins for a for statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 113):
    
    # Assigning a Call to a Name (line 113):
    
    # Call to dot(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'c' (line 113)
    c_414917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'c', False)
    # Getting the type of 'w' (line 113)
    w_414918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'w', False)
    # Processing the call keyword arguments (line 113)
    kwargs_414919 = {}
    # Getting the type of 'dot' (line 113)
    dot_414916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'dot', False)
    # Calling dot(args, kwargs) (line 113)
    dot_call_result_414920 = invoke(stypy.reporting.localization.Localization(__file__, 113, 20), dot_414916, *[c_414917, w_414918], **kwargs_414919)
    
    # Assigning a type to the variable 'alpha' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'alpha', dot_call_result_414920)
    
    # Assigning a Name to a Subscript (line 114):
    
    # Assigning a Name to a Subscript (line 114):
    # Getting the type of 'alpha' (line 114)
    alpha_414921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'alpha')
    # Getting the type of 'B' (line 114)
    B_414922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'B')
    
    # Obtaining an instance of the builtin type 'tuple' (line 114)
    tuple_414923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 114)
    # Adding element type (line 114)
    # Getting the type of 'i' (line 114)
    i_414924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 14), tuple_414923, i_414924)
    # Adding element type (line 114)
    # Getting the type of 'j' (line 114)
    j_414925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 14), tuple_414923, j_414925)
    
    # Storing an element on a container (line 114)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 12), B_414922, (tuple_414923, alpha_414921))
    
    # Assigning a Call to a Name (line 115):
    
    # Assigning a Call to a Name (line 115):
    
    # Call to axpy(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'c' (line 115)
    c_414927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 21), 'c', False)
    # Getting the type of 'w' (line 115)
    w_414928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 24), 'w', False)
    
    # Obtaining the type of the subscript
    int_414929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 35), 'int')
    # Getting the type of 'c' (line 115)
    c_414930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'c', False)
    # Obtaining the member 'shape' of a type (line 115)
    shape_414931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 27), c_414930, 'shape')
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___414932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 27), shape_414931, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_414933 = invoke(stypy.reporting.localization.Localization(__file__, 115, 27), getitem___414932, int_414929)
    
    
    # Getting the type of 'alpha' (line 115)
    alpha_414934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 40), 'alpha', False)
    # Applying the 'usub' unary operator (line 115)
    result___neg___414935 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 39), 'usub', alpha_414934)
    
    # Processing the call keyword arguments (line 115)
    kwargs_414936 = {}
    # Getting the type of 'axpy' (line 115)
    axpy_414926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'axpy', False)
    # Calling axpy(args, kwargs) (line 115)
    axpy_call_result_414937 = invoke(stypy.reporting.localization.Localization(__file__, 115, 16), axpy_414926, *[c_414927, w_414928, subscript_call_result_414933, result___neg___414935], **kwargs_414936)
    
    # Assigning a type to the variable 'w' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'w', axpy_call_result_414937)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 118):
    
    # Assigning a Call to a Name (line 118):
    
    # Call to zeros(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'j' (line 118)
    j_414940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'j', False)
    int_414941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 26), 'int')
    # Applying the binary operator '+' (line 118)
    result_add_414942 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 24), '+', j_414940, int_414941)
    
    # Processing the call keyword arguments (line 118)
    # Getting the type of 'Q' (line 118)
    Q_414943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 35), 'Q', False)
    # Obtaining the member 'dtype' of a type (line 118)
    dtype_414944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 35), Q_414943, 'dtype')
    keyword_414945 = dtype_414944
    kwargs_414946 = {'dtype': keyword_414945}
    # Getting the type of 'np' (line 118)
    np_414938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'np', False)
    # Obtaining the member 'zeros' of a type (line 118)
    zeros_414939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 15), np_414938, 'zeros')
    # Calling zeros(args, kwargs) (line 118)
    zeros_call_result_414947 = invoke(stypy.reporting.localization.Localization(__file__, 118, 15), zeros_414939, *[result_add_414942], **kwargs_414946)
    
    # Assigning a type to the variable 'hcur' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'hcur', zeros_call_result_414947)
    
    
    # Call to enumerate(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'vs' (line 119)
    vs_414949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 30), 'vs', False)
    # Processing the call keyword arguments (line 119)
    kwargs_414950 = {}
    # Getting the type of 'enumerate' (line 119)
    enumerate_414948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 119)
    enumerate_call_result_414951 = invoke(stypy.reporting.localization.Localization(__file__, 119, 20), enumerate_414948, *[vs_414949], **kwargs_414950)
    
    # Testing the type of a for loop iterable (line 119)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 119, 8), enumerate_call_result_414951)
    # Getting the type of the for loop variable (line 119)
    for_loop_var_414952 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 119, 8), enumerate_call_result_414951)
    # Assigning a type to the variable 'i' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 8), for_loop_var_414952))
    # Assigning a type to the variable 'v' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 8), for_loop_var_414952))
    # SSA begins for a for statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 120):
    
    # Assigning a Call to a Name (line 120):
    
    # Call to dot(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'v' (line 120)
    v_414954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 'v', False)
    # Getting the type of 'w' (line 120)
    w_414955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'w', False)
    # Processing the call keyword arguments (line 120)
    kwargs_414956 = {}
    # Getting the type of 'dot' (line 120)
    dot_414953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'dot', False)
    # Calling dot(args, kwargs) (line 120)
    dot_call_result_414957 = invoke(stypy.reporting.localization.Localization(__file__, 120, 20), dot_414953, *[v_414954, w_414955], **kwargs_414956)
    
    # Assigning a type to the variable 'alpha' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'alpha', dot_call_result_414957)
    
    # Assigning a Name to a Subscript (line 121):
    
    # Assigning a Name to a Subscript (line 121):
    # Getting the type of 'alpha' (line 121)
    alpha_414958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 22), 'alpha')
    # Getting the type of 'hcur' (line 121)
    hcur_414959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'hcur')
    # Getting the type of 'i' (line 121)
    i_414960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'i')
    # Storing an element on a container (line 121)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 12), hcur_414959, (i_414960, alpha_414958))
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to axpy(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'v' (line 122)
    v_414962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'v', False)
    # Getting the type of 'w' (line 122)
    w_414963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'w', False)
    
    # Obtaining the type of the subscript
    int_414964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 35), 'int')
    # Getting the type of 'v' (line 122)
    v_414965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'v', False)
    # Obtaining the member 'shape' of a type (line 122)
    shape_414966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 27), v_414965, 'shape')
    # Obtaining the member '__getitem__' of a type (line 122)
    getitem___414967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 27), shape_414966, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 122)
    subscript_call_result_414968 = invoke(stypy.reporting.localization.Localization(__file__, 122, 27), getitem___414967, int_414964)
    
    
    # Getting the type of 'alpha' (line 122)
    alpha_414969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 40), 'alpha', False)
    # Applying the 'usub' unary operator (line 122)
    result___neg___414970 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 39), 'usub', alpha_414969)
    
    # Processing the call keyword arguments (line 122)
    kwargs_414971 = {}
    # Getting the type of 'axpy' (line 122)
    axpy_414961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'axpy', False)
    # Calling axpy(args, kwargs) (line 122)
    axpy_call_result_414972 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), axpy_414961, *[v_414962, w_414963, subscript_call_result_414968, result___neg___414970], **kwargs_414971)
    
    # Assigning a type to the variable 'w' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'w', axpy_call_result_414972)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 123):
    
    # Assigning a Call to a Subscript (line 123):
    
    # Call to nrm2(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'w' (line 123)
    w_414974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 25), 'w', False)
    # Processing the call keyword arguments (line 123)
    kwargs_414975 = {}
    # Getting the type of 'nrm2' (line 123)
    nrm2_414973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 20), 'nrm2', False)
    # Calling nrm2(args, kwargs) (line 123)
    nrm2_call_result_414976 = invoke(stypy.reporting.localization.Localization(__file__, 123, 20), nrm2_414973, *[w_414974], **kwargs_414975)
    
    # Getting the type of 'hcur' (line 123)
    hcur_414977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'hcur')
    # Getting the type of 'i' (line 123)
    i_414978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 13), 'i')
    int_414979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 15), 'int')
    # Applying the binary operator '+' (line 123)
    result_add_414980 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 13), '+', i_414978, int_414979)
    
    # Storing an element on a container (line 123)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 8), hcur_414977, (result_add_414980, nrm2_call_result_414976))
    
    # Call to errstate(...): (line 125)
    # Processing the call keyword arguments (line 125)
    str_414983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 30), 'str', 'ignore')
    keyword_414984 = str_414983
    str_414985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 47), 'str', 'ignore')
    keyword_414986 = str_414985
    kwargs_414987 = {'over': keyword_414984, 'divide': keyword_414986}
    # Getting the type of 'np' (line 125)
    np_414981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'np', False)
    # Obtaining the member 'errstate' of a type (line 125)
    errstate_414982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), np_414981, 'errstate')
    # Calling errstate(args, kwargs) (line 125)
    errstate_call_result_414988 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), errstate_414982, *[], **kwargs_414987)
    
    with_414989 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 125, 13), errstate_call_result_414988, 'with parameter', '__enter__', '__exit__')

    if with_414989:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 125)
        enter___414990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), errstate_call_result_414988, '__enter__')
        with_enter_414991 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), enter___414990)
        
        # Assigning a BinOp to a Name (line 127):
        
        # Assigning a BinOp to a Name (line 127):
        int_414992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 20), 'int')
        
        # Obtaining the type of the subscript
        int_414993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 27), 'int')
        # Getting the type of 'hcur' (line 127)
        hcur_414994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 22), 'hcur')
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___414995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 22), hcur_414994, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_414996 = invoke(stypy.reporting.localization.Localization(__file__, 127, 22), getitem___414995, int_414993)
        
        # Applying the binary operator 'div' (line 127)
        result_div_414997 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 20), 'div', int_414992, subscript_call_result_414996)
        
        # Assigning a type to the variable 'alpha' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'alpha', result_div_414997)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 125)
        exit___414998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), errstate_call_result_414988, '__exit__')
        with_exit_414999 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), exit___414998, None, None, None)

    
    
    # Call to isfinite(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'alpha' (line 129)
    alpha_415002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'alpha', False)
    # Processing the call keyword arguments (line 129)
    kwargs_415003 = {}
    # Getting the type of 'np' (line 129)
    np_415000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 129)
    isfinite_415001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 11), np_415000, 'isfinite')
    # Calling isfinite(args, kwargs) (line 129)
    isfinite_call_result_415004 = invoke(stypy.reporting.localization.Localization(__file__, 129, 11), isfinite_415001, *[alpha_415002], **kwargs_415003)
    
    # Testing the type of an if condition (line 129)
    if_condition_415005 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 8), isfinite_call_result_415004)
    # Assigning a type to the variable 'if_condition_415005' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'if_condition_415005', if_condition_415005)
    # SSA begins for if statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 130):
    
    # Assigning a Call to a Name (line 130):
    
    # Call to scal(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'alpha' (line 130)
    alpha_415007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'alpha', False)
    # Getting the type of 'w' (line 130)
    w_415008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'w', False)
    # Processing the call keyword arguments (line 130)
    kwargs_415009 = {}
    # Getting the type of 'scal' (line 130)
    scal_415006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'scal', False)
    # Calling scal(args, kwargs) (line 130)
    scal_call_result_415010 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), scal_415006, *[alpha_415007, w_415008], **kwargs_415009)
    
    # Assigning a type to the variable 'w' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'w', scal_call_result_415010)
    # SSA join for if statement (line 129)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Obtaining the type of the subscript
    int_415011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 21), 'int')
    # Getting the type of 'hcur' (line 132)
    hcur_415012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'hcur')
    # Obtaining the member '__getitem__' of a type (line 132)
    getitem___415013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), hcur_415012, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 132)
    subscript_call_result_415014 = invoke(stypy.reporting.localization.Localization(__file__, 132, 16), getitem___415013, int_415011)
    
    # Getting the type of 'eps' (line 132)
    eps_415015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'eps')
    # Getting the type of 'w_norm' (line 132)
    w_norm_415016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 33), 'w_norm')
    # Applying the binary operator '*' (line 132)
    result_mul_415017 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 27), '*', eps_415015, w_norm_415016)
    
    # Applying the binary operator '>' (line 132)
    result_gt_415018 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 16), '>', subscript_call_result_415014, result_mul_415017)
    
    # Applying the 'not' unary operator (line 132)
    result_not__415019 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 11), 'not', result_gt_415018)
    
    # Testing the type of an if condition (line 132)
    if_condition_415020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 8), result_not__415019)
    # Assigning a type to the variable 'if_condition_415020' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'if_condition_415020', if_condition_415020)
    # SSA begins for if statement (line 132)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 136):
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'True' (line 136)
    True_415021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'True')
    # Assigning a type to the variable 'breakdown' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'breakdown', True_415021)
    # SSA join for if statement (line 132)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'w' (line 138)
    w_415024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 18), 'w', False)
    # Processing the call keyword arguments (line 138)
    kwargs_415025 = {}
    # Getting the type of 'vs' (line 138)
    vs_415022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'vs', False)
    # Obtaining the member 'append' of a type (line 138)
    append_415023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), vs_415022, 'append')
    # Calling append(args, kwargs) (line 138)
    append_call_result_415026 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), append_415023, *[w_415024], **kwargs_415025)
    
    
    # Call to append(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'z' (line 139)
    z_415029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 18), 'z', False)
    # Processing the call keyword arguments (line 139)
    kwargs_415030 = {}
    # Getting the type of 'zs' (line 139)
    zs_415027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'zs', False)
    # Obtaining the member 'append' of a type (line 139)
    append_415028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), zs_415027, 'append')
    # Calling append(args, kwargs) (line 139)
    append_call_result_415031 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), append_415028, *[z_415029], **kwargs_415030)
    
    
    # Assigning a Call to a Name (line 144):
    
    # Assigning a Call to a Name (line 144):
    
    # Call to zeros(...): (line 144)
    # Processing the call arguments (line 144)
    
    # Obtaining an instance of the builtin type 'tuple' (line 144)
    tuple_415034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 144)
    # Adding element type (line 144)
    # Getting the type of 'j' (line 144)
    j_415035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 23), 'j', False)
    int_415036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 25), 'int')
    # Applying the binary operator '+' (line 144)
    result_add_415037 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 23), '+', j_415035, int_415036)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 23), tuple_415034, result_add_415037)
    # Adding element type (line 144)
    # Getting the type of 'j' (line 144)
    j_415038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 28), 'j', False)
    int_415039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 30), 'int')
    # Applying the binary operator '+' (line 144)
    result_add_415040 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 28), '+', j_415038, int_415039)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 23), tuple_415034, result_add_415040)
    
    # Processing the call keyword arguments (line 144)
    # Getting the type of 'Q' (line 144)
    Q_415041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 40), 'Q', False)
    # Obtaining the member 'dtype' of a type (line 144)
    dtype_415042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 40), Q_415041, 'dtype')
    keyword_415043 = dtype_415042
    str_415044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 55), 'str', 'F')
    keyword_415045 = str_415044
    kwargs_415046 = {'dtype': keyword_415043, 'order': keyword_415045}
    # Getting the type of 'np' (line 144)
    np_415032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 13), 'np', False)
    # Obtaining the member 'zeros' of a type (line 144)
    zeros_415033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 13), np_415032, 'zeros')
    # Calling zeros(args, kwargs) (line 144)
    zeros_call_result_415047 = invoke(stypy.reporting.localization.Localization(__file__, 144, 13), zeros_415033, *[tuple_415034], **kwargs_415046)
    
    # Assigning a type to the variable 'Q2' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'Q2', zeros_call_result_415047)
    
    # Assigning a Name to a Subscript (line 145):
    
    # Assigning a Name to a Subscript (line 145):
    # Getting the type of 'Q' (line 145)
    Q_415048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 24), 'Q')
    # Getting the type of 'Q2' (line 145)
    Q2_415049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'Q2')
    # Getting the type of 'j' (line 145)
    j_415050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'j')
    int_415051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 14), 'int')
    # Applying the binary operator '+' (line 145)
    result_add_415052 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 12), '+', j_415050, int_415051)
    
    slice_415053 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 145, 8), None, result_add_415052, None)
    # Getting the type of 'j' (line 145)
    j_415054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 17), 'j')
    int_415055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 19), 'int')
    # Applying the binary operator '+' (line 145)
    result_add_415056 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 17), '+', j_415054, int_415055)
    
    slice_415057 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 145, 8), None, result_add_415056, None)
    # Storing an element on a container (line 145)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 8), Q2_415049, ((slice_415053, slice_415057), Q_415048))
    
    # Assigning a Num to a Subscript (line 146):
    
    # Assigning a Num to a Subscript (line 146):
    int_415058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 22), 'int')
    # Getting the type of 'Q2' (line 146)
    Q2_415059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'Q2')
    
    # Obtaining an instance of the builtin type 'tuple' (line 146)
    tuple_415060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 146)
    # Adding element type (line 146)
    # Getting the type of 'j' (line 146)
    j_415061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'j')
    int_415062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 13), 'int')
    # Applying the binary operator '+' (line 146)
    result_add_415063 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 11), '+', j_415061, int_415062)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 11), tuple_415060, result_add_415063)
    # Adding element type (line 146)
    # Getting the type of 'j' (line 146)
    j_415064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'j')
    int_415065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 17), 'int')
    # Applying the binary operator '+' (line 146)
    result_add_415066 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 15), '+', j_415064, int_415065)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 11), tuple_415060, result_add_415066)
    
    # Storing an element on a container (line 146)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 8), Q2_415059, (tuple_415060, int_415058))
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to zeros(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Obtaining an instance of the builtin type 'tuple' (line 148)
    tuple_415069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 148)
    # Adding element type (line 148)
    # Getting the type of 'j' (line 148)
    j_415070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'j', False)
    int_415071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 25), 'int')
    # Applying the binary operator '+' (line 148)
    result_add_415072 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 23), '+', j_415070, int_415071)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 23), tuple_415069, result_add_415072)
    # Adding element type (line 148)
    # Getting the type of 'j' (line 148)
    j_415073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 28), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 23), tuple_415069, j_415073)
    
    # Processing the call keyword arguments (line 148)
    # Getting the type of 'R' (line 148)
    R_415074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 38), 'R', False)
    # Obtaining the member 'dtype' of a type (line 148)
    dtype_415075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 38), R_415074, 'dtype')
    keyword_415076 = dtype_415075
    str_415077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 53), 'str', 'F')
    keyword_415078 = str_415077
    kwargs_415079 = {'dtype': keyword_415076, 'order': keyword_415078}
    # Getting the type of 'np' (line 148)
    np_415067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 13), 'np', False)
    # Obtaining the member 'zeros' of a type (line 148)
    zeros_415068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 13), np_415067, 'zeros')
    # Calling zeros(args, kwargs) (line 148)
    zeros_call_result_415080 = invoke(stypy.reporting.localization.Localization(__file__, 148, 13), zeros_415068, *[tuple_415069], **kwargs_415079)
    
    # Assigning a type to the variable 'R2' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'R2', zeros_call_result_415080)
    
    # Assigning a Name to a Subscript (line 149):
    
    # Assigning a Name to a Subscript (line 149):
    # Getting the type of 'R' (line 149)
    R_415081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 21), 'R')
    # Getting the type of 'R2' (line 149)
    R2_415082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'R2')
    # Getting the type of 'j' (line 149)
    j_415083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'j')
    int_415084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 14), 'int')
    # Applying the binary operator '+' (line 149)
    result_add_415085 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 12), '+', j_415083, int_415084)
    
    slice_415086 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 149, 8), None, result_add_415085, None)
    slice_415087 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 149, 8), None, None, None)
    # Storing an element on a container (line 149)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 8), R2_415082, ((slice_415086, slice_415087), R_415081))
    
    # Assigning a Call to a Tuple (line 151):
    
    # Assigning a Subscript to a Name (line 151):
    
    # Obtaining the type of the subscript
    int_415088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'int')
    
    # Call to qr_insert(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'Q2' (line 151)
    Q2_415090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'Q2', False)
    # Getting the type of 'R2' (line 151)
    R2_415091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 29), 'R2', False)
    # Getting the type of 'hcur' (line 151)
    hcur_415092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 33), 'hcur', False)
    # Getting the type of 'j' (line 151)
    j_415093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 39), 'j', False)
    # Processing the call keyword arguments (line 151)
    str_415094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 48), 'str', 'col')
    keyword_415095 = str_415094
    # Getting the type of 'True' (line 152)
    True_415096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 39), 'True', False)
    keyword_415097 = True_415096
    # Getting the type of 'False' (line 152)
    False_415098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 58), 'False', False)
    keyword_415099 = False_415098
    kwargs_415100 = {'overwrite_qru': keyword_415097, 'which': keyword_415095, 'check_finite': keyword_415099}
    # Getting the type of 'qr_insert' (line 151)
    qr_insert_415089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'qr_insert', False)
    # Calling qr_insert(args, kwargs) (line 151)
    qr_insert_call_result_415101 = invoke(stypy.reporting.localization.Localization(__file__, 151, 15), qr_insert_415089, *[Q2_415090, R2_415091, hcur_415092, j_415093], **kwargs_415100)
    
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___415102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), qr_insert_call_result_415101, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_415103 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), getitem___415102, int_415088)
    
    # Assigning a type to the variable 'tuple_var_assignment_414619' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_414619', subscript_call_result_415103)
    
    # Assigning a Subscript to a Name (line 151):
    
    # Obtaining the type of the subscript
    int_415104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'int')
    
    # Call to qr_insert(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'Q2' (line 151)
    Q2_415106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'Q2', False)
    # Getting the type of 'R2' (line 151)
    R2_415107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 29), 'R2', False)
    # Getting the type of 'hcur' (line 151)
    hcur_415108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 33), 'hcur', False)
    # Getting the type of 'j' (line 151)
    j_415109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 39), 'j', False)
    # Processing the call keyword arguments (line 151)
    str_415110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 48), 'str', 'col')
    keyword_415111 = str_415110
    # Getting the type of 'True' (line 152)
    True_415112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 39), 'True', False)
    keyword_415113 = True_415112
    # Getting the type of 'False' (line 152)
    False_415114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 58), 'False', False)
    keyword_415115 = False_415114
    kwargs_415116 = {'overwrite_qru': keyword_415113, 'which': keyword_415111, 'check_finite': keyword_415115}
    # Getting the type of 'qr_insert' (line 151)
    qr_insert_415105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'qr_insert', False)
    # Calling qr_insert(args, kwargs) (line 151)
    qr_insert_call_result_415117 = invoke(stypy.reporting.localization.Localization(__file__, 151, 15), qr_insert_415105, *[Q2_415106, R2_415107, hcur_415108, j_415109], **kwargs_415116)
    
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___415118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), qr_insert_call_result_415117, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_415119 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), getitem___415118, int_415104)
    
    # Assigning a type to the variable 'tuple_var_assignment_414620' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_414620', subscript_call_result_415119)
    
    # Assigning a Name to a Name (line 151):
    # Getting the type of 'tuple_var_assignment_414619' (line 151)
    tuple_var_assignment_414619_415120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_414619')
    # Assigning a type to the variable 'Q' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'Q', tuple_var_assignment_414619_415120)
    
    # Assigning a Name to a Name (line 151):
    # Getting the type of 'tuple_var_assignment_414620' (line 151)
    tuple_var_assignment_414620_415121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_414620')
    # Assigning a type to the variable 'R' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'R', tuple_var_assignment_414620_415121)
    
    # Assigning a Call to a Name (line 159):
    
    # Assigning a Call to a Name (line 159):
    
    # Call to abs(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 159)
    tuple_415123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 159)
    # Adding element type (line 159)
    int_415124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 20), tuple_415123, int_415124)
    # Adding element type (line 159)
    int_415125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 20), tuple_415123, int_415125)
    
    # Getting the type of 'Q' (line 159)
    Q_415126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 18), 'Q', False)
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___415127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 18), Q_415126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_415128 = invoke(stypy.reporting.localization.Localization(__file__, 159, 18), getitem___415127, tuple_415123)
    
    # Processing the call keyword arguments (line 159)
    kwargs_415129 = {}
    # Getting the type of 'abs' (line 159)
    abs_415122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 14), 'abs', False)
    # Calling abs(args, kwargs) (line 159)
    abs_call_result_415130 = invoke(stypy.reporting.localization.Localization(__file__, 159, 14), abs_415122, *[subscript_call_result_415128], **kwargs_415129)
    
    # Assigning a type to the variable 'res' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'res', abs_call_result_415130)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'res' (line 162)
    res_415131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'res')
    # Getting the type of 'atol' (line 162)
    atol_415132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 17), 'atol')
    # Applying the binary operator '<' (line 162)
    result_lt_415133 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 11), '<', res_415131, atol_415132)
    
    # Getting the type of 'breakdown' (line 162)
    breakdown_415134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'breakdown')
    # Applying the binary operator 'or' (line 162)
    result_or_keyword_415135 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 11), 'or', result_lt_415133, breakdown_415134)
    
    # Testing the type of an if condition (line 162)
    if_condition_415136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 8), result_or_keyword_415135)
    # Assigning a type to the variable 'if_condition_415136' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'if_condition_415136', if_condition_415136)
    # SSA begins for if statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 162)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to isfinite(...): (line 165)
    # Processing the call arguments (line 165)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 165)
    tuple_415139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 165)
    # Adding element type (line 165)
    # Getting the type of 'j' (line 165)
    j_415140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 25), tuple_415139, j_415140)
    # Adding element type (line 165)
    # Getting the type of 'j' (line 165)
    j_415141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 27), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 25), tuple_415139, j_415141)
    
    # Getting the type of 'R' (line 165)
    R_415142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___415143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 23), R_415142, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_415144 = invoke(stypy.reporting.localization.Localization(__file__, 165, 23), getitem___415143, tuple_415139)
    
    # Processing the call keyword arguments (line 165)
    kwargs_415145 = {}
    # Getting the type of 'np' (line 165)
    np_415137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 165)
    isfinite_415138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 11), np_415137, 'isfinite')
    # Calling isfinite(args, kwargs) (line 165)
    isfinite_call_result_415146 = invoke(stypy.reporting.localization.Localization(__file__, 165, 11), isfinite_415138, *[subscript_call_result_415144], **kwargs_415145)
    
    # Applying the 'not' unary operator (line 165)
    result_not__415147 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 7), 'not', isfinite_call_result_415146)
    
    # Testing the type of an if condition (line 165)
    if_condition_415148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 4), result_not__415147)
    # Assigning a type to the variable 'if_condition_415148' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'if_condition_415148', if_condition_415148)
    # SSA begins for if statement (line 165)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 167)
    # Processing the call keyword arguments (line 167)
    kwargs_415150 = {}
    # Getting the type of 'LinAlgError' (line 167)
    LinAlgError_415149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 167)
    LinAlgError_call_result_415151 = invoke(stypy.reporting.localization.Localization(__file__, 167, 14), LinAlgError_415149, *[], **kwargs_415150)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 167, 8), LinAlgError_call_result_415151, 'raise parameter', BaseException)
    # SSA join for if statement (line 165)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 174):
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_415152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 4), 'int')
    
    # Call to lstsq(...): (line 174)
    # Processing the call arguments (line 174)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 174)
    j_415154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'j', False)
    int_415155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 29), 'int')
    # Applying the binary operator '+' (line 174)
    result_add_415156 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 27), '+', j_415154, int_415155)
    
    slice_415157 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 24), None, result_add_415156, None)
    # Getting the type of 'j' (line 174)
    j_415158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'j', False)
    int_415159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 34), 'int')
    # Applying the binary operator '+' (line 174)
    result_add_415160 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 32), '+', j_415158, int_415159)
    
    slice_415161 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 24), None, result_add_415160, None)
    # Getting the type of 'R' (line 174)
    R_415162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___415163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 24), R_415162, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_415164 = invoke(stypy.reporting.localization.Localization(__file__, 174, 24), getitem___415163, (slice_415157, slice_415161))
    
    
    # Call to conj(...): (line 174)
    # Processing the call keyword arguments (line 174)
    kwargs_415174 = {}
    
    # Obtaining the type of the subscript
    int_415165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 40), 'int')
    # Getting the type of 'j' (line 174)
    j_415166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 43), 'j', False)
    int_415167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 45), 'int')
    # Applying the binary operator '+' (line 174)
    result_add_415168 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 43), '+', j_415166, int_415167)
    
    slice_415169 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 38), None, result_add_415168, None)
    # Getting the type of 'Q' (line 174)
    Q_415170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'Q', False)
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___415171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 38), Q_415170, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_415172 = invoke(stypy.reporting.localization.Localization(__file__, 174, 38), getitem___415171, (int_415165, slice_415169))
    
    # Obtaining the member 'conj' of a type (line 174)
    conj_415173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 38), subscript_call_result_415172, 'conj')
    # Calling conj(args, kwargs) (line 174)
    conj_call_result_415175 = invoke(stypy.reporting.localization.Localization(__file__, 174, 38), conj_415173, *[], **kwargs_415174)
    
    # Processing the call keyword arguments (line 174)
    kwargs_415176 = {}
    # Getting the type of 'lstsq' (line 174)
    lstsq_415153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 174)
    lstsq_call_result_415177 = invoke(stypy.reporting.localization.Localization(__file__, 174, 18), lstsq_415153, *[subscript_call_result_415164, conj_call_result_415175], **kwargs_415176)
    
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___415178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 4), lstsq_call_result_415177, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_415179 = invoke(stypy.reporting.localization.Localization(__file__, 174, 4), getitem___415178, int_415152)
    
    # Assigning a type to the variable 'tuple_var_assignment_414621' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'tuple_var_assignment_414621', subscript_call_result_415179)
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_415180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 4), 'int')
    
    # Call to lstsq(...): (line 174)
    # Processing the call arguments (line 174)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 174)
    j_415182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'j', False)
    int_415183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 29), 'int')
    # Applying the binary operator '+' (line 174)
    result_add_415184 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 27), '+', j_415182, int_415183)
    
    slice_415185 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 24), None, result_add_415184, None)
    # Getting the type of 'j' (line 174)
    j_415186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'j', False)
    int_415187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 34), 'int')
    # Applying the binary operator '+' (line 174)
    result_add_415188 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 32), '+', j_415186, int_415187)
    
    slice_415189 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 24), None, result_add_415188, None)
    # Getting the type of 'R' (line 174)
    R_415190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___415191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 24), R_415190, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_415192 = invoke(stypy.reporting.localization.Localization(__file__, 174, 24), getitem___415191, (slice_415185, slice_415189))
    
    
    # Call to conj(...): (line 174)
    # Processing the call keyword arguments (line 174)
    kwargs_415202 = {}
    
    # Obtaining the type of the subscript
    int_415193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 40), 'int')
    # Getting the type of 'j' (line 174)
    j_415194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 43), 'j', False)
    int_415195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 45), 'int')
    # Applying the binary operator '+' (line 174)
    result_add_415196 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 43), '+', j_415194, int_415195)
    
    slice_415197 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 38), None, result_add_415196, None)
    # Getting the type of 'Q' (line 174)
    Q_415198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'Q', False)
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___415199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 38), Q_415198, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_415200 = invoke(stypy.reporting.localization.Localization(__file__, 174, 38), getitem___415199, (int_415193, slice_415197))
    
    # Obtaining the member 'conj' of a type (line 174)
    conj_415201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 38), subscript_call_result_415200, 'conj')
    # Calling conj(args, kwargs) (line 174)
    conj_call_result_415203 = invoke(stypy.reporting.localization.Localization(__file__, 174, 38), conj_415201, *[], **kwargs_415202)
    
    # Processing the call keyword arguments (line 174)
    kwargs_415204 = {}
    # Getting the type of 'lstsq' (line 174)
    lstsq_415181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 174)
    lstsq_call_result_415205 = invoke(stypy.reporting.localization.Localization(__file__, 174, 18), lstsq_415181, *[subscript_call_result_415192, conj_call_result_415203], **kwargs_415204)
    
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___415206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 4), lstsq_call_result_415205, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_415207 = invoke(stypy.reporting.localization.Localization(__file__, 174, 4), getitem___415206, int_415180)
    
    # Assigning a type to the variable 'tuple_var_assignment_414622' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'tuple_var_assignment_414622', subscript_call_result_415207)
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_415208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 4), 'int')
    
    # Call to lstsq(...): (line 174)
    # Processing the call arguments (line 174)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 174)
    j_415210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'j', False)
    int_415211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 29), 'int')
    # Applying the binary operator '+' (line 174)
    result_add_415212 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 27), '+', j_415210, int_415211)
    
    slice_415213 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 24), None, result_add_415212, None)
    # Getting the type of 'j' (line 174)
    j_415214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'j', False)
    int_415215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 34), 'int')
    # Applying the binary operator '+' (line 174)
    result_add_415216 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 32), '+', j_415214, int_415215)
    
    slice_415217 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 24), None, result_add_415216, None)
    # Getting the type of 'R' (line 174)
    R_415218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___415219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 24), R_415218, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_415220 = invoke(stypy.reporting.localization.Localization(__file__, 174, 24), getitem___415219, (slice_415213, slice_415217))
    
    
    # Call to conj(...): (line 174)
    # Processing the call keyword arguments (line 174)
    kwargs_415230 = {}
    
    # Obtaining the type of the subscript
    int_415221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 40), 'int')
    # Getting the type of 'j' (line 174)
    j_415222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 43), 'j', False)
    int_415223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 45), 'int')
    # Applying the binary operator '+' (line 174)
    result_add_415224 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 43), '+', j_415222, int_415223)
    
    slice_415225 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 38), None, result_add_415224, None)
    # Getting the type of 'Q' (line 174)
    Q_415226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'Q', False)
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___415227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 38), Q_415226, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_415228 = invoke(stypy.reporting.localization.Localization(__file__, 174, 38), getitem___415227, (int_415221, slice_415225))
    
    # Obtaining the member 'conj' of a type (line 174)
    conj_415229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 38), subscript_call_result_415228, 'conj')
    # Calling conj(args, kwargs) (line 174)
    conj_call_result_415231 = invoke(stypy.reporting.localization.Localization(__file__, 174, 38), conj_415229, *[], **kwargs_415230)
    
    # Processing the call keyword arguments (line 174)
    kwargs_415232 = {}
    # Getting the type of 'lstsq' (line 174)
    lstsq_415209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 174)
    lstsq_call_result_415233 = invoke(stypy.reporting.localization.Localization(__file__, 174, 18), lstsq_415209, *[subscript_call_result_415220, conj_call_result_415231], **kwargs_415232)
    
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___415234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 4), lstsq_call_result_415233, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_415235 = invoke(stypy.reporting.localization.Localization(__file__, 174, 4), getitem___415234, int_415208)
    
    # Assigning a type to the variable 'tuple_var_assignment_414623' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'tuple_var_assignment_414623', subscript_call_result_415235)
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_415236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 4), 'int')
    
    # Call to lstsq(...): (line 174)
    # Processing the call arguments (line 174)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 174)
    j_415238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'j', False)
    int_415239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 29), 'int')
    # Applying the binary operator '+' (line 174)
    result_add_415240 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 27), '+', j_415238, int_415239)
    
    slice_415241 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 24), None, result_add_415240, None)
    # Getting the type of 'j' (line 174)
    j_415242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'j', False)
    int_415243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 34), 'int')
    # Applying the binary operator '+' (line 174)
    result_add_415244 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 32), '+', j_415242, int_415243)
    
    slice_415245 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 24), None, result_add_415244, None)
    # Getting the type of 'R' (line 174)
    R_415246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___415247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 24), R_415246, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_415248 = invoke(stypy.reporting.localization.Localization(__file__, 174, 24), getitem___415247, (slice_415241, slice_415245))
    
    
    # Call to conj(...): (line 174)
    # Processing the call keyword arguments (line 174)
    kwargs_415258 = {}
    
    # Obtaining the type of the subscript
    int_415249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 40), 'int')
    # Getting the type of 'j' (line 174)
    j_415250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 43), 'j', False)
    int_415251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 45), 'int')
    # Applying the binary operator '+' (line 174)
    result_add_415252 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 43), '+', j_415250, int_415251)
    
    slice_415253 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 38), None, result_add_415252, None)
    # Getting the type of 'Q' (line 174)
    Q_415254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'Q', False)
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___415255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 38), Q_415254, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_415256 = invoke(stypy.reporting.localization.Localization(__file__, 174, 38), getitem___415255, (int_415249, slice_415253))
    
    # Obtaining the member 'conj' of a type (line 174)
    conj_415257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 38), subscript_call_result_415256, 'conj')
    # Calling conj(args, kwargs) (line 174)
    conj_call_result_415259 = invoke(stypy.reporting.localization.Localization(__file__, 174, 38), conj_415257, *[], **kwargs_415258)
    
    # Processing the call keyword arguments (line 174)
    kwargs_415260 = {}
    # Getting the type of 'lstsq' (line 174)
    lstsq_415237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 174)
    lstsq_call_result_415261 = invoke(stypy.reporting.localization.Localization(__file__, 174, 18), lstsq_415237, *[subscript_call_result_415248, conj_call_result_415259], **kwargs_415260)
    
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___415262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 4), lstsq_call_result_415261, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_415263 = invoke(stypy.reporting.localization.Localization(__file__, 174, 4), getitem___415262, int_415236)
    
    # Assigning a type to the variable 'tuple_var_assignment_414624' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'tuple_var_assignment_414624', subscript_call_result_415263)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_var_assignment_414621' (line 174)
    tuple_var_assignment_414621_415264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'tuple_var_assignment_414621')
    # Assigning a type to the variable 'y' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'y', tuple_var_assignment_414621_415264)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_var_assignment_414622' (line 174)
    tuple_var_assignment_414622_415265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'tuple_var_assignment_414622')
    # Assigning a type to the variable '_' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 7), '_', tuple_var_assignment_414622_415265)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_var_assignment_414623' (line 174)
    tuple_var_assignment_414623_415266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'tuple_var_assignment_414623')
    # Assigning a type to the variable '_' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 10), '_', tuple_var_assignment_414623_415266)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_var_assignment_414624' (line 174)
    tuple_var_assignment_414624_415267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'tuple_var_assignment_414624')
    # Assigning a type to the variable '_' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 13), '_', tuple_var_assignment_414624_415267)
    
    # Assigning a Subscript to a Name (line 176):
    
    # Assigning a Subscript to a Name (line 176):
    
    # Obtaining the type of the subscript
    slice_415268 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 176, 8), None, None, None)
    # Getting the type of 'j' (line 176)
    j_415269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 13), 'j')
    int_415270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 15), 'int')
    # Applying the binary operator '+' (line 176)
    result_add_415271 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 13), '+', j_415269, int_415270)
    
    slice_415272 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 176, 8), None, result_add_415271, None)
    # Getting the type of 'B' (line 176)
    B_415273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'B')
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___415274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), B_415273, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_415275 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), getitem___415274, (slice_415268, slice_415272))
    
    # Assigning a type to the variable 'B' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'B', subscript_call_result_415275)
    
    # Obtaining an instance of the builtin type 'tuple' (line 178)
    tuple_415276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 178)
    # Adding element type (line 178)
    # Getting the type of 'Q' (line 178)
    Q_415277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 11), 'Q')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 11), tuple_415276, Q_415277)
    # Adding element type (line 178)
    # Getting the type of 'R' (line 178)
    R_415278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 14), 'R')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 11), tuple_415276, R_415278)
    # Adding element type (line 178)
    # Getting the type of 'B' (line 178)
    B_415279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'B')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 11), tuple_415276, B_415279)
    # Adding element type (line 178)
    # Getting the type of 'vs' (line 178)
    vs_415280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'vs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 11), tuple_415276, vs_415280)
    # Adding element type (line 178)
    # Getting the type of 'zs' (line 178)
    zs_415281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'zs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 11), tuple_415276, zs_415281)
    # Adding element type (line 178)
    # Getting the type of 'y' (line 178)
    y_415282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 11), tuple_415276, y_415282)
    
    # Assigning a type to the variable 'stypy_return_type' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type', tuple_415276)
    
    # ################# End of '_fgmres(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fgmres' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_415283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_415283)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fgmres'
    return stypy_return_type_415283

# Assigning a type to the variable '_fgmres' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '_fgmres', _fgmres)

@norecursion
def gcrotmk(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 181)
    None_415284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 21), 'None')
    float_415285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 31), 'float')
    int_415286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 45), 'int')
    # Getting the type of 'None' (line 181)
    None_415287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 53), 'None')
    # Getting the type of 'None' (line 181)
    None_415288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 68), 'None')
    int_415289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 14), 'int')
    # Getting the type of 'None' (line 182)
    None_415290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'None')
    # Getting the type of 'None' (line 182)
    None_415291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 29), 'None')
    # Getting the type of 'False' (line 182)
    False_415292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 45), 'False')
    str_415293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 61), 'str', 'oldest')
    defaults = [None_415284, float_415285, int_415286, None_415287, None_415288, int_415289, None_415290, None_415291, False_415292, str_415293]
    # Create a new context for function 'gcrotmk'
    module_type_store = module_type_store.open_function_context('gcrotmk', 181, 0, False)
    
    # Passed parameters checking function
    gcrotmk.stypy_localization = localization
    gcrotmk.stypy_type_of_self = None
    gcrotmk.stypy_type_store = module_type_store
    gcrotmk.stypy_function_name = 'gcrotmk'
    gcrotmk.stypy_param_names_list = ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback', 'm', 'k', 'CU', 'discard_C', 'truncate']
    gcrotmk.stypy_varargs_param_name = None
    gcrotmk.stypy_kwargs_param_name = None
    gcrotmk.stypy_call_defaults = defaults
    gcrotmk.stypy_call_varargs = varargs
    gcrotmk.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gcrotmk', ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback', 'm', 'k', 'CU', 'discard_C', 'truncate'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gcrotmk', localization, ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback', 'm', 'k', 'CU', 'discard_C', 'truncate'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gcrotmk(...)' code ##################

    str_415294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, (-1)), 'str', "\n    Solve a matrix equation using flexible GCROT(m,k) algorithm.\n\n    Parameters\n    ----------\n    A : {sparse matrix, dense matrix, LinearOperator}\n        The real or complex N-by-N matrix of the linear system.\n    b : {array, matrix}\n        Right hand side of the linear system. Has shape (N,) or (N,1).\n    x0  : {array, matrix}\n        Starting guess for the solution.\n    tol : float, optional\n        Tolerance to achieve. The algorithm terminates when either the relative\n        or the absolute residual is below `tol`.\n    maxiter : int, optional\n        Maximum number of iterations.  Iteration will stop after maxiter\n        steps even if the specified tolerance has not been achieved.\n    M : {sparse matrix, dense matrix, LinearOperator}, optional\n        Preconditioner for A.  The preconditioner should approximate the\n        inverse of A. gcrotmk is a 'flexible' algorithm and the preconditioner\n        can vary from iteration to iteration. Effective preconditioning\n        dramatically improves the rate of convergence, which implies that\n        fewer iterations are needed to reach a given error tolerance.\n    callback : function, optional\n        User-supplied function to call after each iteration.  It is called\n        as callback(xk), where xk is the current solution vector.\n    m : int, optional\n        Number of inner FGMRES iterations per each outer iteration.\n        Default: 20\n    k : int, optional\n        Number of vectors to carry between inner FGMRES iterations.\n        According to [2]_, good values are around m.\n        Default: m\n    CU : list of tuples, optional\n        List of tuples ``(c, u)`` which contain the columns of the matrices\n        C and U in the GCROT(m,k) algorithm. For details, see [2]_.\n        The list given and vectors contained in it are modified in-place.\n        If not given, start from empty matrices. The ``c`` elements in the\n        tuples can be ``None``, in which case the vectors are recomputed\n        via ``c = A u`` on start and orthogonalized as described in [3]_.\n    discard_C : bool, optional\n        Discard the C-vectors at the end. Useful if recycling Krylov subspaces\n        for different linear systems.\n    truncate : {'oldest', 'smallest'}, optional\n        Truncation scheme to use. Drop: oldest vectors, or vectors with\n        smallest singular values using the scheme discussed in [1,2].\n        See [2]_ for detailed comparison.\n        Default: 'oldest'\n\n    Returns\n    -------\n    x : array or matrix\n        The solution found.\n    info : int\n        Provides convergence information:\n\n        * 0  : successful exit\n        * >0 : convergence to tolerance not achieved, number of iterations\n\n    References\n    ----------\n    .. [1] E. de Sturler, ''Truncation strategies for optimal Krylov subspace\n           methods'', SIAM J. Numer. Anal. 36, 864 (1999).\n    .. [2] J.E. Hicken and D.W. Zingg, ''A simplified and flexible variant\n           of GCROT for solving nonsymmetric linear systems'',\n           SIAM J. Sci. Comput. 32, 172 (2010).\n    .. [3] M.L. Parks, E. de Sturler, G. Mackey, D.D. Johnson, S. Maiti,\n           ''Recycling Krylov subspaces for sequences of linear systems'',\n           SIAM J. Sci. Comput. 28, 1651 (2006).\n\n    ")
    
    # Assigning a Call to a Tuple (line 254):
    
    # Assigning a Subscript to a Name (line 254):
    
    # Obtaining the type of the subscript
    int_415295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 4), 'int')
    
    # Call to make_system(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'A' (line 254)
    A_415297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 38), 'A', False)
    # Getting the type of 'M' (line 254)
    M_415298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 40), 'M', False)
    # Getting the type of 'x0' (line 254)
    x0_415299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 42), 'x0', False)
    # Getting the type of 'b' (line 254)
    b_415300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 45), 'b', False)
    # Processing the call keyword arguments (line 254)
    kwargs_415301 = {}
    # Getting the type of 'make_system' (line 254)
    make_system_415296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 254)
    make_system_call_result_415302 = invoke(stypy.reporting.localization.Localization(__file__, 254, 26), make_system_415296, *[A_415297, M_415298, x0_415299, b_415300], **kwargs_415301)
    
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___415303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 4), make_system_call_result_415302, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_415304 = invoke(stypy.reporting.localization.Localization(__file__, 254, 4), getitem___415303, int_415295)
    
    # Assigning a type to the variable 'tuple_var_assignment_414625' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'tuple_var_assignment_414625', subscript_call_result_415304)
    
    # Assigning a Subscript to a Name (line 254):
    
    # Obtaining the type of the subscript
    int_415305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 4), 'int')
    
    # Call to make_system(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'A' (line 254)
    A_415307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 38), 'A', False)
    # Getting the type of 'M' (line 254)
    M_415308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 40), 'M', False)
    # Getting the type of 'x0' (line 254)
    x0_415309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 42), 'x0', False)
    # Getting the type of 'b' (line 254)
    b_415310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 45), 'b', False)
    # Processing the call keyword arguments (line 254)
    kwargs_415311 = {}
    # Getting the type of 'make_system' (line 254)
    make_system_415306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 254)
    make_system_call_result_415312 = invoke(stypy.reporting.localization.Localization(__file__, 254, 26), make_system_415306, *[A_415307, M_415308, x0_415309, b_415310], **kwargs_415311)
    
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___415313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 4), make_system_call_result_415312, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_415314 = invoke(stypy.reporting.localization.Localization(__file__, 254, 4), getitem___415313, int_415305)
    
    # Assigning a type to the variable 'tuple_var_assignment_414626' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'tuple_var_assignment_414626', subscript_call_result_415314)
    
    # Assigning a Subscript to a Name (line 254):
    
    # Obtaining the type of the subscript
    int_415315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 4), 'int')
    
    # Call to make_system(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'A' (line 254)
    A_415317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 38), 'A', False)
    # Getting the type of 'M' (line 254)
    M_415318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 40), 'M', False)
    # Getting the type of 'x0' (line 254)
    x0_415319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 42), 'x0', False)
    # Getting the type of 'b' (line 254)
    b_415320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 45), 'b', False)
    # Processing the call keyword arguments (line 254)
    kwargs_415321 = {}
    # Getting the type of 'make_system' (line 254)
    make_system_415316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 254)
    make_system_call_result_415322 = invoke(stypy.reporting.localization.Localization(__file__, 254, 26), make_system_415316, *[A_415317, M_415318, x0_415319, b_415320], **kwargs_415321)
    
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___415323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 4), make_system_call_result_415322, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_415324 = invoke(stypy.reporting.localization.Localization(__file__, 254, 4), getitem___415323, int_415315)
    
    # Assigning a type to the variable 'tuple_var_assignment_414627' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'tuple_var_assignment_414627', subscript_call_result_415324)
    
    # Assigning a Subscript to a Name (line 254):
    
    # Obtaining the type of the subscript
    int_415325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 4), 'int')
    
    # Call to make_system(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'A' (line 254)
    A_415327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 38), 'A', False)
    # Getting the type of 'M' (line 254)
    M_415328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 40), 'M', False)
    # Getting the type of 'x0' (line 254)
    x0_415329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 42), 'x0', False)
    # Getting the type of 'b' (line 254)
    b_415330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 45), 'b', False)
    # Processing the call keyword arguments (line 254)
    kwargs_415331 = {}
    # Getting the type of 'make_system' (line 254)
    make_system_415326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 254)
    make_system_call_result_415332 = invoke(stypy.reporting.localization.Localization(__file__, 254, 26), make_system_415326, *[A_415327, M_415328, x0_415329, b_415330], **kwargs_415331)
    
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___415333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 4), make_system_call_result_415332, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_415334 = invoke(stypy.reporting.localization.Localization(__file__, 254, 4), getitem___415333, int_415325)
    
    # Assigning a type to the variable 'tuple_var_assignment_414628' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'tuple_var_assignment_414628', subscript_call_result_415334)
    
    # Assigning a Subscript to a Name (line 254):
    
    # Obtaining the type of the subscript
    int_415335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 4), 'int')
    
    # Call to make_system(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'A' (line 254)
    A_415337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 38), 'A', False)
    # Getting the type of 'M' (line 254)
    M_415338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 40), 'M', False)
    # Getting the type of 'x0' (line 254)
    x0_415339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 42), 'x0', False)
    # Getting the type of 'b' (line 254)
    b_415340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 45), 'b', False)
    # Processing the call keyword arguments (line 254)
    kwargs_415341 = {}
    # Getting the type of 'make_system' (line 254)
    make_system_415336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 254)
    make_system_call_result_415342 = invoke(stypy.reporting.localization.Localization(__file__, 254, 26), make_system_415336, *[A_415337, M_415338, x0_415339, b_415340], **kwargs_415341)
    
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___415343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 4), make_system_call_result_415342, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_415344 = invoke(stypy.reporting.localization.Localization(__file__, 254, 4), getitem___415343, int_415335)
    
    # Assigning a type to the variable 'tuple_var_assignment_414629' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'tuple_var_assignment_414629', subscript_call_result_415344)
    
    # Assigning a Name to a Name (line 254):
    # Getting the type of 'tuple_var_assignment_414625' (line 254)
    tuple_var_assignment_414625_415345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'tuple_var_assignment_414625')
    # Assigning a type to the variable 'A' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'A', tuple_var_assignment_414625_415345)
    
    # Assigning a Name to a Name (line 254):
    # Getting the type of 'tuple_var_assignment_414626' (line 254)
    tuple_var_assignment_414626_415346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'tuple_var_assignment_414626')
    # Assigning a type to the variable 'M' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 6), 'M', tuple_var_assignment_414626_415346)
    
    # Assigning a Name to a Name (line 254):
    # Getting the type of 'tuple_var_assignment_414627' (line 254)
    tuple_var_assignment_414627_415347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'tuple_var_assignment_414627')
    # Assigning a type to the variable 'x' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'x', tuple_var_assignment_414627_415347)
    
    # Assigning a Name to a Name (line 254):
    # Getting the type of 'tuple_var_assignment_414628' (line 254)
    tuple_var_assignment_414628_415348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'tuple_var_assignment_414628')
    # Assigning a type to the variable 'b' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 10), 'b', tuple_var_assignment_414628_415348)
    
    # Assigning a Name to a Name (line 254):
    # Getting the type of 'tuple_var_assignment_414629' (line 254)
    tuple_var_assignment_414629_415349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'tuple_var_assignment_414629')
    # Assigning a type to the variable 'postprocess' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'postprocess', tuple_var_assignment_414629_415349)
    
    
    
    # Call to all(...): (line 256)
    # Processing the call keyword arguments (line 256)
    kwargs_415356 = {}
    
    # Call to isfinite(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'b' (line 256)
    b_415352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 23), 'b', False)
    # Processing the call keyword arguments (line 256)
    kwargs_415353 = {}
    # Getting the type of 'np' (line 256)
    np_415350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 256)
    isfinite_415351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 11), np_415350, 'isfinite')
    # Calling isfinite(args, kwargs) (line 256)
    isfinite_call_result_415354 = invoke(stypy.reporting.localization.Localization(__file__, 256, 11), isfinite_415351, *[b_415352], **kwargs_415353)
    
    # Obtaining the member 'all' of a type (line 256)
    all_415355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 11), isfinite_call_result_415354, 'all')
    # Calling all(args, kwargs) (line 256)
    all_call_result_415357 = invoke(stypy.reporting.localization.Localization(__file__, 256, 11), all_415355, *[], **kwargs_415356)
    
    # Applying the 'not' unary operator (line 256)
    result_not__415358 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 7), 'not', all_call_result_415357)
    
    # Testing the type of an if condition (line 256)
    if_condition_415359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 4), result_not__415358)
    # Assigning a type to the variable 'if_condition_415359' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'if_condition_415359', if_condition_415359)
    # SSA begins for if statement (line 256)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 257)
    # Processing the call arguments (line 257)
    str_415361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 25), 'str', 'RHS must contain only finite numbers')
    # Processing the call keyword arguments (line 257)
    kwargs_415362 = {}
    # Getting the type of 'ValueError' (line 257)
    ValueError_415360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 257)
    ValueError_call_result_415363 = invoke(stypy.reporting.localization.Localization(__file__, 257, 14), ValueError_415360, *[str_415361], **kwargs_415362)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 257, 8), ValueError_call_result_415363, 'raise parameter', BaseException)
    # SSA join for if statement (line 256)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'truncate' (line 259)
    truncate_415364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 7), 'truncate')
    
    # Obtaining an instance of the builtin type 'tuple' (line 259)
    tuple_415365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 259)
    # Adding element type (line 259)
    str_415366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 24), 'str', 'oldest')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 24), tuple_415365, str_415366)
    # Adding element type (line 259)
    str_415367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 34), 'str', 'smallest')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 24), tuple_415365, str_415367)
    
    # Applying the binary operator 'notin' (line 259)
    result_contains_415368 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 7), 'notin', truncate_415364, tuple_415365)
    
    # Testing the type of an if condition (line 259)
    if_condition_415369 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 4), result_contains_415368)
    # Assigning a type to the variable 'if_condition_415369' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'if_condition_415369', if_condition_415369)
    # SSA begins for if statement (line 259)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 260)
    # Processing the call arguments (line 260)
    str_415371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 25), 'str', "Invalid value for 'truncate': %r")
    
    # Obtaining an instance of the builtin type 'tuple' (line 260)
    tuple_415372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 63), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 260)
    # Adding element type (line 260)
    # Getting the type of 'truncate' (line 260)
    truncate_415373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 63), 'truncate', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 63), tuple_415372, truncate_415373)
    
    # Applying the binary operator '%' (line 260)
    result_mod_415374 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 25), '%', str_415371, tuple_415372)
    
    # Processing the call keyword arguments (line 260)
    kwargs_415375 = {}
    # Getting the type of 'ValueError' (line 260)
    ValueError_415370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 260)
    ValueError_call_result_415376 = invoke(stypy.reporting.localization.Localization(__file__, 260, 14), ValueError_415370, *[result_mod_415374], **kwargs_415375)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 260, 8), ValueError_call_result_415376, 'raise parameter', BaseException)
    # SSA join for if statement (line 259)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 262):
    
    # Assigning a Attribute to a Name (line 262):
    # Getting the type of 'A' (line 262)
    A_415377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 13), 'A')
    # Obtaining the member 'matvec' of a type (line 262)
    matvec_415378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 13), A_415377, 'matvec')
    # Assigning a type to the variable 'matvec' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'matvec', matvec_415378)
    
    # Assigning a Attribute to a Name (line 263):
    
    # Assigning a Attribute to a Name (line 263):
    # Getting the type of 'M' (line 263)
    M_415379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 13), 'M')
    # Obtaining the member 'matvec' of a type (line 263)
    matvec_415380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 13), M_415379, 'matvec')
    # Assigning a type to the variable 'psolve' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'psolve', matvec_415380)
    
    # Type idiom detected: calculating its left and rigth part (line 265)
    # Getting the type of 'CU' (line 265)
    CU_415381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 7), 'CU')
    # Getting the type of 'None' (line 265)
    None_415382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 13), 'None')
    
    (may_be_415383, more_types_in_union_415384) = may_be_none(CU_415381, None_415382)

    if may_be_415383:

        if more_types_in_union_415384:
            # Runtime conditional SSA (line 265)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Name (line 266):
        
        # Assigning a List to a Name (line 266):
        
        # Obtaining an instance of the builtin type 'list' (line 266)
        list_415385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 266)
        
        # Assigning a type to the variable 'CU' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'CU', list_415385)

        if more_types_in_union_415384:
            # SSA join for if statement (line 265)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 268)
    # Getting the type of 'k' (line 268)
    k_415386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 7), 'k')
    # Getting the type of 'None' (line 268)
    None_415387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'None')
    
    (may_be_415388, more_types_in_union_415389) = may_be_none(k_415386, None_415387)

    if may_be_415388:

        if more_types_in_union_415389:
            # Runtime conditional SSA (line 268)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 269):
        
        # Assigning a Name to a Name (line 269):
        # Getting the type of 'm' (line 269)
        m_415390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'm')
        # Assigning a type to the variable 'k' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'k', m_415390)

        if more_types_in_union_415389:
            # SSA join for if statement (line 268)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Tuple to a Tuple (line 271):
    
    # Assigning a Name to a Name (line 271):
    # Getting the type of 'None' (line 271)
    None_415391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 22), 'None')
    # Assigning a type to the variable 'tuple_assignment_414630' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'tuple_assignment_414630', None_415391)
    
    # Assigning a Name to a Name (line 271):
    # Getting the type of 'None' (line 271)
    None_415392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 28), 'None')
    # Assigning a type to the variable 'tuple_assignment_414631' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'tuple_assignment_414631', None_415392)
    
    # Assigning a Name to a Name (line 271):
    # Getting the type of 'None' (line 271)
    None_415393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 34), 'None')
    # Assigning a type to the variable 'tuple_assignment_414632' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'tuple_assignment_414632', None_415393)
    
    # Assigning a Name to a Name (line 271):
    # Getting the type of 'tuple_assignment_414630' (line 271)
    tuple_assignment_414630_415394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'tuple_assignment_414630')
    # Assigning a type to the variable 'axpy' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'axpy', tuple_assignment_414630_415394)
    
    # Assigning a Name to a Name (line 271):
    # Getting the type of 'tuple_assignment_414631' (line 271)
    tuple_assignment_414631_415395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'tuple_assignment_414631')
    # Assigning a type to the variable 'dot' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 10), 'dot', tuple_assignment_414631_415395)
    
    # Assigning a Name to a Name (line 271):
    # Getting the type of 'tuple_assignment_414632' (line 271)
    tuple_assignment_414632_415396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'tuple_assignment_414632')
    # Assigning a type to the variable 'scal' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'scal', tuple_assignment_414632_415396)
    
    # Assigning a BinOp to a Name (line 273):
    
    # Assigning a BinOp to a Name (line 273):
    # Getting the type of 'b' (line 273)
    b_415397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'b')
    
    # Call to matvec(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'x' (line 273)
    x_415399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 19), 'x', False)
    # Processing the call keyword arguments (line 273)
    kwargs_415400 = {}
    # Getting the type of 'matvec' (line 273)
    matvec_415398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'matvec', False)
    # Calling matvec(args, kwargs) (line 273)
    matvec_call_result_415401 = invoke(stypy.reporting.localization.Localization(__file__, 273, 12), matvec_415398, *[x_415399], **kwargs_415400)
    
    # Applying the binary operator '-' (line 273)
    result_sub_415402 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 8), '-', b_415397, matvec_call_result_415401)
    
    # Assigning a type to the variable 'r' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'r', result_sub_415402)
    
    # Assigning a Call to a Tuple (line 275):
    
    # Assigning a Subscript to a Name (line 275):
    
    # Obtaining the type of the subscript
    int_415403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 4), 'int')
    
    # Call to get_blas_funcs(...): (line 275)
    # Processing the call arguments (line 275)
    
    # Obtaining an instance of the builtin type 'list' (line 275)
    list_415405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 275)
    # Adding element type (line 275)
    str_415406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 44), 'str', 'axpy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415405, str_415406)
    # Adding element type (line 275)
    str_415407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 52), 'str', 'dot')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415405, str_415407)
    # Adding element type (line 275)
    str_415408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 59), 'str', 'scal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415405, str_415408)
    # Adding element type (line 275)
    str_415409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 67), 'str', 'nrm2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415405, str_415409)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 275)
    tuple_415410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 77), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 275)
    # Adding element type (line 275)
    # Getting the type of 'x' (line 275)
    x_415411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 77), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 77), tuple_415410, x_415411)
    # Adding element type (line 275)
    # Getting the type of 'r' (line 275)
    r_415412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 80), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 77), tuple_415410, r_415412)
    
    # Processing the call keyword arguments (line 275)
    kwargs_415413 = {}
    # Getting the type of 'get_blas_funcs' (line 275)
    get_blas_funcs_415404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 28), 'get_blas_funcs', False)
    # Calling get_blas_funcs(args, kwargs) (line 275)
    get_blas_funcs_call_result_415414 = invoke(stypy.reporting.localization.Localization(__file__, 275, 28), get_blas_funcs_415404, *[list_415405, tuple_415410], **kwargs_415413)
    
    # Obtaining the member '__getitem__' of a type (line 275)
    getitem___415415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 4), get_blas_funcs_call_result_415414, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 275)
    subscript_call_result_415416 = invoke(stypy.reporting.localization.Localization(__file__, 275, 4), getitem___415415, int_415403)
    
    # Assigning a type to the variable 'tuple_var_assignment_414633' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'tuple_var_assignment_414633', subscript_call_result_415416)
    
    # Assigning a Subscript to a Name (line 275):
    
    # Obtaining the type of the subscript
    int_415417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 4), 'int')
    
    # Call to get_blas_funcs(...): (line 275)
    # Processing the call arguments (line 275)
    
    # Obtaining an instance of the builtin type 'list' (line 275)
    list_415419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 275)
    # Adding element type (line 275)
    str_415420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 44), 'str', 'axpy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415419, str_415420)
    # Adding element type (line 275)
    str_415421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 52), 'str', 'dot')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415419, str_415421)
    # Adding element type (line 275)
    str_415422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 59), 'str', 'scal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415419, str_415422)
    # Adding element type (line 275)
    str_415423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 67), 'str', 'nrm2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415419, str_415423)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 275)
    tuple_415424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 77), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 275)
    # Adding element type (line 275)
    # Getting the type of 'x' (line 275)
    x_415425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 77), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 77), tuple_415424, x_415425)
    # Adding element type (line 275)
    # Getting the type of 'r' (line 275)
    r_415426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 80), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 77), tuple_415424, r_415426)
    
    # Processing the call keyword arguments (line 275)
    kwargs_415427 = {}
    # Getting the type of 'get_blas_funcs' (line 275)
    get_blas_funcs_415418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 28), 'get_blas_funcs', False)
    # Calling get_blas_funcs(args, kwargs) (line 275)
    get_blas_funcs_call_result_415428 = invoke(stypy.reporting.localization.Localization(__file__, 275, 28), get_blas_funcs_415418, *[list_415419, tuple_415424], **kwargs_415427)
    
    # Obtaining the member '__getitem__' of a type (line 275)
    getitem___415429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 4), get_blas_funcs_call_result_415428, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 275)
    subscript_call_result_415430 = invoke(stypy.reporting.localization.Localization(__file__, 275, 4), getitem___415429, int_415417)
    
    # Assigning a type to the variable 'tuple_var_assignment_414634' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'tuple_var_assignment_414634', subscript_call_result_415430)
    
    # Assigning a Subscript to a Name (line 275):
    
    # Obtaining the type of the subscript
    int_415431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 4), 'int')
    
    # Call to get_blas_funcs(...): (line 275)
    # Processing the call arguments (line 275)
    
    # Obtaining an instance of the builtin type 'list' (line 275)
    list_415433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 275)
    # Adding element type (line 275)
    str_415434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 44), 'str', 'axpy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415433, str_415434)
    # Adding element type (line 275)
    str_415435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 52), 'str', 'dot')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415433, str_415435)
    # Adding element type (line 275)
    str_415436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 59), 'str', 'scal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415433, str_415436)
    # Adding element type (line 275)
    str_415437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 67), 'str', 'nrm2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415433, str_415437)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 275)
    tuple_415438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 77), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 275)
    # Adding element type (line 275)
    # Getting the type of 'x' (line 275)
    x_415439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 77), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 77), tuple_415438, x_415439)
    # Adding element type (line 275)
    # Getting the type of 'r' (line 275)
    r_415440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 80), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 77), tuple_415438, r_415440)
    
    # Processing the call keyword arguments (line 275)
    kwargs_415441 = {}
    # Getting the type of 'get_blas_funcs' (line 275)
    get_blas_funcs_415432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 28), 'get_blas_funcs', False)
    # Calling get_blas_funcs(args, kwargs) (line 275)
    get_blas_funcs_call_result_415442 = invoke(stypy.reporting.localization.Localization(__file__, 275, 28), get_blas_funcs_415432, *[list_415433, tuple_415438], **kwargs_415441)
    
    # Obtaining the member '__getitem__' of a type (line 275)
    getitem___415443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 4), get_blas_funcs_call_result_415442, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 275)
    subscript_call_result_415444 = invoke(stypy.reporting.localization.Localization(__file__, 275, 4), getitem___415443, int_415431)
    
    # Assigning a type to the variable 'tuple_var_assignment_414635' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'tuple_var_assignment_414635', subscript_call_result_415444)
    
    # Assigning a Subscript to a Name (line 275):
    
    # Obtaining the type of the subscript
    int_415445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 4), 'int')
    
    # Call to get_blas_funcs(...): (line 275)
    # Processing the call arguments (line 275)
    
    # Obtaining an instance of the builtin type 'list' (line 275)
    list_415447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 275)
    # Adding element type (line 275)
    str_415448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 44), 'str', 'axpy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415447, str_415448)
    # Adding element type (line 275)
    str_415449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 52), 'str', 'dot')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415447, str_415449)
    # Adding element type (line 275)
    str_415450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 59), 'str', 'scal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415447, str_415450)
    # Adding element type (line 275)
    str_415451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 67), 'str', 'nrm2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 43), list_415447, str_415451)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 275)
    tuple_415452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 77), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 275)
    # Adding element type (line 275)
    # Getting the type of 'x' (line 275)
    x_415453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 77), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 77), tuple_415452, x_415453)
    # Adding element type (line 275)
    # Getting the type of 'r' (line 275)
    r_415454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 80), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 77), tuple_415452, r_415454)
    
    # Processing the call keyword arguments (line 275)
    kwargs_415455 = {}
    # Getting the type of 'get_blas_funcs' (line 275)
    get_blas_funcs_415446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 28), 'get_blas_funcs', False)
    # Calling get_blas_funcs(args, kwargs) (line 275)
    get_blas_funcs_call_result_415456 = invoke(stypy.reporting.localization.Localization(__file__, 275, 28), get_blas_funcs_415446, *[list_415447, tuple_415452], **kwargs_415455)
    
    # Obtaining the member '__getitem__' of a type (line 275)
    getitem___415457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 4), get_blas_funcs_call_result_415456, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 275)
    subscript_call_result_415458 = invoke(stypy.reporting.localization.Localization(__file__, 275, 4), getitem___415457, int_415445)
    
    # Assigning a type to the variable 'tuple_var_assignment_414636' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'tuple_var_assignment_414636', subscript_call_result_415458)
    
    # Assigning a Name to a Name (line 275):
    # Getting the type of 'tuple_var_assignment_414633' (line 275)
    tuple_var_assignment_414633_415459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'tuple_var_assignment_414633')
    # Assigning a type to the variable 'axpy' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'axpy', tuple_var_assignment_414633_415459)
    
    # Assigning a Name to a Name (line 275):
    # Getting the type of 'tuple_var_assignment_414634' (line 275)
    tuple_var_assignment_414634_415460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'tuple_var_assignment_414634')
    # Assigning a type to the variable 'dot' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 10), 'dot', tuple_var_assignment_414634_415460)
    
    # Assigning a Name to a Name (line 275):
    # Getting the type of 'tuple_var_assignment_414635' (line 275)
    tuple_var_assignment_414635_415461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'tuple_var_assignment_414635')
    # Assigning a type to the variable 'scal' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 15), 'scal', tuple_var_assignment_414635_415461)
    
    # Assigning a Name to a Name (line 275):
    # Getting the type of 'tuple_var_assignment_414636' (line 275)
    tuple_var_assignment_414636_415462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'tuple_var_assignment_414636')
    # Assigning a type to the variable 'nrm2' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 21), 'nrm2', tuple_var_assignment_414636_415462)
    
    # Assigning a Call to a Name (line 277):
    
    # Assigning a Call to a Name (line 277):
    
    # Call to nrm2(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'b' (line 277)
    b_415464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 18), 'b', False)
    # Processing the call keyword arguments (line 277)
    kwargs_415465 = {}
    # Getting the type of 'nrm2' (line 277)
    nrm2_415463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 13), 'nrm2', False)
    # Calling nrm2(args, kwargs) (line 277)
    nrm2_call_result_415466 = invoke(stypy.reporting.localization.Localization(__file__, 277, 13), nrm2_415463, *[b_415464], **kwargs_415465)
    
    # Assigning a type to the variable 'b_norm' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'b_norm', nrm2_call_result_415466)
    
    
    # Getting the type of 'b_norm' (line 278)
    b_norm_415467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 7), 'b_norm')
    int_415468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 17), 'int')
    # Applying the binary operator '==' (line 278)
    result_eq_415469 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 7), '==', b_norm_415467, int_415468)
    
    # Testing the type of an if condition (line 278)
    if_condition_415470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 4), result_eq_415469)
    # Assigning a type to the variable 'if_condition_415470' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'if_condition_415470', if_condition_415470)
    # SSA begins for if statement (line 278)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 279):
    
    # Assigning a Num to a Name (line 279):
    int_415471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 17), 'int')
    # Assigning a type to the variable 'b_norm' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'b_norm', int_415471)
    # SSA join for if statement (line 278)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'discard_C' (line 281)
    discard_C_415472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 7), 'discard_C')
    # Testing the type of an if condition (line 281)
    if_condition_415473 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 4), discard_C_415472)
    # Assigning a type to the variable 'if_condition_415473' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'if_condition_415473', if_condition_415473)
    # SSA begins for if statement (line 281)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a ListComp to a Subscript (line 282):
    
    # Assigning a ListComp to a Subscript (line 282):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'CU' (line 282)
    CU_415477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 39), 'CU')
    comprehension_415478 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 17), CU_415477)
    # Assigning a type to the variable 'c' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 17), comprehension_415478))
    # Assigning a type to the variable 'u' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'u', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 17), comprehension_415478))
    
    # Obtaining an instance of the builtin type 'tuple' (line 282)
    tuple_415474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 282)
    # Adding element type (line 282)
    # Getting the type of 'None' (line 282)
    None_415475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 18), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 18), tuple_415474, None_415475)
    # Adding element type (line 282)
    # Getting the type of 'u' (line 282)
    u_415476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'u')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 18), tuple_415474, u_415476)
    
    list_415479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 17), list_415479, tuple_415474)
    # Getting the type of 'CU' (line 282)
    CU_415480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'CU')
    slice_415481 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 282, 8), None, None, None)
    # Storing an element on a container (line 282)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 8), CU_415480, (slice_415481, list_415479))
    # SSA join for if statement (line 281)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'CU' (line 285)
    CU_415482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 7), 'CU')
    # Testing the type of an if condition (line 285)
    if_condition_415483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 4), CU_415482)
    # Assigning a type to the variable 'if_condition_415483' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'if_condition_415483', if_condition_415483)
    # SSA begins for if statement (line 285)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to sort(...): (line 287)
    # Processing the call keyword arguments (line 287)

    @norecursion
    def _stypy_temp_lambda_224(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_224'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_224', 287, 20, True)
        # Passed parameters checking function
        _stypy_temp_lambda_224.stypy_localization = localization
        _stypy_temp_lambda_224.stypy_type_of_self = None
        _stypy_temp_lambda_224.stypy_type_store = module_type_store
        _stypy_temp_lambda_224.stypy_function_name = '_stypy_temp_lambda_224'
        _stypy_temp_lambda_224.stypy_param_names_list = ['cu']
        _stypy_temp_lambda_224.stypy_varargs_param_name = None
        _stypy_temp_lambda_224.stypy_kwargs_param_name = None
        _stypy_temp_lambda_224.stypy_call_defaults = defaults
        _stypy_temp_lambda_224.stypy_call_varargs = varargs
        _stypy_temp_lambda_224.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_224', ['cu'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_224', ['cu'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        
        # Obtaining the type of the subscript
        int_415486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 34), 'int')
        # Getting the type of 'cu' (line 287)
        cu_415487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'cu', False)
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___415488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 31), cu_415487, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_415489 = invoke(stypy.reporting.localization.Localization(__file__, 287, 31), getitem___415488, int_415486)
        
        # Getting the type of 'None' (line 287)
        None_415490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 44), 'None', False)
        # Applying the binary operator 'isnot' (line 287)
        result_is_not_415491 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 31), 'isnot', subscript_call_result_415489, None_415490)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'stypy_return_type', result_is_not_415491)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_224' in the type store
        # Getting the type of 'stypy_return_type' (line 287)
        stypy_return_type_415492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_415492)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_224'
        return stypy_return_type_415492

    # Assigning a type to the variable '_stypy_temp_lambda_224' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), '_stypy_temp_lambda_224', _stypy_temp_lambda_224)
    # Getting the type of '_stypy_temp_lambda_224' (line 287)
    _stypy_temp_lambda_224_415493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), '_stypy_temp_lambda_224')
    keyword_415494 = _stypy_temp_lambda_224_415493
    kwargs_415495 = {'key': keyword_415494}
    # Getting the type of 'CU' (line 287)
    CU_415484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'CU', False)
    # Obtaining the member 'sort' of a type (line 287)
    sort_415485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), CU_415484, 'sort')
    # Calling sort(args, kwargs) (line 287)
    sort_call_result_415496 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), sort_415485, *[], **kwargs_415495)
    
    
    # Assigning a Call to a Name (line 290):
    
    # Assigning a Call to a Name (line 290):
    
    # Call to empty(...): (line 290)
    # Processing the call arguments (line 290)
    
    # Obtaining an instance of the builtin type 'tuple' (line 290)
    tuple_415499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 290)
    # Adding element type (line 290)
    
    # Obtaining the type of the subscript
    int_415500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 30), 'int')
    # Getting the type of 'A' (line 290)
    A_415501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 22), 'A', False)
    # Obtaining the member 'shape' of a type (line 290)
    shape_415502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 22), A_415501, 'shape')
    # Obtaining the member '__getitem__' of a type (line 290)
    getitem___415503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 22), shape_415502, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 290)
    subscript_call_result_415504 = invoke(stypy.reporting.localization.Localization(__file__, 290, 22), getitem___415503, int_415500)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 22), tuple_415499, subscript_call_result_415504)
    # Adding element type (line 290)
    
    # Call to len(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'CU' (line 290)
    CU_415506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 38), 'CU', False)
    # Processing the call keyword arguments (line 290)
    kwargs_415507 = {}
    # Getting the type of 'len' (line 290)
    len_415505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 34), 'len', False)
    # Calling len(args, kwargs) (line 290)
    len_call_result_415508 = invoke(stypy.reporting.localization.Localization(__file__, 290, 34), len_415505, *[CU_415506], **kwargs_415507)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 22), tuple_415499, len_call_result_415508)
    
    # Processing the call keyword arguments (line 290)
    # Getting the type of 'r' (line 290)
    r_415509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 50), 'r', False)
    # Obtaining the member 'dtype' of a type (line 290)
    dtype_415510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 50), r_415509, 'dtype')
    keyword_415511 = dtype_415510
    str_415512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 65), 'str', 'F')
    keyword_415513 = str_415512
    kwargs_415514 = {'dtype': keyword_415511, 'order': keyword_415513}
    # Getting the type of 'np' (line 290)
    np_415497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'np', False)
    # Obtaining the member 'empty' of a type (line 290)
    empty_415498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), np_415497, 'empty')
    # Calling empty(args, kwargs) (line 290)
    empty_call_result_415515 = invoke(stypy.reporting.localization.Localization(__file__, 290, 12), empty_415498, *[tuple_415499], **kwargs_415514)
    
    # Assigning a type to the variable 'C' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'C', empty_call_result_415515)
    
    # Assigning a List to a Name (line 291):
    
    # Assigning a List to a Name (line 291):
    
    # Obtaining an instance of the builtin type 'list' (line 291)
    list_415516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 291)
    
    # Assigning a type to the variable 'us' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'us', list_415516)
    
    # Assigning a Num to a Name (line 292):
    
    # Assigning a Num to a Name (line 292):
    int_415517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 12), 'int')
    # Assigning a type to the variable 'j' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'j', int_415517)
    
    # Getting the type of 'CU' (line 293)
    CU_415518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 14), 'CU')
    # Testing the type of an if condition (line 293)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 8), CU_415518)
    # SSA begins for while statement (line 293)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 295):
    
    # Assigning a Subscript to a Name (line 295):
    
    # Obtaining the type of the subscript
    int_415519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 12), 'int')
    
    # Call to pop(...): (line 295)
    # Processing the call arguments (line 295)
    int_415522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 26), 'int')
    # Processing the call keyword arguments (line 295)
    kwargs_415523 = {}
    # Getting the type of 'CU' (line 295)
    CU_415520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 19), 'CU', False)
    # Obtaining the member 'pop' of a type (line 295)
    pop_415521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 19), CU_415520, 'pop')
    # Calling pop(args, kwargs) (line 295)
    pop_call_result_415524 = invoke(stypy.reporting.localization.Localization(__file__, 295, 19), pop_415521, *[int_415522], **kwargs_415523)
    
    # Obtaining the member '__getitem__' of a type (line 295)
    getitem___415525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 12), pop_call_result_415524, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 295)
    subscript_call_result_415526 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), getitem___415525, int_415519)
    
    # Assigning a type to the variable 'tuple_var_assignment_414637' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'tuple_var_assignment_414637', subscript_call_result_415526)
    
    # Assigning a Subscript to a Name (line 295):
    
    # Obtaining the type of the subscript
    int_415527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 12), 'int')
    
    # Call to pop(...): (line 295)
    # Processing the call arguments (line 295)
    int_415530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 26), 'int')
    # Processing the call keyword arguments (line 295)
    kwargs_415531 = {}
    # Getting the type of 'CU' (line 295)
    CU_415528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 19), 'CU', False)
    # Obtaining the member 'pop' of a type (line 295)
    pop_415529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 19), CU_415528, 'pop')
    # Calling pop(args, kwargs) (line 295)
    pop_call_result_415532 = invoke(stypy.reporting.localization.Localization(__file__, 295, 19), pop_415529, *[int_415530], **kwargs_415531)
    
    # Obtaining the member '__getitem__' of a type (line 295)
    getitem___415533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 12), pop_call_result_415532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 295)
    subscript_call_result_415534 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), getitem___415533, int_415527)
    
    # Assigning a type to the variable 'tuple_var_assignment_414638' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'tuple_var_assignment_414638', subscript_call_result_415534)
    
    # Assigning a Name to a Name (line 295):
    # Getting the type of 'tuple_var_assignment_414637' (line 295)
    tuple_var_assignment_414637_415535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'tuple_var_assignment_414637')
    # Assigning a type to the variable 'c' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'c', tuple_var_assignment_414637_415535)
    
    # Assigning a Name to a Name (line 295):
    # Getting the type of 'tuple_var_assignment_414638' (line 295)
    tuple_var_assignment_414638_415536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'tuple_var_assignment_414638')
    # Assigning a type to the variable 'u' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'u', tuple_var_assignment_414638_415536)
    
    # Type idiom detected: calculating its left and rigth part (line 296)
    # Getting the type of 'c' (line 296)
    c_415537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 15), 'c')
    # Getting the type of 'None' (line 296)
    None_415538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 20), 'None')
    
    (may_be_415539, more_types_in_union_415540) = may_be_none(c_415537, None_415538)

    if may_be_415539:

        if more_types_in_union_415540:
            # Runtime conditional SSA (line 296)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 297):
        
        # Assigning a Call to a Name (line 297):
        
        # Call to matvec(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'u' (line 297)
        u_415542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 27), 'u', False)
        # Processing the call keyword arguments (line 297)
        kwargs_415543 = {}
        # Getting the type of 'matvec' (line 297)
        matvec_415541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 20), 'matvec', False)
        # Calling matvec(args, kwargs) (line 297)
        matvec_call_result_415544 = invoke(stypy.reporting.localization.Localization(__file__, 297, 20), matvec_415541, *[u_415542], **kwargs_415543)
        
        # Assigning a type to the variable 'c' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'c', matvec_call_result_415544)

        if more_types_in_union_415540:
            # SSA join for if statement (line 296)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Subscript (line 298):
    
    # Assigning a Name to a Subscript (line 298):
    # Getting the type of 'c' (line 298)
    c_415545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 21), 'c')
    # Getting the type of 'C' (line 298)
    C_415546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'C')
    slice_415547 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 298, 12), None, None, None)
    # Getting the type of 'j' (line 298)
    j_415548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'j')
    # Storing an element on a container (line 298)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 12), C_415546, ((slice_415547, j_415548), c_415545))
    
    # Getting the type of 'j' (line 299)
    j_415549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'j')
    int_415550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 17), 'int')
    # Applying the binary operator '+=' (line 299)
    result_iadd_415551 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 12), '+=', j_415549, int_415550)
    # Assigning a type to the variable 'j' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'j', result_iadd_415551)
    
    
    # Call to append(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 'u' (line 300)
    u_415554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 22), 'u', False)
    # Processing the call keyword arguments (line 300)
    kwargs_415555 = {}
    # Getting the type of 'us' (line 300)
    us_415552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'us', False)
    # Obtaining the member 'append' of a type (line 300)
    append_415553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 12), us_415552, 'append')
    # Calling append(args, kwargs) (line 300)
    append_call_result_415556 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), append_415553, *[u_415554], **kwargs_415555)
    
    # SSA join for while statement (line 293)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 303):
    
    # Assigning a Subscript to a Name (line 303):
    
    # Obtaining the type of the subscript
    int_415557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 8), 'int')
    
    # Call to qr(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'C' (line 303)
    C_415559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 21), 'C', False)
    # Processing the call keyword arguments (line 303)
    # Getting the type of 'True' (line 303)
    True_415560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 36), 'True', False)
    keyword_415561 = True_415560
    str_415562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 47), 'str', 'economic')
    keyword_415563 = str_415562
    # Getting the type of 'True' (line 303)
    True_415564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 68), 'True', False)
    keyword_415565 = True_415564
    kwargs_415566 = {'pivoting': keyword_415565, 'overwrite_a': keyword_415561, 'mode': keyword_415563}
    # Getting the type of 'qr' (line 303)
    qr_415558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 18), 'qr', False)
    # Calling qr(args, kwargs) (line 303)
    qr_call_result_415567 = invoke(stypy.reporting.localization.Localization(__file__, 303, 18), qr_415558, *[C_415559], **kwargs_415566)
    
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___415568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 8), qr_call_result_415567, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_415569 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), getitem___415568, int_415557)
    
    # Assigning a type to the variable 'tuple_var_assignment_414639' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'tuple_var_assignment_414639', subscript_call_result_415569)
    
    # Assigning a Subscript to a Name (line 303):
    
    # Obtaining the type of the subscript
    int_415570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 8), 'int')
    
    # Call to qr(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'C' (line 303)
    C_415572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 21), 'C', False)
    # Processing the call keyword arguments (line 303)
    # Getting the type of 'True' (line 303)
    True_415573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 36), 'True', False)
    keyword_415574 = True_415573
    str_415575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 47), 'str', 'economic')
    keyword_415576 = str_415575
    # Getting the type of 'True' (line 303)
    True_415577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 68), 'True', False)
    keyword_415578 = True_415577
    kwargs_415579 = {'pivoting': keyword_415578, 'overwrite_a': keyword_415574, 'mode': keyword_415576}
    # Getting the type of 'qr' (line 303)
    qr_415571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 18), 'qr', False)
    # Calling qr(args, kwargs) (line 303)
    qr_call_result_415580 = invoke(stypy.reporting.localization.Localization(__file__, 303, 18), qr_415571, *[C_415572], **kwargs_415579)
    
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___415581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 8), qr_call_result_415580, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_415582 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), getitem___415581, int_415570)
    
    # Assigning a type to the variable 'tuple_var_assignment_414640' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'tuple_var_assignment_414640', subscript_call_result_415582)
    
    # Assigning a Subscript to a Name (line 303):
    
    # Obtaining the type of the subscript
    int_415583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 8), 'int')
    
    # Call to qr(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'C' (line 303)
    C_415585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 21), 'C', False)
    # Processing the call keyword arguments (line 303)
    # Getting the type of 'True' (line 303)
    True_415586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 36), 'True', False)
    keyword_415587 = True_415586
    str_415588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 47), 'str', 'economic')
    keyword_415589 = str_415588
    # Getting the type of 'True' (line 303)
    True_415590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 68), 'True', False)
    keyword_415591 = True_415590
    kwargs_415592 = {'pivoting': keyword_415591, 'overwrite_a': keyword_415587, 'mode': keyword_415589}
    # Getting the type of 'qr' (line 303)
    qr_415584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 18), 'qr', False)
    # Calling qr(args, kwargs) (line 303)
    qr_call_result_415593 = invoke(stypy.reporting.localization.Localization(__file__, 303, 18), qr_415584, *[C_415585], **kwargs_415592)
    
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___415594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 8), qr_call_result_415593, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_415595 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), getitem___415594, int_415583)
    
    # Assigning a type to the variable 'tuple_var_assignment_414641' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'tuple_var_assignment_414641', subscript_call_result_415595)
    
    # Assigning a Name to a Name (line 303):
    # Getting the type of 'tuple_var_assignment_414639' (line 303)
    tuple_var_assignment_414639_415596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'tuple_var_assignment_414639')
    # Assigning a type to the variable 'Q' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'Q', tuple_var_assignment_414639_415596)
    
    # Assigning a Name to a Name (line 303):
    # Getting the type of 'tuple_var_assignment_414640' (line 303)
    tuple_var_assignment_414640_415597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'tuple_var_assignment_414640')
    # Assigning a type to the variable 'R' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 11), 'R', tuple_var_assignment_414640_415597)
    
    # Assigning a Name to a Name (line 303):
    # Getting the type of 'tuple_var_assignment_414641' (line 303)
    tuple_var_assignment_414641_415598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'tuple_var_assignment_414641')
    # Assigning a type to the variable 'P' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 14), 'P', tuple_var_assignment_414641_415598)
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 304, 8), module_type_store, 'C')
    
    # Assigning a Call to a Name (line 307):
    
    # Assigning a Call to a Name (line 307):
    
    # Call to list(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'Q' (line 307)
    Q_415600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 18), 'Q', False)
    # Obtaining the member 'T' of a type (line 307)
    T_415601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 18), Q_415600, 'T')
    # Processing the call keyword arguments (line 307)
    kwargs_415602 = {}
    # Getting the type of 'list' (line 307)
    list_415599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 13), 'list', False)
    # Calling list(args, kwargs) (line 307)
    list_call_result_415603 = invoke(stypy.reporting.localization.Localization(__file__, 307, 13), list_415599, *[T_415601], **kwargs_415602)
    
    # Assigning a type to the variable 'cs' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'cs', list_call_result_415603)
    
    # Assigning a List to a Name (line 310):
    
    # Assigning a List to a Name (line 310):
    
    # Obtaining an instance of the builtin type 'list' (line 310)
    list_415604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 310)
    
    # Assigning a type to the variable 'new_us' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'new_us', list_415604)
    
    
    # Call to xrange(...): (line 311)
    # Processing the call arguments (line 311)
    
    # Call to len(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'cs' (line 311)
    cs_415607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 28), 'cs', False)
    # Processing the call keyword arguments (line 311)
    kwargs_415608 = {}
    # Getting the type of 'len' (line 311)
    len_415606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 24), 'len', False)
    # Calling len(args, kwargs) (line 311)
    len_call_result_415609 = invoke(stypy.reporting.localization.Localization(__file__, 311, 24), len_415606, *[cs_415607], **kwargs_415608)
    
    # Processing the call keyword arguments (line 311)
    kwargs_415610 = {}
    # Getting the type of 'xrange' (line 311)
    xrange_415605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 17), 'xrange', False)
    # Calling xrange(args, kwargs) (line 311)
    xrange_call_result_415611 = invoke(stypy.reporting.localization.Localization(__file__, 311, 17), xrange_415605, *[len_call_result_415609], **kwargs_415610)
    
    # Testing the type of a for loop iterable (line 311)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 311, 8), xrange_call_result_415611)
    # Getting the type of the for loop variable (line 311)
    for_loop_var_415612 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 311, 8), xrange_call_result_415611)
    # Assigning a type to the variable 'j' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'j', for_loop_var_415612)
    # SSA begins for a for statement (line 311)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 312):
    
    # Assigning a Subscript to a Name (line 312):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 312)
    j_415613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 21), 'j')
    # Getting the type of 'P' (line 312)
    P_415614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 19), 'P')
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___415615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 19), P_415614, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_415616 = invoke(stypy.reporting.localization.Localization(__file__, 312, 19), getitem___415615, j_415613)
    
    # Getting the type of 'us' (line 312)
    us_415617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 16), 'us')
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___415618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 16), us_415617, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_415619 = invoke(stypy.reporting.localization.Localization(__file__, 312, 16), getitem___415618, subscript_call_result_415616)
    
    # Assigning a type to the variable 'u' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'u', subscript_call_result_415619)
    
    
    # Call to xrange(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'j' (line 313)
    j_415621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 28), 'j', False)
    # Processing the call keyword arguments (line 313)
    kwargs_415622 = {}
    # Getting the type of 'xrange' (line 313)
    xrange_415620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 21), 'xrange', False)
    # Calling xrange(args, kwargs) (line 313)
    xrange_call_result_415623 = invoke(stypy.reporting.localization.Localization(__file__, 313, 21), xrange_415620, *[j_415621], **kwargs_415622)
    
    # Testing the type of a for loop iterable (line 313)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 313, 12), xrange_call_result_415623)
    # Getting the type of the for loop variable (line 313)
    for_loop_var_415624 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 313, 12), xrange_call_result_415623)
    # Assigning a type to the variable 'i' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'i', for_loop_var_415624)
    # SSA begins for a for statement (line 313)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 314):
    
    # Assigning a Call to a Name (line 314):
    
    # Call to axpy(...): (line 314)
    # Processing the call arguments (line 314)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 314)
    i_415626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 30), 'i', False)
    # Getting the type of 'P' (line 314)
    P_415627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 28), 'P', False)
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___415628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 28), P_415627, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_415629 = invoke(stypy.reporting.localization.Localization(__file__, 314, 28), getitem___415628, i_415626)
    
    # Getting the type of 'us' (line 314)
    us_415630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 25), 'us', False)
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___415631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 25), us_415630, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_415632 = invoke(stypy.reporting.localization.Localization(__file__, 314, 25), getitem___415631, subscript_call_result_415629)
    
    # Getting the type of 'u' (line 314)
    u_415633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 35), 'u', False)
    
    # Obtaining the type of the subscript
    int_415634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 46), 'int')
    # Getting the type of 'u' (line 314)
    u_415635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 38), 'u', False)
    # Obtaining the member 'shape' of a type (line 314)
    shape_415636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 38), u_415635, 'shape')
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___415637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 38), shape_415636, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_415638 = invoke(stypy.reporting.localization.Localization(__file__, 314, 38), getitem___415637, int_415634)
    
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 314)
    tuple_415639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 314)
    # Adding element type (line 314)
    # Getting the type of 'i' (line 314)
    i_415640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 53), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 53), tuple_415639, i_415640)
    # Adding element type (line 314)
    # Getting the type of 'j' (line 314)
    j_415641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 55), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 53), tuple_415639, j_415641)
    
    # Getting the type of 'R' (line 314)
    R_415642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 51), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___415643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 51), R_415642, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_415644 = invoke(stypy.reporting.localization.Localization(__file__, 314, 51), getitem___415643, tuple_415639)
    
    # Applying the 'usub' unary operator (line 314)
    result___neg___415645 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 50), 'usub', subscript_call_result_415644)
    
    # Processing the call keyword arguments (line 314)
    kwargs_415646 = {}
    # Getting the type of 'axpy' (line 314)
    axpy_415625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 20), 'axpy', False)
    # Calling axpy(args, kwargs) (line 314)
    axpy_call_result_415647 = invoke(stypy.reporting.localization.Localization(__file__, 314, 20), axpy_415625, *[subscript_call_result_415632, u_415633, subscript_call_result_415638, result___neg___415645], **kwargs_415646)
    
    # Assigning a type to the variable 'u' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'u', axpy_call_result_415647)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to abs(...): (line 315)
    # Processing the call arguments (line 315)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 315)
    tuple_415649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 315)
    # Adding element type (line 315)
    # Getting the type of 'j' (line 315)
    j_415650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 21), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 21), tuple_415649, j_415650)
    # Adding element type (line 315)
    # Getting the type of 'j' (line 315)
    j_415651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 21), tuple_415649, j_415651)
    
    # Getting the type of 'R' (line 315)
    R_415652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 19), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 315)
    getitem___415653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 19), R_415652, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 315)
    subscript_call_result_415654 = invoke(stypy.reporting.localization.Localization(__file__, 315, 19), getitem___415653, tuple_415649)
    
    # Processing the call keyword arguments (line 315)
    kwargs_415655 = {}
    # Getting the type of 'abs' (line 315)
    abs_415648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 15), 'abs', False)
    # Calling abs(args, kwargs) (line 315)
    abs_call_result_415656 = invoke(stypy.reporting.localization.Localization(__file__, 315, 15), abs_415648, *[subscript_call_result_415654], **kwargs_415655)
    
    float_415657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 29), 'float')
    
    # Call to abs(...): (line 315)
    # Processing the call arguments (line 315)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 315)
    tuple_415659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 315)
    # Adding element type (line 315)
    int_415660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 43), tuple_415659, int_415660)
    # Adding element type (line 315)
    int_415661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 43), tuple_415659, int_415661)
    
    # Getting the type of 'R' (line 315)
    R_415662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 41), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 315)
    getitem___415663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 41), R_415662, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 315)
    subscript_call_result_415664 = invoke(stypy.reporting.localization.Localization(__file__, 315, 41), getitem___415663, tuple_415659)
    
    # Processing the call keyword arguments (line 315)
    kwargs_415665 = {}
    # Getting the type of 'abs' (line 315)
    abs_415658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 37), 'abs', False)
    # Calling abs(args, kwargs) (line 315)
    abs_call_result_415666 = invoke(stypy.reporting.localization.Localization(__file__, 315, 37), abs_415658, *[subscript_call_result_415664], **kwargs_415665)
    
    # Applying the binary operator '*' (line 315)
    result_mul_415667 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 29), '*', float_415657, abs_call_result_415666)
    
    # Applying the binary operator '<' (line 315)
    result_lt_415668 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 15), '<', abs_call_result_415656, result_mul_415667)
    
    # Testing the type of an if condition (line 315)
    if_condition_415669 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 315, 12), result_lt_415668)
    # Assigning a type to the variable 'if_condition_415669' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'if_condition_415669', if_condition_415669)
    # SSA begins for if statement (line 315)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 315)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 318):
    
    # Assigning a Call to a Name (line 318):
    
    # Call to scal(...): (line 318)
    # Processing the call arguments (line 318)
    float_415671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 21), 'float')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 318)
    tuple_415672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 318)
    # Adding element type (line 318)
    # Getting the type of 'j' (line 318)
    j_415673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 27), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 27), tuple_415672, j_415673)
    # Adding element type (line 318)
    # Getting the type of 'j' (line 318)
    j_415674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 29), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 27), tuple_415672, j_415674)
    
    # Getting the type of 'R' (line 318)
    R_415675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 318)
    getitem___415676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 25), R_415675, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 318)
    subscript_call_result_415677 = invoke(stypy.reporting.localization.Localization(__file__, 318, 25), getitem___415676, tuple_415672)
    
    # Applying the binary operator 'div' (line 318)
    result_div_415678 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 21), 'div', float_415671, subscript_call_result_415677)
    
    # Getting the type of 'u' (line 318)
    u_415679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 33), 'u', False)
    # Processing the call keyword arguments (line 318)
    kwargs_415680 = {}
    # Getting the type of 'scal' (line 318)
    scal_415670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'scal', False)
    # Calling scal(args, kwargs) (line 318)
    scal_call_result_415681 = invoke(stypy.reporting.localization.Localization(__file__, 318, 16), scal_415670, *[result_div_415678, u_415679], **kwargs_415680)
    
    # Assigning a type to the variable 'u' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'u', scal_call_result_415681)
    
    # Call to append(...): (line 319)
    # Processing the call arguments (line 319)
    # Getting the type of 'u' (line 319)
    u_415684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 26), 'u', False)
    # Processing the call keyword arguments (line 319)
    kwargs_415685 = {}
    # Getting the type of 'new_us' (line 319)
    new_us_415682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'new_us', False)
    # Obtaining the member 'append' of a type (line 319)
    append_415683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 12), new_us_415682, 'append')
    # Calling append(args, kwargs) (line 319)
    append_call_result_415686 = invoke(stypy.reporting.localization.Localization(__file__, 319, 12), append_415683, *[u_415684], **kwargs_415685)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Subscript (line 322):
    
    # Assigning a Subscript to a Subscript (line 322):
    
    # Obtaining the type of the subscript
    int_415687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 40), 'int')
    slice_415688 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 322, 16), None, None, int_415687)
    
    # Call to list(...): (line 322)
    # Processing the call arguments (line 322)
    
    # Call to zip(...): (line 322)
    # Processing the call arguments (line 322)
    # Getting the type of 'cs' (line 322)
    cs_415691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 'cs', False)
    # Getting the type of 'new_us' (line 322)
    new_us_415692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 29), 'new_us', False)
    # Processing the call keyword arguments (line 322)
    kwargs_415693 = {}
    # Getting the type of 'zip' (line 322)
    zip_415690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 21), 'zip', False)
    # Calling zip(args, kwargs) (line 322)
    zip_call_result_415694 = invoke(stypy.reporting.localization.Localization(__file__, 322, 21), zip_415690, *[cs_415691, new_us_415692], **kwargs_415693)
    
    # Processing the call keyword arguments (line 322)
    kwargs_415695 = {}
    # Getting the type of 'list' (line 322)
    list_415689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 16), 'list', False)
    # Calling list(args, kwargs) (line 322)
    list_call_result_415696 = invoke(stypy.reporting.localization.Localization(__file__, 322, 16), list_415689, *[zip_call_result_415694], **kwargs_415695)
    
    # Obtaining the member '__getitem__' of a type (line 322)
    getitem___415697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 16), list_call_result_415696, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 322)
    subscript_call_result_415698 = invoke(stypy.reporting.localization.Localization(__file__, 322, 16), getitem___415697, slice_415688)
    
    # Getting the type of 'CU' (line 322)
    CU_415699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'CU')
    slice_415700 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 322, 8), None, None, None)
    # Storing an element on a container (line 322)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 8), CU_415699, (slice_415700, subscript_call_result_415698))
    # SSA join for if statement (line 285)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'CU' (line 324)
    CU_415701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 7), 'CU')
    # Testing the type of an if condition (line 324)
    if_condition_415702 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 4), CU_415701)
    # Assigning a type to the variable 'if_condition_415702' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'if_condition_415702', if_condition_415702)
    # SSA begins for if statement (line 324)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 325):
    
    # Assigning a Subscript to a Name (line 325):
    
    # Obtaining the type of the subscript
    int_415703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 8), 'int')
    
    # Call to get_blas_funcs(...): (line 325)
    # Processing the call arguments (line 325)
    
    # Obtaining an instance of the builtin type 'list' (line 325)
    list_415705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 325)
    # Adding element type (line 325)
    str_415706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 36), 'str', 'axpy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 35), list_415705, str_415706)
    # Adding element type (line 325)
    str_415707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 44), 'str', 'dot')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 35), list_415705, str_415707)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 325)
    tuple_415708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 325)
    # Adding element type (line 325)
    # Getting the type of 'r' (line 325)
    r_415709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 53), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 53), tuple_415708, r_415709)
    
    # Processing the call keyword arguments (line 325)
    kwargs_415710 = {}
    # Getting the type of 'get_blas_funcs' (line 325)
    get_blas_funcs_415704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), 'get_blas_funcs', False)
    # Calling get_blas_funcs(args, kwargs) (line 325)
    get_blas_funcs_call_result_415711 = invoke(stypy.reporting.localization.Localization(__file__, 325, 20), get_blas_funcs_415704, *[list_415705, tuple_415708], **kwargs_415710)
    
    # Obtaining the member '__getitem__' of a type (line 325)
    getitem___415712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 8), get_blas_funcs_call_result_415711, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 325)
    subscript_call_result_415713 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), getitem___415712, int_415703)
    
    # Assigning a type to the variable 'tuple_var_assignment_414642' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'tuple_var_assignment_414642', subscript_call_result_415713)
    
    # Assigning a Subscript to a Name (line 325):
    
    # Obtaining the type of the subscript
    int_415714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 8), 'int')
    
    # Call to get_blas_funcs(...): (line 325)
    # Processing the call arguments (line 325)
    
    # Obtaining an instance of the builtin type 'list' (line 325)
    list_415716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 325)
    # Adding element type (line 325)
    str_415717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 36), 'str', 'axpy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 35), list_415716, str_415717)
    # Adding element type (line 325)
    str_415718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 44), 'str', 'dot')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 35), list_415716, str_415718)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 325)
    tuple_415719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 325)
    # Adding element type (line 325)
    # Getting the type of 'r' (line 325)
    r_415720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 53), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 53), tuple_415719, r_415720)
    
    # Processing the call keyword arguments (line 325)
    kwargs_415721 = {}
    # Getting the type of 'get_blas_funcs' (line 325)
    get_blas_funcs_415715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), 'get_blas_funcs', False)
    # Calling get_blas_funcs(args, kwargs) (line 325)
    get_blas_funcs_call_result_415722 = invoke(stypy.reporting.localization.Localization(__file__, 325, 20), get_blas_funcs_415715, *[list_415716, tuple_415719], **kwargs_415721)
    
    # Obtaining the member '__getitem__' of a type (line 325)
    getitem___415723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 8), get_blas_funcs_call_result_415722, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 325)
    subscript_call_result_415724 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), getitem___415723, int_415714)
    
    # Assigning a type to the variable 'tuple_var_assignment_414643' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'tuple_var_assignment_414643', subscript_call_result_415724)
    
    # Assigning a Name to a Name (line 325):
    # Getting the type of 'tuple_var_assignment_414642' (line 325)
    tuple_var_assignment_414642_415725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'tuple_var_assignment_414642')
    # Assigning a type to the variable 'axpy' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'axpy', tuple_var_assignment_414642_415725)
    
    # Assigning a Name to a Name (line 325):
    # Getting the type of 'tuple_var_assignment_414643' (line 325)
    tuple_var_assignment_414643_415726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'tuple_var_assignment_414643')
    # Assigning a type to the variable 'dot' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 14), 'dot', tuple_var_assignment_414643_415726)
    
    # Getting the type of 'CU' (line 335)
    CU_415727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'CU')
    # Testing the type of a for loop iterable (line 335)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 335, 8), CU_415727)
    # Getting the type of the for loop variable (line 335)
    for_loop_var_415728 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 335, 8), CU_415727)
    # Assigning a type to the variable 'c' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 8), for_loop_var_415728))
    # Assigning a type to the variable 'u' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'u', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 8), for_loop_var_415728))
    # SSA begins for a for statement (line 335)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 336):
    
    # Assigning a Call to a Name (line 336):
    
    # Call to dot(...): (line 336)
    # Processing the call arguments (line 336)
    # Getting the type of 'c' (line 336)
    c_415730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 21), 'c', False)
    # Getting the type of 'r' (line 336)
    r_415731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 24), 'r', False)
    # Processing the call keyword arguments (line 336)
    kwargs_415732 = {}
    # Getting the type of 'dot' (line 336)
    dot_415729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 17), 'dot', False)
    # Calling dot(args, kwargs) (line 336)
    dot_call_result_415733 = invoke(stypy.reporting.localization.Localization(__file__, 336, 17), dot_415729, *[c_415730, r_415731], **kwargs_415732)
    
    # Assigning a type to the variable 'yc' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'yc', dot_call_result_415733)
    
    # Assigning a Call to a Name (line 337):
    
    # Assigning a Call to a Name (line 337):
    
    # Call to axpy(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'u' (line 337)
    u_415735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 21), 'u', False)
    # Getting the type of 'x' (line 337)
    x_415736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 24), 'x', False)
    
    # Obtaining the type of the subscript
    int_415737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 35), 'int')
    # Getting the type of 'x' (line 337)
    x_415738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 27), 'x', False)
    # Obtaining the member 'shape' of a type (line 337)
    shape_415739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 27), x_415738, 'shape')
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___415740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 27), shape_415739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_415741 = invoke(stypy.reporting.localization.Localization(__file__, 337, 27), getitem___415740, int_415737)
    
    # Getting the type of 'yc' (line 337)
    yc_415742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 39), 'yc', False)
    # Processing the call keyword arguments (line 337)
    kwargs_415743 = {}
    # Getting the type of 'axpy' (line 337)
    axpy_415734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 16), 'axpy', False)
    # Calling axpy(args, kwargs) (line 337)
    axpy_call_result_415744 = invoke(stypy.reporting.localization.Localization(__file__, 337, 16), axpy_415734, *[u_415735, x_415736, subscript_call_result_415741, yc_415742], **kwargs_415743)
    
    # Assigning a type to the variable 'x' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'x', axpy_call_result_415744)
    
    # Assigning a Call to a Name (line 338):
    
    # Assigning a Call to a Name (line 338):
    
    # Call to axpy(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'c' (line 338)
    c_415746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 21), 'c', False)
    # Getting the type of 'r' (line 338)
    r_415747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 24), 'r', False)
    
    # Obtaining the type of the subscript
    int_415748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 35), 'int')
    # Getting the type of 'r' (line 338)
    r_415749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 27), 'r', False)
    # Obtaining the member 'shape' of a type (line 338)
    shape_415750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 27), r_415749, 'shape')
    # Obtaining the member '__getitem__' of a type (line 338)
    getitem___415751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 27), shape_415750, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 338)
    subscript_call_result_415752 = invoke(stypy.reporting.localization.Localization(__file__, 338, 27), getitem___415751, int_415748)
    
    
    # Getting the type of 'yc' (line 338)
    yc_415753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 40), 'yc', False)
    # Applying the 'usub' unary operator (line 338)
    result___neg___415754 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 39), 'usub', yc_415753)
    
    # Processing the call keyword arguments (line 338)
    kwargs_415755 = {}
    # Getting the type of 'axpy' (line 338)
    axpy_415745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'axpy', False)
    # Calling axpy(args, kwargs) (line 338)
    axpy_call_result_415756 = invoke(stypy.reporting.localization.Localization(__file__, 338, 16), axpy_415745, *[c_415746, r_415747, subscript_call_result_415752, result___neg___415754], **kwargs_415755)
    
    # Assigning a type to the variable 'r' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'r', axpy_call_result_415756)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 324)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to xrange(...): (line 341)
    # Processing the call arguments (line 341)
    # Getting the type of 'maxiter' (line 341)
    maxiter_415758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 26), 'maxiter', False)
    # Processing the call keyword arguments (line 341)
    kwargs_415759 = {}
    # Getting the type of 'xrange' (line 341)
    xrange_415757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 19), 'xrange', False)
    # Calling xrange(args, kwargs) (line 341)
    xrange_call_result_415760 = invoke(stypy.reporting.localization.Localization(__file__, 341, 19), xrange_415757, *[maxiter_415758], **kwargs_415759)
    
    # Testing the type of a for loop iterable (line 341)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 341, 4), xrange_call_result_415760)
    # Getting the type of the for loop variable (line 341)
    for_loop_var_415761 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 341, 4), xrange_call_result_415760)
    # Assigning a type to the variable 'j_outer' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'j_outer', for_loop_var_415761)
    # SSA begins for a for statement (line 341)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Type idiom detected: calculating its left and rigth part (line 343)
    # Getting the type of 'callback' (line 343)
    callback_415762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'callback')
    # Getting the type of 'None' (line 343)
    None_415763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 27), 'None')
    
    (may_be_415764, more_types_in_union_415765) = may_not_be_none(callback_415762, None_415763)

    if may_be_415764:

        if more_types_in_union_415765:
            # Runtime conditional SSA (line 343)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to callback(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'x' (line 344)
        x_415767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 21), 'x', False)
        # Processing the call keyword arguments (line 344)
        kwargs_415768 = {}
        # Getting the type of 'callback' (line 344)
        callback_415766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'callback', False)
        # Calling callback(args, kwargs) (line 344)
        callback_call_result_415769 = invoke(stypy.reporting.localization.Localization(__file__, 344, 12), callback_415766, *[x_415767], **kwargs_415768)
        

        if more_types_in_union_415765:
            # SSA join for if statement (line 343)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 346):
    
    # Assigning a Call to a Name (line 346):
    
    # Call to nrm2(...): (line 346)
    # Processing the call arguments (line 346)
    # Getting the type of 'r' (line 346)
    r_415771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 20), 'r', False)
    # Processing the call keyword arguments (line 346)
    kwargs_415772 = {}
    # Getting the type of 'nrm2' (line 346)
    nrm2_415770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 15), 'nrm2', False)
    # Calling nrm2(args, kwargs) (line 346)
    nrm2_call_result_415773 = invoke(stypy.reporting.localization.Localization(__file__, 346, 15), nrm2_415770, *[r_415771], **kwargs_415772)
    
    # Assigning a type to the variable 'beta' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'beta', nrm2_call_result_415773)
    
    
    # Getting the type of 'beta' (line 349)
    beta_415774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 11), 'beta')
    
    # Call to max(...): (line 349)
    # Processing the call arguments (line 349)
    # Getting the type of 'tol' (line 349)
    tol_415776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 23), 'tol', False)
    # Getting the type of 'tol' (line 349)
    tol_415777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 28), 'tol', False)
    # Getting the type of 'b_norm' (line 349)
    b_norm_415778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 34), 'b_norm', False)
    # Applying the binary operator '*' (line 349)
    result_mul_415779 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 28), '*', tol_415777, b_norm_415778)
    
    # Processing the call keyword arguments (line 349)
    kwargs_415780 = {}
    # Getting the type of 'max' (line 349)
    max_415775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 19), 'max', False)
    # Calling max(args, kwargs) (line 349)
    max_call_result_415781 = invoke(stypy.reporting.localization.Localization(__file__, 349, 19), max_415775, *[tol_415776, result_mul_415779], **kwargs_415780)
    
    # Applying the binary operator '<=' (line 349)
    result_le_415782 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 11), '<=', beta_415774, max_call_result_415781)
    
    # Testing the type of an if condition (line 349)
    if_condition_415783 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 349, 8), result_le_415782)
    # Assigning a type to the variable 'if_condition_415783' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'if_condition_415783', if_condition_415783)
    # SSA begins for if statement (line 349)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 350):
    
    # Assigning a Num to a Name (line 350):
    int_415784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 22), 'int')
    # Assigning a type to the variable 'j_outer' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'j_outer', int_415784)
    # SSA join for if statement (line 349)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 353):
    
    # Assigning a BinOp to a Name (line 353):
    # Getting the type of 'm' (line 353)
    m_415785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 13), 'm')
    
    # Call to max(...): (line 353)
    # Processing the call arguments (line 353)
    # Getting the type of 'k' (line 353)
    k_415787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 21), 'k', False)
    
    # Call to len(...): (line 353)
    # Processing the call arguments (line 353)
    # Getting the type of 'CU' (line 353)
    CU_415789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 29), 'CU', False)
    # Processing the call keyword arguments (line 353)
    kwargs_415790 = {}
    # Getting the type of 'len' (line 353)
    len_415788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 25), 'len', False)
    # Calling len(args, kwargs) (line 353)
    len_call_result_415791 = invoke(stypy.reporting.localization.Localization(__file__, 353, 25), len_415788, *[CU_415789], **kwargs_415790)
    
    # Applying the binary operator '-' (line 353)
    result_sub_415792 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 21), '-', k_415787, len_call_result_415791)
    
    int_415793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 34), 'int')
    # Processing the call keyword arguments (line 353)
    kwargs_415794 = {}
    # Getting the type of 'max' (line 353)
    max_415786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 17), 'max', False)
    # Calling max(args, kwargs) (line 353)
    max_call_result_415795 = invoke(stypy.reporting.localization.Localization(__file__, 353, 17), max_415786, *[result_sub_415792, int_415793], **kwargs_415794)
    
    # Applying the binary operator '+' (line 353)
    result_add_415796 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 13), '+', m_415785, max_call_result_415795)
    
    # Assigning a type to the variable 'ml' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'ml', result_add_415796)
    
    # Assigning a ListComp to a Name (line 355):
    
    # Assigning a ListComp to a Name (line 355):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'CU' (line 355)
    CU_415798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 28), 'CU')
    comprehension_415799 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 14), CU_415798)
    # Assigning a type to the variable 'c' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 14), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 14), comprehension_415799))
    # Assigning a type to the variable 'u' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 14), 'u', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 14), comprehension_415799))
    # Getting the type of 'c' (line 355)
    c_415797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 14), 'c')
    list_415800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 14), list_415800, c_415797)
    # Assigning a type to the variable 'cs' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'cs', list_415800)
    
    
    # SSA begins for try-except statement (line 357)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 358):
    
    # Assigning a Subscript to a Name (line 358):
    
    # Obtaining the type of the subscript
    int_415801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 12), 'int')
    
    # Call to _fgmres(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'matvec' (line 358)
    matvec_415803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 41), 'matvec', False)
    # Getting the type of 'r' (line 359)
    r_415804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 40), 'r', False)
    # Getting the type of 'beta' (line 359)
    beta_415805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 42), 'beta', False)
    # Applying the binary operator 'div' (line 359)
    result_div_415806 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 40), 'div', r_415804, beta_415805)
    
    # Getting the type of 'ml' (line 360)
    ml_415807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 40), 'ml', False)
    # Processing the call keyword arguments (line 358)
    # Getting the type of 'psolve' (line 361)
    psolve_415808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 48), 'psolve', False)
    keyword_415809 = psolve_415808
    # Getting the type of 'tol' (line 362)
    tol_415810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 45), 'tol', False)
    # Getting the type of 'b_norm' (line 362)
    b_norm_415811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 49), 'b_norm', False)
    # Applying the binary operator '*' (line 362)
    result_mul_415812 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 45), '*', tol_415810, b_norm_415811)
    
    # Getting the type of 'beta' (line 362)
    beta_415813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 56), 'beta', False)
    # Applying the binary operator 'div' (line 362)
    result_div_415814 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 55), 'div', result_mul_415812, beta_415813)
    
    keyword_415815 = result_div_415814
    # Getting the type of 'cs' (line 363)
    cs_415816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 43), 'cs', False)
    keyword_415817 = cs_415816
    kwargs_415818 = {'rpsolve': keyword_415809, 'cs': keyword_415817, 'atol': keyword_415815}
    # Getting the type of '_fgmres' (line 358)
    _fgmres_415802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 33), '_fgmres', False)
    # Calling _fgmres(args, kwargs) (line 358)
    _fgmres_call_result_415819 = invoke(stypy.reporting.localization.Localization(__file__, 358, 33), _fgmres_415802, *[matvec_415803, result_div_415806, ml_415807], **kwargs_415818)
    
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___415820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), _fgmres_call_result_415819, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_415821 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), getitem___415820, int_415801)
    
    # Assigning a type to the variable 'tuple_var_assignment_414644' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_414644', subscript_call_result_415821)
    
    # Assigning a Subscript to a Name (line 358):
    
    # Obtaining the type of the subscript
    int_415822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 12), 'int')
    
    # Call to _fgmres(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'matvec' (line 358)
    matvec_415824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 41), 'matvec', False)
    # Getting the type of 'r' (line 359)
    r_415825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 40), 'r', False)
    # Getting the type of 'beta' (line 359)
    beta_415826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 42), 'beta', False)
    # Applying the binary operator 'div' (line 359)
    result_div_415827 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 40), 'div', r_415825, beta_415826)
    
    # Getting the type of 'ml' (line 360)
    ml_415828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 40), 'ml', False)
    # Processing the call keyword arguments (line 358)
    # Getting the type of 'psolve' (line 361)
    psolve_415829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 48), 'psolve', False)
    keyword_415830 = psolve_415829
    # Getting the type of 'tol' (line 362)
    tol_415831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 45), 'tol', False)
    # Getting the type of 'b_norm' (line 362)
    b_norm_415832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 49), 'b_norm', False)
    # Applying the binary operator '*' (line 362)
    result_mul_415833 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 45), '*', tol_415831, b_norm_415832)
    
    # Getting the type of 'beta' (line 362)
    beta_415834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 56), 'beta', False)
    # Applying the binary operator 'div' (line 362)
    result_div_415835 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 55), 'div', result_mul_415833, beta_415834)
    
    keyword_415836 = result_div_415835
    # Getting the type of 'cs' (line 363)
    cs_415837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 43), 'cs', False)
    keyword_415838 = cs_415837
    kwargs_415839 = {'rpsolve': keyword_415830, 'cs': keyword_415838, 'atol': keyword_415836}
    # Getting the type of '_fgmres' (line 358)
    _fgmres_415823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 33), '_fgmres', False)
    # Calling _fgmres(args, kwargs) (line 358)
    _fgmres_call_result_415840 = invoke(stypy.reporting.localization.Localization(__file__, 358, 33), _fgmres_415823, *[matvec_415824, result_div_415827, ml_415828], **kwargs_415839)
    
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___415841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), _fgmres_call_result_415840, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_415842 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), getitem___415841, int_415822)
    
    # Assigning a type to the variable 'tuple_var_assignment_414645' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_414645', subscript_call_result_415842)
    
    # Assigning a Subscript to a Name (line 358):
    
    # Obtaining the type of the subscript
    int_415843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 12), 'int')
    
    # Call to _fgmres(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'matvec' (line 358)
    matvec_415845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 41), 'matvec', False)
    # Getting the type of 'r' (line 359)
    r_415846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 40), 'r', False)
    # Getting the type of 'beta' (line 359)
    beta_415847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 42), 'beta', False)
    # Applying the binary operator 'div' (line 359)
    result_div_415848 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 40), 'div', r_415846, beta_415847)
    
    # Getting the type of 'ml' (line 360)
    ml_415849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 40), 'ml', False)
    # Processing the call keyword arguments (line 358)
    # Getting the type of 'psolve' (line 361)
    psolve_415850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 48), 'psolve', False)
    keyword_415851 = psolve_415850
    # Getting the type of 'tol' (line 362)
    tol_415852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 45), 'tol', False)
    # Getting the type of 'b_norm' (line 362)
    b_norm_415853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 49), 'b_norm', False)
    # Applying the binary operator '*' (line 362)
    result_mul_415854 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 45), '*', tol_415852, b_norm_415853)
    
    # Getting the type of 'beta' (line 362)
    beta_415855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 56), 'beta', False)
    # Applying the binary operator 'div' (line 362)
    result_div_415856 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 55), 'div', result_mul_415854, beta_415855)
    
    keyword_415857 = result_div_415856
    # Getting the type of 'cs' (line 363)
    cs_415858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 43), 'cs', False)
    keyword_415859 = cs_415858
    kwargs_415860 = {'rpsolve': keyword_415851, 'cs': keyword_415859, 'atol': keyword_415857}
    # Getting the type of '_fgmres' (line 358)
    _fgmres_415844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 33), '_fgmres', False)
    # Calling _fgmres(args, kwargs) (line 358)
    _fgmres_call_result_415861 = invoke(stypy.reporting.localization.Localization(__file__, 358, 33), _fgmres_415844, *[matvec_415845, result_div_415848, ml_415849], **kwargs_415860)
    
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___415862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), _fgmres_call_result_415861, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_415863 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), getitem___415862, int_415843)
    
    # Assigning a type to the variable 'tuple_var_assignment_414646' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_414646', subscript_call_result_415863)
    
    # Assigning a Subscript to a Name (line 358):
    
    # Obtaining the type of the subscript
    int_415864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 12), 'int')
    
    # Call to _fgmres(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'matvec' (line 358)
    matvec_415866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 41), 'matvec', False)
    # Getting the type of 'r' (line 359)
    r_415867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 40), 'r', False)
    # Getting the type of 'beta' (line 359)
    beta_415868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 42), 'beta', False)
    # Applying the binary operator 'div' (line 359)
    result_div_415869 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 40), 'div', r_415867, beta_415868)
    
    # Getting the type of 'ml' (line 360)
    ml_415870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 40), 'ml', False)
    # Processing the call keyword arguments (line 358)
    # Getting the type of 'psolve' (line 361)
    psolve_415871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 48), 'psolve', False)
    keyword_415872 = psolve_415871
    # Getting the type of 'tol' (line 362)
    tol_415873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 45), 'tol', False)
    # Getting the type of 'b_norm' (line 362)
    b_norm_415874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 49), 'b_norm', False)
    # Applying the binary operator '*' (line 362)
    result_mul_415875 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 45), '*', tol_415873, b_norm_415874)
    
    # Getting the type of 'beta' (line 362)
    beta_415876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 56), 'beta', False)
    # Applying the binary operator 'div' (line 362)
    result_div_415877 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 55), 'div', result_mul_415875, beta_415876)
    
    keyword_415878 = result_div_415877
    # Getting the type of 'cs' (line 363)
    cs_415879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 43), 'cs', False)
    keyword_415880 = cs_415879
    kwargs_415881 = {'rpsolve': keyword_415872, 'cs': keyword_415880, 'atol': keyword_415878}
    # Getting the type of '_fgmres' (line 358)
    _fgmres_415865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 33), '_fgmres', False)
    # Calling _fgmres(args, kwargs) (line 358)
    _fgmres_call_result_415882 = invoke(stypy.reporting.localization.Localization(__file__, 358, 33), _fgmres_415865, *[matvec_415866, result_div_415869, ml_415870], **kwargs_415881)
    
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___415883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), _fgmres_call_result_415882, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_415884 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), getitem___415883, int_415864)
    
    # Assigning a type to the variable 'tuple_var_assignment_414647' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_414647', subscript_call_result_415884)
    
    # Assigning a Subscript to a Name (line 358):
    
    # Obtaining the type of the subscript
    int_415885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 12), 'int')
    
    # Call to _fgmres(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'matvec' (line 358)
    matvec_415887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 41), 'matvec', False)
    # Getting the type of 'r' (line 359)
    r_415888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 40), 'r', False)
    # Getting the type of 'beta' (line 359)
    beta_415889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 42), 'beta', False)
    # Applying the binary operator 'div' (line 359)
    result_div_415890 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 40), 'div', r_415888, beta_415889)
    
    # Getting the type of 'ml' (line 360)
    ml_415891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 40), 'ml', False)
    # Processing the call keyword arguments (line 358)
    # Getting the type of 'psolve' (line 361)
    psolve_415892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 48), 'psolve', False)
    keyword_415893 = psolve_415892
    # Getting the type of 'tol' (line 362)
    tol_415894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 45), 'tol', False)
    # Getting the type of 'b_norm' (line 362)
    b_norm_415895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 49), 'b_norm', False)
    # Applying the binary operator '*' (line 362)
    result_mul_415896 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 45), '*', tol_415894, b_norm_415895)
    
    # Getting the type of 'beta' (line 362)
    beta_415897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 56), 'beta', False)
    # Applying the binary operator 'div' (line 362)
    result_div_415898 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 55), 'div', result_mul_415896, beta_415897)
    
    keyword_415899 = result_div_415898
    # Getting the type of 'cs' (line 363)
    cs_415900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 43), 'cs', False)
    keyword_415901 = cs_415900
    kwargs_415902 = {'rpsolve': keyword_415893, 'cs': keyword_415901, 'atol': keyword_415899}
    # Getting the type of '_fgmres' (line 358)
    _fgmres_415886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 33), '_fgmres', False)
    # Calling _fgmres(args, kwargs) (line 358)
    _fgmres_call_result_415903 = invoke(stypy.reporting.localization.Localization(__file__, 358, 33), _fgmres_415886, *[matvec_415887, result_div_415890, ml_415891], **kwargs_415902)
    
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___415904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), _fgmres_call_result_415903, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_415905 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), getitem___415904, int_415885)
    
    # Assigning a type to the variable 'tuple_var_assignment_414648' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_414648', subscript_call_result_415905)
    
    # Assigning a Subscript to a Name (line 358):
    
    # Obtaining the type of the subscript
    int_415906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 12), 'int')
    
    # Call to _fgmres(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'matvec' (line 358)
    matvec_415908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 41), 'matvec', False)
    # Getting the type of 'r' (line 359)
    r_415909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 40), 'r', False)
    # Getting the type of 'beta' (line 359)
    beta_415910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 42), 'beta', False)
    # Applying the binary operator 'div' (line 359)
    result_div_415911 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 40), 'div', r_415909, beta_415910)
    
    # Getting the type of 'ml' (line 360)
    ml_415912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 40), 'ml', False)
    # Processing the call keyword arguments (line 358)
    # Getting the type of 'psolve' (line 361)
    psolve_415913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 48), 'psolve', False)
    keyword_415914 = psolve_415913
    # Getting the type of 'tol' (line 362)
    tol_415915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 45), 'tol', False)
    # Getting the type of 'b_norm' (line 362)
    b_norm_415916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 49), 'b_norm', False)
    # Applying the binary operator '*' (line 362)
    result_mul_415917 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 45), '*', tol_415915, b_norm_415916)
    
    # Getting the type of 'beta' (line 362)
    beta_415918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 56), 'beta', False)
    # Applying the binary operator 'div' (line 362)
    result_div_415919 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 55), 'div', result_mul_415917, beta_415918)
    
    keyword_415920 = result_div_415919
    # Getting the type of 'cs' (line 363)
    cs_415921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 43), 'cs', False)
    keyword_415922 = cs_415921
    kwargs_415923 = {'rpsolve': keyword_415914, 'cs': keyword_415922, 'atol': keyword_415920}
    # Getting the type of '_fgmres' (line 358)
    _fgmres_415907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 33), '_fgmres', False)
    # Calling _fgmres(args, kwargs) (line 358)
    _fgmres_call_result_415924 = invoke(stypy.reporting.localization.Localization(__file__, 358, 33), _fgmres_415907, *[matvec_415908, result_div_415911, ml_415912], **kwargs_415923)
    
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___415925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), _fgmres_call_result_415924, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_415926 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), getitem___415925, int_415906)
    
    # Assigning a type to the variable 'tuple_var_assignment_414649' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_414649', subscript_call_result_415926)
    
    # Assigning a Name to a Name (line 358):
    # Getting the type of 'tuple_var_assignment_414644' (line 358)
    tuple_var_assignment_414644_415927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_414644')
    # Assigning a type to the variable 'Q' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'Q', tuple_var_assignment_414644_415927)
    
    # Assigning a Name to a Name (line 358):
    # Getting the type of 'tuple_var_assignment_414645' (line 358)
    tuple_var_assignment_414645_415928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_414645')
    # Assigning a type to the variable 'R' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 15), 'R', tuple_var_assignment_414645_415928)
    
    # Assigning a Name to a Name (line 358):
    # Getting the type of 'tuple_var_assignment_414646' (line 358)
    tuple_var_assignment_414646_415929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_414646')
    # Assigning a type to the variable 'B' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 18), 'B', tuple_var_assignment_414646_415929)
    
    # Assigning a Name to a Name (line 358):
    # Getting the type of 'tuple_var_assignment_414647' (line 358)
    tuple_var_assignment_414647_415930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_414647')
    # Assigning a type to the variable 'vs' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 21), 'vs', tuple_var_assignment_414647_415930)
    
    # Assigning a Name to a Name (line 358):
    # Getting the type of 'tuple_var_assignment_414648' (line 358)
    tuple_var_assignment_414648_415931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_414648')
    # Assigning a type to the variable 'zs' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 25), 'zs', tuple_var_assignment_414648_415931)
    
    # Assigning a Name to a Name (line 358):
    # Getting the type of 'tuple_var_assignment_414649' (line 358)
    tuple_var_assignment_414649_415932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_414649')
    # Assigning a type to the variable 'y' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 29), 'y', tuple_var_assignment_414649_415932)
    
    # Getting the type of 'y' (line 364)
    y_415933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'y')
    # Getting the type of 'beta' (line 364)
    beta_415934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 17), 'beta')
    # Applying the binary operator '*=' (line 364)
    result_imul_415935 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 12), '*=', y_415933, beta_415934)
    # Assigning a type to the variable 'y' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'y', result_imul_415935)
    
    # SSA branch for the except part of a try statement (line 357)
    # SSA branch for the except 'LinAlgError' branch of a try statement (line 357)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 357)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 390):
    
    # Assigning a BinOp to a Name (line 390):
    
    # Obtaining the type of the subscript
    int_415936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 16), 'int')
    # Getting the type of 'zs' (line 390)
    zs_415937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 13), 'zs')
    # Obtaining the member '__getitem__' of a type (line 390)
    getitem___415938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 13), zs_415937, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 390)
    subscript_call_result_415939 = invoke(stypy.reporting.localization.Localization(__file__, 390, 13), getitem___415938, int_415936)
    
    
    # Obtaining the type of the subscript
    int_415940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 21), 'int')
    # Getting the type of 'y' (line 390)
    y_415941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'y')
    # Obtaining the member '__getitem__' of a type (line 390)
    getitem___415942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 19), y_415941, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 390)
    subscript_call_result_415943 = invoke(stypy.reporting.localization.Localization(__file__, 390, 19), getitem___415942, int_415940)
    
    # Applying the binary operator '*' (line 390)
    result_mul_415944 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 13), '*', subscript_call_result_415939, subscript_call_result_415943)
    
    # Assigning a type to the variable 'ux' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'ux', result_mul_415944)
    
    
    # Call to zip(...): (line 391)
    # Processing the call arguments (line 391)
    
    # Obtaining the type of the subscript
    int_415946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 28), 'int')
    slice_415947 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 391, 25), int_415946, None, None)
    # Getting the type of 'zs' (line 391)
    zs_415948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 25), 'zs', False)
    # Obtaining the member '__getitem__' of a type (line 391)
    getitem___415949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 25), zs_415948, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 391)
    subscript_call_result_415950 = invoke(stypy.reporting.localization.Localization(__file__, 391, 25), getitem___415949, slice_415947)
    
    
    # Obtaining the type of the subscript
    int_415951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 35), 'int')
    slice_415952 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 391, 33), int_415951, None, None)
    # Getting the type of 'y' (line 391)
    y_415953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 33), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 391)
    getitem___415954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 33), y_415953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 391)
    subscript_call_result_415955 = invoke(stypy.reporting.localization.Localization(__file__, 391, 33), getitem___415954, slice_415952)
    
    # Processing the call keyword arguments (line 391)
    kwargs_415956 = {}
    # Getting the type of 'zip' (line 391)
    zip_415945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 21), 'zip', False)
    # Calling zip(args, kwargs) (line 391)
    zip_call_result_415957 = invoke(stypy.reporting.localization.Localization(__file__, 391, 21), zip_415945, *[subscript_call_result_415950, subscript_call_result_415955], **kwargs_415956)
    
    # Testing the type of a for loop iterable (line 391)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 391, 8), zip_call_result_415957)
    # Getting the type of the for loop variable (line 391)
    for_loop_var_415958 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 391, 8), zip_call_result_415957)
    # Assigning a type to the variable 'z' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'z', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 8), for_loop_var_415958))
    # Assigning a type to the variable 'yc' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'yc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 8), for_loop_var_415958))
    # SSA begins for a for statement (line 391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 392):
    
    # Assigning a Call to a Name (line 392):
    
    # Call to axpy(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'z' (line 392)
    z_415960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 22), 'z', False)
    # Getting the type of 'ux' (line 392)
    ux_415961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 25), 'ux', False)
    
    # Obtaining the type of the subscript
    int_415962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 38), 'int')
    # Getting the type of 'ux' (line 392)
    ux_415963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 29), 'ux', False)
    # Obtaining the member 'shape' of a type (line 392)
    shape_415964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 29), ux_415963, 'shape')
    # Obtaining the member '__getitem__' of a type (line 392)
    getitem___415965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 29), shape_415964, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 392)
    subscript_call_result_415966 = invoke(stypy.reporting.localization.Localization(__file__, 392, 29), getitem___415965, int_415962)
    
    # Getting the type of 'yc' (line 392)
    yc_415967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 42), 'yc', False)
    # Processing the call keyword arguments (line 392)
    kwargs_415968 = {}
    # Getting the type of 'axpy' (line 392)
    axpy_415959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 17), 'axpy', False)
    # Calling axpy(args, kwargs) (line 392)
    axpy_call_result_415969 = invoke(stypy.reporting.localization.Localization(__file__, 392, 17), axpy_415959, *[z_415960, ux_415961, subscript_call_result_415966, yc_415967], **kwargs_415968)
    
    # Assigning a type to the variable 'ux' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'ux', axpy_call_result_415969)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 393):
    
    # Assigning a Call to a Name (line 393):
    
    # Call to dot(...): (line 393)
    # Processing the call arguments (line 393)
    # Getting the type of 'y' (line 393)
    y_415972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 19), 'y', False)
    # Processing the call keyword arguments (line 393)
    kwargs_415973 = {}
    # Getting the type of 'B' (line 393)
    B_415970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 13), 'B', False)
    # Obtaining the member 'dot' of a type (line 393)
    dot_415971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 13), B_415970, 'dot')
    # Calling dot(args, kwargs) (line 393)
    dot_call_result_415974 = invoke(stypy.reporting.localization.Localization(__file__, 393, 13), dot_415971, *[y_415972], **kwargs_415973)
    
    # Assigning a type to the variable 'by' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'by', dot_call_result_415974)
    
    
    # Call to zip(...): (line 394)
    # Processing the call arguments (line 394)
    # Getting the type of 'CU' (line 394)
    CU_415976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 27), 'CU', False)
    # Getting the type of 'by' (line 394)
    by_415977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 31), 'by', False)
    # Processing the call keyword arguments (line 394)
    kwargs_415978 = {}
    # Getting the type of 'zip' (line 394)
    zip_415975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 23), 'zip', False)
    # Calling zip(args, kwargs) (line 394)
    zip_call_result_415979 = invoke(stypy.reporting.localization.Localization(__file__, 394, 23), zip_415975, *[CU_415976, by_415977], **kwargs_415978)
    
    # Testing the type of a for loop iterable (line 394)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 394, 8), zip_call_result_415979)
    # Getting the type of the for loop variable (line 394)
    for_loop_var_415980 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 394, 8), zip_call_result_415979)
    # Assigning a type to the variable 'cu' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'cu', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 8), for_loop_var_415980))
    # Assigning a type to the variable 'byc' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'byc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 8), for_loop_var_415980))
    # SSA begins for a for statement (line 394)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Tuple (line 395):
    
    # Assigning a Subscript to a Name (line 395):
    
    # Obtaining the type of the subscript
    int_415981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 12), 'int')
    # Getting the type of 'cu' (line 395)
    cu_415982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 19), 'cu')
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___415983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), cu_415982, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 395)
    subscript_call_result_415984 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), getitem___415983, int_415981)
    
    # Assigning a type to the variable 'tuple_var_assignment_414650' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'tuple_var_assignment_414650', subscript_call_result_415984)
    
    # Assigning a Subscript to a Name (line 395):
    
    # Obtaining the type of the subscript
    int_415985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 12), 'int')
    # Getting the type of 'cu' (line 395)
    cu_415986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 19), 'cu')
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___415987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), cu_415986, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 395)
    subscript_call_result_415988 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), getitem___415987, int_415985)
    
    # Assigning a type to the variable 'tuple_var_assignment_414651' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'tuple_var_assignment_414651', subscript_call_result_415988)
    
    # Assigning a Name to a Name (line 395):
    # Getting the type of 'tuple_var_assignment_414650' (line 395)
    tuple_var_assignment_414650_415989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'tuple_var_assignment_414650')
    # Assigning a type to the variable 'c' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'c', tuple_var_assignment_414650_415989)
    
    # Assigning a Name to a Name (line 395):
    # Getting the type of 'tuple_var_assignment_414651' (line 395)
    tuple_var_assignment_414651_415990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'tuple_var_assignment_414651')
    # Assigning a type to the variable 'u' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 15), 'u', tuple_var_assignment_414651_415990)
    
    # Assigning a Call to a Name (line 396):
    
    # Assigning a Call to a Name (line 396):
    
    # Call to axpy(...): (line 396)
    # Processing the call arguments (line 396)
    # Getting the type of 'u' (line 396)
    u_415992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 22), 'u', False)
    # Getting the type of 'ux' (line 396)
    ux_415993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 25), 'ux', False)
    
    # Obtaining the type of the subscript
    int_415994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 38), 'int')
    # Getting the type of 'ux' (line 396)
    ux_415995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 29), 'ux', False)
    # Obtaining the member 'shape' of a type (line 396)
    shape_415996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 29), ux_415995, 'shape')
    # Obtaining the member '__getitem__' of a type (line 396)
    getitem___415997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 29), shape_415996, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 396)
    subscript_call_result_415998 = invoke(stypy.reporting.localization.Localization(__file__, 396, 29), getitem___415997, int_415994)
    
    
    # Getting the type of 'byc' (line 396)
    byc_415999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 43), 'byc', False)
    # Applying the 'usub' unary operator (line 396)
    result___neg___416000 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 42), 'usub', byc_415999)
    
    # Processing the call keyword arguments (line 396)
    kwargs_416001 = {}
    # Getting the type of 'axpy' (line 396)
    axpy_415991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 17), 'axpy', False)
    # Calling axpy(args, kwargs) (line 396)
    axpy_call_result_416002 = invoke(stypy.reporting.localization.Localization(__file__, 396, 17), axpy_415991, *[u_415992, ux_415993, subscript_call_result_415998, result___neg___416000], **kwargs_416001)
    
    # Assigning a type to the variable 'ux' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'ux', axpy_call_result_416002)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 399):
    
    # Assigning a Call to a Name (line 399):
    
    # Call to dot(...): (line 399)
    # Processing the call arguments (line 399)
    
    # Call to dot(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'y' (line 399)
    y_416007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 25), 'y', False)
    # Processing the call keyword arguments (line 399)
    kwargs_416008 = {}
    # Getting the type of 'R' (line 399)
    R_416005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 19), 'R', False)
    # Obtaining the member 'dot' of a type (line 399)
    dot_416006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 19), R_416005, 'dot')
    # Calling dot(args, kwargs) (line 399)
    dot_call_result_416009 = invoke(stypy.reporting.localization.Localization(__file__, 399, 19), dot_416006, *[y_416007], **kwargs_416008)
    
    # Processing the call keyword arguments (line 399)
    kwargs_416010 = {}
    # Getting the type of 'Q' (line 399)
    Q_416003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 13), 'Q', False)
    # Obtaining the member 'dot' of a type (line 399)
    dot_416004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 13), Q_416003, 'dot')
    # Calling dot(args, kwargs) (line 399)
    dot_call_result_416011 = invoke(stypy.reporting.localization.Localization(__file__, 399, 13), dot_416004, *[dot_call_result_416009], **kwargs_416010)
    
    # Assigning a type to the variable 'hy' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'hy', dot_call_result_416011)
    
    # Assigning a BinOp to a Name (line 400):
    
    # Assigning a BinOp to a Name (line 400):
    
    # Obtaining the type of the subscript
    int_416012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 16), 'int')
    # Getting the type of 'vs' (line 400)
    vs_416013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 13), 'vs')
    # Obtaining the member '__getitem__' of a type (line 400)
    getitem___416014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 13), vs_416013, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 400)
    subscript_call_result_416015 = invoke(stypy.reporting.localization.Localization(__file__, 400, 13), getitem___416014, int_416012)
    
    
    # Obtaining the type of the subscript
    int_416016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 24), 'int')
    # Getting the type of 'hy' (line 400)
    hy_416017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 21), 'hy')
    # Obtaining the member '__getitem__' of a type (line 400)
    getitem___416018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 21), hy_416017, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 400)
    subscript_call_result_416019 = invoke(stypy.reporting.localization.Localization(__file__, 400, 21), getitem___416018, int_416016)
    
    # Applying the binary operator '*' (line 400)
    result_mul_416020 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 13), '*', subscript_call_result_416015, subscript_call_result_416019)
    
    # Assigning a type to the variable 'cx' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'cx', result_mul_416020)
    
    
    # Call to zip(...): (line 401)
    # Processing the call arguments (line 401)
    
    # Obtaining the type of the subscript
    int_416022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 29), 'int')
    slice_416023 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 401, 26), int_416022, None, None)
    # Getting the type of 'vs' (line 401)
    vs_416024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 26), 'vs', False)
    # Obtaining the member '__getitem__' of a type (line 401)
    getitem___416025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 26), vs_416024, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 401)
    subscript_call_result_416026 = invoke(stypy.reporting.localization.Localization(__file__, 401, 26), getitem___416025, slice_416023)
    
    
    # Obtaining the type of the subscript
    int_416027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 37), 'int')
    slice_416028 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 401, 34), int_416027, None, None)
    # Getting the type of 'hy' (line 401)
    hy_416029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 34), 'hy', False)
    # Obtaining the member '__getitem__' of a type (line 401)
    getitem___416030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 34), hy_416029, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 401)
    subscript_call_result_416031 = invoke(stypy.reporting.localization.Localization(__file__, 401, 34), getitem___416030, slice_416028)
    
    # Processing the call keyword arguments (line 401)
    kwargs_416032 = {}
    # Getting the type of 'zip' (line 401)
    zip_416021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 22), 'zip', False)
    # Calling zip(args, kwargs) (line 401)
    zip_call_result_416033 = invoke(stypy.reporting.localization.Localization(__file__, 401, 22), zip_416021, *[subscript_call_result_416026, subscript_call_result_416031], **kwargs_416032)
    
    # Testing the type of a for loop iterable (line 401)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 401, 8), zip_call_result_416033)
    # Getting the type of the for loop variable (line 401)
    for_loop_var_416034 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 401, 8), zip_call_result_416033)
    # Assigning a type to the variable 'v' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 8), for_loop_var_416034))
    # Assigning a type to the variable 'hyc' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'hyc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 8), for_loop_var_416034))
    # SSA begins for a for statement (line 401)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 402):
    
    # Assigning a Call to a Name (line 402):
    
    # Call to axpy(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of 'v' (line 402)
    v_416036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 22), 'v', False)
    # Getting the type of 'cx' (line 402)
    cx_416037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 25), 'cx', False)
    
    # Obtaining the type of the subscript
    int_416038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 38), 'int')
    # Getting the type of 'cx' (line 402)
    cx_416039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 29), 'cx', False)
    # Obtaining the member 'shape' of a type (line 402)
    shape_416040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 29), cx_416039, 'shape')
    # Obtaining the member '__getitem__' of a type (line 402)
    getitem___416041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 29), shape_416040, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 402)
    subscript_call_result_416042 = invoke(stypy.reporting.localization.Localization(__file__, 402, 29), getitem___416041, int_416038)
    
    # Getting the type of 'hyc' (line 402)
    hyc_416043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 42), 'hyc', False)
    # Processing the call keyword arguments (line 402)
    kwargs_416044 = {}
    # Getting the type of 'axpy' (line 402)
    axpy_416035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 17), 'axpy', False)
    # Calling axpy(args, kwargs) (line 402)
    axpy_call_result_416045 = invoke(stypy.reporting.localization.Localization(__file__, 402, 17), axpy_416035, *[v_416036, cx_416037, subscript_call_result_416042, hyc_416043], **kwargs_416044)
    
    # Assigning a type to the variable 'cx' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'cx', axpy_call_result_416045)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 406)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a BinOp to a Name (line 407):
    
    # Assigning a BinOp to a Name (line 407):
    int_416046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 20), 'int')
    
    # Call to nrm2(...): (line 407)
    # Processing the call arguments (line 407)
    # Getting the type of 'cx' (line 407)
    cx_416048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 27), 'cx', False)
    # Processing the call keyword arguments (line 407)
    kwargs_416049 = {}
    # Getting the type of 'nrm2' (line 407)
    nrm2_416047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 22), 'nrm2', False)
    # Calling nrm2(args, kwargs) (line 407)
    nrm2_call_result_416050 = invoke(stypy.reporting.localization.Localization(__file__, 407, 22), nrm2_416047, *[cx_416048], **kwargs_416049)
    
    # Applying the binary operator 'div' (line 407)
    result_div_416051 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 20), 'div', int_416046, nrm2_call_result_416050)
    
    # Assigning a type to the variable 'alpha' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'alpha', result_div_416051)
    
    
    
    # Call to isfinite(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'alpha' (line 408)
    alpha_416054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 31), 'alpha', False)
    # Processing the call keyword arguments (line 408)
    kwargs_416055 = {}
    # Getting the type of 'np' (line 408)
    np_416052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 19), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 408)
    isfinite_416053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 19), np_416052, 'isfinite')
    # Calling isfinite(args, kwargs) (line 408)
    isfinite_call_result_416056 = invoke(stypy.reporting.localization.Localization(__file__, 408, 19), isfinite_416053, *[alpha_416054], **kwargs_416055)
    
    # Applying the 'not' unary operator (line 408)
    result_not__416057 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 15), 'not', isfinite_call_result_416056)
    
    # Testing the type of an if condition (line 408)
    if_condition_416058 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 408, 12), result_not__416057)
    # Assigning a type to the variable 'if_condition_416058' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'if_condition_416058', if_condition_416058)
    # SSA begins for if statement (line 408)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to FloatingPointError(...): (line 409)
    # Processing the call keyword arguments (line 409)
    kwargs_416060 = {}
    # Getting the type of 'FloatingPointError' (line 409)
    FloatingPointError_416059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 22), 'FloatingPointError', False)
    # Calling FloatingPointError(args, kwargs) (line 409)
    FloatingPointError_call_result_416061 = invoke(stypy.reporting.localization.Localization(__file__, 409, 22), FloatingPointError_416059, *[], **kwargs_416060)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 409, 16), FloatingPointError_call_result_416061, 'raise parameter', BaseException)
    # SSA join for if statement (line 408)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 406)
    # SSA branch for the except 'Tuple' branch of a try statement (line 406)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 406)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 414):
    
    # Assigning a Call to a Name (line 414):
    
    # Call to scal(...): (line 414)
    # Processing the call arguments (line 414)
    # Getting the type of 'alpha' (line 414)
    alpha_416063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 18), 'alpha', False)
    # Getting the type of 'cx' (line 414)
    cx_416064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 25), 'cx', False)
    # Processing the call keyword arguments (line 414)
    kwargs_416065 = {}
    # Getting the type of 'scal' (line 414)
    scal_416062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 13), 'scal', False)
    # Calling scal(args, kwargs) (line 414)
    scal_call_result_416066 = invoke(stypy.reporting.localization.Localization(__file__, 414, 13), scal_416062, *[alpha_416063, cx_416064], **kwargs_416065)
    
    # Assigning a type to the variable 'cx' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'cx', scal_call_result_416066)
    
    # Assigning a Call to a Name (line 415):
    
    # Assigning a Call to a Name (line 415):
    
    # Call to scal(...): (line 415)
    # Processing the call arguments (line 415)
    # Getting the type of 'alpha' (line 415)
    alpha_416068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 18), 'alpha', False)
    # Getting the type of 'ux' (line 415)
    ux_416069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 25), 'ux', False)
    # Processing the call keyword arguments (line 415)
    kwargs_416070 = {}
    # Getting the type of 'scal' (line 415)
    scal_416067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 13), 'scal', False)
    # Calling scal(args, kwargs) (line 415)
    scal_call_result_416071 = invoke(stypy.reporting.localization.Localization(__file__, 415, 13), scal_416067, *[alpha_416068, ux_416069], **kwargs_416070)
    
    # Assigning a type to the variable 'ux' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'ux', scal_call_result_416071)
    
    # Assigning a Call to a Name (line 418):
    
    # Assigning a Call to a Name (line 418):
    
    # Call to dot(...): (line 418)
    # Processing the call arguments (line 418)
    # Getting the type of 'cx' (line 418)
    cx_416073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 20), 'cx', False)
    # Getting the type of 'r' (line 418)
    r_416074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 24), 'r', False)
    # Processing the call keyword arguments (line 418)
    kwargs_416075 = {}
    # Getting the type of 'dot' (line 418)
    dot_416072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'dot', False)
    # Calling dot(args, kwargs) (line 418)
    dot_call_result_416076 = invoke(stypy.reporting.localization.Localization(__file__, 418, 16), dot_416072, *[cx_416073, r_416074], **kwargs_416075)
    
    # Assigning a type to the variable 'gamma' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'gamma', dot_call_result_416076)
    
    # Assigning a Call to a Name (line 419):
    
    # Assigning a Call to a Name (line 419):
    
    # Call to axpy(...): (line 419)
    # Processing the call arguments (line 419)
    # Getting the type of 'cx' (line 419)
    cx_416078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 17), 'cx', False)
    # Getting the type of 'r' (line 419)
    r_416079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 21), 'r', False)
    
    # Obtaining the type of the subscript
    int_416080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 32), 'int')
    # Getting the type of 'r' (line 419)
    r_416081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 24), 'r', False)
    # Obtaining the member 'shape' of a type (line 419)
    shape_416082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 24), r_416081, 'shape')
    # Obtaining the member '__getitem__' of a type (line 419)
    getitem___416083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 24), shape_416082, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 419)
    subscript_call_result_416084 = invoke(stypy.reporting.localization.Localization(__file__, 419, 24), getitem___416083, int_416080)
    
    
    # Getting the type of 'gamma' (line 419)
    gamma_416085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 37), 'gamma', False)
    # Applying the 'usub' unary operator (line 419)
    result___neg___416086 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 36), 'usub', gamma_416085)
    
    # Processing the call keyword arguments (line 419)
    kwargs_416087 = {}
    # Getting the type of 'axpy' (line 419)
    axpy_416077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'axpy', False)
    # Calling axpy(args, kwargs) (line 419)
    axpy_call_result_416088 = invoke(stypy.reporting.localization.Localization(__file__, 419, 12), axpy_416077, *[cx_416078, r_416079, subscript_call_result_416084, result___neg___416086], **kwargs_416087)
    
    # Assigning a type to the variable 'r' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'r', axpy_call_result_416088)
    
    # Assigning a Call to a Name (line 420):
    
    # Assigning a Call to a Name (line 420):
    
    # Call to axpy(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'ux' (line 420)
    ux_416090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 17), 'ux', False)
    # Getting the type of 'x' (line 420)
    x_416091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 21), 'x', False)
    
    # Obtaining the type of the subscript
    int_416092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 32), 'int')
    # Getting the type of 'x' (line 420)
    x_416093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 24), 'x', False)
    # Obtaining the member 'shape' of a type (line 420)
    shape_416094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 24), x_416093, 'shape')
    # Obtaining the member '__getitem__' of a type (line 420)
    getitem___416095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 24), shape_416094, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 420)
    subscript_call_result_416096 = invoke(stypy.reporting.localization.Localization(__file__, 420, 24), getitem___416095, int_416092)
    
    # Getting the type of 'gamma' (line 420)
    gamma_416097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 36), 'gamma', False)
    # Processing the call keyword arguments (line 420)
    kwargs_416098 = {}
    # Getting the type of 'axpy' (line 420)
    axpy_416089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'axpy', False)
    # Calling axpy(args, kwargs) (line 420)
    axpy_call_result_416099 = invoke(stypy.reporting.localization.Localization(__file__, 420, 12), axpy_416089, *[ux_416090, x_416091, subscript_call_result_416096, gamma_416097], **kwargs_416098)
    
    # Assigning a type to the variable 'x' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'x', axpy_call_result_416099)
    
    
    # Getting the type of 'truncate' (line 423)
    truncate_416100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'truncate')
    str_416101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 23), 'str', 'oldest')
    # Applying the binary operator '==' (line 423)
    result_eq_416102 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 11), '==', truncate_416100, str_416101)
    
    # Testing the type of an if condition (line 423)
    if_condition_416103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 8), result_eq_416102)
    # Assigning a type to the variable 'if_condition_416103' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'if_condition_416103', if_condition_416103)
    # SSA begins for if statement (line 423)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'CU' (line 424)
    CU_416105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 22), 'CU', False)
    # Processing the call keyword arguments (line 424)
    kwargs_416106 = {}
    # Getting the type of 'len' (line 424)
    len_416104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 18), 'len', False)
    # Calling len(args, kwargs) (line 424)
    len_call_result_416107 = invoke(stypy.reporting.localization.Localization(__file__, 424, 18), len_416104, *[CU_416105], **kwargs_416106)
    
    # Getting the type of 'k' (line 424)
    k_416108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 29), 'k')
    # Applying the binary operator '>=' (line 424)
    result_ge_416109 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 18), '>=', len_call_result_416107, k_416108)
    
    # Getting the type of 'CU' (line 424)
    CU_416110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 35), 'CU')
    # Applying the binary operator 'and' (line 424)
    result_and_keyword_416111 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 18), 'and', result_ge_416109, CU_416110)
    
    # Testing the type of an if condition (line 424)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 424, 12), result_and_keyword_416111)
    # SSA begins for while statement (line 424)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    # Deleting a member
    # Getting the type of 'CU' (line 425)
    CU_416112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 20), 'CU')
    
    # Obtaining the type of the subscript
    int_416113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 23), 'int')
    # Getting the type of 'CU' (line 425)
    CU_416114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 20), 'CU')
    # Obtaining the member '__getitem__' of a type (line 425)
    getitem___416115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 20), CU_416114, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 425)
    subscript_call_result_416116 = invoke(stypy.reporting.localization.Localization(__file__, 425, 20), getitem___416115, int_416113)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 16), CU_416112, subscript_call_result_416116)
    # SSA join for while statement (line 424)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 423)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'truncate' (line 426)
    truncate_416117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 13), 'truncate')
    str_416118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 25), 'str', 'smallest')
    # Applying the binary operator '==' (line 426)
    result_eq_416119 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 13), '==', truncate_416117, str_416118)
    
    # Testing the type of an if condition (line 426)
    if_condition_416120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 426, 13), result_eq_416119)
    # Assigning a type to the variable 'if_condition_416120' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 13), 'if_condition_416120', if_condition_416120)
    # SSA begins for if statement (line 426)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'CU' (line 427)
    CU_416122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 19), 'CU', False)
    # Processing the call keyword arguments (line 427)
    kwargs_416123 = {}
    # Getting the type of 'len' (line 427)
    len_416121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 15), 'len', False)
    # Calling len(args, kwargs) (line 427)
    len_call_result_416124 = invoke(stypy.reporting.localization.Localization(__file__, 427, 15), len_416121, *[CU_416122], **kwargs_416123)
    
    # Getting the type of 'k' (line 427)
    k_416125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 26), 'k')
    # Applying the binary operator '>=' (line 427)
    result_ge_416126 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 15), '>=', len_call_result_416124, k_416125)
    
    # Getting the type of 'CU' (line 427)
    CU_416127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 32), 'CU')
    # Applying the binary operator 'and' (line 427)
    result_and_keyword_416128 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 15), 'and', result_ge_416126, CU_416127)
    
    # Testing the type of an if condition (line 427)
    if_condition_416129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 427, 12), result_and_keyword_416128)
    # Assigning a type to the variable 'if_condition_416129' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'if_condition_416129', if_condition_416129)
    # SSA begins for if statement (line 427)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 429):
    
    # Assigning a Attribute to a Name (line 429):
    
    # Call to solve(...): (line 429)
    # Processing the call arguments (line 429)
    
    # Obtaining the type of the subscript
    int_416131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 29), 'int')
    slice_416132 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 429, 26), None, int_416131, None)
    slice_416133 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 429, 26), None, None, None)
    # Getting the type of 'R' (line 429)
    R_416134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 26), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 429)
    getitem___416135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 26), R_416134, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 429)
    subscript_call_result_416136 = invoke(stypy.reporting.localization.Localization(__file__, 429, 26), getitem___416135, (slice_416132, slice_416133))
    
    # Obtaining the member 'T' of a type (line 429)
    T_416137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 26), subscript_call_result_416136, 'T')
    # Getting the type of 'B' (line 429)
    B_416138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 38), 'B', False)
    # Obtaining the member 'T' of a type (line 429)
    T_416139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 38), B_416138, 'T')
    # Processing the call keyword arguments (line 429)
    kwargs_416140 = {}
    # Getting the type of 'solve' (line 429)
    solve_416130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 20), 'solve', False)
    # Calling solve(args, kwargs) (line 429)
    solve_call_result_416141 = invoke(stypy.reporting.localization.Localization(__file__, 429, 20), solve_416130, *[T_416137, T_416139], **kwargs_416140)
    
    # Obtaining the member 'T' of a type (line 429)
    T_416142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 20), solve_call_result_416141, 'T')
    # Assigning a type to the variable 'D' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 16), 'D', T_416142)
    
    # Assigning a Call to a Tuple (line 430):
    
    # Assigning a Subscript to a Name (line 430):
    
    # Obtaining the type of the subscript
    int_416143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 16), 'int')
    
    # Call to svd(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'D' (line 430)
    D_416145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 34), 'D', False)
    # Processing the call keyword arguments (line 430)
    kwargs_416146 = {}
    # Getting the type of 'svd' (line 430)
    svd_416144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 30), 'svd', False)
    # Calling svd(args, kwargs) (line 430)
    svd_call_result_416147 = invoke(stypy.reporting.localization.Localization(__file__, 430, 30), svd_416144, *[D_416145], **kwargs_416146)
    
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___416148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 16), svd_call_result_416147, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 430)
    subscript_call_result_416149 = invoke(stypy.reporting.localization.Localization(__file__, 430, 16), getitem___416148, int_416143)
    
    # Assigning a type to the variable 'tuple_var_assignment_414652' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'tuple_var_assignment_414652', subscript_call_result_416149)
    
    # Assigning a Subscript to a Name (line 430):
    
    # Obtaining the type of the subscript
    int_416150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 16), 'int')
    
    # Call to svd(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'D' (line 430)
    D_416152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 34), 'D', False)
    # Processing the call keyword arguments (line 430)
    kwargs_416153 = {}
    # Getting the type of 'svd' (line 430)
    svd_416151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 30), 'svd', False)
    # Calling svd(args, kwargs) (line 430)
    svd_call_result_416154 = invoke(stypy.reporting.localization.Localization(__file__, 430, 30), svd_416151, *[D_416152], **kwargs_416153)
    
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___416155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 16), svd_call_result_416154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 430)
    subscript_call_result_416156 = invoke(stypy.reporting.localization.Localization(__file__, 430, 16), getitem___416155, int_416150)
    
    # Assigning a type to the variable 'tuple_var_assignment_414653' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'tuple_var_assignment_414653', subscript_call_result_416156)
    
    # Assigning a Subscript to a Name (line 430):
    
    # Obtaining the type of the subscript
    int_416157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 16), 'int')
    
    # Call to svd(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'D' (line 430)
    D_416159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 34), 'D', False)
    # Processing the call keyword arguments (line 430)
    kwargs_416160 = {}
    # Getting the type of 'svd' (line 430)
    svd_416158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 30), 'svd', False)
    # Calling svd(args, kwargs) (line 430)
    svd_call_result_416161 = invoke(stypy.reporting.localization.Localization(__file__, 430, 30), svd_416158, *[D_416159], **kwargs_416160)
    
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___416162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 16), svd_call_result_416161, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 430)
    subscript_call_result_416163 = invoke(stypy.reporting.localization.Localization(__file__, 430, 16), getitem___416162, int_416157)
    
    # Assigning a type to the variable 'tuple_var_assignment_414654' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'tuple_var_assignment_414654', subscript_call_result_416163)
    
    # Assigning a Name to a Name (line 430):
    # Getting the type of 'tuple_var_assignment_414652' (line 430)
    tuple_var_assignment_414652_416164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'tuple_var_assignment_414652')
    # Assigning a type to the variable 'W' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'W', tuple_var_assignment_414652_416164)
    
    # Assigning a Name to a Name (line 430):
    # Getting the type of 'tuple_var_assignment_414653' (line 430)
    tuple_var_assignment_414653_416165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'tuple_var_assignment_414653')
    # Assigning a type to the variable 'sigma' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 19), 'sigma', tuple_var_assignment_414653_416165)
    
    # Assigning a Name to a Name (line 430):
    # Getting the type of 'tuple_var_assignment_414654' (line 430)
    tuple_var_assignment_414654_416166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'tuple_var_assignment_414654')
    # Assigning a type to the variable 'V' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 26), 'V', tuple_var_assignment_414654_416166)
    
    # Assigning a List to a Name (line 433):
    
    # Assigning a List to a Name (line 433):
    
    # Obtaining an instance of the builtin type 'list' (line 433)
    list_416167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 433)
    
    # Assigning a type to the variable 'new_CU' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 16), 'new_CU', list_416167)
    
    
    # Call to enumerate(...): (line 434)
    # Processing the call arguments (line 434)
    
    # Obtaining the type of the subscript
    slice_416169 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 434, 38), None, None, None)
    # Getting the type of 'k' (line 434)
    k_416170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 43), 'k', False)
    int_416171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 45), 'int')
    # Applying the binary operator '-' (line 434)
    result_sub_416172 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 43), '-', k_416170, int_416171)
    
    slice_416173 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 434, 38), None, result_sub_416172, None)
    # Getting the type of 'W' (line 434)
    W_416174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 38), 'W', False)
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___416175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 38), W_416174, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 434)
    subscript_call_result_416176 = invoke(stypy.reporting.localization.Localization(__file__, 434, 38), getitem___416175, (slice_416169, slice_416173))
    
    # Obtaining the member 'T' of a type (line 434)
    T_416177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 38), subscript_call_result_416176, 'T')
    # Processing the call keyword arguments (line 434)
    kwargs_416178 = {}
    # Getting the type of 'enumerate' (line 434)
    enumerate_416168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 28), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 434)
    enumerate_call_result_416179 = invoke(stypy.reporting.localization.Localization(__file__, 434, 28), enumerate_416168, *[T_416177], **kwargs_416178)
    
    # Testing the type of a for loop iterable (line 434)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 434, 16), enumerate_call_result_416179)
    # Getting the type of the for loop variable (line 434)
    for_loop_var_416180 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 434, 16), enumerate_call_result_416179)
    # Assigning a type to the variable 'j' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 16), for_loop_var_416180))
    # Assigning a type to the variable 'w' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 16), for_loop_var_416180))
    # SSA begins for a for statement (line 434)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Tuple (line 435):
    
    # Assigning a Subscript to a Name (line 435):
    
    # Obtaining the type of the subscript
    int_416181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 20), 'int')
    
    # Obtaining the type of the subscript
    int_416182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 30), 'int')
    # Getting the type of 'CU' (line 435)
    CU_416183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 27), 'CU')
    # Obtaining the member '__getitem__' of a type (line 435)
    getitem___416184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 27), CU_416183, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 435)
    subscript_call_result_416185 = invoke(stypy.reporting.localization.Localization(__file__, 435, 27), getitem___416184, int_416182)
    
    # Obtaining the member '__getitem__' of a type (line 435)
    getitem___416186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 20), subscript_call_result_416185, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 435)
    subscript_call_result_416187 = invoke(stypy.reporting.localization.Localization(__file__, 435, 20), getitem___416186, int_416181)
    
    # Assigning a type to the variable 'tuple_var_assignment_414655' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'tuple_var_assignment_414655', subscript_call_result_416187)
    
    # Assigning a Subscript to a Name (line 435):
    
    # Obtaining the type of the subscript
    int_416188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 20), 'int')
    
    # Obtaining the type of the subscript
    int_416189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 30), 'int')
    # Getting the type of 'CU' (line 435)
    CU_416190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 27), 'CU')
    # Obtaining the member '__getitem__' of a type (line 435)
    getitem___416191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 27), CU_416190, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 435)
    subscript_call_result_416192 = invoke(stypy.reporting.localization.Localization(__file__, 435, 27), getitem___416191, int_416189)
    
    # Obtaining the member '__getitem__' of a type (line 435)
    getitem___416193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 20), subscript_call_result_416192, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 435)
    subscript_call_result_416194 = invoke(stypy.reporting.localization.Localization(__file__, 435, 20), getitem___416193, int_416188)
    
    # Assigning a type to the variable 'tuple_var_assignment_414656' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'tuple_var_assignment_414656', subscript_call_result_416194)
    
    # Assigning a Name to a Name (line 435):
    # Getting the type of 'tuple_var_assignment_414655' (line 435)
    tuple_var_assignment_414655_416195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'tuple_var_assignment_414655')
    # Assigning a type to the variable 'c' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'c', tuple_var_assignment_414655_416195)
    
    # Assigning a Name to a Name (line 435):
    # Getting the type of 'tuple_var_assignment_414656' (line 435)
    tuple_var_assignment_414656_416196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'tuple_var_assignment_414656')
    # Assigning a type to the variable 'u' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 23), 'u', tuple_var_assignment_414656_416196)
    
    # Assigning a BinOp to a Name (line 436):
    
    # Assigning a BinOp to a Name (line 436):
    # Getting the type of 'c' (line 436)
    c_416197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 24), 'c')
    
    # Obtaining the type of the subscript
    int_416198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 30), 'int')
    # Getting the type of 'w' (line 436)
    w_416199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 28), 'w')
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___416200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 28), w_416199, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_416201 = invoke(stypy.reporting.localization.Localization(__file__, 436, 28), getitem___416200, int_416198)
    
    # Applying the binary operator '*' (line 436)
    result_mul_416202 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 24), '*', c_416197, subscript_call_result_416201)
    
    # Assigning a type to the variable 'c' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 20), 'c', result_mul_416202)
    
    # Assigning a BinOp to a Name (line 437):
    
    # Assigning a BinOp to a Name (line 437):
    # Getting the type of 'u' (line 437)
    u_416203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 24), 'u')
    
    # Obtaining the type of the subscript
    int_416204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 30), 'int')
    # Getting the type of 'w' (line 437)
    w_416205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 28), 'w')
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___416206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 28), w_416205, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_416207 = invoke(stypy.reporting.localization.Localization(__file__, 437, 28), getitem___416206, int_416204)
    
    # Applying the binary operator '*' (line 437)
    result_mul_416208 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 24), '*', u_416203, subscript_call_result_416207)
    
    # Assigning a type to the variable 'u' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 20), 'u', result_mul_416208)
    
    
    # Call to zip(...): (line 438)
    # Processing the call arguments (line 438)
    
    # Obtaining the type of the subscript
    int_416210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 42), 'int')
    slice_416211 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 438, 39), int_416210, None, None)
    # Getting the type of 'CU' (line 438)
    CU_416212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 39), 'CU', False)
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___416213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 39), CU_416212, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_416214 = invoke(stypy.reporting.localization.Localization(__file__, 438, 39), getitem___416213, slice_416211)
    
    
    # Obtaining the type of the subscript
    int_416215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 49), 'int')
    slice_416216 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 438, 47), int_416215, None, None)
    # Getting the type of 'w' (line 438)
    w_416217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 47), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___416218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 47), w_416217, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_416219 = invoke(stypy.reporting.localization.Localization(__file__, 438, 47), getitem___416218, slice_416216)
    
    # Processing the call keyword arguments (line 438)
    kwargs_416220 = {}
    # Getting the type of 'zip' (line 438)
    zip_416209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 35), 'zip', False)
    # Calling zip(args, kwargs) (line 438)
    zip_call_result_416221 = invoke(stypy.reporting.localization.Localization(__file__, 438, 35), zip_416209, *[subscript_call_result_416214, subscript_call_result_416219], **kwargs_416220)
    
    # Testing the type of a for loop iterable (line 438)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 438, 20), zip_call_result_416221)
    # Getting the type of the for loop variable (line 438)
    for_loop_var_416222 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 438, 20), zip_call_result_416221)
    # Assigning a type to the variable 'cup' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 20), 'cup', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 20), for_loop_var_416222))
    # Assigning a type to the variable 'wp' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 20), 'wp', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 20), for_loop_var_416222))
    # SSA begins for a for statement (line 438)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Tuple (line 439):
    
    # Assigning a Subscript to a Name (line 439):
    
    # Obtaining the type of the subscript
    int_416223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 24), 'int')
    # Getting the type of 'cup' (line 439)
    cup_416224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 33), 'cup')
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___416225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 24), cup_416224, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_416226 = invoke(stypy.reporting.localization.Localization(__file__, 439, 24), getitem___416225, int_416223)
    
    # Assigning a type to the variable 'tuple_var_assignment_414657' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 24), 'tuple_var_assignment_414657', subscript_call_result_416226)
    
    # Assigning a Subscript to a Name (line 439):
    
    # Obtaining the type of the subscript
    int_416227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 24), 'int')
    # Getting the type of 'cup' (line 439)
    cup_416228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 33), 'cup')
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___416229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 24), cup_416228, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_416230 = invoke(stypy.reporting.localization.Localization(__file__, 439, 24), getitem___416229, int_416227)
    
    # Assigning a type to the variable 'tuple_var_assignment_414658' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 24), 'tuple_var_assignment_414658', subscript_call_result_416230)
    
    # Assigning a Name to a Name (line 439):
    # Getting the type of 'tuple_var_assignment_414657' (line 439)
    tuple_var_assignment_414657_416231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 24), 'tuple_var_assignment_414657')
    # Assigning a type to the variable 'cp' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 24), 'cp', tuple_var_assignment_414657_416231)
    
    # Assigning a Name to a Name (line 439):
    # Getting the type of 'tuple_var_assignment_414658' (line 439)
    tuple_var_assignment_414658_416232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 24), 'tuple_var_assignment_414658')
    # Assigning a type to the variable 'up' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 28), 'up', tuple_var_assignment_414658_416232)
    
    # Assigning a Call to a Name (line 440):
    
    # Assigning a Call to a Name (line 440):
    
    # Call to axpy(...): (line 440)
    # Processing the call arguments (line 440)
    # Getting the type of 'cp' (line 440)
    cp_416234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 33), 'cp', False)
    # Getting the type of 'c' (line 440)
    c_416235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 37), 'c', False)
    
    # Obtaining the type of the subscript
    int_416236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 48), 'int')
    # Getting the type of 'c' (line 440)
    c_416237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 40), 'c', False)
    # Obtaining the member 'shape' of a type (line 440)
    shape_416238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 40), c_416237, 'shape')
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___416239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 40), shape_416238, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_416240 = invoke(stypy.reporting.localization.Localization(__file__, 440, 40), getitem___416239, int_416236)
    
    # Getting the type of 'wp' (line 440)
    wp_416241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 52), 'wp', False)
    # Processing the call keyword arguments (line 440)
    kwargs_416242 = {}
    # Getting the type of 'axpy' (line 440)
    axpy_416233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 28), 'axpy', False)
    # Calling axpy(args, kwargs) (line 440)
    axpy_call_result_416243 = invoke(stypy.reporting.localization.Localization(__file__, 440, 28), axpy_416233, *[cp_416234, c_416235, subscript_call_result_416240, wp_416241], **kwargs_416242)
    
    # Assigning a type to the variable 'c' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 24), 'c', axpy_call_result_416243)
    
    # Assigning a Call to a Name (line 441):
    
    # Assigning a Call to a Name (line 441):
    
    # Call to axpy(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'up' (line 441)
    up_416245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 33), 'up', False)
    # Getting the type of 'u' (line 441)
    u_416246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 37), 'u', False)
    
    # Obtaining the type of the subscript
    int_416247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 48), 'int')
    # Getting the type of 'u' (line 441)
    u_416248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 40), 'u', False)
    # Obtaining the member 'shape' of a type (line 441)
    shape_416249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 40), u_416248, 'shape')
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___416250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 40), shape_416249, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_416251 = invoke(stypy.reporting.localization.Localization(__file__, 441, 40), getitem___416250, int_416247)
    
    # Getting the type of 'wp' (line 441)
    wp_416252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 52), 'wp', False)
    # Processing the call keyword arguments (line 441)
    kwargs_416253 = {}
    # Getting the type of 'axpy' (line 441)
    axpy_416244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 28), 'axpy', False)
    # Calling axpy(args, kwargs) (line 441)
    axpy_call_result_416254 = invoke(stypy.reporting.localization.Localization(__file__, 441, 28), axpy_416244, *[up_416245, u_416246, subscript_call_result_416251, wp_416252], **kwargs_416253)
    
    # Assigning a type to the variable 'u' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 24), 'u', axpy_call_result_416254)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'new_CU' (line 446)
    new_CU_416255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 34), 'new_CU')
    # Testing the type of a for loop iterable (line 446)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 446, 20), new_CU_416255)
    # Getting the type of the for loop variable (line 446)
    for_loop_var_416256 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 446, 20), new_CU_416255)
    # Assigning a type to the variable 'cp' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 20), 'cp', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 20), for_loop_var_416256))
    # Assigning a type to the variable 'up' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 20), 'up', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 20), for_loop_var_416256))
    # SSA begins for a for statement (line 446)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 447):
    
    # Assigning a Call to a Name (line 447):
    
    # Call to dot(...): (line 447)
    # Processing the call arguments (line 447)
    # Getting the type of 'cp' (line 447)
    cp_416258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 36), 'cp', False)
    # Getting the type of 'c' (line 447)
    c_416259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 40), 'c', False)
    # Processing the call keyword arguments (line 447)
    kwargs_416260 = {}
    # Getting the type of 'dot' (line 447)
    dot_416257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 32), 'dot', False)
    # Calling dot(args, kwargs) (line 447)
    dot_call_result_416261 = invoke(stypy.reporting.localization.Localization(__file__, 447, 32), dot_416257, *[cp_416258, c_416259], **kwargs_416260)
    
    # Assigning a type to the variable 'alpha' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 24), 'alpha', dot_call_result_416261)
    
    # Assigning a Call to a Name (line 448):
    
    # Assigning a Call to a Name (line 448):
    
    # Call to axpy(...): (line 448)
    # Processing the call arguments (line 448)
    # Getting the type of 'cp' (line 448)
    cp_416263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 33), 'cp', False)
    # Getting the type of 'c' (line 448)
    c_416264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 37), 'c', False)
    
    # Obtaining the type of the subscript
    int_416265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 48), 'int')
    # Getting the type of 'c' (line 448)
    c_416266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 40), 'c', False)
    # Obtaining the member 'shape' of a type (line 448)
    shape_416267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 40), c_416266, 'shape')
    # Obtaining the member '__getitem__' of a type (line 448)
    getitem___416268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 40), shape_416267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 448)
    subscript_call_result_416269 = invoke(stypy.reporting.localization.Localization(__file__, 448, 40), getitem___416268, int_416265)
    
    
    # Getting the type of 'alpha' (line 448)
    alpha_416270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 53), 'alpha', False)
    # Applying the 'usub' unary operator (line 448)
    result___neg___416271 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 52), 'usub', alpha_416270)
    
    # Processing the call keyword arguments (line 448)
    kwargs_416272 = {}
    # Getting the type of 'axpy' (line 448)
    axpy_416262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 28), 'axpy', False)
    # Calling axpy(args, kwargs) (line 448)
    axpy_call_result_416273 = invoke(stypy.reporting.localization.Localization(__file__, 448, 28), axpy_416262, *[cp_416263, c_416264, subscript_call_result_416269, result___neg___416271], **kwargs_416272)
    
    # Assigning a type to the variable 'c' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 24), 'c', axpy_call_result_416273)
    
    # Assigning a Call to a Name (line 449):
    
    # Assigning a Call to a Name (line 449):
    
    # Call to axpy(...): (line 449)
    # Processing the call arguments (line 449)
    # Getting the type of 'up' (line 449)
    up_416275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 33), 'up', False)
    # Getting the type of 'u' (line 449)
    u_416276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 37), 'u', False)
    
    # Obtaining the type of the subscript
    int_416277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 48), 'int')
    # Getting the type of 'u' (line 449)
    u_416278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 40), 'u', False)
    # Obtaining the member 'shape' of a type (line 449)
    shape_416279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 40), u_416278, 'shape')
    # Obtaining the member '__getitem__' of a type (line 449)
    getitem___416280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 40), shape_416279, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 449)
    subscript_call_result_416281 = invoke(stypy.reporting.localization.Localization(__file__, 449, 40), getitem___416280, int_416277)
    
    
    # Getting the type of 'alpha' (line 449)
    alpha_416282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 53), 'alpha', False)
    # Applying the 'usub' unary operator (line 449)
    result___neg___416283 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 52), 'usub', alpha_416282)
    
    # Processing the call keyword arguments (line 449)
    kwargs_416284 = {}
    # Getting the type of 'axpy' (line 449)
    axpy_416274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 28), 'axpy', False)
    # Calling axpy(args, kwargs) (line 449)
    axpy_call_result_416285 = invoke(stypy.reporting.localization.Localization(__file__, 449, 28), axpy_416274, *[up_416275, u_416276, subscript_call_result_416281, result___neg___416283], **kwargs_416284)
    
    # Assigning a type to the variable 'u' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 24), 'u', axpy_call_result_416285)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 450):
    
    # Assigning a Call to a Name (line 450):
    
    # Call to nrm2(...): (line 450)
    # Processing the call arguments (line 450)
    # Getting the type of 'c' (line 450)
    c_416287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 33), 'c', False)
    # Processing the call keyword arguments (line 450)
    kwargs_416288 = {}
    # Getting the type of 'nrm2' (line 450)
    nrm2_416286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 28), 'nrm2', False)
    # Calling nrm2(args, kwargs) (line 450)
    nrm2_call_result_416289 = invoke(stypy.reporting.localization.Localization(__file__, 450, 28), nrm2_416286, *[c_416287], **kwargs_416288)
    
    # Assigning a type to the variable 'alpha' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 20), 'alpha', nrm2_call_result_416289)
    
    # Assigning a Call to a Name (line 451):
    
    # Assigning a Call to a Name (line 451):
    
    # Call to scal(...): (line 451)
    # Processing the call arguments (line 451)
    float_416291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 29), 'float')
    # Getting the type of 'alpha' (line 451)
    alpha_416292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 33), 'alpha', False)
    # Applying the binary operator 'div' (line 451)
    result_div_416293 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 29), 'div', float_416291, alpha_416292)
    
    # Getting the type of 'c' (line 451)
    c_416294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 40), 'c', False)
    # Processing the call keyword arguments (line 451)
    kwargs_416295 = {}
    # Getting the type of 'scal' (line 451)
    scal_416290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 24), 'scal', False)
    # Calling scal(args, kwargs) (line 451)
    scal_call_result_416296 = invoke(stypy.reporting.localization.Localization(__file__, 451, 24), scal_416290, *[result_div_416293, c_416294], **kwargs_416295)
    
    # Assigning a type to the variable 'c' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 20), 'c', scal_call_result_416296)
    
    # Assigning a Call to a Name (line 452):
    
    # Assigning a Call to a Name (line 452):
    
    # Call to scal(...): (line 452)
    # Processing the call arguments (line 452)
    float_416298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 29), 'float')
    # Getting the type of 'alpha' (line 452)
    alpha_416299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 33), 'alpha', False)
    # Applying the binary operator 'div' (line 452)
    result_div_416300 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 29), 'div', float_416298, alpha_416299)
    
    # Getting the type of 'u' (line 452)
    u_416301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 40), 'u', False)
    # Processing the call keyword arguments (line 452)
    kwargs_416302 = {}
    # Getting the type of 'scal' (line 452)
    scal_416297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 24), 'scal', False)
    # Calling scal(args, kwargs) (line 452)
    scal_call_result_416303 = invoke(stypy.reporting.localization.Localization(__file__, 452, 24), scal_416297, *[result_div_416300, u_416301], **kwargs_416302)
    
    # Assigning a type to the variable 'u' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 20), 'u', scal_call_result_416303)
    
    # Call to append(...): (line 454)
    # Processing the call arguments (line 454)
    
    # Obtaining an instance of the builtin type 'tuple' (line 454)
    tuple_416306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 454)
    # Adding element type (line 454)
    # Getting the type of 'c' (line 454)
    c_416307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 35), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 35), tuple_416306, c_416307)
    # Adding element type (line 454)
    # Getting the type of 'u' (line 454)
    u_416308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 38), 'u', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 35), tuple_416306, u_416308)
    
    # Processing the call keyword arguments (line 454)
    kwargs_416309 = {}
    # Getting the type of 'new_CU' (line 454)
    new_CU_416304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 20), 'new_CU', False)
    # Obtaining the member 'append' of a type (line 454)
    append_416305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 20), new_CU_416304, 'append')
    # Calling append(args, kwargs) (line 454)
    append_call_result_416310 = invoke(stypy.reporting.localization.Localization(__file__, 454, 20), append_416305, *[tuple_416306], **kwargs_416309)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 455):
    
    # Assigning a Name to a Subscript (line 455):
    # Getting the type of 'new_CU' (line 455)
    new_CU_416311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 24), 'new_CU')
    # Getting the type of 'CU' (line 455)
    CU_416312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'CU')
    slice_416313 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 455, 16), None, None, None)
    # Storing an element on a container (line 455)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 16), CU_416312, (slice_416313, new_CU_416311))
    # SSA join for if statement (line 427)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 426)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 423)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 458)
    # Processing the call arguments (line 458)
    
    # Obtaining an instance of the builtin type 'tuple' (line 458)
    tuple_416316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 458)
    # Adding element type (line 458)
    # Getting the type of 'cx' (line 458)
    cx_416317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 19), 'cx', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 19), tuple_416316, cx_416317)
    # Adding element type (line 458)
    # Getting the type of 'ux' (line 458)
    ux_416318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 23), 'ux', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 19), tuple_416316, ux_416318)
    
    # Processing the call keyword arguments (line 458)
    kwargs_416319 = {}
    # Getting the type of 'CU' (line 458)
    CU_416314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'CU', False)
    # Obtaining the member 'append' of a type (line 458)
    append_416315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), CU_416314, 'append')
    # Calling append(args, kwargs) (line 458)
    append_call_result_416320 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), append_416315, *[tuple_416316], **kwargs_416319)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 461)
    # Processing the call arguments (line 461)
    
    # Obtaining an instance of the builtin type 'tuple' (line 461)
    tuple_416323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 461)
    # Adding element type (line 461)
    # Getting the type of 'None' (line 461)
    None_416324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 15), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 15), tuple_416323, None_416324)
    # Adding element type (line 461)
    
    # Call to copy(...): (line 461)
    # Processing the call keyword arguments (line 461)
    kwargs_416327 = {}
    # Getting the type of 'x' (line 461)
    x_416325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 21), 'x', False)
    # Obtaining the member 'copy' of a type (line 461)
    copy_416326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 21), x_416325, 'copy')
    # Calling copy(args, kwargs) (line 461)
    copy_call_result_416328 = invoke(stypy.reporting.localization.Localization(__file__, 461, 21), copy_416326, *[], **kwargs_416327)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 15), tuple_416323, copy_call_result_416328)
    
    # Processing the call keyword arguments (line 461)
    kwargs_416329 = {}
    # Getting the type of 'CU' (line 461)
    CU_416321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'CU', False)
    # Obtaining the member 'append' of a type (line 461)
    append_416322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 4), CU_416321, 'append')
    # Calling append(args, kwargs) (line 461)
    append_call_result_416330 = invoke(stypy.reporting.localization.Localization(__file__, 461, 4), append_416322, *[tuple_416323], **kwargs_416329)
    
    
    # Getting the type of 'discard_C' (line 462)
    discard_C_416331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 7), 'discard_C')
    # Testing the type of an if condition (line 462)
    if_condition_416332 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 4), discard_C_416331)
    # Assigning a type to the variable 'if_condition_416332' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'if_condition_416332', if_condition_416332)
    # SSA begins for if statement (line 462)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a ListComp to a Subscript (line 463):
    
    # Assigning a ListComp to a Subscript (line 463):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'CU' (line 463)
    CU_416336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 42), 'CU')
    comprehension_416337 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 17), CU_416336)
    # Assigning a type to the variable 'cz' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 17), 'cz', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 17), comprehension_416337))
    # Assigning a type to the variable 'uz' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 17), 'uz', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 17), comprehension_416337))
    
    # Obtaining an instance of the builtin type 'tuple' (line 463)
    tuple_416333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 463)
    # Adding element type (line 463)
    # Getting the type of 'None' (line 463)
    None_416334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 18), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 18), tuple_416333, None_416334)
    # Adding element type (line 463)
    # Getting the type of 'uz' (line 463)
    uz_416335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 24), 'uz')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 18), tuple_416333, uz_416335)
    
    list_416338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 17), list_416338, tuple_416333)
    # Getting the type of 'CU' (line 463)
    CU_416339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'CU')
    slice_416340 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 463, 8), None, None, None)
    # Storing an element on a container (line 463)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 8), CU_416339, (slice_416340, list_416338))
    # SSA join for if statement (line 462)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 465)
    tuple_416341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 465)
    # Adding element type (line 465)
    
    # Call to postprocess(...): (line 465)
    # Processing the call arguments (line 465)
    # Getting the type of 'x' (line 465)
    x_416343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 23), 'x', False)
    # Processing the call keyword arguments (line 465)
    kwargs_416344 = {}
    # Getting the type of 'postprocess' (line 465)
    postprocess_416342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 11), 'postprocess', False)
    # Calling postprocess(args, kwargs) (line 465)
    postprocess_call_result_416345 = invoke(stypy.reporting.localization.Localization(__file__, 465, 11), postprocess_416342, *[x_416343], **kwargs_416344)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 11), tuple_416341, postprocess_call_result_416345)
    # Adding element type (line 465)
    # Getting the type of 'j_outer' (line 465)
    j_outer_416346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 27), 'j_outer')
    int_416347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 37), 'int')
    # Applying the binary operator '+' (line 465)
    result_add_416348 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 27), '+', j_outer_416346, int_416347)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 11), tuple_416341, result_add_416348)
    
    # Assigning a type to the variable 'stypy_return_type' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'stypy_return_type', tuple_416341)
    
    # ################# End of 'gcrotmk(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gcrotmk' in the type store
    # Getting the type of 'stypy_return_type' (line 181)
    stypy_return_type_416349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_416349)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gcrotmk'
    return stypy_return_type_416349

# Assigning a type to the variable 'gcrotmk' (line 181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'gcrotmk', gcrotmk)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
