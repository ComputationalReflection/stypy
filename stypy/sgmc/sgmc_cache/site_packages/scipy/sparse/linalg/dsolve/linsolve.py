
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from warnings import warn
4: 
5: import numpy as np
6: from numpy import asarray, empty, ravel, nonzero
7: from scipy.sparse import (isspmatrix_csc, isspmatrix_csr, isspmatrix,
8:                           SparseEfficiencyWarning, csc_matrix, csr_matrix)
9: from scipy.linalg import LinAlgError
10: 
11: from . import _superlu
12: 
13: noScikit = False
14: try:
15:     import scikits.umfpack as umfpack
16: except ImportError:
17:     noScikit = True
18: 
19: useUmfpack = not noScikit
20: 
21: __all__ = ['use_solver', 'spsolve', 'splu', 'spilu', 'factorized',
22:            'MatrixRankWarning', 'spsolve_triangular']
23: 
24: 
25: class MatrixRankWarning(UserWarning):
26:     pass
27: 
28: 
29: def use_solver(**kwargs):
30:     '''
31:     Select default sparse direct solver to be used.
32: 
33:     Parameters
34:     ----------
35:     useUmfpack : bool, optional
36:         Use UMFPACK over SuperLU. Has effect only if scikits.umfpack is
37:         installed. Default: True
38:     assumeSortedIndices : bool, optional
39:         Allow UMFPACK to skip the step of sorting indices for a CSR/CSC matrix.
40:         Has effect only if useUmfpack is True and scikits.umfpack is installed.
41:         Default: False
42: 
43:     Notes
44:     -----
45:     The default sparse solver is umfpack when available
46:     (scikits.umfpack is installed). This can be changed by passing
47:     useUmfpack = False, which then causes the always present SuperLU
48:     based solver to be used.
49: 
50:     Umfpack requires a CSR/CSC matrix to have sorted column/row indices. If
51:     sure that the matrix fulfills this, pass ``assumeSortedIndices=True``
52:     to gain some speed.
53: 
54:     '''
55:     if 'useUmfpack' in kwargs:
56:         globals()['useUmfpack'] = kwargs['useUmfpack']
57:     if useUmfpack and 'assumeSortedIndices' in kwargs:
58:         umfpack.configure(assumeSortedIndices=kwargs['assumeSortedIndices'])
59: 
60: def _get_umf_family(A):
61:     '''Get umfpack family string given the sparse matrix dtype.'''
62:     _families = {
63:         (np.float64, np.int32): 'di',
64:         (np.complex128, np.int32): 'zi',
65:         (np.float64, np.int64): 'dl',
66:         (np.complex128, np.int64): 'zl'
67:     }
68: 
69:     f_type = np.sctypeDict[A.dtype.name]
70:     i_type = np.sctypeDict[A.indices.dtype.name]
71: 
72:     try:
73:         family = _families[(f_type, i_type)]
74: 
75:     except KeyError:
76:         msg = 'only float64 or complex128 matrices with int32 or int64' \
77:             ' indices are supported! (got: matrix: %s, indices: %s)' \
78:             % (f_type, i_type)
79:         raise ValueError(msg)
80: 
81:     return family
82: 
83: def spsolve(A, b, permc_spec=None, use_umfpack=True):
84:     '''Solve the sparse linear system Ax=b, where b may be a vector or a matrix.
85: 
86:     Parameters
87:     ----------
88:     A : ndarray or sparse matrix
89:         The square matrix A will be converted into CSC or CSR form
90:     b : ndarray or sparse matrix
91:         The matrix or vector representing the right hand side of the equation.
92:         If a vector, b.shape must be (n,) or (n, 1).
93:     permc_spec : str, optional
94:         How to permute the columns of the matrix for sparsity preservation.
95:         (default: 'COLAMD')
96: 
97:         - ``NATURAL``: natural ordering.
98:         - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
99:         - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
100:         - ``COLAMD``: approximate minimum degree column ordering
101:     use_umfpack : bool, optional
102:         if True (default) then use umfpack for the solution.  This is
103:         only referenced if b is a vector and ``scikit-umfpack`` is installed.
104: 
105:     Returns
106:     -------
107:     x : ndarray or sparse matrix
108:         the solution of the sparse linear equation.
109:         If b is a vector, then x is a vector of size A.shape[1]
110:         If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])
111: 
112:     Notes
113:     -----
114:     For solving the matrix expression AX = B, this solver assumes the resulting
115:     matrix X is sparse, as is often the case for very sparse inputs.  If the
116:     resulting X is dense, the construction of this sparse result will be
117:     relatively expensive.  In that case, consider converting A to a dense
118:     matrix and using scipy.linalg.solve or its variants.
119: 
120:     Examples
121:     --------
122:     >>> from scipy.sparse import csc_matrix
123:     >>> from scipy.sparse.linalg import spsolve
124:     >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
125:     >>> B = csc_matrix([[2, 0], [-1, 0], [2, 0]], dtype=float)
126:     >>> x = spsolve(A, B)
127:     >>> np.allclose(A.dot(x).todense(), B.todense())
128:     True
129:     '''
130:     if not (isspmatrix_csc(A) or isspmatrix_csr(A)):
131:         A = csc_matrix(A)
132:         warn('spsolve requires A be CSC or CSR matrix format',
133:                 SparseEfficiencyWarning)
134: 
135:     # b is a vector only if b have shape (n,) or (n, 1)
136:     b_is_sparse = isspmatrix(b)
137:     if not b_is_sparse:
138:         b = asarray(b)
139:     b_is_vector = ((b.ndim == 1) or (b.ndim == 2 and b.shape[1] == 1))
140: 
141:     A.sort_indices()
142:     A = A.asfptype()  # upcast to a floating point format
143:     result_dtype = np.promote_types(A.dtype, b.dtype)
144:     if A.dtype != result_dtype:
145:         A = A.astype(result_dtype)
146:     if b.dtype != result_dtype:
147:         b = b.astype(result_dtype)
148: 
149:     # validate input shapes
150:     M, N = A.shape
151:     if (M != N):
152:         raise ValueError("matrix must be square (has shape %s)" % ((M, N),))
153: 
154:     if M != b.shape[0]:
155:         raise ValueError("matrix - rhs dimension mismatch (%s - %s)"
156:                          % (A.shape, b.shape[0]))
157: 
158:     use_umfpack = use_umfpack and useUmfpack
159: 
160:     if b_is_vector and use_umfpack:
161:         if b_is_sparse:
162:             b_vec = b.toarray()
163:         else:
164:             b_vec = b
165:         b_vec = asarray(b_vec, dtype=A.dtype).ravel()
166: 
167:         if noScikit:
168:             raise RuntimeError('Scikits.umfpack not installed.')
169: 
170:         if A.dtype.char not in 'dD':
171:             raise ValueError("convert matrix data to double, please, using"
172:                   " .astype(), or set linsolve.useUmfpack = False")
173: 
174:         umf = umfpack.UmfpackContext(_get_umf_family(A))
175:         x = umf.linsolve(umfpack.UMFPACK_A, A, b_vec,
176:                          autoTranspose=True)
177:     else:
178:         if b_is_vector and b_is_sparse:
179:             b = b.toarray()
180:             b_is_sparse = False
181: 
182:         if not b_is_sparse:
183:             if isspmatrix_csc(A):
184:                 flag = 1  # CSC format
185:             else:
186:                 flag = 0  # CSR format
187: 
188:             options = dict(ColPerm=permc_spec)
189:             x, info = _superlu.gssv(N, A.nnz, A.data, A.indices, A.indptr,
190:                                     b, flag, options=options)
191:             if info != 0:
192:                 warn("Matrix is exactly singular", MatrixRankWarning)
193:                 x.fill(np.nan)
194:             if b_is_vector:
195:                 x = x.ravel()
196:         else:
197:             # b is sparse
198:             Afactsolve = factorized(A)
199: 
200:             if not isspmatrix_csc(b):
201:                 warn('spsolve is more efficient when sparse b '
202:                      'is in the CSC matrix format', SparseEfficiencyWarning)
203:                 b = csc_matrix(b)
204: 
205:             # Create a sparse output matrix by repeatedly applying
206:             # the sparse factorization to solve columns of b.
207:             data_segs = []
208:             row_segs = []
209:             col_segs = []
210:             for j in range(b.shape[1]):
211:                 bj = b[:, j].A.ravel()
212:                 xj = Afactsolve(bj)
213:                 w = np.flatnonzero(xj)
214:                 segment_length = w.shape[0]
215:                 row_segs.append(w)
216:                 col_segs.append(np.ones(segment_length, dtype=int)*j)
217:                 data_segs.append(np.asarray(xj[w], dtype=A.dtype))
218:             sparse_data = np.concatenate(data_segs)
219:             sparse_row = np.concatenate(row_segs)
220:             sparse_col = np.concatenate(col_segs)
221:             x = A.__class__((sparse_data, (sparse_row, sparse_col)),
222:                            shape=b.shape, dtype=A.dtype)
223: 
224:     return x
225: 
226: 
227: def splu(A, permc_spec=None, diag_pivot_thresh=None,
228:          relax=None, panel_size=None, options=dict()):
229:     '''
230:     Compute the LU decomposition of a sparse, square matrix.
231: 
232:     Parameters
233:     ----------
234:     A : sparse matrix
235:         Sparse matrix to factorize. Should be in CSR or CSC format.
236:     permc_spec : str, optional
237:         How to permute the columns of the matrix for sparsity preservation.
238:         (default: 'COLAMD')
239: 
240:         - ``NATURAL``: natural ordering.
241:         - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
242:         - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
243:         - ``COLAMD``: approximate minimum degree column ordering
244: 
245:     diag_pivot_thresh : float, optional
246:         Threshold used for a diagonal entry to be an acceptable pivot.
247:         See SuperLU user's guide for details [1]_
248:     relax : int, optional
249:         Expert option for customizing the degree of relaxing supernodes.
250:         See SuperLU user's guide for details [1]_
251:     panel_size : int, optional
252:         Expert option for customizing the panel size.
253:         See SuperLU user's guide for details [1]_
254:     options : dict, optional
255:         Dictionary containing additional expert options to SuperLU.
256:         See SuperLU user guide [1]_ (section 2.4 on the 'Options' argument)
257:         for more details. For example, you can specify
258:         ``options=dict(Equil=False, IterRefine='SINGLE'))``
259:         to turn equilibration off and perform a single iterative refinement.
260: 
261:     Returns
262:     -------
263:     invA : scipy.sparse.linalg.SuperLU
264:         Object, which has a ``solve`` method.
265: 
266:     See also
267:     --------
268:     spilu : incomplete LU decomposition
269: 
270:     Notes
271:     -----
272:     This function uses the SuperLU library.
273: 
274:     References
275:     ----------
276:     .. [1] SuperLU http://crd.lbl.gov/~xiaoye/SuperLU/
277: 
278:     Examples
279:     --------
280:     >>> from scipy.sparse import csc_matrix
281:     >>> from scipy.sparse.linalg import splu
282:     >>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
283:     >>> B = splu(A)
284:     >>> x = np.array([1., 2., 3.], dtype=float)
285:     >>> B.solve(x)
286:     array([ 1. , -3. , -1.5])
287:     >>> A.dot(B.solve(x))
288:     array([ 1.,  2.,  3.])
289:     >>> B.solve(A.dot(x))
290:     array([ 1.,  2.,  3.])
291:     '''
292: 
293:     if not isspmatrix_csc(A):
294:         A = csc_matrix(A)
295:         warn('splu requires CSC matrix format', SparseEfficiencyWarning)
296: 
297:     A.sort_indices()
298:     A = A.asfptype()  # upcast to a floating point format
299: 
300:     M, N = A.shape
301:     if (M != N):
302:         raise ValueError("can only factor square matrices")  # is this true?
303: 
304:     _options = dict(DiagPivotThresh=diag_pivot_thresh, ColPerm=permc_spec,
305:                     PanelSize=panel_size, Relax=relax)
306:     if options is not None:
307:         _options.update(options)
308:     return _superlu.gstrf(N, A.nnz, A.data, A.indices, A.indptr,
309:                           ilu=False, options=_options)
310: 
311: 
312: def spilu(A, drop_tol=None, fill_factor=None, drop_rule=None, permc_spec=None,
313:           diag_pivot_thresh=None, relax=None, panel_size=None, options=None):
314:     '''
315:     Compute an incomplete LU decomposition for a sparse, square matrix.
316: 
317:     The resulting object is an approximation to the inverse of `A`.
318: 
319:     Parameters
320:     ----------
321:     A : (N, N) array_like
322:         Sparse matrix to factorize
323:     drop_tol : float, optional
324:         Drop tolerance (0 <= tol <= 1) for an incomplete LU decomposition.
325:         (default: 1e-4)
326:     fill_factor : float, optional
327:         Specifies the fill ratio upper bound (>= 1.0) for ILU. (default: 10)
328:     drop_rule : str, optional
329:         Comma-separated string of drop rules to use.
330:         Available rules: ``basic``, ``prows``, ``column``, ``area``,
331:         ``secondary``, ``dynamic``, ``interp``. (Default: ``basic,area``)
332: 
333:         See SuperLU documentation for details.
334: 
335:     Remaining other options
336:         Same as for `splu`
337: 
338:     Returns
339:     -------
340:     invA_approx : scipy.sparse.linalg.SuperLU
341:         Object, which has a ``solve`` method.
342: 
343:     See also
344:     --------
345:     splu : complete LU decomposition
346: 
347:     Notes
348:     -----
349:     To improve the better approximation to the inverse, you may need to
350:     increase `fill_factor` AND decrease `drop_tol`.
351: 
352:     This function uses the SuperLU library.
353: 
354:     Examples
355:     --------
356:     >>> from scipy.sparse import csc_matrix
357:     >>> from scipy.sparse.linalg import spilu
358:     >>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
359:     >>> B = spilu(A)
360:     >>> x = np.array([1., 2., 3.], dtype=float)
361:     >>> B.solve(x)
362:     array([ 1. , -3. , -1.5])
363:     >>> A.dot(B.solve(x))
364:     array([ 1.,  2.,  3.])
365:     >>> B.solve(A.dot(x))
366:     array([ 1.,  2.,  3.])
367:     '''
368:     if not isspmatrix_csc(A):
369:         A = csc_matrix(A)
370:         warn('splu requires CSC matrix format', SparseEfficiencyWarning)
371: 
372:     A.sort_indices()
373:     A = A.asfptype()  # upcast to a floating point format
374: 
375:     M, N = A.shape
376:     if (M != N):
377:         raise ValueError("can only factor square matrices")  # is this true?
378: 
379:     _options = dict(ILU_DropRule=drop_rule, ILU_DropTol=drop_tol,
380:                     ILU_FillFactor=fill_factor,
381:                     DiagPivotThresh=diag_pivot_thresh, ColPerm=permc_spec,
382:                     PanelSize=panel_size, Relax=relax)
383:     if options is not None:
384:         _options.update(options)
385:     return _superlu.gstrf(N, A.nnz, A.data, A.indices, A.indptr,
386:                           ilu=True, options=_options)
387: 
388: 
389: def factorized(A):
390:     '''
391:     Return a function for solving a sparse linear system, with A pre-factorized.
392: 
393:     Parameters
394:     ----------
395:     A : (N, N) array_like
396:         Input.
397: 
398:     Returns
399:     -------
400:     solve : callable
401:         To solve the linear system of equations given in `A`, the `solve`
402:         callable should be passed an ndarray of shape (N,).
403: 
404:     Examples
405:     --------
406:     >>> from scipy.sparse.linalg import factorized
407:     >>> A = np.array([[ 3. ,  2. , -1. ],
408:     ...               [ 2. , -2. ,  4. ],
409:     ...               [-1. ,  0.5, -1. ]])
410:     >>> solve = factorized(A) # Makes LU decomposition.
411:     >>> rhs1 = np.array([1, -2, 0])
412:     >>> solve(rhs1) # Uses the LU factors.
413:     array([ 1., -2., -2.])
414: 
415:     '''
416:     if useUmfpack:
417:         if noScikit:
418:             raise RuntimeError('Scikits.umfpack not installed.')
419: 
420:         if not isspmatrix_csc(A):
421:             A = csc_matrix(A)
422:             warn('splu requires CSC matrix format', SparseEfficiencyWarning)
423: 
424:         A = A.asfptype()  # upcast to a floating point format
425: 
426:         if A.dtype.char not in 'dD':
427:             raise ValueError("convert matrix data to double, please, using"
428:                   " .astype(), or set linsolve.useUmfpack = False")
429: 
430:         umf = umfpack.UmfpackContext(_get_umf_family(A))
431: 
432:         # Make LU decomposition.
433:         umf.numeric(A)
434: 
435:         def solve(b):
436:             return umf.solve(umfpack.UMFPACK_A, A, b, autoTranspose=True)
437: 
438:         return solve
439:     else:
440:         return splu(A).solve
441: 
442: 
443: def spsolve_triangular(A, b, lower=True, overwrite_A=False, overwrite_b=False):
444:     '''
445:     Solve the equation `A x = b` for `x`, assuming A is a triangular matrix.
446: 
447:     Parameters
448:     ----------
449:     A : (M, M) sparse matrix
450:         A sparse square triangular matrix. Should be in CSR format.
451:     b : (M,) or (M, N) array_like
452:         Right-hand side matrix in `A x = b`
453:     lower : bool, optional
454:         Whether `A` is a lower or upper triangular matrix.
455:         Default is lower triangular matrix.
456:     overwrite_A : bool, optional
457:         Allow changing `A`. The indices of `A` are going to be sorted and zero
458:         entries are going to be removed.
459:         Enabling gives a performance gain. Default is False.
460:     overwrite_b : bool, optional
461:         Allow overwriting data in `b`.
462:         Enabling gives a performance gain. Default is False.
463:         If `overwrite_b` is True, it should be ensured that
464:         `b` has an appropriate dtype to be able to store the result.
465: 
466:     Returns
467:     -------
468:     x : (M,) or (M, N) ndarray
469:         Solution to the system `A x = b`.  Shape of return matches shape of `b`.
470: 
471:     Raises
472:     ------
473:     LinAlgError
474:         If `A` is singular or not triangular.
475:     ValueError
476:         If shape of `A` or shape of `b` do not match the requirements.
477: 
478:     Notes
479:     -----
480:     .. versionadded:: 0.19.0
481: 
482:     Examples
483:     --------
484:     >>> from scipy.sparse import csr_matrix
485:     >>> from scipy.sparse.linalg import spsolve_triangular
486:     >>> A = csr_matrix([[3, 0, 0], [1, -1, 0], [2, 0, 1]], dtype=float)
487:     >>> B = np.array([[2, 0], [-1, 0], [2, 0]], dtype=float)
488:     >>> x = spsolve_triangular(A, B)
489:     >>> np.allclose(A.dot(x), B)
490:     True
491:     '''
492: 
493:     # Check the input for correct type and format.
494:     if not isspmatrix_csr(A):
495:         warn('CSR matrix format is required. Converting to CSR matrix.',
496:              SparseEfficiencyWarning)
497:         A = csr_matrix(A)
498:     elif not overwrite_A:
499:         A = A.copy()
500: 
501:     if A.shape[0] != A.shape[1]:
502:         raise ValueError(
503:             'A must be a square matrix but its shape is {}.'.format(A.shape))
504: 
505:     A.eliminate_zeros()
506:     A.sort_indices()
507: 
508:     b = np.asanyarray(b)
509: 
510:     if b.ndim not in [1, 2]:
511:         raise ValueError(
512:             'b must have 1 or 2 dims but its shape is {}.'.format(b.shape))
513:     if A.shape[0] != b.shape[0]:
514:         raise ValueError(
515:             'The size of the dimensions of A must be equal to '
516:             'the size of the first dimension of b but the shape of A is '
517:             '{} and the shape of b is {}.'.format(A.shape, b.shape))
518: 
519:     # Init x as (a copy of) b.
520:     x_dtype = np.result_type(A.data, b, np.float)
521:     if overwrite_b:
522:         if np.can_cast(b.dtype, x_dtype, casting='same_kind'):
523:             x = b
524:         else:
525:             raise ValueError(
526:                 'Cannot overwrite b (dtype {}) with result '
527:                 'of type {}.'.format(b.dtype, x_dtype))
528:     else:
529:         x = b.astype(x_dtype, copy=True)
530: 
531:     # Choose forward or backward order.
532:     if lower:
533:         row_indices = range(len(b))
534:     else:
535:         row_indices = range(len(b) - 1, -1, -1)
536: 
537:     # Fill x iteratively.
538:     for i in row_indices:
539: 
540:         # Get indices for i-th row.
541:         indptr_start = A.indptr[i]
542:         indptr_stop = A.indptr[i + 1]
543:         if lower:
544:             A_diagonal_index_row_i = indptr_stop - 1
545:             A_off_diagonal_indices_row_i = slice(indptr_start, indptr_stop - 1)
546:         else:
547:             A_diagonal_index_row_i = indptr_start
548:             A_off_diagonal_indices_row_i = slice(indptr_start + 1, indptr_stop)
549: 
550:         # Check regularity and triangularity of A.
551:         if indptr_stop <= indptr_start or A.indices[A_diagonal_index_row_i] < i:
552:             raise LinAlgError(
553:                 'A is singular: diagonal {} is zero.'.format(i))
554:         if A.indices[A_diagonal_index_row_i] > i:
555:             raise LinAlgError(
556:                 'A is not triangular: A[{}, {}] is nonzero.'
557:                 ''.format(i, A.indices[A_diagonal_index_row_i]))
558: 
559:         # Incorporate off-diagonal entries.
560:         A_column_indices_in_row_i = A.indices[A_off_diagonal_indices_row_i]
561:         A_values_in_row_i = A.data[A_off_diagonal_indices_row_i]
562:         x[i] -= np.dot(x[A_column_indices_in_row_i].T, A_values_in_row_i)
563: 
564:         # Compute i-th entry of x.
565:         x[i] /= A.data[A_diagonal_index_row_i]
566: 
567:     return x
568: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from warnings import warn' statement (line 3)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
import_391666 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_391666) is not StypyTypeError):

    if (import_391666 != 'pyd_module'):
        __import__(import_391666)
        sys_modules_391667 = sys.modules[import_391666]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_391667.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_391666)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy import asarray, empty, ravel, nonzero' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
import_391668 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_391668) is not StypyTypeError):

    if (import_391668 != 'pyd_module'):
        __import__(import_391668)
        sys_modules_391669 = sys.modules[import_391668]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', sys_modules_391669.module_type_store, module_type_store, ['asarray', 'empty', 'ravel', 'nonzero'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_391669, sys_modules_391669.module_type_store, module_type_store)
    else:
        from numpy import asarray, empty, ravel, nonzero

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', None, module_type_store, ['asarray', 'empty', 'ravel', 'nonzero'], [asarray, empty, ravel, nonzero])

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_391668)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.sparse import isspmatrix_csc, isspmatrix_csr, isspmatrix, SparseEfficiencyWarning, csc_matrix, csr_matrix' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
import_391670 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse')

if (type(import_391670) is not StypyTypeError):

    if (import_391670 != 'pyd_module'):
        __import__(import_391670)
        sys_modules_391671 = sys.modules[import_391670]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', sys_modules_391671.module_type_store, module_type_store, ['isspmatrix_csc', 'isspmatrix_csr', 'isspmatrix', 'SparseEfficiencyWarning', 'csc_matrix', 'csr_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_391671, sys_modules_391671.module_type_store, module_type_store)
    else:
        from scipy.sparse import isspmatrix_csc, isspmatrix_csr, isspmatrix, SparseEfficiencyWarning, csc_matrix, csr_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', None, module_type_store, ['isspmatrix_csc', 'isspmatrix_csr', 'isspmatrix', 'SparseEfficiencyWarning', 'csc_matrix', 'csr_matrix'], [isspmatrix_csc, isspmatrix_csr, isspmatrix, SparseEfficiencyWarning, csc_matrix, csr_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', import_391670)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.linalg import LinAlgError' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
import_391672 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg')

if (type(import_391672) is not StypyTypeError):

    if (import_391672 != 'pyd_module'):
        __import__(import_391672)
        sys_modules_391673 = sys.modules[import_391672]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', sys_modules_391673.module_type_store, module_type_store, ['LinAlgError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_391673, sys_modules_391673.module_type_store, module_type_store)
    else:
        from scipy.linalg import LinAlgError

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', None, module_type_store, ['LinAlgError'], [LinAlgError])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', import_391672)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.sparse.linalg.dsolve import _superlu' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
import_391674 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.dsolve')

if (type(import_391674) is not StypyTypeError):

    if (import_391674 != 'pyd_module'):
        __import__(import_391674)
        sys_modules_391675 = sys.modules[import_391674]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.dsolve', sys_modules_391675.module_type_store, module_type_store, ['_superlu'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_391675, sys_modules_391675.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.dsolve import _superlu

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.dsolve', None, module_type_store, ['_superlu'], [_superlu])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.dsolve' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.dsolve', import_391674)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')


# Assigning a Name to a Name (line 13):

# Assigning a Name to a Name (line 13):
# Getting the type of 'False' (line 13)
False_391676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'False')
# Assigning a type to the variable 'noScikit' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'noScikit', False_391676)


# SSA begins for try-except statement (line 14)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 4))

# 'import scikits.umfpack' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
import_391677 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'scikits.umfpack')

if (type(import_391677) is not StypyTypeError):

    if (import_391677 != 'pyd_module'):
        __import__(import_391677)
        sys_modules_391678 = sys.modules[import_391677]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'umfpack', sys_modules_391678.module_type_store, module_type_store)
    else:
        import scikits.umfpack as umfpack

        import_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'umfpack', scikits.umfpack, module_type_store)

else:
    # Assigning a type to the variable 'scikits.umfpack' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'scikits.umfpack', import_391677)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')

# SSA branch for the except part of a try statement (line 14)
# SSA branch for the except 'ImportError' branch of a try statement (line 14)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 17):

# Assigning a Name to a Name (line 17):
# Getting the type of 'True' (line 17)
True_391679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'True')
# Assigning a type to the variable 'noScikit' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'noScikit', True_391679)
# SSA join for try-except statement (line 14)
module_type_store = module_type_store.join_ssa_context()


# Assigning a UnaryOp to a Name (line 19):

# Assigning a UnaryOp to a Name (line 19):

# Getting the type of 'noScikit' (line 19)
noScikit_391680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 17), 'noScikit')
# Applying the 'not' unary operator (line 19)
result_not__391681 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 13), 'not', noScikit_391680)

# Assigning a type to the variable 'useUmfpack' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'useUmfpack', result_not__391681)

# Assigning a List to a Name (line 21):

# Assigning a List to a Name (line 21):
__all__ = ['use_solver', 'spsolve', 'splu', 'spilu', 'factorized', 'MatrixRankWarning', 'spsolve_triangular']
module_type_store.set_exportable_members(['use_solver', 'spsolve', 'splu', 'spilu', 'factorized', 'MatrixRankWarning', 'spsolve_triangular'])

# Obtaining an instance of the builtin type 'list' (line 21)
list_391682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_391683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'str', 'use_solver')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_391682, str_391683)
# Adding element type (line 21)
str_391684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'str', 'spsolve')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_391682, str_391684)
# Adding element type (line 21)
str_391685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 36), 'str', 'splu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_391682, str_391685)
# Adding element type (line 21)
str_391686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 44), 'str', 'spilu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_391682, str_391686)
# Adding element type (line 21)
str_391687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 53), 'str', 'factorized')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_391682, str_391687)
# Adding element type (line 21)
str_391688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'str', 'MatrixRankWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_391682, str_391688)
# Adding element type (line 21)
str_391689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'str', 'spsolve_triangular')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_391682, str_391689)

# Assigning a type to the variable '__all__' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), '__all__', list_391682)
# Declaration of the 'MatrixRankWarning' class
# Getting the type of 'UserWarning' (line 25)
UserWarning_391690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'UserWarning')

class MatrixRankWarning(UserWarning_391690, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 25, 0, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixRankWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MatrixRankWarning' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'MatrixRankWarning', MatrixRankWarning)

@norecursion
def use_solver(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'use_solver'
    module_type_store = module_type_store.open_function_context('use_solver', 29, 0, False)
    
    # Passed parameters checking function
    use_solver.stypy_localization = localization
    use_solver.stypy_type_of_self = None
    use_solver.stypy_type_store = module_type_store
    use_solver.stypy_function_name = 'use_solver'
    use_solver.stypy_param_names_list = []
    use_solver.stypy_varargs_param_name = None
    use_solver.stypy_kwargs_param_name = 'kwargs'
    use_solver.stypy_call_defaults = defaults
    use_solver.stypy_call_varargs = varargs
    use_solver.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'use_solver', [], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'use_solver', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'use_solver(...)' code ##################

    str_391691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', '\n    Select default sparse direct solver to be used.\n\n    Parameters\n    ----------\n    useUmfpack : bool, optional\n        Use UMFPACK over SuperLU. Has effect only if scikits.umfpack is\n        installed. Default: True\n    assumeSortedIndices : bool, optional\n        Allow UMFPACK to skip the step of sorting indices for a CSR/CSC matrix.\n        Has effect only if useUmfpack is True and scikits.umfpack is installed.\n        Default: False\n\n    Notes\n    -----\n    The default sparse solver is umfpack when available\n    (scikits.umfpack is installed). This can be changed by passing\n    useUmfpack = False, which then causes the always present SuperLU\n    based solver to be used.\n\n    Umfpack requires a CSR/CSC matrix to have sorted column/row indices. If\n    sure that the matrix fulfills this, pass ``assumeSortedIndices=True``\n    to gain some speed.\n\n    ')
    
    
    str_391692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 7), 'str', 'useUmfpack')
    # Getting the type of 'kwargs' (line 55)
    kwargs_391693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'kwargs')
    # Applying the binary operator 'in' (line 55)
    result_contains_391694 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 7), 'in', str_391692, kwargs_391693)
    
    # Testing the type of an if condition (line 55)
    if_condition_391695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 4), result_contains_391694)
    # Assigning a type to the variable 'if_condition_391695' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'if_condition_391695', if_condition_391695)
    # SSA begins for if statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 56):
    
    # Assigning a Subscript to a Subscript (line 56):
    
    # Obtaining the type of the subscript
    str_391696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 41), 'str', 'useUmfpack')
    # Getting the type of 'kwargs' (line 56)
    kwargs_391697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 34), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___391698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 34), kwargs_391697, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_391699 = invoke(stypy.reporting.localization.Localization(__file__, 56, 34), getitem___391698, str_391696)
    
    
    # Call to globals(...): (line 56)
    # Processing the call keyword arguments (line 56)
    kwargs_391701 = {}
    # Getting the type of 'globals' (line 56)
    globals_391700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'globals', False)
    # Calling globals(args, kwargs) (line 56)
    globals_call_result_391702 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), globals_391700, *[], **kwargs_391701)
    
    str_391703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 18), 'str', 'useUmfpack')
    # Storing an element on a container (line 56)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 8), globals_call_result_391702, (str_391703, subscript_call_result_391699))
    # SSA join for if statement (line 55)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'useUmfpack' (line 57)
    useUmfpack_391704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 7), 'useUmfpack')
    
    str_391705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'str', 'assumeSortedIndices')
    # Getting the type of 'kwargs' (line 57)
    kwargs_391706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 47), 'kwargs')
    # Applying the binary operator 'in' (line 57)
    result_contains_391707 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 22), 'in', str_391705, kwargs_391706)
    
    # Applying the binary operator 'and' (line 57)
    result_and_keyword_391708 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 7), 'and', useUmfpack_391704, result_contains_391707)
    
    # Testing the type of an if condition (line 57)
    if_condition_391709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 4), result_and_keyword_391708)
    # Assigning a type to the variable 'if_condition_391709' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'if_condition_391709', if_condition_391709)
    # SSA begins for if statement (line 57)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to configure(...): (line 58)
    # Processing the call keyword arguments (line 58)
    
    # Obtaining the type of the subscript
    str_391712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 53), 'str', 'assumeSortedIndices')
    # Getting the type of 'kwargs' (line 58)
    kwargs_391713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 46), 'kwargs', False)
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___391714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 46), kwargs_391713, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 58)
    subscript_call_result_391715 = invoke(stypy.reporting.localization.Localization(__file__, 58, 46), getitem___391714, str_391712)
    
    keyword_391716 = subscript_call_result_391715
    kwargs_391717 = {'assumeSortedIndices': keyword_391716}
    # Getting the type of 'umfpack' (line 58)
    umfpack_391710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'umfpack', False)
    # Obtaining the member 'configure' of a type (line 58)
    configure_391711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), umfpack_391710, 'configure')
    # Calling configure(args, kwargs) (line 58)
    configure_call_result_391718 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), configure_391711, *[], **kwargs_391717)
    
    # SSA join for if statement (line 57)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'use_solver(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'use_solver' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_391719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_391719)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'use_solver'
    return stypy_return_type_391719

# Assigning a type to the variable 'use_solver' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'use_solver', use_solver)

@norecursion
def _get_umf_family(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_umf_family'
    module_type_store = module_type_store.open_function_context('_get_umf_family', 60, 0, False)
    
    # Passed parameters checking function
    _get_umf_family.stypy_localization = localization
    _get_umf_family.stypy_type_of_self = None
    _get_umf_family.stypy_type_store = module_type_store
    _get_umf_family.stypy_function_name = '_get_umf_family'
    _get_umf_family.stypy_param_names_list = ['A']
    _get_umf_family.stypy_varargs_param_name = None
    _get_umf_family.stypy_kwargs_param_name = None
    _get_umf_family.stypy_call_defaults = defaults
    _get_umf_family.stypy_call_varargs = varargs
    _get_umf_family.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_umf_family', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_umf_family', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_umf_family(...)' code ##################

    str_391720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'str', 'Get umfpack family string given the sparse matrix dtype.')
    
    # Assigning a Dict to a Name (line 62):
    
    # Assigning a Dict to a Name (line 62):
    
    # Obtaining an instance of the builtin type 'dict' (line 62)
    dict_391721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 62)
    # Adding element type (key, value) (line 62)
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_391722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    # Getting the type of 'np' (line 63)
    np_391723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 9), 'np')
    # Obtaining the member 'float64' of a type (line 63)
    float64_391724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 9), np_391723, 'float64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 9), tuple_391722, float64_391724)
    # Adding element type (line 63)
    # Getting the type of 'np' (line 63)
    np_391725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 21), 'np')
    # Obtaining the member 'int32' of a type (line 63)
    int32_391726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 21), np_391725, 'int32')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 9), tuple_391722, int32_391726)
    
    str_391727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 32), 'str', 'di')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 16), dict_391721, (tuple_391722, str_391727))
    # Adding element type (key, value) (line 62)
    
    # Obtaining an instance of the builtin type 'tuple' (line 64)
    tuple_391728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 64)
    # Adding element type (line 64)
    # Getting the type of 'np' (line 64)
    np_391729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 9), 'np')
    # Obtaining the member 'complex128' of a type (line 64)
    complex128_391730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 9), np_391729, 'complex128')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 9), tuple_391728, complex128_391730)
    # Adding element type (line 64)
    # Getting the type of 'np' (line 64)
    np_391731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'np')
    # Obtaining the member 'int32' of a type (line 64)
    int32_391732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 24), np_391731, 'int32')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 9), tuple_391728, int32_391732)
    
    str_391733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 35), 'str', 'zi')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 16), dict_391721, (tuple_391728, str_391733))
    # Adding element type (key, value) (line 62)
    
    # Obtaining an instance of the builtin type 'tuple' (line 65)
    tuple_391734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 65)
    # Adding element type (line 65)
    # Getting the type of 'np' (line 65)
    np_391735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 9), 'np')
    # Obtaining the member 'float64' of a type (line 65)
    float64_391736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 9), np_391735, 'float64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_391734, float64_391736)
    # Adding element type (line 65)
    # Getting the type of 'np' (line 65)
    np_391737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'np')
    # Obtaining the member 'int64' of a type (line 65)
    int64_391738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 21), np_391737, 'int64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_391734, int64_391738)
    
    str_391739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 32), 'str', 'dl')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 16), dict_391721, (tuple_391734, str_391739))
    # Adding element type (key, value) (line 62)
    
    # Obtaining an instance of the builtin type 'tuple' (line 66)
    tuple_391740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 66)
    # Adding element type (line 66)
    # Getting the type of 'np' (line 66)
    np_391741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 9), 'np')
    # Obtaining the member 'complex128' of a type (line 66)
    complex128_391742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 9), np_391741, 'complex128')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 9), tuple_391740, complex128_391742)
    # Adding element type (line 66)
    # Getting the type of 'np' (line 66)
    np_391743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'np')
    # Obtaining the member 'int64' of a type (line 66)
    int64_391744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 24), np_391743, 'int64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 9), tuple_391740, int64_391744)
    
    str_391745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 35), 'str', 'zl')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 16), dict_391721, (tuple_391740, str_391745))
    
    # Assigning a type to the variable '_families' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), '_families', dict_391721)
    
    # Assigning a Subscript to a Name (line 69):
    
    # Assigning a Subscript to a Name (line 69):
    
    # Obtaining the type of the subscript
    # Getting the type of 'A' (line 69)
    A_391746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'A')
    # Obtaining the member 'dtype' of a type (line 69)
    dtype_391747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 27), A_391746, 'dtype')
    # Obtaining the member 'name' of a type (line 69)
    name_391748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 27), dtype_391747, 'name')
    # Getting the type of 'np' (line 69)
    np_391749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'np')
    # Obtaining the member 'sctypeDict' of a type (line 69)
    sctypeDict_391750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), np_391749, 'sctypeDict')
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___391751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), sctypeDict_391750, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_391752 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), getitem___391751, name_391748)
    
    # Assigning a type to the variable 'f_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'f_type', subscript_call_result_391752)
    
    # Assigning a Subscript to a Name (line 70):
    
    # Assigning a Subscript to a Name (line 70):
    
    # Obtaining the type of the subscript
    # Getting the type of 'A' (line 70)
    A_391753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 27), 'A')
    # Obtaining the member 'indices' of a type (line 70)
    indices_391754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 27), A_391753, 'indices')
    # Obtaining the member 'dtype' of a type (line 70)
    dtype_391755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 27), indices_391754, 'dtype')
    # Obtaining the member 'name' of a type (line 70)
    name_391756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 27), dtype_391755, 'name')
    # Getting the type of 'np' (line 70)
    np_391757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 13), 'np')
    # Obtaining the member 'sctypeDict' of a type (line 70)
    sctypeDict_391758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 13), np_391757, 'sctypeDict')
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___391759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 13), sctypeDict_391758, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_391760 = invoke(stypy.reporting.localization.Localization(__file__, 70, 13), getitem___391759, name_391756)
    
    # Assigning a type to the variable 'i_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'i_type', subscript_call_result_391760)
    
    
    # SSA begins for try-except statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 73):
    
    # Assigning a Subscript to a Name (line 73):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 73)
    tuple_391761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 73)
    # Adding element type (line 73)
    # Getting the type of 'f_type' (line 73)
    f_type_391762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 28), 'f_type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), tuple_391761, f_type_391762)
    # Adding element type (line 73)
    # Getting the type of 'i_type' (line 73)
    i_type_391763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 36), 'i_type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), tuple_391761, i_type_391763)
    
    # Getting the type of '_families' (line 73)
    _families_391764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), '_families')
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___391765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 17), _families_391764, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
    subscript_call_result_391766 = invoke(stypy.reporting.localization.Localization(__file__, 73, 17), getitem___391765, tuple_391761)
    
    # Assigning a type to the variable 'family' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'family', subscript_call_result_391766)
    # SSA branch for the except part of a try statement (line 72)
    # SSA branch for the except 'KeyError' branch of a try statement (line 72)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a BinOp to a Name (line 76):
    
    # Assigning a BinOp to a Name (line 76):
    str_391767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 14), 'str', 'only float64 or complex128 matrices with int32 or int64 indices are supported! (got: matrix: %s, indices: %s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_391768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    # Getting the type of 'f_type' (line 78)
    f_type_391769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'f_type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 15), tuple_391768, f_type_391769)
    # Adding element type (line 78)
    # Getting the type of 'i_type' (line 78)
    i_type_391770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'i_type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 15), tuple_391768, i_type_391770)
    
    # Applying the binary operator '%' (line 76)
    result_mod_391771 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 14), '%', str_391767, tuple_391768)
    
    # Assigning a type to the variable 'msg' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'msg', result_mod_391771)
    
    # Call to ValueError(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'msg' (line 79)
    msg_391773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'msg', False)
    # Processing the call keyword arguments (line 79)
    kwargs_391774 = {}
    # Getting the type of 'ValueError' (line 79)
    ValueError_391772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 79)
    ValueError_call_result_391775 = invoke(stypy.reporting.localization.Localization(__file__, 79, 14), ValueError_391772, *[msg_391773], **kwargs_391774)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 79, 8), ValueError_call_result_391775, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 72)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'family' (line 81)
    family_391776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'family')
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type', family_391776)
    
    # ################# End of '_get_umf_family(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_umf_family' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_391777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_391777)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_umf_family'
    return stypy_return_type_391777

# Assigning a type to the variable '_get_umf_family' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), '_get_umf_family', _get_umf_family)

@norecursion
def spsolve(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 83)
    None_391778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 29), 'None')
    # Getting the type of 'True' (line 83)
    True_391779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 47), 'True')
    defaults = [None_391778, True_391779]
    # Create a new context for function 'spsolve'
    module_type_store = module_type_store.open_function_context('spsolve', 83, 0, False)
    
    # Passed parameters checking function
    spsolve.stypy_localization = localization
    spsolve.stypy_type_of_self = None
    spsolve.stypy_type_store = module_type_store
    spsolve.stypy_function_name = 'spsolve'
    spsolve.stypy_param_names_list = ['A', 'b', 'permc_spec', 'use_umfpack']
    spsolve.stypy_varargs_param_name = None
    spsolve.stypy_kwargs_param_name = None
    spsolve.stypy_call_defaults = defaults
    spsolve.stypy_call_varargs = varargs
    spsolve.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spsolve', ['A', 'b', 'permc_spec', 'use_umfpack'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spsolve', localization, ['A', 'b', 'permc_spec', 'use_umfpack'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spsolve(...)' code ##################

    str_391780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, (-1)), 'str', "Solve the sparse linear system Ax=b, where b may be a vector or a matrix.\n\n    Parameters\n    ----------\n    A : ndarray or sparse matrix\n        The square matrix A will be converted into CSC or CSR form\n    b : ndarray or sparse matrix\n        The matrix or vector representing the right hand side of the equation.\n        If a vector, b.shape must be (n,) or (n, 1).\n    permc_spec : str, optional\n        How to permute the columns of the matrix for sparsity preservation.\n        (default: 'COLAMD')\n\n        - ``NATURAL``: natural ordering.\n        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.\n        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.\n        - ``COLAMD``: approximate minimum degree column ordering\n    use_umfpack : bool, optional\n        if True (default) then use umfpack for the solution.  This is\n        only referenced if b is a vector and ``scikit-umfpack`` is installed.\n\n    Returns\n    -------\n    x : ndarray or sparse matrix\n        the solution of the sparse linear equation.\n        If b is a vector, then x is a vector of size A.shape[1]\n        If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])\n\n    Notes\n    -----\n    For solving the matrix expression AX = B, this solver assumes the resulting\n    matrix X is sparse, as is often the case for very sparse inputs.  If the\n    resulting X is dense, the construction of this sparse result will be\n    relatively expensive.  In that case, consider converting A to a dense\n    matrix and using scipy.linalg.solve or its variants.\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import spsolve\n    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)\n    >>> B = csc_matrix([[2, 0], [-1, 0], [2, 0]], dtype=float)\n    >>> x = spsolve(A, B)\n    >>> np.allclose(A.dot(x).todense(), B.todense())\n    True\n    ")
    
    
    
    # Evaluating a boolean operation
    
    # Call to isspmatrix_csc(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'A' (line 130)
    A_391782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 27), 'A', False)
    # Processing the call keyword arguments (line 130)
    kwargs_391783 = {}
    # Getting the type of 'isspmatrix_csc' (line 130)
    isspmatrix_csc_391781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'isspmatrix_csc', False)
    # Calling isspmatrix_csc(args, kwargs) (line 130)
    isspmatrix_csc_call_result_391784 = invoke(stypy.reporting.localization.Localization(__file__, 130, 12), isspmatrix_csc_391781, *[A_391782], **kwargs_391783)
    
    
    # Call to isspmatrix_csr(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'A' (line 130)
    A_391786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 48), 'A', False)
    # Processing the call keyword arguments (line 130)
    kwargs_391787 = {}
    # Getting the type of 'isspmatrix_csr' (line 130)
    isspmatrix_csr_391785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 33), 'isspmatrix_csr', False)
    # Calling isspmatrix_csr(args, kwargs) (line 130)
    isspmatrix_csr_call_result_391788 = invoke(stypy.reporting.localization.Localization(__file__, 130, 33), isspmatrix_csr_391785, *[A_391786], **kwargs_391787)
    
    # Applying the binary operator 'or' (line 130)
    result_or_keyword_391789 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 12), 'or', isspmatrix_csc_call_result_391784, isspmatrix_csr_call_result_391788)
    
    # Applying the 'not' unary operator (line 130)
    result_not__391790 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 7), 'not', result_or_keyword_391789)
    
    # Testing the type of an if condition (line 130)
    if_condition_391791 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 4), result_not__391790)
    # Assigning a type to the variable 'if_condition_391791' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'if_condition_391791', if_condition_391791)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to csc_matrix(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'A' (line 131)
    A_391793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 23), 'A', False)
    # Processing the call keyword arguments (line 131)
    kwargs_391794 = {}
    # Getting the type of 'csc_matrix' (line 131)
    csc_matrix_391792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 131)
    csc_matrix_call_result_391795 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), csc_matrix_391792, *[A_391793], **kwargs_391794)
    
    # Assigning a type to the variable 'A' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'A', csc_matrix_call_result_391795)
    
    # Call to warn(...): (line 132)
    # Processing the call arguments (line 132)
    str_391797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 13), 'str', 'spsolve requires A be CSC or CSR matrix format')
    # Getting the type of 'SparseEfficiencyWarning' (line 133)
    SparseEfficiencyWarning_391798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'SparseEfficiencyWarning', False)
    # Processing the call keyword arguments (line 132)
    kwargs_391799 = {}
    # Getting the type of 'warn' (line 132)
    warn_391796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 132)
    warn_call_result_391800 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), warn_391796, *[str_391797, SparseEfficiencyWarning_391798], **kwargs_391799)
    
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 136):
    
    # Assigning a Call to a Name (line 136):
    
    # Call to isspmatrix(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'b' (line 136)
    b_391802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 29), 'b', False)
    # Processing the call keyword arguments (line 136)
    kwargs_391803 = {}
    # Getting the type of 'isspmatrix' (line 136)
    isspmatrix_391801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 18), 'isspmatrix', False)
    # Calling isspmatrix(args, kwargs) (line 136)
    isspmatrix_call_result_391804 = invoke(stypy.reporting.localization.Localization(__file__, 136, 18), isspmatrix_391801, *[b_391802], **kwargs_391803)
    
    # Assigning a type to the variable 'b_is_sparse' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'b_is_sparse', isspmatrix_call_result_391804)
    
    
    # Getting the type of 'b_is_sparse' (line 137)
    b_is_sparse_391805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'b_is_sparse')
    # Applying the 'not' unary operator (line 137)
    result_not__391806 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 7), 'not', b_is_sparse_391805)
    
    # Testing the type of an if condition (line 137)
    if_condition_391807 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 4), result_not__391806)
    # Assigning a type to the variable 'if_condition_391807' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'if_condition_391807', if_condition_391807)
    # SSA begins for if statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 138):
    
    # Assigning a Call to a Name (line 138):
    
    # Call to asarray(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'b' (line 138)
    b_391809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'b', False)
    # Processing the call keyword arguments (line 138)
    kwargs_391810 = {}
    # Getting the type of 'asarray' (line 138)
    asarray_391808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'asarray', False)
    # Calling asarray(args, kwargs) (line 138)
    asarray_call_result_391811 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), asarray_391808, *[b_391809], **kwargs_391810)
    
    # Assigning a type to the variable 'b' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'b', asarray_call_result_391811)
    # SSA join for if statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 139):
    
    # Assigning a BoolOp to a Name (line 139):
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 139)
    b_391812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'b')
    # Obtaining the member 'ndim' of a type (line 139)
    ndim_391813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 20), b_391812, 'ndim')
    int_391814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 30), 'int')
    # Applying the binary operator '==' (line 139)
    result_eq_391815 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 20), '==', ndim_391813, int_391814)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 139)
    b_391816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 37), 'b')
    # Obtaining the member 'ndim' of a type (line 139)
    ndim_391817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 37), b_391816, 'ndim')
    int_391818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 47), 'int')
    # Applying the binary operator '==' (line 139)
    result_eq_391819 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 37), '==', ndim_391817, int_391818)
    
    
    
    # Obtaining the type of the subscript
    int_391820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 61), 'int')
    # Getting the type of 'b' (line 139)
    b_391821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 53), 'b')
    # Obtaining the member 'shape' of a type (line 139)
    shape_391822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 53), b_391821, 'shape')
    # Obtaining the member '__getitem__' of a type (line 139)
    getitem___391823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 53), shape_391822, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 139)
    subscript_call_result_391824 = invoke(stypy.reporting.localization.Localization(__file__, 139, 53), getitem___391823, int_391820)
    
    int_391825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 67), 'int')
    # Applying the binary operator '==' (line 139)
    result_eq_391826 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 53), '==', subscript_call_result_391824, int_391825)
    
    # Applying the binary operator 'and' (line 139)
    result_and_keyword_391827 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 37), 'and', result_eq_391819, result_eq_391826)
    
    # Applying the binary operator 'or' (line 139)
    result_or_keyword_391828 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 19), 'or', result_eq_391815, result_and_keyword_391827)
    
    # Assigning a type to the variable 'b_is_vector' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'b_is_vector', result_or_keyword_391828)
    
    # Call to sort_indices(...): (line 141)
    # Processing the call keyword arguments (line 141)
    kwargs_391831 = {}
    # Getting the type of 'A' (line 141)
    A_391829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'A', False)
    # Obtaining the member 'sort_indices' of a type (line 141)
    sort_indices_391830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 4), A_391829, 'sort_indices')
    # Calling sort_indices(args, kwargs) (line 141)
    sort_indices_call_result_391832 = invoke(stypy.reporting.localization.Localization(__file__, 141, 4), sort_indices_391830, *[], **kwargs_391831)
    
    
    # Assigning a Call to a Name (line 142):
    
    # Assigning a Call to a Name (line 142):
    
    # Call to asfptype(...): (line 142)
    # Processing the call keyword arguments (line 142)
    kwargs_391835 = {}
    # Getting the type of 'A' (line 142)
    A_391833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'A', False)
    # Obtaining the member 'asfptype' of a type (line 142)
    asfptype_391834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), A_391833, 'asfptype')
    # Calling asfptype(args, kwargs) (line 142)
    asfptype_call_result_391836 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), asfptype_391834, *[], **kwargs_391835)
    
    # Assigning a type to the variable 'A' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'A', asfptype_call_result_391836)
    
    # Assigning a Call to a Name (line 143):
    
    # Assigning a Call to a Name (line 143):
    
    # Call to promote_types(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'A' (line 143)
    A_391839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 36), 'A', False)
    # Obtaining the member 'dtype' of a type (line 143)
    dtype_391840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 36), A_391839, 'dtype')
    # Getting the type of 'b' (line 143)
    b_391841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 45), 'b', False)
    # Obtaining the member 'dtype' of a type (line 143)
    dtype_391842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 45), b_391841, 'dtype')
    # Processing the call keyword arguments (line 143)
    kwargs_391843 = {}
    # Getting the type of 'np' (line 143)
    np_391837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 19), 'np', False)
    # Obtaining the member 'promote_types' of a type (line 143)
    promote_types_391838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 19), np_391837, 'promote_types')
    # Calling promote_types(args, kwargs) (line 143)
    promote_types_call_result_391844 = invoke(stypy.reporting.localization.Localization(__file__, 143, 19), promote_types_391838, *[dtype_391840, dtype_391842], **kwargs_391843)
    
    # Assigning a type to the variable 'result_dtype' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'result_dtype', promote_types_call_result_391844)
    
    
    # Getting the type of 'A' (line 144)
    A_391845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 7), 'A')
    # Obtaining the member 'dtype' of a type (line 144)
    dtype_391846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 7), A_391845, 'dtype')
    # Getting the type of 'result_dtype' (line 144)
    result_dtype_391847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 18), 'result_dtype')
    # Applying the binary operator '!=' (line 144)
    result_ne_391848 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 7), '!=', dtype_391846, result_dtype_391847)
    
    # Testing the type of an if condition (line 144)
    if_condition_391849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 4), result_ne_391848)
    # Assigning a type to the variable 'if_condition_391849' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'if_condition_391849', if_condition_391849)
    # SSA begins for if statement (line 144)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 145):
    
    # Assigning a Call to a Name (line 145):
    
    # Call to astype(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'result_dtype' (line 145)
    result_dtype_391852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 21), 'result_dtype', False)
    # Processing the call keyword arguments (line 145)
    kwargs_391853 = {}
    # Getting the type of 'A' (line 145)
    A_391850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'A', False)
    # Obtaining the member 'astype' of a type (line 145)
    astype_391851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), A_391850, 'astype')
    # Calling astype(args, kwargs) (line 145)
    astype_call_result_391854 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), astype_391851, *[result_dtype_391852], **kwargs_391853)
    
    # Assigning a type to the variable 'A' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'A', astype_call_result_391854)
    # SSA join for if statement (line 144)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'b' (line 146)
    b_391855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 7), 'b')
    # Obtaining the member 'dtype' of a type (line 146)
    dtype_391856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 7), b_391855, 'dtype')
    # Getting the type of 'result_dtype' (line 146)
    result_dtype_391857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'result_dtype')
    # Applying the binary operator '!=' (line 146)
    result_ne_391858 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 7), '!=', dtype_391856, result_dtype_391857)
    
    # Testing the type of an if condition (line 146)
    if_condition_391859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 4), result_ne_391858)
    # Assigning a type to the variable 'if_condition_391859' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'if_condition_391859', if_condition_391859)
    # SSA begins for if statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 147):
    
    # Call to astype(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'result_dtype' (line 147)
    result_dtype_391862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'result_dtype', False)
    # Processing the call keyword arguments (line 147)
    kwargs_391863 = {}
    # Getting the type of 'b' (line 147)
    b_391860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'b', False)
    # Obtaining the member 'astype' of a type (line 147)
    astype_391861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), b_391860, 'astype')
    # Calling astype(args, kwargs) (line 147)
    astype_call_result_391864 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), astype_391861, *[result_dtype_391862], **kwargs_391863)
    
    # Assigning a type to the variable 'b' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'b', astype_call_result_391864)
    # SSA join for if statement (line 146)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 150):
    
    # Assigning a Subscript to a Name (line 150):
    
    # Obtaining the type of the subscript
    int_391865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 4), 'int')
    # Getting the type of 'A' (line 150)
    A_391866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'A')
    # Obtaining the member 'shape' of a type (line 150)
    shape_391867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 11), A_391866, 'shape')
    # Obtaining the member '__getitem__' of a type (line 150)
    getitem___391868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 4), shape_391867, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 150)
    subscript_call_result_391869 = invoke(stypy.reporting.localization.Localization(__file__, 150, 4), getitem___391868, int_391865)
    
    # Assigning a type to the variable 'tuple_var_assignment_391658' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_var_assignment_391658', subscript_call_result_391869)
    
    # Assigning a Subscript to a Name (line 150):
    
    # Obtaining the type of the subscript
    int_391870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 4), 'int')
    # Getting the type of 'A' (line 150)
    A_391871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'A')
    # Obtaining the member 'shape' of a type (line 150)
    shape_391872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 11), A_391871, 'shape')
    # Obtaining the member '__getitem__' of a type (line 150)
    getitem___391873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 4), shape_391872, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 150)
    subscript_call_result_391874 = invoke(stypy.reporting.localization.Localization(__file__, 150, 4), getitem___391873, int_391870)
    
    # Assigning a type to the variable 'tuple_var_assignment_391659' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_var_assignment_391659', subscript_call_result_391874)
    
    # Assigning a Name to a Name (line 150):
    # Getting the type of 'tuple_var_assignment_391658' (line 150)
    tuple_var_assignment_391658_391875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_var_assignment_391658')
    # Assigning a type to the variable 'M' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'M', tuple_var_assignment_391658_391875)
    
    # Assigning a Name to a Name (line 150):
    # Getting the type of 'tuple_var_assignment_391659' (line 150)
    tuple_var_assignment_391659_391876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_var_assignment_391659')
    # Assigning a type to the variable 'N' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 7), 'N', tuple_var_assignment_391659_391876)
    
    
    # Getting the type of 'M' (line 151)
    M_391877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'M')
    # Getting the type of 'N' (line 151)
    N_391878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'N')
    # Applying the binary operator '!=' (line 151)
    result_ne_391879 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 8), '!=', M_391877, N_391878)
    
    # Testing the type of an if condition (line 151)
    if_condition_391880 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 4), result_ne_391879)
    # Assigning a type to the variable 'if_condition_391880' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'if_condition_391880', if_condition_391880)
    # SSA begins for if statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 152)
    # Processing the call arguments (line 152)
    str_391882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 25), 'str', 'matrix must be square (has shape %s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 152)
    tuple_391883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 67), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 152)
    # Adding element type (line 152)
    
    # Obtaining an instance of the builtin type 'tuple' (line 152)
    tuple_391884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 68), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 152)
    # Adding element type (line 152)
    # Getting the type of 'M' (line 152)
    M_391885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 68), 'M', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 68), tuple_391884, M_391885)
    # Adding element type (line 152)
    # Getting the type of 'N' (line 152)
    N_391886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 71), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 68), tuple_391884, N_391886)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 67), tuple_391883, tuple_391884)
    
    # Applying the binary operator '%' (line 152)
    result_mod_391887 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 25), '%', str_391882, tuple_391883)
    
    # Processing the call keyword arguments (line 152)
    kwargs_391888 = {}
    # Getting the type of 'ValueError' (line 152)
    ValueError_391881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 152)
    ValueError_call_result_391889 = invoke(stypy.reporting.localization.Localization(__file__, 152, 14), ValueError_391881, *[result_mod_391887], **kwargs_391888)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 152, 8), ValueError_call_result_391889, 'raise parameter', BaseException)
    # SSA join for if statement (line 151)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'M' (line 154)
    M_391890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 7), 'M')
    
    # Obtaining the type of the subscript
    int_391891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'int')
    # Getting the type of 'b' (line 154)
    b_391892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'b')
    # Obtaining the member 'shape' of a type (line 154)
    shape_391893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), b_391892, 'shape')
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___391894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), shape_391893, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_391895 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), getitem___391894, int_391891)
    
    # Applying the binary operator '!=' (line 154)
    result_ne_391896 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 7), '!=', M_391890, subscript_call_result_391895)
    
    # Testing the type of an if condition (line 154)
    if_condition_391897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 4), result_ne_391896)
    # Assigning a type to the variable 'if_condition_391897' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'if_condition_391897', if_condition_391897)
    # SSA begins for if statement (line 154)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 155)
    # Processing the call arguments (line 155)
    str_391899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 25), 'str', 'matrix - rhs dimension mismatch (%s - %s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 156)
    tuple_391900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 156)
    # Adding element type (line 156)
    # Getting the type of 'A' (line 156)
    A_391901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 28), 'A', False)
    # Obtaining the member 'shape' of a type (line 156)
    shape_391902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 28), A_391901, 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 28), tuple_391900, shape_391902)
    # Adding element type (line 156)
    
    # Obtaining the type of the subscript
    int_391903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 45), 'int')
    # Getting the type of 'b' (line 156)
    b_391904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'b', False)
    # Obtaining the member 'shape' of a type (line 156)
    shape_391905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 37), b_391904, 'shape')
    # Obtaining the member '__getitem__' of a type (line 156)
    getitem___391906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 37), shape_391905, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 156)
    subscript_call_result_391907 = invoke(stypy.reporting.localization.Localization(__file__, 156, 37), getitem___391906, int_391903)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 28), tuple_391900, subscript_call_result_391907)
    
    # Applying the binary operator '%' (line 155)
    result_mod_391908 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 25), '%', str_391899, tuple_391900)
    
    # Processing the call keyword arguments (line 155)
    kwargs_391909 = {}
    # Getting the type of 'ValueError' (line 155)
    ValueError_391898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 155)
    ValueError_call_result_391910 = invoke(stypy.reporting.localization.Localization(__file__, 155, 14), ValueError_391898, *[result_mod_391908], **kwargs_391909)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 155, 8), ValueError_call_result_391910, 'raise parameter', BaseException)
    # SSA join for if statement (line 154)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 158):
    
    # Assigning a BoolOp to a Name (line 158):
    
    # Evaluating a boolean operation
    # Getting the type of 'use_umfpack' (line 158)
    use_umfpack_391911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 18), 'use_umfpack')
    # Getting the type of 'useUmfpack' (line 158)
    useUmfpack_391912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 34), 'useUmfpack')
    # Applying the binary operator 'and' (line 158)
    result_and_keyword_391913 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 18), 'and', use_umfpack_391911, useUmfpack_391912)
    
    # Assigning a type to the variable 'use_umfpack' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'use_umfpack', result_and_keyword_391913)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'b_is_vector' (line 160)
    b_is_vector_391914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 7), 'b_is_vector')
    # Getting the type of 'use_umfpack' (line 160)
    use_umfpack_391915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 23), 'use_umfpack')
    # Applying the binary operator 'and' (line 160)
    result_and_keyword_391916 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 7), 'and', b_is_vector_391914, use_umfpack_391915)
    
    # Testing the type of an if condition (line 160)
    if_condition_391917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 4), result_and_keyword_391916)
    # Assigning a type to the variable 'if_condition_391917' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'if_condition_391917', if_condition_391917)
    # SSA begins for if statement (line 160)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'b_is_sparse' (line 161)
    b_is_sparse_391918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'b_is_sparse')
    # Testing the type of an if condition (line 161)
    if_condition_391919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 8), b_is_sparse_391918)
    # Assigning a type to the variable 'if_condition_391919' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'if_condition_391919', if_condition_391919)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 162):
    
    # Assigning a Call to a Name (line 162):
    
    # Call to toarray(...): (line 162)
    # Processing the call keyword arguments (line 162)
    kwargs_391922 = {}
    # Getting the type of 'b' (line 162)
    b_391920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'b', False)
    # Obtaining the member 'toarray' of a type (line 162)
    toarray_391921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 20), b_391920, 'toarray')
    # Calling toarray(args, kwargs) (line 162)
    toarray_call_result_391923 = invoke(stypy.reporting.localization.Localization(__file__, 162, 20), toarray_391921, *[], **kwargs_391922)
    
    # Assigning a type to the variable 'b_vec' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'b_vec', toarray_call_result_391923)
    # SSA branch for the else part of an if statement (line 161)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 164):
    
    # Assigning a Name to a Name (line 164):
    # Getting the type of 'b' (line 164)
    b_391924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'b')
    # Assigning a type to the variable 'b_vec' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'b_vec', b_391924)
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 165):
    
    # Assigning a Call to a Name (line 165):
    
    # Call to ravel(...): (line 165)
    # Processing the call keyword arguments (line 165)
    kwargs_391933 = {}
    
    # Call to asarray(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'b_vec' (line 165)
    b_vec_391926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 24), 'b_vec', False)
    # Processing the call keyword arguments (line 165)
    # Getting the type of 'A' (line 165)
    A_391927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 37), 'A', False)
    # Obtaining the member 'dtype' of a type (line 165)
    dtype_391928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 37), A_391927, 'dtype')
    keyword_391929 = dtype_391928
    kwargs_391930 = {'dtype': keyword_391929}
    # Getting the type of 'asarray' (line 165)
    asarray_391925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'asarray', False)
    # Calling asarray(args, kwargs) (line 165)
    asarray_call_result_391931 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), asarray_391925, *[b_vec_391926], **kwargs_391930)
    
    # Obtaining the member 'ravel' of a type (line 165)
    ravel_391932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), asarray_call_result_391931, 'ravel')
    # Calling ravel(args, kwargs) (line 165)
    ravel_call_result_391934 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), ravel_391932, *[], **kwargs_391933)
    
    # Assigning a type to the variable 'b_vec' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'b_vec', ravel_call_result_391934)
    
    # Getting the type of 'noScikit' (line 167)
    noScikit_391935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'noScikit')
    # Testing the type of an if condition (line 167)
    if_condition_391936 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 8), noScikit_391935)
    # Assigning a type to the variable 'if_condition_391936' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'if_condition_391936', if_condition_391936)
    # SSA begins for if statement (line 167)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 168)
    # Processing the call arguments (line 168)
    str_391938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 31), 'str', 'Scikits.umfpack not installed.')
    # Processing the call keyword arguments (line 168)
    kwargs_391939 = {}
    # Getting the type of 'RuntimeError' (line 168)
    RuntimeError_391937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 168)
    RuntimeError_call_result_391940 = invoke(stypy.reporting.localization.Localization(__file__, 168, 18), RuntimeError_391937, *[str_391938], **kwargs_391939)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 168, 12), RuntimeError_call_result_391940, 'raise parameter', BaseException)
    # SSA join for if statement (line 167)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'A' (line 170)
    A_391941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'A')
    # Obtaining the member 'dtype' of a type (line 170)
    dtype_391942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 11), A_391941, 'dtype')
    # Obtaining the member 'char' of a type (line 170)
    char_391943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 11), dtype_391942, 'char')
    str_391944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 31), 'str', 'dD')
    # Applying the binary operator 'notin' (line 170)
    result_contains_391945 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 11), 'notin', char_391943, str_391944)
    
    # Testing the type of an if condition (line 170)
    if_condition_391946 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 8), result_contains_391945)
    # Assigning a type to the variable 'if_condition_391946' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'if_condition_391946', if_condition_391946)
    # SSA begins for if statement (line 170)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 171)
    # Processing the call arguments (line 171)
    str_391948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 29), 'str', 'convert matrix data to double, please, using .astype(), or set linsolve.useUmfpack = False')
    # Processing the call keyword arguments (line 171)
    kwargs_391949 = {}
    # Getting the type of 'ValueError' (line 171)
    ValueError_391947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 171)
    ValueError_call_result_391950 = invoke(stypy.reporting.localization.Localization(__file__, 171, 18), ValueError_391947, *[str_391948], **kwargs_391949)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 171, 12), ValueError_call_result_391950, 'raise parameter', BaseException)
    # SSA join for if statement (line 170)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 174):
    
    # Assigning a Call to a Name (line 174):
    
    # Call to UmfpackContext(...): (line 174)
    # Processing the call arguments (line 174)
    
    # Call to _get_umf_family(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'A' (line 174)
    A_391954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 53), 'A', False)
    # Processing the call keyword arguments (line 174)
    kwargs_391955 = {}
    # Getting the type of '_get_umf_family' (line 174)
    _get_umf_family_391953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 37), '_get_umf_family', False)
    # Calling _get_umf_family(args, kwargs) (line 174)
    _get_umf_family_call_result_391956 = invoke(stypy.reporting.localization.Localization(__file__, 174, 37), _get_umf_family_391953, *[A_391954], **kwargs_391955)
    
    # Processing the call keyword arguments (line 174)
    kwargs_391957 = {}
    # Getting the type of 'umfpack' (line 174)
    umfpack_391951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 14), 'umfpack', False)
    # Obtaining the member 'UmfpackContext' of a type (line 174)
    UmfpackContext_391952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 14), umfpack_391951, 'UmfpackContext')
    # Calling UmfpackContext(args, kwargs) (line 174)
    UmfpackContext_call_result_391958 = invoke(stypy.reporting.localization.Localization(__file__, 174, 14), UmfpackContext_391952, *[_get_umf_family_call_result_391956], **kwargs_391957)
    
    # Assigning a type to the variable 'umf' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'umf', UmfpackContext_call_result_391958)
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to linsolve(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'umfpack' (line 175)
    umfpack_391961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'umfpack', False)
    # Obtaining the member 'UMFPACK_A' of a type (line 175)
    UMFPACK_A_391962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 25), umfpack_391961, 'UMFPACK_A')
    # Getting the type of 'A' (line 175)
    A_391963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 44), 'A', False)
    # Getting the type of 'b_vec' (line 175)
    b_vec_391964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 47), 'b_vec', False)
    # Processing the call keyword arguments (line 175)
    # Getting the type of 'True' (line 176)
    True_391965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 39), 'True', False)
    keyword_391966 = True_391965
    kwargs_391967 = {'autoTranspose': keyword_391966}
    # Getting the type of 'umf' (line 175)
    umf_391959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'umf', False)
    # Obtaining the member 'linsolve' of a type (line 175)
    linsolve_391960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), umf_391959, 'linsolve')
    # Calling linsolve(args, kwargs) (line 175)
    linsolve_call_result_391968 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), linsolve_391960, *[UMFPACK_A_391962, A_391963, b_vec_391964], **kwargs_391967)
    
    # Assigning a type to the variable 'x' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'x', linsolve_call_result_391968)
    # SSA branch for the else part of an if statement (line 160)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'b_is_vector' (line 178)
    b_is_vector_391969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 11), 'b_is_vector')
    # Getting the type of 'b_is_sparse' (line 178)
    b_is_sparse_391970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'b_is_sparse')
    # Applying the binary operator 'and' (line 178)
    result_and_keyword_391971 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 11), 'and', b_is_vector_391969, b_is_sparse_391970)
    
    # Testing the type of an if condition (line 178)
    if_condition_391972 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 8), result_and_keyword_391971)
    # Assigning a type to the variable 'if_condition_391972' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'if_condition_391972', if_condition_391972)
    # SSA begins for if statement (line 178)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 179):
    
    # Assigning a Call to a Name (line 179):
    
    # Call to toarray(...): (line 179)
    # Processing the call keyword arguments (line 179)
    kwargs_391975 = {}
    # Getting the type of 'b' (line 179)
    b_391973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'b', False)
    # Obtaining the member 'toarray' of a type (line 179)
    toarray_391974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 16), b_391973, 'toarray')
    # Calling toarray(args, kwargs) (line 179)
    toarray_call_result_391976 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), toarray_391974, *[], **kwargs_391975)
    
    # Assigning a type to the variable 'b' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'b', toarray_call_result_391976)
    
    # Assigning a Name to a Name (line 180):
    
    # Assigning a Name to a Name (line 180):
    # Getting the type of 'False' (line 180)
    False_391977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 26), 'False')
    # Assigning a type to the variable 'b_is_sparse' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'b_is_sparse', False_391977)
    # SSA join for if statement (line 178)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'b_is_sparse' (line 182)
    b_is_sparse_391978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'b_is_sparse')
    # Applying the 'not' unary operator (line 182)
    result_not__391979 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 11), 'not', b_is_sparse_391978)
    
    # Testing the type of an if condition (line 182)
    if_condition_391980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 8), result_not__391979)
    # Assigning a type to the variable 'if_condition_391980' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'if_condition_391980', if_condition_391980)
    # SSA begins for if statement (line 182)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to isspmatrix_csc(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'A' (line 183)
    A_391982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 30), 'A', False)
    # Processing the call keyword arguments (line 183)
    kwargs_391983 = {}
    # Getting the type of 'isspmatrix_csc' (line 183)
    isspmatrix_csc_391981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'isspmatrix_csc', False)
    # Calling isspmatrix_csc(args, kwargs) (line 183)
    isspmatrix_csc_call_result_391984 = invoke(stypy.reporting.localization.Localization(__file__, 183, 15), isspmatrix_csc_391981, *[A_391982], **kwargs_391983)
    
    # Testing the type of an if condition (line 183)
    if_condition_391985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 12), isspmatrix_csc_call_result_391984)
    # Assigning a type to the variable 'if_condition_391985' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'if_condition_391985', if_condition_391985)
    # SSA begins for if statement (line 183)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 184):
    
    # Assigning a Num to a Name (line 184):
    int_391986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 23), 'int')
    # Assigning a type to the variable 'flag' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'flag', int_391986)
    # SSA branch for the else part of an if statement (line 183)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 186):
    
    # Assigning a Num to a Name (line 186):
    int_391987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 23), 'int')
    # Assigning a type to the variable 'flag' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'flag', int_391987)
    # SSA join for if statement (line 183)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 188):
    
    # Assigning a Call to a Name (line 188):
    
    # Call to dict(...): (line 188)
    # Processing the call keyword arguments (line 188)
    # Getting the type of 'permc_spec' (line 188)
    permc_spec_391989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 35), 'permc_spec', False)
    keyword_391990 = permc_spec_391989
    kwargs_391991 = {'ColPerm': keyword_391990}
    # Getting the type of 'dict' (line 188)
    dict_391988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 22), 'dict', False)
    # Calling dict(args, kwargs) (line 188)
    dict_call_result_391992 = invoke(stypy.reporting.localization.Localization(__file__, 188, 22), dict_391988, *[], **kwargs_391991)
    
    # Assigning a type to the variable 'options' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'options', dict_call_result_391992)
    
    # Assigning a Call to a Tuple (line 189):
    
    # Assigning a Subscript to a Name (line 189):
    
    # Obtaining the type of the subscript
    int_391993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 12), 'int')
    
    # Call to gssv(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'N' (line 189)
    N_391996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 36), 'N', False)
    # Getting the type of 'A' (line 189)
    A_391997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 39), 'A', False)
    # Obtaining the member 'nnz' of a type (line 189)
    nnz_391998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 39), A_391997, 'nnz')
    # Getting the type of 'A' (line 189)
    A_391999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 46), 'A', False)
    # Obtaining the member 'data' of a type (line 189)
    data_392000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 46), A_391999, 'data')
    # Getting the type of 'A' (line 189)
    A_392001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 54), 'A', False)
    # Obtaining the member 'indices' of a type (line 189)
    indices_392002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 54), A_392001, 'indices')
    # Getting the type of 'A' (line 189)
    A_392003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 65), 'A', False)
    # Obtaining the member 'indptr' of a type (line 189)
    indptr_392004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 65), A_392003, 'indptr')
    # Getting the type of 'b' (line 190)
    b_392005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 36), 'b', False)
    # Getting the type of 'flag' (line 190)
    flag_392006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 39), 'flag', False)
    # Processing the call keyword arguments (line 189)
    # Getting the type of 'options' (line 190)
    options_392007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 53), 'options', False)
    keyword_392008 = options_392007
    kwargs_392009 = {'options': keyword_392008}
    # Getting the type of '_superlu' (line 189)
    _superlu_391994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 22), '_superlu', False)
    # Obtaining the member 'gssv' of a type (line 189)
    gssv_391995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 22), _superlu_391994, 'gssv')
    # Calling gssv(args, kwargs) (line 189)
    gssv_call_result_392010 = invoke(stypy.reporting.localization.Localization(__file__, 189, 22), gssv_391995, *[N_391996, nnz_391998, data_392000, indices_392002, indptr_392004, b_392005, flag_392006], **kwargs_392009)
    
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___392011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), gssv_call_result_392010, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_392012 = invoke(stypy.reporting.localization.Localization(__file__, 189, 12), getitem___392011, int_391993)
    
    # Assigning a type to the variable 'tuple_var_assignment_391660' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'tuple_var_assignment_391660', subscript_call_result_392012)
    
    # Assigning a Subscript to a Name (line 189):
    
    # Obtaining the type of the subscript
    int_392013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 12), 'int')
    
    # Call to gssv(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'N' (line 189)
    N_392016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 36), 'N', False)
    # Getting the type of 'A' (line 189)
    A_392017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 39), 'A', False)
    # Obtaining the member 'nnz' of a type (line 189)
    nnz_392018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 39), A_392017, 'nnz')
    # Getting the type of 'A' (line 189)
    A_392019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 46), 'A', False)
    # Obtaining the member 'data' of a type (line 189)
    data_392020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 46), A_392019, 'data')
    # Getting the type of 'A' (line 189)
    A_392021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 54), 'A', False)
    # Obtaining the member 'indices' of a type (line 189)
    indices_392022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 54), A_392021, 'indices')
    # Getting the type of 'A' (line 189)
    A_392023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 65), 'A', False)
    # Obtaining the member 'indptr' of a type (line 189)
    indptr_392024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 65), A_392023, 'indptr')
    # Getting the type of 'b' (line 190)
    b_392025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 36), 'b', False)
    # Getting the type of 'flag' (line 190)
    flag_392026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 39), 'flag', False)
    # Processing the call keyword arguments (line 189)
    # Getting the type of 'options' (line 190)
    options_392027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 53), 'options', False)
    keyword_392028 = options_392027
    kwargs_392029 = {'options': keyword_392028}
    # Getting the type of '_superlu' (line 189)
    _superlu_392014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 22), '_superlu', False)
    # Obtaining the member 'gssv' of a type (line 189)
    gssv_392015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 22), _superlu_392014, 'gssv')
    # Calling gssv(args, kwargs) (line 189)
    gssv_call_result_392030 = invoke(stypy.reporting.localization.Localization(__file__, 189, 22), gssv_392015, *[N_392016, nnz_392018, data_392020, indices_392022, indptr_392024, b_392025, flag_392026], **kwargs_392029)
    
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___392031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), gssv_call_result_392030, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_392032 = invoke(stypy.reporting.localization.Localization(__file__, 189, 12), getitem___392031, int_392013)
    
    # Assigning a type to the variable 'tuple_var_assignment_391661' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'tuple_var_assignment_391661', subscript_call_result_392032)
    
    # Assigning a Name to a Name (line 189):
    # Getting the type of 'tuple_var_assignment_391660' (line 189)
    tuple_var_assignment_391660_392033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'tuple_var_assignment_391660')
    # Assigning a type to the variable 'x' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'x', tuple_var_assignment_391660_392033)
    
    # Assigning a Name to a Name (line 189):
    # Getting the type of 'tuple_var_assignment_391661' (line 189)
    tuple_var_assignment_391661_392034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'tuple_var_assignment_391661')
    # Assigning a type to the variable 'info' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'info', tuple_var_assignment_391661_392034)
    
    
    # Getting the type of 'info' (line 191)
    info_392035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'info')
    int_392036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 23), 'int')
    # Applying the binary operator '!=' (line 191)
    result_ne_392037 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), '!=', info_392035, int_392036)
    
    # Testing the type of an if condition (line 191)
    if_condition_392038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 12), result_ne_392037)
    # Assigning a type to the variable 'if_condition_392038' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'if_condition_392038', if_condition_392038)
    # SSA begins for if statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 192)
    # Processing the call arguments (line 192)
    str_392040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 21), 'str', 'Matrix is exactly singular')
    # Getting the type of 'MatrixRankWarning' (line 192)
    MatrixRankWarning_392041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 51), 'MatrixRankWarning', False)
    # Processing the call keyword arguments (line 192)
    kwargs_392042 = {}
    # Getting the type of 'warn' (line 192)
    warn_392039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'warn', False)
    # Calling warn(args, kwargs) (line 192)
    warn_call_result_392043 = invoke(stypy.reporting.localization.Localization(__file__, 192, 16), warn_392039, *[str_392040, MatrixRankWarning_392041], **kwargs_392042)
    
    
    # Call to fill(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'np' (line 193)
    np_392046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'np', False)
    # Obtaining the member 'nan' of a type (line 193)
    nan_392047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 23), np_392046, 'nan')
    # Processing the call keyword arguments (line 193)
    kwargs_392048 = {}
    # Getting the type of 'x' (line 193)
    x_392044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'x', False)
    # Obtaining the member 'fill' of a type (line 193)
    fill_392045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 16), x_392044, 'fill')
    # Calling fill(args, kwargs) (line 193)
    fill_call_result_392049 = invoke(stypy.reporting.localization.Localization(__file__, 193, 16), fill_392045, *[nan_392047], **kwargs_392048)
    
    # SSA join for if statement (line 191)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'b_is_vector' (line 194)
    b_is_vector_392050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'b_is_vector')
    # Testing the type of an if condition (line 194)
    if_condition_392051 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 12), b_is_vector_392050)
    # Assigning a type to the variable 'if_condition_392051' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'if_condition_392051', if_condition_392051)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 195):
    
    # Assigning a Call to a Name (line 195):
    
    # Call to ravel(...): (line 195)
    # Processing the call keyword arguments (line 195)
    kwargs_392054 = {}
    # Getting the type of 'x' (line 195)
    x_392052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'x', False)
    # Obtaining the member 'ravel' of a type (line 195)
    ravel_392053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 20), x_392052, 'ravel')
    # Calling ravel(args, kwargs) (line 195)
    ravel_call_result_392055 = invoke(stypy.reporting.localization.Localization(__file__, 195, 20), ravel_392053, *[], **kwargs_392054)
    
    # Assigning a type to the variable 'x' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'x', ravel_call_result_392055)
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 182)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to factorized(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'A' (line 198)
    A_392057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 36), 'A', False)
    # Processing the call keyword arguments (line 198)
    kwargs_392058 = {}
    # Getting the type of 'factorized' (line 198)
    factorized_392056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 25), 'factorized', False)
    # Calling factorized(args, kwargs) (line 198)
    factorized_call_result_392059 = invoke(stypy.reporting.localization.Localization(__file__, 198, 25), factorized_392056, *[A_392057], **kwargs_392058)
    
    # Assigning a type to the variable 'Afactsolve' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'Afactsolve', factorized_call_result_392059)
    
    
    
    # Call to isspmatrix_csc(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'b' (line 200)
    b_392061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 34), 'b', False)
    # Processing the call keyword arguments (line 200)
    kwargs_392062 = {}
    # Getting the type of 'isspmatrix_csc' (line 200)
    isspmatrix_csc_392060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'isspmatrix_csc', False)
    # Calling isspmatrix_csc(args, kwargs) (line 200)
    isspmatrix_csc_call_result_392063 = invoke(stypy.reporting.localization.Localization(__file__, 200, 19), isspmatrix_csc_392060, *[b_392061], **kwargs_392062)
    
    # Applying the 'not' unary operator (line 200)
    result_not__392064 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 15), 'not', isspmatrix_csc_call_result_392063)
    
    # Testing the type of an if condition (line 200)
    if_condition_392065 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 12), result_not__392064)
    # Assigning a type to the variable 'if_condition_392065' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'if_condition_392065', if_condition_392065)
    # SSA begins for if statement (line 200)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 201)
    # Processing the call arguments (line 201)
    str_392067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 21), 'str', 'spsolve is more efficient when sparse b is in the CSC matrix format')
    # Getting the type of 'SparseEfficiencyWarning' (line 202)
    SparseEfficiencyWarning_392068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 52), 'SparseEfficiencyWarning', False)
    # Processing the call keyword arguments (line 201)
    kwargs_392069 = {}
    # Getting the type of 'warn' (line 201)
    warn_392066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'warn', False)
    # Calling warn(args, kwargs) (line 201)
    warn_call_result_392070 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), warn_392066, *[str_392067, SparseEfficiencyWarning_392068], **kwargs_392069)
    
    
    # Assigning a Call to a Name (line 203):
    
    # Assigning a Call to a Name (line 203):
    
    # Call to csc_matrix(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'b' (line 203)
    b_392072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 31), 'b', False)
    # Processing the call keyword arguments (line 203)
    kwargs_392073 = {}
    # Getting the type of 'csc_matrix' (line 203)
    csc_matrix_392071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 20), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 203)
    csc_matrix_call_result_392074 = invoke(stypy.reporting.localization.Localization(__file__, 203, 20), csc_matrix_392071, *[b_392072], **kwargs_392073)
    
    # Assigning a type to the variable 'b' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'b', csc_matrix_call_result_392074)
    # SSA join for if statement (line 200)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 207):
    
    # Assigning a List to a Name (line 207):
    
    # Obtaining an instance of the builtin type 'list' (line 207)
    list_392075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 207)
    
    # Assigning a type to the variable 'data_segs' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'data_segs', list_392075)
    
    # Assigning a List to a Name (line 208):
    
    # Assigning a List to a Name (line 208):
    
    # Obtaining an instance of the builtin type 'list' (line 208)
    list_392076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 208)
    
    # Assigning a type to the variable 'row_segs' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'row_segs', list_392076)
    
    # Assigning a List to a Name (line 209):
    
    # Assigning a List to a Name (line 209):
    
    # Obtaining an instance of the builtin type 'list' (line 209)
    list_392077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 209)
    
    # Assigning a type to the variable 'col_segs' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'col_segs', list_392077)
    
    
    # Call to range(...): (line 210)
    # Processing the call arguments (line 210)
    
    # Obtaining the type of the subscript
    int_392079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 35), 'int')
    # Getting the type of 'b' (line 210)
    b_392080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 27), 'b', False)
    # Obtaining the member 'shape' of a type (line 210)
    shape_392081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 27), b_392080, 'shape')
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___392082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 27), shape_392081, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 210)
    subscript_call_result_392083 = invoke(stypy.reporting.localization.Localization(__file__, 210, 27), getitem___392082, int_392079)
    
    # Processing the call keyword arguments (line 210)
    kwargs_392084 = {}
    # Getting the type of 'range' (line 210)
    range_392078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 21), 'range', False)
    # Calling range(args, kwargs) (line 210)
    range_call_result_392085 = invoke(stypy.reporting.localization.Localization(__file__, 210, 21), range_392078, *[subscript_call_result_392083], **kwargs_392084)
    
    # Testing the type of a for loop iterable (line 210)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 210, 12), range_call_result_392085)
    # Getting the type of the for loop variable (line 210)
    for_loop_var_392086 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 210, 12), range_call_result_392085)
    # Assigning a type to the variable 'j' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'j', for_loop_var_392086)
    # SSA begins for a for statement (line 210)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 211):
    
    # Assigning a Call to a Name (line 211):
    
    # Call to ravel(...): (line 211)
    # Processing the call keyword arguments (line 211)
    kwargs_392094 = {}
    
    # Obtaining the type of the subscript
    slice_392087 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 211, 21), None, None, None)
    # Getting the type of 'j' (line 211)
    j_392088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 26), 'j', False)
    # Getting the type of 'b' (line 211)
    b_392089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 211)
    getitem___392090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 21), b_392089, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 211)
    subscript_call_result_392091 = invoke(stypy.reporting.localization.Localization(__file__, 211, 21), getitem___392090, (slice_392087, j_392088))
    
    # Obtaining the member 'A' of a type (line 211)
    A_392092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 21), subscript_call_result_392091, 'A')
    # Obtaining the member 'ravel' of a type (line 211)
    ravel_392093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 21), A_392092, 'ravel')
    # Calling ravel(args, kwargs) (line 211)
    ravel_call_result_392095 = invoke(stypy.reporting.localization.Localization(__file__, 211, 21), ravel_392093, *[], **kwargs_392094)
    
    # Assigning a type to the variable 'bj' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'bj', ravel_call_result_392095)
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Call to Afactsolve(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'bj' (line 212)
    bj_392097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 32), 'bj', False)
    # Processing the call keyword arguments (line 212)
    kwargs_392098 = {}
    # Getting the type of 'Afactsolve' (line 212)
    Afactsolve_392096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 21), 'Afactsolve', False)
    # Calling Afactsolve(args, kwargs) (line 212)
    Afactsolve_call_result_392099 = invoke(stypy.reporting.localization.Localization(__file__, 212, 21), Afactsolve_392096, *[bj_392097], **kwargs_392098)
    
    # Assigning a type to the variable 'xj' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'xj', Afactsolve_call_result_392099)
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to flatnonzero(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'xj' (line 213)
    xj_392102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 35), 'xj', False)
    # Processing the call keyword arguments (line 213)
    kwargs_392103 = {}
    # Getting the type of 'np' (line 213)
    np_392100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'np', False)
    # Obtaining the member 'flatnonzero' of a type (line 213)
    flatnonzero_392101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 20), np_392100, 'flatnonzero')
    # Calling flatnonzero(args, kwargs) (line 213)
    flatnonzero_call_result_392104 = invoke(stypy.reporting.localization.Localization(__file__, 213, 20), flatnonzero_392101, *[xj_392102], **kwargs_392103)
    
    # Assigning a type to the variable 'w' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'w', flatnonzero_call_result_392104)
    
    # Assigning a Subscript to a Name (line 214):
    
    # Assigning a Subscript to a Name (line 214):
    
    # Obtaining the type of the subscript
    int_392105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 41), 'int')
    # Getting the type of 'w' (line 214)
    w_392106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 33), 'w')
    # Obtaining the member 'shape' of a type (line 214)
    shape_392107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 33), w_392106, 'shape')
    # Obtaining the member '__getitem__' of a type (line 214)
    getitem___392108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 33), shape_392107, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 214)
    subscript_call_result_392109 = invoke(stypy.reporting.localization.Localization(__file__, 214, 33), getitem___392108, int_392105)
    
    # Assigning a type to the variable 'segment_length' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'segment_length', subscript_call_result_392109)
    
    # Call to append(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'w' (line 215)
    w_392112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 32), 'w', False)
    # Processing the call keyword arguments (line 215)
    kwargs_392113 = {}
    # Getting the type of 'row_segs' (line 215)
    row_segs_392110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'row_segs', False)
    # Obtaining the member 'append' of a type (line 215)
    append_392111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 16), row_segs_392110, 'append')
    # Calling append(args, kwargs) (line 215)
    append_call_result_392114 = invoke(stypy.reporting.localization.Localization(__file__, 215, 16), append_392111, *[w_392112], **kwargs_392113)
    
    
    # Call to append(...): (line 216)
    # Processing the call arguments (line 216)
    
    # Call to ones(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'segment_length' (line 216)
    segment_length_392119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 40), 'segment_length', False)
    # Processing the call keyword arguments (line 216)
    # Getting the type of 'int' (line 216)
    int_392120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 62), 'int', False)
    keyword_392121 = int_392120
    kwargs_392122 = {'dtype': keyword_392121}
    # Getting the type of 'np' (line 216)
    np_392117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 32), 'np', False)
    # Obtaining the member 'ones' of a type (line 216)
    ones_392118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 32), np_392117, 'ones')
    # Calling ones(args, kwargs) (line 216)
    ones_call_result_392123 = invoke(stypy.reporting.localization.Localization(__file__, 216, 32), ones_392118, *[segment_length_392119], **kwargs_392122)
    
    # Getting the type of 'j' (line 216)
    j_392124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 67), 'j', False)
    # Applying the binary operator '*' (line 216)
    result_mul_392125 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 32), '*', ones_call_result_392123, j_392124)
    
    # Processing the call keyword arguments (line 216)
    kwargs_392126 = {}
    # Getting the type of 'col_segs' (line 216)
    col_segs_392115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'col_segs', False)
    # Obtaining the member 'append' of a type (line 216)
    append_392116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), col_segs_392115, 'append')
    # Calling append(args, kwargs) (line 216)
    append_call_result_392127 = invoke(stypy.reporting.localization.Localization(__file__, 216, 16), append_392116, *[result_mul_392125], **kwargs_392126)
    
    
    # Call to append(...): (line 217)
    # Processing the call arguments (line 217)
    
    # Call to asarray(...): (line 217)
    # Processing the call arguments (line 217)
    
    # Obtaining the type of the subscript
    # Getting the type of 'w' (line 217)
    w_392132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 47), 'w', False)
    # Getting the type of 'xj' (line 217)
    xj_392133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 44), 'xj', False)
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___392134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 44), xj_392133, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_392135 = invoke(stypy.reporting.localization.Localization(__file__, 217, 44), getitem___392134, w_392132)
    
    # Processing the call keyword arguments (line 217)
    # Getting the type of 'A' (line 217)
    A_392136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 57), 'A', False)
    # Obtaining the member 'dtype' of a type (line 217)
    dtype_392137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 57), A_392136, 'dtype')
    keyword_392138 = dtype_392137
    kwargs_392139 = {'dtype': keyword_392138}
    # Getting the type of 'np' (line 217)
    np_392130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 33), 'np', False)
    # Obtaining the member 'asarray' of a type (line 217)
    asarray_392131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 33), np_392130, 'asarray')
    # Calling asarray(args, kwargs) (line 217)
    asarray_call_result_392140 = invoke(stypy.reporting.localization.Localization(__file__, 217, 33), asarray_392131, *[subscript_call_result_392135], **kwargs_392139)
    
    # Processing the call keyword arguments (line 217)
    kwargs_392141 = {}
    # Getting the type of 'data_segs' (line 217)
    data_segs_392128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'data_segs', False)
    # Obtaining the member 'append' of a type (line 217)
    append_392129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 16), data_segs_392128, 'append')
    # Calling append(args, kwargs) (line 217)
    append_call_result_392142 = invoke(stypy.reporting.localization.Localization(__file__, 217, 16), append_392129, *[asarray_call_result_392140], **kwargs_392141)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 218):
    
    # Assigning a Call to a Name (line 218):
    
    # Call to concatenate(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'data_segs' (line 218)
    data_segs_392145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 41), 'data_segs', False)
    # Processing the call keyword arguments (line 218)
    kwargs_392146 = {}
    # Getting the type of 'np' (line 218)
    np_392143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 26), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 218)
    concatenate_392144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 26), np_392143, 'concatenate')
    # Calling concatenate(args, kwargs) (line 218)
    concatenate_call_result_392147 = invoke(stypy.reporting.localization.Localization(__file__, 218, 26), concatenate_392144, *[data_segs_392145], **kwargs_392146)
    
    # Assigning a type to the variable 'sparse_data' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'sparse_data', concatenate_call_result_392147)
    
    # Assigning a Call to a Name (line 219):
    
    # Assigning a Call to a Name (line 219):
    
    # Call to concatenate(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'row_segs' (line 219)
    row_segs_392150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 40), 'row_segs', False)
    # Processing the call keyword arguments (line 219)
    kwargs_392151 = {}
    # Getting the type of 'np' (line 219)
    np_392148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 25), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 219)
    concatenate_392149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 25), np_392148, 'concatenate')
    # Calling concatenate(args, kwargs) (line 219)
    concatenate_call_result_392152 = invoke(stypy.reporting.localization.Localization(__file__, 219, 25), concatenate_392149, *[row_segs_392150], **kwargs_392151)
    
    # Assigning a type to the variable 'sparse_row' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'sparse_row', concatenate_call_result_392152)
    
    # Assigning a Call to a Name (line 220):
    
    # Assigning a Call to a Name (line 220):
    
    # Call to concatenate(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'col_segs' (line 220)
    col_segs_392155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 40), 'col_segs', False)
    # Processing the call keyword arguments (line 220)
    kwargs_392156 = {}
    # Getting the type of 'np' (line 220)
    np_392153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 25), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 220)
    concatenate_392154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 25), np_392153, 'concatenate')
    # Calling concatenate(args, kwargs) (line 220)
    concatenate_call_result_392157 = invoke(stypy.reporting.localization.Localization(__file__, 220, 25), concatenate_392154, *[col_segs_392155], **kwargs_392156)
    
    # Assigning a type to the variable 'sparse_col' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'sparse_col', concatenate_call_result_392157)
    
    # Assigning a Call to a Name (line 221):
    
    # Assigning a Call to a Name (line 221):
    
    # Call to __class__(...): (line 221)
    # Processing the call arguments (line 221)
    
    # Obtaining an instance of the builtin type 'tuple' (line 221)
    tuple_392160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 221)
    # Adding element type (line 221)
    # Getting the type of 'sparse_data' (line 221)
    sparse_data_392161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 29), 'sparse_data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 29), tuple_392160, sparse_data_392161)
    # Adding element type (line 221)
    
    # Obtaining an instance of the builtin type 'tuple' (line 221)
    tuple_392162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 221)
    # Adding element type (line 221)
    # Getting the type of 'sparse_row' (line 221)
    sparse_row_392163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), 'sparse_row', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 43), tuple_392162, sparse_row_392163)
    # Adding element type (line 221)
    # Getting the type of 'sparse_col' (line 221)
    sparse_col_392164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 55), 'sparse_col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 43), tuple_392162, sparse_col_392164)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 29), tuple_392160, tuple_392162)
    
    # Processing the call keyword arguments (line 221)
    # Getting the type of 'b' (line 222)
    b_392165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 33), 'b', False)
    # Obtaining the member 'shape' of a type (line 222)
    shape_392166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 33), b_392165, 'shape')
    keyword_392167 = shape_392166
    # Getting the type of 'A' (line 222)
    A_392168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 48), 'A', False)
    # Obtaining the member 'dtype' of a type (line 222)
    dtype_392169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 48), A_392168, 'dtype')
    keyword_392170 = dtype_392169
    kwargs_392171 = {'dtype': keyword_392170, 'shape': keyword_392167}
    # Getting the type of 'A' (line 221)
    A_392158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'A', False)
    # Obtaining the member '__class__' of a type (line 221)
    class___392159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), A_392158, '__class__')
    # Calling __class__(args, kwargs) (line 221)
    class___call_result_392172 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), class___392159, *[tuple_392160], **kwargs_392171)
    
    # Assigning a type to the variable 'x' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'x', class___call_result_392172)
    # SSA join for if statement (line 182)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 160)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 224)
    x_392173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'stypy_return_type', x_392173)
    
    # ################# End of 'spsolve(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spsolve' in the type store
    # Getting the type of 'stypy_return_type' (line 83)
    stypy_return_type_392174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_392174)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spsolve'
    return stypy_return_type_392174

# Assigning a type to the variable 'spsolve' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'spsolve', spsolve)

@norecursion
def splu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 227)
    None_392175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 23), 'None')
    # Getting the type of 'None' (line 227)
    None_392176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 47), 'None')
    # Getting the type of 'None' (line 228)
    None_392177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'None')
    # Getting the type of 'None' (line 228)
    None_392178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 32), 'None')
    
    # Call to dict(...): (line 228)
    # Processing the call keyword arguments (line 228)
    kwargs_392180 = {}
    # Getting the type of 'dict' (line 228)
    dict_392179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 46), 'dict', False)
    # Calling dict(args, kwargs) (line 228)
    dict_call_result_392181 = invoke(stypy.reporting.localization.Localization(__file__, 228, 46), dict_392179, *[], **kwargs_392180)
    
    defaults = [None_392175, None_392176, None_392177, None_392178, dict_call_result_392181]
    # Create a new context for function 'splu'
    module_type_store = module_type_store.open_function_context('splu', 227, 0, False)
    
    # Passed parameters checking function
    splu.stypy_localization = localization
    splu.stypy_type_of_self = None
    splu.stypy_type_store = module_type_store
    splu.stypy_function_name = 'splu'
    splu.stypy_param_names_list = ['A', 'permc_spec', 'diag_pivot_thresh', 'relax', 'panel_size', 'options']
    splu.stypy_varargs_param_name = None
    splu.stypy_kwargs_param_name = None
    splu.stypy_call_defaults = defaults
    splu.stypy_call_varargs = varargs
    splu.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'splu', ['A', 'permc_spec', 'diag_pivot_thresh', 'relax', 'panel_size', 'options'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'splu', localization, ['A', 'permc_spec', 'diag_pivot_thresh', 'relax', 'panel_size', 'options'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'splu(...)' code ##################

    str_392182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, (-1)), 'str', "\n    Compute the LU decomposition of a sparse, square matrix.\n\n    Parameters\n    ----------\n    A : sparse matrix\n        Sparse matrix to factorize. Should be in CSR or CSC format.\n    permc_spec : str, optional\n        How to permute the columns of the matrix for sparsity preservation.\n        (default: 'COLAMD')\n\n        - ``NATURAL``: natural ordering.\n        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.\n        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.\n        - ``COLAMD``: approximate minimum degree column ordering\n\n    diag_pivot_thresh : float, optional\n        Threshold used for a diagonal entry to be an acceptable pivot.\n        See SuperLU user's guide for details [1]_\n    relax : int, optional\n        Expert option for customizing the degree of relaxing supernodes.\n        See SuperLU user's guide for details [1]_\n    panel_size : int, optional\n        Expert option for customizing the panel size.\n        See SuperLU user's guide for details [1]_\n    options : dict, optional\n        Dictionary containing additional expert options to SuperLU.\n        See SuperLU user guide [1]_ (section 2.4 on the 'Options' argument)\n        for more details. For example, you can specify\n        ``options=dict(Equil=False, IterRefine='SINGLE'))``\n        to turn equilibration off and perform a single iterative refinement.\n\n    Returns\n    -------\n    invA : scipy.sparse.linalg.SuperLU\n        Object, which has a ``solve`` method.\n\n    See also\n    --------\n    spilu : incomplete LU decomposition\n\n    Notes\n    -----\n    This function uses the SuperLU library.\n\n    References\n    ----------\n    .. [1] SuperLU http://crd.lbl.gov/~xiaoye/SuperLU/\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import splu\n    >>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)\n    >>> B = splu(A)\n    >>> x = np.array([1., 2., 3.], dtype=float)\n    >>> B.solve(x)\n    array([ 1. , -3. , -1.5])\n    >>> A.dot(B.solve(x))\n    array([ 1.,  2.,  3.])\n    >>> B.solve(A.dot(x))\n    array([ 1.,  2.,  3.])\n    ")
    
    
    
    # Call to isspmatrix_csc(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'A' (line 293)
    A_392184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'A', False)
    # Processing the call keyword arguments (line 293)
    kwargs_392185 = {}
    # Getting the type of 'isspmatrix_csc' (line 293)
    isspmatrix_csc_392183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 11), 'isspmatrix_csc', False)
    # Calling isspmatrix_csc(args, kwargs) (line 293)
    isspmatrix_csc_call_result_392186 = invoke(stypy.reporting.localization.Localization(__file__, 293, 11), isspmatrix_csc_392183, *[A_392184], **kwargs_392185)
    
    # Applying the 'not' unary operator (line 293)
    result_not__392187 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 7), 'not', isspmatrix_csc_call_result_392186)
    
    # Testing the type of an if condition (line 293)
    if_condition_392188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 4), result_not__392187)
    # Assigning a type to the variable 'if_condition_392188' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'if_condition_392188', if_condition_392188)
    # SSA begins for if statement (line 293)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 294):
    
    # Assigning a Call to a Name (line 294):
    
    # Call to csc_matrix(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'A' (line 294)
    A_392190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 23), 'A', False)
    # Processing the call keyword arguments (line 294)
    kwargs_392191 = {}
    # Getting the type of 'csc_matrix' (line 294)
    csc_matrix_392189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 294)
    csc_matrix_call_result_392192 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), csc_matrix_392189, *[A_392190], **kwargs_392191)
    
    # Assigning a type to the variable 'A' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'A', csc_matrix_call_result_392192)
    
    # Call to warn(...): (line 295)
    # Processing the call arguments (line 295)
    str_392194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 13), 'str', 'splu requires CSC matrix format')
    # Getting the type of 'SparseEfficiencyWarning' (line 295)
    SparseEfficiencyWarning_392195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 48), 'SparseEfficiencyWarning', False)
    # Processing the call keyword arguments (line 295)
    kwargs_392196 = {}
    # Getting the type of 'warn' (line 295)
    warn_392193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 295)
    warn_call_result_392197 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), warn_392193, *[str_392194, SparseEfficiencyWarning_392195], **kwargs_392196)
    
    # SSA join for if statement (line 293)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to sort_indices(...): (line 297)
    # Processing the call keyword arguments (line 297)
    kwargs_392200 = {}
    # Getting the type of 'A' (line 297)
    A_392198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'A', False)
    # Obtaining the member 'sort_indices' of a type (line 297)
    sort_indices_392199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 4), A_392198, 'sort_indices')
    # Calling sort_indices(args, kwargs) (line 297)
    sort_indices_call_result_392201 = invoke(stypy.reporting.localization.Localization(__file__, 297, 4), sort_indices_392199, *[], **kwargs_392200)
    
    
    # Assigning a Call to a Name (line 298):
    
    # Assigning a Call to a Name (line 298):
    
    # Call to asfptype(...): (line 298)
    # Processing the call keyword arguments (line 298)
    kwargs_392204 = {}
    # Getting the type of 'A' (line 298)
    A_392202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'A', False)
    # Obtaining the member 'asfptype' of a type (line 298)
    asfptype_392203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), A_392202, 'asfptype')
    # Calling asfptype(args, kwargs) (line 298)
    asfptype_call_result_392205 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), asfptype_392203, *[], **kwargs_392204)
    
    # Assigning a type to the variable 'A' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'A', asfptype_call_result_392205)
    
    # Assigning a Attribute to a Tuple (line 300):
    
    # Assigning a Subscript to a Name (line 300):
    
    # Obtaining the type of the subscript
    int_392206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 4), 'int')
    # Getting the type of 'A' (line 300)
    A_392207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 11), 'A')
    # Obtaining the member 'shape' of a type (line 300)
    shape_392208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 11), A_392207, 'shape')
    # Obtaining the member '__getitem__' of a type (line 300)
    getitem___392209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 4), shape_392208, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 300)
    subscript_call_result_392210 = invoke(stypy.reporting.localization.Localization(__file__, 300, 4), getitem___392209, int_392206)
    
    # Assigning a type to the variable 'tuple_var_assignment_391662' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'tuple_var_assignment_391662', subscript_call_result_392210)
    
    # Assigning a Subscript to a Name (line 300):
    
    # Obtaining the type of the subscript
    int_392211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 4), 'int')
    # Getting the type of 'A' (line 300)
    A_392212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 11), 'A')
    # Obtaining the member 'shape' of a type (line 300)
    shape_392213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 11), A_392212, 'shape')
    # Obtaining the member '__getitem__' of a type (line 300)
    getitem___392214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 4), shape_392213, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 300)
    subscript_call_result_392215 = invoke(stypy.reporting.localization.Localization(__file__, 300, 4), getitem___392214, int_392211)
    
    # Assigning a type to the variable 'tuple_var_assignment_391663' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'tuple_var_assignment_391663', subscript_call_result_392215)
    
    # Assigning a Name to a Name (line 300):
    # Getting the type of 'tuple_var_assignment_391662' (line 300)
    tuple_var_assignment_391662_392216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'tuple_var_assignment_391662')
    # Assigning a type to the variable 'M' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'M', tuple_var_assignment_391662_392216)
    
    # Assigning a Name to a Name (line 300):
    # Getting the type of 'tuple_var_assignment_391663' (line 300)
    tuple_var_assignment_391663_392217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'tuple_var_assignment_391663')
    # Assigning a type to the variable 'N' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 7), 'N', tuple_var_assignment_391663_392217)
    
    
    # Getting the type of 'M' (line 301)
    M_392218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'M')
    # Getting the type of 'N' (line 301)
    N_392219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 13), 'N')
    # Applying the binary operator '!=' (line 301)
    result_ne_392220 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 8), '!=', M_392218, N_392219)
    
    # Testing the type of an if condition (line 301)
    if_condition_392221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 4), result_ne_392220)
    # Assigning a type to the variable 'if_condition_392221' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'if_condition_392221', if_condition_392221)
    # SSA begins for if statement (line 301)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 302)
    # Processing the call arguments (line 302)
    str_392223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 25), 'str', 'can only factor square matrices')
    # Processing the call keyword arguments (line 302)
    kwargs_392224 = {}
    # Getting the type of 'ValueError' (line 302)
    ValueError_392222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 302)
    ValueError_call_result_392225 = invoke(stypy.reporting.localization.Localization(__file__, 302, 14), ValueError_392222, *[str_392223], **kwargs_392224)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 302, 8), ValueError_call_result_392225, 'raise parameter', BaseException)
    # SSA join for if statement (line 301)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 304):
    
    # Assigning a Call to a Name (line 304):
    
    # Call to dict(...): (line 304)
    # Processing the call keyword arguments (line 304)
    # Getting the type of 'diag_pivot_thresh' (line 304)
    diag_pivot_thresh_392227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 36), 'diag_pivot_thresh', False)
    keyword_392228 = diag_pivot_thresh_392227
    # Getting the type of 'permc_spec' (line 304)
    permc_spec_392229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 63), 'permc_spec', False)
    keyword_392230 = permc_spec_392229
    # Getting the type of 'panel_size' (line 305)
    panel_size_392231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 30), 'panel_size', False)
    keyword_392232 = panel_size_392231
    # Getting the type of 'relax' (line 305)
    relax_392233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 48), 'relax', False)
    keyword_392234 = relax_392233
    kwargs_392235 = {'DiagPivotThresh': keyword_392228, 'ColPerm': keyword_392230, 'PanelSize': keyword_392232, 'Relax': keyword_392234}
    # Getting the type of 'dict' (line 304)
    dict_392226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'dict', False)
    # Calling dict(args, kwargs) (line 304)
    dict_call_result_392236 = invoke(stypy.reporting.localization.Localization(__file__, 304, 15), dict_392226, *[], **kwargs_392235)
    
    # Assigning a type to the variable '_options' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), '_options', dict_call_result_392236)
    
    # Type idiom detected: calculating its left and rigth part (line 306)
    # Getting the type of 'options' (line 306)
    options_392237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'options')
    # Getting the type of 'None' (line 306)
    None_392238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 22), 'None')
    
    (may_be_392239, more_types_in_union_392240) = may_not_be_none(options_392237, None_392238)

    if may_be_392239:

        if more_types_in_union_392240:
            # Runtime conditional SSA (line 306)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to update(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'options' (line 307)
        options_392243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 24), 'options', False)
        # Processing the call keyword arguments (line 307)
        kwargs_392244 = {}
        # Getting the type of '_options' (line 307)
        _options_392241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), '_options', False)
        # Obtaining the member 'update' of a type (line 307)
        update_392242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), _options_392241, 'update')
        # Calling update(args, kwargs) (line 307)
        update_call_result_392245 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), update_392242, *[options_392243], **kwargs_392244)
        

        if more_types_in_union_392240:
            # SSA join for if statement (line 306)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to gstrf(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'N' (line 308)
    N_392248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 26), 'N', False)
    # Getting the type of 'A' (line 308)
    A_392249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 29), 'A', False)
    # Obtaining the member 'nnz' of a type (line 308)
    nnz_392250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 29), A_392249, 'nnz')
    # Getting the type of 'A' (line 308)
    A_392251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 36), 'A', False)
    # Obtaining the member 'data' of a type (line 308)
    data_392252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 36), A_392251, 'data')
    # Getting the type of 'A' (line 308)
    A_392253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 44), 'A', False)
    # Obtaining the member 'indices' of a type (line 308)
    indices_392254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 44), A_392253, 'indices')
    # Getting the type of 'A' (line 308)
    A_392255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 55), 'A', False)
    # Obtaining the member 'indptr' of a type (line 308)
    indptr_392256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 55), A_392255, 'indptr')
    # Processing the call keyword arguments (line 308)
    # Getting the type of 'False' (line 309)
    False_392257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 30), 'False', False)
    keyword_392258 = False_392257
    # Getting the type of '_options' (line 309)
    _options_392259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 45), '_options', False)
    keyword_392260 = _options_392259
    kwargs_392261 = {'ilu': keyword_392258, 'options': keyword_392260}
    # Getting the type of '_superlu' (line 308)
    _superlu_392246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 11), '_superlu', False)
    # Obtaining the member 'gstrf' of a type (line 308)
    gstrf_392247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 11), _superlu_392246, 'gstrf')
    # Calling gstrf(args, kwargs) (line 308)
    gstrf_call_result_392262 = invoke(stypy.reporting.localization.Localization(__file__, 308, 11), gstrf_392247, *[N_392248, nnz_392250, data_392252, indices_392254, indptr_392256], **kwargs_392261)
    
    # Assigning a type to the variable 'stypy_return_type' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'stypy_return_type', gstrf_call_result_392262)
    
    # ################# End of 'splu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splu' in the type store
    # Getting the type of 'stypy_return_type' (line 227)
    stypy_return_type_392263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_392263)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splu'
    return stypy_return_type_392263

# Assigning a type to the variable 'splu' (line 227)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 0), 'splu', splu)

@norecursion
def spilu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 312)
    None_392264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 22), 'None')
    # Getting the type of 'None' (line 312)
    None_392265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 40), 'None')
    # Getting the type of 'None' (line 312)
    None_392266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 56), 'None')
    # Getting the type of 'None' (line 312)
    None_392267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 73), 'None')
    # Getting the type of 'None' (line 313)
    None_392268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 28), 'None')
    # Getting the type of 'None' (line 313)
    None_392269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 40), 'None')
    # Getting the type of 'None' (line 313)
    None_392270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 57), 'None')
    # Getting the type of 'None' (line 313)
    None_392271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 71), 'None')
    defaults = [None_392264, None_392265, None_392266, None_392267, None_392268, None_392269, None_392270, None_392271]
    # Create a new context for function 'spilu'
    module_type_store = module_type_store.open_function_context('spilu', 312, 0, False)
    
    # Passed parameters checking function
    spilu.stypy_localization = localization
    spilu.stypy_type_of_self = None
    spilu.stypy_type_store = module_type_store
    spilu.stypy_function_name = 'spilu'
    spilu.stypy_param_names_list = ['A', 'drop_tol', 'fill_factor', 'drop_rule', 'permc_spec', 'diag_pivot_thresh', 'relax', 'panel_size', 'options']
    spilu.stypy_varargs_param_name = None
    spilu.stypy_kwargs_param_name = None
    spilu.stypy_call_defaults = defaults
    spilu.stypy_call_varargs = varargs
    spilu.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spilu', ['A', 'drop_tol', 'fill_factor', 'drop_rule', 'permc_spec', 'diag_pivot_thresh', 'relax', 'panel_size', 'options'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spilu', localization, ['A', 'drop_tol', 'fill_factor', 'drop_rule', 'permc_spec', 'diag_pivot_thresh', 'relax', 'panel_size', 'options'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spilu(...)' code ##################

    str_392272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, (-1)), 'str', '\n    Compute an incomplete LU decomposition for a sparse, square matrix.\n\n    The resulting object is an approximation to the inverse of `A`.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Sparse matrix to factorize\n    drop_tol : float, optional\n        Drop tolerance (0 <= tol <= 1) for an incomplete LU decomposition.\n        (default: 1e-4)\n    fill_factor : float, optional\n        Specifies the fill ratio upper bound (>= 1.0) for ILU. (default: 10)\n    drop_rule : str, optional\n        Comma-separated string of drop rules to use.\n        Available rules: ``basic``, ``prows``, ``column``, ``area``,\n        ``secondary``, ``dynamic``, ``interp``. (Default: ``basic,area``)\n\n        See SuperLU documentation for details.\n\n    Remaining other options\n        Same as for `splu`\n\n    Returns\n    -------\n    invA_approx : scipy.sparse.linalg.SuperLU\n        Object, which has a ``solve`` method.\n\n    See also\n    --------\n    splu : complete LU decomposition\n\n    Notes\n    -----\n    To improve the better approximation to the inverse, you may need to\n    increase `fill_factor` AND decrease `drop_tol`.\n\n    This function uses the SuperLU library.\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import spilu\n    >>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)\n    >>> B = spilu(A)\n    >>> x = np.array([1., 2., 3.], dtype=float)\n    >>> B.solve(x)\n    array([ 1. , -3. , -1.5])\n    >>> A.dot(B.solve(x))\n    array([ 1.,  2.,  3.])\n    >>> B.solve(A.dot(x))\n    array([ 1.,  2.,  3.])\n    ')
    
    
    
    # Call to isspmatrix_csc(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'A' (line 368)
    A_392274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 26), 'A', False)
    # Processing the call keyword arguments (line 368)
    kwargs_392275 = {}
    # Getting the type of 'isspmatrix_csc' (line 368)
    isspmatrix_csc_392273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 11), 'isspmatrix_csc', False)
    # Calling isspmatrix_csc(args, kwargs) (line 368)
    isspmatrix_csc_call_result_392276 = invoke(stypy.reporting.localization.Localization(__file__, 368, 11), isspmatrix_csc_392273, *[A_392274], **kwargs_392275)
    
    # Applying the 'not' unary operator (line 368)
    result_not__392277 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 7), 'not', isspmatrix_csc_call_result_392276)
    
    # Testing the type of an if condition (line 368)
    if_condition_392278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 4), result_not__392277)
    # Assigning a type to the variable 'if_condition_392278' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'if_condition_392278', if_condition_392278)
    # SSA begins for if statement (line 368)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 369):
    
    # Assigning a Call to a Name (line 369):
    
    # Call to csc_matrix(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'A' (line 369)
    A_392280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 23), 'A', False)
    # Processing the call keyword arguments (line 369)
    kwargs_392281 = {}
    # Getting the type of 'csc_matrix' (line 369)
    csc_matrix_392279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 369)
    csc_matrix_call_result_392282 = invoke(stypy.reporting.localization.Localization(__file__, 369, 12), csc_matrix_392279, *[A_392280], **kwargs_392281)
    
    # Assigning a type to the variable 'A' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'A', csc_matrix_call_result_392282)
    
    # Call to warn(...): (line 370)
    # Processing the call arguments (line 370)
    str_392284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 13), 'str', 'splu requires CSC matrix format')
    # Getting the type of 'SparseEfficiencyWarning' (line 370)
    SparseEfficiencyWarning_392285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 48), 'SparseEfficiencyWarning', False)
    # Processing the call keyword arguments (line 370)
    kwargs_392286 = {}
    # Getting the type of 'warn' (line 370)
    warn_392283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 370)
    warn_call_result_392287 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), warn_392283, *[str_392284, SparseEfficiencyWarning_392285], **kwargs_392286)
    
    # SSA join for if statement (line 368)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to sort_indices(...): (line 372)
    # Processing the call keyword arguments (line 372)
    kwargs_392290 = {}
    # Getting the type of 'A' (line 372)
    A_392288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'A', False)
    # Obtaining the member 'sort_indices' of a type (line 372)
    sort_indices_392289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 4), A_392288, 'sort_indices')
    # Calling sort_indices(args, kwargs) (line 372)
    sort_indices_call_result_392291 = invoke(stypy.reporting.localization.Localization(__file__, 372, 4), sort_indices_392289, *[], **kwargs_392290)
    
    
    # Assigning a Call to a Name (line 373):
    
    # Assigning a Call to a Name (line 373):
    
    # Call to asfptype(...): (line 373)
    # Processing the call keyword arguments (line 373)
    kwargs_392294 = {}
    # Getting the type of 'A' (line 373)
    A_392292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'A', False)
    # Obtaining the member 'asfptype' of a type (line 373)
    asfptype_392293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 8), A_392292, 'asfptype')
    # Calling asfptype(args, kwargs) (line 373)
    asfptype_call_result_392295 = invoke(stypy.reporting.localization.Localization(__file__, 373, 8), asfptype_392293, *[], **kwargs_392294)
    
    # Assigning a type to the variable 'A' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'A', asfptype_call_result_392295)
    
    # Assigning a Attribute to a Tuple (line 375):
    
    # Assigning a Subscript to a Name (line 375):
    
    # Obtaining the type of the subscript
    int_392296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 4), 'int')
    # Getting the type of 'A' (line 375)
    A_392297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'A')
    # Obtaining the member 'shape' of a type (line 375)
    shape_392298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 11), A_392297, 'shape')
    # Obtaining the member '__getitem__' of a type (line 375)
    getitem___392299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 4), shape_392298, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 375)
    subscript_call_result_392300 = invoke(stypy.reporting.localization.Localization(__file__, 375, 4), getitem___392299, int_392296)
    
    # Assigning a type to the variable 'tuple_var_assignment_391664' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'tuple_var_assignment_391664', subscript_call_result_392300)
    
    # Assigning a Subscript to a Name (line 375):
    
    # Obtaining the type of the subscript
    int_392301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 4), 'int')
    # Getting the type of 'A' (line 375)
    A_392302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'A')
    # Obtaining the member 'shape' of a type (line 375)
    shape_392303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 11), A_392302, 'shape')
    # Obtaining the member '__getitem__' of a type (line 375)
    getitem___392304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 4), shape_392303, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 375)
    subscript_call_result_392305 = invoke(stypy.reporting.localization.Localization(__file__, 375, 4), getitem___392304, int_392301)
    
    # Assigning a type to the variable 'tuple_var_assignment_391665' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'tuple_var_assignment_391665', subscript_call_result_392305)
    
    # Assigning a Name to a Name (line 375):
    # Getting the type of 'tuple_var_assignment_391664' (line 375)
    tuple_var_assignment_391664_392306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'tuple_var_assignment_391664')
    # Assigning a type to the variable 'M' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'M', tuple_var_assignment_391664_392306)
    
    # Assigning a Name to a Name (line 375):
    # Getting the type of 'tuple_var_assignment_391665' (line 375)
    tuple_var_assignment_391665_392307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'tuple_var_assignment_391665')
    # Assigning a type to the variable 'N' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 7), 'N', tuple_var_assignment_391665_392307)
    
    
    # Getting the type of 'M' (line 376)
    M_392308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'M')
    # Getting the type of 'N' (line 376)
    N_392309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 13), 'N')
    # Applying the binary operator '!=' (line 376)
    result_ne_392310 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 8), '!=', M_392308, N_392309)
    
    # Testing the type of an if condition (line 376)
    if_condition_392311 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 4), result_ne_392310)
    # Assigning a type to the variable 'if_condition_392311' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'if_condition_392311', if_condition_392311)
    # SSA begins for if statement (line 376)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 377)
    # Processing the call arguments (line 377)
    str_392313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 25), 'str', 'can only factor square matrices')
    # Processing the call keyword arguments (line 377)
    kwargs_392314 = {}
    # Getting the type of 'ValueError' (line 377)
    ValueError_392312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 377)
    ValueError_call_result_392315 = invoke(stypy.reporting.localization.Localization(__file__, 377, 14), ValueError_392312, *[str_392313], **kwargs_392314)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 377, 8), ValueError_call_result_392315, 'raise parameter', BaseException)
    # SSA join for if statement (line 376)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 379):
    
    # Assigning a Call to a Name (line 379):
    
    # Call to dict(...): (line 379)
    # Processing the call keyword arguments (line 379)
    # Getting the type of 'drop_rule' (line 379)
    drop_rule_392317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 33), 'drop_rule', False)
    keyword_392318 = drop_rule_392317
    # Getting the type of 'drop_tol' (line 379)
    drop_tol_392319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 56), 'drop_tol', False)
    keyword_392320 = drop_tol_392319
    # Getting the type of 'fill_factor' (line 380)
    fill_factor_392321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 35), 'fill_factor', False)
    keyword_392322 = fill_factor_392321
    # Getting the type of 'diag_pivot_thresh' (line 381)
    diag_pivot_thresh_392323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 36), 'diag_pivot_thresh', False)
    keyword_392324 = diag_pivot_thresh_392323
    # Getting the type of 'permc_spec' (line 381)
    permc_spec_392325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 63), 'permc_spec', False)
    keyword_392326 = permc_spec_392325
    # Getting the type of 'panel_size' (line 382)
    panel_size_392327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 30), 'panel_size', False)
    keyword_392328 = panel_size_392327
    # Getting the type of 'relax' (line 382)
    relax_392329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 48), 'relax', False)
    keyword_392330 = relax_392329
    kwargs_392331 = {'ColPerm': keyword_392326, 'Relax': keyword_392330, 'PanelSize': keyword_392328, 'ILU_FillFactor': keyword_392322, 'ILU_DropRule': keyword_392318, 'DiagPivotThresh': keyword_392324, 'ILU_DropTol': keyword_392320}
    # Getting the type of 'dict' (line 379)
    dict_392316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'dict', False)
    # Calling dict(args, kwargs) (line 379)
    dict_call_result_392332 = invoke(stypy.reporting.localization.Localization(__file__, 379, 15), dict_392316, *[], **kwargs_392331)
    
    # Assigning a type to the variable '_options' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), '_options', dict_call_result_392332)
    
    # Type idiom detected: calculating its left and rigth part (line 383)
    # Getting the type of 'options' (line 383)
    options_392333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'options')
    # Getting the type of 'None' (line 383)
    None_392334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 22), 'None')
    
    (may_be_392335, more_types_in_union_392336) = may_not_be_none(options_392333, None_392334)

    if may_be_392335:

        if more_types_in_union_392336:
            # Runtime conditional SSA (line 383)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to update(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'options' (line 384)
        options_392339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 24), 'options', False)
        # Processing the call keyword arguments (line 384)
        kwargs_392340 = {}
        # Getting the type of '_options' (line 384)
        _options_392337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), '_options', False)
        # Obtaining the member 'update' of a type (line 384)
        update_392338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), _options_392337, 'update')
        # Calling update(args, kwargs) (line 384)
        update_call_result_392341 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), update_392338, *[options_392339], **kwargs_392340)
        

        if more_types_in_union_392336:
            # SSA join for if statement (line 383)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to gstrf(...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'N' (line 385)
    N_392344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 26), 'N', False)
    # Getting the type of 'A' (line 385)
    A_392345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 29), 'A', False)
    # Obtaining the member 'nnz' of a type (line 385)
    nnz_392346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 29), A_392345, 'nnz')
    # Getting the type of 'A' (line 385)
    A_392347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 36), 'A', False)
    # Obtaining the member 'data' of a type (line 385)
    data_392348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 36), A_392347, 'data')
    # Getting the type of 'A' (line 385)
    A_392349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 44), 'A', False)
    # Obtaining the member 'indices' of a type (line 385)
    indices_392350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 44), A_392349, 'indices')
    # Getting the type of 'A' (line 385)
    A_392351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 55), 'A', False)
    # Obtaining the member 'indptr' of a type (line 385)
    indptr_392352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 55), A_392351, 'indptr')
    # Processing the call keyword arguments (line 385)
    # Getting the type of 'True' (line 386)
    True_392353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 30), 'True', False)
    keyword_392354 = True_392353
    # Getting the type of '_options' (line 386)
    _options_392355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 44), '_options', False)
    keyword_392356 = _options_392355
    kwargs_392357 = {'ilu': keyword_392354, 'options': keyword_392356}
    # Getting the type of '_superlu' (line 385)
    _superlu_392342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 11), '_superlu', False)
    # Obtaining the member 'gstrf' of a type (line 385)
    gstrf_392343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 11), _superlu_392342, 'gstrf')
    # Calling gstrf(args, kwargs) (line 385)
    gstrf_call_result_392358 = invoke(stypy.reporting.localization.Localization(__file__, 385, 11), gstrf_392343, *[N_392344, nnz_392346, data_392348, indices_392350, indptr_392352], **kwargs_392357)
    
    # Assigning a type to the variable 'stypy_return_type' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'stypy_return_type', gstrf_call_result_392358)
    
    # ################# End of 'spilu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spilu' in the type store
    # Getting the type of 'stypy_return_type' (line 312)
    stypy_return_type_392359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_392359)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spilu'
    return stypy_return_type_392359

# Assigning a type to the variable 'spilu' (line 312)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'spilu', spilu)

@norecursion
def factorized(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'factorized'
    module_type_store = module_type_store.open_function_context('factorized', 389, 0, False)
    
    # Passed parameters checking function
    factorized.stypy_localization = localization
    factorized.stypy_type_of_self = None
    factorized.stypy_type_store = module_type_store
    factorized.stypy_function_name = 'factorized'
    factorized.stypy_param_names_list = ['A']
    factorized.stypy_varargs_param_name = None
    factorized.stypy_kwargs_param_name = None
    factorized.stypy_call_defaults = defaults
    factorized.stypy_call_varargs = varargs
    factorized.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'factorized', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'factorized', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'factorized(...)' code ##################

    str_392360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, (-1)), 'str', '\n    Return a function for solving a sparse linear system, with A pre-factorized.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Input.\n\n    Returns\n    -------\n    solve : callable\n        To solve the linear system of equations given in `A`, the `solve`\n        callable should be passed an ndarray of shape (N,).\n\n    Examples\n    --------\n    >>> from scipy.sparse.linalg import factorized\n    >>> A = np.array([[ 3. ,  2. , -1. ],\n    ...               [ 2. , -2. ,  4. ],\n    ...               [-1. ,  0.5, -1. ]])\n    >>> solve = factorized(A) # Makes LU decomposition.\n    >>> rhs1 = np.array([1, -2, 0])\n    >>> solve(rhs1) # Uses the LU factors.\n    array([ 1., -2., -2.])\n\n    ')
    
    # Getting the type of 'useUmfpack' (line 416)
    useUmfpack_392361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 7), 'useUmfpack')
    # Testing the type of an if condition (line 416)
    if_condition_392362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 416, 4), useUmfpack_392361)
    # Assigning a type to the variable 'if_condition_392362' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'if_condition_392362', if_condition_392362)
    # SSA begins for if statement (line 416)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'noScikit' (line 417)
    noScikit_392363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 11), 'noScikit')
    # Testing the type of an if condition (line 417)
    if_condition_392364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 417, 8), noScikit_392363)
    # Assigning a type to the variable 'if_condition_392364' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'if_condition_392364', if_condition_392364)
    # SSA begins for if statement (line 417)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 418)
    # Processing the call arguments (line 418)
    str_392366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 31), 'str', 'Scikits.umfpack not installed.')
    # Processing the call keyword arguments (line 418)
    kwargs_392367 = {}
    # Getting the type of 'RuntimeError' (line 418)
    RuntimeError_392365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 418)
    RuntimeError_call_result_392368 = invoke(stypy.reporting.localization.Localization(__file__, 418, 18), RuntimeError_392365, *[str_392366], **kwargs_392367)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 418, 12), RuntimeError_call_result_392368, 'raise parameter', BaseException)
    # SSA join for if statement (line 417)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to isspmatrix_csc(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'A' (line 420)
    A_392370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 30), 'A', False)
    # Processing the call keyword arguments (line 420)
    kwargs_392371 = {}
    # Getting the type of 'isspmatrix_csc' (line 420)
    isspmatrix_csc_392369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 15), 'isspmatrix_csc', False)
    # Calling isspmatrix_csc(args, kwargs) (line 420)
    isspmatrix_csc_call_result_392372 = invoke(stypy.reporting.localization.Localization(__file__, 420, 15), isspmatrix_csc_392369, *[A_392370], **kwargs_392371)
    
    # Applying the 'not' unary operator (line 420)
    result_not__392373 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 11), 'not', isspmatrix_csc_call_result_392372)
    
    # Testing the type of an if condition (line 420)
    if_condition_392374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 8), result_not__392373)
    # Assigning a type to the variable 'if_condition_392374' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'if_condition_392374', if_condition_392374)
    # SSA begins for if statement (line 420)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 421):
    
    # Assigning a Call to a Name (line 421):
    
    # Call to csc_matrix(...): (line 421)
    # Processing the call arguments (line 421)
    # Getting the type of 'A' (line 421)
    A_392376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 27), 'A', False)
    # Processing the call keyword arguments (line 421)
    kwargs_392377 = {}
    # Getting the type of 'csc_matrix' (line 421)
    csc_matrix_392375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 16), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 421)
    csc_matrix_call_result_392378 = invoke(stypy.reporting.localization.Localization(__file__, 421, 16), csc_matrix_392375, *[A_392376], **kwargs_392377)
    
    # Assigning a type to the variable 'A' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'A', csc_matrix_call_result_392378)
    
    # Call to warn(...): (line 422)
    # Processing the call arguments (line 422)
    str_392380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 17), 'str', 'splu requires CSC matrix format')
    # Getting the type of 'SparseEfficiencyWarning' (line 422)
    SparseEfficiencyWarning_392381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 52), 'SparseEfficiencyWarning', False)
    # Processing the call keyword arguments (line 422)
    kwargs_392382 = {}
    # Getting the type of 'warn' (line 422)
    warn_392379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'warn', False)
    # Calling warn(args, kwargs) (line 422)
    warn_call_result_392383 = invoke(stypy.reporting.localization.Localization(__file__, 422, 12), warn_392379, *[str_392380, SparseEfficiencyWarning_392381], **kwargs_392382)
    
    # SSA join for if statement (line 420)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 424):
    
    # Assigning a Call to a Name (line 424):
    
    # Call to asfptype(...): (line 424)
    # Processing the call keyword arguments (line 424)
    kwargs_392386 = {}
    # Getting the type of 'A' (line 424)
    A_392384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'A', False)
    # Obtaining the member 'asfptype' of a type (line 424)
    asfptype_392385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), A_392384, 'asfptype')
    # Calling asfptype(args, kwargs) (line 424)
    asfptype_call_result_392387 = invoke(stypy.reporting.localization.Localization(__file__, 424, 12), asfptype_392385, *[], **kwargs_392386)
    
    # Assigning a type to the variable 'A' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'A', asfptype_call_result_392387)
    
    
    # Getting the type of 'A' (line 426)
    A_392388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 11), 'A')
    # Obtaining the member 'dtype' of a type (line 426)
    dtype_392389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 11), A_392388, 'dtype')
    # Obtaining the member 'char' of a type (line 426)
    char_392390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 11), dtype_392389, 'char')
    str_392391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 31), 'str', 'dD')
    # Applying the binary operator 'notin' (line 426)
    result_contains_392392 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 11), 'notin', char_392390, str_392391)
    
    # Testing the type of an if condition (line 426)
    if_condition_392393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 426, 8), result_contains_392392)
    # Assigning a type to the variable 'if_condition_392393' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'if_condition_392393', if_condition_392393)
    # SSA begins for if statement (line 426)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 427)
    # Processing the call arguments (line 427)
    str_392395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 29), 'str', 'convert matrix data to double, please, using .astype(), or set linsolve.useUmfpack = False')
    # Processing the call keyword arguments (line 427)
    kwargs_392396 = {}
    # Getting the type of 'ValueError' (line 427)
    ValueError_392394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 427)
    ValueError_call_result_392397 = invoke(stypy.reporting.localization.Localization(__file__, 427, 18), ValueError_392394, *[str_392395], **kwargs_392396)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 427, 12), ValueError_call_result_392397, 'raise parameter', BaseException)
    # SSA join for if statement (line 426)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 430):
    
    # Assigning a Call to a Name (line 430):
    
    # Call to UmfpackContext(...): (line 430)
    # Processing the call arguments (line 430)
    
    # Call to _get_umf_family(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'A' (line 430)
    A_392401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 53), 'A', False)
    # Processing the call keyword arguments (line 430)
    kwargs_392402 = {}
    # Getting the type of '_get_umf_family' (line 430)
    _get_umf_family_392400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 37), '_get_umf_family', False)
    # Calling _get_umf_family(args, kwargs) (line 430)
    _get_umf_family_call_result_392403 = invoke(stypy.reporting.localization.Localization(__file__, 430, 37), _get_umf_family_392400, *[A_392401], **kwargs_392402)
    
    # Processing the call keyword arguments (line 430)
    kwargs_392404 = {}
    # Getting the type of 'umfpack' (line 430)
    umfpack_392398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 14), 'umfpack', False)
    # Obtaining the member 'UmfpackContext' of a type (line 430)
    UmfpackContext_392399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 14), umfpack_392398, 'UmfpackContext')
    # Calling UmfpackContext(args, kwargs) (line 430)
    UmfpackContext_call_result_392405 = invoke(stypy.reporting.localization.Localization(__file__, 430, 14), UmfpackContext_392399, *[_get_umf_family_call_result_392403], **kwargs_392404)
    
    # Assigning a type to the variable 'umf' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'umf', UmfpackContext_call_result_392405)
    
    # Call to numeric(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'A' (line 433)
    A_392408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'A', False)
    # Processing the call keyword arguments (line 433)
    kwargs_392409 = {}
    # Getting the type of 'umf' (line 433)
    umf_392406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'umf', False)
    # Obtaining the member 'numeric' of a type (line 433)
    numeric_392407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), umf_392406, 'numeric')
    # Calling numeric(args, kwargs) (line 433)
    numeric_call_result_392410 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), numeric_392407, *[A_392408], **kwargs_392409)
    

    @norecursion
    def solve(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'solve'
        module_type_store = module_type_store.open_function_context('solve', 435, 8, False)
        
        # Passed parameters checking function
        solve.stypy_localization = localization
        solve.stypy_type_of_self = None
        solve.stypy_type_store = module_type_store
        solve.stypy_function_name = 'solve'
        solve.stypy_param_names_list = ['b']
        solve.stypy_varargs_param_name = None
        solve.stypy_kwargs_param_name = None
        solve.stypy_call_defaults = defaults
        solve.stypy_call_varargs = varargs
        solve.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'solve', ['b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'solve', localization, ['b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'solve(...)' code ##################

        
        # Call to solve(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'umfpack' (line 436)
        umfpack_392413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 29), 'umfpack', False)
        # Obtaining the member 'UMFPACK_A' of a type (line 436)
        UMFPACK_A_392414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 29), umfpack_392413, 'UMFPACK_A')
        # Getting the type of 'A' (line 436)
        A_392415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 48), 'A', False)
        # Getting the type of 'b' (line 436)
        b_392416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 51), 'b', False)
        # Processing the call keyword arguments (line 436)
        # Getting the type of 'True' (line 436)
        True_392417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 68), 'True', False)
        keyword_392418 = True_392417
        kwargs_392419 = {'autoTranspose': keyword_392418}
        # Getting the type of 'umf' (line 436)
        umf_392411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 19), 'umf', False)
        # Obtaining the member 'solve' of a type (line 436)
        solve_392412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 19), umf_392411, 'solve')
        # Calling solve(args, kwargs) (line 436)
        solve_call_result_392420 = invoke(stypy.reporting.localization.Localization(__file__, 436, 19), solve_392412, *[UMFPACK_A_392414, A_392415, b_392416], **kwargs_392419)
        
        # Assigning a type to the variable 'stypy_return_type' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'stypy_return_type', solve_call_result_392420)
        
        # ################# End of 'solve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'solve' in the type store
        # Getting the type of 'stypy_return_type' (line 435)
        stypy_return_type_392421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_392421)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'solve'
        return stypy_return_type_392421

    # Assigning a type to the variable 'solve' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'solve', solve)
    # Getting the type of 'solve' (line 438)
    solve_392422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 15), 'solve')
    # Assigning a type to the variable 'stypy_return_type' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'stypy_return_type', solve_392422)
    # SSA branch for the else part of an if statement (line 416)
    module_type_store.open_ssa_branch('else')
    
    # Call to splu(...): (line 440)
    # Processing the call arguments (line 440)
    # Getting the type of 'A' (line 440)
    A_392424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 20), 'A', False)
    # Processing the call keyword arguments (line 440)
    kwargs_392425 = {}
    # Getting the type of 'splu' (line 440)
    splu_392423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'splu', False)
    # Calling splu(args, kwargs) (line 440)
    splu_call_result_392426 = invoke(stypy.reporting.localization.Localization(__file__, 440, 15), splu_392423, *[A_392424], **kwargs_392425)
    
    # Obtaining the member 'solve' of a type (line 440)
    solve_392427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 15), splu_call_result_392426, 'solve')
    # Assigning a type to the variable 'stypy_return_type' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'stypy_return_type', solve_392427)
    # SSA join for if statement (line 416)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'factorized(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'factorized' in the type store
    # Getting the type of 'stypy_return_type' (line 389)
    stypy_return_type_392428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_392428)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'factorized'
    return stypy_return_type_392428

# Assigning a type to the variable 'factorized' (line 389)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 0), 'factorized', factorized)

@norecursion
def spsolve_triangular(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 443)
    True_392429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 35), 'True')
    # Getting the type of 'False' (line 443)
    False_392430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 53), 'False')
    # Getting the type of 'False' (line 443)
    False_392431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 72), 'False')
    defaults = [True_392429, False_392430, False_392431]
    # Create a new context for function 'spsolve_triangular'
    module_type_store = module_type_store.open_function_context('spsolve_triangular', 443, 0, False)
    
    # Passed parameters checking function
    spsolve_triangular.stypy_localization = localization
    spsolve_triangular.stypy_type_of_self = None
    spsolve_triangular.stypy_type_store = module_type_store
    spsolve_triangular.stypy_function_name = 'spsolve_triangular'
    spsolve_triangular.stypy_param_names_list = ['A', 'b', 'lower', 'overwrite_A', 'overwrite_b']
    spsolve_triangular.stypy_varargs_param_name = None
    spsolve_triangular.stypy_kwargs_param_name = None
    spsolve_triangular.stypy_call_defaults = defaults
    spsolve_triangular.stypy_call_varargs = varargs
    spsolve_triangular.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spsolve_triangular', ['A', 'b', 'lower', 'overwrite_A', 'overwrite_b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spsolve_triangular', localization, ['A', 'b', 'lower', 'overwrite_A', 'overwrite_b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spsolve_triangular(...)' code ##################

    str_392432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, (-1)), 'str', '\n    Solve the equation `A x = b` for `x`, assuming A is a triangular matrix.\n\n    Parameters\n    ----------\n    A : (M, M) sparse matrix\n        A sparse square triangular matrix. Should be in CSR format.\n    b : (M,) or (M, N) array_like\n        Right-hand side matrix in `A x = b`\n    lower : bool, optional\n        Whether `A` is a lower or upper triangular matrix.\n        Default is lower triangular matrix.\n    overwrite_A : bool, optional\n        Allow changing `A`. The indices of `A` are going to be sorted and zero\n        entries are going to be removed.\n        Enabling gives a performance gain. Default is False.\n    overwrite_b : bool, optional\n        Allow overwriting data in `b`.\n        Enabling gives a performance gain. Default is False.\n        If `overwrite_b` is True, it should be ensured that\n        `b` has an appropriate dtype to be able to store the result.\n\n    Returns\n    -------\n    x : (M,) or (M, N) ndarray\n        Solution to the system `A x = b`.  Shape of return matches shape of `b`.\n\n    Raises\n    ------\n    LinAlgError\n        If `A` is singular or not triangular.\n    ValueError\n        If shape of `A` or shape of `b` do not match the requirements.\n\n    Notes\n    -----\n    .. versionadded:: 0.19.0\n\n    Examples\n    --------\n    >>> from scipy.sparse import csr_matrix\n    >>> from scipy.sparse.linalg import spsolve_triangular\n    >>> A = csr_matrix([[3, 0, 0], [1, -1, 0], [2, 0, 1]], dtype=float)\n    >>> B = np.array([[2, 0], [-1, 0], [2, 0]], dtype=float)\n    >>> x = spsolve_triangular(A, B)\n    >>> np.allclose(A.dot(x), B)\n    True\n    ')
    
    
    
    # Call to isspmatrix_csr(...): (line 494)
    # Processing the call arguments (line 494)
    # Getting the type of 'A' (line 494)
    A_392434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 26), 'A', False)
    # Processing the call keyword arguments (line 494)
    kwargs_392435 = {}
    # Getting the type of 'isspmatrix_csr' (line 494)
    isspmatrix_csr_392433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 11), 'isspmatrix_csr', False)
    # Calling isspmatrix_csr(args, kwargs) (line 494)
    isspmatrix_csr_call_result_392436 = invoke(stypy.reporting.localization.Localization(__file__, 494, 11), isspmatrix_csr_392433, *[A_392434], **kwargs_392435)
    
    # Applying the 'not' unary operator (line 494)
    result_not__392437 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 7), 'not', isspmatrix_csr_call_result_392436)
    
    # Testing the type of an if condition (line 494)
    if_condition_392438 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 494, 4), result_not__392437)
    # Assigning a type to the variable 'if_condition_392438' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'if_condition_392438', if_condition_392438)
    # SSA begins for if statement (line 494)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 495)
    # Processing the call arguments (line 495)
    str_392440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 13), 'str', 'CSR matrix format is required. Converting to CSR matrix.')
    # Getting the type of 'SparseEfficiencyWarning' (line 496)
    SparseEfficiencyWarning_392441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 13), 'SparseEfficiencyWarning', False)
    # Processing the call keyword arguments (line 495)
    kwargs_392442 = {}
    # Getting the type of 'warn' (line 495)
    warn_392439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 495)
    warn_call_result_392443 = invoke(stypy.reporting.localization.Localization(__file__, 495, 8), warn_392439, *[str_392440, SparseEfficiencyWarning_392441], **kwargs_392442)
    
    
    # Assigning a Call to a Name (line 497):
    
    # Assigning a Call to a Name (line 497):
    
    # Call to csr_matrix(...): (line 497)
    # Processing the call arguments (line 497)
    # Getting the type of 'A' (line 497)
    A_392445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 23), 'A', False)
    # Processing the call keyword arguments (line 497)
    kwargs_392446 = {}
    # Getting the type of 'csr_matrix' (line 497)
    csr_matrix_392444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 497)
    csr_matrix_call_result_392447 = invoke(stypy.reporting.localization.Localization(__file__, 497, 12), csr_matrix_392444, *[A_392445], **kwargs_392446)
    
    # Assigning a type to the variable 'A' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'A', csr_matrix_call_result_392447)
    # SSA branch for the else part of an if statement (line 494)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'overwrite_A' (line 498)
    overwrite_A_392448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 13), 'overwrite_A')
    # Applying the 'not' unary operator (line 498)
    result_not__392449 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 9), 'not', overwrite_A_392448)
    
    # Testing the type of an if condition (line 498)
    if_condition_392450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 498, 9), result_not__392449)
    # Assigning a type to the variable 'if_condition_392450' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 9), 'if_condition_392450', if_condition_392450)
    # SSA begins for if statement (line 498)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 499):
    
    # Assigning a Call to a Name (line 499):
    
    # Call to copy(...): (line 499)
    # Processing the call keyword arguments (line 499)
    kwargs_392453 = {}
    # Getting the type of 'A' (line 499)
    A_392451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 12), 'A', False)
    # Obtaining the member 'copy' of a type (line 499)
    copy_392452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 12), A_392451, 'copy')
    # Calling copy(args, kwargs) (line 499)
    copy_call_result_392454 = invoke(stypy.reporting.localization.Localization(__file__, 499, 12), copy_392452, *[], **kwargs_392453)
    
    # Assigning a type to the variable 'A' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'A', copy_call_result_392454)
    # SSA join for if statement (line 498)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 494)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_392455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 15), 'int')
    # Getting the type of 'A' (line 501)
    A_392456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 7), 'A')
    # Obtaining the member 'shape' of a type (line 501)
    shape_392457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 7), A_392456, 'shape')
    # Obtaining the member '__getitem__' of a type (line 501)
    getitem___392458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 7), shape_392457, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 501)
    subscript_call_result_392459 = invoke(stypy.reporting.localization.Localization(__file__, 501, 7), getitem___392458, int_392455)
    
    
    # Obtaining the type of the subscript
    int_392460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 29), 'int')
    # Getting the type of 'A' (line 501)
    A_392461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 21), 'A')
    # Obtaining the member 'shape' of a type (line 501)
    shape_392462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 21), A_392461, 'shape')
    # Obtaining the member '__getitem__' of a type (line 501)
    getitem___392463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 21), shape_392462, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 501)
    subscript_call_result_392464 = invoke(stypy.reporting.localization.Localization(__file__, 501, 21), getitem___392463, int_392460)
    
    # Applying the binary operator '!=' (line 501)
    result_ne_392465 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 7), '!=', subscript_call_result_392459, subscript_call_result_392464)
    
    # Testing the type of an if condition (line 501)
    if_condition_392466 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 501, 4), result_ne_392465)
    # Assigning a type to the variable 'if_condition_392466' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'if_condition_392466', if_condition_392466)
    # SSA begins for if statement (line 501)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 502)
    # Processing the call arguments (line 502)
    
    # Call to format(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'A' (line 503)
    A_392470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 68), 'A', False)
    # Obtaining the member 'shape' of a type (line 503)
    shape_392471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 68), A_392470, 'shape')
    # Processing the call keyword arguments (line 503)
    kwargs_392472 = {}
    str_392468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 12), 'str', 'A must be a square matrix but its shape is {}.')
    # Obtaining the member 'format' of a type (line 503)
    format_392469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 12), str_392468, 'format')
    # Calling format(args, kwargs) (line 503)
    format_call_result_392473 = invoke(stypy.reporting.localization.Localization(__file__, 503, 12), format_392469, *[shape_392471], **kwargs_392472)
    
    # Processing the call keyword arguments (line 502)
    kwargs_392474 = {}
    # Getting the type of 'ValueError' (line 502)
    ValueError_392467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 502)
    ValueError_call_result_392475 = invoke(stypy.reporting.localization.Localization(__file__, 502, 14), ValueError_392467, *[format_call_result_392473], **kwargs_392474)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 502, 8), ValueError_call_result_392475, 'raise parameter', BaseException)
    # SSA join for if statement (line 501)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to eliminate_zeros(...): (line 505)
    # Processing the call keyword arguments (line 505)
    kwargs_392478 = {}
    # Getting the type of 'A' (line 505)
    A_392476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'A', False)
    # Obtaining the member 'eliminate_zeros' of a type (line 505)
    eliminate_zeros_392477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 4), A_392476, 'eliminate_zeros')
    # Calling eliminate_zeros(args, kwargs) (line 505)
    eliminate_zeros_call_result_392479 = invoke(stypy.reporting.localization.Localization(__file__, 505, 4), eliminate_zeros_392477, *[], **kwargs_392478)
    
    
    # Call to sort_indices(...): (line 506)
    # Processing the call keyword arguments (line 506)
    kwargs_392482 = {}
    # Getting the type of 'A' (line 506)
    A_392480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'A', False)
    # Obtaining the member 'sort_indices' of a type (line 506)
    sort_indices_392481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 4), A_392480, 'sort_indices')
    # Calling sort_indices(args, kwargs) (line 506)
    sort_indices_call_result_392483 = invoke(stypy.reporting.localization.Localization(__file__, 506, 4), sort_indices_392481, *[], **kwargs_392482)
    
    
    # Assigning a Call to a Name (line 508):
    
    # Assigning a Call to a Name (line 508):
    
    # Call to asanyarray(...): (line 508)
    # Processing the call arguments (line 508)
    # Getting the type of 'b' (line 508)
    b_392486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 22), 'b', False)
    # Processing the call keyword arguments (line 508)
    kwargs_392487 = {}
    # Getting the type of 'np' (line 508)
    np_392484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 508)
    asanyarray_392485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 8), np_392484, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 508)
    asanyarray_call_result_392488 = invoke(stypy.reporting.localization.Localization(__file__, 508, 8), asanyarray_392485, *[b_392486], **kwargs_392487)
    
    # Assigning a type to the variable 'b' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'b', asanyarray_call_result_392488)
    
    
    # Getting the type of 'b' (line 510)
    b_392489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 7), 'b')
    # Obtaining the member 'ndim' of a type (line 510)
    ndim_392490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 7), b_392489, 'ndim')
    
    # Obtaining an instance of the builtin type 'list' (line 510)
    list_392491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 510)
    # Adding element type (line 510)
    int_392492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 21), list_392491, int_392492)
    # Adding element type (line 510)
    int_392493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 21), list_392491, int_392493)
    
    # Applying the binary operator 'notin' (line 510)
    result_contains_392494 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 7), 'notin', ndim_392490, list_392491)
    
    # Testing the type of an if condition (line 510)
    if_condition_392495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 510, 4), result_contains_392494)
    # Assigning a type to the variable 'if_condition_392495' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'if_condition_392495', if_condition_392495)
    # SSA begins for if statement (line 510)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 511)
    # Processing the call arguments (line 511)
    
    # Call to format(...): (line 512)
    # Processing the call arguments (line 512)
    # Getting the type of 'b' (line 512)
    b_392499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 66), 'b', False)
    # Obtaining the member 'shape' of a type (line 512)
    shape_392500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 66), b_392499, 'shape')
    # Processing the call keyword arguments (line 512)
    kwargs_392501 = {}
    str_392497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 12), 'str', 'b must have 1 or 2 dims but its shape is {}.')
    # Obtaining the member 'format' of a type (line 512)
    format_392498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 12), str_392497, 'format')
    # Calling format(args, kwargs) (line 512)
    format_call_result_392502 = invoke(stypy.reporting.localization.Localization(__file__, 512, 12), format_392498, *[shape_392500], **kwargs_392501)
    
    # Processing the call keyword arguments (line 511)
    kwargs_392503 = {}
    # Getting the type of 'ValueError' (line 511)
    ValueError_392496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 511)
    ValueError_call_result_392504 = invoke(stypy.reporting.localization.Localization(__file__, 511, 14), ValueError_392496, *[format_call_result_392502], **kwargs_392503)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 511, 8), ValueError_call_result_392504, 'raise parameter', BaseException)
    # SSA join for if statement (line 510)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_392505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 15), 'int')
    # Getting the type of 'A' (line 513)
    A_392506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 7), 'A')
    # Obtaining the member 'shape' of a type (line 513)
    shape_392507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 7), A_392506, 'shape')
    # Obtaining the member '__getitem__' of a type (line 513)
    getitem___392508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 7), shape_392507, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 513)
    subscript_call_result_392509 = invoke(stypy.reporting.localization.Localization(__file__, 513, 7), getitem___392508, int_392505)
    
    
    # Obtaining the type of the subscript
    int_392510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 29), 'int')
    # Getting the type of 'b' (line 513)
    b_392511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 21), 'b')
    # Obtaining the member 'shape' of a type (line 513)
    shape_392512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 21), b_392511, 'shape')
    # Obtaining the member '__getitem__' of a type (line 513)
    getitem___392513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 21), shape_392512, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 513)
    subscript_call_result_392514 = invoke(stypy.reporting.localization.Localization(__file__, 513, 21), getitem___392513, int_392510)
    
    # Applying the binary operator '!=' (line 513)
    result_ne_392515 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 7), '!=', subscript_call_result_392509, subscript_call_result_392514)
    
    # Testing the type of an if condition (line 513)
    if_condition_392516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 513, 4), result_ne_392515)
    # Assigning a type to the variable 'if_condition_392516' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'if_condition_392516', if_condition_392516)
    # SSA begins for if statement (line 513)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 514)
    # Processing the call arguments (line 514)
    
    # Call to format(...): (line 515)
    # Processing the call arguments (line 515)
    # Getting the type of 'A' (line 517)
    A_392520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 50), 'A', False)
    # Obtaining the member 'shape' of a type (line 517)
    shape_392521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 50), A_392520, 'shape')
    # Getting the type of 'b' (line 517)
    b_392522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 59), 'b', False)
    # Obtaining the member 'shape' of a type (line 517)
    shape_392523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 59), b_392522, 'shape')
    # Processing the call keyword arguments (line 515)
    kwargs_392524 = {}
    str_392518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 12), 'str', 'The size of the dimensions of A must be equal to the size of the first dimension of b but the shape of A is {} and the shape of b is {}.')
    # Obtaining the member 'format' of a type (line 515)
    format_392519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 12), str_392518, 'format')
    # Calling format(args, kwargs) (line 515)
    format_call_result_392525 = invoke(stypy.reporting.localization.Localization(__file__, 515, 12), format_392519, *[shape_392521, shape_392523], **kwargs_392524)
    
    # Processing the call keyword arguments (line 514)
    kwargs_392526 = {}
    # Getting the type of 'ValueError' (line 514)
    ValueError_392517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 514)
    ValueError_call_result_392527 = invoke(stypy.reporting.localization.Localization(__file__, 514, 14), ValueError_392517, *[format_call_result_392525], **kwargs_392526)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 514, 8), ValueError_call_result_392527, 'raise parameter', BaseException)
    # SSA join for if statement (line 513)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 520):
    
    # Assigning a Call to a Name (line 520):
    
    # Call to result_type(...): (line 520)
    # Processing the call arguments (line 520)
    # Getting the type of 'A' (line 520)
    A_392530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 29), 'A', False)
    # Obtaining the member 'data' of a type (line 520)
    data_392531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 29), A_392530, 'data')
    # Getting the type of 'b' (line 520)
    b_392532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 37), 'b', False)
    # Getting the type of 'np' (line 520)
    np_392533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 40), 'np', False)
    # Obtaining the member 'float' of a type (line 520)
    float_392534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 40), np_392533, 'float')
    # Processing the call keyword arguments (line 520)
    kwargs_392535 = {}
    # Getting the type of 'np' (line 520)
    np_392528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 14), 'np', False)
    # Obtaining the member 'result_type' of a type (line 520)
    result_type_392529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 14), np_392528, 'result_type')
    # Calling result_type(args, kwargs) (line 520)
    result_type_call_result_392536 = invoke(stypy.reporting.localization.Localization(__file__, 520, 14), result_type_392529, *[data_392531, b_392532, float_392534], **kwargs_392535)
    
    # Assigning a type to the variable 'x_dtype' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'x_dtype', result_type_call_result_392536)
    
    # Getting the type of 'overwrite_b' (line 521)
    overwrite_b_392537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 7), 'overwrite_b')
    # Testing the type of an if condition (line 521)
    if_condition_392538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 521, 4), overwrite_b_392537)
    # Assigning a type to the variable 'if_condition_392538' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'if_condition_392538', if_condition_392538)
    # SSA begins for if statement (line 521)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to can_cast(...): (line 522)
    # Processing the call arguments (line 522)
    # Getting the type of 'b' (line 522)
    b_392541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 23), 'b', False)
    # Obtaining the member 'dtype' of a type (line 522)
    dtype_392542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 23), b_392541, 'dtype')
    # Getting the type of 'x_dtype' (line 522)
    x_dtype_392543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 32), 'x_dtype', False)
    # Processing the call keyword arguments (line 522)
    str_392544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 49), 'str', 'same_kind')
    keyword_392545 = str_392544
    kwargs_392546 = {'casting': keyword_392545}
    # Getting the type of 'np' (line 522)
    np_392539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 11), 'np', False)
    # Obtaining the member 'can_cast' of a type (line 522)
    can_cast_392540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 11), np_392539, 'can_cast')
    # Calling can_cast(args, kwargs) (line 522)
    can_cast_call_result_392547 = invoke(stypy.reporting.localization.Localization(__file__, 522, 11), can_cast_392540, *[dtype_392542, x_dtype_392543], **kwargs_392546)
    
    # Testing the type of an if condition (line 522)
    if_condition_392548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 8), can_cast_call_result_392547)
    # Assigning a type to the variable 'if_condition_392548' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'if_condition_392548', if_condition_392548)
    # SSA begins for if statement (line 522)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 523):
    
    # Assigning a Name to a Name (line 523):
    # Getting the type of 'b' (line 523)
    b_392549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 'b')
    # Assigning a type to the variable 'x' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'x', b_392549)
    # SSA branch for the else part of an if statement (line 522)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 525)
    # Processing the call arguments (line 525)
    
    # Call to format(...): (line 526)
    # Processing the call arguments (line 526)
    # Getting the type of 'b' (line 527)
    b_392553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 37), 'b', False)
    # Obtaining the member 'dtype' of a type (line 527)
    dtype_392554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 37), b_392553, 'dtype')
    # Getting the type of 'x_dtype' (line 527)
    x_dtype_392555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 46), 'x_dtype', False)
    # Processing the call keyword arguments (line 526)
    kwargs_392556 = {}
    str_392551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 16), 'str', 'Cannot overwrite b (dtype {}) with result of type {}.')
    # Obtaining the member 'format' of a type (line 526)
    format_392552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 16), str_392551, 'format')
    # Calling format(args, kwargs) (line 526)
    format_call_result_392557 = invoke(stypy.reporting.localization.Localization(__file__, 526, 16), format_392552, *[dtype_392554, x_dtype_392555], **kwargs_392556)
    
    # Processing the call keyword arguments (line 525)
    kwargs_392558 = {}
    # Getting the type of 'ValueError' (line 525)
    ValueError_392550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 525)
    ValueError_call_result_392559 = invoke(stypy.reporting.localization.Localization(__file__, 525, 18), ValueError_392550, *[format_call_result_392557], **kwargs_392558)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 525, 12), ValueError_call_result_392559, 'raise parameter', BaseException)
    # SSA join for if statement (line 522)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 521)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 529):
    
    # Assigning a Call to a Name (line 529):
    
    # Call to astype(...): (line 529)
    # Processing the call arguments (line 529)
    # Getting the type of 'x_dtype' (line 529)
    x_dtype_392562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 21), 'x_dtype', False)
    # Processing the call keyword arguments (line 529)
    # Getting the type of 'True' (line 529)
    True_392563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 35), 'True', False)
    keyword_392564 = True_392563
    kwargs_392565 = {'copy': keyword_392564}
    # Getting the type of 'b' (line 529)
    b_392560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'b', False)
    # Obtaining the member 'astype' of a type (line 529)
    astype_392561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 12), b_392560, 'astype')
    # Calling astype(args, kwargs) (line 529)
    astype_call_result_392566 = invoke(stypy.reporting.localization.Localization(__file__, 529, 12), astype_392561, *[x_dtype_392562], **kwargs_392565)
    
    # Assigning a type to the variable 'x' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'x', astype_call_result_392566)
    # SSA join for if statement (line 521)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'lower' (line 532)
    lower_392567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 7), 'lower')
    # Testing the type of an if condition (line 532)
    if_condition_392568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 532, 4), lower_392567)
    # Assigning a type to the variable 'if_condition_392568' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'if_condition_392568', if_condition_392568)
    # SSA begins for if statement (line 532)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 533):
    
    # Assigning a Call to a Name (line 533):
    
    # Call to range(...): (line 533)
    # Processing the call arguments (line 533)
    
    # Call to len(...): (line 533)
    # Processing the call arguments (line 533)
    # Getting the type of 'b' (line 533)
    b_392571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 32), 'b', False)
    # Processing the call keyword arguments (line 533)
    kwargs_392572 = {}
    # Getting the type of 'len' (line 533)
    len_392570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 28), 'len', False)
    # Calling len(args, kwargs) (line 533)
    len_call_result_392573 = invoke(stypy.reporting.localization.Localization(__file__, 533, 28), len_392570, *[b_392571], **kwargs_392572)
    
    # Processing the call keyword arguments (line 533)
    kwargs_392574 = {}
    # Getting the type of 'range' (line 533)
    range_392569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 22), 'range', False)
    # Calling range(args, kwargs) (line 533)
    range_call_result_392575 = invoke(stypy.reporting.localization.Localization(__file__, 533, 22), range_392569, *[len_call_result_392573], **kwargs_392574)
    
    # Assigning a type to the variable 'row_indices' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'row_indices', range_call_result_392575)
    # SSA branch for the else part of an if statement (line 532)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 535):
    
    # Assigning a Call to a Name (line 535):
    
    # Call to range(...): (line 535)
    # Processing the call arguments (line 535)
    
    # Call to len(...): (line 535)
    # Processing the call arguments (line 535)
    # Getting the type of 'b' (line 535)
    b_392578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 32), 'b', False)
    # Processing the call keyword arguments (line 535)
    kwargs_392579 = {}
    # Getting the type of 'len' (line 535)
    len_392577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 28), 'len', False)
    # Calling len(args, kwargs) (line 535)
    len_call_result_392580 = invoke(stypy.reporting.localization.Localization(__file__, 535, 28), len_392577, *[b_392578], **kwargs_392579)
    
    int_392581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 37), 'int')
    # Applying the binary operator '-' (line 535)
    result_sub_392582 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 28), '-', len_call_result_392580, int_392581)
    
    int_392583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 40), 'int')
    int_392584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 44), 'int')
    # Processing the call keyword arguments (line 535)
    kwargs_392585 = {}
    # Getting the type of 'range' (line 535)
    range_392576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 22), 'range', False)
    # Calling range(args, kwargs) (line 535)
    range_call_result_392586 = invoke(stypy.reporting.localization.Localization(__file__, 535, 22), range_392576, *[result_sub_392582, int_392583, int_392584], **kwargs_392585)
    
    # Assigning a type to the variable 'row_indices' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'row_indices', range_call_result_392586)
    # SSA join for if statement (line 532)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'row_indices' (line 538)
    row_indices_392587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 13), 'row_indices')
    # Testing the type of a for loop iterable (line 538)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 538, 4), row_indices_392587)
    # Getting the type of the for loop variable (line 538)
    for_loop_var_392588 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 538, 4), row_indices_392587)
    # Assigning a type to the variable 'i' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'i', for_loop_var_392588)
    # SSA begins for a for statement (line 538)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 541):
    
    # Assigning a Subscript to a Name (line 541):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 541)
    i_392589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 32), 'i')
    # Getting the type of 'A' (line 541)
    A_392590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 23), 'A')
    # Obtaining the member 'indptr' of a type (line 541)
    indptr_392591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 23), A_392590, 'indptr')
    # Obtaining the member '__getitem__' of a type (line 541)
    getitem___392592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 23), indptr_392591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 541)
    subscript_call_result_392593 = invoke(stypy.reporting.localization.Localization(__file__, 541, 23), getitem___392592, i_392589)
    
    # Assigning a type to the variable 'indptr_start' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'indptr_start', subscript_call_result_392593)
    
    # Assigning a Subscript to a Name (line 542):
    
    # Assigning a Subscript to a Name (line 542):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 542)
    i_392594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 31), 'i')
    int_392595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 35), 'int')
    # Applying the binary operator '+' (line 542)
    result_add_392596 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 31), '+', i_392594, int_392595)
    
    # Getting the type of 'A' (line 542)
    A_392597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 22), 'A')
    # Obtaining the member 'indptr' of a type (line 542)
    indptr_392598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 22), A_392597, 'indptr')
    # Obtaining the member '__getitem__' of a type (line 542)
    getitem___392599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 22), indptr_392598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 542)
    subscript_call_result_392600 = invoke(stypy.reporting.localization.Localization(__file__, 542, 22), getitem___392599, result_add_392596)
    
    # Assigning a type to the variable 'indptr_stop' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'indptr_stop', subscript_call_result_392600)
    
    # Getting the type of 'lower' (line 543)
    lower_392601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 11), 'lower')
    # Testing the type of an if condition (line 543)
    if_condition_392602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 543, 8), lower_392601)
    # Assigning a type to the variable 'if_condition_392602' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'if_condition_392602', if_condition_392602)
    # SSA begins for if statement (line 543)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 544):
    
    # Assigning a BinOp to a Name (line 544):
    # Getting the type of 'indptr_stop' (line 544)
    indptr_stop_392603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 37), 'indptr_stop')
    int_392604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 51), 'int')
    # Applying the binary operator '-' (line 544)
    result_sub_392605 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 37), '-', indptr_stop_392603, int_392604)
    
    # Assigning a type to the variable 'A_diagonal_index_row_i' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'A_diagonal_index_row_i', result_sub_392605)
    
    # Assigning a Call to a Name (line 545):
    
    # Assigning a Call to a Name (line 545):
    
    # Call to slice(...): (line 545)
    # Processing the call arguments (line 545)
    # Getting the type of 'indptr_start' (line 545)
    indptr_start_392607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 49), 'indptr_start', False)
    # Getting the type of 'indptr_stop' (line 545)
    indptr_stop_392608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 63), 'indptr_stop', False)
    int_392609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 77), 'int')
    # Applying the binary operator '-' (line 545)
    result_sub_392610 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 63), '-', indptr_stop_392608, int_392609)
    
    # Processing the call keyword arguments (line 545)
    kwargs_392611 = {}
    # Getting the type of 'slice' (line 545)
    slice_392606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 43), 'slice', False)
    # Calling slice(args, kwargs) (line 545)
    slice_call_result_392612 = invoke(stypy.reporting.localization.Localization(__file__, 545, 43), slice_392606, *[indptr_start_392607, result_sub_392610], **kwargs_392611)
    
    # Assigning a type to the variable 'A_off_diagonal_indices_row_i' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'A_off_diagonal_indices_row_i', slice_call_result_392612)
    # SSA branch for the else part of an if statement (line 543)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 547):
    
    # Assigning a Name to a Name (line 547):
    # Getting the type of 'indptr_start' (line 547)
    indptr_start_392613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 37), 'indptr_start')
    # Assigning a type to the variable 'A_diagonal_index_row_i' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'A_diagonal_index_row_i', indptr_start_392613)
    
    # Assigning a Call to a Name (line 548):
    
    # Assigning a Call to a Name (line 548):
    
    # Call to slice(...): (line 548)
    # Processing the call arguments (line 548)
    # Getting the type of 'indptr_start' (line 548)
    indptr_start_392615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 49), 'indptr_start', False)
    int_392616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 64), 'int')
    # Applying the binary operator '+' (line 548)
    result_add_392617 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 49), '+', indptr_start_392615, int_392616)
    
    # Getting the type of 'indptr_stop' (line 548)
    indptr_stop_392618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 67), 'indptr_stop', False)
    # Processing the call keyword arguments (line 548)
    kwargs_392619 = {}
    # Getting the type of 'slice' (line 548)
    slice_392614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 43), 'slice', False)
    # Calling slice(args, kwargs) (line 548)
    slice_call_result_392620 = invoke(stypy.reporting.localization.Localization(__file__, 548, 43), slice_392614, *[result_add_392617, indptr_stop_392618], **kwargs_392619)
    
    # Assigning a type to the variable 'A_off_diagonal_indices_row_i' (line 548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'A_off_diagonal_indices_row_i', slice_call_result_392620)
    # SSA join for if statement (line 543)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'indptr_stop' (line 551)
    indptr_stop_392621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 11), 'indptr_stop')
    # Getting the type of 'indptr_start' (line 551)
    indptr_start_392622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 26), 'indptr_start')
    # Applying the binary operator '<=' (line 551)
    result_le_392623 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 11), '<=', indptr_stop_392621, indptr_start_392622)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'A_diagonal_index_row_i' (line 551)
    A_diagonal_index_row_i_392624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 52), 'A_diagonal_index_row_i')
    # Getting the type of 'A' (line 551)
    A_392625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 42), 'A')
    # Obtaining the member 'indices' of a type (line 551)
    indices_392626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 42), A_392625, 'indices')
    # Obtaining the member '__getitem__' of a type (line 551)
    getitem___392627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 42), indices_392626, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 551)
    subscript_call_result_392628 = invoke(stypy.reporting.localization.Localization(__file__, 551, 42), getitem___392627, A_diagonal_index_row_i_392624)
    
    # Getting the type of 'i' (line 551)
    i_392629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 78), 'i')
    # Applying the binary operator '<' (line 551)
    result_lt_392630 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 42), '<', subscript_call_result_392628, i_392629)
    
    # Applying the binary operator 'or' (line 551)
    result_or_keyword_392631 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 11), 'or', result_le_392623, result_lt_392630)
    
    # Testing the type of an if condition (line 551)
    if_condition_392632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 551, 8), result_or_keyword_392631)
    # Assigning a type to the variable 'if_condition_392632' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'if_condition_392632', if_condition_392632)
    # SSA begins for if statement (line 551)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 552)
    # Processing the call arguments (line 552)
    
    # Call to format(...): (line 553)
    # Processing the call arguments (line 553)
    # Getting the type of 'i' (line 553)
    i_392636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 61), 'i', False)
    # Processing the call keyword arguments (line 553)
    kwargs_392637 = {}
    str_392634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 16), 'str', 'A is singular: diagonal {} is zero.')
    # Obtaining the member 'format' of a type (line 553)
    format_392635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 16), str_392634, 'format')
    # Calling format(args, kwargs) (line 553)
    format_call_result_392638 = invoke(stypy.reporting.localization.Localization(__file__, 553, 16), format_392635, *[i_392636], **kwargs_392637)
    
    # Processing the call keyword arguments (line 552)
    kwargs_392639 = {}
    # Getting the type of 'LinAlgError' (line 552)
    LinAlgError_392633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 18), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 552)
    LinAlgError_call_result_392640 = invoke(stypy.reporting.localization.Localization(__file__, 552, 18), LinAlgError_392633, *[format_call_result_392638], **kwargs_392639)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 552, 12), LinAlgError_call_result_392640, 'raise parameter', BaseException)
    # SSA join for if statement (line 551)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'A_diagonal_index_row_i' (line 554)
    A_diagonal_index_row_i_392641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 21), 'A_diagonal_index_row_i')
    # Getting the type of 'A' (line 554)
    A_392642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 11), 'A')
    # Obtaining the member 'indices' of a type (line 554)
    indices_392643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 11), A_392642, 'indices')
    # Obtaining the member '__getitem__' of a type (line 554)
    getitem___392644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 11), indices_392643, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 554)
    subscript_call_result_392645 = invoke(stypy.reporting.localization.Localization(__file__, 554, 11), getitem___392644, A_diagonal_index_row_i_392641)
    
    # Getting the type of 'i' (line 554)
    i_392646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 47), 'i')
    # Applying the binary operator '>' (line 554)
    result_gt_392647 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 11), '>', subscript_call_result_392645, i_392646)
    
    # Testing the type of an if condition (line 554)
    if_condition_392648 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 8), result_gt_392647)
    # Assigning a type to the variable 'if_condition_392648' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'if_condition_392648', if_condition_392648)
    # SSA begins for if statement (line 554)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 555)
    # Processing the call arguments (line 555)
    
    # Call to format(...): (line 556)
    # Processing the call arguments (line 556)
    # Getting the type of 'i' (line 557)
    i_392652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 26), 'i', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'A_diagonal_index_row_i' (line 557)
    A_diagonal_index_row_i_392653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 39), 'A_diagonal_index_row_i', False)
    # Getting the type of 'A' (line 557)
    A_392654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 29), 'A', False)
    # Obtaining the member 'indices' of a type (line 557)
    indices_392655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 29), A_392654, 'indices')
    # Obtaining the member '__getitem__' of a type (line 557)
    getitem___392656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 29), indices_392655, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 557)
    subscript_call_result_392657 = invoke(stypy.reporting.localization.Localization(__file__, 557, 29), getitem___392656, A_diagonal_index_row_i_392653)
    
    # Processing the call keyword arguments (line 556)
    kwargs_392658 = {}
    str_392650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 16), 'str', 'A is not triangular: A[{}, {}] is nonzero.')
    # Obtaining the member 'format' of a type (line 556)
    format_392651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 16), str_392650, 'format')
    # Calling format(args, kwargs) (line 556)
    format_call_result_392659 = invoke(stypy.reporting.localization.Localization(__file__, 556, 16), format_392651, *[i_392652, subscript_call_result_392657], **kwargs_392658)
    
    # Processing the call keyword arguments (line 555)
    kwargs_392660 = {}
    # Getting the type of 'LinAlgError' (line 555)
    LinAlgError_392649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 18), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 555)
    LinAlgError_call_result_392661 = invoke(stypy.reporting.localization.Localization(__file__, 555, 18), LinAlgError_392649, *[format_call_result_392659], **kwargs_392660)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 555, 12), LinAlgError_call_result_392661, 'raise parameter', BaseException)
    # SSA join for if statement (line 554)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 560):
    
    # Assigning a Subscript to a Name (line 560):
    
    # Obtaining the type of the subscript
    # Getting the type of 'A_off_diagonal_indices_row_i' (line 560)
    A_off_diagonal_indices_row_i_392662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 46), 'A_off_diagonal_indices_row_i')
    # Getting the type of 'A' (line 560)
    A_392663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 36), 'A')
    # Obtaining the member 'indices' of a type (line 560)
    indices_392664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 36), A_392663, 'indices')
    # Obtaining the member '__getitem__' of a type (line 560)
    getitem___392665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 36), indices_392664, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 560)
    subscript_call_result_392666 = invoke(stypy.reporting.localization.Localization(__file__, 560, 36), getitem___392665, A_off_diagonal_indices_row_i_392662)
    
    # Assigning a type to the variable 'A_column_indices_in_row_i' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'A_column_indices_in_row_i', subscript_call_result_392666)
    
    # Assigning a Subscript to a Name (line 561):
    
    # Assigning a Subscript to a Name (line 561):
    
    # Obtaining the type of the subscript
    # Getting the type of 'A_off_diagonal_indices_row_i' (line 561)
    A_off_diagonal_indices_row_i_392667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 35), 'A_off_diagonal_indices_row_i')
    # Getting the type of 'A' (line 561)
    A_392668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 28), 'A')
    # Obtaining the member 'data' of a type (line 561)
    data_392669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 28), A_392668, 'data')
    # Obtaining the member '__getitem__' of a type (line 561)
    getitem___392670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 28), data_392669, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 561)
    subscript_call_result_392671 = invoke(stypy.reporting.localization.Localization(__file__, 561, 28), getitem___392670, A_off_diagonal_indices_row_i_392667)
    
    # Assigning a type to the variable 'A_values_in_row_i' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'A_values_in_row_i', subscript_call_result_392671)
    
    # Getting the type of 'x' (line 562)
    x_392672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'x')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 562)
    i_392673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 10), 'i')
    # Getting the type of 'x' (line 562)
    x_392674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'x')
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___392675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 8), x_392674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_392676 = invoke(stypy.reporting.localization.Localization(__file__, 562, 8), getitem___392675, i_392673)
    
    
    # Call to dot(...): (line 562)
    # Processing the call arguments (line 562)
    
    # Obtaining the type of the subscript
    # Getting the type of 'A_column_indices_in_row_i' (line 562)
    A_column_indices_in_row_i_392679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 25), 'A_column_indices_in_row_i', False)
    # Getting the type of 'x' (line 562)
    x_392680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 23), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___392681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 23), x_392680, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_392682 = invoke(stypy.reporting.localization.Localization(__file__, 562, 23), getitem___392681, A_column_indices_in_row_i_392679)
    
    # Obtaining the member 'T' of a type (line 562)
    T_392683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 23), subscript_call_result_392682, 'T')
    # Getting the type of 'A_values_in_row_i' (line 562)
    A_values_in_row_i_392684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 55), 'A_values_in_row_i', False)
    # Processing the call keyword arguments (line 562)
    kwargs_392685 = {}
    # Getting the type of 'np' (line 562)
    np_392677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 16), 'np', False)
    # Obtaining the member 'dot' of a type (line 562)
    dot_392678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 16), np_392677, 'dot')
    # Calling dot(args, kwargs) (line 562)
    dot_call_result_392686 = invoke(stypy.reporting.localization.Localization(__file__, 562, 16), dot_392678, *[T_392683, A_values_in_row_i_392684], **kwargs_392685)
    
    # Applying the binary operator '-=' (line 562)
    result_isub_392687 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 8), '-=', subscript_call_result_392676, dot_call_result_392686)
    # Getting the type of 'x' (line 562)
    x_392688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'x')
    # Getting the type of 'i' (line 562)
    i_392689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 10), 'i')
    # Storing an element on a container (line 562)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 8), x_392688, (i_392689, result_isub_392687))
    
    
    # Getting the type of 'x' (line 565)
    x_392690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'x')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 565)
    i_392691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 10), 'i')
    # Getting the type of 'x' (line 565)
    x_392692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'x')
    # Obtaining the member '__getitem__' of a type (line 565)
    getitem___392693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 8), x_392692, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 565)
    subscript_call_result_392694 = invoke(stypy.reporting.localization.Localization(__file__, 565, 8), getitem___392693, i_392691)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'A_diagonal_index_row_i' (line 565)
    A_diagonal_index_row_i_392695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 23), 'A_diagonal_index_row_i')
    # Getting the type of 'A' (line 565)
    A_392696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 16), 'A')
    # Obtaining the member 'data' of a type (line 565)
    data_392697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 16), A_392696, 'data')
    # Obtaining the member '__getitem__' of a type (line 565)
    getitem___392698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 16), data_392697, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 565)
    subscript_call_result_392699 = invoke(stypy.reporting.localization.Localization(__file__, 565, 16), getitem___392698, A_diagonal_index_row_i_392695)
    
    # Applying the binary operator 'div=' (line 565)
    result_div_392700 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 8), 'div=', subscript_call_result_392694, subscript_call_result_392699)
    # Getting the type of 'x' (line 565)
    x_392701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'x')
    # Getting the type of 'i' (line 565)
    i_392702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 10), 'i')
    # Storing an element on a container (line 565)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 8), x_392701, (i_392702, result_div_392700))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 567)
    x_392703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'stypy_return_type', x_392703)
    
    # ################# End of 'spsolve_triangular(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spsolve_triangular' in the type store
    # Getting the type of 'stypy_return_type' (line 443)
    stypy_return_type_392704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_392704)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spsolve_triangular'
    return stypy_return_type_392704

# Assigning a type to the variable 'spsolve_triangular' (line 443)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 0), 'spsolve_triangular', spsolve_triangular)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
