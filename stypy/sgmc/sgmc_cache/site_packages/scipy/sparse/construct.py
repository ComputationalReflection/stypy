
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Functions to construct sparse matrices
2: '''
3: from __future__ import division, print_function, absolute_import
4: 
5: __docformat__ = "restructuredtext en"
6: 
7: __all__ = ['spdiags', 'eye', 'identity', 'kron', 'kronsum',
8:            'hstack', 'vstack', 'bmat', 'rand', 'random', 'diags', 'block_diag']
9: 
10: 
11: import numpy as np
12: 
13: from scipy._lib.six import xrange
14: 
15: from .sputils import upcast, get_index_dtype, isscalarlike
16: 
17: from .csr import csr_matrix
18: from .csc import csc_matrix
19: from .bsr import bsr_matrix
20: from .coo import coo_matrix
21: from .dia import dia_matrix
22: 
23: from .base import issparse
24: 
25: 
26: def spdiags(data, diags, m, n, format=None):
27:     '''
28:     Return a sparse matrix from diagonals.
29: 
30:     Parameters
31:     ----------
32:     data : array_like
33:         matrix diagonals stored row-wise
34:     diags : diagonals to set
35:         - k = 0  the main diagonal
36:         - k > 0  the k-th upper diagonal
37:         - k < 0  the k-th lower diagonal
38:     m, n : int
39:         shape of the result
40:     format : str, optional
41:         Format of the result. By default (format=None) an appropriate sparse
42:         matrix format is returned.  This choice is subject to change.
43: 
44:     See Also
45:     --------
46:     diags : more convenient form of this function
47:     dia_matrix : the sparse DIAgonal format.
48: 
49:     Examples
50:     --------
51:     >>> from scipy.sparse import spdiags
52:     >>> data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
53:     >>> diags = np.array([0, -1, 2])
54:     >>> spdiags(data, diags, 4, 4).toarray()
55:     array([[1, 0, 3, 0],
56:            [1, 2, 0, 4],
57:            [0, 2, 3, 0],
58:            [0, 0, 3, 4]])
59: 
60:     '''
61:     return dia_matrix((data, diags), shape=(m,n)).asformat(format)
62: 
63: 
64: def diags(diagonals, offsets=0, shape=None, format=None, dtype=None):
65:     '''
66:     Construct a sparse matrix from diagonals.
67: 
68:     Parameters
69:     ----------
70:     diagonals : sequence of array_like
71:         Sequence of arrays containing the matrix diagonals,
72:         corresponding to `offsets`.
73:     offsets : sequence of int or an int, optional
74:         Diagonals to set:
75:           - k = 0  the main diagonal (default)
76:           - k > 0  the k-th upper diagonal
77:           - k < 0  the k-th lower diagonal
78:     shape : tuple of int, optional
79:         Shape of the result. If omitted, a square matrix large enough
80:         to contain the diagonals is returned.
81:     format : {"dia", "csr", "csc", "lil", ...}, optional
82:         Matrix format of the result.  By default (format=None) an
83:         appropriate sparse matrix format is returned.  This choice is
84:         subject to change.
85:     dtype : dtype, optional
86:         Data type of the matrix.
87: 
88:     See Also
89:     --------
90:     spdiags : construct matrix from diagonals
91: 
92:     Notes
93:     -----
94:     This function differs from `spdiags` in the way it handles
95:     off-diagonals.
96: 
97:     The result from `diags` is the sparse equivalent of::
98: 
99:         np.diag(diagonals[0], offsets[0])
100:         + ...
101:         + np.diag(diagonals[k], offsets[k])
102: 
103:     Repeated diagonal offsets are disallowed.
104: 
105:     .. versionadded:: 0.11
106: 
107:     Examples
108:     --------
109:     >>> from scipy.sparse import diags
110:     >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
111:     >>> diags(diagonals, [0, -1, 2]).toarray()
112:     array([[1, 0, 1, 0],
113:            [1, 2, 0, 2],
114:            [0, 2, 3, 0],
115:            [0, 0, 3, 4]])
116: 
117:     Broadcasting of scalars is supported (but shape needs to be
118:     specified):
119: 
120:     >>> diags([1, -2, 1], [-1, 0, 1], shape=(4, 4)).toarray()
121:     array([[-2.,  1.,  0.,  0.],
122:            [ 1., -2.,  1.,  0.],
123:            [ 0.,  1., -2.,  1.],
124:            [ 0.,  0.,  1., -2.]])
125: 
126: 
127:     If only one diagonal is wanted (as in `numpy.diag`), the following
128:     works as well:
129: 
130:     >>> diags([1, 2, 3], 1).toarray()
131:     array([[ 0.,  1.,  0.,  0.],
132:            [ 0.,  0.,  2.,  0.],
133:            [ 0.,  0.,  0.,  3.],
134:            [ 0.,  0.,  0.,  0.]])
135:     '''
136:     # if offsets is not a sequence, assume that there's only one diagonal
137:     if isscalarlike(offsets):
138:         # now check that there's actually only one diagonal
139:         if len(diagonals) == 0 or isscalarlike(diagonals[0]):
140:             diagonals = [np.atleast_1d(diagonals)]
141:         else:
142:             raise ValueError("Different number of diagonals and offsets.")
143:     else:
144:         diagonals = list(map(np.atleast_1d, diagonals))
145: 
146:     offsets = np.atleast_1d(offsets)
147: 
148:     # Basic check
149:     if len(diagonals) != len(offsets):
150:         raise ValueError("Different number of diagonals and offsets.")
151: 
152:     # Determine shape, if omitted
153:     if shape is None:
154:         m = len(diagonals[0]) + abs(int(offsets[0]))
155:         shape = (m, m)
156: 
157:     # Determine data type, if omitted
158:     if dtype is None:
159:         dtype = np.common_type(*diagonals)
160: 
161:     # Construct data array
162:     m, n = shape
163: 
164:     M = max([min(m + offset, n - offset) + max(0, offset)
165:              for offset in offsets])
166:     M = max(0, M)
167:     data_arr = np.zeros((len(offsets), M), dtype=dtype)
168: 
169:     K = min(m, n)
170: 
171:     for j, diagonal in enumerate(diagonals):
172:         offset = offsets[j]
173:         k = max(0, offset)
174:         length = min(m + offset, n - offset, K)
175:         if length < 0:
176:             raise ValueError("Offset %d (index %d) out of bounds" % (offset, j))
177:         try:
178:             data_arr[j, k:k+length] = diagonal[...,:length]
179:         except ValueError:
180:             if len(diagonal) != length and len(diagonal) != 1:
181:                 raise ValueError(
182:                     "Diagonal length (index %d: %d at offset %d) does not "
183:                     "agree with matrix size (%d, %d)." % (
184:                     j, len(diagonal), offset, m, n))
185:             raise
186: 
187:     return dia_matrix((data_arr, offsets), shape=(m, n)).asformat(format)
188: 
189: 
190: def identity(n, dtype='d', format=None):
191:     '''Identity matrix in sparse format
192: 
193:     Returns an identity matrix with shape (n,n) using a given
194:     sparse format and dtype.
195: 
196:     Parameters
197:     ----------
198:     n : int
199:         Shape of the identity matrix.
200:     dtype : dtype, optional
201:         Data type of the matrix
202:     format : str, optional
203:         Sparse format of the result, e.g. format="csr", etc.
204: 
205:     Examples
206:     --------
207:     >>> from scipy.sparse import identity
208:     >>> identity(3).toarray()
209:     array([[ 1.,  0.,  0.],
210:            [ 0.,  1.,  0.],
211:            [ 0.,  0.,  1.]])
212:     >>> identity(3, dtype='int8', format='dia')
213:     <3x3 sparse matrix of type '<class 'numpy.int8'>'
214:             with 3 stored elements (1 diagonals) in DIAgonal format>
215: 
216:     '''
217:     return eye(n, n, dtype=dtype, format=format)
218: 
219: 
220: def eye(m, n=None, k=0, dtype=float, format=None):
221:     '''Sparse matrix with ones on diagonal
222: 
223:     Returns a sparse (m x n) matrix where the k-th diagonal
224:     is all ones and everything else is zeros.
225: 
226:     Parameters
227:     ----------
228:     m : int
229:         Number of rows in the matrix.
230:     n : int, optional
231:         Number of columns. Default: `m`.
232:     k : int, optional
233:         Diagonal to place ones on. Default: 0 (main diagonal).
234:     dtype : dtype, optional
235:         Data type of the matrix.
236:     format : str, optional
237:         Sparse format of the result, e.g. format="csr", etc.
238: 
239:     Examples
240:     --------
241:     >>> from scipy import sparse
242:     >>> sparse.eye(3).toarray()
243:     array([[ 1.,  0.,  0.],
244:            [ 0.,  1.,  0.],
245:            [ 0.,  0.,  1.]])
246:     >>> sparse.eye(3, dtype=np.int8)
247:     <3x3 sparse matrix of type '<class 'numpy.int8'>'
248:         with 3 stored elements (1 diagonals) in DIAgonal format>
249: 
250:     '''
251:     if n is None:
252:         n = m
253:     m,n = int(m),int(n)
254: 
255:     if m == n and k == 0:
256:         # fast branch for special formats
257:         if format in ['csr', 'csc']:
258:             idx_dtype = get_index_dtype(maxval=n)
259:             indptr = np.arange(n+1, dtype=idx_dtype)
260:             indices = np.arange(n, dtype=idx_dtype)
261:             data = np.ones(n, dtype=dtype)
262:             cls = {'csr': csr_matrix, 'csc': csc_matrix}[format]
263:             return cls((data,indices,indptr),(n,n))
264:         elif format == 'coo':
265:             idx_dtype = get_index_dtype(maxval=n)
266:             row = np.arange(n, dtype=idx_dtype)
267:             col = np.arange(n, dtype=idx_dtype)
268:             data = np.ones(n, dtype=dtype)
269:             return coo_matrix((data,(row,col)),(n,n))
270: 
271:     diags = np.ones((1, max(0, min(m + k, n))), dtype=dtype)
272:     return spdiags(diags, k, m, n).asformat(format)
273: 
274: 
275: def kron(A, B, format=None):
276:     '''kronecker product of sparse matrices A and B
277: 
278:     Parameters
279:     ----------
280:     A : sparse or dense matrix
281:         first matrix of the product
282:     B : sparse or dense matrix
283:         second matrix of the product
284:     format : str, optional
285:         format of the result (e.g. "csr")
286: 
287:     Returns
288:     -------
289:     kronecker product in a sparse matrix format
290: 
291: 
292:     Examples
293:     --------
294:     >>> from scipy import sparse
295:     >>> A = sparse.csr_matrix(np.array([[0, 2], [5, 0]]))
296:     >>> B = sparse.csr_matrix(np.array([[1, 2], [3, 4]]))
297:     >>> sparse.kron(A, B).toarray()
298:     array([[ 0,  0,  2,  4],
299:            [ 0,  0,  6,  8],
300:            [ 5, 10,  0,  0],
301:            [15, 20,  0,  0]])
302: 
303:     >>> sparse.kron(A, [[1, 2], [3, 4]]).toarray()
304:     array([[ 0,  0,  2,  4],
305:            [ 0,  0,  6,  8],
306:            [ 5, 10,  0,  0],
307:            [15, 20,  0,  0]])
308: 
309:     '''
310:     B = coo_matrix(B)
311: 
312:     if (format is None or format == "bsr") and 2*B.nnz >= B.shape[0] * B.shape[1]:
313:         # B is fairly dense, use BSR
314:         A = csr_matrix(A,copy=True)
315: 
316:         output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])
317: 
318:         if A.nnz == 0 or B.nnz == 0:
319:             # kronecker product is the zero matrix
320:             return coo_matrix(output_shape)
321: 
322:         B = B.toarray()
323:         data = A.data.repeat(B.size).reshape(-1,B.shape[0],B.shape[1])
324:         data = data * B
325: 
326:         return bsr_matrix((data,A.indices,A.indptr), shape=output_shape)
327:     else:
328:         # use COO
329:         A = coo_matrix(A)
330:         output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])
331: 
332:         if A.nnz == 0 or B.nnz == 0:
333:             # kronecker product is the zero matrix
334:             return coo_matrix(output_shape)
335: 
336:         # expand entries of a into blocks
337:         row = A.row.repeat(B.nnz)
338:         col = A.col.repeat(B.nnz)
339:         data = A.data.repeat(B.nnz)
340: 
341:         row *= B.shape[0]
342:         col *= B.shape[1]
343: 
344:         # increment block indices
345:         row,col = row.reshape(-1,B.nnz),col.reshape(-1,B.nnz)
346:         row += B.row
347:         col += B.col
348:         row,col = row.reshape(-1),col.reshape(-1)
349: 
350:         # compute block entries
351:         data = data.reshape(-1,B.nnz) * B.data
352:         data = data.reshape(-1)
353: 
354:         return coo_matrix((data,(row,col)), shape=output_shape).asformat(format)
355: 
356: 
357: def kronsum(A, B, format=None):
358:     '''kronecker sum of sparse matrices A and B
359: 
360:     Kronecker sum of two sparse matrices is a sum of two Kronecker
361:     products kron(I_n,A) + kron(B,I_m) where A has shape (m,m)
362:     and B has shape (n,n) and I_m and I_n are identity matrices
363:     of shape (m,m) and (n,n) respectively.
364: 
365:     Parameters
366:     ----------
367:     A
368:         square matrix
369:     B
370:         square matrix
371:     format : str
372:         format of the result (e.g. "csr")
373: 
374:     Returns
375:     -------
376:     kronecker sum in a sparse matrix format
377: 
378:     Examples
379:     --------
380: 
381: 
382:     '''
383:     A = coo_matrix(A)
384:     B = coo_matrix(B)
385: 
386:     if A.shape[0] != A.shape[1]:
387:         raise ValueError('A is not square')
388: 
389:     if B.shape[0] != B.shape[1]:
390:         raise ValueError('B is not square')
391: 
392:     dtype = upcast(A.dtype, B.dtype)
393: 
394:     L = kron(eye(B.shape[0],dtype=dtype), A, format=format)
395:     R = kron(B, eye(A.shape[0],dtype=dtype), format=format)
396: 
397:     return (L+R).asformat(format)  # since L + R is not always same format
398: 
399: 
400: def _compressed_sparse_stack(blocks, axis):
401:     '''
402:     Stacking fast path for CSR/CSC matrices
403:     (i) vstack for CSR, (ii) hstack for CSC.
404:     '''
405:     other_axis = 1 if axis == 0 else 0
406:     data = np.concatenate([b.data for b in blocks])
407:     constant_dim = blocks[0].shape[other_axis]
408:     idx_dtype = get_index_dtype(arrays=[b.indptr for b in blocks],
409:                                 maxval=max(data.size, constant_dim))
410:     indices = np.empty(data.size, dtype=idx_dtype)
411:     indptr = np.empty(sum(b.shape[axis] for b in blocks) + 1, dtype=idx_dtype)
412:     last_indptr = idx_dtype(0)
413:     sum_dim = 0
414:     sum_indices = 0
415:     for b in blocks:
416:         if b.shape[other_axis] != constant_dim:
417:             raise ValueError('incompatible dimensions for axis %d' % other_axis)
418:         indices[sum_indices:sum_indices+b.indices.size] = b.indices
419:         sum_indices += b.indices.size
420:         idxs = slice(sum_dim, sum_dim + b.shape[axis])
421:         indptr[idxs] = b.indptr[:-1]
422:         indptr[idxs] += last_indptr
423:         sum_dim += b.shape[axis]
424:         last_indptr += b.indptr[-1]
425:     indptr[-1] = last_indptr
426:     if axis == 0:
427:         return csr_matrix((data, indices, indptr),
428:                           shape=(sum_dim, constant_dim))
429:     else:
430:         return csc_matrix((data, indices, indptr),
431:                           shape=(constant_dim, sum_dim))
432: 
433: 
434: def hstack(blocks, format=None, dtype=None):
435:     '''
436:     Stack sparse matrices horizontally (column wise)
437: 
438:     Parameters
439:     ----------
440:     blocks
441:         sequence of sparse matrices with compatible shapes
442:     format : str
443:         sparse format of the result (e.g. "csr")
444:         by default an appropriate sparse matrix format is returned.
445:         This choice is subject to change.
446:     dtype : dtype, optional
447:         The data-type of the output matrix.  If not given, the dtype is
448:         determined from that of `blocks`.
449: 
450:     See Also
451:     --------
452:     vstack : stack sparse matrices vertically (row wise)
453: 
454:     Examples
455:     --------
456:     >>> from scipy.sparse import coo_matrix, hstack
457:     >>> A = coo_matrix([[1, 2], [3, 4]])
458:     >>> B = coo_matrix([[5], [6]])
459:     >>> hstack([A,B]).toarray()
460:     array([[1, 2, 5],
461:            [3, 4, 6]])
462: 
463:     '''
464:     return bmat([blocks], format=format, dtype=dtype)
465: 
466: 
467: def vstack(blocks, format=None, dtype=None):
468:     '''
469:     Stack sparse matrices vertically (row wise)
470: 
471:     Parameters
472:     ----------
473:     blocks
474:         sequence of sparse matrices with compatible shapes
475:     format : str, optional
476:         sparse format of the result (e.g. "csr")
477:         by default an appropriate sparse matrix format is returned.
478:         This choice is subject to change.
479:     dtype : dtype, optional
480:         The data-type of the output matrix.  If not given, the dtype is
481:         determined from that of `blocks`.
482: 
483:     See Also
484:     --------
485:     hstack : stack sparse matrices horizontally (column wise)
486: 
487:     Examples
488:     --------
489:     >>> from scipy.sparse import coo_matrix, vstack
490:     >>> A = coo_matrix([[1, 2], [3, 4]])
491:     >>> B = coo_matrix([[5, 6]])
492:     >>> vstack([A, B]).toarray()
493:     array([[1, 2],
494:            [3, 4],
495:            [5, 6]])
496: 
497:     '''
498:     return bmat([[b] for b in blocks], format=format, dtype=dtype)
499: 
500: 
501: def bmat(blocks, format=None, dtype=None):
502:     '''
503:     Build a sparse matrix from sparse sub-blocks
504: 
505:     Parameters
506:     ----------
507:     blocks : array_like
508:         Grid of sparse matrices with compatible shapes.
509:         An entry of None implies an all-zero matrix.
510:     format : {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional
511:         The sparse format of the result (e.g. "csr").  By default an
512:         appropriate sparse matrix format is returned.
513:         This choice is subject to change.
514:     dtype : dtype, optional
515:         The data-type of the output matrix.  If not given, the dtype is
516:         determined from that of `blocks`.
517: 
518:     Returns
519:     -------
520:     bmat : sparse matrix
521: 
522:     See Also
523:     --------
524:     block_diag, diags
525: 
526:     Examples
527:     --------
528:     >>> from scipy.sparse import coo_matrix, bmat
529:     >>> A = coo_matrix([[1, 2], [3, 4]])
530:     >>> B = coo_matrix([[5], [6]])
531:     >>> C = coo_matrix([[7]])
532:     >>> bmat([[A, B], [None, C]]).toarray()
533:     array([[1, 2, 5],
534:            [3, 4, 6],
535:            [0, 0, 7]])
536: 
537:     >>> bmat([[A, None], [None, C]]).toarray()
538:     array([[1, 2, 0],
539:            [3, 4, 0],
540:            [0, 0, 7]])
541: 
542:     '''
543: 
544:     blocks = np.asarray(blocks, dtype='object')
545: 
546:     if blocks.ndim != 2:
547:         raise ValueError('blocks must be 2-D')
548: 
549:     M,N = blocks.shape
550: 
551:     # check for fast path cases
552:     if (N == 1 and format in (None, 'csr') and all(isinstance(b, csr_matrix)
553:                                                    for b in blocks.flat)):
554:         A = _compressed_sparse_stack(blocks[:,0], 0)
555:         if dtype is not None:
556:             A = A.astype(dtype)
557:         return A
558:     elif (M == 1 and format in (None, 'csc')
559:           and all(isinstance(b, csc_matrix) for b in blocks.flat)):
560:         A = _compressed_sparse_stack(blocks[0,:], 1)
561:         if dtype is not None:
562:             A = A.astype(dtype)
563:         return A
564: 
565:     block_mask = np.zeros(blocks.shape, dtype=bool)
566:     brow_lengths = np.zeros(M, dtype=np.int64)
567:     bcol_lengths = np.zeros(N, dtype=np.int64)
568: 
569:     # convert everything to COO format
570:     for i in range(M):
571:         for j in range(N):
572:             if blocks[i,j] is not None:
573:                 A = coo_matrix(blocks[i,j])
574:                 blocks[i,j] = A
575:                 block_mask[i,j] = True
576: 
577:                 if brow_lengths[i] == 0:
578:                     brow_lengths[i] = A.shape[0]
579:                 elif brow_lengths[i] != A.shape[0]:
580:                     msg = ('blocks[{i},:] has incompatible row dimensions. '
581:                            'Got blocks[{i},{j}].shape[0] == {got}, '
582:                            'expected {exp}.'.format(i=i, j=j,
583:                                                     exp=brow_lengths[i],
584:                                                     got=A.shape[0]))
585:                     raise ValueError(msg)
586: 
587:                 if bcol_lengths[j] == 0:
588:                     bcol_lengths[j] = A.shape[1]
589:                 elif bcol_lengths[j] != A.shape[1]:
590:                     msg = ('blocks[:,{j}] has incompatible row dimensions. '
591:                            'Got blocks[{i},{j}].shape[1] == {got}, '
592:                            'expected {exp}.'.format(i=i, j=j,
593:                                                     exp=bcol_lengths[j],
594:                                                     got=A.shape[1]))
595:                     raise ValueError(msg)
596: 
597:     nnz = sum(block.nnz for block in blocks[block_mask])
598:     if dtype is None:
599:         all_dtypes = [blk.dtype for blk in blocks[block_mask]]
600:         dtype = upcast(*all_dtypes) if all_dtypes else None
601: 
602:     row_offsets = np.append(0, np.cumsum(brow_lengths))
603:     col_offsets = np.append(0, np.cumsum(bcol_lengths))
604: 
605:     shape = (row_offsets[-1], col_offsets[-1])
606: 
607:     data = np.empty(nnz, dtype=dtype)
608:     idx_dtype = get_index_dtype(maxval=max(shape))
609:     row = np.empty(nnz, dtype=idx_dtype)
610:     col = np.empty(nnz, dtype=idx_dtype)
611: 
612:     nnz = 0
613:     ii, jj = np.nonzero(block_mask)
614:     for i, j in zip(ii, jj):
615:         B = blocks[i, j]
616:         idx = slice(nnz, nnz + B.nnz)
617:         data[idx] = B.data
618:         row[idx] = B.row + row_offsets[i]
619:         col[idx] = B.col + col_offsets[j]
620:         nnz += B.nnz
621: 
622:     return coo_matrix((data, (row, col)), shape=shape).asformat(format)
623: 
624: 
625: def block_diag(mats, format=None, dtype=None):
626:     '''
627:     Build a block diagonal sparse matrix from provided matrices.
628: 
629:     Parameters
630:     ----------
631:     mats : sequence of matrices
632:         Input matrices.
633:     format : str, optional
634:         The sparse format of the result (e.g. "csr").  If not given, the matrix
635:         is returned in "coo" format.
636:     dtype : dtype specifier, optional
637:         The data-type of the output matrix.  If not given, the dtype is
638:         determined from that of `blocks`.
639: 
640:     Returns
641:     -------
642:     res : sparse matrix
643: 
644:     Notes
645:     -----
646: 
647:     .. versionadded:: 0.11.0
648: 
649:     See Also
650:     --------
651:     bmat, diags
652: 
653:     Examples
654:     --------
655:     >>> from scipy.sparse import coo_matrix, block_diag
656:     >>> A = coo_matrix([[1, 2], [3, 4]])
657:     >>> B = coo_matrix([[5], [6]])
658:     >>> C = coo_matrix([[7]])
659:     >>> block_diag((A, B, C)).toarray()
660:     array([[1, 2, 0, 0],
661:            [3, 4, 0, 0],
662:            [0, 0, 5, 0],
663:            [0, 0, 6, 0],
664:            [0, 0, 0, 7]])
665: 
666:     '''
667:     nmat = len(mats)
668:     rows = []
669:     for ia, a in enumerate(mats):
670:         row = [None]*nmat
671:         if issparse(a):
672:             row[ia] = a
673:         else:
674:             row[ia] = coo_matrix(a)
675:         rows.append(row)
676:     return bmat(rows, format=format, dtype=dtype)
677: 
678: 
679: def random(m, n, density=0.01, format='coo', dtype=None,
680:            random_state=None, data_rvs=None):
681:     '''Generate a sparse matrix of the given shape and density with randomly
682:     distributed values.
683: 
684:     Parameters
685:     ----------
686:     m, n : int
687:         shape of the matrix
688:     density : real, optional
689:         density of the generated matrix: density equal to one means a full
690:         matrix, density of 0 means a matrix with no non-zero items.
691:     format : str, optional
692:         sparse matrix format.
693:     dtype : dtype, optional
694:         type of the returned matrix values.
695:     random_state : {numpy.random.RandomState, int}, optional
696:         Random number generator or random seed. If not given, the singleton
697:         numpy.random will be used.  This random state will be used
698:         for sampling the sparsity structure, but not necessarily for sampling
699:         the values of the structurally nonzero entries of the matrix.
700:     data_rvs : callable, optional
701:         Samples a requested number of random values.
702:         This function should take a single argument specifying the length
703:         of the ndarray that it will return.  The structurally nonzero entries
704:         of the sparse random matrix will be taken from the array sampled
705:         by this function.  By default, uniform [0, 1) random values will be
706:         sampled using the same random state as is used for sampling
707:         the sparsity structure.
708: 
709:     Returns
710:     -------
711:     res : sparse matrix
712: 
713:     Examples
714:     --------
715:     >>> from scipy.sparse import random
716:     >>> from scipy import stats
717:     >>> class CustomRandomState(object):
718:     ...     def randint(self, k):
719:     ...         i = np.random.randint(k)
720:     ...         return i - i % 2
721:     >>> rs = CustomRandomState()
722:     >>> rvs = stats.poisson(25, loc=10).rvs
723:     >>> S = random(3, 4, density=0.25, random_state=rs, data_rvs=rvs)
724:     >>> S.A
725:     array([[ 36.,   0.,  33.,   0.],   # random
726:            [  0.,   0.,   0.,   0.],
727:            [  0.,   0.,  36.,   0.]])
728: 
729:     Notes
730:     -----
731:     Only float types are supported for now.
732:     '''
733:     if density < 0 or density > 1:
734:         raise ValueError("density expected to be 0 <= density <= 1")
735:     dtype = np.dtype(dtype)
736:     if dtype.char not in 'fdg':
737:         raise NotImplementedError("type %s not supported" % dtype)
738: 
739:     mn = m * n
740: 
741:     tp = np.intc
742:     if mn > np.iinfo(tp).max:
743:         tp = np.int64
744: 
745:     if mn > np.iinfo(tp).max:
746:         msg = '''\
747: Trying to generate a random sparse matrix such as the product of dimensions is
748: greater than %d - this is not supported on this machine
749: '''
750:         raise ValueError(msg % np.iinfo(tp).max)
751: 
752:     # Number of non zero values
753:     k = int(density * m * n)
754: 
755:     if random_state is None:
756:         random_state = np.random
757:     elif isinstance(random_state, (int, np.integer)):
758:         random_state = np.random.RandomState(random_state)
759:     if data_rvs is None:
760:         data_rvs = random_state.rand
761: 
762:     # Use the algorithm from python's random.sample for k < mn/3.
763:     if mn < 3*k:
764:         ind = random_state.choice(mn, size=k, replace=False)
765:     else:
766:         ind = np.empty(k, dtype=tp)
767:         selected = set()
768:         for i in xrange(k):
769:             j = random_state.randint(mn)
770:             while j in selected:
771:                 j = random_state.randint(mn)
772:             selected.add(j)
773:             ind[i] = j
774: 
775:     j = np.floor(ind * 1. / m).astype(tp)
776:     i = (ind - j * m).astype(tp)
777:     vals = data_rvs(k).astype(dtype)
778:     return coo_matrix((vals, (i, j)), shape=(m, n)).asformat(format)
779: 
780: 
781: def rand(m, n, density=0.01, format="coo", dtype=None, random_state=None):
782:     '''Generate a sparse matrix of the given shape and density with uniformly
783:     distributed values.
784: 
785:     Parameters
786:     ----------
787:     m, n : int
788:         shape of the matrix
789:     density : real, optional
790:         density of the generated matrix: density equal to one means a full
791:         matrix, density of 0 means a matrix with no non-zero items.
792:     format : str, optional
793:         sparse matrix format.
794:     dtype : dtype, optional
795:         type of the returned matrix values.
796:     random_state : {numpy.random.RandomState, int}, optional
797:         Random number generator or random seed. If not given, the singleton
798:         numpy.random will be used.
799: 
800:     Returns
801:     -------
802:     res : sparse matrix
803: 
804:     Notes
805:     -----
806:     Only float types are supported for now.
807: 
808:     See Also
809:     --------
810:     scipy.sparse.random : Similar function that allows a user-specified random
811:         data source.
812: 
813:     Examples
814:     --------
815:     >>> from scipy.sparse import rand
816:     >>> matrix = rand(3, 4, density=0.25, format="csr", random_state=42)
817:     >>> matrix
818:     <3x4 sparse matrix of type '<class 'numpy.float64'>'
819:        with 3 stored elements in Compressed Sparse Row format>
820:     >>> matrix.todense()
821:     matrix([[ 0.        ,  0.59685016,  0.779691  ,  0.        ],
822:             [ 0.        ,  0.        ,  0.        ,  0.44583275],
823:             [ 0.        ,  0.        ,  0.        ,  0.        ]])
824:     '''
825:     return random(m, n, density, format, dtype, random_state)
826: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_366309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'Functions to construct sparse matrices\n')

# Assigning a Str to a Name (line 5):

# Assigning a Str to a Name (line 5):
str_366310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__docformat__', str_366310)

# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['spdiags', 'eye', 'identity', 'kron', 'kronsum', 'hstack', 'vstack', 'bmat', 'rand', 'random', 'diags', 'block_diag']
module_type_store.set_exportable_members(['spdiags', 'eye', 'identity', 'kron', 'kronsum', 'hstack', 'vstack', 'bmat', 'rand', 'random', 'diags', 'block_diag'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_366311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_366312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'spdiags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_366311, str_366312)
# Adding element type (line 7)
str_366313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 22), 'str', 'eye')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_366311, str_366313)
# Adding element type (line 7)
str_366314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 29), 'str', 'identity')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_366311, str_366314)
# Adding element type (line 7)
str_366315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 41), 'str', 'kron')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_366311, str_366315)
# Adding element type (line 7)
str_366316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 49), 'str', 'kronsum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_366311, str_366316)
# Adding element type (line 7)
str_366317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'hstack')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_366311, str_366317)
# Adding element type (line 7)
str_366318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 21), 'str', 'vstack')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_366311, str_366318)
# Adding element type (line 7)
str_366319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 31), 'str', 'bmat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_366311, str_366319)
# Adding element type (line 7)
str_366320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 39), 'str', 'rand')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_366311, str_366320)
# Adding element type (line 7)
str_366321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 47), 'str', 'random')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_366311, str_366321)
# Adding element type (line 7)
str_366322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 57), 'str', 'diags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_366311, str_366322)
# Adding element type (line 7)
str_366323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 66), 'str', 'block_diag')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_366311, str_366323)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_366311)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_366324 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_366324) is not StypyTypeError):

    if (import_366324 != 'pyd_module'):
        __import__(import_366324)
        sys_modules_366325 = sys.modules[import_366324]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_366325.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_366324)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy._lib.six import xrange' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_366326 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six')

if (type(import_366326) is not StypyTypeError):

    if (import_366326 != 'pyd_module'):
        __import__(import_366326)
        sys_modules_366327 = sys.modules[import_366326]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', sys_modules_366327.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_366327, sys_modules_366327.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', import_366326)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.sparse.sputils import upcast, get_index_dtype, isscalarlike' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_366328 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.sputils')

if (type(import_366328) is not StypyTypeError):

    if (import_366328 != 'pyd_module'):
        __import__(import_366328)
        sys_modules_366329 = sys.modules[import_366328]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.sputils', sys_modules_366329.module_type_store, module_type_store, ['upcast', 'get_index_dtype', 'isscalarlike'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_366329, sys_modules_366329.module_type_store, module_type_store)
    else:
        from scipy.sparse.sputils import upcast, get_index_dtype, isscalarlike

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.sputils', None, module_type_store, ['upcast', 'get_index_dtype', 'isscalarlike'], [upcast, get_index_dtype, isscalarlike])

else:
    # Assigning a type to the variable 'scipy.sparse.sputils' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.sputils', import_366328)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.sparse.csr import csr_matrix' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_366330 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.csr')

if (type(import_366330) is not StypyTypeError):

    if (import_366330 != 'pyd_module'):
        __import__(import_366330)
        sys_modules_366331 = sys.modules[import_366330]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.csr', sys_modules_366331.module_type_store, module_type_store, ['csr_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_366331, sys_modules_366331.module_type_store, module_type_store)
    else:
        from scipy.sparse.csr import csr_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.csr', None, module_type_store, ['csr_matrix'], [csr_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse.csr' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.csr', import_366330)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.sparse.csc import csc_matrix' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_366332 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse.csc')

if (type(import_366332) is not StypyTypeError):

    if (import_366332 != 'pyd_module'):
        __import__(import_366332)
        sys_modules_366333 = sys.modules[import_366332]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse.csc', sys_modules_366333.module_type_store, module_type_store, ['csc_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_366333, sys_modules_366333.module_type_store, module_type_store)
    else:
        from scipy.sparse.csc import csc_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse.csc', None, module_type_store, ['csc_matrix'], [csc_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse.csc' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse.csc', import_366332)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy.sparse.bsr import bsr_matrix' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_366334 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.sparse.bsr')

if (type(import_366334) is not StypyTypeError):

    if (import_366334 != 'pyd_module'):
        __import__(import_366334)
        sys_modules_366335 = sys.modules[import_366334]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.sparse.bsr', sys_modules_366335.module_type_store, module_type_store, ['bsr_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_366335, sys_modules_366335.module_type_store, module_type_store)
    else:
        from scipy.sparse.bsr import bsr_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.sparse.bsr', None, module_type_store, ['bsr_matrix'], [bsr_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse.bsr' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.sparse.bsr', import_366334)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy.sparse.coo import coo_matrix' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_366336 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.coo')

if (type(import_366336) is not StypyTypeError):

    if (import_366336 != 'pyd_module'):
        __import__(import_366336)
        sys_modules_366337 = sys.modules[import_366336]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.coo', sys_modules_366337.module_type_store, module_type_store, ['coo_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_366337, sys_modules_366337.module_type_store, module_type_store)
    else:
        from scipy.sparse.coo import coo_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.coo', None, module_type_store, ['coo_matrix'], [coo_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse.coo' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.coo', import_366336)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from scipy.sparse.dia import dia_matrix' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_366338 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.sparse.dia')

if (type(import_366338) is not StypyTypeError):

    if (import_366338 != 'pyd_module'):
        __import__(import_366338)
        sys_modules_366339 = sys.modules[import_366338]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.sparse.dia', sys_modules_366339.module_type_store, module_type_store, ['dia_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_366339, sys_modules_366339.module_type_store, module_type_store)
    else:
        from scipy.sparse.dia import dia_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.sparse.dia', None, module_type_store, ['dia_matrix'], [dia_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse.dia' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.sparse.dia', import_366338)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from scipy.sparse.base import issparse' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_366340 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy.sparse.base')

if (type(import_366340) is not StypyTypeError):

    if (import_366340 != 'pyd_module'):
        __import__(import_366340)
        sys_modules_366341 = sys.modules[import_366340]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy.sparse.base', sys_modules_366341.module_type_store, module_type_store, ['issparse'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_366341, sys_modules_366341.module_type_store, module_type_store)
    else:
        from scipy.sparse.base import issparse

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy.sparse.base', None, module_type_store, ['issparse'], [issparse])

else:
    # Assigning a type to the variable 'scipy.sparse.base' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy.sparse.base', import_366340)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')


@norecursion
def spdiags(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 26)
    None_366342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 38), 'None')
    defaults = [None_366342]
    # Create a new context for function 'spdiags'
    module_type_store = module_type_store.open_function_context('spdiags', 26, 0, False)
    
    # Passed parameters checking function
    spdiags.stypy_localization = localization
    spdiags.stypy_type_of_self = None
    spdiags.stypy_type_store = module_type_store
    spdiags.stypy_function_name = 'spdiags'
    spdiags.stypy_param_names_list = ['data', 'diags', 'm', 'n', 'format']
    spdiags.stypy_varargs_param_name = None
    spdiags.stypy_kwargs_param_name = None
    spdiags.stypy_call_defaults = defaults
    spdiags.stypy_call_varargs = varargs
    spdiags.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spdiags', ['data', 'diags', 'm', 'n', 'format'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spdiags', localization, ['data', 'diags', 'm', 'n', 'format'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spdiags(...)' code ##################

    str_366343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, (-1)), 'str', '\n    Return a sparse matrix from diagonals.\n\n    Parameters\n    ----------\n    data : array_like\n        matrix diagonals stored row-wise\n    diags : diagonals to set\n        - k = 0  the main diagonal\n        - k > 0  the k-th upper diagonal\n        - k < 0  the k-th lower diagonal\n    m, n : int\n        shape of the result\n    format : str, optional\n        Format of the result. By default (format=None) an appropriate sparse\n        matrix format is returned.  This choice is subject to change.\n\n    See Also\n    --------\n    diags : more convenient form of this function\n    dia_matrix : the sparse DIAgonal format.\n\n    Examples\n    --------\n    >>> from scipy.sparse import spdiags\n    >>> data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])\n    >>> diags = np.array([0, -1, 2])\n    >>> spdiags(data, diags, 4, 4).toarray()\n    array([[1, 0, 3, 0],\n           [1, 2, 0, 4],\n           [0, 2, 3, 0],\n           [0, 0, 3, 4]])\n\n    ')
    
    # Call to asformat(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'format' (line 61)
    format_366355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 59), 'format', False)
    # Processing the call keyword arguments (line 61)
    kwargs_366356 = {}
    
    # Call to dia_matrix(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Obtaining an instance of the builtin type 'tuple' (line 61)
    tuple_366345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 61)
    # Adding element type (line 61)
    # Getting the type of 'data' (line 61)
    data_366346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 23), tuple_366345, data_366346)
    # Adding element type (line 61)
    # Getting the type of 'diags' (line 61)
    diags_366347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'diags', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 23), tuple_366345, diags_366347)
    
    # Processing the call keyword arguments (line 61)
    
    # Obtaining an instance of the builtin type 'tuple' (line 61)
    tuple_366348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 61)
    # Adding element type (line 61)
    # Getting the type of 'm' (line 61)
    m_366349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 44), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 44), tuple_366348, m_366349)
    # Adding element type (line 61)
    # Getting the type of 'n' (line 61)
    n_366350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 46), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 44), tuple_366348, n_366350)
    
    keyword_366351 = tuple_366348
    kwargs_366352 = {'shape': keyword_366351}
    # Getting the type of 'dia_matrix' (line 61)
    dia_matrix_366344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'dia_matrix', False)
    # Calling dia_matrix(args, kwargs) (line 61)
    dia_matrix_call_result_366353 = invoke(stypy.reporting.localization.Localization(__file__, 61, 11), dia_matrix_366344, *[tuple_366345], **kwargs_366352)
    
    # Obtaining the member 'asformat' of a type (line 61)
    asformat_366354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 11), dia_matrix_call_result_366353, 'asformat')
    # Calling asformat(args, kwargs) (line 61)
    asformat_call_result_366357 = invoke(stypy.reporting.localization.Localization(__file__, 61, 11), asformat_366354, *[format_366355], **kwargs_366356)
    
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', asformat_call_result_366357)
    
    # ################# End of 'spdiags(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spdiags' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_366358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_366358)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spdiags'
    return stypy_return_type_366358

# Assigning a type to the variable 'spdiags' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'spdiags', spdiags)

@norecursion
def diags(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_366359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 29), 'int')
    # Getting the type of 'None' (line 64)
    None_366360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 38), 'None')
    # Getting the type of 'None' (line 64)
    None_366361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 51), 'None')
    # Getting the type of 'None' (line 64)
    None_366362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 63), 'None')
    defaults = [int_366359, None_366360, None_366361, None_366362]
    # Create a new context for function 'diags'
    module_type_store = module_type_store.open_function_context('diags', 64, 0, False)
    
    # Passed parameters checking function
    diags.stypy_localization = localization
    diags.stypy_type_of_self = None
    diags.stypy_type_store = module_type_store
    diags.stypy_function_name = 'diags'
    diags.stypy_param_names_list = ['diagonals', 'offsets', 'shape', 'format', 'dtype']
    diags.stypy_varargs_param_name = None
    diags.stypy_kwargs_param_name = None
    diags.stypy_call_defaults = defaults
    diags.stypy_call_varargs = varargs
    diags.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'diags', ['diagonals', 'offsets', 'shape', 'format', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'diags', localization, ['diagonals', 'offsets', 'shape', 'format', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'diags(...)' code ##################

    str_366363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, (-1)), 'str', '\n    Construct a sparse matrix from diagonals.\n\n    Parameters\n    ----------\n    diagonals : sequence of array_like\n        Sequence of arrays containing the matrix diagonals,\n        corresponding to `offsets`.\n    offsets : sequence of int or an int, optional\n        Diagonals to set:\n          - k = 0  the main diagonal (default)\n          - k > 0  the k-th upper diagonal\n          - k < 0  the k-th lower diagonal\n    shape : tuple of int, optional\n        Shape of the result. If omitted, a square matrix large enough\n        to contain the diagonals is returned.\n    format : {"dia", "csr", "csc", "lil", ...}, optional\n        Matrix format of the result.  By default (format=None) an\n        appropriate sparse matrix format is returned.  This choice is\n        subject to change.\n    dtype : dtype, optional\n        Data type of the matrix.\n\n    See Also\n    --------\n    spdiags : construct matrix from diagonals\n\n    Notes\n    -----\n    This function differs from `spdiags` in the way it handles\n    off-diagonals.\n\n    The result from `diags` is the sparse equivalent of::\n\n        np.diag(diagonals[0], offsets[0])\n        + ...\n        + np.diag(diagonals[k], offsets[k])\n\n    Repeated diagonal offsets are disallowed.\n\n    .. versionadded:: 0.11\n\n    Examples\n    --------\n    >>> from scipy.sparse import diags\n    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]\n    >>> diags(diagonals, [0, -1, 2]).toarray()\n    array([[1, 0, 1, 0],\n           [1, 2, 0, 2],\n           [0, 2, 3, 0],\n           [0, 0, 3, 4]])\n\n    Broadcasting of scalars is supported (but shape needs to be\n    specified):\n\n    >>> diags([1, -2, 1], [-1, 0, 1], shape=(4, 4)).toarray()\n    array([[-2.,  1.,  0.,  0.],\n           [ 1., -2.,  1.,  0.],\n           [ 0.,  1., -2.,  1.],\n           [ 0.,  0.,  1., -2.]])\n\n\n    If only one diagonal is wanted (as in `numpy.diag`), the following\n    works as well:\n\n    >>> diags([1, 2, 3], 1).toarray()\n    array([[ 0.,  1.,  0.,  0.],\n           [ 0.,  0.,  2.,  0.],\n           [ 0.,  0.,  0.,  3.],\n           [ 0.,  0.,  0.,  0.]])\n    ')
    
    
    # Call to isscalarlike(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'offsets' (line 137)
    offsets_366365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 20), 'offsets', False)
    # Processing the call keyword arguments (line 137)
    kwargs_366366 = {}
    # Getting the type of 'isscalarlike' (line 137)
    isscalarlike_366364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 7), 'isscalarlike', False)
    # Calling isscalarlike(args, kwargs) (line 137)
    isscalarlike_call_result_366367 = invoke(stypy.reporting.localization.Localization(__file__, 137, 7), isscalarlike_366364, *[offsets_366365], **kwargs_366366)
    
    # Testing the type of an if condition (line 137)
    if_condition_366368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 4), isscalarlike_call_result_366367)
    # Assigning a type to the variable 'if_condition_366368' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'if_condition_366368', if_condition_366368)
    # SSA begins for if statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'diagonals' (line 139)
    diagonals_366370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'diagonals', False)
    # Processing the call keyword arguments (line 139)
    kwargs_366371 = {}
    # Getting the type of 'len' (line 139)
    len_366369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'len', False)
    # Calling len(args, kwargs) (line 139)
    len_call_result_366372 = invoke(stypy.reporting.localization.Localization(__file__, 139, 11), len_366369, *[diagonals_366370], **kwargs_366371)
    
    int_366373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 29), 'int')
    # Applying the binary operator '==' (line 139)
    result_eq_366374 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 11), '==', len_call_result_366372, int_366373)
    
    
    # Call to isscalarlike(...): (line 139)
    # Processing the call arguments (line 139)
    
    # Obtaining the type of the subscript
    int_366376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 57), 'int')
    # Getting the type of 'diagonals' (line 139)
    diagonals_366377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 47), 'diagonals', False)
    # Obtaining the member '__getitem__' of a type (line 139)
    getitem___366378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 47), diagonals_366377, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 139)
    subscript_call_result_366379 = invoke(stypy.reporting.localization.Localization(__file__, 139, 47), getitem___366378, int_366376)
    
    # Processing the call keyword arguments (line 139)
    kwargs_366380 = {}
    # Getting the type of 'isscalarlike' (line 139)
    isscalarlike_366375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 34), 'isscalarlike', False)
    # Calling isscalarlike(args, kwargs) (line 139)
    isscalarlike_call_result_366381 = invoke(stypy.reporting.localization.Localization(__file__, 139, 34), isscalarlike_366375, *[subscript_call_result_366379], **kwargs_366380)
    
    # Applying the binary operator 'or' (line 139)
    result_or_keyword_366382 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 11), 'or', result_eq_366374, isscalarlike_call_result_366381)
    
    # Testing the type of an if condition (line 139)
    if_condition_366383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 8), result_or_keyword_366382)
    # Assigning a type to the variable 'if_condition_366383' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'if_condition_366383', if_condition_366383)
    # SSA begins for if statement (line 139)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 140):
    
    # Assigning a List to a Name (line 140):
    
    # Obtaining an instance of the builtin type 'list' (line 140)
    list_366384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 140)
    # Adding element type (line 140)
    
    # Call to atleast_1d(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'diagonals' (line 140)
    diagonals_366387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 39), 'diagonals', False)
    # Processing the call keyword arguments (line 140)
    kwargs_366388 = {}
    # Getting the type of 'np' (line 140)
    np_366385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 25), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 140)
    atleast_1d_366386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 25), np_366385, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 140)
    atleast_1d_call_result_366389 = invoke(stypy.reporting.localization.Localization(__file__, 140, 25), atleast_1d_366386, *[diagonals_366387], **kwargs_366388)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 24), list_366384, atleast_1d_call_result_366389)
    
    # Assigning a type to the variable 'diagonals' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'diagonals', list_366384)
    # SSA branch for the else part of an if statement (line 139)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 142)
    # Processing the call arguments (line 142)
    str_366391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 29), 'str', 'Different number of diagonals and offsets.')
    # Processing the call keyword arguments (line 142)
    kwargs_366392 = {}
    # Getting the type of 'ValueError' (line 142)
    ValueError_366390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 142)
    ValueError_call_result_366393 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), ValueError_366390, *[str_366391], **kwargs_366392)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 142, 12), ValueError_call_result_366393, 'raise parameter', BaseException)
    # SSA join for if statement (line 139)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 137)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 144):
    
    # Assigning a Call to a Name (line 144):
    
    # Call to list(...): (line 144)
    # Processing the call arguments (line 144)
    
    # Call to map(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'np' (line 144)
    np_366396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 29), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 144)
    atleast_1d_366397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 29), np_366396, 'atleast_1d')
    # Getting the type of 'diagonals' (line 144)
    diagonals_366398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 44), 'diagonals', False)
    # Processing the call keyword arguments (line 144)
    kwargs_366399 = {}
    # Getting the type of 'map' (line 144)
    map_366395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'map', False)
    # Calling map(args, kwargs) (line 144)
    map_call_result_366400 = invoke(stypy.reporting.localization.Localization(__file__, 144, 25), map_366395, *[atleast_1d_366397, diagonals_366398], **kwargs_366399)
    
    # Processing the call keyword arguments (line 144)
    kwargs_366401 = {}
    # Getting the type of 'list' (line 144)
    list_366394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'list', False)
    # Calling list(args, kwargs) (line 144)
    list_call_result_366402 = invoke(stypy.reporting.localization.Localization(__file__, 144, 20), list_366394, *[map_call_result_366400], **kwargs_366401)
    
    # Assigning a type to the variable 'diagonals' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'diagonals', list_call_result_366402)
    # SSA join for if statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 146):
    
    # Assigning a Call to a Name (line 146):
    
    # Call to atleast_1d(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'offsets' (line 146)
    offsets_366405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 28), 'offsets', False)
    # Processing the call keyword arguments (line 146)
    kwargs_366406 = {}
    # Getting the type of 'np' (line 146)
    np_366403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 14), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 146)
    atleast_1d_366404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 14), np_366403, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 146)
    atleast_1d_call_result_366407 = invoke(stypy.reporting.localization.Localization(__file__, 146, 14), atleast_1d_366404, *[offsets_366405], **kwargs_366406)
    
    # Assigning a type to the variable 'offsets' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'offsets', atleast_1d_call_result_366407)
    
    
    
    # Call to len(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'diagonals' (line 149)
    diagonals_366409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'diagonals', False)
    # Processing the call keyword arguments (line 149)
    kwargs_366410 = {}
    # Getting the type of 'len' (line 149)
    len_366408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 7), 'len', False)
    # Calling len(args, kwargs) (line 149)
    len_call_result_366411 = invoke(stypy.reporting.localization.Localization(__file__, 149, 7), len_366408, *[diagonals_366409], **kwargs_366410)
    
    
    # Call to len(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'offsets' (line 149)
    offsets_366413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 29), 'offsets', False)
    # Processing the call keyword arguments (line 149)
    kwargs_366414 = {}
    # Getting the type of 'len' (line 149)
    len_366412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'len', False)
    # Calling len(args, kwargs) (line 149)
    len_call_result_366415 = invoke(stypy.reporting.localization.Localization(__file__, 149, 25), len_366412, *[offsets_366413], **kwargs_366414)
    
    # Applying the binary operator '!=' (line 149)
    result_ne_366416 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 7), '!=', len_call_result_366411, len_call_result_366415)
    
    # Testing the type of an if condition (line 149)
    if_condition_366417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 4), result_ne_366416)
    # Assigning a type to the variable 'if_condition_366417' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'if_condition_366417', if_condition_366417)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 150)
    # Processing the call arguments (line 150)
    str_366419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 25), 'str', 'Different number of diagonals and offsets.')
    # Processing the call keyword arguments (line 150)
    kwargs_366420 = {}
    # Getting the type of 'ValueError' (line 150)
    ValueError_366418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 150)
    ValueError_call_result_366421 = invoke(stypy.reporting.localization.Localization(__file__, 150, 14), ValueError_366418, *[str_366419], **kwargs_366420)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 150, 8), ValueError_call_result_366421, 'raise parameter', BaseException)
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 153)
    # Getting the type of 'shape' (line 153)
    shape_366422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 7), 'shape')
    # Getting the type of 'None' (line 153)
    None_366423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'None')
    
    (may_be_366424, more_types_in_union_366425) = may_be_none(shape_366422, None_366423)

    if may_be_366424:

        if more_types_in_union_366425:
            # Runtime conditional SSA (line 153)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 154):
        
        # Assigning a BinOp to a Name (line 154):
        
        # Call to len(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Obtaining the type of the subscript
        int_366427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 26), 'int')
        # Getting the type of 'diagonals' (line 154)
        diagonals_366428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'diagonals', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___366429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 16), diagonals_366428, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_366430 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), getitem___366429, int_366427)
        
        # Processing the call keyword arguments (line 154)
        kwargs_366431 = {}
        # Getting the type of 'len' (line 154)
        len_366426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'len', False)
        # Calling len(args, kwargs) (line 154)
        len_call_result_366432 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), len_366426, *[subscript_call_result_366430], **kwargs_366431)
        
        
        # Call to abs(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Call to int(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Obtaining the type of the subscript
        int_366435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 48), 'int')
        # Getting the type of 'offsets' (line 154)
        offsets_366436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 40), 'offsets', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___366437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 40), offsets_366436, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_366438 = invoke(stypy.reporting.localization.Localization(__file__, 154, 40), getitem___366437, int_366435)
        
        # Processing the call keyword arguments (line 154)
        kwargs_366439 = {}
        # Getting the type of 'int' (line 154)
        int_366434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 36), 'int', False)
        # Calling int(args, kwargs) (line 154)
        int_call_result_366440 = invoke(stypy.reporting.localization.Localization(__file__, 154, 36), int_366434, *[subscript_call_result_366438], **kwargs_366439)
        
        # Processing the call keyword arguments (line 154)
        kwargs_366441 = {}
        # Getting the type of 'abs' (line 154)
        abs_366433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 32), 'abs', False)
        # Calling abs(args, kwargs) (line 154)
        abs_call_result_366442 = invoke(stypy.reporting.localization.Localization(__file__, 154, 32), abs_366433, *[int_call_result_366440], **kwargs_366441)
        
        # Applying the binary operator '+' (line 154)
        result_add_366443 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 12), '+', len_call_result_366432, abs_call_result_366442)
        
        # Assigning a type to the variable 'm' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'm', result_add_366443)
        
        # Assigning a Tuple to a Name (line 155):
        
        # Assigning a Tuple to a Name (line 155):
        
        # Obtaining an instance of the builtin type 'tuple' (line 155)
        tuple_366444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 155)
        # Adding element type (line 155)
        # Getting the type of 'm' (line 155)
        m_366445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 17), 'm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 17), tuple_366444, m_366445)
        # Adding element type (line 155)
        # Getting the type of 'm' (line 155)
        m_366446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 17), tuple_366444, m_366446)
        
        # Assigning a type to the variable 'shape' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'shape', tuple_366444)

        if more_types_in_union_366425:
            # SSA join for if statement (line 153)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 158)
    # Getting the type of 'dtype' (line 158)
    dtype_366447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 7), 'dtype')
    # Getting the type of 'None' (line 158)
    None_366448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'None')
    
    (may_be_366449, more_types_in_union_366450) = may_be_none(dtype_366447, None_366448)

    if may_be_366449:

        if more_types_in_union_366450:
            # Runtime conditional SSA (line 158)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 159):
        
        # Assigning a Call to a Name (line 159):
        
        # Call to common_type(...): (line 159)
        # Getting the type of 'diagonals' (line 159)
        diagonals_366453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 'diagonals', False)
        # Processing the call keyword arguments (line 159)
        kwargs_366454 = {}
        # Getting the type of 'np' (line 159)
        np_366451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'np', False)
        # Obtaining the member 'common_type' of a type (line 159)
        common_type_366452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 16), np_366451, 'common_type')
        # Calling common_type(args, kwargs) (line 159)
        common_type_call_result_366455 = invoke(stypy.reporting.localization.Localization(__file__, 159, 16), common_type_366452, *[diagonals_366453], **kwargs_366454)
        
        # Assigning a type to the variable 'dtype' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'dtype', common_type_call_result_366455)

        if more_types_in_union_366450:
            # SSA join for if statement (line 158)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Tuple (line 162):
    
    # Assigning a Subscript to a Name (line 162):
    
    # Obtaining the type of the subscript
    int_366456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 4), 'int')
    # Getting the type of 'shape' (line 162)
    shape_366457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'shape')
    # Obtaining the member '__getitem__' of a type (line 162)
    getitem___366458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 4), shape_366457, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 162)
    subscript_call_result_366459 = invoke(stypy.reporting.localization.Localization(__file__, 162, 4), getitem___366458, int_366456)
    
    # Assigning a type to the variable 'tuple_var_assignment_366297' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'tuple_var_assignment_366297', subscript_call_result_366459)
    
    # Assigning a Subscript to a Name (line 162):
    
    # Obtaining the type of the subscript
    int_366460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 4), 'int')
    # Getting the type of 'shape' (line 162)
    shape_366461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'shape')
    # Obtaining the member '__getitem__' of a type (line 162)
    getitem___366462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 4), shape_366461, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 162)
    subscript_call_result_366463 = invoke(stypy.reporting.localization.Localization(__file__, 162, 4), getitem___366462, int_366460)
    
    # Assigning a type to the variable 'tuple_var_assignment_366298' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'tuple_var_assignment_366298', subscript_call_result_366463)
    
    # Assigning a Name to a Name (line 162):
    # Getting the type of 'tuple_var_assignment_366297' (line 162)
    tuple_var_assignment_366297_366464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'tuple_var_assignment_366297')
    # Assigning a type to the variable 'm' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'm', tuple_var_assignment_366297_366464)
    
    # Assigning a Name to a Name (line 162):
    # Getting the type of 'tuple_var_assignment_366298' (line 162)
    tuple_var_assignment_366298_366465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'tuple_var_assignment_366298')
    # Assigning a type to the variable 'n' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 7), 'n', tuple_var_assignment_366298_366465)
    
    # Assigning a Call to a Name (line 164):
    
    # Assigning a Call to a Name (line 164):
    
    # Call to max(...): (line 164)
    # Processing the call arguments (line 164)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'offsets' (line 165)
    offsets_366482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 27), 'offsets', False)
    comprehension_366483 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 13), offsets_366482)
    # Assigning a type to the variable 'offset' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'offset', comprehension_366483)
    
    # Call to min(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'm' (line 164)
    m_366468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 17), 'm', False)
    # Getting the type of 'offset' (line 164)
    offset_366469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'offset', False)
    # Applying the binary operator '+' (line 164)
    result_add_366470 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 17), '+', m_366468, offset_366469)
    
    # Getting the type of 'n' (line 164)
    n_366471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 29), 'n', False)
    # Getting the type of 'offset' (line 164)
    offset_366472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 33), 'offset', False)
    # Applying the binary operator '-' (line 164)
    result_sub_366473 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 29), '-', n_366471, offset_366472)
    
    # Processing the call keyword arguments (line 164)
    kwargs_366474 = {}
    # Getting the type of 'min' (line 164)
    min_366467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'min', False)
    # Calling min(args, kwargs) (line 164)
    min_call_result_366475 = invoke(stypy.reporting.localization.Localization(__file__, 164, 13), min_366467, *[result_add_366470, result_sub_366473], **kwargs_366474)
    
    
    # Call to max(...): (line 164)
    # Processing the call arguments (line 164)
    int_366477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 47), 'int')
    # Getting the type of 'offset' (line 164)
    offset_366478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 50), 'offset', False)
    # Processing the call keyword arguments (line 164)
    kwargs_366479 = {}
    # Getting the type of 'max' (line 164)
    max_366476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 43), 'max', False)
    # Calling max(args, kwargs) (line 164)
    max_call_result_366480 = invoke(stypy.reporting.localization.Localization(__file__, 164, 43), max_366476, *[int_366477, offset_366478], **kwargs_366479)
    
    # Applying the binary operator '+' (line 164)
    result_add_366481 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 13), '+', min_call_result_366475, max_call_result_366480)
    
    list_366484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 13), list_366484, result_add_366481)
    # Processing the call keyword arguments (line 164)
    kwargs_366485 = {}
    # Getting the type of 'max' (line 164)
    max_366466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'max', False)
    # Calling max(args, kwargs) (line 164)
    max_call_result_366486 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), max_366466, *[list_366484], **kwargs_366485)
    
    # Assigning a type to the variable 'M' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'M', max_call_result_366486)
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to max(...): (line 166)
    # Processing the call arguments (line 166)
    int_366488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 12), 'int')
    # Getting the type of 'M' (line 166)
    M_366489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'M', False)
    # Processing the call keyword arguments (line 166)
    kwargs_366490 = {}
    # Getting the type of 'max' (line 166)
    max_366487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'max', False)
    # Calling max(args, kwargs) (line 166)
    max_call_result_366491 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), max_366487, *[int_366488, M_366489], **kwargs_366490)
    
    # Assigning a type to the variable 'M' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'M', max_call_result_366491)
    
    # Assigning a Call to a Name (line 167):
    
    # Assigning a Call to a Name (line 167):
    
    # Call to zeros(...): (line 167)
    # Processing the call arguments (line 167)
    
    # Obtaining an instance of the builtin type 'tuple' (line 167)
    tuple_366494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 167)
    # Adding element type (line 167)
    
    # Call to len(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'offsets' (line 167)
    offsets_366496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 29), 'offsets', False)
    # Processing the call keyword arguments (line 167)
    kwargs_366497 = {}
    # Getting the type of 'len' (line 167)
    len_366495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'len', False)
    # Calling len(args, kwargs) (line 167)
    len_call_result_366498 = invoke(stypy.reporting.localization.Localization(__file__, 167, 25), len_366495, *[offsets_366496], **kwargs_366497)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 25), tuple_366494, len_call_result_366498)
    # Adding element type (line 167)
    # Getting the type of 'M' (line 167)
    M_366499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'M', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 25), tuple_366494, M_366499)
    
    # Processing the call keyword arguments (line 167)
    # Getting the type of 'dtype' (line 167)
    dtype_366500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 49), 'dtype', False)
    keyword_366501 = dtype_366500
    kwargs_366502 = {'dtype': keyword_366501}
    # Getting the type of 'np' (line 167)
    np_366492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), 'np', False)
    # Obtaining the member 'zeros' of a type (line 167)
    zeros_366493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 15), np_366492, 'zeros')
    # Calling zeros(args, kwargs) (line 167)
    zeros_call_result_366503 = invoke(stypy.reporting.localization.Localization(__file__, 167, 15), zeros_366493, *[tuple_366494], **kwargs_366502)
    
    # Assigning a type to the variable 'data_arr' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'data_arr', zeros_call_result_366503)
    
    # Assigning a Call to a Name (line 169):
    
    # Assigning a Call to a Name (line 169):
    
    # Call to min(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'm' (line 169)
    m_366505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'm', False)
    # Getting the type of 'n' (line 169)
    n_366506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'n', False)
    # Processing the call keyword arguments (line 169)
    kwargs_366507 = {}
    # Getting the type of 'min' (line 169)
    min_366504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'min', False)
    # Calling min(args, kwargs) (line 169)
    min_call_result_366508 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), min_366504, *[m_366505, n_366506], **kwargs_366507)
    
    # Assigning a type to the variable 'K' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'K', min_call_result_366508)
    
    
    # Call to enumerate(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'diagonals' (line 171)
    diagonals_366510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 33), 'diagonals', False)
    # Processing the call keyword arguments (line 171)
    kwargs_366511 = {}
    # Getting the type of 'enumerate' (line 171)
    enumerate_366509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 23), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 171)
    enumerate_call_result_366512 = invoke(stypy.reporting.localization.Localization(__file__, 171, 23), enumerate_366509, *[diagonals_366510], **kwargs_366511)
    
    # Testing the type of a for loop iterable (line 171)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 171, 4), enumerate_call_result_366512)
    # Getting the type of the for loop variable (line 171)
    for_loop_var_366513 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 171, 4), enumerate_call_result_366512)
    # Assigning a type to the variable 'j' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 4), for_loop_var_366513))
    # Assigning a type to the variable 'diagonal' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'diagonal', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 4), for_loop_var_366513))
    # SSA begins for a for statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 172):
    
    # Assigning a Subscript to a Name (line 172):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 172)
    j_366514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 'j')
    # Getting the type of 'offsets' (line 172)
    offsets_366515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 17), 'offsets')
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___366516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 17), offsets_366515, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_366517 = invoke(stypy.reporting.localization.Localization(__file__, 172, 17), getitem___366516, j_366514)
    
    # Assigning a type to the variable 'offset' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'offset', subscript_call_result_366517)
    
    # Assigning a Call to a Name (line 173):
    
    # Assigning a Call to a Name (line 173):
    
    # Call to max(...): (line 173)
    # Processing the call arguments (line 173)
    int_366519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 16), 'int')
    # Getting the type of 'offset' (line 173)
    offset_366520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'offset', False)
    # Processing the call keyword arguments (line 173)
    kwargs_366521 = {}
    # Getting the type of 'max' (line 173)
    max_366518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'max', False)
    # Calling max(args, kwargs) (line 173)
    max_call_result_366522 = invoke(stypy.reporting.localization.Localization(__file__, 173, 12), max_366518, *[int_366519, offset_366520], **kwargs_366521)
    
    # Assigning a type to the variable 'k' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'k', max_call_result_366522)
    
    # Assigning a Call to a Name (line 174):
    
    # Assigning a Call to a Name (line 174):
    
    # Call to min(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'm' (line 174)
    m_366524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 21), 'm', False)
    # Getting the type of 'offset' (line 174)
    offset_366525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'offset', False)
    # Applying the binary operator '+' (line 174)
    result_add_366526 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 21), '+', m_366524, offset_366525)
    
    # Getting the type of 'n' (line 174)
    n_366527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 33), 'n', False)
    # Getting the type of 'offset' (line 174)
    offset_366528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 37), 'offset', False)
    # Applying the binary operator '-' (line 174)
    result_sub_366529 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 33), '-', n_366527, offset_366528)
    
    # Getting the type of 'K' (line 174)
    K_366530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 45), 'K', False)
    # Processing the call keyword arguments (line 174)
    kwargs_366531 = {}
    # Getting the type of 'min' (line 174)
    min_366523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 17), 'min', False)
    # Calling min(args, kwargs) (line 174)
    min_call_result_366532 = invoke(stypy.reporting.localization.Localization(__file__, 174, 17), min_366523, *[result_add_366526, result_sub_366529, K_366530], **kwargs_366531)
    
    # Assigning a type to the variable 'length' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'length', min_call_result_366532)
    
    
    # Getting the type of 'length' (line 175)
    length_366533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 11), 'length')
    int_366534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 20), 'int')
    # Applying the binary operator '<' (line 175)
    result_lt_366535 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 11), '<', length_366533, int_366534)
    
    # Testing the type of an if condition (line 175)
    if_condition_366536 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 8), result_lt_366535)
    # Assigning a type to the variable 'if_condition_366536' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'if_condition_366536', if_condition_366536)
    # SSA begins for if statement (line 175)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 176)
    # Processing the call arguments (line 176)
    str_366538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 29), 'str', 'Offset %d (index %d) out of bounds')
    
    # Obtaining an instance of the builtin type 'tuple' (line 176)
    tuple_366539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 69), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 176)
    # Adding element type (line 176)
    # Getting the type of 'offset' (line 176)
    offset_366540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 69), 'offset', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 69), tuple_366539, offset_366540)
    # Adding element type (line 176)
    # Getting the type of 'j' (line 176)
    j_366541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 77), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 69), tuple_366539, j_366541)
    
    # Applying the binary operator '%' (line 176)
    result_mod_366542 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 29), '%', str_366538, tuple_366539)
    
    # Processing the call keyword arguments (line 176)
    kwargs_366543 = {}
    # Getting the type of 'ValueError' (line 176)
    ValueError_366537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 176)
    ValueError_call_result_366544 = invoke(stypy.reporting.localization.Localization(__file__, 176, 18), ValueError_366537, *[result_mod_366542], **kwargs_366543)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 176, 12), ValueError_call_result_366544, 'raise parameter', BaseException)
    # SSA join for if statement (line 175)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Subscript (line 178):
    
    # Assigning a Subscript to a Subscript (line 178):
    
    # Obtaining the type of the subscript
    Ellipsis_366545 = Ellipsis
    # Getting the type of 'length' (line 178)
    length_366546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 52), 'length')
    slice_366547 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 178, 38), None, length_366546, None)
    # Getting the type of 'diagonal' (line 178)
    diagonal_366548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 38), 'diagonal')
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___366549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 38), diagonal_366548, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_366550 = invoke(stypy.reporting.localization.Localization(__file__, 178, 38), getitem___366549, (Ellipsis_366545, slice_366547))
    
    # Getting the type of 'data_arr' (line 178)
    data_arr_366551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'data_arr')
    # Getting the type of 'j' (line 178)
    j_366552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'j')
    # Getting the type of 'k' (line 178)
    k_366553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'k')
    # Getting the type of 'k' (line 178)
    k_366554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 26), 'k')
    # Getting the type of 'length' (line 178)
    length_366555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'length')
    # Applying the binary operator '+' (line 178)
    result_add_366556 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 26), '+', k_366554, length_366555)
    
    slice_366557 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 178, 12), k_366553, result_add_366556, None)
    # Storing an element on a container (line 178)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 12), data_arr_366551, ((j_366552, slice_366557), subscript_call_result_366550))
    # SSA branch for the except part of a try statement (line 177)
    # SSA branch for the except 'ValueError' branch of a try statement (line 177)
    module_type_store.open_ssa_branch('except')
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'diagonal' (line 180)
    diagonal_366559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'diagonal', False)
    # Processing the call keyword arguments (line 180)
    kwargs_366560 = {}
    # Getting the type of 'len' (line 180)
    len_366558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'len', False)
    # Calling len(args, kwargs) (line 180)
    len_call_result_366561 = invoke(stypy.reporting.localization.Localization(__file__, 180, 15), len_366558, *[diagonal_366559], **kwargs_366560)
    
    # Getting the type of 'length' (line 180)
    length_366562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 32), 'length')
    # Applying the binary operator '!=' (line 180)
    result_ne_366563 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 15), '!=', len_call_result_366561, length_366562)
    
    
    
    # Call to len(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'diagonal' (line 180)
    diagonal_366565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 47), 'diagonal', False)
    # Processing the call keyword arguments (line 180)
    kwargs_366566 = {}
    # Getting the type of 'len' (line 180)
    len_366564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 43), 'len', False)
    # Calling len(args, kwargs) (line 180)
    len_call_result_366567 = invoke(stypy.reporting.localization.Localization(__file__, 180, 43), len_366564, *[diagonal_366565], **kwargs_366566)
    
    int_366568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 60), 'int')
    # Applying the binary operator '!=' (line 180)
    result_ne_366569 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 43), '!=', len_call_result_366567, int_366568)
    
    # Applying the binary operator 'and' (line 180)
    result_and_keyword_366570 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 15), 'and', result_ne_366563, result_ne_366569)
    
    # Testing the type of an if condition (line 180)
    if_condition_366571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 12), result_and_keyword_366570)
    # Assigning a type to the variable 'if_condition_366571' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'if_condition_366571', if_condition_366571)
    # SSA begins for if statement (line 180)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 181)
    # Processing the call arguments (line 181)
    str_366573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 20), 'str', 'Diagonal length (index %d: %d at offset %d) does not agree with matrix size (%d, %d).')
    
    # Obtaining an instance of the builtin type 'tuple' (line 184)
    tuple_366574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 184)
    # Adding element type (line 184)
    # Getting the type of 'j' (line 184)
    j_366575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 20), tuple_366574, j_366575)
    # Adding element type (line 184)
    
    # Call to len(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'diagonal' (line 184)
    diagonal_366577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'diagonal', False)
    # Processing the call keyword arguments (line 184)
    kwargs_366578 = {}
    # Getting the type of 'len' (line 184)
    len_366576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 23), 'len', False)
    # Calling len(args, kwargs) (line 184)
    len_call_result_366579 = invoke(stypy.reporting.localization.Localization(__file__, 184, 23), len_366576, *[diagonal_366577], **kwargs_366578)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 20), tuple_366574, len_call_result_366579)
    # Adding element type (line 184)
    # Getting the type of 'offset' (line 184)
    offset_366580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 38), 'offset', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 20), tuple_366574, offset_366580)
    # Adding element type (line 184)
    # Getting the type of 'm' (line 184)
    m_366581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 46), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 20), tuple_366574, m_366581)
    # Adding element type (line 184)
    # Getting the type of 'n' (line 184)
    n_366582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 49), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 20), tuple_366574, n_366582)
    
    # Applying the binary operator '%' (line 182)
    result_mod_366583 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 20), '%', str_366573, tuple_366574)
    
    # Processing the call keyword arguments (line 181)
    kwargs_366584 = {}
    # Getting the type of 'ValueError' (line 181)
    ValueError_366572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 181)
    ValueError_call_result_366585 = invoke(stypy.reporting.localization.Localization(__file__, 181, 22), ValueError_366572, *[result_mod_366583], **kwargs_366584)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 181, 16), ValueError_call_result_366585, 'raise parameter', BaseException)
    # SSA join for if statement (line 180)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 177)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to asformat(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'format' (line 187)
    format_366597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 66), 'format', False)
    # Processing the call keyword arguments (line 187)
    kwargs_366598 = {}
    
    # Call to dia_matrix(...): (line 187)
    # Processing the call arguments (line 187)
    
    # Obtaining an instance of the builtin type 'tuple' (line 187)
    tuple_366587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 187)
    # Adding element type (line 187)
    # Getting the type of 'data_arr' (line 187)
    data_arr_366588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'data_arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 23), tuple_366587, data_arr_366588)
    # Adding element type (line 187)
    # Getting the type of 'offsets' (line 187)
    offsets_366589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 33), 'offsets', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 23), tuple_366587, offsets_366589)
    
    # Processing the call keyword arguments (line 187)
    
    # Obtaining an instance of the builtin type 'tuple' (line 187)
    tuple_366590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 187)
    # Adding element type (line 187)
    # Getting the type of 'm' (line 187)
    m_366591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 50), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 50), tuple_366590, m_366591)
    # Adding element type (line 187)
    # Getting the type of 'n' (line 187)
    n_366592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 53), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 50), tuple_366590, n_366592)
    
    keyword_366593 = tuple_366590
    kwargs_366594 = {'shape': keyword_366593}
    # Getting the type of 'dia_matrix' (line 187)
    dia_matrix_366586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'dia_matrix', False)
    # Calling dia_matrix(args, kwargs) (line 187)
    dia_matrix_call_result_366595 = invoke(stypy.reporting.localization.Localization(__file__, 187, 11), dia_matrix_366586, *[tuple_366587], **kwargs_366594)
    
    # Obtaining the member 'asformat' of a type (line 187)
    asformat_366596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 11), dia_matrix_call_result_366595, 'asformat')
    # Calling asformat(args, kwargs) (line 187)
    asformat_call_result_366599 = invoke(stypy.reporting.localization.Localization(__file__, 187, 11), asformat_366596, *[format_366597], **kwargs_366598)
    
    # Assigning a type to the variable 'stypy_return_type' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type', asformat_call_result_366599)
    
    # ################# End of 'diags(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'diags' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_366600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_366600)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'diags'
    return stypy_return_type_366600

# Assigning a type to the variable 'diags' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'diags', diags)

@norecursion
def identity(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_366601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 22), 'str', 'd')
    # Getting the type of 'None' (line 190)
    None_366602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 34), 'None')
    defaults = [str_366601, None_366602]
    # Create a new context for function 'identity'
    module_type_store = module_type_store.open_function_context('identity', 190, 0, False)
    
    # Passed parameters checking function
    identity.stypy_localization = localization
    identity.stypy_type_of_self = None
    identity.stypy_type_store = module_type_store
    identity.stypy_function_name = 'identity'
    identity.stypy_param_names_list = ['n', 'dtype', 'format']
    identity.stypy_varargs_param_name = None
    identity.stypy_kwargs_param_name = None
    identity.stypy_call_defaults = defaults
    identity.stypy_call_varargs = varargs
    identity.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'identity', ['n', 'dtype', 'format'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'identity', localization, ['n', 'dtype', 'format'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'identity(...)' code ##################

    str_366603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, (-1)), 'str', 'Identity matrix in sparse format\n\n    Returns an identity matrix with shape (n,n) using a given\n    sparse format and dtype.\n\n    Parameters\n    ----------\n    n : int\n        Shape of the identity matrix.\n    dtype : dtype, optional\n        Data type of the matrix\n    format : str, optional\n        Sparse format of the result, e.g. format="csr", etc.\n\n    Examples\n    --------\n    >>> from scipy.sparse import identity\n    >>> identity(3).toarray()\n    array([[ 1.,  0.,  0.],\n           [ 0.,  1.,  0.],\n           [ 0.,  0.,  1.]])\n    >>> identity(3, dtype=\'int8\', format=\'dia\')\n    <3x3 sparse matrix of type \'<class \'numpy.int8\'>\'\n            with 3 stored elements (1 diagonals) in DIAgonal format>\n\n    ')
    
    # Call to eye(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'n' (line 217)
    n_366605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'n', False)
    # Getting the type of 'n' (line 217)
    n_366606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'n', False)
    # Processing the call keyword arguments (line 217)
    # Getting the type of 'dtype' (line 217)
    dtype_366607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 27), 'dtype', False)
    keyword_366608 = dtype_366607
    # Getting the type of 'format' (line 217)
    format_366609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 41), 'format', False)
    keyword_366610 = format_366609
    kwargs_366611 = {'dtype': keyword_366608, 'format': keyword_366610}
    # Getting the type of 'eye' (line 217)
    eye_366604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'eye', False)
    # Calling eye(args, kwargs) (line 217)
    eye_call_result_366612 = invoke(stypy.reporting.localization.Localization(__file__, 217, 11), eye_366604, *[n_366605, n_366606], **kwargs_366611)
    
    # Assigning a type to the variable 'stypy_return_type' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type', eye_call_result_366612)
    
    # ################# End of 'identity(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'identity' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_366613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_366613)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'identity'
    return stypy_return_type_366613

# Assigning a type to the variable 'identity' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'identity', identity)

@norecursion
def eye(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 220)
    None_366614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 13), 'None')
    int_366615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 21), 'int')
    # Getting the type of 'float' (line 220)
    float_366616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 30), 'float')
    # Getting the type of 'None' (line 220)
    None_366617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 44), 'None')
    defaults = [None_366614, int_366615, float_366616, None_366617]
    # Create a new context for function 'eye'
    module_type_store = module_type_store.open_function_context('eye', 220, 0, False)
    
    # Passed parameters checking function
    eye.stypy_localization = localization
    eye.stypy_type_of_self = None
    eye.stypy_type_store = module_type_store
    eye.stypy_function_name = 'eye'
    eye.stypy_param_names_list = ['m', 'n', 'k', 'dtype', 'format']
    eye.stypy_varargs_param_name = None
    eye.stypy_kwargs_param_name = None
    eye.stypy_call_defaults = defaults
    eye.stypy_call_varargs = varargs
    eye.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eye', ['m', 'n', 'k', 'dtype', 'format'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eye', localization, ['m', 'n', 'k', 'dtype', 'format'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eye(...)' code ##################

    str_366618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, (-1)), 'str', 'Sparse matrix with ones on diagonal\n\n    Returns a sparse (m x n) matrix where the k-th diagonal\n    is all ones and everything else is zeros.\n\n    Parameters\n    ----------\n    m : int\n        Number of rows in the matrix.\n    n : int, optional\n        Number of columns. Default: `m`.\n    k : int, optional\n        Diagonal to place ones on. Default: 0 (main diagonal).\n    dtype : dtype, optional\n        Data type of the matrix.\n    format : str, optional\n        Sparse format of the result, e.g. format="csr", etc.\n\n    Examples\n    --------\n    >>> from scipy import sparse\n    >>> sparse.eye(3).toarray()\n    array([[ 1.,  0.,  0.],\n           [ 0.,  1.,  0.],\n           [ 0.,  0.,  1.]])\n    >>> sparse.eye(3, dtype=np.int8)\n    <3x3 sparse matrix of type \'<class \'numpy.int8\'>\'\n        with 3 stored elements (1 diagonals) in DIAgonal format>\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 251)
    # Getting the type of 'n' (line 251)
    n_366619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 7), 'n')
    # Getting the type of 'None' (line 251)
    None_366620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'None')
    
    (may_be_366621, more_types_in_union_366622) = may_be_none(n_366619, None_366620)

    if may_be_366621:

        if more_types_in_union_366622:
            # Runtime conditional SSA (line 251)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 252):
        
        # Assigning a Name to a Name (line 252):
        # Getting the type of 'm' (line 252)
        m_366623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'm')
        # Assigning a type to the variable 'n' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'n', m_366623)

        if more_types_in_union_366622:
            # SSA join for if statement (line 251)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Tuple to a Tuple (line 253):
    
    # Assigning a Call to a Name (line 253):
    
    # Call to int(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 'm' (line 253)
    m_366625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 14), 'm', False)
    # Processing the call keyword arguments (line 253)
    kwargs_366626 = {}
    # Getting the type of 'int' (line 253)
    int_366624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 10), 'int', False)
    # Calling int(args, kwargs) (line 253)
    int_call_result_366627 = invoke(stypy.reporting.localization.Localization(__file__, 253, 10), int_366624, *[m_366625], **kwargs_366626)
    
    # Assigning a type to the variable 'tuple_assignment_366299' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_assignment_366299', int_call_result_366627)
    
    # Assigning a Call to a Name (line 253):
    
    # Call to int(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 'n' (line 253)
    n_366629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 21), 'n', False)
    # Processing the call keyword arguments (line 253)
    kwargs_366630 = {}
    # Getting the type of 'int' (line 253)
    int_366628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 17), 'int', False)
    # Calling int(args, kwargs) (line 253)
    int_call_result_366631 = invoke(stypy.reporting.localization.Localization(__file__, 253, 17), int_366628, *[n_366629], **kwargs_366630)
    
    # Assigning a type to the variable 'tuple_assignment_366300' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_assignment_366300', int_call_result_366631)
    
    # Assigning a Name to a Name (line 253):
    # Getting the type of 'tuple_assignment_366299' (line 253)
    tuple_assignment_366299_366632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_assignment_366299')
    # Assigning a type to the variable 'm' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'm', tuple_assignment_366299_366632)
    
    # Assigning a Name to a Name (line 253):
    # Getting the type of 'tuple_assignment_366300' (line 253)
    tuple_assignment_366300_366633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_assignment_366300')
    # Assigning a type to the variable 'n' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 6), 'n', tuple_assignment_366300_366633)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'm' (line 255)
    m_366634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 7), 'm')
    # Getting the type of 'n' (line 255)
    n_366635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'n')
    # Applying the binary operator '==' (line 255)
    result_eq_366636 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 7), '==', m_366634, n_366635)
    
    
    # Getting the type of 'k' (line 255)
    k_366637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 18), 'k')
    int_366638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 23), 'int')
    # Applying the binary operator '==' (line 255)
    result_eq_366639 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 18), '==', k_366637, int_366638)
    
    # Applying the binary operator 'and' (line 255)
    result_and_keyword_366640 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 7), 'and', result_eq_366636, result_eq_366639)
    
    # Testing the type of an if condition (line 255)
    if_condition_366641 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 4), result_and_keyword_366640)
    # Assigning a type to the variable 'if_condition_366641' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'if_condition_366641', if_condition_366641)
    # SSA begins for if statement (line 255)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'format' (line 257)
    format_366642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 'format')
    
    # Obtaining an instance of the builtin type 'list' (line 257)
    list_366643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 257)
    # Adding element type (line 257)
    str_366644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 22), 'str', 'csr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 21), list_366643, str_366644)
    # Adding element type (line 257)
    str_366645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 29), 'str', 'csc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 21), list_366643, str_366645)
    
    # Applying the binary operator 'in' (line 257)
    result_contains_366646 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 11), 'in', format_366642, list_366643)
    
    # Testing the type of an if condition (line 257)
    if_condition_366647 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 8), result_contains_366646)
    # Assigning a type to the variable 'if_condition_366647' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'if_condition_366647', if_condition_366647)
    # SSA begins for if statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 258):
    
    # Assigning a Call to a Name (line 258):
    
    # Call to get_index_dtype(...): (line 258)
    # Processing the call keyword arguments (line 258)
    # Getting the type of 'n' (line 258)
    n_366649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 47), 'n', False)
    keyword_366650 = n_366649
    kwargs_366651 = {'maxval': keyword_366650}
    # Getting the type of 'get_index_dtype' (line 258)
    get_index_dtype_366648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 24), 'get_index_dtype', False)
    # Calling get_index_dtype(args, kwargs) (line 258)
    get_index_dtype_call_result_366652 = invoke(stypy.reporting.localization.Localization(__file__, 258, 24), get_index_dtype_366648, *[], **kwargs_366651)
    
    # Assigning a type to the variable 'idx_dtype' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'idx_dtype', get_index_dtype_call_result_366652)
    
    # Assigning a Call to a Name (line 259):
    
    # Assigning a Call to a Name (line 259):
    
    # Call to arange(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'n' (line 259)
    n_366655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 31), 'n', False)
    int_366656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 33), 'int')
    # Applying the binary operator '+' (line 259)
    result_add_366657 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 31), '+', n_366655, int_366656)
    
    # Processing the call keyword arguments (line 259)
    # Getting the type of 'idx_dtype' (line 259)
    idx_dtype_366658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 42), 'idx_dtype', False)
    keyword_366659 = idx_dtype_366658
    kwargs_366660 = {'dtype': keyword_366659}
    # Getting the type of 'np' (line 259)
    np_366653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 21), 'np', False)
    # Obtaining the member 'arange' of a type (line 259)
    arange_366654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 21), np_366653, 'arange')
    # Calling arange(args, kwargs) (line 259)
    arange_call_result_366661 = invoke(stypy.reporting.localization.Localization(__file__, 259, 21), arange_366654, *[result_add_366657], **kwargs_366660)
    
    # Assigning a type to the variable 'indptr' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'indptr', arange_call_result_366661)
    
    # Assigning a Call to a Name (line 260):
    
    # Assigning a Call to a Name (line 260):
    
    # Call to arange(...): (line 260)
    # Processing the call arguments (line 260)
    # Getting the type of 'n' (line 260)
    n_366664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 32), 'n', False)
    # Processing the call keyword arguments (line 260)
    # Getting the type of 'idx_dtype' (line 260)
    idx_dtype_366665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 41), 'idx_dtype', False)
    keyword_366666 = idx_dtype_366665
    kwargs_366667 = {'dtype': keyword_366666}
    # Getting the type of 'np' (line 260)
    np_366662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 22), 'np', False)
    # Obtaining the member 'arange' of a type (line 260)
    arange_366663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 22), np_366662, 'arange')
    # Calling arange(args, kwargs) (line 260)
    arange_call_result_366668 = invoke(stypy.reporting.localization.Localization(__file__, 260, 22), arange_366663, *[n_366664], **kwargs_366667)
    
    # Assigning a type to the variable 'indices' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'indices', arange_call_result_366668)
    
    # Assigning a Call to a Name (line 261):
    
    # Assigning a Call to a Name (line 261):
    
    # Call to ones(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'n' (line 261)
    n_366671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 27), 'n', False)
    # Processing the call keyword arguments (line 261)
    # Getting the type of 'dtype' (line 261)
    dtype_366672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 36), 'dtype', False)
    keyword_366673 = dtype_366672
    kwargs_366674 = {'dtype': keyword_366673}
    # Getting the type of 'np' (line 261)
    np_366669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 19), 'np', False)
    # Obtaining the member 'ones' of a type (line 261)
    ones_366670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 19), np_366669, 'ones')
    # Calling ones(args, kwargs) (line 261)
    ones_call_result_366675 = invoke(stypy.reporting.localization.Localization(__file__, 261, 19), ones_366670, *[n_366671], **kwargs_366674)
    
    # Assigning a type to the variable 'data' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'data', ones_call_result_366675)
    
    # Assigning a Subscript to a Name (line 262):
    
    # Assigning a Subscript to a Name (line 262):
    
    # Obtaining the type of the subscript
    # Getting the type of 'format' (line 262)
    format_366676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 57), 'format')
    
    # Obtaining an instance of the builtin type 'dict' (line 262)
    dict_366677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 18), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 262)
    # Adding element type (key, value) (line 262)
    str_366678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 19), 'str', 'csr')
    # Getting the type of 'csr_matrix' (line 262)
    csr_matrix_366679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 26), 'csr_matrix')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 18), dict_366677, (str_366678, csr_matrix_366679))
    # Adding element type (key, value) (line 262)
    str_366680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 38), 'str', 'csc')
    # Getting the type of 'csc_matrix' (line 262)
    csc_matrix_366681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 45), 'csc_matrix')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 18), dict_366677, (str_366680, csc_matrix_366681))
    
    # Obtaining the member '__getitem__' of a type (line 262)
    getitem___366682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 18), dict_366677, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 262)
    subscript_call_result_366683 = invoke(stypy.reporting.localization.Localization(__file__, 262, 18), getitem___366682, format_366676)
    
    # Assigning a type to the variable 'cls' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'cls', subscript_call_result_366683)
    
    # Call to cls(...): (line 263)
    # Processing the call arguments (line 263)
    
    # Obtaining an instance of the builtin type 'tuple' (line 263)
    tuple_366685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 263)
    # Adding element type (line 263)
    # Getting the type of 'data' (line 263)
    data_366686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 24), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 24), tuple_366685, data_366686)
    # Adding element type (line 263)
    # Getting the type of 'indices' (line 263)
    indices_366687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 29), 'indices', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 24), tuple_366685, indices_366687)
    # Adding element type (line 263)
    # Getting the type of 'indptr' (line 263)
    indptr_366688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 37), 'indptr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 24), tuple_366685, indptr_366688)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 263)
    tuple_366689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 263)
    # Adding element type (line 263)
    # Getting the type of 'n' (line 263)
    n_366690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 46), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 46), tuple_366689, n_366690)
    # Adding element type (line 263)
    # Getting the type of 'n' (line 263)
    n_366691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 48), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 46), tuple_366689, n_366691)
    
    # Processing the call keyword arguments (line 263)
    kwargs_366692 = {}
    # Getting the type of 'cls' (line 263)
    cls_366684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), 'cls', False)
    # Calling cls(args, kwargs) (line 263)
    cls_call_result_366693 = invoke(stypy.reporting.localization.Localization(__file__, 263, 19), cls_366684, *[tuple_366685, tuple_366689], **kwargs_366692)
    
    # Assigning a type to the variable 'stypy_return_type' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'stypy_return_type', cls_call_result_366693)
    # SSA branch for the else part of an if statement (line 257)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'format' (line 264)
    format_366694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 13), 'format')
    str_366695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 23), 'str', 'coo')
    # Applying the binary operator '==' (line 264)
    result_eq_366696 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 13), '==', format_366694, str_366695)
    
    # Testing the type of an if condition (line 264)
    if_condition_366697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 264, 13), result_eq_366696)
    # Assigning a type to the variable 'if_condition_366697' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 13), 'if_condition_366697', if_condition_366697)
    # SSA begins for if statement (line 264)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 265):
    
    # Assigning a Call to a Name (line 265):
    
    # Call to get_index_dtype(...): (line 265)
    # Processing the call keyword arguments (line 265)
    # Getting the type of 'n' (line 265)
    n_366699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 47), 'n', False)
    keyword_366700 = n_366699
    kwargs_366701 = {'maxval': keyword_366700}
    # Getting the type of 'get_index_dtype' (line 265)
    get_index_dtype_366698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 24), 'get_index_dtype', False)
    # Calling get_index_dtype(args, kwargs) (line 265)
    get_index_dtype_call_result_366702 = invoke(stypy.reporting.localization.Localization(__file__, 265, 24), get_index_dtype_366698, *[], **kwargs_366701)
    
    # Assigning a type to the variable 'idx_dtype' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'idx_dtype', get_index_dtype_call_result_366702)
    
    # Assigning a Call to a Name (line 266):
    
    # Assigning a Call to a Name (line 266):
    
    # Call to arange(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'n' (line 266)
    n_366705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 28), 'n', False)
    # Processing the call keyword arguments (line 266)
    # Getting the type of 'idx_dtype' (line 266)
    idx_dtype_366706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 37), 'idx_dtype', False)
    keyword_366707 = idx_dtype_366706
    kwargs_366708 = {'dtype': keyword_366707}
    # Getting the type of 'np' (line 266)
    np_366703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 18), 'np', False)
    # Obtaining the member 'arange' of a type (line 266)
    arange_366704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 18), np_366703, 'arange')
    # Calling arange(args, kwargs) (line 266)
    arange_call_result_366709 = invoke(stypy.reporting.localization.Localization(__file__, 266, 18), arange_366704, *[n_366705], **kwargs_366708)
    
    # Assigning a type to the variable 'row' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'row', arange_call_result_366709)
    
    # Assigning a Call to a Name (line 267):
    
    # Assigning a Call to a Name (line 267):
    
    # Call to arange(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'n' (line 267)
    n_366712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 28), 'n', False)
    # Processing the call keyword arguments (line 267)
    # Getting the type of 'idx_dtype' (line 267)
    idx_dtype_366713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 37), 'idx_dtype', False)
    keyword_366714 = idx_dtype_366713
    kwargs_366715 = {'dtype': keyword_366714}
    # Getting the type of 'np' (line 267)
    np_366710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 18), 'np', False)
    # Obtaining the member 'arange' of a type (line 267)
    arange_366711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 18), np_366710, 'arange')
    # Calling arange(args, kwargs) (line 267)
    arange_call_result_366716 = invoke(stypy.reporting.localization.Localization(__file__, 267, 18), arange_366711, *[n_366712], **kwargs_366715)
    
    # Assigning a type to the variable 'col' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'col', arange_call_result_366716)
    
    # Assigning a Call to a Name (line 268):
    
    # Assigning a Call to a Name (line 268):
    
    # Call to ones(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'n' (line 268)
    n_366719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 27), 'n', False)
    # Processing the call keyword arguments (line 268)
    # Getting the type of 'dtype' (line 268)
    dtype_366720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 36), 'dtype', False)
    keyword_366721 = dtype_366720
    kwargs_366722 = {'dtype': keyword_366721}
    # Getting the type of 'np' (line 268)
    np_366717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'np', False)
    # Obtaining the member 'ones' of a type (line 268)
    ones_366718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 19), np_366717, 'ones')
    # Calling ones(args, kwargs) (line 268)
    ones_call_result_366723 = invoke(stypy.reporting.localization.Localization(__file__, 268, 19), ones_366718, *[n_366719], **kwargs_366722)
    
    # Assigning a type to the variable 'data' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'data', ones_call_result_366723)
    
    # Call to coo_matrix(...): (line 269)
    # Processing the call arguments (line 269)
    
    # Obtaining an instance of the builtin type 'tuple' (line 269)
    tuple_366725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 269)
    # Adding element type (line 269)
    # Getting the type of 'data' (line 269)
    data_366726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 31), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 31), tuple_366725, data_366726)
    # Adding element type (line 269)
    
    # Obtaining an instance of the builtin type 'tuple' (line 269)
    tuple_366727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 269)
    # Adding element type (line 269)
    # Getting the type of 'row' (line 269)
    row_366728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 37), 'row', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 37), tuple_366727, row_366728)
    # Adding element type (line 269)
    # Getting the type of 'col' (line 269)
    col_366729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 41), 'col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 37), tuple_366727, col_366729)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 31), tuple_366725, tuple_366727)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 269)
    tuple_366730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 269)
    # Adding element type (line 269)
    # Getting the type of 'n' (line 269)
    n_366731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 48), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 48), tuple_366730, n_366731)
    # Adding element type (line 269)
    # Getting the type of 'n' (line 269)
    n_366732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 50), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 48), tuple_366730, n_366732)
    
    # Processing the call keyword arguments (line 269)
    kwargs_366733 = {}
    # Getting the type of 'coo_matrix' (line 269)
    coo_matrix_366724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 19), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 269)
    coo_matrix_call_result_366734 = invoke(stypy.reporting.localization.Localization(__file__, 269, 19), coo_matrix_366724, *[tuple_366725, tuple_366730], **kwargs_366733)
    
    # Assigning a type to the variable 'stypy_return_type' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'stypy_return_type', coo_matrix_call_result_366734)
    # SSA join for if statement (line 264)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 257)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 255)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 271):
    
    # Assigning a Call to a Name (line 271):
    
    # Call to ones(...): (line 271)
    # Processing the call arguments (line 271)
    
    # Obtaining an instance of the builtin type 'tuple' (line 271)
    tuple_366737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 271)
    # Adding element type (line 271)
    int_366738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 21), tuple_366737, int_366738)
    # Adding element type (line 271)
    
    # Call to max(...): (line 271)
    # Processing the call arguments (line 271)
    int_366740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 28), 'int')
    
    # Call to min(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'm' (line 271)
    m_366742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 35), 'm', False)
    # Getting the type of 'k' (line 271)
    k_366743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 39), 'k', False)
    # Applying the binary operator '+' (line 271)
    result_add_366744 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 35), '+', m_366742, k_366743)
    
    # Getting the type of 'n' (line 271)
    n_366745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 42), 'n', False)
    # Processing the call keyword arguments (line 271)
    kwargs_366746 = {}
    # Getting the type of 'min' (line 271)
    min_366741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 31), 'min', False)
    # Calling min(args, kwargs) (line 271)
    min_call_result_366747 = invoke(stypy.reporting.localization.Localization(__file__, 271, 31), min_366741, *[result_add_366744, n_366745], **kwargs_366746)
    
    # Processing the call keyword arguments (line 271)
    kwargs_366748 = {}
    # Getting the type of 'max' (line 271)
    max_366739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 24), 'max', False)
    # Calling max(args, kwargs) (line 271)
    max_call_result_366749 = invoke(stypy.reporting.localization.Localization(__file__, 271, 24), max_366739, *[int_366740, min_call_result_366747], **kwargs_366748)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 21), tuple_366737, max_call_result_366749)
    
    # Processing the call keyword arguments (line 271)
    # Getting the type of 'dtype' (line 271)
    dtype_366750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 54), 'dtype', False)
    keyword_366751 = dtype_366750
    kwargs_366752 = {'dtype': keyword_366751}
    # Getting the type of 'np' (line 271)
    np_366735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'np', False)
    # Obtaining the member 'ones' of a type (line 271)
    ones_366736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), np_366735, 'ones')
    # Calling ones(args, kwargs) (line 271)
    ones_call_result_366753 = invoke(stypy.reporting.localization.Localization(__file__, 271, 12), ones_366736, *[tuple_366737], **kwargs_366752)
    
    # Assigning a type to the variable 'diags' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'diags', ones_call_result_366753)
    
    # Call to asformat(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'format' (line 272)
    format_366762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 44), 'format', False)
    # Processing the call keyword arguments (line 272)
    kwargs_366763 = {}
    
    # Call to spdiags(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'diags' (line 272)
    diags_366755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 19), 'diags', False)
    # Getting the type of 'k' (line 272)
    k_366756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 26), 'k', False)
    # Getting the type of 'm' (line 272)
    m_366757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 29), 'm', False)
    # Getting the type of 'n' (line 272)
    n_366758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 32), 'n', False)
    # Processing the call keyword arguments (line 272)
    kwargs_366759 = {}
    # Getting the type of 'spdiags' (line 272)
    spdiags_366754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'spdiags', False)
    # Calling spdiags(args, kwargs) (line 272)
    spdiags_call_result_366760 = invoke(stypy.reporting.localization.Localization(__file__, 272, 11), spdiags_366754, *[diags_366755, k_366756, m_366757, n_366758], **kwargs_366759)
    
    # Obtaining the member 'asformat' of a type (line 272)
    asformat_366761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 11), spdiags_call_result_366760, 'asformat')
    # Calling asformat(args, kwargs) (line 272)
    asformat_call_result_366764 = invoke(stypy.reporting.localization.Localization(__file__, 272, 11), asformat_366761, *[format_366762], **kwargs_366763)
    
    # Assigning a type to the variable 'stypy_return_type' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type', asformat_call_result_366764)
    
    # ################# End of 'eye(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eye' in the type store
    # Getting the type of 'stypy_return_type' (line 220)
    stypy_return_type_366765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_366765)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eye'
    return stypy_return_type_366765

# Assigning a type to the variable 'eye' (line 220)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'eye', eye)

@norecursion
def kron(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 275)
    None_366766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 22), 'None')
    defaults = [None_366766]
    # Create a new context for function 'kron'
    module_type_store = module_type_store.open_function_context('kron', 275, 0, False)
    
    # Passed parameters checking function
    kron.stypy_localization = localization
    kron.stypy_type_of_self = None
    kron.stypy_type_store = module_type_store
    kron.stypy_function_name = 'kron'
    kron.stypy_param_names_list = ['A', 'B', 'format']
    kron.stypy_varargs_param_name = None
    kron.stypy_kwargs_param_name = None
    kron.stypy_call_defaults = defaults
    kron.stypy_call_varargs = varargs
    kron.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'kron', ['A', 'B', 'format'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'kron', localization, ['A', 'B', 'format'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'kron(...)' code ##################

    str_366767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, (-1)), 'str', 'kronecker product of sparse matrices A and B\n\n    Parameters\n    ----------\n    A : sparse or dense matrix\n        first matrix of the product\n    B : sparse or dense matrix\n        second matrix of the product\n    format : str, optional\n        format of the result (e.g. "csr")\n\n    Returns\n    -------\n    kronecker product in a sparse matrix format\n\n\n    Examples\n    --------\n    >>> from scipy import sparse\n    >>> A = sparse.csr_matrix(np.array([[0, 2], [5, 0]]))\n    >>> B = sparse.csr_matrix(np.array([[1, 2], [3, 4]]))\n    >>> sparse.kron(A, B).toarray()\n    array([[ 0,  0,  2,  4],\n           [ 0,  0,  6,  8],\n           [ 5, 10,  0,  0],\n           [15, 20,  0,  0]])\n\n    >>> sparse.kron(A, [[1, 2], [3, 4]]).toarray()\n    array([[ 0,  0,  2,  4],\n           [ 0,  0,  6,  8],\n           [ 5, 10,  0,  0],\n           [15, 20,  0,  0]])\n\n    ')
    
    # Assigning a Call to a Name (line 310):
    
    # Assigning a Call to a Name (line 310):
    
    # Call to coo_matrix(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'B' (line 310)
    B_366769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 'B', False)
    # Processing the call keyword arguments (line 310)
    kwargs_366770 = {}
    # Getting the type of 'coo_matrix' (line 310)
    coo_matrix_366768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 310)
    coo_matrix_call_result_366771 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), coo_matrix_366768, *[B_366769], **kwargs_366770)
    
    # Assigning a type to the variable 'B' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'B', coo_matrix_call_result_366771)
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Getting the type of 'format' (line 312)
    format_366772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'format')
    # Getting the type of 'None' (line 312)
    None_366773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 18), 'None')
    # Applying the binary operator 'is' (line 312)
    result_is__366774 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 8), 'is', format_366772, None_366773)
    
    
    # Getting the type of 'format' (line 312)
    format_366775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 26), 'format')
    str_366776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 36), 'str', 'bsr')
    # Applying the binary operator '==' (line 312)
    result_eq_366777 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 26), '==', format_366775, str_366776)
    
    # Applying the binary operator 'or' (line 312)
    result_or_keyword_366778 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 8), 'or', result_is__366774, result_eq_366777)
    
    
    int_366779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 47), 'int')
    # Getting the type of 'B' (line 312)
    B_366780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 49), 'B')
    # Obtaining the member 'nnz' of a type (line 312)
    nnz_366781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 49), B_366780, 'nnz')
    # Applying the binary operator '*' (line 312)
    result_mul_366782 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 47), '*', int_366779, nnz_366781)
    
    
    # Obtaining the type of the subscript
    int_366783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 66), 'int')
    # Getting the type of 'B' (line 312)
    B_366784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 58), 'B')
    # Obtaining the member 'shape' of a type (line 312)
    shape_366785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 58), B_366784, 'shape')
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___366786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 58), shape_366785, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_366787 = invoke(stypy.reporting.localization.Localization(__file__, 312, 58), getitem___366786, int_366783)
    
    
    # Obtaining the type of the subscript
    int_366788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 79), 'int')
    # Getting the type of 'B' (line 312)
    B_366789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 71), 'B')
    # Obtaining the member 'shape' of a type (line 312)
    shape_366790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 71), B_366789, 'shape')
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___366791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 71), shape_366790, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_366792 = invoke(stypy.reporting.localization.Localization(__file__, 312, 71), getitem___366791, int_366788)
    
    # Applying the binary operator '*' (line 312)
    result_mul_366793 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 58), '*', subscript_call_result_366787, subscript_call_result_366792)
    
    # Applying the binary operator '>=' (line 312)
    result_ge_366794 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 47), '>=', result_mul_366782, result_mul_366793)
    
    # Applying the binary operator 'and' (line 312)
    result_and_keyword_366795 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 7), 'and', result_or_keyword_366778, result_ge_366794)
    
    # Testing the type of an if condition (line 312)
    if_condition_366796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 4), result_and_keyword_366795)
    # Assigning a type to the variable 'if_condition_366796' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'if_condition_366796', if_condition_366796)
    # SSA begins for if statement (line 312)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 314):
    
    # Assigning a Call to a Name (line 314):
    
    # Call to csr_matrix(...): (line 314)
    # Processing the call arguments (line 314)
    # Getting the type of 'A' (line 314)
    A_366798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 23), 'A', False)
    # Processing the call keyword arguments (line 314)
    # Getting the type of 'True' (line 314)
    True_366799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 30), 'True', False)
    keyword_366800 = True_366799
    kwargs_366801 = {'copy': keyword_366800}
    # Getting the type of 'csr_matrix' (line 314)
    csr_matrix_366797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 314)
    csr_matrix_call_result_366802 = invoke(stypy.reporting.localization.Localization(__file__, 314, 12), csr_matrix_366797, *[A_366798], **kwargs_366801)
    
    # Assigning a type to the variable 'A' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'A', csr_matrix_call_result_366802)
    
    # Assigning a Tuple to a Name (line 316):
    
    # Assigning a Tuple to a Name (line 316):
    
    # Obtaining an instance of the builtin type 'tuple' (line 316)
    tuple_366803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 316)
    # Adding element type (line 316)
    
    # Obtaining the type of the subscript
    int_366804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 32), 'int')
    # Getting the type of 'A' (line 316)
    A_366805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 24), 'A')
    # Obtaining the member 'shape' of a type (line 316)
    shape_366806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 24), A_366805, 'shape')
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___366807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 24), shape_366806, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_366808 = invoke(stypy.reporting.localization.Localization(__file__, 316, 24), getitem___366807, int_366804)
    
    
    # Obtaining the type of the subscript
    int_366809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 43), 'int')
    # Getting the type of 'B' (line 316)
    B_366810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 35), 'B')
    # Obtaining the member 'shape' of a type (line 316)
    shape_366811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 35), B_366810, 'shape')
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___366812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 35), shape_366811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_366813 = invoke(stypy.reporting.localization.Localization(__file__, 316, 35), getitem___366812, int_366809)
    
    # Applying the binary operator '*' (line 316)
    result_mul_366814 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 24), '*', subscript_call_result_366808, subscript_call_result_366813)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 24), tuple_366803, result_mul_366814)
    # Adding element type (line 316)
    
    # Obtaining the type of the subscript
    int_366815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 55), 'int')
    # Getting the type of 'A' (line 316)
    A_366816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 47), 'A')
    # Obtaining the member 'shape' of a type (line 316)
    shape_366817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 47), A_366816, 'shape')
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___366818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 47), shape_366817, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_366819 = invoke(stypy.reporting.localization.Localization(__file__, 316, 47), getitem___366818, int_366815)
    
    
    # Obtaining the type of the subscript
    int_366820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 66), 'int')
    # Getting the type of 'B' (line 316)
    B_366821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 58), 'B')
    # Obtaining the member 'shape' of a type (line 316)
    shape_366822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 58), B_366821, 'shape')
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___366823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 58), shape_366822, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_366824 = invoke(stypy.reporting.localization.Localization(__file__, 316, 58), getitem___366823, int_366820)
    
    # Applying the binary operator '*' (line 316)
    result_mul_366825 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 47), '*', subscript_call_result_366819, subscript_call_result_366824)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 24), tuple_366803, result_mul_366825)
    
    # Assigning a type to the variable 'output_shape' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'output_shape', tuple_366803)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'A' (line 318)
    A_366826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 11), 'A')
    # Obtaining the member 'nnz' of a type (line 318)
    nnz_366827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 11), A_366826, 'nnz')
    int_366828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 20), 'int')
    # Applying the binary operator '==' (line 318)
    result_eq_366829 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 11), '==', nnz_366827, int_366828)
    
    
    # Getting the type of 'B' (line 318)
    B_366830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 'B')
    # Obtaining the member 'nnz' of a type (line 318)
    nnz_366831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 25), B_366830, 'nnz')
    int_366832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 34), 'int')
    # Applying the binary operator '==' (line 318)
    result_eq_366833 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 25), '==', nnz_366831, int_366832)
    
    # Applying the binary operator 'or' (line 318)
    result_or_keyword_366834 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 11), 'or', result_eq_366829, result_eq_366833)
    
    # Testing the type of an if condition (line 318)
    if_condition_366835 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 8), result_or_keyword_366834)
    # Assigning a type to the variable 'if_condition_366835' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'if_condition_366835', if_condition_366835)
    # SSA begins for if statement (line 318)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to coo_matrix(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'output_shape' (line 320)
    output_shape_366837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 30), 'output_shape', False)
    # Processing the call keyword arguments (line 320)
    kwargs_366838 = {}
    # Getting the type of 'coo_matrix' (line 320)
    coo_matrix_366836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 19), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 320)
    coo_matrix_call_result_366839 = invoke(stypy.reporting.localization.Localization(__file__, 320, 19), coo_matrix_366836, *[output_shape_366837], **kwargs_366838)
    
    # Assigning a type to the variable 'stypy_return_type' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'stypy_return_type', coo_matrix_call_result_366839)
    # SSA join for if statement (line 318)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 322):
    
    # Assigning a Call to a Name (line 322):
    
    # Call to toarray(...): (line 322)
    # Processing the call keyword arguments (line 322)
    kwargs_366842 = {}
    # Getting the type of 'B' (line 322)
    B_366840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'B', False)
    # Obtaining the member 'toarray' of a type (line 322)
    toarray_366841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 12), B_366840, 'toarray')
    # Calling toarray(args, kwargs) (line 322)
    toarray_call_result_366843 = invoke(stypy.reporting.localization.Localization(__file__, 322, 12), toarray_366841, *[], **kwargs_366842)
    
    # Assigning a type to the variable 'B' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'B', toarray_call_result_366843)
    
    # Assigning a Call to a Name (line 323):
    
    # Assigning a Call to a Name (line 323):
    
    # Call to reshape(...): (line 323)
    # Processing the call arguments (line 323)
    int_366852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 45), 'int')
    
    # Obtaining the type of the subscript
    int_366853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 56), 'int')
    # Getting the type of 'B' (line 323)
    B_366854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 48), 'B', False)
    # Obtaining the member 'shape' of a type (line 323)
    shape_366855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 48), B_366854, 'shape')
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___366856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 48), shape_366855, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
    subscript_call_result_366857 = invoke(stypy.reporting.localization.Localization(__file__, 323, 48), getitem___366856, int_366853)
    
    
    # Obtaining the type of the subscript
    int_366858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 67), 'int')
    # Getting the type of 'B' (line 323)
    B_366859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 59), 'B', False)
    # Obtaining the member 'shape' of a type (line 323)
    shape_366860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 59), B_366859, 'shape')
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___366861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 59), shape_366860, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
    subscript_call_result_366862 = invoke(stypy.reporting.localization.Localization(__file__, 323, 59), getitem___366861, int_366858)
    
    # Processing the call keyword arguments (line 323)
    kwargs_366863 = {}
    
    # Call to repeat(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'B' (line 323)
    B_366847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 29), 'B', False)
    # Obtaining the member 'size' of a type (line 323)
    size_366848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 29), B_366847, 'size')
    # Processing the call keyword arguments (line 323)
    kwargs_366849 = {}
    # Getting the type of 'A' (line 323)
    A_366844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), 'A', False)
    # Obtaining the member 'data' of a type (line 323)
    data_366845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 15), A_366844, 'data')
    # Obtaining the member 'repeat' of a type (line 323)
    repeat_366846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 15), data_366845, 'repeat')
    # Calling repeat(args, kwargs) (line 323)
    repeat_call_result_366850 = invoke(stypy.reporting.localization.Localization(__file__, 323, 15), repeat_366846, *[size_366848], **kwargs_366849)
    
    # Obtaining the member 'reshape' of a type (line 323)
    reshape_366851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 15), repeat_call_result_366850, 'reshape')
    # Calling reshape(args, kwargs) (line 323)
    reshape_call_result_366864 = invoke(stypy.reporting.localization.Localization(__file__, 323, 15), reshape_366851, *[int_366852, subscript_call_result_366857, subscript_call_result_366862], **kwargs_366863)
    
    # Assigning a type to the variable 'data' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'data', reshape_call_result_366864)
    
    # Assigning a BinOp to a Name (line 324):
    
    # Assigning a BinOp to a Name (line 324):
    # Getting the type of 'data' (line 324)
    data_366865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 15), 'data')
    # Getting the type of 'B' (line 324)
    B_366866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 22), 'B')
    # Applying the binary operator '*' (line 324)
    result_mul_366867 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 15), '*', data_366865, B_366866)
    
    # Assigning a type to the variable 'data' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'data', result_mul_366867)
    
    # Call to bsr_matrix(...): (line 326)
    # Processing the call arguments (line 326)
    
    # Obtaining an instance of the builtin type 'tuple' (line 326)
    tuple_366869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 326)
    # Adding element type (line 326)
    # Getting the type of 'data' (line 326)
    data_366870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 27), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 27), tuple_366869, data_366870)
    # Adding element type (line 326)
    # Getting the type of 'A' (line 326)
    A_366871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 32), 'A', False)
    # Obtaining the member 'indices' of a type (line 326)
    indices_366872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 32), A_366871, 'indices')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 27), tuple_366869, indices_366872)
    # Adding element type (line 326)
    # Getting the type of 'A' (line 326)
    A_366873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 42), 'A', False)
    # Obtaining the member 'indptr' of a type (line 326)
    indptr_366874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 42), A_366873, 'indptr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 27), tuple_366869, indptr_366874)
    
    # Processing the call keyword arguments (line 326)
    # Getting the type of 'output_shape' (line 326)
    output_shape_366875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 59), 'output_shape', False)
    keyword_366876 = output_shape_366875
    kwargs_366877 = {'shape': keyword_366876}
    # Getting the type of 'bsr_matrix' (line 326)
    bsr_matrix_366868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 15), 'bsr_matrix', False)
    # Calling bsr_matrix(args, kwargs) (line 326)
    bsr_matrix_call_result_366878 = invoke(stypy.reporting.localization.Localization(__file__, 326, 15), bsr_matrix_366868, *[tuple_366869], **kwargs_366877)
    
    # Assigning a type to the variable 'stypy_return_type' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'stypy_return_type', bsr_matrix_call_result_366878)
    # SSA branch for the else part of an if statement (line 312)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 329):
    
    # Assigning a Call to a Name (line 329):
    
    # Call to coo_matrix(...): (line 329)
    # Processing the call arguments (line 329)
    # Getting the type of 'A' (line 329)
    A_366880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 23), 'A', False)
    # Processing the call keyword arguments (line 329)
    kwargs_366881 = {}
    # Getting the type of 'coo_matrix' (line 329)
    coo_matrix_366879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 329)
    coo_matrix_call_result_366882 = invoke(stypy.reporting.localization.Localization(__file__, 329, 12), coo_matrix_366879, *[A_366880], **kwargs_366881)
    
    # Assigning a type to the variable 'A' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'A', coo_matrix_call_result_366882)
    
    # Assigning a Tuple to a Name (line 330):
    
    # Assigning a Tuple to a Name (line 330):
    
    # Obtaining an instance of the builtin type 'tuple' (line 330)
    tuple_366883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 330)
    # Adding element type (line 330)
    
    # Obtaining the type of the subscript
    int_366884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 32), 'int')
    # Getting the type of 'A' (line 330)
    A_366885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 24), 'A')
    # Obtaining the member 'shape' of a type (line 330)
    shape_366886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 24), A_366885, 'shape')
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___366887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 24), shape_366886, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_366888 = invoke(stypy.reporting.localization.Localization(__file__, 330, 24), getitem___366887, int_366884)
    
    
    # Obtaining the type of the subscript
    int_366889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 43), 'int')
    # Getting the type of 'B' (line 330)
    B_366890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 35), 'B')
    # Obtaining the member 'shape' of a type (line 330)
    shape_366891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 35), B_366890, 'shape')
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___366892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 35), shape_366891, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_366893 = invoke(stypy.reporting.localization.Localization(__file__, 330, 35), getitem___366892, int_366889)
    
    # Applying the binary operator '*' (line 330)
    result_mul_366894 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 24), '*', subscript_call_result_366888, subscript_call_result_366893)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 24), tuple_366883, result_mul_366894)
    # Adding element type (line 330)
    
    # Obtaining the type of the subscript
    int_366895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 55), 'int')
    # Getting the type of 'A' (line 330)
    A_366896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 47), 'A')
    # Obtaining the member 'shape' of a type (line 330)
    shape_366897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 47), A_366896, 'shape')
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___366898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 47), shape_366897, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_366899 = invoke(stypy.reporting.localization.Localization(__file__, 330, 47), getitem___366898, int_366895)
    
    
    # Obtaining the type of the subscript
    int_366900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 66), 'int')
    # Getting the type of 'B' (line 330)
    B_366901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 58), 'B')
    # Obtaining the member 'shape' of a type (line 330)
    shape_366902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 58), B_366901, 'shape')
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___366903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 58), shape_366902, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_366904 = invoke(stypy.reporting.localization.Localization(__file__, 330, 58), getitem___366903, int_366900)
    
    # Applying the binary operator '*' (line 330)
    result_mul_366905 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 47), '*', subscript_call_result_366899, subscript_call_result_366904)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 24), tuple_366883, result_mul_366905)
    
    # Assigning a type to the variable 'output_shape' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'output_shape', tuple_366883)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'A' (line 332)
    A_366906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 11), 'A')
    # Obtaining the member 'nnz' of a type (line 332)
    nnz_366907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 11), A_366906, 'nnz')
    int_366908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 20), 'int')
    # Applying the binary operator '==' (line 332)
    result_eq_366909 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 11), '==', nnz_366907, int_366908)
    
    
    # Getting the type of 'B' (line 332)
    B_366910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 25), 'B')
    # Obtaining the member 'nnz' of a type (line 332)
    nnz_366911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 25), B_366910, 'nnz')
    int_366912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 34), 'int')
    # Applying the binary operator '==' (line 332)
    result_eq_366913 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 25), '==', nnz_366911, int_366912)
    
    # Applying the binary operator 'or' (line 332)
    result_or_keyword_366914 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 11), 'or', result_eq_366909, result_eq_366913)
    
    # Testing the type of an if condition (line 332)
    if_condition_366915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 8), result_or_keyword_366914)
    # Assigning a type to the variable 'if_condition_366915' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'if_condition_366915', if_condition_366915)
    # SSA begins for if statement (line 332)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to coo_matrix(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'output_shape' (line 334)
    output_shape_366917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 30), 'output_shape', False)
    # Processing the call keyword arguments (line 334)
    kwargs_366918 = {}
    # Getting the type of 'coo_matrix' (line 334)
    coo_matrix_366916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 334)
    coo_matrix_call_result_366919 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), coo_matrix_366916, *[output_shape_366917], **kwargs_366918)
    
    # Assigning a type to the variable 'stypy_return_type' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'stypy_return_type', coo_matrix_call_result_366919)
    # SSA join for if statement (line 332)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 337):
    
    # Assigning a Call to a Name (line 337):
    
    # Call to repeat(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'B' (line 337)
    B_366923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 27), 'B', False)
    # Obtaining the member 'nnz' of a type (line 337)
    nnz_366924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 27), B_366923, 'nnz')
    # Processing the call keyword arguments (line 337)
    kwargs_366925 = {}
    # Getting the type of 'A' (line 337)
    A_366920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 14), 'A', False)
    # Obtaining the member 'row' of a type (line 337)
    row_366921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 14), A_366920, 'row')
    # Obtaining the member 'repeat' of a type (line 337)
    repeat_366922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 14), row_366921, 'repeat')
    # Calling repeat(args, kwargs) (line 337)
    repeat_call_result_366926 = invoke(stypy.reporting.localization.Localization(__file__, 337, 14), repeat_366922, *[nnz_366924], **kwargs_366925)
    
    # Assigning a type to the variable 'row' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'row', repeat_call_result_366926)
    
    # Assigning a Call to a Name (line 338):
    
    # Assigning a Call to a Name (line 338):
    
    # Call to repeat(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'B' (line 338)
    B_366930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 27), 'B', False)
    # Obtaining the member 'nnz' of a type (line 338)
    nnz_366931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 27), B_366930, 'nnz')
    # Processing the call keyword arguments (line 338)
    kwargs_366932 = {}
    # Getting the type of 'A' (line 338)
    A_366927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 14), 'A', False)
    # Obtaining the member 'col' of a type (line 338)
    col_366928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 14), A_366927, 'col')
    # Obtaining the member 'repeat' of a type (line 338)
    repeat_366929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 14), col_366928, 'repeat')
    # Calling repeat(args, kwargs) (line 338)
    repeat_call_result_366933 = invoke(stypy.reporting.localization.Localization(__file__, 338, 14), repeat_366929, *[nnz_366931], **kwargs_366932)
    
    # Assigning a type to the variable 'col' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'col', repeat_call_result_366933)
    
    # Assigning a Call to a Name (line 339):
    
    # Assigning a Call to a Name (line 339):
    
    # Call to repeat(...): (line 339)
    # Processing the call arguments (line 339)
    # Getting the type of 'B' (line 339)
    B_366937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 29), 'B', False)
    # Obtaining the member 'nnz' of a type (line 339)
    nnz_366938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 29), B_366937, 'nnz')
    # Processing the call keyword arguments (line 339)
    kwargs_366939 = {}
    # Getting the type of 'A' (line 339)
    A_366934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'A', False)
    # Obtaining the member 'data' of a type (line 339)
    data_366935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 15), A_366934, 'data')
    # Obtaining the member 'repeat' of a type (line 339)
    repeat_366936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 15), data_366935, 'repeat')
    # Calling repeat(args, kwargs) (line 339)
    repeat_call_result_366940 = invoke(stypy.reporting.localization.Localization(__file__, 339, 15), repeat_366936, *[nnz_366938], **kwargs_366939)
    
    # Assigning a type to the variable 'data' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'data', repeat_call_result_366940)
    
    # Getting the type of 'row' (line 341)
    row_366941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'row')
    
    # Obtaining the type of the subscript
    int_366942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 23), 'int')
    # Getting the type of 'B' (line 341)
    B_366943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'B')
    # Obtaining the member 'shape' of a type (line 341)
    shape_366944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), B_366943, 'shape')
    # Obtaining the member '__getitem__' of a type (line 341)
    getitem___366945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), shape_366944, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 341)
    subscript_call_result_366946 = invoke(stypy.reporting.localization.Localization(__file__, 341, 15), getitem___366945, int_366942)
    
    # Applying the binary operator '*=' (line 341)
    result_imul_366947 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 8), '*=', row_366941, subscript_call_result_366946)
    # Assigning a type to the variable 'row' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'row', result_imul_366947)
    
    
    # Getting the type of 'col' (line 342)
    col_366948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'col')
    
    # Obtaining the type of the subscript
    int_366949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 23), 'int')
    # Getting the type of 'B' (line 342)
    B_366950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 'B')
    # Obtaining the member 'shape' of a type (line 342)
    shape_366951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), B_366950, 'shape')
    # Obtaining the member '__getitem__' of a type (line 342)
    getitem___366952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), shape_366951, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 342)
    subscript_call_result_366953 = invoke(stypy.reporting.localization.Localization(__file__, 342, 15), getitem___366952, int_366949)
    
    # Applying the binary operator '*=' (line 342)
    result_imul_366954 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 8), '*=', col_366948, subscript_call_result_366953)
    # Assigning a type to the variable 'col' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'col', result_imul_366954)
    
    
    # Assigning a Tuple to a Tuple (line 345):
    
    # Assigning a Call to a Name (line 345):
    
    # Call to reshape(...): (line 345)
    # Processing the call arguments (line 345)
    int_366957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 30), 'int')
    # Getting the type of 'B' (line 345)
    B_366958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 33), 'B', False)
    # Obtaining the member 'nnz' of a type (line 345)
    nnz_366959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 33), B_366958, 'nnz')
    # Processing the call keyword arguments (line 345)
    kwargs_366960 = {}
    # Getting the type of 'row' (line 345)
    row_366955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 18), 'row', False)
    # Obtaining the member 'reshape' of a type (line 345)
    reshape_366956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 18), row_366955, 'reshape')
    # Calling reshape(args, kwargs) (line 345)
    reshape_call_result_366961 = invoke(stypy.reporting.localization.Localization(__file__, 345, 18), reshape_366956, *[int_366957, nnz_366959], **kwargs_366960)
    
    # Assigning a type to the variable 'tuple_assignment_366301' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_assignment_366301', reshape_call_result_366961)
    
    # Assigning a Call to a Name (line 345):
    
    # Call to reshape(...): (line 345)
    # Processing the call arguments (line 345)
    int_366964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 52), 'int')
    # Getting the type of 'B' (line 345)
    B_366965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 55), 'B', False)
    # Obtaining the member 'nnz' of a type (line 345)
    nnz_366966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 55), B_366965, 'nnz')
    # Processing the call keyword arguments (line 345)
    kwargs_366967 = {}
    # Getting the type of 'col' (line 345)
    col_366962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 40), 'col', False)
    # Obtaining the member 'reshape' of a type (line 345)
    reshape_366963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 40), col_366962, 'reshape')
    # Calling reshape(args, kwargs) (line 345)
    reshape_call_result_366968 = invoke(stypy.reporting.localization.Localization(__file__, 345, 40), reshape_366963, *[int_366964, nnz_366966], **kwargs_366967)
    
    # Assigning a type to the variable 'tuple_assignment_366302' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_assignment_366302', reshape_call_result_366968)
    
    # Assigning a Name to a Name (line 345):
    # Getting the type of 'tuple_assignment_366301' (line 345)
    tuple_assignment_366301_366969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_assignment_366301')
    # Assigning a type to the variable 'row' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'row', tuple_assignment_366301_366969)
    
    # Assigning a Name to a Name (line 345):
    # Getting the type of 'tuple_assignment_366302' (line 345)
    tuple_assignment_366302_366970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_assignment_366302')
    # Assigning a type to the variable 'col' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'col', tuple_assignment_366302_366970)
    
    # Getting the type of 'row' (line 346)
    row_366971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'row')
    # Getting the type of 'B' (line 346)
    B_366972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 15), 'B')
    # Obtaining the member 'row' of a type (line 346)
    row_366973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 15), B_366972, 'row')
    # Applying the binary operator '+=' (line 346)
    result_iadd_366974 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 8), '+=', row_366971, row_366973)
    # Assigning a type to the variable 'row' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'row', result_iadd_366974)
    
    
    # Getting the type of 'col' (line 347)
    col_366975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'col')
    # Getting the type of 'B' (line 347)
    B_366976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 15), 'B')
    # Obtaining the member 'col' of a type (line 347)
    col_366977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 15), B_366976, 'col')
    # Applying the binary operator '+=' (line 347)
    result_iadd_366978 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 8), '+=', col_366975, col_366977)
    # Assigning a type to the variable 'col' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'col', result_iadd_366978)
    
    
    # Assigning a Tuple to a Tuple (line 348):
    
    # Assigning a Call to a Name (line 348):
    
    # Call to reshape(...): (line 348)
    # Processing the call arguments (line 348)
    int_366981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 30), 'int')
    # Processing the call keyword arguments (line 348)
    kwargs_366982 = {}
    # Getting the type of 'row' (line 348)
    row_366979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 18), 'row', False)
    # Obtaining the member 'reshape' of a type (line 348)
    reshape_366980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 18), row_366979, 'reshape')
    # Calling reshape(args, kwargs) (line 348)
    reshape_call_result_366983 = invoke(stypy.reporting.localization.Localization(__file__, 348, 18), reshape_366980, *[int_366981], **kwargs_366982)
    
    # Assigning a type to the variable 'tuple_assignment_366303' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'tuple_assignment_366303', reshape_call_result_366983)
    
    # Assigning a Call to a Name (line 348):
    
    # Call to reshape(...): (line 348)
    # Processing the call arguments (line 348)
    int_366986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 46), 'int')
    # Processing the call keyword arguments (line 348)
    kwargs_366987 = {}
    # Getting the type of 'col' (line 348)
    col_366984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 34), 'col', False)
    # Obtaining the member 'reshape' of a type (line 348)
    reshape_366985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 34), col_366984, 'reshape')
    # Calling reshape(args, kwargs) (line 348)
    reshape_call_result_366988 = invoke(stypy.reporting.localization.Localization(__file__, 348, 34), reshape_366985, *[int_366986], **kwargs_366987)
    
    # Assigning a type to the variable 'tuple_assignment_366304' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'tuple_assignment_366304', reshape_call_result_366988)
    
    # Assigning a Name to a Name (line 348):
    # Getting the type of 'tuple_assignment_366303' (line 348)
    tuple_assignment_366303_366989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'tuple_assignment_366303')
    # Assigning a type to the variable 'row' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'row', tuple_assignment_366303_366989)
    
    # Assigning a Name to a Name (line 348):
    # Getting the type of 'tuple_assignment_366304' (line 348)
    tuple_assignment_366304_366990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'tuple_assignment_366304')
    # Assigning a type to the variable 'col' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'col', tuple_assignment_366304_366990)
    
    # Assigning a BinOp to a Name (line 351):
    
    # Assigning a BinOp to a Name (line 351):
    
    # Call to reshape(...): (line 351)
    # Processing the call arguments (line 351)
    int_366993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 28), 'int')
    # Getting the type of 'B' (line 351)
    B_366994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 31), 'B', False)
    # Obtaining the member 'nnz' of a type (line 351)
    nnz_366995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 31), B_366994, 'nnz')
    # Processing the call keyword arguments (line 351)
    kwargs_366996 = {}
    # Getting the type of 'data' (line 351)
    data_366991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 15), 'data', False)
    # Obtaining the member 'reshape' of a type (line 351)
    reshape_366992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 15), data_366991, 'reshape')
    # Calling reshape(args, kwargs) (line 351)
    reshape_call_result_366997 = invoke(stypy.reporting.localization.Localization(__file__, 351, 15), reshape_366992, *[int_366993, nnz_366995], **kwargs_366996)
    
    # Getting the type of 'B' (line 351)
    B_366998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 40), 'B')
    # Obtaining the member 'data' of a type (line 351)
    data_366999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 40), B_366998, 'data')
    # Applying the binary operator '*' (line 351)
    result_mul_367000 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 15), '*', reshape_call_result_366997, data_366999)
    
    # Assigning a type to the variable 'data' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'data', result_mul_367000)
    
    # Assigning a Call to a Name (line 352):
    
    # Assigning a Call to a Name (line 352):
    
    # Call to reshape(...): (line 352)
    # Processing the call arguments (line 352)
    int_367003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 28), 'int')
    # Processing the call keyword arguments (line 352)
    kwargs_367004 = {}
    # Getting the type of 'data' (line 352)
    data_367001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 15), 'data', False)
    # Obtaining the member 'reshape' of a type (line 352)
    reshape_367002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 15), data_367001, 'reshape')
    # Calling reshape(args, kwargs) (line 352)
    reshape_call_result_367005 = invoke(stypy.reporting.localization.Localization(__file__, 352, 15), reshape_367002, *[int_367003], **kwargs_367004)
    
    # Assigning a type to the variable 'data' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'data', reshape_call_result_367005)
    
    # Call to asformat(...): (line 354)
    # Processing the call arguments (line 354)
    # Getting the type of 'format' (line 354)
    format_367017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 73), 'format', False)
    # Processing the call keyword arguments (line 354)
    kwargs_367018 = {}
    
    # Call to coo_matrix(...): (line 354)
    # Processing the call arguments (line 354)
    
    # Obtaining an instance of the builtin type 'tuple' (line 354)
    tuple_367007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 354)
    # Adding element type (line 354)
    # Getting the type of 'data' (line 354)
    data_367008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 27), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 27), tuple_367007, data_367008)
    # Adding element type (line 354)
    
    # Obtaining an instance of the builtin type 'tuple' (line 354)
    tuple_367009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 354)
    # Adding element type (line 354)
    # Getting the type of 'row' (line 354)
    row_367010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 33), 'row', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 33), tuple_367009, row_367010)
    # Adding element type (line 354)
    # Getting the type of 'col' (line 354)
    col_367011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 37), 'col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 33), tuple_367009, col_367011)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 27), tuple_367007, tuple_367009)
    
    # Processing the call keyword arguments (line 354)
    # Getting the type of 'output_shape' (line 354)
    output_shape_367012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 50), 'output_shape', False)
    keyword_367013 = output_shape_367012
    kwargs_367014 = {'shape': keyword_367013}
    # Getting the type of 'coo_matrix' (line 354)
    coo_matrix_367006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 15), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 354)
    coo_matrix_call_result_367015 = invoke(stypy.reporting.localization.Localization(__file__, 354, 15), coo_matrix_367006, *[tuple_367007], **kwargs_367014)
    
    # Obtaining the member 'asformat' of a type (line 354)
    asformat_367016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 15), coo_matrix_call_result_367015, 'asformat')
    # Calling asformat(args, kwargs) (line 354)
    asformat_call_result_367019 = invoke(stypy.reporting.localization.Localization(__file__, 354, 15), asformat_367016, *[format_367017], **kwargs_367018)
    
    # Assigning a type to the variable 'stypy_return_type' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'stypy_return_type', asformat_call_result_367019)
    # SSA join for if statement (line 312)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'kron(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'kron' in the type store
    # Getting the type of 'stypy_return_type' (line 275)
    stypy_return_type_367020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_367020)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'kron'
    return stypy_return_type_367020

# Assigning a type to the variable 'kron' (line 275)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 0), 'kron', kron)

@norecursion
def kronsum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 357)
    None_367021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 25), 'None')
    defaults = [None_367021]
    # Create a new context for function 'kronsum'
    module_type_store = module_type_store.open_function_context('kronsum', 357, 0, False)
    
    # Passed parameters checking function
    kronsum.stypy_localization = localization
    kronsum.stypy_type_of_self = None
    kronsum.stypy_type_store = module_type_store
    kronsum.stypy_function_name = 'kronsum'
    kronsum.stypy_param_names_list = ['A', 'B', 'format']
    kronsum.stypy_varargs_param_name = None
    kronsum.stypy_kwargs_param_name = None
    kronsum.stypy_call_defaults = defaults
    kronsum.stypy_call_varargs = varargs
    kronsum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'kronsum', ['A', 'B', 'format'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'kronsum', localization, ['A', 'B', 'format'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'kronsum(...)' code ##################

    str_367022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, (-1)), 'str', 'kronecker sum of sparse matrices A and B\n\n    Kronecker sum of two sparse matrices is a sum of two Kronecker\n    products kron(I_n,A) + kron(B,I_m) where A has shape (m,m)\n    and B has shape (n,n) and I_m and I_n are identity matrices\n    of shape (m,m) and (n,n) respectively.\n\n    Parameters\n    ----------\n    A\n        square matrix\n    B\n        square matrix\n    format : str\n        format of the result (e.g. "csr")\n\n    Returns\n    -------\n    kronecker sum in a sparse matrix format\n\n    Examples\n    --------\n\n\n    ')
    
    # Assigning a Call to a Name (line 383):
    
    # Assigning a Call to a Name (line 383):
    
    # Call to coo_matrix(...): (line 383)
    # Processing the call arguments (line 383)
    # Getting the type of 'A' (line 383)
    A_367024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), 'A', False)
    # Processing the call keyword arguments (line 383)
    kwargs_367025 = {}
    # Getting the type of 'coo_matrix' (line 383)
    coo_matrix_367023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 383)
    coo_matrix_call_result_367026 = invoke(stypy.reporting.localization.Localization(__file__, 383, 8), coo_matrix_367023, *[A_367024], **kwargs_367025)
    
    # Assigning a type to the variable 'A' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'A', coo_matrix_call_result_367026)
    
    # Assigning a Call to a Name (line 384):
    
    # Assigning a Call to a Name (line 384):
    
    # Call to coo_matrix(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 'B' (line 384)
    B_367028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), 'B', False)
    # Processing the call keyword arguments (line 384)
    kwargs_367029 = {}
    # Getting the type of 'coo_matrix' (line 384)
    coo_matrix_367027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 384)
    coo_matrix_call_result_367030 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), coo_matrix_367027, *[B_367028], **kwargs_367029)
    
    # Assigning a type to the variable 'B' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'B', coo_matrix_call_result_367030)
    
    
    
    # Obtaining the type of the subscript
    int_367031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 15), 'int')
    # Getting the type of 'A' (line 386)
    A_367032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 7), 'A')
    # Obtaining the member 'shape' of a type (line 386)
    shape_367033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 7), A_367032, 'shape')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___367034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 7), shape_367033, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_367035 = invoke(stypy.reporting.localization.Localization(__file__, 386, 7), getitem___367034, int_367031)
    
    
    # Obtaining the type of the subscript
    int_367036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 29), 'int')
    # Getting the type of 'A' (line 386)
    A_367037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 21), 'A')
    # Obtaining the member 'shape' of a type (line 386)
    shape_367038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 21), A_367037, 'shape')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___367039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 21), shape_367038, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_367040 = invoke(stypy.reporting.localization.Localization(__file__, 386, 21), getitem___367039, int_367036)
    
    # Applying the binary operator '!=' (line 386)
    result_ne_367041 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 7), '!=', subscript_call_result_367035, subscript_call_result_367040)
    
    # Testing the type of an if condition (line 386)
    if_condition_367042 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 386, 4), result_ne_367041)
    # Assigning a type to the variable 'if_condition_367042' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'if_condition_367042', if_condition_367042)
    # SSA begins for if statement (line 386)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 387)
    # Processing the call arguments (line 387)
    str_367044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 25), 'str', 'A is not square')
    # Processing the call keyword arguments (line 387)
    kwargs_367045 = {}
    # Getting the type of 'ValueError' (line 387)
    ValueError_367043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 387)
    ValueError_call_result_367046 = invoke(stypy.reporting.localization.Localization(__file__, 387, 14), ValueError_367043, *[str_367044], **kwargs_367045)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 387, 8), ValueError_call_result_367046, 'raise parameter', BaseException)
    # SSA join for if statement (line 386)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_367047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 15), 'int')
    # Getting the type of 'B' (line 389)
    B_367048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 7), 'B')
    # Obtaining the member 'shape' of a type (line 389)
    shape_367049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 7), B_367048, 'shape')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___367050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 7), shape_367049, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_367051 = invoke(stypy.reporting.localization.Localization(__file__, 389, 7), getitem___367050, int_367047)
    
    
    # Obtaining the type of the subscript
    int_367052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 29), 'int')
    # Getting the type of 'B' (line 389)
    B_367053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 21), 'B')
    # Obtaining the member 'shape' of a type (line 389)
    shape_367054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 21), B_367053, 'shape')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___367055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 21), shape_367054, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_367056 = invoke(stypy.reporting.localization.Localization(__file__, 389, 21), getitem___367055, int_367052)
    
    # Applying the binary operator '!=' (line 389)
    result_ne_367057 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 7), '!=', subscript_call_result_367051, subscript_call_result_367056)
    
    # Testing the type of an if condition (line 389)
    if_condition_367058 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 4), result_ne_367057)
    # Assigning a type to the variable 'if_condition_367058' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'if_condition_367058', if_condition_367058)
    # SSA begins for if statement (line 389)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 390)
    # Processing the call arguments (line 390)
    str_367060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 25), 'str', 'B is not square')
    # Processing the call keyword arguments (line 390)
    kwargs_367061 = {}
    # Getting the type of 'ValueError' (line 390)
    ValueError_367059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 390)
    ValueError_call_result_367062 = invoke(stypy.reporting.localization.Localization(__file__, 390, 14), ValueError_367059, *[str_367060], **kwargs_367061)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 390, 8), ValueError_call_result_367062, 'raise parameter', BaseException)
    # SSA join for if statement (line 389)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 392):
    
    # Assigning a Call to a Name (line 392):
    
    # Call to upcast(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'A' (line 392)
    A_367064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'A', False)
    # Obtaining the member 'dtype' of a type (line 392)
    dtype_367065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 19), A_367064, 'dtype')
    # Getting the type of 'B' (line 392)
    B_367066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 28), 'B', False)
    # Obtaining the member 'dtype' of a type (line 392)
    dtype_367067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 28), B_367066, 'dtype')
    # Processing the call keyword arguments (line 392)
    kwargs_367068 = {}
    # Getting the type of 'upcast' (line 392)
    upcast_367063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'upcast', False)
    # Calling upcast(args, kwargs) (line 392)
    upcast_call_result_367069 = invoke(stypy.reporting.localization.Localization(__file__, 392, 12), upcast_367063, *[dtype_367065, dtype_367067], **kwargs_367068)
    
    # Assigning a type to the variable 'dtype' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'dtype', upcast_call_result_367069)
    
    # Assigning a Call to a Name (line 394):
    
    # Assigning a Call to a Name (line 394):
    
    # Call to kron(...): (line 394)
    # Processing the call arguments (line 394)
    
    # Call to eye(...): (line 394)
    # Processing the call arguments (line 394)
    
    # Obtaining the type of the subscript
    int_367072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 25), 'int')
    # Getting the type of 'B' (line 394)
    B_367073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 17), 'B', False)
    # Obtaining the member 'shape' of a type (line 394)
    shape_367074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 17), B_367073, 'shape')
    # Obtaining the member '__getitem__' of a type (line 394)
    getitem___367075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 17), shape_367074, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 394)
    subscript_call_result_367076 = invoke(stypy.reporting.localization.Localization(__file__, 394, 17), getitem___367075, int_367072)
    
    # Processing the call keyword arguments (line 394)
    # Getting the type of 'dtype' (line 394)
    dtype_367077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 34), 'dtype', False)
    keyword_367078 = dtype_367077
    kwargs_367079 = {'dtype': keyword_367078}
    # Getting the type of 'eye' (line 394)
    eye_367071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 13), 'eye', False)
    # Calling eye(args, kwargs) (line 394)
    eye_call_result_367080 = invoke(stypy.reporting.localization.Localization(__file__, 394, 13), eye_367071, *[subscript_call_result_367076], **kwargs_367079)
    
    # Getting the type of 'A' (line 394)
    A_367081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 42), 'A', False)
    # Processing the call keyword arguments (line 394)
    # Getting the type of 'format' (line 394)
    format_367082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 52), 'format', False)
    keyword_367083 = format_367082
    kwargs_367084 = {'format': keyword_367083}
    # Getting the type of 'kron' (line 394)
    kron_367070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'kron', False)
    # Calling kron(args, kwargs) (line 394)
    kron_call_result_367085 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), kron_367070, *[eye_call_result_367080, A_367081], **kwargs_367084)
    
    # Assigning a type to the variable 'L' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'L', kron_call_result_367085)
    
    # Assigning a Call to a Name (line 395):
    
    # Assigning a Call to a Name (line 395):
    
    # Call to kron(...): (line 395)
    # Processing the call arguments (line 395)
    # Getting the type of 'B' (line 395)
    B_367087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 13), 'B', False)
    
    # Call to eye(...): (line 395)
    # Processing the call arguments (line 395)
    
    # Obtaining the type of the subscript
    int_367089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 28), 'int')
    # Getting the type of 'A' (line 395)
    A_367090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 20), 'A', False)
    # Obtaining the member 'shape' of a type (line 395)
    shape_367091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 20), A_367090, 'shape')
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___367092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 20), shape_367091, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 395)
    subscript_call_result_367093 = invoke(stypy.reporting.localization.Localization(__file__, 395, 20), getitem___367092, int_367089)
    
    # Processing the call keyword arguments (line 395)
    # Getting the type of 'dtype' (line 395)
    dtype_367094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 37), 'dtype', False)
    keyword_367095 = dtype_367094
    kwargs_367096 = {'dtype': keyword_367095}
    # Getting the type of 'eye' (line 395)
    eye_367088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'eye', False)
    # Calling eye(args, kwargs) (line 395)
    eye_call_result_367097 = invoke(stypy.reporting.localization.Localization(__file__, 395, 16), eye_367088, *[subscript_call_result_367093], **kwargs_367096)
    
    # Processing the call keyword arguments (line 395)
    # Getting the type of 'format' (line 395)
    format_367098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 52), 'format', False)
    keyword_367099 = format_367098
    kwargs_367100 = {'format': keyword_367099}
    # Getting the type of 'kron' (line 395)
    kron_367086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'kron', False)
    # Calling kron(args, kwargs) (line 395)
    kron_call_result_367101 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), kron_367086, *[B_367087, eye_call_result_367097], **kwargs_367100)
    
    # Assigning a type to the variable 'R' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'R', kron_call_result_367101)
    
    # Call to asformat(...): (line 397)
    # Processing the call arguments (line 397)
    # Getting the type of 'format' (line 397)
    format_367106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 26), 'format', False)
    # Processing the call keyword arguments (line 397)
    kwargs_367107 = {}
    # Getting the type of 'L' (line 397)
    L_367102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'L', False)
    # Getting the type of 'R' (line 397)
    R_367103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 14), 'R', False)
    # Applying the binary operator '+' (line 397)
    result_add_367104 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 12), '+', L_367102, R_367103)
    
    # Obtaining the member 'asformat' of a type (line 397)
    asformat_367105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 12), result_add_367104, 'asformat')
    # Calling asformat(args, kwargs) (line 397)
    asformat_call_result_367108 = invoke(stypy.reporting.localization.Localization(__file__, 397, 12), asformat_367105, *[format_367106], **kwargs_367107)
    
    # Assigning a type to the variable 'stypy_return_type' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'stypy_return_type', asformat_call_result_367108)
    
    # ################# End of 'kronsum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'kronsum' in the type store
    # Getting the type of 'stypy_return_type' (line 357)
    stypy_return_type_367109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_367109)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'kronsum'
    return stypy_return_type_367109

# Assigning a type to the variable 'kronsum' (line 357)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 0), 'kronsum', kronsum)

@norecursion
def _compressed_sparse_stack(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_compressed_sparse_stack'
    module_type_store = module_type_store.open_function_context('_compressed_sparse_stack', 400, 0, False)
    
    # Passed parameters checking function
    _compressed_sparse_stack.stypy_localization = localization
    _compressed_sparse_stack.stypy_type_of_self = None
    _compressed_sparse_stack.stypy_type_store = module_type_store
    _compressed_sparse_stack.stypy_function_name = '_compressed_sparse_stack'
    _compressed_sparse_stack.stypy_param_names_list = ['blocks', 'axis']
    _compressed_sparse_stack.stypy_varargs_param_name = None
    _compressed_sparse_stack.stypy_kwargs_param_name = None
    _compressed_sparse_stack.stypy_call_defaults = defaults
    _compressed_sparse_stack.stypy_call_varargs = varargs
    _compressed_sparse_stack.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_compressed_sparse_stack', ['blocks', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_compressed_sparse_stack', localization, ['blocks', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_compressed_sparse_stack(...)' code ##################

    str_367110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, (-1)), 'str', '\n    Stacking fast path for CSR/CSC matrices\n    (i) vstack for CSR, (ii) hstack for CSC.\n    ')
    
    # Assigning a IfExp to a Name (line 405):
    
    # Assigning a IfExp to a Name (line 405):
    
    
    # Getting the type of 'axis' (line 405)
    axis_367111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 22), 'axis')
    int_367112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 30), 'int')
    # Applying the binary operator '==' (line 405)
    result_eq_367113 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 22), '==', axis_367111, int_367112)
    
    # Testing the type of an if expression (line 405)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 405, 17), result_eq_367113)
    # SSA begins for if expression (line 405)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    int_367114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 17), 'int')
    # SSA branch for the else part of an if expression (line 405)
    module_type_store.open_ssa_branch('if expression else')
    int_367115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 37), 'int')
    # SSA join for if expression (line 405)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_367116 = union_type.UnionType.add(int_367114, int_367115)
    
    # Assigning a type to the variable 'other_axis' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'other_axis', if_exp_367116)
    
    # Assigning a Call to a Name (line 406):
    
    # Assigning a Call to a Name (line 406):
    
    # Call to concatenate(...): (line 406)
    # Processing the call arguments (line 406)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'blocks' (line 406)
    blocks_367121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 43), 'blocks', False)
    comprehension_367122 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 27), blocks_367121)
    # Assigning a type to the variable 'b' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 27), 'b', comprehension_367122)
    # Getting the type of 'b' (line 406)
    b_367119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 27), 'b', False)
    # Obtaining the member 'data' of a type (line 406)
    data_367120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 27), b_367119, 'data')
    list_367123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 27), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 27), list_367123, data_367120)
    # Processing the call keyword arguments (line 406)
    kwargs_367124 = {}
    # Getting the type of 'np' (line 406)
    np_367117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 406)
    concatenate_367118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 11), np_367117, 'concatenate')
    # Calling concatenate(args, kwargs) (line 406)
    concatenate_call_result_367125 = invoke(stypy.reporting.localization.Localization(__file__, 406, 11), concatenate_367118, *[list_367123], **kwargs_367124)
    
    # Assigning a type to the variable 'data' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'data', concatenate_call_result_367125)
    
    # Assigning a Subscript to a Name (line 407):
    
    # Assigning a Subscript to a Name (line 407):
    
    # Obtaining the type of the subscript
    # Getting the type of 'other_axis' (line 407)
    other_axis_367126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 35), 'other_axis')
    
    # Obtaining the type of the subscript
    int_367127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 26), 'int')
    # Getting the type of 'blocks' (line 407)
    blocks_367128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 19), 'blocks')
    # Obtaining the member '__getitem__' of a type (line 407)
    getitem___367129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 19), blocks_367128, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 407)
    subscript_call_result_367130 = invoke(stypy.reporting.localization.Localization(__file__, 407, 19), getitem___367129, int_367127)
    
    # Obtaining the member 'shape' of a type (line 407)
    shape_367131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 19), subscript_call_result_367130, 'shape')
    # Obtaining the member '__getitem__' of a type (line 407)
    getitem___367132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 19), shape_367131, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 407)
    subscript_call_result_367133 = invoke(stypy.reporting.localization.Localization(__file__, 407, 19), getitem___367132, other_axis_367126)
    
    # Assigning a type to the variable 'constant_dim' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'constant_dim', subscript_call_result_367133)
    
    # Assigning a Call to a Name (line 408):
    
    # Assigning a Call to a Name (line 408):
    
    # Call to get_index_dtype(...): (line 408)
    # Processing the call keyword arguments (line 408)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'blocks' (line 408)
    blocks_367137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 58), 'blocks', False)
    comprehension_367138 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 40), blocks_367137)
    # Assigning a type to the variable 'b' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 40), 'b', comprehension_367138)
    # Getting the type of 'b' (line 408)
    b_367135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 40), 'b', False)
    # Obtaining the member 'indptr' of a type (line 408)
    indptr_367136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 40), b_367135, 'indptr')
    list_367139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 40), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 40), list_367139, indptr_367136)
    keyword_367140 = list_367139
    
    # Call to max(...): (line 409)
    # Processing the call arguments (line 409)
    # Getting the type of 'data' (line 409)
    data_367142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 43), 'data', False)
    # Obtaining the member 'size' of a type (line 409)
    size_367143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 43), data_367142, 'size')
    # Getting the type of 'constant_dim' (line 409)
    constant_dim_367144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 54), 'constant_dim', False)
    # Processing the call keyword arguments (line 409)
    kwargs_367145 = {}
    # Getting the type of 'max' (line 409)
    max_367141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 39), 'max', False)
    # Calling max(args, kwargs) (line 409)
    max_call_result_367146 = invoke(stypy.reporting.localization.Localization(__file__, 409, 39), max_367141, *[size_367143, constant_dim_367144], **kwargs_367145)
    
    keyword_367147 = max_call_result_367146
    kwargs_367148 = {'arrays': keyword_367140, 'maxval': keyword_367147}
    # Getting the type of 'get_index_dtype' (line 408)
    get_index_dtype_367134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 16), 'get_index_dtype', False)
    # Calling get_index_dtype(args, kwargs) (line 408)
    get_index_dtype_call_result_367149 = invoke(stypy.reporting.localization.Localization(__file__, 408, 16), get_index_dtype_367134, *[], **kwargs_367148)
    
    # Assigning a type to the variable 'idx_dtype' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'idx_dtype', get_index_dtype_call_result_367149)
    
    # Assigning a Call to a Name (line 410):
    
    # Assigning a Call to a Name (line 410):
    
    # Call to empty(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 'data' (line 410)
    data_367152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 23), 'data', False)
    # Obtaining the member 'size' of a type (line 410)
    size_367153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 23), data_367152, 'size')
    # Processing the call keyword arguments (line 410)
    # Getting the type of 'idx_dtype' (line 410)
    idx_dtype_367154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 40), 'idx_dtype', False)
    keyword_367155 = idx_dtype_367154
    kwargs_367156 = {'dtype': keyword_367155}
    # Getting the type of 'np' (line 410)
    np_367150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 14), 'np', False)
    # Obtaining the member 'empty' of a type (line 410)
    empty_367151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 14), np_367150, 'empty')
    # Calling empty(args, kwargs) (line 410)
    empty_call_result_367157 = invoke(stypy.reporting.localization.Localization(__file__, 410, 14), empty_367151, *[size_367153], **kwargs_367156)
    
    # Assigning a type to the variable 'indices' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'indices', empty_call_result_367157)
    
    # Assigning a Call to a Name (line 411):
    
    # Assigning a Call to a Name (line 411):
    
    # Call to empty(...): (line 411)
    # Processing the call arguments (line 411)
    
    # Call to sum(...): (line 411)
    # Processing the call arguments (line 411)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 411, 26, True)
    # Calculating comprehension expression
    # Getting the type of 'blocks' (line 411)
    blocks_367166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 49), 'blocks', False)
    comprehension_367167 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 26), blocks_367166)
    # Assigning a type to the variable 'b' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 26), 'b', comprehension_367167)
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 411)
    axis_367161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 34), 'axis', False)
    # Getting the type of 'b' (line 411)
    b_367162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 26), 'b', False)
    # Obtaining the member 'shape' of a type (line 411)
    shape_367163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 26), b_367162, 'shape')
    # Obtaining the member '__getitem__' of a type (line 411)
    getitem___367164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 26), shape_367163, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 411)
    subscript_call_result_367165 = invoke(stypy.reporting.localization.Localization(__file__, 411, 26), getitem___367164, axis_367161)
    
    list_367168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 26), list_367168, subscript_call_result_367165)
    # Processing the call keyword arguments (line 411)
    kwargs_367169 = {}
    # Getting the type of 'sum' (line 411)
    sum_367160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 22), 'sum', False)
    # Calling sum(args, kwargs) (line 411)
    sum_call_result_367170 = invoke(stypy.reporting.localization.Localization(__file__, 411, 22), sum_367160, *[list_367168], **kwargs_367169)
    
    int_367171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 59), 'int')
    # Applying the binary operator '+' (line 411)
    result_add_367172 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 22), '+', sum_call_result_367170, int_367171)
    
    # Processing the call keyword arguments (line 411)
    # Getting the type of 'idx_dtype' (line 411)
    idx_dtype_367173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 68), 'idx_dtype', False)
    keyword_367174 = idx_dtype_367173
    kwargs_367175 = {'dtype': keyword_367174}
    # Getting the type of 'np' (line 411)
    np_367158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 13), 'np', False)
    # Obtaining the member 'empty' of a type (line 411)
    empty_367159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 13), np_367158, 'empty')
    # Calling empty(args, kwargs) (line 411)
    empty_call_result_367176 = invoke(stypy.reporting.localization.Localization(__file__, 411, 13), empty_367159, *[result_add_367172], **kwargs_367175)
    
    # Assigning a type to the variable 'indptr' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'indptr', empty_call_result_367176)
    
    # Assigning a Call to a Name (line 412):
    
    # Assigning a Call to a Name (line 412):
    
    # Call to idx_dtype(...): (line 412)
    # Processing the call arguments (line 412)
    int_367178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 28), 'int')
    # Processing the call keyword arguments (line 412)
    kwargs_367179 = {}
    # Getting the type of 'idx_dtype' (line 412)
    idx_dtype_367177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 18), 'idx_dtype', False)
    # Calling idx_dtype(args, kwargs) (line 412)
    idx_dtype_call_result_367180 = invoke(stypy.reporting.localization.Localization(__file__, 412, 18), idx_dtype_367177, *[int_367178], **kwargs_367179)
    
    # Assigning a type to the variable 'last_indptr' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'last_indptr', idx_dtype_call_result_367180)
    
    # Assigning a Num to a Name (line 413):
    
    # Assigning a Num to a Name (line 413):
    int_367181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 14), 'int')
    # Assigning a type to the variable 'sum_dim' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'sum_dim', int_367181)
    
    # Assigning a Num to a Name (line 414):
    
    # Assigning a Num to a Name (line 414):
    int_367182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 18), 'int')
    # Assigning a type to the variable 'sum_indices' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'sum_indices', int_367182)
    
    # Getting the type of 'blocks' (line 415)
    blocks_367183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 13), 'blocks')
    # Testing the type of a for loop iterable (line 415)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 415, 4), blocks_367183)
    # Getting the type of the for loop variable (line 415)
    for_loop_var_367184 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 415, 4), blocks_367183)
    # Assigning a type to the variable 'b' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'b', for_loop_var_367184)
    # SSA begins for a for statement (line 415)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'other_axis' (line 416)
    other_axis_367185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 19), 'other_axis')
    # Getting the type of 'b' (line 416)
    b_367186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 11), 'b')
    # Obtaining the member 'shape' of a type (line 416)
    shape_367187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 11), b_367186, 'shape')
    # Obtaining the member '__getitem__' of a type (line 416)
    getitem___367188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 11), shape_367187, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 416)
    subscript_call_result_367189 = invoke(stypy.reporting.localization.Localization(__file__, 416, 11), getitem___367188, other_axis_367185)
    
    # Getting the type of 'constant_dim' (line 416)
    constant_dim_367190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 34), 'constant_dim')
    # Applying the binary operator '!=' (line 416)
    result_ne_367191 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 11), '!=', subscript_call_result_367189, constant_dim_367190)
    
    # Testing the type of an if condition (line 416)
    if_condition_367192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 416, 8), result_ne_367191)
    # Assigning a type to the variable 'if_condition_367192' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'if_condition_367192', if_condition_367192)
    # SSA begins for if statement (line 416)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 417)
    # Processing the call arguments (line 417)
    str_367194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 29), 'str', 'incompatible dimensions for axis %d')
    # Getting the type of 'other_axis' (line 417)
    other_axis_367195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 69), 'other_axis', False)
    # Applying the binary operator '%' (line 417)
    result_mod_367196 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 29), '%', str_367194, other_axis_367195)
    
    # Processing the call keyword arguments (line 417)
    kwargs_367197 = {}
    # Getting the type of 'ValueError' (line 417)
    ValueError_367193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 417)
    ValueError_call_result_367198 = invoke(stypy.reporting.localization.Localization(__file__, 417, 18), ValueError_367193, *[result_mod_367196], **kwargs_367197)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 417, 12), ValueError_call_result_367198, 'raise parameter', BaseException)
    # SSA join for if statement (line 416)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Subscript (line 418):
    
    # Assigning a Attribute to a Subscript (line 418):
    # Getting the type of 'b' (line 418)
    b_367199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 58), 'b')
    # Obtaining the member 'indices' of a type (line 418)
    indices_367200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 58), b_367199, 'indices')
    # Getting the type of 'indices' (line 418)
    indices_367201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'indices')
    # Getting the type of 'sum_indices' (line 418)
    sum_indices_367202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'sum_indices')
    # Getting the type of 'sum_indices' (line 418)
    sum_indices_367203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 28), 'sum_indices')
    # Getting the type of 'b' (line 418)
    b_367204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 40), 'b')
    # Obtaining the member 'indices' of a type (line 418)
    indices_367205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 40), b_367204, 'indices')
    # Obtaining the member 'size' of a type (line 418)
    size_367206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 40), indices_367205, 'size')
    # Applying the binary operator '+' (line 418)
    result_add_367207 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 28), '+', sum_indices_367203, size_367206)
    
    slice_367208 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 418, 8), sum_indices_367202, result_add_367207, None)
    # Storing an element on a container (line 418)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 8), indices_367201, (slice_367208, indices_367200))
    
    # Getting the type of 'sum_indices' (line 419)
    sum_indices_367209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'sum_indices')
    # Getting the type of 'b' (line 419)
    b_367210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 23), 'b')
    # Obtaining the member 'indices' of a type (line 419)
    indices_367211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 23), b_367210, 'indices')
    # Obtaining the member 'size' of a type (line 419)
    size_367212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 23), indices_367211, 'size')
    # Applying the binary operator '+=' (line 419)
    result_iadd_367213 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 8), '+=', sum_indices_367209, size_367212)
    # Assigning a type to the variable 'sum_indices' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'sum_indices', result_iadd_367213)
    
    
    # Assigning a Call to a Name (line 420):
    
    # Assigning a Call to a Name (line 420):
    
    # Call to slice(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'sum_dim' (line 420)
    sum_dim_367215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 21), 'sum_dim', False)
    # Getting the type of 'sum_dim' (line 420)
    sum_dim_367216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 30), 'sum_dim', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 420)
    axis_367217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 48), 'axis', False)
    # Getting the type of 'b' (line 420)
    b_367218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 40), 'b', False)
    # Obtaining the member 'shape' of a type (line 420)
    shape_367219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 40), b_367218, 'shape')
    # Obtaining the member '__getitem__' of a type (line 420)
    getitem___367220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 40), shape_367219, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 420)
    subscript_call_result_367221 = invoke(stypy.reporting.localization.Localization(__file__, 420, 40), getitem___367220, axis_367217)
    
    # Applying the binary operator '+' (line 420)
    result_add_367222 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 30), '+', sum_dim_367216, subscript_call_result_367221)
    
    # Processing the call keyword arguments (line 420)
    kwargs_367223 = {}
    # Getting the type of 'slice' (line 420)
    slice_367214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 15), 'slice', False)
    # Calling slice(args, kwargs) (line 420)
    slice_call_result_367224 = invoke(stypy.reporting.localization.Localization(__file__, 420, 15), slice_367214, *[sum_dim_367215, result_add_367222], **kwargs_367223)
    
    # Assigning a type to the variable 'idxs' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'idxs', slice_call_result_367224)
    
    # Assigning a Subscript to a Subscript (line 421):
    
    # Assigning a Subscript to a Subscript (line 421):
    
    # Obtaining the type of the subscript
    int_367225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 33), 'int')
    slice_367226 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 421, 23), None, int_367225, None)
    # Getting the type of 'b' (line 421)
    b_367227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 23), 'b')
    # Obtaining the member 'indptr' of a type (line 421)
    indptr_367228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 23), b_367227, 'indptr')
    # Obtaining the member '__getitem__' of a type (line 421)
    getitem___367229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 23), indptr_367228, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 421)
    subscript_call_result_367230 = invoke(stypy.reporting.localization.Localization(__file__, 421, 23), getitem___367229, slice_367226)
    
    # Getting the type of 'indptr' (line 421)
    indptr_367231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'indptr')
    # Getting the type of 'idxs' (line 421)
    idxs_367232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 15), 'idxs')
    # Storing an element on a container (line 421)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 8), indptr_367231, (idxs_367232, subscript_call_result_367230))
    
    # Getting the type of 'indptr' (line 422)
    indptr_367233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'indptr')
    
    # Obtaining the type of the subscript
    # Getting the type of 'idxs' (line 422)
    idxs_367234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'idxs')
    # Getting the type of 'indptr' (line 422)
    indptr_367235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'indptr')
    # Obtaining the member '__getitem__' of a type (line 422)
    getitem___367236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 8), indptr_367235, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 422)
    subscript_call_result_367237 = invoke(stypy.reporting.localization.Localization(__file__, 422, 8), getitem___367236, idxs_367234)
    
    # Getting the type of 'last_indptr' (line 422)
    last_indptr_367238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 24), 'last_indptr')
    # Applying the binary operator '+=' (line 422)
    result_iadd_367239 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 8), '+=', subscript_call_result_367237, last_indptr_367238)
    # Getting the type of 'indptr' (line 422)
    indptr_367240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'indptr')
    # Getting the type of 'idxs' (line 422)
    idxs_367241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'idxs')
    # Storing an element on a container (line 422)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 8), indptr_367240, (idxs_367241, result_iadd_367239))
    
    
    # Getting the type of 'sum_dim' (line 423)
    sum_dim_367242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'sum_dim')
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 423)
    axis_367243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 27), 'axis')
    # Getting the type of 'b' (line 423)
    b_367244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 19), 'b')
    # Obtaining the member 'shape' of a type (line 423)
    shape_367245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 19), b_367244, 'shape')
    # Obtaining the member '__getitem__' of a type (line 423)
    getitem___367246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 19), shape_367245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 423)
    subscript_call_result_367247 = invoke(stypy.reporting.localization.Localization(__file__, 423, 19), getitem___367246, axis_367243)
    
    # Applying the binary operator '+=' (line 423)
    result_iadd_367248 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 8), '+=', sum_dim_367242, subscript_call_result_367247)
    # Assigning a type to the variable 'sum_dim' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'sum_dim', result_iadd_367248)
    
    
    # Getting the type of 'last_indptr' (line 424)
    last_indptr_367249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'last_indptr')
    
    # Obtaining the type of the subscript
    int_367250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 32), 'int')
    # Getting the type of 'b' (line 424)
    b_367251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 23), 'b')
    # Obtaining the member 'indptr' of a type (line 424)
    indptr_367252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 23), b_367251, 'indptr')
    # Obtaining the member '__getitem__' of a type (line 424)
    getitem___367253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 23), indptr_367252, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 424)
    subscript_call_result_367254 = invoke(stypy.reporting.localization.Localization(__file__, 424, 23), getitem___367253, int_367250)
    
    # Applying the binary operator '+=' (line 424)
    result_iadd_367255 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 8), '+=', last_indptr_367249, subscript_call_result_367254)
    # Assigning a type to the variable 'last_indptr' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'last_indptr', result_iadd_367255)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 425):
    
    # Assigning a Name to a Subscript (line 425):
    # Getting the type of 'last_indptr' (line 425)
    last_indptr_367256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 17), 'last_indptr')
    # Getting the type of 'indptr' (line 425)
    indptr_367257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'indptr')
    int_367258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 11), 'int')
    # Storing an element on a container (line 425)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 4), indptr_367257, (int_367258, last_indptr_367256))
    
    
    # Getting the type of 'axis' (line 426)
    axis_367259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 7), 'axis')
    int_367260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 15), 'int')
    # Applying the binary operator '==' (line 426)
    result_eq_367261 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 7), '==', axis_367259, int_367260)
    
    # Testing the type of an if condition (line 426)
    if_condition_367262 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 426, 4), result_eq_367261)
    # Assigning a type to the variable 'if_condition_367262' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'if_condition_367262', if_condition_367262)
    # SSA begins for if statement (line 426)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to csr_matrix(...): (line 427)
    # Processing the call arguments (line 427)
    
    # Obtaining an instance of the builtin type 'tuple' (line 427)
    tuple_367264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 427)
    # Adding element type (line 427)
    # Getting the type of 'data' (line 427)
    data_367265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 27), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 27), tuple_367264, data_367265)
    # Adding element type (line 427)
    # Getting the type of 'indices' (line 427)
    indices_367266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 33), 'indices', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 27), tuple_367264, indices_367266)
    # Adding element type (line 427)
    # Getting the type of 'indptr' (line 427)
    indptr_367267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 42), 'indptr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 27), tuple_367264, indptr_367267)
    
    # Processing the call keyword arguments (line 427)
    
    # Obtaining an instance of the builtin type 'tuple' (line 428)
    tuple_367268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 428)
    # Adding element type (line 428)
    # Getting the type of 'sum_dim' (line 428)
    sum_dim_367269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 33), 'sum_dim', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 33), tuple_367268, sum_dim_367269)
    # Adding element type (line 428)
    # Getting the type of 'constant_dim' (line 428)
    constant_dim_367270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 42), 'constant_dim', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 33), tuple_367268, constant_dim_367270)
    
    keyword_367271 = tuple_367268
    kwargs_367272 = {'shape': keyword_367271}
    # Getting the type of 'csr_matrix' (line 427)
    csr_matrix_367263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 15), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 427)
    csr_matrix_call_result_367273 = invoke(stypy.reporting.localization.Localization(__file__, 427, 15), csr_matrix_367263, *[tuple_367264], **kwargs_367272)
    
    # Assigning a type to the variable 'stypy_return_type' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'stypy_return_type', csr_matrix_call_result_367273)
    # SSA branch for the else part of an if statement (line 426)
    module_type_store.open_ssa_branch('else')
    
    # Call to csc_matrix(...): (line 430)
    # Processing the call arguments (line 430)
    
    # Obtaining an instance of the builtin type 'tuple' (line 430)
    tuple_367275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 430)
    # Adding element type (line 430)
    # Getting the type of 'data' (line 430)
    data_367276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 27), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 27), tuple_367275, data_367276)
    # Adding element type (line 430)
    # Getting the type of 'indices' (line 430)
    indices_367277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 33), 'indices', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 27), tuple_367275, indices_367277)
    # Adding element type (line 430)
    # Getting the type of 'indptr' (line 430)
    indptr_367278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 42), 'indptr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 27), tuple_367275, indptr_367278)
    
    # Processing the call keyword arguments (line 430)
    
    # Obtaining an instance of the builtin type 'tuple' (line 431)
    tuple_367279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 431)
    # Adding element type (line 431)
    # Getting the type of 'constant_dim' (line 431)
    constant_dim_367280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 33), 'constant_dim', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 33), tuple_367279, constant_dim_367280)
    # Adding element type (line 431)
    # Getting the type of 'sum_dim' (line 431)
    sum_dim_367281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 47), 'sum_dim', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 33), tuple_367279, sum_dim_367281)
    
    keyword_367282 = tuple_367279
    kwargs_367283 = {'shape': keyword_367282}
    # Getting the type of 'csc_matrix' (line 430)
    csc_matrix_367274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 15), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 430)
    csc_matrix_call_result_367284 = invoke(stypy.reporting.localization.Localization(__file__, 430, 15), csc_matrix_367274, *[tuple_367275], **kwargs_367283)
    
    # Assigning a type to the variable 'stypy_return_type' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'stypy_return_type', csc_matrix_call_result_367284)
    # SSA join for if statement (line 426)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_compressed_sparse_stack(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_compressed_sparse_stack' in the type store
    # Getting the type of 'stypy_return_type' (line 400)
    stypy_return_type_367285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_367285)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_compressed_sparse_stack'
    return stypy_return_type_367285

# Assigning a type to the variable '_compressed_sparse_stack' (line 400)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 0), '_compressed_sparse_stack', _compressed_sparse_stack)

@norecursion
def hstack(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 434)
    None_367286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 26), 'None')
    # Getting the type of 'None' (line 434)
    None_367287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 38), 'None')
    defaults = [None_367286, None_367287]
    # Create a new context for function 'hstack'
    module_type_store = module_type_store.open_function_context('hstack', 434, 0, False)
    
    # Passed parameters checking function
    hstack.stypy_localization = localization
    hstack.stypy_type_of_self = None
    hstack.stypy_type_store = module_type_store
    hstack.stypy_function_name = 'hstack'
    hstack.stypy_param_names_list = ['blocks', 'format', 'dtype']
    hstack.stypy_varargs_param_name = None
    hstack.stypy_kwargs_param_name = None
    hstack.stypy_call_defaults = defaults
    hstack.stypy_call_varargs = varargs
    hstack.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hstack', ['blocks', 'format', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hstack', localization, ['blocks', 'format', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hstack(...)' code ##################

    str_367288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, (-1)), 'str', '\n    Stack sparse matrices horizontally (column wise)\n\n    Parameters\n    ----------\n    blocks\n        sequence of sparse matrices with compatible shapes\n    format : str\n        sparse format of the result (e.g. "csr")\n        by default an appropriate sparse matrix format is returned.\n        This choice is subject to change.\n    dtype : dtype, optional\n        The data-type of the output matrix.  If not given, the dtype is\n        determined from that of `blocks`.\n\n    See Also\n    --------\n    vstack : stack sparse matrices vertically (row wise)\n\n    Examples\n    --------\n    >>> from scipy.sparse import coo_matrix, hstack\n    >>> A = coo_matrix([[1, 2], [3, 4]])\n    >>> B = coo_matrix([[5], [6]])\n    >>> hstack([A,B]).toarray()\n    array([[1, 2, 5],\n           [3, 4, 6]])\n\n    ')
    
    # Call to bmat(...): (line 464)
    # Processing the call arguments (line 464)
    
    # Obtaining an instance of the builtin type 'list' (line 464)
    list_367290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 464)
    # Adding element type (line 464)
    # Getting the type of 'blocks' (line 464)
    blocks_367291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 17), 'blocks', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 16), list_367290, blocks_367291)
    
    # Processing the call keyword arguments (line 464)
    # Getting the type of 'format' (line 464)
    format_367292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 33), 'format', False)
    keyword_367293 = format_367292
    # Getting the type of 'dtype' (line 464)
    dtype_367294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 47), 'dtype', False)
    keyword_367295 = dtype_367294
    kwargs_367296 = {'dtype': keyword_367295, 'format': keyword_367293}
    # Getting the type of 'bmat' (line 464)
    bmat_367289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 11), 'bmat', False)
    # Calling bmat(args, kwargs) (line 464)
    bmat_call_result_367297 = invoke(stypy.reporting.localization.Localization(__file__, 464, 11), bmat_367289, *[list_367290], **kwargs_367296)
    
    # Assigning a type to the variable 'stypy_return_type' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'stypy_return_type', bmat_call_result_367297)
    
    # ################# End of 'hstack(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hstack' in the type store
    # Getting the type of 'stypy_return_type' (line 434)
    stypy_return_type_367298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_367298)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hstack'
    return stypy_return_type_367298

# Assigning a type to the variable 'hstack' (line 434)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 0), 'hstack', hstack)

@norecursion
def vstack(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 467)
    None_367299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 26), 'None')
    # Getting the type of 'None' (line 467)
    None_367300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 38), 'None')
    defaults = [None_367299, None_367300]
    # Create a new context for function 'vstack'
    module_type_store = module_type_store.open_function_context('vstack', 467, 0, False)
    
    # Passed parameters checking function
    vstack.stypy_localization = localization
    vstack.stypy_type_of_self = None
    vstack.stypy_type_store = module_type_store
    vstack.stypy_function_name = 'vstack'
    vstack.stypy_param_names_list = ['blocks', 'format', 'dtype']
    vstack.stypy_varargs_param_name = None
    vstack.stypy_kwargs_param_name = None
    vstack.stypy_call_defaults = defaults
    vstack.stypy_call_varargs = varargs
    vstack.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'vstack', ['blocks', 'format', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'vstack', localization, ['blocks', 'format', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'vstack(...)' code ##################

    str_367301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, (-1)), 'str', '\n    Stack sparse matrices vertically (row wise)\n\n    Parameters\n    ----------\n    blocks\n        sequence of sparse matrices with compatible shapes\n    format : str, optional\n        sparse format of the result (e.g. "csr")\n        by default an appropriate sparse matrix format is returned.\n        This choice is subject to change.\n    dtype : dtype, optional\n        The data-type of the output matrix.  If not given, the dtype is\n        determined from that of `blocks`.\n\n    See Also\n    --------\n    hstack : stack sparse matrices horizontally (column wise)\n\n    Examples\n    --------\n    >>> from scipy.sparse import coo_matrix, vstack\n    >>> A = coo_matrix([[1, 2], [3, 4]])\n    >>> B = coo_matrix([[5, 6]])\n    >>> vstack([A, B]).toarray()\n    array([[1, 2],\n           [3, 4],\n           [5, 6]])\n\n    ')
    
    # Call to bmat(...): (line 498)
    # Processing the call arguments (line 498)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'blocks' (line 498)
    blocks_367305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 30), 'blocks', False)
    comprehension_367306 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 17), blocks_367305)
    # Assigning a type to the variable 'b' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 17), 'b', comprehension_367306)
    
    # Obtaining an instance of the builtin type 'list' (line 498)
    list_367303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 498)
    # Adding element type (line 498)
    # Getting the type of 'b' (line 498)
    b_367304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 18), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 17), list_367303, b_367304)
    
    list_367307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 17), list_367307, list_367303)
    # Processing the call keyword arguments (line 498)
    # Getting the type of 'format' (line 498)
    format_367308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 46), 'format', False)
    keyword_367309 = format_367308
    # Getting the type of 'dtype' (line 498)
    dtype_367310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 60), 'dtype', False)
    keyword_367311 = dtype_367310
    kwargs_367312 = {'dtype': keyword_367311, 'format': keyword_367309}
    # Getting the type of 'bmat' (line 498)
    bmat_367302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 11), 'bmat', False)
    # Calling bmat(args, kwargs) (line 498)
    bmat_call_result_367313 = invoke(stypy.reporting.localization.Localization(__file__, 498, 11), bmat_367302, *[list_367307], **kwargs_367312)
    
    # Assigning a type to the variable 'stypy_return_type' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'stypy_return_type', bmat_call_result_367313)
    
    # ################# End of 'vstack(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'vstack' in the type store
    # Getting the type of 'stypy_return_type' (line 467)
    stypy_return_type_367314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_367314)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'vstack'
    return stypy_return_type_367314

# Assigning a type to the variable 'vstack' (line 467)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 0), 'vstack', vstack)

@norecursion
def bmat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 501)
    None_367315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 24), 'None')
    # Getting the type of 'None' (line 501)
    None_367316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 36), 'None')
    defaults = [None_367315, None_367316]
    # Create a new context for function 'bmat'
    module_type_store = module_type_store.open_function_context('bmat', 501, 0, False)
    
    # Passed parameters checking function
    bmat.stypy_localization = localization
    bmat.stypy_type_of_self = None
    bmat.stypy_type_store = module_type_store
    bmat.stypy_function_name = 'bmat'
    bmat.stypy_param_names_list = ['blocks', 'format', 'dtype']
    bmat.stypy_varargs_param_name = None
    bmat.stypy_kwargs_param_name = None
    bmat.stypy_call_defaults = defaults
    bmat.stypy_call_varargs = varargs
    bmat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bmat', ['blocks', 'format', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bmat', localization, ['blocks', 'format', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bmat(...)' code ##################

    str_367317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, (-1)), 'str', '\n    Build a sparse matrix from sparse sub-blocks\n\n    Parameters\n    ----------\n    blocks : array_like\n        Grid of sparse matrices with compatible shapes.\n        An entry of None implies an all-zero matrix.\n    format : {\'bsr\', \'coo\', \'csc\', \'csr\', \'dia\', \'dok\', \'lil\'}, optional\n        The sparse format of the result (e.g. "csr").  By default an\n        appropriate sparse matrix format is returned.\n        This choice is subject to change.\n    dtype : dtype, optional\n        The data-type of the output matrix.  If not given, the dtype is\n        determined from that of `blocks`.\n\n    Returns\n    -------\n    bmat : sparse matrix\n\n    See Also\n    --------\n    block_diag, diags\n\n    Examples\n    --------\n    >>> from scipy.sparse import coo_matrix, bmat\n    >>> A = coo_matrix([[1, 2], [3, 4]])\n    >>> B = coo_matrix([[5], [6]])\n    >>> C = coo_matrix([[7]])\n    >>> bmat([[A, B], [None, C]]).toarray()\n    array([[1, 2, 5],\n           [3, 4, 6],\n           [0, 0, 7]])\n\n    >>> bmat([[A, None], [None, C]]).toarray()\n    array([[1, 2, 0],\n           [3, 4, 0],\n           [0, 0, 7]])\n\n    ')
    
    # Assigning a Call to a Name (line 544):
    
    # Assigning a Call to a Name (line 544):
    
    # Call to asarray(...): (line 544)
    # Processing the call arguments (line 544)
    # Getting the type of 'blocks' (line 544)
    blocks_367320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 24), 'blocks', False)
    # Processing the call keyword arguments (line 544)
    str_367321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 38), 'str', 'object')
    keyword_367322 = str_367321
    kwargs_367323 = {'dtype': keyword_367322}
    # Getting the type of 'np' (line 544)
    np_367318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 13), 'np', False)
    # Obtaining the member 'asarray' of a type (line 544)
    asarray_367319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 13), np_367318, 'asarray')
    # Calling asarray(args, kwargs) (line 544)
    asarray_call_result_367324 = invoke(stypy.reporting.localization.Localization(__file__, 544, 13), asarray_367319, *[blocks_367320], **kwargs_367323)
    
    # Assigning a type to the variable 'blocks' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'blocks', asarray_call_result_367324)
    
    
    # Getting the type of 'blocks' (line 546)
    blocks_367325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 7), 'blocks')
    # Obtaining the member 'ndim' of a type (line 546)
    ndim_367326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 7), blocks_367325, 'ndim')
    int_367327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 22), 'int')
    # Applying the binary operator '!=' (line 546)
    result_ne_367328 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 7), '!=', ndim_367326, int_367327)
    
    # Testing the type of an if condition (line 546)
    if_condition_367329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 546, 4), result_ne_367328)
    # Assigning a type to the variable 'if_condition_367329' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'if_condition_367329', if_condition_367329)
    # SSA begins for if statement (line 546)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 547)
    # Processing the call arguments (line 547)
    str_367331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 25), 'str', 'blocks must be 2-D')
    # Processing the call keyword arguments (line 547)
    kwargs_367332 = {}
    # Getting the type of 'ValueError' (line 547)
    ValueError_367330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 547)
    ValueError_call_result_367333 = invoke(stypy.reporting.localization.Localization(__file__, 547, 14), ValueError_367330, *[str_367331], **kwargs_367332)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 547, 8), ValueError_call_result_367333, 'raise parameter', BaseException)
    # SSA join for if statement (line 546)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 549):
    
    # Assigning a Subscript to a Name (line 549):
    
    # Obtaining the type of the subscript
    int_367334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 4), 'int')
    # Getting the type of 'blocks' (line 549)
    blocks_367335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 10), 'blocks')
    # Obtaining the member 'shape' of a type (line 549)
    shape_367336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 10), blocks_367335, 'shape')
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___367337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 4), shape_367336, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_367338 = invoke(stypy.reporting.localization.Localization(__file__, 549, 4), getitem___367337, int_367334)
    
    # Assigning a type to the variable 'tuple_var_assignment_366305' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_366305', subscript_call_result_367338)
    
    # Assigning a Subscript to a Name (line 549):
    
    # Obtaining the type of the subscript
    int_367339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 4), 'int')
    # Getting the type of 'blocks' (line 549)
    blocks_367340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 10), 'blocks')
    # Obtaining the member 'shape' of a type (line 549)
    shape_367341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 10), blocks_367340, 'shape')
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___367342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 4), shape_367341, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_367343 = invoke(stypy.reporting.localization.Localization(__file__, 549, 4), getitem___367342, int_367339)
    
    # Assigning a type to the variable 'tuple_var_assignment_366306' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_366306', subscript_call_result_367343)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'tuple_var_assignment_366305' (line 549)
    tuple_var_assignment_366305_367344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_366305')
    # Assigning a type to the variable 'M' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'M', tuple_var_assignment_366305_367344)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'tuple_var_assignment_366306' (line 549)
    tuple_var_assignment_366306_367345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_366306')
    # Assigning a type to the variable 'N' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 6), 'N', tuple_var_assignment_366306_367345)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'N' (line 552)
    N_367346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'N')
    int_367347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 13), 'int')
    # Applying the binary operator '==' (line 552)
    result_eq_367348 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 8), '==', N_367346, int_367347)
    
    
    # Getting the type of 'format' (line 552)
    format_367349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 19), 'format')
    
    # Obtaining an instance of the builtin type 'tuple' (line 552)
    tuple_367350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 552)
    # Adding element type (line 552)
    # Getting the type of 'None' (line 552)
    None_367351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 30), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 30), tuple_367350, None_367351)
    # Adding element type (line 552)
    str_367352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 36), 'str', 'csr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 30), tuple_367350, str_367352)
    
    # Applying the binary operator 'in' (line 552)
    result_contains_367353 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 19), 'in', format_367349, tuple_367350)
    
    # Applying the binary operator 'and' (line 552)
    result_and_keyword_367354 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 8), 'and', result_eq_367348, result_contains_367353)
    
    # Call to all(...): (line 552)
    # Processing the call arguments (line 552)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 552, 51, True)
    # Calculating comprehension expression
    # Getting the type of 'blocks' (line 553)
    blocks_367361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 60), 'blocks', False)
    # Obtaining the member 'flat' of a type (line 553)
    flat_367362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 60), blocks_367361, 'flat')
    comprehension_367363 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 51), flat_367362)
    # Assigning a type to the variable 'b' (line 552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 51), 'b', comprehension_367363)
    
    # Call to isinstance(...): (line 552)
    # Processing the call arguments (line 552)
    # Getting the type of 'b' (line 552)
    b_367357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 62), 'b', False)
    # Getting the type of 'csr_matrix' (line 552)
    csr_matrix_367358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 65), 'csr_matrix', False)
    # Processing the call keyword arguments (line 552)
    kwargs_367359 = {}
    # Getting the type of 'isinstance' (line 552)
    isinstance_367356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 51), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 552)
    isinstance_call_result_367360 = invoke(stypy.reporting.localization.Localization(__file__, 552, 51), isinstance_367356, *[b_367357, csr_matrix_367358], **kwargs_367359)
    
    list_367364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 51), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 51), list_367364, isinstance_call_result_367360)
    # Processing the call keyword arguments (line 552)
    kwargs_367365 = {}
    # Getting the type of 'all' (line 552)
    all_367355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 47), 'all', False)
    # Calling all(args, kwargs) (line 552)
    all_call_result_367366 = invoke(stypy.reporting.localization.Localization(__file__, 552, 47), all_367355, *[list_367364], **kwargs_367365)
    
    # Applying the binary operator 'and' (line 552)
    result_and_keyword_367367 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 8), 'and', result_and_keyword_367354, all_call_result_367366)
    
    # Testing the type of an if condition (line 552)
    if_condition_367368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 552, 4), result_and_keyword_367367)
    # Assigning a type to the variable 'if_condition_367368' (line 552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'if_condition_367368', if_condition_367368)
    # SSA begins for if statement (line 552)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 554):
    
    # Assigning a Call to a Name (line 554):
    
    # Call to _compressed_sparse_stack(...): (line 554)
    # Processing the call arguments (line 554)
    
    # Obtaining the type of the subscript
    slice_367370 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 554, 37), None, None, None)
    int_367371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 46), 'int')
    # Getting the type of 'blocks' (line 554)
    blocks_367372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 37), 'blocks', False)
    # Obtaining the member '__getitem__' of a type (line 554)
    getitem___367373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 37), blocks_367372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 554)
    subscript_call_result_367374 = invoke(stypy.reporting.localization.Localization(__file__, 554, 37), getitem___367373, (slice_367370, int_367371))
    
    int_367375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 50), 'int')
    # Processing the call keyword arguments (line 554)
    kwargs_367376 = {}
    # Getting the type of '_compressed_sparse_stack' (line 554)
    _compressed_sparse_stack_367369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), '_compressed_sparse_stack', False)
    # Calling _compressed_sparse_stack(args, kwargs) (line 554)
    _compressed_sparse_stack_call_result_367377 = invoke(stypy.reporting.localization.Localization(__file__, 554, 12), _compressed_sparse_stack_367369, *[subscript_call_result_367374, int_367375], **kwargs_367376)
    
    # Assigning a type to the variable 'A' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'A', _compressed_sparse_stack_call_result_367377)
    
    # Type idiom detected: calculating its left and rigth part (line 555)
    # Getting the type of 'dtype' (line 555)
    dtype_367378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'dtype')
    # Getting the type of 'None' (line 555)
    None_367379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 24), 'None')
    
    (may_be_367380, more_types_in_union_367381) = may_not_be_none(dtype_367378, None_367379)

    if may_be_367380:

        if more_types_in_union_367381:
            # Runtime conditional SSA (line 555)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 556):
        
        # Assigning a Call to a Name (line 556):
        
        # Call to astype(...): (line 556)
        # Processing the call arguments (line 556)
        # Getting the type of 'dtype' (line 556)
        dtype_367384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 25), 'dtype', False)
        # Processing the call keyword arguments (line 556)
        kwargs_367385 = {}
        # Getting the type of 'A' (line 556)
        A_367382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), 'A', False)
        # Obtaining the member 'astype' of a type (line 556)
        astype_367383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 16), A_367382, 'astype')
        # Calling astype(args, kwargs) (line 556)
        astype_call_result_367386 = invoke(stypy.reporting.localization.Localization(__file__, 556, 16), astype_367383, *[dtype_367384], **kwargs_367385)
        
        # Assigning a type to the variable 'A' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'A', astype_call_result_367386)

        if more_types_in_union_367381:
            # SSA join for if statement (line 555)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'A' (line 557)
    A_367387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 15), 'A')
    # Assigning a type to the variable 'stypy_return_type' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'stypy_return_type', A_367387)
    # SSA branch for the else part of an if statement (line 552)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'M' (line 558)
    M_367388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 10), 'M')
    int_367389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 15), 'int')
    # Applying the binary operator '==' (line 558)
    result_eq_367390 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 10), '==', M_367388, int_367389)
    
    
    # Getting the type of 'format' (line 558)
    format_367391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 21), 'format')
    
    # Obtaining an instance of the builtin type 'tuple' (line 558)
    tuple_367392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 558)
    # Adding element type (line 558)
    # Getting the type of 'None' (line 558)
    None_367393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 32), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 32), tuple_367392, None_367393)
    # Adding element type (line 558)
    str_367394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 38), 'str', 'csc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 32), tuple_367392, str_367394)
    
    # Applying the binary operator 'in' (line 558)
    result_contains_367395 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 21), 'in', format_367391, tuple_367392)
    
    # Applying the binary operator 'and' (line 558)
    result_and_keyword_367396 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 10), 'and', result_eq_367390, result_contains_367395)
    
    # Call to all(...): (line 559)
    # Processing the call arguments (line 559)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 559, 18, True)
    # Calculating comprehension expression
    # Getting the type of 'blocks' (line 559)
    blocks_367403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 53), 'blocks', False)
    # Obtaining the member 'flat' of a type (line 559)
    flat_367404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 53), blocks_367403, 'flat')
    comprehension_367405 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 18), flat_367404)
    # Assigning a type to the variable 'b' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 18), 'b', comprehension_367405)
    
    # Call to isinstance(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'b' (line 559)
    b_367399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 29), 'b', False)
    # Getting the type of 'csc_matrix' (line 559)
    csc_matrix_367400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 32), 'csc_matrix', False)
    # Processing the call keyword arguments (line 559)
    kwargs_367401 = {}
    # Getting the type of 'isinstance' (line 559)
    isinstance_367398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 18), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 559)
    isinstance_call_result_367402 = invoke(stypy.reporting.localization.Localization(__file__, 559, 18), isinstance_367398, *[b_367399, csc_matrix_367400], **kwargs_367401)
    
    list_367406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 18), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 18), list_367406, isinstance_call_result_367402)
    # Processing the call keyword arguments (line 559)
    kwargs_367407 = {}
    # Getting the type of 'all' (line 559)
    all_367397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 14), 'all', False)
    # Calling all(args, kwargs) (line 559)
    all_call_result_367408 = invoke(stypy.reporting.localization.Localization(__file__, 559, 14), all_367397, *[list_367406], **kwargs_367407)
    
    # Applying the binary operator 'and' (line 558)
    result_and_keyword_367409 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 10), 'and', result_and_keyword_367396, all_call_result_367408)
    
    # Testing the type of an if condition (line 558)
    if_condition_367410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 558, 9), result_and_keyword_367409)
    # Assigning a type to the variable 'if_condition_367410' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 9), 'if_condition_367410', if_condition_367410)
    # SSA begins for if statement (line 558)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 560):
    
    # Assigning a Call to a Name (line 560):
    
    # Call to _compressed_sparse_stack(...): (line 560)
    # Processing the call arguments (line 560)
    
    # Obtaining the type of the subscript
    int_367412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 44), 'int')
    slice_367413 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 560, 37), None, None, None)
    # Getting the type of 'blocks' (line 560)
    blocks_367414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 37), 'blocks', False)
    # Obtaining the member '__getitem__' of a type (line 560)
    getitem___367415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 37), blocks_367414, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 560)
    subscript_call_result_367416 = invoke(stypy.reporting.localization.Localization(__file__, 560, 37), getitem___367415, (int_367412, slice_367413))
    
    int_367417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 50), 'int')
    # Processing the call keyword arguments (line 560)
    kwargs_367418 = {}
    # Getting the type of '_compressed_sparse_stack' (line 560)
    _compressed_sparse_stack_367411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 12), '_compressed_sparse_stack', False)
    # Calling _compressed_sparse_stack(args, kwargs) (line 560)
    _compressed_sparse_stack_call_result_367419 = invoke(stypy.reporting.localization.Localization(__file__, 560, 12), _compressed_sparse_stack_367411, *[subscript_call_result_367416, int_367417], **kwargs_367418)
    
    # Assigning a type to the variable 'A' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'A', _compressed_sparse_stack_call_result_367419)
    
    # Type idiom detected: calculating its left and rigth part (line 561)
    # Getting the type of 'dtype' (line 561)
    dtype_367420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'dtype')
    # Getting the type of 'None' (line 561)
    None_367421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 24), 'None')
    
    (may_be_367422, more_types_in_union_367423) = may_not_be_none(dtype_367420, None_367421)

    if may_be_367422:

        if more_types_in_union_367423:
            # Runtime conditional SSA (line 561)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 562):
        
        # Assigning a Call to a Name (line 562):
        
        # Call to astype(...): (line 562)
        # Processing the call arguments (line 562)
        # Getting the type of 'dtype' (line 562)
        dtype_367426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 25), 'dtype', False)
        # Processing the call keyword arguments (line 562)
        kwargs_367427 = {}
        # Getting the type of 'A' (line 562)
        A_367424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 16), 'A', False)
        # Obtaining the member 'astype' of a type (line 562)
        astype_367425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 16), A_367424, 'astype')
        # Calling astype(args, kwargs) (line 562)
        astype_call_result_367428 = invoke(stypy.reporting.localization.Localization(__file__, 562, 16), astype_367425, *[dtype_367426], **kwargs_367427)
        
        # Assigning a type to the variable 'A' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'A', astype_call_result_367428)

        if more_types_in_union_367423:
            # SSA join for if statement (line 561)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'A' (line 563)
    A_367429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 15), 'A')
    # Assigning a type to the variable 'stypy_return_type' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'stypy_return_type', A_367429)
    # SSA join for if statement (line 558)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 552)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 565):
    
    # Assigning a Call to a Name (line 565):
    
    # Call to zeros(...): (line 565)
    # Processing the call arguments (line 565)
    # Getting the type of 'blocks' (line 565)
    blocks_367432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 26), 'blocks', False)
    # Obtaining the member 'shape' of a type (line 565)
    shape_367433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 26), blocks_367432, 'shape')
    # Processing the call keyword arguments (line 565)
    # Getting the type of 'bool' (line 565)
    bool_367434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 46), 'bool', False)
    keyword_367435 = bool_367434
    kwargs_367436 = {'dtype': keyword_367435}
    # Getting the type of 'np' (line 565)
    np_367430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 565)
    zeros_367431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 17), np_367430, 'zeros')
    # Calling zeros(args, kwargs) (line 565)
    zeros_call_result_367437 = invoke(stypy.reporting.localization.Localization(__file__, 565, 17), zeros_367431, *[shape_367433], **kwargs_367436)
    
    # Assigning a type to the variable 'block_mask' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'block_mask', zeros_call_result_367437)
    
    # Assigning a Call to a Name (line 566):
    
    # Assigning a Call to a Name (line 566):
    
    # Call to zeros(...): (line 566)
    # Processing the call arguments (line 566)
    # Getting the type of 'M' (line 566)
    M_367440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 28), 'M', False)
    # Processing the call keyword arguments (line 566)
    # Getting the type of 'np' (line 566)
    np_367441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 37), 'np', False)
    # Obtaining the member 'int64' of a type (line 566)
    int64_367442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 37), np_367441, 'int64')
    keyword_367443 = int64_367442
    kwargs_367444 = {'dtype': keyword_367443}
    # Getting the type of 'np' (line 566)
    np_367438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 19), 'np', False)
    # Obtaining the member 'zeros' of a type (line 566)
    zeros_367439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 19), np_367438, 'zeros')
    # Calling zeros(args, kwargs) (line 566)
    zeros_call_result_367445 = invoke(stypy.reporting.localization.Localization(__file__, 566, 19), zeros_367439, *[M_367440], **kwargs_367444)
    
    # Assigning a type to the variable 'brow_lengths' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'brow_lengths', zeros_call_result_367445)
    
    # Assigning a Call to a Name (line 567):
    
    # Assigning a Call to a Name (line 567):
    
    # Call to zeros(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'N' (line 567)
    N_367448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 28), 'N', False)
    # Processing the call keyword arguments (line 567)
    # Getting the type of 'np' (line 567)
    np_367449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 37), 'np', False)
    # Obtaining the member 'int64' of a type (line 567)
    int64_367450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 37), np_367449, 'int64')
    keyword_367451 = int64_367450
    kwargs_367452 = {'dtype': keyword_367451}
    # Getting the type of 'np' (line 567)
    np_367446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 19), 'np', False)
    # Obtaining the member 'zeros' of a type (line 567)
    zeros_367447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 19), np_367446, 'zeros')
    # Calling zeros(args, kwargs) (line 567)
    zeros_call_result_367453 = invoke(stypy.reporting.localization.Localization(__file__, 567, 19), zeros_367447, *[N_367448], **kwargs_367452)
    
    # Assigning a type to the variable 'bcol_lengths' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'bcol_lengths', zeros_call_result_367453)
    
    
    # Call to range(...): (line 570)
    # Processing the call arguments (line 570)
    # Getting the type of 'M' (line 570)
    M_367455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 19), 'M', False)
    # Processing the call keyword arguments (line 570)
    kwargs_367456 = {}
    # Getting the type of 'range' (line 570)
    range_367454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 13), 'range', False)
    # Calling range(args, kwargs) (line 570)
    range_call_result_367457 = invoke(stypy.reporting.localization.Localization(__file__, 570, 13), range_367454, *[M_367455], **kwargs_367456)
    
    # Testing the type of a for loop iterable (line 570)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 570, 4), range_call_result_367457)
    # Getting the type of the for loop variable (line 570)
    for_loop_var_367458 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 570, 4), range_call_result_367457)
    # Assigning a type to the variable 'i' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'i', for_loop_var_367458)
    # SSA begins for a for statement (line 570)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 571)
    # Processing the call arguments (line 571)
    # Getting the type of 'N' (line 571)
    N_367460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 23), 'N', False)
    # Processing the call keyword arguments (line 571)
    kwargs_367461 = {}
    # Getting the type of 'range' (line 571)
    range_367459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 17), 'range', False)
    # Calling range(args, kwargs) (line 571)
    range_call_result_367462 = invoke(stypy.reporting.localization.Localization(__file__, 571, 17), range_367459, *[N_367460], **kwargs_367461)
    
    # Testing the type of a for loop iterable (line 571)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 571, 8), range_call_result_367462)
    # Getting the type of the for loop variable (line 571)
    for_loop_var_367463 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 571, 8), range_call_result_367462)
    # Assigning a type to the variable 'j' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'j', for_loop_var_367463)
    # SSA begins for a for statement (line 571)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 572)
    tuple_367464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 572)
    # Adding element type (line 572)
    # Getting the type of 'i' (line 572)
    i_367465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 22), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 22), tuple_367464, i_367465)
    # Adding element type (line 572)
    # Getting the type of 'j' (line 572)
    j_367466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 24), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 22), tuple_367464, j_367466)
    
    # Getting the type of 'blocks' (line 572)
    blocks_367467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 15), 'blocks')
    # Obtaining the member '__getitem__' of a type (line 572)
    getitem___367468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 15), blocks_367467, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 572)
    subscript_call_result_367469 = invoke(stypy.reporting.localization.Localization(__file__, 572, 15), getitem___367468, tuple_367464)
    
    # Getting the type of 'None' (line 572)
    None_367470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 34), 'None')
    # Applying the binary operator 'isnot' (line 572)
    result_is_not_367471 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 15), 'isnot', subscript_call_result_367469, None_367470)
    
    # Testing the type of an if condition (line 572)
    if_condition_367472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 572, 12), result_is_not_367471)
    # Assigning a type to the variable 'if_condition_367472' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'if_condition_367472', if_condition_367472)
    # SSA begins for if statement (line 572)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 573):
    
    # Assigning a Call to a Name (line 573):
    
    # Call to coo_matrix(...): (line 573)
    # Processing the call arguments (line 573)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 573)
    tuple_367474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 573)
    # Adding element type (line 573)
    # Getting the type of 'i' (line 573)
    i_367475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 38), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 38), tuple_367474, i_367475)
    # Adding element type (line 573)
    # Getting the type of 'j' (line 573)
    j_367476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 40), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 38), tuple_367474, j_367476)
    
    # Getting the type of 'blocks' (line 573)
    blocks_367477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 31), 'blocks', False)
    # Obtaining the member '__getitem__' of a type (line 573)
    getitem___367478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 31), blocks_367477, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 573)
    subscript_call_result_367479 = invoke(stypy.reporting.localization.Localization(__file__, 573, 31), getitem___367478, tuple_367474)
    
    # Processing the call keyword arguments (line 573)
    kwargs_367480 = {}
    # Getting the type of 'coo_matrix' (line 573)
    coo_matrix_367473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 20), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 573)
    coo_matrix_call_result_367481 = invoke(stypy.reporting.localization.Localization(__file__, 573, 20), coo_matrix_367473, *[subscript_call_result_367479], **kwargs_367480)
    
    # Assigning a type to the variable 'A' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 16), 'A', coo_matrix_call_result_367481)
    
    # Assigning a Name to a Subscript (line 574):
    
    # Assigning a Name to a Subscript (line 574):
    # Getting the type of 'A' (line 574)
    A_367482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 30), 'A')
    # Getting the type of 'blocks' (line 574)
    blocks_367483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 16), 'blocks')
    
    # Obtaining an instance of the builtin type 'tuple' (line 574)
    tuple_367484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 574)
    # Adding element type (line 574)
    # Getting the type of 'i' (line 574)
    i_367485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 23), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 23), tuple_367484, i_367485)
    # Adding element type (line 574)
    # Getting the type of 'j' (line 574)
    j_367486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 25), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 23), tuple_367484, j_367486)
    
    # Storing an element on a container (line 574)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 16), blocks_367483, (tuple_367484, A_367482))
    
    # Assigning a Name to a Subscript (line 575):
    
    # Assigning a Name to a Subscript (line 575):
    # Getting the type of 'True' (line 575)
    True_367487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 34), 'True')
    # Getting the type of 'block_mask' (line 575)
    block_mask_367488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'block_mask')
    
    # Obtaining an instance of the builtin type 'tuple' (line 575)
    tuple_367489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 575)
    # Adding element type (line 575)
    # Getting the type of 'i' (line 575)
    i_367490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 27), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 27), tuple_367489, i_367490)
    # Adding element type (line 575)
    # Getting the type of 'j' (line 575)
    j_367491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 29), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 27), tuple_367489, j_367491)
    
    # Storing an element on a container (line 575)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 16), block_mask_367488, (tuple_367489, True_367487))
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 577)
    i_367492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 32), 'i')
    # Getting the type of 'brow_lengths' (line 577)
    brow_lengths_367493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 19), 'brow_lengths')
    # Obtaining the member '__getitem__' of a type (line 577)
    getitem___367494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 19), brow_lengths_367493, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 577)
    subscript_call_result_367495 = invoke(stypy.reporting.localization.Localization(__file__, 577, 19), getitem___367494, i_367492)
    
    int_367496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 38), 'int')
    # Applying the binary operator '==' (line 577)
    result_eq_367497 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 19), '==', subscript_call_result_367495, int_367496)
    
    # Testing the type of an if condition (line 577)
    if_condition_367498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 577, 16), result_eq_367497)
    # Assigning a type to the variable 'if_condition_367498' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'if_condition_367498', if_condition_367498)
    # SSA begins for if statement (line 577)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 578):
    
    # Assigning a Subscript to a Subscript (line 578):
    
    # Obtaining the type of the subscript
    int_367499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 46), 'int')
    # Getting the type of 'A' (line 578)
    A_367500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 38), 'A')
    # Obtaining the member 'shape' of a type (line 578)
    shape_367501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 38), A_367500, 'shape')
    # Obtaining the member '__getitem__' of a type (line 578)
    getitem___367502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 38), shape_367501, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 578)
    subscript_call_result_367503 = invoke(stypy.reporting.localization.Localization(__file__, 578, 38), getitem___367502, int_367499)
    
    # Getting the type of 'brow_lengths' (line 578)
    brow_lengths_367504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 20), 'brow_lengths')
    # Getting the type of 'i' (line 578)
    i_367505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 33), 'i')
    # Storing an element on a container (line 578)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 20), brow_lengths_367504, (i_367505, subscript_call_result_367503))
    # SSA branch for the else part of an if statement (line 577)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 579)
    i_367506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 34), 'i')
    # Getting the type of 'brow_lengths' (line 579)
    brow_lengths_367507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 21), 'brow_lengths')
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___367508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 21), brow_lengths_367507, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_367509 = invoke(stypy.reporting.localization.Localization(__file__, 579, 21), getitem___367508, i_367506)
    
    
    # Obtaining the type of the subscript
    int_367510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 48), 'int')
    # Getting the type of 'A' (line 579)
    A_367511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 40), 'A')
    # Obtaining the member 'shape' of a type (line 579)
    shape_367512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 40), A_367511, 'shape')
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___367513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 40), shape_367512, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_367514 = invoke(stypy.reporting.localization.Localization(__file__, 579, 40), getitem___367513, int_367510)
    
    # Applying the binary operator '!=' (line 579)
    result_ne_367515 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 21), '!=', subscript_call_result_367509, subscript_call_result_367514)
    
    # Testing the type of an if condition (line 579)
    if_condition_367516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 579, 21), result_ne_367515)
    # Assigning a type to the variable 'if_condition_367516' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 21), 'if_condition_367516', if_condition_367516)
    # SSA begins for if statement (line 579)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 580):
    
    # Assigning a Call to a Name (line 580):
    
    # Call to format(...): (line 580)
    # Processing the call keyword arguments (line 580)
    # Getting the type of 'i' (line 582)
    i_367519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 54), 'i', False)
    keyword_367520 = i_367519
    # Getting the type of 'j' (line 582)
    j_367521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 59), 'j', False)
    keyword_367522 = j_367521
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 583)
    i_367523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 69), 'i', False)
    # Getting the type of 'brow_lengths' (line 583)
    brow_lengths_367524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 56), 'brow_lengths', False)
    # Obtaining the member '__getitem__' of a type (line 583)
    getitem___367525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 56), brow_lengths_367524, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 583)
    subscript_call_result_367526 = invoke(stypy.reporting.localization.Localization(__file__, 583, 56), getitem___367525, i_367523)
    
    keyword_367527 = subscript_call_result_367526
    
    # Obtaining the type of the subscript
    int_367528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 64), 'int')
    # Getting the type of 'A' (line 584)
    A_367529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 56), 'A', False)
    # Obtaining the member 'shape' of a type (line 584)
    shape_367530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 56), A_367529, 'shape')
    # Obtaining the member '__getitem__' of a type (line 584)
    getitem___367531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 56), shape_367530, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 584)
    subscript_call_result_367532 = invoke(stypy.reporting.localization.Localization(__file__, 584, 56), getitem___367531, int_367528)
    
    keyword_367533 = subscript_call_result_367532
    kwargs_367534 = {'i': keyword_367520, 'got': keyword_367533, 'j': keyword_367522, 'exp': keyword_367527}
    str_367517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 27), 'str', 'blocks[{i},:] has incompatible row dimensions. Got blocks[{i},{j}].shape[0] == {got}, expected {exp}.')
    # Obtaining the member 'format' of a type (line 580)
    format_367518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 27), str_367517, 'format')
    # Calling format(args, kwargs) (line 580)
    format_call_result_367535 = invoke(stypy.reporting.localization.Localization(__file__, 580, 27), format_367518, *[], **kwargs_367534)
    
    # Assigning a type to the variable 'msg' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 20), 'msg', format_call_result_367535)
    
    # Call to ValueError(...): (line 585)
    # Processing the call arguments (line 585)
    # Getting the type of 'msg' (line 585)
    msg_367537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 37), 'msg', False)
    # Processing the call keyword arguments (line 585)
    kwargs_367538 = {}
    # Getting the type of 'ValueError' (line 585)
    ValueError_367536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 26), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 585)
    ValueError_call_result_367539 = invoke(stypy.reporting.localization.Localization(__file__, 585, 26), ValueError_367536, *[msg_367537], **kwargs_367538)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 585, 20), ValueError_call_result_367539, 'raise parameter', BaseException)
    # SSA join for if statement (line 579)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 577)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 587)
    j_367540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 32), 'j')
    # Getting the type of 'bcol_lengths' (line 587)
    bcol_lengths_367541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 19), 'bcol_lengths')
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___367542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 19), bcol_lengths_367541, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_367543 = invoke(stypy.reporting.localization.Localization(__file__, 587, 19), getitem___367542, j_367540)
    
    int_367544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 38), 'int')
    # Applying the binary operator '==' (line 587)
    result_eq_367545 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 19), '==', subscript_call_result_367543, int_367544)
    
    # Testing the type of an if condition (line 587)
    if_condition_367546 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 587, 16), result_eq_367545)
    # Assigning a type to the variable 'if_condition_367546' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'if_condition_367546', if_condition_367546)
    # SSA begins for if statement (line 587)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 588):
    
    # Assigning a Subscript to a Subscript (line 588):
    
    # Obtaining the type of the subscript
    int_367547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 46), 'int')
    # Getting the type of 'A' (line 588)
    A_367548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 38), 'A')
    # Obtaining the member 'shape' of a type (line 588)
    shape_367549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 38), A_367548, 'shape')
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___367550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 38), shape_367549, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_367551 = invoke(stypy.reporting.localization.Localization(__file__, 588, 38), getitem___367550, int_367547)
    
    # Getting the type of 'bcol_lengths' (line 588)
    bcol_lengths_367552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 20), 'bcol_lengths')
    # Getting the type of 'j' (line 588)
    j_367553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 33), 'j')
    # Storing an element on a container (line 588)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 20), bcol_lengths_367552, (j_367553, subscript_call_result_367551))
    # SSA branch for the else part of an if statement (line 587)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 589)
    j_367554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 34), 'j')
    # Getting the type of 'bcol_lengths' (line 589)
    bcol_lengths_367555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'bcol_lengths')
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___367556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 21), bcol_lengths_367555, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 589)
    subscript_call_result_367557 = invoke(stypy.reporting.localization.Localization(__file__, 589, 21), getitem___367556, j_367554)
    
    
    # Obtaining the type of the subscript
    int_367558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 48), 'int')
    # Getting the type of 'A' (line 589)
    A_367559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 40), 'A')
    # Obtaining the member 'shape' of a type (line 589)
    shape_367560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 40), A_367559, 'shape')
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___367561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 40), shape_367560, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 589)
    subscript_call_result_367562 = invoke(stypy.reporting.localization.Localization(__file__, 589, 40), getitem___367561, int_367558)
    
    # Applying the binary operator '!=' (line 589)
    result_ne_367563 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 21), '!=', subscript_call_result_367557, subscript_call_result_367562)
    
    # Testing the type of an if condition (line 589)
    if_condition_367564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 589, 21), result_ne_367563)
    # Assigning a type to the variable 'if_condition_367564' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'if_condition_367564', if_condition_367564)
    # SSA begins for if statement (line 589)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 590):
    
    # Assigning a Call to a Name (line 590):
    
    # Call to format(...): (line 590)
    # Processing the call keyword arguments (line 590)
    # Getting the type of 'i' (line 592)
    i_367567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 54), 'i', False)
    keyword_367568 = i_367567
    # Getting the type of 'j' (line 592)
    j_367569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 59), 'j', False)
    keyword_367570 = j_367569
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 593)
    j_367571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 69), 'j', False)
    # Getting the type of 'bcol_lengths' (line 593)
    bcol_lengths_367572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 56), 'bcol_lengths', False)
    # Obtaining the member '__getitem__' of a type (line 593)
    getitem___367573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 56), bcol_lengths_367572, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 593)
    subscript_call_result_367574 = invoke(stypy.reporting.localization.Localization(__file__, 593, 56), getitem___367573, j_367571)
    
    keyword_367575 = subscript_call_result_367574
    
    # Obtaining the type of the subscript
    int_367576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 64), 'int')
    # Getting the type of 'A' (line 594)
    A_367577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 56), 'A', False)
    # Obtaining the member 'shape' of a type (line 594)
    shape_367578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 56), A_367577, 'shape')
    # Obtaining the member '__getitem__' of a type (line 594)
    getitem___367579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 56), shape_367578, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 594)
    subscript_call_result_367580 = invoke(stypy.reporting.localization.Localization(__file__, 594, 56), getitem___367579, int_367576)
    
    keyword_367581 = subscript_call_result_367580
    kwargs_367582 = {'i': keyword_367568, 'got': keyword_367581, 'j': keyword_367570, 'exp': keyword_367575}
    str_367565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 27), 'str', 'blocks[:,{j}] has incompatible row dimensions. Got blocks[{i},{j}].shape[1] == {got}, expected {exp}.')
    # Obtaining the member 'format' of a type (line 590)
    format_367566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 27), str_367565, 'format')
    # Calling format(args, kwargs) (line 590)
    format_call_result_367583 = invoke(stypy.reporting.localization.Localization(__file__, 590, 27), format_367566, *[], **kwargs_367582)
    
    # Assigning a type to the variable 'msg' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 20), 'msg', format_call_result_367583)
    
    # Call to ValueError(...): (line 595)
    # Processing the call arguments (line 595)
    # Getting the type of 'msg' (line 595)
    msg_367585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 37), 'msg', False)
    # Processing the call keyword arguments (line 595)
    kwargs_367586 = {}
    # Getting the type of 'ValueError' (line 595)
    ValueError_367584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 26), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 595)
    ValueError_call_result_367587 = invoke(stypy.reporting.localization.Localization(__file__, 595, 26), ValueError_367584, *[msg_367585], **kwargs_367586)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 595, 20), ValueError_call_result_367587, 'raise parameter', BaseException)
    # SSA join for if statement (line 589)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 587)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 572)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 597):
    
    # Assigning a Call to a Name (line 597):
    
    # Call to sum(...): (line 597)
    # Processing the call arguments (line 597)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 597, 14, True)
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    # Getting the type of 'block_mask' (line 597)
    block_mask_367591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 44), 'block_mask', False)
    # Getting the type of 'blocks' (line 597)
    blocks_367592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 37), 'blocks', False)
    # Obtaining the member '__getitem__' of a type (line 597)
    getitem___367593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 37), blocks_367592, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 597)
    subscript_call_result_367594 = invoke(stypy.reporting.localization.Localization(__file__, 597, 37), getitem___367593, block_mask_367591)
    
    comprehension_367595 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 14), subscript_call_result_367594)
    # Assigning a type to the variable 'block' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 14), 'block', comprehension_367595)
    # Getting the type of 'block' (line 597)
    block_367589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 14), 'block', False)
    # Obtaining the member 'nnz' of a type (line 597)
    nnz_367590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 14), block_367589, 'nnz')
    list_367596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 14), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 14), list_367596, nnz_367590)
    # Processing the call keyword arguments (line 597)
    kwargs_367597 = {}
    # Getting the type of 'sum' (line 597)
    sum_367588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 10), 'sum', False)
    # Calling sum(args, kwargs) (line 597)
    sum_call_result_367598 = invoke(stypy.reporting.localization.Localization(__file__, 597, 10), sum_367588, *[list_367596], **kwargs_367597)
    
    # Assigning a type to the variable 'nnz' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'nnz', sum_call_result_367598)
    
    # Type idiom detected: calculating its left and rigth part (line 598)
    # Getting the type of 'dtype' (line 598)
    dtype_367599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 7), 'dtype')
    # Getting the type of 'None' (line 598)
    None_367600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 16), 'None')
    
    (may_be_367601, more_types_in_union_367602) = may_be_none(dtype_367599, None_367600)

    if may_be_367601:

        if more_types_in_union_367602:
            # Runtime conditional SSA (line 598)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a ListComp to a Name (line 599):
        
        # Assigning a ListComp to a Name (line 599):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining the type of the subscript
        # Getting the type of 'block_mask' (line 599)
        block_mask_367605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 50), 'block_mask')
        # Getting the type of 'blocks' (line 599)
        blocks_367606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 43), 'blocks')
        # Obtaining the member '__getitem__' of a type (line 599)
        getitem___367607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 43), blocks_367606, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 599)
        subscript_call_result_367608 = invoke(stypy.reporting.localization.Localization(__file__, 599, 43), getitem___367607, block_mask_367605)
        
        comprehension_367609 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 22), subscript_call_result_367608)
        # Assigning a type to the variable 'blk' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 22), 'blk', comprehension_367609)
        # Getting the type of 'blk' (line 599)
        blk_367603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 22), 'blk')
        # Obtaining the member 'dtype' of a type (line 599)
        dtype_367604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 22), blk_367603, 'dtype')
        list_367610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 22), list_367610, dtype_367604)
        # Assigning a type to the variable 'all_dtypes' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'all_dtypes', list_367610)
        
        # Assigning a IfExp to a Name (line 600):
        
        # Assigning a IfExp to a Name (line 600):
        
        # Getting the type of 'all_dtypes' (line 600)
        all_dtypes_367611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 39), 'all_dtypes')
        # Testing the type of an if expression (line 600)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 600, 16), all_dtypes_367611)
        # SSA begins for if expression (line 600)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to upcast(...): (line 600)
        # Getting the type of 'all_dtypes' (line 600)
        all_dtypes_367613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 24), 'all_dtypes', False)
        # Processing the call keyword arguments (line 600)
        kwargs_367614 = {}
        # Getting the type of 'upcast' (line 600)
        upcast_367612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'upcast', False)
        # Calling upcast(args, kwargs) (line 600)
        upcast_call_result_367615 = invoke(stypy.reporting.localization.Localization(__file__, 600, 16), upcast_367612, *[all_dtypes_367613], **kwargs_367614)
        
        # SSA branch for the else part of an if expression (line 600)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'None' (line 600)
        None_367616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 55), 'None')
        # SSA join for if expression (line 600)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_367617 = union_type.UnionType.add(upcast_call_result_367615, None_367616)
        
        # Assigning a type to the variable 'dtype' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'dtype', if_exp_367617)

        if more_types_in_union_367602:
            # SSA join for if statement (line 598)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 602):
    
    # Assigning a Call to a Name (line 602):
    
    # Call to append(...): (line 602)
    # Processing the call arguments (line 602)
    int_367620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 28), 'int')
    
    # Call to cumsum(...): (line 602)
    # Processing the call arguments (line 602)
    # Getting the type of 'brow_lengths' (line 602)
    brow_lengths_367623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 41), 'brow_lengths', False)
    # Processing the call keyword arguments (line 602)
    kwargs_367624 = {}
    # Getting the type of 'np' (line 602)
    np_367621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 31), 'np', False)
    # Obtaining the member 'cumsum' of a type (line 602)
    cumsum_367622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 31), np_367621, 'cumsum')
    # Calling cumsum(args, kwargs) (line 602)
    cumsum_call_result_367625 = invoke(stypy.reporting.localization.Localization(__file__, 602, 31), cumsum_367622, *[brow_lengths_367623], **kwargs_367624)
    
    # Processing the call keyword arguments (line 602)
    kwargs_367626 = {}
    # Getting the type of 'np' (line 602)
    np_367618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 18), 'np', False)
    # Obtaining the member 'append' of a type (line 602)
    append_367619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 18), np_367618, 'append')
    # Calling append(args, kwargs) (line 602)
    append_call_result_367627 = invoke(stypy.reporting.localization.Localization(__file__, 602, 18), append_367619, *[int_367620, cumsum_call_result_367625], **kwargs_367626)
    
    # Assigning a type to the variable 'row_offsets' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'row_offsets', append_call_result_367627)
    
    # Assigning a Call to a Name (line 603):
    
    # Assigning a Call to a Name (line 603):
    
    # Call to append(...): (line 603)
    # Processing the call arguments (line 603)
    int_367630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 28), 'int')
    
    # Call to cumsum(...): (line 603)
    # Processing the call arguments (line 603)
    # Getting the type of 'bcol_lengths' (line 603)
    bcol_lengths_367633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 41), 'bcol_lengths', False)
    # Processing the call keyword arguments (line 603)
    kwargs_367634 = {}
    # Getting the type of 'np' (line 603)
    np_367631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 31), 'np', False)
    # Obtaining the member 'cumsum' of a type (line 603)
    cumsum_367632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 31), np_367631, 'cumsum')
    # Calling cumsum(args, kwargs) (line 603)
    cumsum_call_result_367635 = invoke(stypy.reporting.localization.Localization(__file__, 603, 31), cumsum_367632, *[bcol_lengths_367633], **kwargs_367634)
    
    # Processing the call keyword arguments (line 603)
    kwargs_367636 = {}
    # Getting the type of 'np' (line 603)
    np_367628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 18), 'np', False)
    # Obtaining the member 'append' of a type (line 603)
    append_367629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 18), np_367628, 'append')
    # Calling append(args, kwargs) (line 603)
    append_call_result_367637 = invoke(stypy.reporting.localization.Localization(__file__, 603, 18), append_367629, *[int_367630, cumsum_call_result_367635], **kwargs_367636)
    
    # Assigning a type to the variable 'col_offsets' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'col_offsets', append_call_result_367637)
    
    # Assigning a Tuple to a Name (line 605):
    
    # Assigning a Tuple to a Name (line 605):
    
    # Obtaining an instance of the builtin type 'tuple' (line 605)
    tuple_367638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 605)
    # Adding element type (line 605)
    
    # Obtaining the type of the subscript
    int_367639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 25), 'int')
    # Getting the type of 'row_offsets' (line 605)
    row_offsets_367640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 13), 'row_offsets')
    # Obtaining the member '__getitem__' of a type (line 605)
    getitem___367641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 13), row_offsets_367640, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 605)
    subscript_call_result_367642 = invoke(stypy.reporting.localization.Localization(__file__, 605, 13), getitem___367641, int_367639)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 13), tuple_367638, subscript_call_result_367642)
    # Adding element type (line 605)
    
    # Obtaining the type of the subscript
    int_367643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 42), 'int')
    # Getting the type of 'col_offsets' (line 605)
    col_offsets_367644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 30), 'col_offsets')
    # Obtaining the member '__getitem__' of a type (line 605)
    getitem___367645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 30), col_offsets_367644, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 605)
    subscript_call_result_367646 = invoke(stypy.reporting.localization.Localization(__file__, 605, 30), getitem___367645, int_367643)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 13), tuple_367638, subscript_call_result_367646)
    
    # Assigning a type to the variable 'shape' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 4), 'shape', tuple_367638)
    
    # Assigning a Call to a Name (line 607):
    
    # Assigning a Call to a Name (line 607):
    
    # Call to empty(...): (line 607)
    # Processing the call arguments (line 607)
    # Getting the type of 'nnz' (line 607)
    nnz_367649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 20), 'nnz', False)
    # Processing the call keyword arguments (line 607)
    # Getting the type of 'dtype' (line 607)
    dtype_367650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 31), 'dtype', False)
    keyword_367651 = dtype_367650
    kwargs_367652 = {'dtype': keyword_367651}
    # Getting the type of 'np' (line 607)
    np_367647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 11), 'np', False)
    # Obtaining the member 'empty' of a type (line 607)
    empty_367648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 11), np_367647, 'empty')
    # Calling empty(args, kwargs) (line 607)
    empty_call_result_367653 = invoke(stypy.reporting.localization.Localization(__file__, 607, 11), empty_367648, *[nnz_367649], **kwargs_367652)
    
    # Assigning a type to the variable 'data' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'data', empty_call_result_367653)
    
    # Assigning a Call to a Name (line 608):
    
    # Assigning a Call to a Name (line 608):
    
    # Call to get_index_dtype(...): (line 608)
    # Processing the call keyword arguments (line 608)
    
    # Call to max(...): (line 608)
    # Processing the call arguments (line 608)
    # Getting the type of 'shape' (line 608)
    shape_367656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 43), 'shape', False)
    # Processing the call keyword arguments (line 608)
    kwargs_367657 = {}
    # Getting the type of 'max' (line 608)
    max_367655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 39), 'max', False)
    # Calling max(args, kwargs) (line 608)
    max_call_result_367658 = invoke(stypy.reporting.localization.Localization(__file__, 608, 39), max_367655, *[shape_367656], **kwargs_367657)
    
    keyword_367659 = max_call_result_367658
    kwargs_367660 = {'maxval': keyword_367659}
    # Getting the type of 'get_index_dtype' (line 608)
    get_index_dtype_367654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 16), 'get_index_dtype', False)
    # Calling get_index_dtype(args, kwargs) (line 608)
    get_index_dtype_call_result_367661 = invoke(stypy.reporting.localization.Localization(__file__, 608, 16), get_index_dtype_367654, *[], **kwargs_367660)
    
    # Assigning a type to the variable 'idx_dtype' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 4), 'idx_dtype', get_index_dtype_call_result_367661)
    
    # Assigning a Call to a Name (line 609):
    
    # Assigning a Call to a Name (line 609):
    
    # Call to empty(...): (line 609)
    # Processing the call arguments (line 609)
    # Getting the type of 'nnz' (line 609)
    nnz_367664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 19), 'nnz', False)
    # Processing the call keyword arguments (line 609)
    # Getting the type of 'idx_dtype' (line 609)
    idx_dtype_367665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 30), 'idx_dtype', False)
    keyword_367666 = idx_dtype_367665
    kwargs_367667 = {'dtype': keyword_367666}
    # Getting the type of 'np' (line 609)
    np_367662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 609)
    empty_367663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 10), np_367662, 'empty')
    # Calling empty(args, kwargs) (line 609)
    empty_call_result_367668 = invoke(stypy.reporting.localization.Localization(__file__, 609, 10), empty_367663, *[nnz_367664], **kwargs_367667)
    
    # Assigning a type to the variable 'row' (line 609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 4), 'row', empty_call_result_367668)
    
    # Assigning a Call to a Name (line 610):
    
    # Assigning a Call to a Name (line 610):
    
    # Call to empty(...): (line 610)
    # Processing the call arguments (line 610)
    # Getting the type of 'nnz' (line 610)
    nnz_367671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 19), 'nnz', False)
    # Processing the call keyword arguments (line 610)
    # Getting the type of 'idx_dtype' (line 610)
    idx_dtype_367672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 30), 'idx_dtype', False)
    keyword_367673 = idx_dtype_367672
    kwargs_367674 = {'dtype': keyword_367673}
    # Getting the type of 'np' (line 610)
    np_367669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 610)
    empty_367670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 10), np_367669, 'empty')
    # Calling empty(args, kwargs) (line 610)
    empty_call_result_367675 = invoke(stypy.reporting.localization.Localization(__file__, 610, 10), empty_367670, *[nnz_367671], **kwargs_367674)
    
    # Assigning a type to the variable 'col' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'col', empty_call_result_367675)
    
    # Assigning a Num to a Name (line 612):
    
    # Assigning a Num to a Name (line 612):
    int_367676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 10), 'int')
    # Assigning a type to the variable 'nnz' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 4), 'nnz', int_367676)
    
    # Assigning a Call to a Tuple (line 613):
    
    # Assigning a Subscript to a Name (line 613):
    
    # Obtaining the type of the subscript
    int_367677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 4), 'int')
    
    # Call to nonzero(...): (line 613)
    # Processing the call arguments (line 613)
    # Getting the type of 'block_mask' (line 613)
    block_mask_367680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 24), 'block_mask', False)
    # Processing the call keyword arguments (line 613)
    kwargs_367681 = {}
    # Getting the type of 'np' (line 613)
    np_367678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 13), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 613)
    nonzero_367679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 13), np_367678, 'nonzero')
    # Calling nonzero(args, kwargs) (line 613)
    nonzero_call_result_367682 = invoke(stypy.reporting.localization.Localization(__file__, 613, 13), nonzero_367679, *[block_mask_367680], **kwargs_367681)
    
    # Obtaining the member '__getitem__' of a type (line 613)
    getitem___367683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 4), nonzero_call_result_367682, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 613)
    subscript_call_result_367684 = invoke(stypy.reporting.localization.Localization(__file__, 613, 4), getitem___367683, int_367677)
    
    # Assigning a type to the variable 'tuple_var_assignment_366307' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'tuple_var_assignment_366307', subscript_call_result_367684)
    
    # Assigning a Subscript to a Name (line 613):
    
    # Obtaining the type of the subscript
    int_367685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 4), 'int')
    
    # Call to nonzero(...): (line 613)
    # Processing the call arguments (line 613)
    # Getting the type of 'block_mask' (line 613)
    block_mask_367688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 24), 'block_mask', False)
    # Processing the call keyword arguments (line 613)
    kwargs_367689 = {}
    # Getting the type of 'np' (line 613)
    np_367686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 13), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 613)
    nonzero_367687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 13), np_367686, 'nonzero')
    # Calling nonzero(args, kwargs) (line 613)
    nonzero_call_result_367690 = invoke(stypy.reporting.localization.Localization(__file__, 613, 13), nonzero_367687, *[block_mask_367688], **kwargs_367689)
    
    # Obtaining the member '__getitem__' of a type (line 613)
    getitem___367691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 4), nonzero_call_result_367690, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 613)
    subscript_call_result_367692 = invoke(stypy.reporting.localization.Localization(__file__, 613, 4), getitem___367691, int_367685)
    
    # Assigning a type to the variable 'tuple_var_assignment_366308' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'tuple_var_assignment_366308', subscript_call_result_367692)
    
    # Assigning a Name to a Name (line 613):
    # Getting the type of 'tuple_var_assignment_366307' (line 613)
    tuple_var_assignment_366307_367693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'tuple_var_assignment_366307')
    # Assigning a type to the variable 'ii' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'ii', tuple_var_assignment_366307_367693)
    
    # Assigning a Name to a Name (line 613):
    # Getting the type of 'tuple_var_assignment_366308' (line 613)
    tuple_var_assignment_366308_367694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'tuple_var_assignment_366308')
    # Assigning a type to the variable 'jj' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'jj', tuple_var_assignment_366308_367694)
    
    
    # Call to zip(...): (line 614)
    # Processing the call arguments (line 614)
    # Getting the type of 'ii' (line 614)
    ii_367696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 20), 'ii', False)
    # Getting the type of 'jj' (line 614)
    jj_367697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 24), 'jj', False)
    # Processing the call keyword arguments (line 614)
    kwargs_367698 = {}
    # Getting the type of 'zip' (line 614)
    zip_367695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 16), 'zip', False)
    # Calling zip(args, kwargs) (line 614)
    zip_call_result_367699 = invoke(stypy.reporting.localization.Localization(__file__, 614, 16), zip_367695, *[ii_367696, jj_367697], **kwargs_367698)
    
    # Testing the type of a for loop iterable (line 614)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 614, 4), zip_call_result_367699)
    # Getting the type of the for loop variable (line 614)
    for_loop_var_367700 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 614, 4), zip_call_result_367699)
    # Assigning a type to the variable 'i' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 4), for_loop_var_367700))
    # Assigning a type to the variable 'j' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 4), for_loop_var_367700))
    # SSA begins for a for statement (line 614)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 615):
    
    # Assigning a Subscript to a Name (line 615):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 615)
    tuple_367701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 615)
    # Adding element type (line 615)
    # Getting the type of 'i' (line 615)
    i_367702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 19), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 19), tuple_367701, i_367702)
    # Adding element type (line 615)
    # Getting the type of 'j' (line 615)
    j_367703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 22), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 19), tuple_367701, j_367703)
    
    # Getting the type of 'blocks' (line 615)
    blocks_367704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'blocks')
    # Obtaining the member '__getitem__' of a type (line 615)
    getitem___367705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 12), blocks_367704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 615)
    subscript_call_result_367706 = invoke(stypy.reporting.localization.Localization(__file__, 615, 12), getitem___367705, tuple_367701)
    
    # Assigning a type to the variable 'B' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'B', subscript_call_result_367706)
    
    # Assigning a Call to a Name (line 616):
    
    # Assigning a Call to a Name (line 616):
    
    # Call to slice(...): (line 616)
    # Processing the call arguments (line 616)
    # Getting the type of 'nnz' (line 616)
    nnz_367708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 20), 'nnz', False)
    # Getting the type of 'nnz' (line 616)
    nnz_367709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 25), 'nnz', False)
    # Getting the type of 'B' (line 616)
    B_367710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 31), 'B', False)
    # Obtaining the member 'nnz' of a type (line 616)
    nnz_367711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 31), B_367710, 'nnz')
    # Applying the binary operator '+' (line 616)
    result_add_367712 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 25), '+', nnz_367709, nnz_367711)
    
    # Processing the call keyword arguments (line 616)
    kwargs_367713 = {}
    # Getting the type of 'slice' (line 616)
    slice_367707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 14), 'slice', False)
    # Calling slice(args, kwargs) (line 616)
    slice_call_result_367714 = invoke(stypy.reporting.localization.Localization(__file__, 616, 14), slice_367707, *[nnz_367708, result_add_367712], **kwargs_367713)
    
    # Assigning a type to the variable 'idx' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'idx', slice_call_result_367714)
    
    # Assigning a Attribute to a Subscript (line 617):
    
    # Assigning a Attribute to a Subscript (line 617):
    # Getting the type of 'B' (line 617)
    B_367715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 20), 'B')
    # Obtaining the member 'data' of a type (line 617)
    data_367716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 20), B_367715, 'data')
    # Getting the type of 'data' (line 617)
    data_367717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'data')
    # Getting the type of 'idx' (line 617)
    idx_367718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 13), 'idx')
    # Storing an element on a container (line 617)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 8), data_367717, (idx_367718, data_367716))
    
    # Assigning a BinOp to a Subscript (line 618):
    
    # Assigning a BinOp to a Subscript (line 618):
    # Getting the type of 'B' (line 618)
    B_367719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 19), 'B')
    # Obtaining the member 'row' of a type (line 618)
    row_367720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 19), B_367719, 'row')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 618)
    i_367721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 39), 'i')
    # Getting the type of 'row_offsets' (line 618)
    row_offsets_367722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 27), 'row_offsets')
    # Obtaining the member '__getitem__' of a type (line 618)
    getitem___367723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 27), row_offsets_367722, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 618)
    subscript_call_result_367724 = invoke(stypy.reporting.localization.Localization(__file__, 618, 27), getitem___367723, i_367721)
    
    # Applying the binary operator '+' (line 618)
    result_add_367725 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 19), '+', row_367720, subscript_call_result_367724)
    
    # Getting the type of 'row' (line 618)
    row_367726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'row')
    # Getting the type of 'idx' (line 618)
    idx_367727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 12), 'idx')
    # Storing an element on a container (line 618)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 8), row_367726, (idx_367727, result_add_367725))
    
    # Assigning a BinOp to a Subscript (line 619):
    
    # Assigning a BinOp to a Subscript (line 619):
    # Getting the type of 'B' (line 619)
    B_367728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 19), 'B')
    # Obtaining the member 'col' of a type (line 619)
    col_367729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 19), B_367728, 'col')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 619)
    j_367730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 39), 'j')
    # Getting the type of 'col_offsets' (line 619)
    col_offsets_367731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 27), 'col_offsets')
    # Obtaining the member '__getitem__' of a type (line 619)
    getitem___367732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 27), col_offsets_367731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 619)
    subscript_call_result_367733 = invoke(stypy.reporting.localization.Localization(__file__, 619, 27), getitem___367732, j_367730)
    
    # Applying the binary operator '+' (line 619)
    result_add_367734 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 19), '+', col_367729, subscript_call_result_367733)
    
    # Getting the type of 'col' (line 619)
    col_367735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'col')
    # Getting the type of 'idx' (line 619)
    idx_367736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'idx')
    # Storing an element on a container (line 619)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 8), col_367735, (idx_367736, result_add_367734))
    
    # Getting the type of 'nnz' (line 620)
    nnz_367737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'nnz')
    # Getting the type of 'B' (line 620)
    B_367738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 15), 'B')
    # Obtaining the member 'nnz' of a type (line 620)
    nnz_367739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 15), B_367738, 'nnz')
    # Applying the binary operator '+=' (line 620)
    result_iadd_367740 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 8), '+=', nnz_367737, nnz_367739)
    # Assigning a type to the variable 'nnz' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'nnz', result_iadd_367740)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to asformat(...): (line 622)
    # Processing the call arguments (line 622)
    # Getting the type of 'format' (line 622)
    format_367752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 64), 'format', False)
    # Processing the call keyword arguments (line 622)
    kwargs_367753 = {}
    
    # Call to coo_matrix(...): (line 622)
    # Processing the call arguments (line 622)
    
    # Obtaining an instance of the builtin type 'tuple' (line 622)
    tuple_367742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 622)
    # Adding element type (line 622)
    # Getting the type of 'data' (line 622)
    data_367743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 23), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 23), tuple_367742, data_367743)
    # Adding element type (line 622)
    
    # Obtaining an instance of the builtin type 'tuple' (line 622)
    tuple_367744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 622)
    # Adding element type (line 622)
    # Getting the type of 'row' (line 622)
    row_367745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 30), 'row', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 30), tuple_367744, row_367745)
    # Adding element type (line 622)
    # Getting the type of 'col' (line 622)
    col_367746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 35), 'col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 30), tuple_367744, col_367746)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 23), tuple_367742, tuple_367744)
    
    # Processing the call keyword arguments (line 622)
    # Getting the type of 'shape' (line 622)
    shape_367747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 48), 'shape', False)
    keyword_367748 = shape_367747
    kwargs_367749 = {'shape': keyword_367748}
    # Getting the type of 'coo_matrix' (line 622)
    coo_matrix_367741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 11), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 622)
    coo_matrix_call_result_367750 = invoke(stypy.reporting.localization.Localization(__file__, 622, 11), coo_matrix_367741, *[tuple_367742], **kwargs_367749)
    
    # Obtaining the member 'asformat' of a type (line 622)
    asformat_367751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 11), coo_matrix_call_result_367750, 'asformat')
    # Calling asformat(args, kwargs) (line 622)
    asformat_call_result_367754 = invoke(stypy.reporting.localization.Localization(__file__, 622, 11), asformat_367751, *[format_367752], **kwargs_367753)
    
    # Assigning a type to the variable 'stypy_return_type' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'stypy_return_type', asformat_call_result_367754)
    
    # ################# End of 'bmat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bmat' in the type store
    # Getting the type of 'stypy_return_type' (line 501)
    stypy_return_type_367755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_367755)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bmat'
    return stypy_return_type_367755

# Assigning a type to the variable 'bmat' (line 501)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 0), 'bmat', bmat)

@norecursion
def block_diag(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 625)
    None_367756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 28), 'None')
    # Getting the type of 'None' (line 625)
    None_367757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 40), 'None')
    defaults = [None_367756, None_367757]
    # Create a new context for function 'block_diag'
    module_type_store = module_type_store.open_function_context('block_diag', 625, 0, False)
    
    # Passed parameters checking function
    block_diag.stypy_localization = localization
    block_diag.stypy_type_of_self = None
    block_diag.stypy_type_store = module_type_store
    block_diag.stypy_function_name = 'block_diag'
    block_diag.stypy_param_names_list = ['mats', 'format', 'dtype']
    block_diag.stypy_varargs_param_name = None
    block_diag.stypy_kwargs_param_name = None
    block_diag.stypy_call_defaults = defaults
    block_diag.stypy_call_varargs = varargs
    block_diag.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'block_diag', ['mats', 'format', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'block_diag', localization, ['mats', 'format', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'block_diag(...)' code ##################

    str_367758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, (-1)), 'str', '\n    Build a block diagonal sparse matrix from provided matrices.\n\n    Parameters\n    ----------\n    mats : sequence of matrices\n        Input matrices.\n    format : str, optional\n        The sparse format of the result (e.g. "csr").  If not given, the matrix\n        is returned in "coo" format.\n    dtype : dtype specifier, optional\n        The data-type of the output matrix.  If not given, the dtype is\n        determined from that of `blocks`.\n\n    Returns\n    -------\n    res : sparse matrix\n\n    Notes\n    -----\n\n    .. versionadded:: 0.11.0\n\n    See Also\n    --------\n    bmat, diags\n\n    Examples\n    --------\n    >>> from scipy.sparse import coo_matrix, block_diag\n    >>> A = coo_matrix([[1, 2], [3, 4]])\n    >>> B = coo_matrix([[5], [6]])\n    >>> C = coo_matrix([[7]])\n    >>> block_diag((A, B, C)).toarray()\n    array([[1, 2, 0, 0],\n           [3, 4, 0, 0],\n           [0, 0, 5, 0],\n           [0, 0, 6, 0],\n           [0, 0, 0, 7]])\n\n    ')
    
    # Assigning a Call to a Name (line 667):
    
    # Assigning a Call to a Name (line 667):
    
    # Call to len(...): (line 667)
    # Processing the call arguments (line 667)
    # Getting the type of 'mats' (line 667)
    mats_367760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 15), 'mats', False)
    # Processing the call keyword arguments (line 667)
    kwargs_367761 = {}
    # Getting the type of 'len' (line 667)
    len_367759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 11), 'len', False)
    # Calling len(args, kwargs) (line 667)
    len_call_result_367762 = invoke(stypy.reporting.localization.Localization(__file__, 667, 11), len_367759, *[mats_367760], **kwargs_367761)
    
    # Assigning a type to the variable 'nmat' (line 667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 4), 'nmat', len_call_result_367762)
    
    # Assigning a List to a Name (line 668):
    
    # Assigning a List to a Name (line 668):
    
    # Obtaining an instance of the builtin type 'list' (line 668)
    list_367763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 668)
    
    # Assigning a type to the variable 'rows' (line 668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 4), 'rows', list_367763)
    
    
    # Call to enumerate(...): (line 669)
    # Processing the call arguments (line 669)
    # Getting the type of 'mats' (line 669)
    mats_367765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 27), 'mats', False)
    # Processing the call keyword arguments (line 669)
    kwargs_367766 = {}
    # Getting the type of 'enumerate' (line 669)
    enumerate_367764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 17), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 669)
    enumerate_call_result_367767 = invoke(stypy.reporting.localization.Localization(__file__, 669, 17), enumerate_367764, *[mats_367765], **kwargs_367766)
    
    # Testing the type of a for loop iterable (line 669)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 669, 4), enumerate_call_result_367767)
    # Getting the type of the for loop variable (line 669)
    for_loop_var_367768 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 669, 4), enumerate_call_result_367767)
    # Assigning a type to the variable 'ia' (line 669)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 4), 'ia', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 669, 4), for_loop_var_367768))
    # Assigning a type to the variable 'a' (line 669)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 4), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 669, 4), for_loop_var_367768))
    # SSA begins for a for statement (line 669)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 670):
    
    # Assigning a BinOp to a Name (line 670):
    
    # Obtaining an instance of the builtin type 'list' (line 670)
    list_367769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 670)
    # Adding element type (line 670)
    # Getting the type of 'None' (line 670)
    None_367770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 15), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 670, 14), list_367769, None_367770)
    
    # Getting the type of 'nmat' (line 670)
    nmat_367771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 21), 'nmat')
    # Applying the binary operator '*' (line 670)
    result_mul_367772 = python_operator(stypy.reporting.localization.Localization(__file__, 670, 14), '*', list_367769, nmat_367771)
    
    # Assigning a type to the variable 'row' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 8), 'row', result_mul_367772)
    
    
    # Call to issparse(...): (line 671)
    # Processing the call arguments (line 671)
    # Getting the type of 'a' (line 671)
    a_367774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 20), 'a', False)
    # Processing the call keyword arguments (line 671)
    kwargs_367775 = {}
    # Getting the type of 'issparse' (line 671)
    issparse_367773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 11), 'issparse', False)
    # Calling issparse(args, kwargs) (line 671)
    issparse_call_result_367776 = invoke(stypy.reporting.localization.Localization(__file__, 671, 11), issparse_367773, *[a_367774], **kwargs_367775)
    
    # Testing the type of an if condition (line 671)
    if_condition_367777 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 671, 8), issparse_call_result_367776)
    # Assigning a type to the variable 'if_condition_367777' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'if_condition_367777', if_condition_367777)
    # SSA begins for if statement (line 671)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 672):
    
    # Assigning a Name to a Subscript (line 672):
    # Getting the type of 'a' (line 672)
    a_367778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 22), 'a')
    # Getting the type of 'row' (line 672)
    row_367779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'row')
    # Getting the type of 'ia' (line 672)
    ia_367780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 16), 'ia')
    # Storing an element on a container (line 672)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 12), row_367779, (ia_367780, a_367778))
    # SSA branch for the else part of an if statement (line 671)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Subscript (line 674):
    
    # Assigning a Call to a Subscript (line 674):
    
    # Call to coo_matrix(...): (line 674)
    # Processing the call arguments (line 674)
    # Getting the type of 'a' (line 674)
    a_367782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 33), 'a', False)
    # Processing the call keyword arguments (line 674)
    kwargs_367783 = {}
    # Getting the type of 'coo_matrix' (line 674)
    coo_matrix_367781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 22), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 674)
    coo_matrix_call_result_367784 = invoke(stypy.reporting.localization.Localization(__file__, 674, 22), coo_matrix_367781, *[a_367782], **kwargs_367783)
    
    # Getting the type of 'row' (line 674)
    row_367785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'row')
    # Getting the type of 'ia' (line 674)
    ia_367786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 16), 'ia')
    # Storing an element on a container (line 674)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 674, 12), row_367785, (ia_367786, coo_matrix_call_result_367784))
    # SSA join for if statement (line 671)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 675)
    # Processing the call arguments (line 675)
    # Getting the type of 'row' (line 675)
    row_367789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 20), 'row', False)
    # Processing the call keyword arguments (line 675)
    kwargs_367790 = {}
    # Getting the type of 'rows' (line 675)
    rows_367787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 8), 'rows', False)
    # Obtaining the member 'append' of a type (line 675)
    append_367788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 8), rows_367787, 'append')
    # Calling append(args, kwargs) (line 675)
    append_call_result_367791 = invoke(stypy.reporting.localization.Localization(__file__, 675, 8), append_367788, *[row_367789], **kwargs_367790)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to bmat(...): (line 676)
    # Processing the call arguments (line 676)
    # Getting the type of 'rows' (line 676)
    rows_367793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'rows', False)
    # Processing the call keyword arguments (line 676)
    # Getting the type of 'format' (line 676)
    format_367794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 29), 'format', False)
    keyword_367795 = format_367794
    # Getting the type of 'dtype' (line 676)
    dtype_367796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 43), 'dtype', False)
    keyword_367797 = dtype_367796
    kwargs_367798 = {'dtype': keyword_367797, 'format': keyword_367795}
    # Getting the type of 'bmat' (line 676)
    bmat_367792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 11), 'bmat', False)
    # Calling bmat(args, kwargs) (line 676)
    bmat_call_result_367799 = invoke(stypy.reporting.localization.Localization(__file__, 676, 11), bmat_367792, *[rows_367793], **kwargs_367798)
    
    # Assigning a type to the variable 'stypy_return_type' (line 676)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 4), 'stypy_return_type', bmat_call_result_367799)
    
    # ################# End of 'block_diag(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'block_diag' in the type store
    # Getting the type of 'stypy_return_type' (line 625)
    stypy_return_type_367800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_367800)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'block_diag'
    return stypy_return_type_367800

# Assigning a type to the variable 'block_diag' (line 625)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 0), 'block_diag', block_diag)

@norecursion
def random(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_367801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 25), 'float')
    str_367802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 38), 'str', 'coo')
    # Getting the type of 'None' (line 679)
    None_367803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 51), 'None')
    # Getting the type of 'None' (line 680)
    None_367804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 24), 'None')
    # Getting the type of 'None' (line 680)
    None_367805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 39), 'None')
    defaults = [float_367801, str_367802, None_367803, None_367804, None_367805]
    # Create a new context for function 'random'
    module_type_store = module_type_store.open_function_context('random', 679, 0, False)
    
    # Passed parameters checking function
    random.stypy_localization = localization
    random.stypy_type_of_self = None
    random.stypy_type_store = module_type_store
    random.stypy_function_name = 'random'
    random.stypy_param_names_list = ['m', 'n', 'density', 'format', 'dtype', 'random_state', 'data_rvs']
    random.stypy_varargs_param_name = None
    random.stypy_kwargs_param_name = None
    random.stypy_call_defaults = defaults
    random.stypy_call_varargs = varargs
    random.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'random', ['m', 'n', 'density', 'format', 'dtype', 'random_state', 'data_rvs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'random', localization, ['m', 'n', 'density', 'format', 'dtype', 'random_state', 'data_rvs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'random(...)' code ##################

    str_367806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, (-1)), 'str', 'Generate a sparse matrix of the given shape and density with randomly\n    distributed values.\n\n    Parameters\n    ----------\n    m, n : int\n        shape of the matrix\n    density : real, optional\n        density of the generated matrix: density equal to one means a full\n        matrix, density of 0 means a matrix with no non-zero items.\n    format : str, optional\n        sparse matrix format.\n    dtype : dtype, optional\n        type of the returned matrix values.\n    random_state : {numpy.random.RandomState, int}, optional\n        Random number generator or random seed. If not given, the singleton\n        numpy.random will be used.  This random state will be used\n        for sampling the sparsity structure, but not necessarily for sampling\n        the values of the structurally nonzero entries of the matrix.\n    data_rvs : callable, optional\n        Samples a requested number of random values.\n        This function should take a single argument specifying the length\n        of the ndarray that it will return.  The structurally nonzero entries\n        of the sparse random matrix will be taken from the array sampled\n        by this function.  By default, uniform [0, 1) random values will be\n        sampled using the same random state as is used for sampling\n        the sparsity structure.\n\n    Returns\n    -------\n    res : sparse matrix\n\n    Examples\n    --------\n    >>> from scipy.sparse import random\n    >>> from scipy import stats\n    >>> class CustomRandomState(object):\n    ...     def randint(self, k):\n    ...         i = np.random.randint(k)\n    ...         return i - i % 2\n    >>> rs = CustomRandomState()\n    >>> rvs = stats.poisson(25, loc=10).rvs\n    >>> S = random(3, 4, density=0.25, random_state=rs, data_rvs=rvs)\n    >>> S.A\n    array([[ 36.,   0.,  33.,   0.],   # random\n           [  0.,   0.,   0.,   0.],\n           [  0.,   0.,  36.,   0.]])\n\n    Notes\n    -----\n    Only float types are supported for now.\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'density' (line 733)
    density_367807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 7), 'density')
    int_367808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 17), 'int')
    # Applying the binary operator '<' (line 733)
    result_lt_367809 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 7), '<', density_367807, int_367808)
    
    
    # Getting the type of 'density' (line 733)
    density_367810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 22), 'density')
    int_367811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 32), 'int')
    # Applying the binary operator '>' (line 733)
    result_gt_367812 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 22), '>', density_367810, int_367811)
    
    # Applying the binary operator 'or' (line 733)
    result_or_keyword_367813 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 7), 'or', result_lt_367809, result_gt_367812)
    
    # Testing the type of an if condition (line 733)
    if_condition_367814 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 733, 4), result_or_keyword_367813)
    # Assigning a type to the variable 'if_condition_367814' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), 'if_condition_367814', if_condition_367814)
    # SSA begins for if statement (line 733)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 734)
    # Processing the call arguments (line 734)
    str_367816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 25), 'str', 'density expected to be 0 <= density <= 1')
    # Processing the call keyword arguments (line 734)
    kwargs_367817 = {}
    # Getting the type of 'ValueError' (line 734)
    ValueError_367815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 734)
    ValueError_call_result_367818 = invoke(stypy.reporting.localization.Localization(__file__, 734, 14), ValueError_367815, *[str_367816], **kwargs_367817)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 734, 8), ValueError_call_result_367818, 'raise parameter', BaseException)
    # SSA join for if statement (line 733)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 735):
    
    # Assigning a Call to a Name (line 735):
    
    # Call to dtype(...): (line 735)
    # Processing the call arguments (line 735)
    # Getting the type of 'dtype' (line 735)
    dtype_367821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 21), 'dtype', False)
    # Processing the call keyword arguments (line 735)
    kwargs_367822 = {}
    # Getting the type of 'np' (line 735)
    np_367819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 12), 'np', False)
    # Obtaining the member 'dtype' of a type (line 735)
    dtype_367820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 12), np_367819, 'dtype')
    # Calling dtype(args, kwargs) (line 735)
    dtype_call_result_367823 = invoke(stypy.reporting.localization.Localization(__file__, 735, 12), dtype_367820, *[dtype_367821], **kwargs_367822)
    
    # Assigning a type to the variable 'dtype' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'dtype', dtype_call_result_367823)
    
    
    # Getting the type of 'dtype' (line 736)
    dtype_367824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 7), 'dtype')
    # Obtaining the member 'char' of a type (line 736)
    char_367825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 7), dtype_367824, 'char')
    str_367826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 25), 'str', 'fdg')
    # Applying the binary operator 'notin' (line 736)
    result_contains_367827 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 7), 'notin', char_367825, str_367826)
    
    # Testing the type of an if condition (line 736)
    if_condition_367828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 736, 4), result_contains_367827)
    # Assigning a type to the variable 'if_condition_367828' (line 736)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 4), 'if_condition_367828', if_condition_367828)
    # SSA begins for if statement (line 736)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to NotImplementedError(...): (line 737)
    # Processing the call arguments (line 737)
    str_367830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 34), 'str', 'type %s not supported')
    # Getting the type of 'dtype' (line 737)
    dtype_367831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 60), 'dtype', False)
    # Applying the binary operator '%' (line 737)
    result_mod_367832 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 34), '%', str_367830, dtype_367831)
    
    # Processing the call keyword arguments (line 737)
    kwargs_367833 = {}
    # Getting the type of 'NotImplementedError' (line 737)
    NotImplementedError_367829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 14), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 737)
    NotImplementedError_call_result_367834 = invoke(stypy.reporting.localization.Localization(__file__, 737, 14), NotImplementedError_367829, *[result_mod_367832], **kwargs_367833)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 737, 8), NotImplementedError_call_result_367834, 'raise parameter', BaseException)
    # SSA join for if statement (line 736)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 739):
    
    # Assigning a BinOp to a Name (line 739):
    # Getting the type of 'm' (line 739)
    m_367835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 9), 'm')
    # Getting the type of 'n' (line 739)
    n_367836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 13), 'n')
    # Applying the binary operator '*' (line 739)
    result_mul_367837 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 9), '*', m_367835, n_367836)
    
    # Assigning a type to the variable 'mn' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 4), 'mn', result_mul_367837)
    
    # Assigning a Attribute to a Name (line 741):
    
    # Assigning a Attribute to a Name (line 741):
    # Getting the type of 'np' (line 741)
    np_367838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 9), 'np')
    # Obtaining the member 'intc' of a type (line 741)
    intc_367839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 9), np_367838, 'intc')
    # Assigning a type to the variable 'tp' (line 741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 4), 'tp', intc_367839)
    
    
    # Getting the type of 'mn' (line 742)
    mn_367840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 7), 'mn')
    
    # Call to iinfo(...): (line 742)
    # Processing the call arguments (line 742)
    # Getting the type of 'tp' (line 742)
    tp_367843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 21), 'tp', False)
    # Processing the call keyword arguments (line 742)
    kwargs_367844 = {}
    # Getting the type of 'np' (line 742)
    np_367841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 12), 'np', False)
    # Obtaining the member 'iinfo' of a type (line 742)
    iinfo_367842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 12), np_367841, 'iinfo')
    # Calling iinfo(args, kwargs) (line 742)
    iinfo_call_result_367845 = invoke(stypy.reporting.localization.Localization(__file__, 742, 12), iinfo_367842, *[tp_367843], **kwargs_367844)
    
    # Obtaining the member 'max' of a type (line 742)
    max_367846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 12), iinfo_call_result_367845, 'max')
    # Applying the binary operator '>' (line 742)
    result_gt_367847 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 7), '>', mn_367840, max_367846)
    
    # Testing the type of an if condition (line 742)
    if_condition_367848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 742, 4), result_gt_367847)
    # Assigning a type to the variable 'if_condition_367848' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'if_condition_367848', if_condition_367848)
    # SSA begins for if statement (line 742)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 743):
    
    # Assigning a Attribute to a Name (line 743):
    # Getting the type of 'np' (line 743)
    np_367849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 13), 'np')
    # Obtaining the member 'int64' of a type (line 743)
    int64_367850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 13), np_367849, 'int64')
    # Assigning a type to the variable 'tp' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'tp', int64_367850)
    # SSA join for if statement (line 742)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mn' (line 745)
    mn_367851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 7), 'mn')
    
    # Call to iinfo(...): (line 745)
    # Processing the call arguments (line 745)
    # Getting the type of 'tp' (line 745)
    tp_367854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 21), 'tp', False)
    # Processing the call keyword arguments (line 745)
    kwargs_367855 = {}
    # Getting the type of 'np' (line 745)
    np_367852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 12), 'np', False)
    # Obtaining the member 'iinfo' of a type (line 745)
    iinfo_367853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 12), np_367852, 'iinfo')
    # Calling iinfo(args, kwargs) (line 745)
    iinfo_call_result_367856 = invoke(stypy.reporting.localization.Localization(__file__, 745, 12), iinfo_367853, *[tp_367854], **kwargs_367855)
    
    # Obtaining the member 'max' of a type (line 745)
    max_367857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 12), iinfo_call_result_367856, 'max')
    # Applying the binary operator '>' (line 745)
    result_gt_367858 = python_operator(stypy.reporting.localization.Localization(__file__, 745, 7), '>', mn_367851, max_367857)
    
    # Testing the type of an if condition (line 745)
    if_condition_367859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 745, 4), result_gt_367858)
    # Assigning a type to the variable 'if_condition_367859' (line 745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 4), 'if_condition_367859', if_condition_367859)
    # SSA begins for if statement (line 745)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 746):
    
    # Assigning a Str to a Name (line 746):
    str_367860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, (-1)), 'str', 'Trying to generate a random sparse matrix such as the product of dimensions is\ngreater than %d - this is not supported on this machine\n')
    # Assigning a type to the variable 'msg' (line 746)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 8), 'msg', str_367860)
    
    # Call to ValueError(...): (line 750)
    # Processing the call arguments (line 750)
    # Getting the type of 'msg' (line 750)
    msg_367862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 25), 'msg', False)
    
    # Call to iinfo(...): (line 750)
    # Processing the call arguments (line 750)
    # Getting the type of 'tp' (line 750)
    tp_367865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 40), 'tp', False)
    # Processing the call keyword arguments (line 750)
    kwargs_367866 = {}
    # Getting the type of 'np' (line 750)
    np_367863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 31), 'np', False)
    # Obtaining the member 'iinfo' of a type (line 750)
    iinfo_367864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 31), np_367863, 'iinfo')
    # Calling iinfo(args, kwargs) (line 750)
    iinfo_call_result_367867 = invoke(stypy.reporting.localization.Localization(__file__, 750, 31), iinfo_367864, *[tp_367865], **kwargs_367866)
    
    # Obtaining the member 'max' of a type (line 750)
    max_367868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 31), iinfo_call_result_367867, 'max')
    # Applying the binary operator '%' (line 750)
    result_mod_367869 = python_operator(stypy.reporting.localization.Localization(__file__, 750, 25), '%', msg_367862, max_367868)
    
    # Processing the call keyword arguments (line 750)
    kwargs_367870 = {}
    # Getting the type of 'ValueError' (line 750)
    ValueError_367861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 750)
    ValueError_call_result_367871 = invoke(stypy.reporting.localization.Localization(__file__, 750, 14), ValueError_367861, *[result_mod_367869], **kwargs_367870)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 750, 8), ValueError_call_result_367871, 'raise parameter', BaseException)
    # SSA join for if statement (line 745)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 753):
    
    # Assigning a Call to a Name (line 753):
    
    # Call to int(...): (line 753)
    # Processing the call arguments (line 753)
    # Getting the type of 'density' (line 753)
    density_367873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 12), 'density', False)
    # Getting the type of 'm' (line 753)
    m_367874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 22), 'm', False)
    # Applying the binary operator '*' (line 753)
    result_mul_367875 = python_operator(stypy.reporting.localization.Localization(__file__, 753, 12), '*', density_367873, m_367874)
    
    # Getting the type of 'n' (line 753)
    n_367876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 26), 'n', False)
    # Applying the binary operator '*' (line 753)
    result_mul_367877 = python_operator(stypy.reporting.localization.Localization(__file__, 753, 24), '*', result_mul_367875, n_367876)
    
    # Processing the call keyword arguments (line 753)
    kwargs_367878 = {}
    # Getting the type of 'int' (line 753)
    int_367872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 8), 'int', False)
    # Calling int(args, kwargs) (line 753)
    int_call_result_367879 = invoke(stypy.reporting.localization.Localization(__file__, 753, 8), int_367872, *[result_mul_367877], **kwargs_367878)
    
    # Assigning a type to the variable 'k' (line 753)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 4), 'k', int_call_result_367879)
    
    # Type idiom detected: calculating its left and rigth part (line 755)
    # Getting the type of 'random_state' (line 755)
    random_state_367880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 7), 'random_state')
    # Getting the type of 'None' (line 755)
    None_367881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 23), 'None')
    
    (may_be_367882, more_types_in_union_367883) = may_be_none(random_state_367880, None_367881)

    if may_be_367882:

        if more_types_in_union_367883:
            # Runtime conditional SSA (line 755)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 756):
        
        # Assigning a Attribute to a Name (line 756):
        # Getting the type of 'np' (line 756)
        np_367884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 23), 'np')
        # Obtaining the member 'random' of a type (line 756)
        random_367885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 23), np_367884, 'random')
        # Assigning a type to the variable 'random_state' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'random_state', random_367885)

        if more_types_in_union_367883:
            # Runtime conditional SSA for else branch (line 755)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_367882) or more_types_in_union_367883):
        
        
        # Call to isinstance(...): (line 757)
        # Processing the call arguments (line 757)
        # Getting the type of 'random_state' (line 757)
        random_state_367887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 20), 'random_state', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 757)
        tuple_367888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 757)
        # Adding element type (line 757)
        # Getting the type of 'int' (line 757)
        int_367889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 35), 'int', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 35), tuple_367888, int_367889)
        # Adding element type (line 757)
        # Getting the type of 'np' (line 757)
        np_367890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 40), 'np', False)
        # Obtaining the member 'integer' of a type (line 757)
        integer_367891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 40), np_367890, 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 35), tuple_367888, integer_367891)
        
        # Processing the call keyword arguments (line 757)
        kwargs_367892 = {}
        # Getting the type of 'isinstance' (line 757)
        isinstance_367886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 9), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 757)
        isinstance_call_result_367893 = invoke(stypy.reporting.localization.Localization(__file__, 757, 9), isinstance_367886, *[random_state_367887, tuple_367888], **kwargs_367892)
        
        # Testing the type of an if condition (line 757)
        if_condition_367894 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 757, 9), isinstance_call_result_367893)
        # Assigning a type to the variable 'if_condition_367894' (line 757)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 9), 'if_condition_367894', if_condition_367894)
        # SSA begins for if statement (line 757)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 758):
        
        # Assigning a Call to a Name (line 758):
        
        # Call to RandomState(...): (line 758)
        # Processing the call arguments (line 758)
        # Getting the type of 'random_state' (line 758)
        random_state_367898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 45), 'random_state', False)
        # Processing the call keyword arguments (line 758)
        kwargs_367899 = {}
        # Getting the type of 'np' (line 758)
        np_367895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 23), 'np', False)
        # Obtaining the member 'random' of a type (line 758)
        random_367896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 23), np_367895, 'random')
        # Obtaining the member 'RandomState' of a type (line 758)
        RandomState_367897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 23), random_367896, 'RandomState')
        # Calling RandomState(args, kwargs) (line 758)
        RandomState_call_result_367900 = invoke(stypy.reporting.localization.Localization(__file__, 758, 23), RandomState_367897, *[random_state_367898], **kwargs_367899)
        
        # Assigning a type to the variable 'random_state' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'random_state', RandomState_call_result_367900)
        # SSA join for if statement (line 757)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_367882 and more_types_in_union_367883):
            # SSA join for if statement (line 755)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 759)
    # Getting the type of 'data_rvs' (line 759)
    data_rvs_367901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 7), 'data_rvs')
    # Getting the type of 'None' (line 759)
    None_367902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 19), 'None')
    
    (may_be_367903, more_types_in_union_367904) = may_be_none(data_rvs_367901, None_367902)

    if may_be_367903:

        if more_types_in_union_367904:
            # Runtime conditional SSA (line 759)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 760):
        
        # Assigning a Attribute to a Name (line 760):
        # Getting the type of 'random_state' (line 760)
        random_state_367905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 19), 'random_state')
        # Obtaining the member 'rand' of a type (line 760)
        rand_367906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 19), random_state_367905, 'rand')
        # Assigning a type to the variable 'data_rvs' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'data_rvs', rand_367906)

        if more_types_in_union_367904:
            # SSA join for if statement (line 759)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'mn' (line 763)
    mn_367907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 7), 'mn')
    int_367908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 12), 'int')
    # Getting the type of 'k' (line 763)
    k_367909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 14), 'k')
    # Applying the binary operator '*' (line 763)
    result_mul_367910 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 12), '*', int_367908, k_367909)
    
    # Applying the binary operator '<' (line 763)
    result_lt_367911 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 7), '<', mn_367907, result_mul_367910)
    
    # Testing the type of an if condition (line 763)
    if_condition_367912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 763, 4), result_lt_367911)
    # Assigning a type to the variable 'if_condition_367912' (line 763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 4), 'if_condition_367912', if_condition_367912)
    # SSA begins for if statement (line 763)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 764):
    
    # Assigning a Call to a Name (line 764):
    
    # Call to choice(...): (line 764)
    # Processing the call arguments (line 764)
    # Getting the type of 'mn' (line 764)
    mn_367915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 34), 'mn', False)
    # Processing the call keyword arguments (line 764)
    # Getting the type of 'k' (line 764)
    k_367916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 43), 'k', False)
    keyword_367917 = k_367916
    # Getting the type of 'False' (line 764)
    False_367918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 54), 'False', False)
    keyword_367919 = False_367918
    kwargs_367920 = {'replace': keyword_367919, 'size': keyword_367917}
    # Getting the type of 'random_state' (line 764)
    random_state_367913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 14), 'random_state', False)
    # Obtaining the member 'choice' of a type (line 764)
    choice_367914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 14), random_state_367913, 'choice')
    # Calling choice(args, kwargs) (line 764)
    choice_call_result_367921 = invoke(stypy.reporting.localization.Localization(__file__, 764, 14), choice_367914, *[mn_367915], **kwargs_367920)
    
    # Assigning a type to the variable 'ind' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 8), 'ind', choice_call_result_367921)
    # SSA branch for the else part of an if statement (line 763)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 766):
    
    # Assigning a Call to a Name (line 766):
    
    # Call to empty(...): (line 766)
    # Processing the call arguments (line 766)
    # Getting the type of 'k' (line 766)
    k_367924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 23), 'k', False)
    # Processing the call keyword arguments (line 766)
    # Getting the type of 'tp' (line 766)
    tp_367925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 32), 'tp', False)
    keyword_367926 = tp_367925
    kwargs_367927 = {'dtype': keyword_367926}
    # Getting the type of 'np' (line 766)
    np_367922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 14), 'np', False)
    # Obtaining the member 'empty' of a type (line 766)
    empty_367923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 14), np_367922, 'empty')
    # Calling empty(args, kwargs) (line 766)
    empty_call_result_367928 = invoke(stypy.reporting.localization.Localization(__file__, 766, 14), empty_367923, *[k_367924], **kwargs_367927)
    
    # Assigning a type to the variable 'ind' (line 766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'ind', empty_call_result_367928)
    
    # Assigning a Call to a Name (line 767):
    
    # Assigning a Call to a Name (line 767):
    
    # Call to set(...): (line 767)
    # Processing the call keyword arguments (line 767)
    kwargs_367930 = {}
    # Getting the type of 'set' (line 767)
    set_367929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 19), 'set', False)
    # Calling set(args, kwargs) (line 767)
    set_call_result_367931 = invoke(stypy.reporting.localization.Localization(__file__, 767, 19), set_367929, *[], **kwargs_367930)
    
    # Assigning a type to the variable 'selected' (line 767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 8), 'selected', set_call_result_367931)
    
    
    # Call to xrange(...): (line 768)
    # Processing the call arguments (line 768)
    # Getting the type of 'k' (line 768)
    k_367933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 24), 'k', False)
    # Processing the call keyword arguments (line 768)
    kwargs_367934 = {}
    # Getting the type of 'xrange' (line 768)
    xrange_367932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 17), 'xrange', False)
    # Calling xrange(args, kwargs) (line 768)
    xrange_call_result_367935 = invoke(stypy.reporting.localization.Localization(__file__, 768, 17), xrange_367932, *[k_367933], **kwargs_367934)
    
    # Testing the type of a for loop iterable (line 768)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 768, 8), xrange_call_result_367935)
    # Getting the type of the for loop variable (line 768)
    for_loop_var_367936 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 768, 8), xrange_call_result_367935)
    # Assigning a type to the variable 'i' (line 768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'i', for_loop_var_367936)
    # SSA begins for a for statement (line 768)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 769):
    
    # Assigning a Call to a Name (line 769):
    
    # Call to randint(...): (line 769)
    # Processing the call arguments (line 769)
    # Getting the type of 'mn' (line 769)
    mn_367939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 37), 'mn', False)
    # Processing the call keyword arguments (line 769)
    kwargs_367940 = {}
    # Getting the type of 'random_state' (line 769)
    random_state_367937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 16), 'random_state', False)
    # Obtaining the member 'randint' of a type (line 769)
    randint_367938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 16), random_state_367937, 'randint')
    # Calling randint(args, kwargs) (line 769)
    randint_call_result_367941 = invoke(stypy.reporting.localization.Localization(__file__, 769, 16), randint_367938, *[mn_367939], **kwargs_367940)
    
    # Assigning a type to the variable 'j' (line 769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 12), 'j', randint_call_result_367941)
    
    
    # Getting the type of 'j' (line 770)
    j_367942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 18), 'j')
    # Getting the type of 'selected' (line 770)
    selected_367943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 23), 'selected')
    # Applying the binary operator 'in' (line 770)
    result_contains_367944 = python_operator(stypy.reporting.localization.Localization(__file__, 770, 18), 'in', j_367942, selected_367943)
    
    # Testing the type of an if condition (line 770)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 770, 12), result_contains_367944)
    # SSA begins for while statement (line 770)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 771):
    
    # Assigning a Call to a Name (line 771):
    
    # Call to randint(...): (line 771)
    # Processing the call arguments (line 771)
    # Getting the type of 'mn' (line 771)
    mn_367947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 41), 'mn', False)
    # Processing the call keyword arguments (line 771)
    kwargs_367948 = {}
    # Getting the type of 'random_state' (line 771)
    random_state_367945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 20), 'random_state', False)
    # Obtaining the member 'randint' of a type (line 771)
    randint_367946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 20), random_state_367945, 'randint')
    # Calling randint(args, kwargs) (line 771)
    randint_call_result_367949 = invoke(stypy.reporting.localization.Localization(__file__, 771, 20), randint_367946, *[mn_367947], **kwargs_367948)
    
    # Assigning a type to the variable 'j' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 16), 'j', randint_call_result_367949)
    # SSA join for while statement (line 770)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to add(...): (line 772)
    # Processing the call arguments (line 772)
    # Getting the type of 'j' (line 772)
    j_367952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 25), 'j', False)
    # Processing the call keyword arguments (line 772)
    kwargs_367953 = {}
    # Getting the type of 'selected' (line 772)
    selected_367950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 12), 'selected', False)
    # Obtaining the member 'add' of a type (line 772)
    add_367951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 12), selected_367950, 'add')
    # Calling add(args, kwargs) (line 772)
    add_call_result_367954 = invoke(stypy.reporting.localization.Localization(__file__, 772, 12), add_367951, *[j_367952], **kwargs_367953)
    
    
    # Assigning a Name to a Subscript (line 773):
    
    # Assigning a Name to a Subscript (line 773):
    # Getting the type of 'j' (line 773)
    j_367955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 21), 'j')
    # Getting the type of 'ind' (line 773)
    ind_367956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 12), 'ind')
    # Getting the type of 'i' (line 773)
    i_367957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 16), 'i')
    # Storing an element on a container (line 773)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 12), ind_367956, (i_367957, j_367955))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 763)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 775):
    
    # Assigning a Call to a Name (line 775):
    
    # Call to astype(...): (line 775)
    # Processing the call arguments (line 775)
    # Getting the type of 'tp' (line 775)
    tp_367968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 38), 'tp', False)
    # Processing the call keyword arguments (line 775)
    kwargs_367969 = {}
    
    # Call to floor(...): (line 775)
    # Processing the call arguments (line 775)
    # Getting the type of 'ind' (line 775)
    ind_367960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 17), 'ind', False)
    float_367961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 23), 'float')
    # Applying the binary operator '*' (line 775)
    result_mul_367962 = python_operator(stypy.reporting.localization.Localization(__file__, 775, 17), '*', ind_367960, float_367961)
    
    # Getting the type of 'm' (line 775)
    m_367963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 28), 'm', False)
    # Applying the binary operator 'div' (line 775)
    result_div_367964 = python_operator(stypy.reporting.localization.Localization(__file__, 775, 26), 'div', result_mul_367962, m_367963)
    
    # Processing the call keyword arguments (line 775)
    kwargs_367965 = {}
    # Getting the type of 'np' (line 775)
    np_367958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'np', False)
    # Obtaining the member 'floor' of a type (line 775)
    floor_367959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 8), np_367958, 'floor')
    # Calling floor(args, kwargs) (line 775)
    floor_call_result_367966 = invoke(stypy.reporting.localization.Localization(__file__, 775, 8), floor_367959, *[result_div_367964], **kwargs_367965)
    
    # Obtaining the member 'astype' of a type (line 775)
    astype_367967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 8), floor_call_result_367966, 'astype')
    # Calling astype(args, kwargs) (line 775)
    astype_call_result_367970 = invoke(stypy.reporting.localization.Localization(__file__, 775, 8), astype_367967, *[tp_367968], **kwargs_367969)
    
    # Assigning a type to the variable 'j' (line 775)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 4), 'j', astype_call_result_367970)
    
    # Assigning a Call to a Name (line 776):
    
    # Assigning a Call to a Name (line 776):
    
    # Call to astype(...): (line 776)
    # Processing the call arguments (line 776)
    # Getting the type of 'tp' (line 776)
    tp_367977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 29), 'tp', False)
    # Processing the call keyword arguments (line 776)
    kwargs_367978 = {}
    # Getting the type of 'ind' (line 776)
    ind_367971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 9), 'ind', False)
    # Getting the type of 'j' (line 776)
    j_367972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 15), 'j', False)
    # Getting the type of 'm' (line 776)
    m_367973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 19), 'm', False)
    # Applying the binary operator '*' (line 776)
    result_mul_367974 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 15), '*', j_367972, m_367973)
    
    # Applying the binary operator '-' (line 776)
    result_sub_367975 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 9), '-', ind_367971, result_mul_367974)
    
    # Obtaining the member 'astype' of a type (line 776)
    astype_367976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 9), result_sub_367975, 'astype')
    # Calling astype(args, kwargs) (line 776)
    astype_call_result_367979 = invoke(stypy.reporting.localization.Localization(__file__, 776, 9), astype_367976, *[tp_367977], **kwargs_367978)
    
    # Assigning a type to the variable 'i' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'i', astype_call_result_367979)
    
    # Assigning a Call to a Name (line 777):
    
    # Assigning a Call to a Name (line 777):
    
    # Call to astype(...): (line 777)
    # Processing the call arguments (line 777)
    # Getting the type of 'dtype' (line 777)
    dtype_367985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 30), 'dtype', False)
    # Processing the call keyword arguments (line 777)
    kwargs_367986 = {}
    
    # Call to data_rvs(...): (line 777)
    # Processing the call arguments (line 777)
    # Getting the type of 'k' (line 777)
    k_367981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 20), 'k', False)
    # Processing the call keyword arguments (line 777)
    kwargs_367982 = {}
    # Getting the type of 'data_rvs' (line 777)
    data_rvs_367980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 11), 'data_rvs', False)
    # Calling data_rvs(args, kwargs) (line 777)
    data_rvs_call_result_367983 = invoke(stypy.reporting.localization.Localization(__file__, 777, 11), data_rvs_367980, *[k_367981], **kwargs_367982)
    
    # Obtaining the member 'astype' of a type (line 777)
    astype_367984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 11), data_rvs_call_result_367983, 'astype')
    # Calling astype(args, kwargs) (line 777)
    astype_call_result_367987 = invoke(stypy.reporting.localization.Localization(__file__, 777, 11), astype_367984, *[dtype_367985], **kwargs_367986)
    
    # Assigning a type to the variable 'vals' (line 777)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 4), 'vals', astype_call_result_367987)
    
    # Call to asformat(...): (line 778)
    # Processing the call arguments (line 778)
    # Getting the type of 'format' (line 778)
    format_368001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 61), 'format', False)
    # Processing the call keyword arguments (line 778)
    kwargs_368002 = {}
    
    # Call to coo_matrix(...): (line 778)
    # Processing the call arguments (line 778)
    
    # Obtaining an instance of the builtin type 'tuple' (line 778)
    tuple_367989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 778)
    # Adding element type (line 778)
    # Getting the type of 'vals' (line 778)
    vals_367990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 23), 'vals', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 23), tuple_367989, vals_367990)
    # Adding element type (line 778)
    
    # Obtaining an instance of the builtin type 'tuple' (line 778)
    tuple_367991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 778)
    # Adding element type (line 778)
    # Getting the type of 'i' (line 778)
    i_367992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 30), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 30), tuple_367991, i_367992)
    # Adding element type (line 778)
    # Getting the type of 'j' (line 778)
    j_367993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 33), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 30), tuple_367991, j_367993)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 23), tuple_367989, tuple_367991)
    
    # Processing the call keyword arguments (line 778)
    
    # Obtaining an instance of the builtin type 'tuple' (line 778)
    tuple_367994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 778)
    # Adding element type (line 778)
    # Getting the type of 'm' (line 778)
    m_367995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 45), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 45), tuple_367994, m_367995)
    # Adding element type (line 778)
    # Getting the type of 'n' (line 778)
    n_367996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 48), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 45), tuple_367994, n_367996)
    
    keyword_367997 = tuple_367994
    kwargs_367998 = {'shape': keyword_367997}
    # Getting the type of 'coo_matrix' (line 778)
    coo_matrix_367988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 11), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 778)
    coo_matrix_call_result_367999 = invoke(stypy.reporting.localization.Localization(__file__, 778, 11), coo_matrix_367988, *[tuple_367989], **kwargs_367998)
    
    # Obtaining the member 'asformat' of a type (line 778)
    asformat_368000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 11), coo_matrix_call_result_367999, 'asformat')
    # Calling asformat(args, kwargs) (line 778)
    asformat_call_result_368003 = invoke(stypy.reporting.localization.Localization(__file__, 778, 11), asformat_368000, *[format_368001], **kwargs_368002)
    
    # Assigning a type to the variable 'stypy_return_type' (line 778)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 4), 'stypy_return_type', asformat_call_result_368003)
    
    # ################# End of 'random(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'random' in the type store
    # Getting the type of 'stypy_return_type' (line 679)
    stypy_return_type_368004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_368004)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'random'
    return stypy_return_type_368004

# Assigning a type to the variable 'random' (line 679)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 0), 'random', random)

@norecursion
def rand(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_368005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 23), 'float')
    str_368006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 36), 'str', 'coo')
    # Getting the type of 'None' (line 781)
    None_368007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 49), 'None')
    # Getting the type of 'None' (line 781)
    None_368008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 68), 'None')
    defaults = [float_368005, str_368006, None_368007, None_368008]
    # Create a new context for function 'rand'
    module_type_store = module_type_store.open_function_context('rand', 781, 0, False)
    
    # Passed parameters checking function
    rand.stypy_localization = localization
    rand.stypy_type_of_self = None
    rand.stypy_type_store = module_type_store
    rand.stypy_function_name = 'rand'
    rand.stypy_param_names_list = ['m', 'n', 'density', 'format', 'dtype', 'random_state']
    rand.stypy_varargs_param_name = None
    rand.stypy_kwargs_param_name = None
    rand.stypy_call_defaults = defaults
    rand.stypy_call_varargs = varargs
    rand.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rand', ['m', 'n', 'density', 'format', 'dtype', 'random_state'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rand', localization, ['m', 'n', 'density', 'format', 'dtype', 'random_state'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rand(...)' code ##################

    str_368009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, (-1)), 'str', 'Generate a sparse matrix of the given shape and density with uniformly\n    distributed values.\n\n    Parameters\n    ----------\n    m, n : int\n        shape of the matrix\n    density : real, optional\n        density of the generated matrix: density equal to one means a full\n        matrix, density of 0 means a matrix with no non-zero items.\n    format : str, optional\n        sparse matrix format.\n    dtype : dtype, optional\n        type of the returned matrix values.\n    random_state : {numpy.random.RandomState, int}, optional\n        Random number generator or random seed. If not given, the singleton\n        numpy.random will be used.\n\n    Returns\n    -------\n    res : sparse matrix\n\n    Notes\n    -----\n    Only float types are supported for now.\n\n    See Also\n    --------\n    scipy.sparse.random : Similar function that allows a user-specified random\n        data source.\n\n    Examples\n    --------\n    >>> from scipy.sparse import rand\n    >>> matrix = rand(3, 4, density=0.25, format="csr", random_state=42)\n    >>> matrix\n    <3x4 sparse matrix of type \'<class \'numpy.float64\'>\'\n       with 3 stored elements in Compressed Sparse Row format>\n    >>> matrix.todense()\n    matrix([[ 0.        ,  0.59685016,  0.779691  ,  0.        ],\n            [ 0.        ,  0.        ,  0.        ,  0.44583275],\n            [ 0.        ,  0.        ,  0.        ,  0.        ]])\n    ')
    
    # Call to random(...): (line 825)
    # Processing the call arguments (line 825)
    # Getting the type of 'm' (line 825)
    m_368011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 18), 'm', False)
    # Getting the type of 'n' (line 825)
    n_368012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 21), 'n', False)
    # Getting the type of 'density' (line 825)
    density_368013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 24), 'density', False)
    # Getting the type of 'format' (line 825)
    format_368014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 33), 'format', False)
    # Getting the type of 'dtype' (line 825)
    dtype_368015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 41), 'dtype', False)
    # Getting the type of 'random_state' (line 825)
    random_state_368016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 48), 'random_state', False)
    # Processing the call keyword arguments (line 825)
    kwargs_368017 = {}
    # Getting the type of 'random' (line 825)
    random_368010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 11), 'random', False)
    # Calling random(args, kwargs) (line 825)
    random_call_result_368018 = invoke(stypy.reporting.localization.Localization(__file__, 825, 11), random_368010, *[m_368011, n_368012, density_368013, format_368014, dtype_368015, random_state_368016], **kwargs_368017)
    
    # Assigning a type to the variable 'stypy_return_type' (line 825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 4), 'stypy_return_type', random_call_result_368018)
    
    # ################# End of 'rand(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rand' in the type store
    # Getting the type of 'stypy_return_type' (line 781)
    stypy_return_type_368019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_368019)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rand'
    return stypy_return_type_368019

# Assigning a type to the variable 'rand' (line 781)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 0), 'rand', rand)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
