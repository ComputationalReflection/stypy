
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Compressed Block Sparse Row matrix format'''
2: from __future__ import division, print_function, absolute_import
3: 
4: 
5: __docformat__ = "restructuredtext en"
6: 
7: __all__ = ['bsr_matrix', 'isspmatrix_bsr']
8: 
9: from warnings import warn
10: 
11: import numpy as np
12: 
13: from .data import _data_matrix, _minmax_mixin
14: from .compressed import _cs_matrix
15: from .base import isspmatrix, _formats, spmatrix
16: from .sputils import isshape, getdtype, to_native, upcast, get_index_dtype
17: from . import _sparsetools
18: from ._sparsetools import (bsr_matvec, bsr_matvecs, csr_matmat_pass1,
19:                            bsr_matmat_pass2, bsr_transpose, bsr_sort_indices)
20: 
21: 
22: class bsr_matrix(_cs_matrix, _minmax_mixin):
23:     '''Block Sparse Row matrix
24: 
25:     This can be instantiated in several ways:
26:         bsr_matrix(D, [blocksize=(R,C)])
27:             where D is a dense matrix or 2-D ndarray.
28: 
29:         bsr_matrix(S, [blocksize=(R,C)])
30:             with another sparse matrix S (equivalent to S.tobsr())
31: 
32:         bsr_matrix((M, N), [blocksize=(R,C), dtype])
33:             to construct an empty matrix with shape (M, N)
34:             dtype is optional, defaulting to dtype='d'.
35: 
36:         bsr_matrix((data, ij), [blocksize=(R,C), shape=(M, N)])
37:             where ``data`` and ``ij`` satisfy ``a[ij[0, k], ij[1, k]] = data[k]``
38: 
39:         bsr_matrix((data, indices, indptr), [shape=(M, N)])
40:             is the standard BSR representation where the block column
41:             indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]``
42:             and their corresponding block values are stored in
43:             ``data[ indptr[i]: indptr[i+1] ]``.  If the shape parameter is not
44:             supplied, the matrix dimensions are inferred from the index arrays.
45: 
46:     Attributes
47:     ----------
48:     dtype : dtype
49:         Data type of the matrix
50:     shape : 2-tuple
51:         Shape of the matrix
52:     ndim : int
53:         Number of dimensions (this is always 2)
54:     nnz
55:         Number of nonzero elements
56:     data
57:         Data array of the matrix
58:     indices
59:         BSR format index array
60:     indptr
61:         BSR format index pointer array
62:     blocksize
63:         Block size of the matrix
64:     has_sorted_indices
65:         Whether indices are sorted
66: 
67:     Notes
68:     -----
69:     Sparse matrices can be used in arithmetic operations: they support
70:     addition, subtraction, multiplication, division, and matrix power.
71: 
72:     **Summary of BSR format**
73: 
74:     The Block Compressed Row (BSR) format is very similar to the Compressed
75:     Sparse Row (CSR) format.  BSR is appropriate for sparse matrices with dense
76:     sub matrices like the last example below.  Block matrices often arise in
77:     vector-valued finite element discretizations.  In such cases, BSR is
78:     considerably more efficient than CSR and CSC for many sparse arithmetic
79:     operations.
80: 
81:     **Blocksize**
82: 
83:     The blocksize (R,C) must evenly divide the shape of the matrix (M,N).
84:     That is, R and C must satisfy the relationship ``M % R = 0`` and
85:     ``N % C = 0``.
86: 
87:     If no blocksize is specified, a simple heuristic is applied to determine
88:     an appropriate blocksize.
89: 
90:     Examples
91:     --------
92:     >>> from scipy.sparse import bsr_matrix
93:     >>> bsr_matrix((3, 4), dtype=np.int8).toarray()
94:     array([[0, 0, 0, 0],
95:            [0, 0, 0, 0],
96:            [0, 0, 0, 0]], dtype=int8)
97: 
98:     >>> row = np.array([0, 0, 1, 2, 2, 2])
99:     >>> col = np.array([0, 2, 2, 0, 1, 2])
100:     >>> data = np.array([1, 2, 3 ,4, 5, 6])
101:     >>> bsr_matrix((data, (row, col)), shape=(3, 3)).toarray()
102:     array([[1, 0, 2],
103:            [0, 0, 3],
104:            [4, 5, 6]])
105: 
106:     >>> indptr = np.array([0, 2, 3, 6])
107:     >>> indices = np.array([0, 2, 2, 0, 1, 2])
108:     >>> data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
109:     >>> bsr_matrix((data,indices,indptr), shape=(6, 6)).toarray()
110:     array([[1, 1, 0, 0, 2, 2],
111:            [1, 1, 0, 0, 2, 2],
112:            [0, 0, 0, 0, 3, 3],
113:            [0, 0, 0, 0, 3, 3],
114:            [4, 4, 5, 5, 6, 6],
115:            [4, 4, 5, 5, 6, 6]])
116: 
117:     '''
118:     format = 'bsr'
119: 
120:     def __init__(self, arg1, shape=None, dtype=None, copy=False, blocksize=None):
121:         _data_matrix.__init__(self)
122: 
123:         if isspmatrix(arg1):
124:             if isspmatrix_bsr(arg1) and copy:
125:                 arg1 = arg1.copy()
126:             else:
127:                 arg1 = arg1.tobsr(blocksize=blocksize)
128:             self._set_self(arg1)
129: 
130:         elif isinstance(arg1,tuple):
131:             if isshape(arg1):
132:                 # it's a tuple of matrix dimensions (M,N)
133:                 self.shape = arg1
134:                 M,N = self.shape
135:                 # process blocksize
136:                 if blocksize is None:
137:                     blocksize = (1,1)
138:                 else:
139:                     if not isshape(blocksize):
140:                         raise ValueError('invalid blocksize=%s' % blocksize)
141:                     blocksize = tuple(blocksize)
142:                 self.data = np.zeros((0,) + blocksize, getdtype(dtype, default=float))
143: 
144:                 R,C = blocksize
145:                 if (M % R) != 0 or (N % C) != 0:
146:                     raise ValueError('shape must be multiple of blocksize')
147: 
148:                 # Select index dtype large enough to pass array and
149:                 # scalar parameters to sparsetools
150:                 idx_dtype = get_index_dtype(maxval=max(M//R, N//C, R, C))
151:                 self.indices = np.zeros(0, dtype=idx_dtype)
152:                 self.indptr = np.zeros(M//R + 1, dtype=idx_dtype)
153: 
154:             elif len(arg1) == 2:
155:                 # (data,(row,col)) format
156:                 from .coo import coo_matrix
157:                 self._set_self(coo_matrix(arg1, dtype=dtype).tobsr(blocksize=blocksize))
158: 
159:             elif len(arg1) == 3:
160:                 # (data,indices,indptr) format
161:                 (data, indices, indptr) = arg1
162: 
163:                 # Select index dtype large enough to pass array and
164:                 # scalar parameters to sparsetools
165:                 maxval = 1
166:                 if shape is not None:
167:                     maxval = max(shape)
168:                 if blocksize is not None:
169:                     maxval = max(maxval, max(blocksize))
170:                 idx_dtype = get_index_dtype((indices, indptr), maxval=maxval, check_contents=True)
171: 
172:                 self.indices = np.array(indices, copy=copy, dtype=idx_dtype)
173:                 self.indptr = np.array(indptr, copy=copy, dtype=idx_dtype)
174:                 self.data = np.array(data, copy=copy, dtype=getdtype(dtype, data))
175:             else:
176:                 raise ValueError('unrecognized bsr_matrix constructor usage')
177:         else:
178:             # must be dense
179:             try:
180:                 arg1 = np.asarray(arg1)
181:             except:
182:                 raise ValueError("unrecognized form for"
183:                         " %s_matrix constructor" % self.format)
184:             from .coo import coo_matrix
185:             arg1 = coo_matrix(arg1, dtype=dtype).tobsr(blocksize=blocksize)
186:             self._set_self(arg1)
187: 
188:         if shape is not None:
189:             self.shape = shape   # spmatrix will check for errors
190:         else:
191:             if self.shape is None:
192:                 # shape not already set, try to infer dimensions
193:                 try:
194:                     M = len(self.indptr) - 1
195:                     N = self.indices.max() + 1
196:                 except:
197:                     raise ValueError('unable to infer matrix dimensions')
198:                 else:
199:                     R,C = self.blocksize
200:                     self.shape = (M*R,N*C)
201: 
202:         if self.shape is None:
203:             if shape is None:
204:                 # TODO infer shape here
205:                 raise ValueError('need to infer shape')
206:             else:
207:                 self.shape = shape
208: 
209:         if dtype is not None:
210:             self.data = self.data.astype(dtype)
211: 
212:         self.check_format(full_check=False)
213: 
214:     def check_format(self, full_check=True):
215:         '''check whether the matrix format is valid
216: 
217:             *Parameters*:
218:                 full_check:
219:                     True  - rigorous check, O(N) operations : default
220:                     False - basic check, O(1) operations
221: 
222:         '''
223:         M,N = self.shape
224:         R,C = self.blocksize
225: 
226:         # index arrays should have integer data types
227:         if self.indptr.dtype.kind != 'i':
228:             warn("indptr array has non-integer dtype (%s)"
229:                     % self.indptr.dtype.name)
230:         if self.indices.dtype.kind != 'i':
231:             warn("indices array has non-integer dtype (%s)"
232:                     % self.indices.dtype.name)
233: 
234:         idx_dtype = get_index_dtype((self.indices, self.indptr))
235:         self.indptr = np.asarray(self.indptr, dtype=idx_dtype)
236:         self.indices = np.asarray(self.indices, dtype=idx_dtype)
237:         self.data = to_native(self.data)
238: 
239:         # check array shapes
240:         if self.indices.ndim != 1 or self.indptr.ndim != 1:
241:             raise ValueError("indices, and indptr should be 1-D")
242:         if self.data.ndim != 3:
243:             raise ValueError("data should be 3-D")
244: 
245:         # check index pointer
246:         if (len(self.indptr) != M//R + 1):
247:             raise ValueError("index pointer size (%d) should be (%d)" %
248:                                 (len(self.indptr), M//R + 1))
249:         if (self.indptr[0] != 0):
250:             raise ValueError("index pointer should start with 0")
251: 
252:         # check index and data arrays
253:         if (len(self.indices) != len(self.data)):
254:             raise ValueError("indices and data should have the same size")
255:         if (self.indptr[-1] > len(self.indices)):
256:             raise ValueError("Last value of index pointer should be less than "
257:                                 "the size of index and data arrays")
258: 
259:         self.prune()
260: 
261:         if full_check:
262:             # check format validity (more expensive)
263:             if self.nnz > 0:
264:                 if self.indices.max() >= N//C:
265:                     raise ValueError("column index values must be < %d (now max %d)" % (N//C, self.indices.max()))
266:                 if self.indices.min() < 0:
267:                     raise ValueError("column index values must be >= 0")
268:                 if np.diff(self.indptr).min() < 0:
269:                     raise ValueError("index pointer values must form a "
270:                                         "non-decreasing sequence")
271: 
272:         # if not self.has_sorted_indices():
273:         #    warn('Indices were not in sorted order. Sorting indices.')
274:         #    self.sort_indices(check_first=False)
275: 
276:     def _get_blocksize(self):
277:         return self.data.shape[1:]
278:     blocksize = property(fget=_get_blocksize)
279: 
280:     def getnnz(self, axis=None):
281:         if axis is not None:
282:             raise NotImplementedError("getnnz over an axis is not implemented "
283:                                       "for BSR format")
284:         R,C = self.blocksize
285:         return int(self.indptr[-1] * R * C)
286: 
287:     getnnz.__doc__ = spmatrix.getnnz.__doc__
288: 
289:     def __repr__(self):
290:         format = _formats[self.getformat()][1]
291:         return ("<%dx%d sparse matrix of type '%s'\n"
292:                 "\twith %d stored elements (blocksize = %dx%d) in %s format>" %
293:                 (self.shape + (self.dtype.type, self.nnz) + self.blocksize +
294:                  (format,)))
295: 
296:     def diagonal(self, k=0):
297:         rows, cols = self.shape
298:         if k <= -rows or k >= cols:
299:             raise ValueError("k exceeds matrix dimensions")
300:         R, C = self.blocksize
301:         y = np.zeros(min(rows + min(k, 0), cols - max(k, 0)),
302:                      dtype=upcast(self.dtype))
303:         _sparsetools.bsr_diagonal(k, rows // R, cols // C, R, C,
304:                                   self.indptr, self.indices,
305:                                   np.ravel(self.data), y)
306:         return y
307: 
308:     diagonal.__doc__ = spmatrix.diagonal.__doc__
309: 
310:     ##########################
311:     # NotImplemented methods #
312:     ##########################
313: 
314:     def __getitem__(self,key):
315:         raise NotImplementedError
316: 
317:     def __setitem__(self,key,val):
318:         raise NotImplementedError
319: 
320:     ######################
321:     # Arithmetic methods #
322:     ######################
323: 
324:     @np.deprecate(message="BSR matvec is deprecated in scipy 0.19.0. "
325:                           "Use * operator instead.")
326:     def matvec(self, other):
327:         '''Multiply matrix by vector.'''
328:         return self * other
329: 
330:     @np.deprecate(message="BSR matmat is deprecated in scipy 0.19.0. "
331:                           "Use * operator instead.")
332:     def matmat(self, other):
333:         '''Multiply this sparse matrix by other matrix.'''
334:         return self * other
335: 
336:     def _add_dense(self, other):
337:         return self.tocoo(copy=False)._add_dense(other)
338: 
339:     def _mul_vector(self, other):
340:         M,N = self.shape
341:         R,C = self.blocksize
342: 
343:         result = np.zeros(self.shape[0], dtype=upcast(self.dtype, other.dtype))
344: 
345:         bsr_matvec(M//R, N//C, R, C,
346:             self.indptr, self.indices, self.data.ravel(),
347:             other, result)
348: 
349:         return result
350: 
351:     def _mul_multivector(self,other):
352:         R,C = self.blocksize
353:         M,N = self.shape
354:         n_vecs = other.shape[1]  # number of column vectors
355: 
356:         result = np.zeros((M,n_vecs), dtype=upcast(self.dtype,other.dtype))
357: 
358:         bsr_matvecs(M//R, N//C, n_vecs, R, C,
359:                 self.indptr, self.indices, self.data.ravel(),
360:                 other.ravel(), result.ravel())
361: 
362:         return result
363: 
364:     def _mul_sparse_matrix(self, other):
365:         M, K1 = self.shape
366:         K2, N = other.shape
367: 
368:         R,n = self.blocksize
369: 
370:         # convert to this format
371:         if isspmatrix_bsr(other):
372:             C = other.blocksize[1]
373:         else:
374:             C = 1
375: 
376:         from .csr import isspmatrix_csr
377: 
378:         if isspmatrix_csr(other) and n == 1:
379:             other = other.tobsr(blocksize=(n,C), copy=False)  # lightweight conversion
380:         else:
381:             other = other.tobsr(blocksize=(n,C))
382: 
383:         idx_dtype = get_index_dtype((self.indptr, self.indices,
384:                                      other.indptr, other.indices),
385:                                     maxval=(M//R)*(N//C))
386:         indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
387: 
388:         csr_matmat_pass1(M//R, N//C,
389:                          self.indptr.astype(idx_dtype),
390:                          self.indices.astype(idx_dtype),
391:                          other.indptr.astype(idx_dtype),
392:                          other.indices.astype(idx_dtype),
393:                          indptr)
394: 
395:         bnnz = indptr[-1]
396: 
397:         idx_dtype = get_index_dtype((self.indptr, self.indices,
398:                                      other.indptr, other.indices),
399:                                     maxval=bnnz)
400:         indptr = indptr.astype(idx_dtype)
401:         indices = np.empty(bnnz, dtype=idx_dtype)
402:         data = np.empty(R*C*bnnz, dtype=upcast(self.dtype,other.dtype))
403: 
404:         bsr_matmat_pass2(M//R, N//C, R, C, n,
405:                          self.indptr.astype(idx_dtype),
406:                          self.indices.astype(idx_dtype),
407:                          np.ravel(self.data),
408:                          other.indptr.astype(idx_dtype),
409:                          other.indices.astype(idx_dtype),
410:                          np.ravel(other.data),
411:                          indptr,
412:                          indices,
413:                          data)
414: 
415:         data = data.reshape(-1,R,C)
416: 
417:         # TODO eliminate zeros
418: 
419:         return bsr_matrix((data,indices,indptr),shape=(M,N),blocksize=(R,C))
420: 
421:     ######################
422:     # Conversion methods #
423:     ######################
424: 
425:     def tobsr(self, blocksize=None, copy=False):
426:         '''Convert this matrix into Block Sparse Row Format.
427: 
428:         With copy=False, the data/indices may be shared between this
429:         matrix and the resultant bsr_matrix.
430: 
431:         If blocksize=(R, C) is provided, it will be used for determining
432:         block size of the bsr_matrix.
433:         '''
434:         if blocksize not in [None, self.blocksize]:
435:             return self.tocsr().tobsr(blocksize=blocksize)
436:         if copy:
437:             return self.copy()
438:         else:
439:             return self
440: 
441:     def tocsr(self, copy=False):
442:         return self.tocoo(copy=False).tocsr(copy=copy)
443:         # TODO make this more efficient
444: 
445:     tocsr.__doc__ = spmatrix.tocsr.__doc__
446: 
447:     def tocsc(self, copy=False):
448:         return self.tocoo(copy=False).tocsc(copy=copy)
449: 
450:     tocsc.__doc__ = spmatrix.tocsc.__doc__
451: 
452:     def tocoo(self, copy=True):
453:         '''Convert this matrix to COOrdinate format.
454: 
455:         When copy=False the data array will be shared between
456:         this matrix and the resultant coo_matrix.
457:         '''
458: 
459:         M,N = self.shape
460:         R,C = self.blocksize
461: 
462:         indptr_diff = np.diff(self.indptr)
463:         if indptr_diff.dtype.itemsize > np.dtype(np.intp).itemsize:
464:             # Check for potential overflow
465:             indptr_diff_limited = indptr_diff.astype(np.intp)
466:             if np.any(indptr_diff_limited != indptr_diff):
467:                 raise ValueError("Matrix too big to convert")
468:             indptr_diff = indptr_diff_limited
469: 
470:         row = (R * np.arange(M//R)).repeat(indptr_diff)
471:         row = row.repeat(R*C).reshape(-1,R,C)
472:         row += np.tile(np.arange(R).reshape(-1,1), (1,C))
473:         row = row.reshape(-1)
474: 
475:         col = (C * self.indices).repeat(R*C).reshape(-1,R,C)
476:         col += np.tile(np.arange(C), (R,1))
477:         col = col.reshape(-1)
478: 
479:         data = self.data.reshape(-1)
480: 
481:         if copy:
482:             data = data.copy()
483: 
484:         from .coo import coo_matrix
485:         return coo_matrix((data,(row,col)), shape=self.shape)
486: 
487:     def toarray(self, order=None, out=None):
488:         return self.tocoo(copy=False).toarray(order=order, out=out)
489: 
490:     toarray.__doc__ = spmatrix.toarray.__doc__
491: 
492:     def transpose(self, axes=None, copy=False):
493:         if axes is not None:
494:             raise ValueError(("Sparse matrices do not support "
495:                               "an 'axes' parameter because swapping "
496:                               "dimensions is the only logical permutation."))
497: 
498:         R, C = self.blocksize
499:         M, N = self.shape
500:         NBLK = self.nnz//(R*C)
501: 
502:         if self.nnz == 0:
503:             return bsr_matrix((N, M), blocksize=(C, R),
504:                               dtype=self.dtype, copy=copy)
505: 
506:         indptr = np.empty(N//C + 1, dtype=self.indptr.dtype)
507:         indices = np.empty(NBLK, dtype=self.indices.dtype)
508:         data = np.empty((NBLK, C, R), dtype=self.data.dtype)
509: 
510:         bsr_transpose(M//R, N//C, R, C,
511:                       self.indptr, self.indices, self.data.ravel(),
512:                       indptr, indices, data.ravel())
513: 
514:         return bsr_matrix((data, indices, indptr),
515:                           shape=(N, M), copy=copy)
516: 
517:     transpose.__doc__ = spmatrix.transpose.__doc__
518: 
519:     ##############################################################
520:     # methods that examine or modify the internal data structure #
521:     ##############################################################
522: 
523:     def eliminate_zeros(self):
524:         '''Remove zero elements in-place.'''
525:         R,C = self.blocksize
526:         M,N = self.shape
527: 
528:         mask = (self.data != 0).reshape(-1,R*C).sum(axis=1)  # nonzero blocks
529: 
530:         nonzero_blocks = mask.nonzero()[0]
531: 
532:         if len(nonzero_blocks) == 0:
533:             return  # nothing to do
534: 
535:         self.data[:len(nonzero_blocks)] = self.data[nonzero_blocks]
536: 
537:         # modifies self.indptr and self.indices *in place*
538:         _sparsetools.csr_eliminate_zeros(M//R, N//C, self.indptr,
539:                                          self.indices, mask)
540:         self.prune()
541: 
542:     def sum_duplicates(self):
543:         '''Eliminate duplicate matrix entries by adding them together
544: 
545:         The is an *in place* operation
546:         '''
547:         if self.has_canonical_format:
548:             return
549:         self.sort_indices()
550:         R, C = self.blocksize
551:         M, N = self.shape
552: 
553:         # port of _sparsetools.csr_sum_duplicates
554:         n_row = M // R
555:         nnz = 0
556:         row_end = 0
557:         for i in range(n_row):
558:             jj = row_end
559:             row_end = self.indptr[i+1]
560:             while jj < row_end:
561:                 j = self.indices[jj]
562:                 x = self.data[jj]
563:                 jj += 1
564:                 while jj < row_end and self.indices[jj] == j:
565:                     x += self.data[jj]
566:                     jj += 1
567:                 self.indices[nnz] = j
568:                 self.data[nnz] = x
569:                 nnz += 1
570:             self.indptr[i+1] = nnz
571: 
572:         self.prune()  # nnz may have changed
573:         self.has_canonical_format = True
574: 
575:     def sort_indices(self):
576:         '''Sort the indices of this matrix *in place*
577:         '''
578:         if self.has_sorted_indices:
579:             return
580: 
581:         R,C = self.blocksize
582:         M,N = self.shape
583: 
584:         bsr_sort_indices(M//R, N//C, R, C, self.indptr, self.indices, self.data.ravel())
585: 
586:         self.has_sorted_indices = True
587: 
588:     def prune(self):
589:         ''' Remove empty space after all non-zero elements.
590:         '''
591: 
592:         R,C = self.blocksize
593:         M,N = self.shape
594: 
595:         if len(self.indptr) != M//R + 1:
596:             raise ValueError("index pointer has invalid length")
597: 
598:         bnnz = self.indptr[-1]
599: 
600:         if len(self.indices) < bnnz:
601:             raise ValueError("indices array has too few elements")
602:         if len(self.data) < bnnz:
603:             raise ValueError("data array has too few elements")
604: 
605:         self.data = self.data[:bnnz]
606:         self.indices = self.indices[:bnnz]
607: 
608:     # utility functions
609:     def _binopt(self, other, op, in_shape=None, out_shape=None):
610:         '''Apply the binary operation fn to two sparse matrices.'''
611: 
612:         # Ideally we'd take the GCDs of the blocksize dimensions
613:         # and explode self and other to match.
614:         other = self.__class__(other, blocksize=self.blocksize)
615: 
616:         # e.g. bsr_plus_bsr, etc.
617:         fn = getattr(_sparsetools, self.format + op + self.format)
618: 
619:         R,C = self.blocksize
620: 
621:         max_bnnz = len(self.data) + len(other.data)
622:         idx_dtype = get_index_dtype((self.indptr, self.indices,
623:                                      other.indptr, other.indices),
624:                                     maxval=max_bnnz)
625:         indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
626:         indices = np.empty(max_bnnz, dtype=idx_dtype)
627: 
628:         bool_ops = ['_ne_', '_lt_', '_gt_', '_le_', '_ge_']
629:         if op in bool_ops:
630:             data = np.empty(R*C*max_bnnz, dtype=np.bool_)
631:         else:
632:             data = np.empty(R*C*max_bnnz, dtype=upcast(self.dtype,other.dtype))
633: 
634:         fn(self.shape[0]//R, self.shape[1]//C, R, C,
635:            self.indptr.astype(idx_dtype),
636:            self.indices.astype(idx_dtype),
637:            self.data,
638:            other.indptr.astype(idx_dtype),
639:            other.indices.astype(idx_dtype),
640:            np.ravel(other.data),
641:            indptr,
642:            indices,
643:            data)
644: 
645:         actual_bnnz = indptr[-1]
646:         indices = indices[:actual_bnnz]
647:         data = data[:R*C*actual_bnnz]
648: 
649:         if actual_bnnz < max_bnnz/2:
650:             indices = indices.copy()
651:             data = data.copy()
652: 
653:         data = data.reshape(-1,R,C)
654: 
655:         return self.__class__((data, indices, indptr), shape=self.shape)
656: 
657:     # needed by _data_matrix
658:     def _with_data(self,data,copy=True):
659:         '''Returns a matrix with the same sparsity structure as self,
660:         but with different data.  By default the structure arrays
661:         (i.e. .indptr and .indices) are copied.
662:         '''
663:         if copy:
664:             return self.__class__((data,self.indices.copy(),self.indptr.copy()),
665:                                    shape=self.shape,dtype=data.dtype)
666:         else:
667:             return self.__class__((data,self.indices,self.indptr),
668:                                    shape=self.shape,dtype=data.dtype)
669: 
670: #    # these functions are used by the parent class
671: #    # to remove redudancy between bsc_matrix and bsr_matrix
672: #    def _swap(self,x):
673: #        '''swap the members of x if this is a column-oriented matrix
674: #        '''
675: #        return (x[0],x[1])
676: 
677: 
678: def isspmatrix_bsr(x):
679:     '''Is x of a bsr_matrix type?
680: 
681:     Parameters
682:     ----------
683:     x
684:         object to check for being a bsr matrix
685: 
686:     Returns
687:     -------
688:     bool
689:         True if x is a bsr matrix, False otherwise
690: 
691:     Examples
692:     --------
693:     >>> from scipy.sparse import bsr_matrix, isspmatrix_bsr
694:     >>> isspmatrix_bsr(bsr_matrix([[5]]))
695:     True
696: 
697:     >>> from scipy.sparse import bsr_matrix, csr_matrix, isspmatrix_bsr
698:     >>> isspmatrix_bsr(csr_matrix([[5]]))
699:     False
700:     '''
701:     return isinstance(x, bsr_matrix)
702: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_358915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Compressed Block Sparse Row matrix format')

# Assigning a Str to a Name (line 5):

# Assigning a Str to a Name (line 5):
str_358916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__docformat__', str_358916)

# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['bsr_matrix', 'isspmatrix_bsr']
module_type_store.set_exportable_members(['bsr_matrix', 'isspmatrix_bsr'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_358917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_358918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'bsr_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_358917, str_358918)
# Adding element type (line 7)
str_358919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'str', 'isspmatrix_bsr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_358917, str_358919)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_358917)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from warnings import warn' statement (line 9)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_358920 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_358920) is not StypyTypeError):

    if (import_358920 != 'pyd_module'):
        __import__(import_358920)
        sys_modules_358921 = sys.modules[import_358920]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_358921.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_358920)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.sparse.data import _data_matrix, _minmax_mixin' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_358922 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.data')

if (type(import_358922) is not StypyTypeError):

    if (import_358922 != 'pyd_module'):
        __import__(import_358922)
        sys_modules_358923 = sys.modules[import_358922]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.data', sys_modules_358923.module_type_store, module_type_store, ['_data_matrix', '_minmax_mixin'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_358923, sys_modules_358923.module_type_store, module_type_store)
    else:
        from scipy.sparse.data import _data_matrix, _minmax_mixin

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.data', None, module_type_store, ['_data_matrix', '_minmax_mixin'], [_data_matrix, _minmax_mixin])

else:
    # Assigning a type to the variable 'scipy.sparse.data' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.data', import_358922)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.sparse.compressed import _cs_matrix' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_358924 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.compressed')

if (type(import_358924) is not StypyTypeError):

    if (import_358924 != 'pyd_module'):
        __import__(import_358924)
        sys_modules_358925 = sys.modules[import_358924]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.compressed', sys_modules_358925.module_type_store, module_type_store, ['_cs_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_358925, sys_modules_358925.module_type_store, module_type_store)
    else:
        from scipy.sparse.compressed import _cs_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.compressed', None, module_type_store, ['_cs_matrix'], [_cs_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse.compressed' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.compressed', import_358924)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.sparse.base import isspmatrix, _formats, spmatrix' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_358926 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.base')

if (type(import_358926) is not StypyTypeError):

    if (import_358926 != 'pyd_module'):
        __import__(import_358926)
        sys_modules_358927 = sys.modules[import_358926]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.base', sys_modules_358927.module_type_store, module_type_store, ['isspmatrix', '_formats', 'spmatrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_358927, sys_modules_358927.module_type_store, module_type_store)
    else:
        from scipy.sparse.base import isspmatrix, _formats, spmatrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.base', None, module_type_store, ['isspmatrix', '_formats', 'spmatrix'], [isspmatrix, _formats, spmatrix])

else:
    # Assigning a type to the variable 'scipy.sparse.base' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.base', import_358926)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.sparse.sputils import isshape, getdtype, to_native, upcast, get_index_dtype' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_358928 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.sputils')

if (type(import_358928) is not StypyTypeError):

    if (import_358928 != 'pyd_module'):
        __import__(import_358928)
        sys_modules_358929 = sys.modules[import_358928]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.sputils', sys_modules_358929.module_type_store, module_type_store, ['isshape', 'getdtype', 'to_native', 'upcast', 'get_index_dtype'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_358929, sys_modules_358929.module_type_store, module_type_store)
    else:
        from scipy.sparse.sputils import isshape, getdtype, to_native, upcast, get_index_dtype

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.sputils', None, module_type_store, ['isshape', 'getdtype', 'to_native', 'upcast', 'get_index_dtype'], [isshape, getdtype, to_native, upcast, get_index_dtype])

else:
    # Assigning a type to the variable 'scipy.sparse.sputils' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.sputils', import_358928)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.sparse import _sparsetools' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_358930 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse')

if (type(import_358930) is not StypyTypeError):

    if (import_358930 != 'pyd_module'):
        __import__(import_358930)
        sys_modules_358931 = sys.modules[import_358930]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse', sys_modules_358931.module_type_store, module_type_store, ['_sparsetools'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_358931, sys_modules_358931.module_type_store, module_type_store)
    else:
        from scipy.sparse import _sparsetools

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse', None, module_type_store, ['_sparsetools'], [_sparsetools])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse', import_358930)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.sparse._sparsetools import bsr_matvec, bsr_matvecs, csr_matmat_pass1, bsr_matmat_pass2, bsr_transpose, bsr_sort_indices' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_358932 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse._sparsetools')

if (type(import_358932) is not StypyTypeError):

    if (import_358932 != 'pyd_module'):
        __import__(import_358932)
        sys_modules_358933 = sys.modules[import_358932]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse._sparsetools', sys_modules_358933.module_type_store, module_type_store, ['bsr_matvec', 'bsr_matvecs', 'csr_matmat_pass1', 'bsr_matmat_pass2', 'bsr_transpose', 'bsr_sort_indices'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_358933, sys_modules_358933.module_type_store, module_type_store)
    else:
        from scipy.sparse._sparsetools import bsr_matvec, bsr_matvecs, csr_matmat_pass1, bsr_matmat_pass2, bsr_transpose, bsr_sort_indices

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse._sparsetools', None, module_type_store, ['bsr_matvec', 'bsr_matvecs', 'csr_matmat_pass1', 'bsr_matmat_pass2', 'bsr_transpose', 'bsr_sort_indices'], [bsr_matvec, bsr_matvecs, csr_matmat_pass1, bsr_matmat_pass2, bsr_transpose, bsr_sort_indices])

else:
    # Assigning a type to the variable 'scipy.sparse._sparsetools' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse._sparsetools', import_358932)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

# Declaration of the 'bsr_matrix' class
# Getting the type of '_cs_matrix' (line 22)
_cs_matrix_358934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), '_cs_matrix')
# Getting the type of '_minmax_mixin' (line 22)
_minmax_mixin_358935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 29), '_minmax_mixin')

class bsr_matrix(_cs_matrix_358934, _minmax_mixin_358935, ):
    str_358936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', "Block Sparse Row matrix\n\n    This can be instantiated in several ways:\n        bsr_matrix(D, [blocksize=(R,C)])\n            where D is a dense matrix or 2-D ndarray.\n\n        bsr_matrix(S, [blocksize=(R,C)])\n            with another sparse matrix S (equivalent to S.tobsr())\n\n        bsr_matrix((M, N), [blocksize=(R,C), dtype])\n            to construct an empty matrix with shape (M, N)\n            dtype is optional, defaulting to dtype='d'.\n\n        bsr_matrix((data, ij), [blocksize=(R,C), shape=(M, N)])\n            where ``data`` and ``ij`` satisfy ``a[ij[0, k], ij[1, k]] = data[k]``\n\n        bsr_matrix((data, indices, indptr), [shape=(M, N)])\n            is the standard BSR representation where the block column\n            indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]``\n            and their corresponding block values are stored in\n            ``data[ indptr[i]: indptr[i+1] ]``.  If the shape parameter is not\n            supplied, the matrix dimensions are inferred from the index arrays.\n\n    Attributes\n    ----------\n    dtype : dtype\n        Data type of the matrix\n    shape : 2-tuple\n        Shape of the matrix\n    ndim : int\n        Number of dimensions (this is always 2)\n    nnz\n        Number of nonzero elements\n    data\n        Data array of the matrix\n    indices\n        BSR format index array\n    indptr\n        BSR format index pointer array\n    blocksize\n        Block size of the matrix\n    has_sorted_indices\n        Whether indices are sorted\n\n    Notes\n    -----\n    Sparse matrices can be used in arithmetic operations: they support\n    addition, subtraction, multiplication, division, and matrix power.\n\n    **Summary of BSR format**\n\n    The Block Compressed Row (BSR) format is very similar to the Compressed\n    Sparse Row (CSR) format.  BSR is appropriate for sparse matrices with dense\n    sub matrices like the last example below.  Block matrices often arise in\n    vector-valued finite element discretizations.  In such cases, BSR is\n    considerably more efficient than CSR and CSC for many sparse arithmetic\n    operations.\n\n    **Blocksize**\n\n    The blocksize (R,C) must evenly divide the shape of the matrix (M,N).\n    That is, R and C must satisfy the relationship ``M % R = 0`` and\n    ``N % C = 0``.\n\n    If no blocksize is specified, a simple heuristic is applied to determine\n    an appropriate blocksize.\n\n    Examples\n    --------\n    >>> from scipy.sparse import bsr_matrix\n    >>> bsr_matrix((3, 4), dtype=np.int8).toarray()\n    array([[0, 0, 0, 0],\n           [0, 0, 0, 0],\n           [0, 0, 0, 0]], dtype=int8)\n\n    >>> row = np.array([0, 0, 1, 2, 2, 2])\n    >>> col = np.array([0, 2, 2, 0, 1, 2])\n    >>> data = np.array([1, 2, 3 ,4, 5, 6])\n    >>> bsr_matrix((data, (row, col)), shape=(3, 3)).toarray()\n    array([[1, 0, 2],\n           [0, 0, 3],\n           [4, 5, 6]])\n\n    >>> indptr = np.array([0, 2, 3, 6])\n    >>> indices = np.array([0, 2, 2, 0, 1, 2])\n    >>> data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)\n    >>> bsr_matrix((data,indices,indptr), shape=(6, 6)).toarray()\n    array([[1, 1, 0, 0, 2, 2],\n           [1, 1, 0, 0, 2, 2],\n           [0, 0, 0, 0, 3, 3],\n           [0, 0, 0, 0, 3, 3],\n           [4, 4, 5, 5, 6, 6],\n           [4, 4, 5, 5, 6, 6]])\n\n    ")
    
    # Assigning a Str to a Name (line 118):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 120)
        None_358937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 35), 'None')
        # Getting the type of 'None' (line 120)
        None_358938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 47), 'None')
        # Getting the type of 'False' (line 120)
        False_358939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 58), 'False')
        # Getting the type of 'None' (line 120)
        None_358940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 75), 'None')
        defaults = [None_358937, None_358938, False_358939, None_358940]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.__init__', ['arg1', 'shape', 'dtype', 'copy', 'blocksize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['arg1', 'shape', 'dtype', 'copy', 'blocksize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'self' (line 121)
        self_358943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 30), 'self', False)
        # Processing the call keyword arguments (line 121)
        kwargs_358944 = {}
        # Getting the type of '_data_matrix' (line 121)
        _data_matrix_358941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), '_data_matrix', False)
        # Obtaining the member '__init__' of a type (line 121)
        init___358942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), _data_matrix_358941, '__init__')
        # Calling __init__(args, kwargs) (line 121)
        init___call_result_358945 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), init___358942, *[self_358943], **kwargs_358944)
        
        
        
        # Call to isspmatrix(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'arg1' (line 123)
        arg1_358947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 'arg1', False)
        # Processing the call keyword arguments (line 123)
        kwargs_358948 = {}
        # Getting the type of 'isspmatrix' (line 123)
        isspmatrix_358946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 123)
        isspmatrix_call_result_358949 = invoke(stypy.reporting.localization.Localization(__file__, 123, 11), isspmatrix_358946, *[arg1_358947], **kwargs_358948)
        
        # Testing the type of an if condition (line 123)
        if_condition_358950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 8), isspmatrix_call_result_358949)
        # Assigning a type to the variable 'if_condition_358950' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'if_condition_358950', if_condition_358950)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Call to isspmatrix_bsr(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'arg1' (line 124)
        arg1_358952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 'arg1', False)
        # Processing the call keyword arguments (line 124)
        kwargs_358953 = {}
        # Getting the type of 'isspmatrix_bsr' (line 124)
        isspmatrix_bsr_358951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'isspmatrix_bsr', False)
        # Calling isspmatrix_bsr(args, kwargs) (line 124)
        isspmatrix_bsr_call_result_358954 = invoke(stypy.reporting.localization.Localization(__file__, 124, 15), isspmatrix_bsr_358951, *[arg1_358952], **kwargs_358953)
        
        # Getting the type of 'copy' (line 124)
        copy_358955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'copy')
        # Applying the binary operator 'and' (line 124)
        result_and_keyword_358956 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 15), 'and', isspmatrix_bsr_call_result_358954, copy_358955)
        
        # Testing the type of an if condition (line 124)
        if_condition_358957 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 12), result_and_keyword_358956)
        # Assigning a type to the variable 'if_condition_358957' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'if_condition_358957', if_condition_358957)
        # SSA begins for if statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 125):
        
        # Assigning a Call to a Name (line 125):
        
        # Call to copy(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_358960 = {}
        # Getting the type of 'arg1' (line 125)
        arg1_358958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 23), 'arg1', False)
        # Obtaining the member 'copy' of a type (line 125)
        copy_358959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 23), arg1_358958, 'copy')
        # Calling copy(args, kwargs) (line 125)
        copy_call_result_358961 = invoke(stypy.reporting.localization.Localization(__file__, 125, 23), copy_358959, *[], **kwargs_358960)
        
        # Assigning a type to the variable 'arg1' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'arg1', copy_call_result_358961)
        # SSA branch for the else part of an if statement (line 124)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 127):
        
        # Assigning a Call to a Name (line 127):
        
        # Call to tobsr(...): (line 127)
        # Processing the call keyword arguments (line 127)
        # Getting the type of 'blocksize' (line 127)
        blocksize_358964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 44), 'blocksize', False)
        keyword_358965 = blocksize_358964
        kwargs_358966 = {'blocksize': keyword_358965}
        # Getting the type of 'arg1' (line 127)
        arg1_358962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'arg1', False)
        # Obtaining the member 'tobsr' of a type (line 127)
        tobsr_358963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 23), arg1_358962, 'tobsr')
        # Calling tobsr(args, kwargs) (line 127)
        tobsr_call_result_358967 = invoke(stypy.reporting.localization.Localization(__file__, 127, 23), tobsr_358963, *[], **kwargs_358966)
        
        # Assigning a type to the variable 'arg1' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'arg1', tobsr_call_result_358967)
        # SSA join for if statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _set_self(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'arg1' (line 128)
        arg1_358970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 'arg1', False)
        # Processing the call keyword arguments (line 128)
        kwargs_358971 = {}
        # Getting the type of 'self' (line 128)
        self_358968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'self', False)
        # Obtaining the member '_set_self' of a type (line 128)
        _set_self_358969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), self_358968, '_set_self')
        # Calling _set_self(args, kwargs) (line 128)
        _set_self_call_result_358972 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), _set_self_358969, *[arg1_358970], **kwargs_358971)
        
        # SSA branch for the else part of an if statement (line 123)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 130)
        # Getting the type of 'tuple' (line 130)
        tuple_358973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 29), 'tuple')
        # Getting the type of 'arg1' (line 130)
        arg1_358974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'arg1')
        
        (may_be_358975, more_types_in_union_358976) = may_be_subtype(tuple_358973, arg1_358974)

        if may_be_358975:

            if more_types_in_union_358976:
                # Runtime conditional SSA (line 130)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'arg1' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'arg1', remove_not_subtype_from_union(arg1_358974, tuple))
            
            
            # Call to isshape(...): (line 131)
            # Processing the call arguments (line 131)
            # Getting the type of 'arg1' (line 131)
            arg1_358978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 23), 'arg1', False)
            # Processing the call keyword arguments (line 131)
            kwargs_358979 = {}
            # Getting the type of 'isshape' (line 131)
            isshape_358977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'isshape', False)
            # Calling isshape(args, kwargs) (line 131)
            isshape_call_result_358980 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), isshape_358977, *[arg1_358978], **kwargs_358979)
            
            # Testing the type of an if condition (line 131)
            if_condition_358981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 12), isshape_call_result_358980)
            # Assigning a type to the variable 'if_condition_358981' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'if_condition_358981', if_condition_358981)
            # SSA begins for if statement (line 131)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 133):
            
            # Assigning a Name to a Attribute (line 133):
            # Getting the type of 'arg1' (line 133)
            arg1_358982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 29), 'arg1')
            # Getting the type of 'self' (line 133)
            self_358983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'self')
            # Setting the type of the member 'shape' of a type (line 133)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), self_358983, 'shape', arg1_358982)
            
            # Assigning a Attribute to a Tuple (line 134):
            
            # Assigning a Subscript to a Name (line 134):
            
            # Obtaining the type of the subscript
            int_358984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 16), 'int')
            # Getting the type of 'self' (line 134)
            self_358985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'self')
            # Obtaining the member 'shape' of a type (line 134)
            shape_358986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 22), self_358985, 'shape')
            # Obtaining the member '__getitem__' of a type (line 134)
            getitem___358987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 16), shape_358986, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 134)
            subscript_call_result_358988 = invoke(stypy.reporting.localization.Localization(__file__, 134, 16), getitem___358987, int_358984)
            
            # Assigning a type to the variable 'tuple_var_assignment_358856' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'tuple_var_assignment_358856', subscript_call_result_358988)
            
            # Assigning a Subscript to a Name (line 134):
            
            # Obtaining the type of the subscript
            int_358989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 16), 'int')
            # Getting the type of 'self' (line 134)
            self_358990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'self')
            # Obtaining the member 'shape' of a type (line 134)
            shape_358991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 22), self_358990, 'shape')
            # Obtaining the member '__getitem__' of a type (line 134)
            getitem___358992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 16), shape_358991, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 134)
            subscript_call_result_358993 = invoke(stypy.reporting.localization.Localization(__file__, 134, 16), getitem___358992, int_358989)
            
            # Assigning a type to the variable 'tuple_var_assignment_358857' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'tuple_var_assignment_358857', subscript_call_result_358993)
            
            # Assigning a Name to a Name (line 134):
            # Getting the type of 'tuple_var_assignment_358856' (line 134)
            tuple_var_assignment_358856_358994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'tuple_var_assignment_358856')
            # Assigning a type to the variable 'M' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'M', tuple_var_assignment_358856_358994)
            
            # Assigning a Name to a Name (line 134):
            # Getting the type of 'tuple_var_assignment_358857' (line 134)
            tuple_var_assignment_358857_358995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'tuple_var_assignment_358857')
            # Assigning a type to the variable 'N' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'N', tuple_var_assignment_358857_358995)
            
            # Type idiom detected: calculating its left and rigth part (line 136)
            # Getting the type of 'blocksize' (line 136)
            blocksize_358996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'blocksize')
            # Getting the type of 'None' (line 136)
            None_358997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 32), 'None')
            
            (may_be_358998, more_types_in_union_358999) = may_be_none(blocksize_358996, None_358997)

            if may_be_358998:

                if more_types_in_union_358999:
                    # Runtime conditional SSA (line 136)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Tuple to a Name (line 137):
                
                # Assigning a Tuple to a Name (line 137):
                
                # Obtaining an instance of the builtin type 'tuple' (line 137)
                tuple_359000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 33), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 137)
                # Adding element type (line 137)
                int_359001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 33), 'int')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 33), tuple_359000, int_359001)
                # Adding element type (line 137)
                int_359002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 35), 'int')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 33), tuple_359000, int_359002)
                
                # Assigning a type to the variable 'blocksize' (line 137)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 20), 'blocksize', tuple_359000)

                if more_types_in_union_358999:
                    # Runtime conditional SSA for else branch (line 136)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_358998) or more_types_in_union_358999):
                
                
                
                # Call to isshape(...): (line 139)
                # Processing the call arguments (line 139)
                # Getting the type of 'blocksize' (line 139)
                blocksize_359004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 35), 'blocksize', False)
                # Processing the call keyword arguments (line 139)
                kwargs_359005 = {}
                # Getting the type of 'isshape' (line 139)
                isshape_359003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 27), 'isshape', False)
                # Calling isshape(args, kwargs) (line 139)
                isshape_call_result_359006 = invoke(stypy.reporting.localization.Localization(__file__, 139, 27), isshape_359003, *[blocksize_359004], **kwargs_359005)
                
                # Applying the 'not' unary operator (line 139)
                result_not__359007 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 23), 'not', isshape_call_result_359006)
                
                # Testing the type of an if condition (line 139)
                if_condition_359008 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 20), result_not__359007)
                # Assigning a type to the variable 'if_condition_359008' (line 139)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'if_condition_359008', if_condition_359008)
                # SSA begins for if statement (line 139)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to ValueError(...): (line 140)
                # Processing the call arguments (line 140)
                str_359010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 41), 'str', 'invalid blocksize=%s')
                # Getting the type of 'blocksize' (line 140)
                blocksize_359011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 66), 'blocksize', False)
                # Applying the binary operator '%' (line 140)
                result_mod_359012 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 41), '%', str_359010, blocksize_359011)
                
                # Processing the call keyword arguments (line 140)
                kwargs_359013 = {}
                # Getting the type of 'ValueError' (line 140)
                ValueError_359009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 140)
                ValueError_call_result_359014 = invoke(stypy.reporting.localization.Localization(__file__, 140, 30), ValueError_359009, *[result_mod_359012], **kwargs_359013)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 140, 24), ValueError_call_result_359014, 'raise parameter', BaseException)
                # SSA join for if statement (line 139)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a Call to a Name (line 141):
                
                # Assigning a Call to a Name (line 141):
                
                # Call to tuple(...): (line 141)
                # Processing the call arguments (line 141)
                # Getting the type of 'blocksize' (line 141)
                blocksize_359016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 38), 'blocksize', False)
                # Processing the call keyword arguments (line 141)
                kwargs_359017 = {}
                # Getting the type of 'tuple' (line 141)
                tuple_359015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 32), 'tuple', False)
                # Calling tuple(args, kwargs) (line 141)
                tuple_call_result_359018 = invoke(stypy.reporting.localization.Localization(__file__, 141, 32), tuple_359015, *[blocksize_359016], **kwargs_359017)
                
                # Assigning a type to the variable 'blocksize' (line 141)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'blocksize', tuple_call_result_359018)

                if (may_be_358998 and more_types_in_union_358999):
                    # SSA join for if statement (line 136)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Attribute (line 142):
            
            # Assigning a Call to a Attribute (line 142):
            
            # Call to zeros(...): (line 142)
            # Processing the call arguments (line 142)
            
            # Obtaining an instance of the builtin type 'tuple' (line 142)
            tuple_359021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 38), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 142)
            # Adding element type (line 142)
            int_359022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 38), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 38), tuple_359021, int_359022)
            
            # Getting the type of 'blocksize' (line 142)
            blocksize_359023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 44), 'blocksize', False)
            # Applying the binary operator '+' (line 142)
            result_add_359024 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 37), '+', tuple_359021, blocksize_359023)
            
            
            # Call to getdtype(...): (line 142)
            # Processing the call arguments (line 142)
            # Getting the type of 'dtype' (line 142)
            dtype_359026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 64), 'dtype', False)
            # Processing the call keyword arguments (line 142)
            # Getting the type of 'float' (line 142)
            float_359027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 79), 'float', False)
            keyword_359028 = float_359027
            kwargs_359029 = {'default': keyword_359028}
            # Getting the type of 'getdtype' (line 142)
            getdtype_359025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 55), 'getdtype', False)
            # Calling getdtype(args, kwargs) (line 142)
            getdtype_call_result_359030 = invoke(stypy.reporting.localization.Localization(__file__, 142, 55), getdtype_359025, *[dtype_359026], **kwargs_359029)
            
            # Processing the call keyword arguments (line 142)
            kwargs_359031 = {}
            # Getting the type of 'np' (line 142)
            np_359019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 28), 'np', False)
            # Obtaining the member 'zeros' of a type (line 142)
            zeros_359020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 28), np_359019, 'zeros')
            # Calling zeros(args, kwargs) (line 142)
            zeros_call_result_359032 = invoke(stypy.reporting.localization.Localization(__file__, 142, 28), zeros_359020, *[result_add_359024, getdtype_call_result_359030], **kwargs_359031)
            
            # Getting the type of 'self' (line 142)
            self_359033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'self')
            # Setting the type of the member 'data' of a type (line 142)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 16), self_359033, 'data', zeros_call_result_359032)
            
            # Assigning a Name to a Tuple (line 144):
            
            # Assigning a Subscript to a Name (line 144):
            
            # Obtaining the type of the subscript
            int_359034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 16), 'int')
            # Getting the type of 'blocksize' (line 144)
            blocksize_359035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 22), 'blocksize')
            # Obtaining the member '__getitem__' of a type (line 144)
            getitem___359036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 16), blocksize_359035, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 144)
            subscript_call_result_359037 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), getitem___359036, int_359034)
            
            # Assigning a type to the variable 'tuple_var_assignment_358858' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'tuple_var_assignment_358858', subscript_call_result_359037)
            
            # Assigning a Subscript to a Name (line 144):
            
            # Obtaining the type of the subscript
            int_359038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 16), 'int')
            # Getting the type of 'blocksize' (line 144)
            blocksize_359039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 22), 'blocksize')
            # Obtaining the member '__getitem__' of a type (line 144)
            getitem___359040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 16), blocksize_359039, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 144)
            subscript_call_result_359041 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), getitem___359040, int_359038)
            
            # Assigning a type to the variable 'tuple_var_assignment_358859' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'tuple_var_assignment_358859', subscript_call_result_359041)
            
            # Assigning a Name to a Name (line 144):
            # Getting the type of 'tuple_var_assignment_358858' (line 144)
            tuple_var_assignment_358858_359042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'tuple_var_assignment_358858')
            # Assigning a type to the variable 'R' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'R', tuple_var_assignment_358858_359042)
            
            # Assigning a Name to a Name (line 144):
            # Getting the type of 'tuple_var_assignment_358859' (line 144)
            tuple_var_assignment_358859_359043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'tuple_var_assignment_358859')
            # Assigning a type to the variable 'C' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 18), 'C', tuple_var_assignment_358859_359043)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'M' (line 145)
            M_359044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 20), 'M')
            # Getting the type of 'R' (line 145)
            R_359045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 24), 'R')
            # Applying the binary operator '%' (line 145)
            result_mod_359046 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 20), '%', M_359044, R_359045)
            
            int_359047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 30), 'int')
            # Applying the binary operator '!=' (line 145)
            result_ne_359048 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 19), '!=', result_mod_359046, int_359047)
            
            
            # Getting the type of 'N' (line 145)
            N_359049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 36), 'N')
            # Getting the type of 'C' (line 145)
            C_359050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 40), 'C')
            # Applying the binary operator '%' (line 145)
            result_mod_359051 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 36), '%', N_359049, C_359050)
            
            int_359052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 46), 'int')
            # Applying the binary operator '!=' (line 145)
            result_ne_359053 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 35), '!=', result_mod_359051, int_359052)
            
            # Applying the binary operator 'or' (line 145)
            result_or_keyword_359054 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 19), 'or', result_ne_359048, result_ne_359053)
            
            # Testing the type of an if condition (line 145)
            if_condition_359055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 16), result_or_keyword_359054)
            # Assigning a type to the variable 'if_condition_359055' (line 145)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'if_condition_359055', if_condition_359055)
            # SSA begins for if statement (line 145)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 146)
            # Processing the call arguments (line 146)
            str_359057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 37), 'str', 'shape must be multiple of blocksize')
            # Processing the call keyword arguments (line 146)
            kwargs_359058 = {}
            # Getting the type of 'ValueError' (line 146)
            ValueError_359056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 146)
            ValueError_call_result_359059 = invoke(stypy.reporting.localization.Localization(__file__, 146, 26), ValueError_359056, *[str_359057], **kwargs_359058)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 146, 20), ValueError_call_result_359059, 'raise parameter', BaseException)
            # SSA join for if statement (line 145)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 150):
            
            # Assigning a Call to a Name (line 150):
            
            # Call to get_index_dtype(...): (line 150)
            # Processing the call keyword arguments (line 150)
            
            # Call to max(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'M' (line 150)
            M_359062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 55), 'M', False)
            # Getting the type of 'R' (line 150)
            R_359063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 58), 'R', False)
            # Applying the binary operator '//' (line 150)
            result_floordiv_359064 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 55), '//', M_359062, R_359063)
            
            # Getting the type of 'N' (line 150)
            N_359065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 61), 'N', False)
            # Getting the type of 'C' (line 150)
            C_359066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 64), 'C', False)
            # Applying the binary operator '//' (line 150)
            result_floordiv_359067 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 61), '//', N_359065, C_359066)
            
            # Getting the type of 'R' (line 150)
            R_359068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 67), 'R', False)
            # Getting the type of 'C' (line 150)
            C_359069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 70), 'C', False)
            # Processing the call keyword arguments (line 150)
            kwargs_359070 = {}
            # Getting the type of 'max' (line 150)
            max_359061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 51), 'max', False)
            # Calling max(args, kwargs) (line 150)
            max_call_result_359071 = invoke(stypy.reporting.localization.Localization(__file__, 150, 51), max_359061, *[result_floordiv_359064, result_floordiv_359067, R_359068, C_359069], **kwargs_359070)
            
            keyword_359072 = max_call_result_359071
            kwargs_359073 = {'maxval': keyword_359072}
            # Getting the type of 'get_index_dtype' (line 150)
            get_index_dtype_359060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 28), 'get_index_dtype', False)
            # Calling get_index_dtype(args, kwargs) (line 150)
            get_index_dtype_call_result_359074 = invoke(stypy.reporting.localization.Localization(__file__, 150, 28), get_index_dtype_359060, *[], **kwargs_359073)
            
            # Assigning a type to the variable 'idx_dtype' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'idx_dtype', get_index_dtype_call_result_359074)
            
            # Assigning a Call to a Attribute (line 151):
            
            # Assigning a Call to a Attribute (line 151):
            
            # Call to zeros(...): (line 151)
            # Processing the call arguments (line 151)
            int_359077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 40), 'int')
            # Processing the call keyword arguments (line 151)
            # Getting the type of 'idx_dtype' (line 151)
            idx_dtype_359078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 49), 'idx_dtype', False)
            keyword_359079 = idx_dtype_359078
            kwargs_359080 = {'dtype': keyword_359079}
            # Getting the type of 'np' (line 151)
            np_359075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 31), 'np', False)
            # Obtaining the member 'zeros' of a type (line 151)
            zeros_359076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 31), np_359075, 'zeros')
            # Calling zeros(args, kwargs) (line 151)
            zeros_call_result_359081 = invoke(stypy.reporting.localization.Localization(__file__, 151, 31), zeros_359076, *[int_359077], **kwargs_359080)
            
            # Getting the type of 'self' (line 151)
            self_359082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'self')
            # Setting the type of the member 'indices' of a type (line 151)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 16), self_359082, 'indices', zeros_call_result_359081)
            
            # Assigning a Call to a Attribute (line 152):
            
            # Assigning a Call to a Attribute (line 152):
            
            # Call to zeros(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'M' (line 152)
            M_359085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 39), 'M', False)
            # Getting the type of 'R' (line 152)
            R_359086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 42), 'R', False)
            # Applying the binary operator '//' (line 152)
            result_floordiv_359087 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 39), '//', M_359085, R_359086)
            
            int_359088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 46), 'int')
            # Applying the binary operator '+' (line 152)
            result_add_359089 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 39), '+', result_floordiv_359087, int_359088)
            
            # Processing the call keyword arguments (line 152)
            # Getting the type of 'idx_dtype' (line 152)
            idx_dtype_359090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 55), 'idx_dtype', False)
            keyword_359091 = idx_dtype_359090
            kwargs_359092 = {'dtype': keyword_359091}
            # Getting the type of 'np' (line 152)
            np_359083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'np', False)
            # Obtaining the member 'zeros' of a type (line 152)
            zeros_359084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 30), np_359083, 'zeros')
            # Calling zeros(args, kwargs) (line 152)
            zeros_call_result_359093 = invoke(stypy.reporting.localization.Localization(__file__, 152, 30), zeros_359084, *[result_add_359089], **kwargs_359092)
            
            # Getting the type of 'self' (line 152)
            self_359094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'self')
            # Setting the type of the member 'indptr' of a type (line 152)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), self_359094, 'indptr', zeros_call_result_359093)
            # SSA branch for the else part of an if statement (line 131)
            module_type_store.open_ssa_branch('else')
            
            
            
            # Call to len(...): (line 154)
            # Processing the call arguments (line 154)
            # Getting the type of 'arg1' (line 154)
            arg1_359096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 21), 'arg1', False)
            # Processing the call keyword arguments (line 154)
            kwargs_359097 = {}
            # Getting the type of 'len' (line 154)
            len_359095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 17), 'len', False)
            # Calling len(args, kwargs) (line 154)
            len_call_result_359098 = invoke(stypy.reporting.localization.Localization(__file__, 154, 17), len_359095, *[arg1_359096], **kwargs_359097)
            
            int_359099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 30), 'int')
            # Applying the binary operator '==' (line 154)
            result_eq_359100 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 17), '==', len_call_result_359098, int_359099)
            
            # Testing the type of an if condition (line 154)
            if_condition_359101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 17), result_eq_359100)
            # Assigning a type to the variable 'if_condition_359101' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 17), 'if_condition_359101', if_condition_359101)
            # SSA begins for if statement (line 154)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 156, 16))
            
            # 'from scipy.sparse.coo import coo_matrix' statement (line 156)
            update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
            import_359102 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 156, 16), 'scipy.sparse.coo')

            if (type(import_359102) is not StypyTypeError):

                if (import_359102 != 'pyd_module'):
                    __import__(import_359102)
                    sys_modules_359103 = sys.modules[import_359102]
                    import_from_module(stypy.reporting.localization.Localization(__file__, 156, 16), 'scipy.sparse.coo', sys_modules_359103.module_type_store, module_type_store, ['coo_matrix'])
                    nest_module(stypy.reporting.localization.Localization(__file__, 156, 16), __file__, sys_modules_359103, sys_modules_359103.module_type_store, module_type_store)
                else:
                    from scipy.sparse.coo import coo_matrix

                    import_from_module(stypy.reporting.localization.Localization(__file__, 156, 16), 'scipy.sparse.coo', None, module_type_store, ['coo_matrix'], [coo_matrix])

            else:
                # Assigning a type to the variable 'scipy.sparse.coo' (line 156)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'scipy.sparse.coo', import_359102)

            remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
            
            
            # Call to _set_self(...): (line 157)
            # Processing the call arguments (line 157)
            
            # Call to tobsr(...): (line 157)
            # Processing the call keyword arguments (line 157)
            # Getting the type of 'blocksize' (line 157)
            blocksize_359113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 77), 'blocksize', False)
            keyword_359114 = blocksize_359113
            kwargs_359115 = {'blocksize': keyword_359114}
            
            # Call to coo_matrix(...): (line 157)
            # Processing the call arguments (line 157)
            # Getting the type of 'arg1' (line 157)
            arg1_359107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 42), 'arg1', False)
            # Processing the call keyword arguments (line 157)
            # Getting the type of 'dtype' (line 157)
            dtype_359108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 54), 'dtype', False)
            keyword_359109 = dtype_359108
            kwargs_359110 = {'dtype': keyword_359109}
            # Getting the type of 'coo_matrix' (line 157)
            coo_matrix_359106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 31), 'coo_matrix', False)
            # Calling coo_matrix(args, kwargs) (line 157)
            coo_matrix_call_result_359111 = invoke(stypy.reporting.localization.Localization(__file__, 157, 31), coo_matrix_359106, *[arg1_359107], **kwargs_359110)
            
            # Obtaining the member 'tobsr' of a type (line 157)
            tobsr_359112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 31), coo_matrix_call_result_359111, 'tobsr')
            # Calling tobsr(args, kwargs) (line 157)
            tobsr_call_result_359116 = invoke(stypy.reporting.localization.Localization(__file__, 157, 31), tobsr_359112, *[], **kwargs_359115)
            
            # Processing the call keyword arguments (line 157)
            kwargs_359117 = {}
            # Getting the type of 'self' (line 157)
            self_359104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'self', False)
            # Obtaining the member '_set_self' of a type (line 157)
            _set_self_359105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 16), self_359104, '_set_self')
            # Calling _set_self(args, kwargs) (line 157)
            _set_self_call_result_359118 = invoke(stypy.reporting.localization.Localization(__file__, 157, 16), _set_self_359105, *[tobsr_call_result_359116], **kwargs_359117)
            
            # SSA branch for the else part of an if statement (line 154)
            module_type_store.open_ssa_branch('else')
            
            
            
            # Call to len(...): (line 159)
            # Processing the call arguments (line 159)
            # Getting the type of 'arg1' (line 159)
            arg1_359120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 21), 'arg1', False)
            # Processing the call keyword arguments (line 159)
            kwargs_359121 = {}
            # Getting the type of 'len' (line 159)
            len_359119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 17), 'len', False)
            # Calling len(args, kwargs) (line 159)
            len_call_result_359122 = invoke(stypy.reporting.localization.Localization(__file__, 159, 17), len_359119, *[arg1_359120], **kwargs_359121)
            
            int_359123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 30), 'int')
            # Applying the binary operator '==' (line 159)
            result_eq_359124 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 17), '==', len_call_result_359122, int_359123)
            
            # Testing the type of an if condition (line 159)
            if_condition_359125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 17), result_eq_359124)
            # Assigning a type to the variable 'if_condition_359125' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 17), 'if_condition_359125', if_condition_359125)
            # SSA begins for if statement (line 159)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Tuple (line 161):
            
            # Assigning a Subscript to a Name (line 161):
            
            # Obtaining the type of the subscript
            int_359126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'int')
            # Getting the type of 'arg1' (line 161)
            arg1_359127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 42), 'arg1')
            # Obtaining the member '__getitem__' of a type (line 161)
            getitem___359128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), arg1_359127, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 161)
            subscript_call_result_359129 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), getitem___359128, int_359126)
            
            # Assigning a type to the variable 'tuple_var_assignment_358860' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_358860', subscript_call_result_359129)
            
            # Assigning a Subscript to a Name (line 161):
            
            # Obtaining the type of the subscript
            int_359130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'int')
            # Getting the type of 'arg1' (line 161)
            arg1_359131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 42), 'arg1')
            # Obtaining the member '__getitem__' of a type (line 161)
            getitem___359132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), arg1_359131, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 161)
            subscript_call_result_359133 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), getitem___359132, int_359130)
            
            # Assigning a type to the variable 'tuple_var_assignment_358861' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_358861', subscript_call_result_359133)
            
            # Assigning a Subscript to a Name (line 161):
            
            # Obtaining the type of the subscript
            int_359134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'int')
            # Getting the type of 'arg1' (line 161)
            arg1_359135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 42), 'arg1')
            # Obtaining the member '__getitem__' of a type (line 161)
            getitem___359136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), arg1_359135, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 161)
            subscript_call_result_359137 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), getitem___359136, int_359134)
            
            # Assigning a type to the variable 'tuple_var_assignment_358862' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_358862', subscript_call_result_359137)
            
            # Assigning a Name to a Name (line 161):
            # Getting the type of 'tuple_var_assignment_358860' (line 161)
            tuple_var_assignment_358860_359138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_358860')
            # Assigning a type to the variable 'data' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 17), 'data', tuple_var_assignment_358860_359138)
            
            # Assigning a Name to a Name (line 161):
            # Getting the type of 'tuple_var_assignment_358861' (line 161)
            tuple_var_assignment_358861_359139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_358861')
            # Assigning a type to the variable 'indices' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'indices', tuple_var_assignment_358861_359139)
            
            # Assigning a Name to a Name (line 161):
            # Getting the type of 'tuple_var_assignment_358862' (line 161)
            tuple_var_assignment_358862_359140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_358862')
            # Assigning a type to the variable 'indptr' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 32), 'indptr', tuple_var_assignment_358862_359140)
            
            # Assigning a Num to a Name (line 165):
            
            # Assigning a Num to a Name (line 165):
            int_359141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 25), 'int')
            # Assigning a type to the variable 'maxval' (line 165)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'maxval', int_359141)
            
            # Type idiom detected: calculating its left and rigth part (line 166)
            # Getting the type of 'shape' (line 166)
            shape_359142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'shape')
            # Getting the type of 'None' (line 166)
            None_359143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 32), 'None')
            
            (may_be_359144, more_types_in_union_359145) = may_not_be_none(shape_359142, None_359143)

            if may_be_359144:

                if more_types_in_union_359145:
                    # Runtime conditional SSA (line 166)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Name (line 167):
                
                # Assigning a Call to a Name (line 167):
                
                # Call to max(...): (line 167)
                # Processing the call arguments (line 167)
                # Getting the type of 'shape' (line 167)
                shape_359147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'shape', False)
                # Processing the call keyword arguments (line 167)
                kwargs_359148 = {}
                # Getting the type of 'max' (line 167)
                max_359146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 29), 'max', False)
                # Calling max(args, kwargs) (line 167)
                max_call_result_359149 = invoke(stypy.reporting.localization.Localization(__file__, 167, 29), max_359146, *[shape_359147], **kwargs_359148)
                
                # Assigning a type to the variable 'maxval' (line 167)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'maxval', max_call_result_359149)

                if more_types_in_union_359145:
                    # SSA join for if statement (line 166)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 168)
            # Getting the type of 'blocksize' (line 168)
            blocksize_359150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'blocksize')
            # Getting the type of 'None' (line 168)
            None_359151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 36), 'None')
            
            (may_be_359152, more_types_in_union_359153) = may_not_be_none(blocksize_359150, None_359151)

            if may_be_359152:

                if more_types_in_union_359153:
                    # Runtime conditional SSA (line 168)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Name (line 169):
                
                # Assigning a Call to a Name (line 169):
                
                # Call to max(...): (line 169)
                # Processing the call arguments (line 169)
                # Getting the type of 'maxval' (line 169)
                maxval_359155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 33), 'maxval', False)
                
                # Call to max(...): (line 169)
                # Processing the call arguments (line 169)
                # Getting the type of 'blocksize' (line 169)
                blocksize_359157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 45), 'blocksize', False)
                # Processing the call keyword arguments (line 169)
                kwargs_359158 = {}
                # Getting the type of 'max' (line 169)
                max_359156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 41), 'max', False)
                # Calling max(args, kwargs) (line 169)
                max_call_result_359159 = invoke(stypy.reporting.localization.Localization(__file__, 169, 41), max_359156, *[blocksize_359157], **kwargs_359158)
                
                # Processing the call keyword arguments (line 169)
                kwargs_359160 = {}
                # Getting the type of 'max' (line 169)
                max_359154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'max', False)
                # Calling max(args, kwargs) (line 169)
                max_call_result_359161 = invoke(stypy.reporting.localization.Localization(__file__, 169, 29), max_359154, *[maxval_359155, max_call_result_359159], **kwargs_359160)
                
                # Assigning a type to the variable 'maxval' (line 169)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'maxval', max_call_result_359161)

                if more_types_in_union_359153:
                    # SSA join for if statement (line 168)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Name (line 170):
            
            # Assigning a Call to a Name (line 170):
            
            # Call to get_index_dtype(...): (line 170)
            # Processing the call arguments (line 170)
            
            # Obtaining an instance of the builtin type 'tuple' (line 170)
            tuple_359163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 45), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 170)
            # Adding element type (line 170)
            # Getting the type of 'indices' (line 170)
            indices_359164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 45), 'indices', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 45), tuple_359163, indices_359164)
            # Adding element type (line 170)
            # Getting the type of 'indptr' (line 170)
            indptr_359165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 54), 'indptr', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 45), tuple_359163, indptr_359165)
            
            # Processing the call keyword arguments (line 170)
            # Getting the type of 'maxval' (line 170)
            maxval_359166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 70), 'maxval', False)
            keyword_359167 = maxval_359166
            # Getting the type of 'True' (line 170)
            True_359168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 93), 'True', False)
            keyword_359169 = True_359168
            kwargs_359170 = {'maxval': keyword_359167, 'check_contents': keyword_359169}
            # Getting the type of 'get_index_dtype' (line 170)
            get_index_dtype_359162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 28), 'get_index_dtype', False)
            # Calling get_index_dtype(args, kwargs) (line 170)
            get_index_dtype_call_result_359171 = invoke(stypy.reporting.localization.Localization(__file__, 170, 28), get_index_dtype_359162, *[tuple_359163], **kwargs_359170)
            
            # Assigning a type to the variable 'idx_dtype' (line 170)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'idx_dtype', get_index_dtype_call_result_359171)
            
            # Assigning a Call to a Attribute (line 172):
            
            # Assigning a Call to a Attribute (line 172):
            
            # Call to array(...): (line 172)
            # Processing the call arguments (line 172)
            # Getting the type of 'indices' (line 172)
            indices_359174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 40), 'indices', False)
            # Processing the call keyword arguments (line 172)
            # Getting the type of 'copy' (line 172)
            copy_359175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 54), 'copy', False)
            keyword_359176 = copy_359175
            # Getting the type of 'idx_dtype' (line 172)
            idx_dtype_359177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 66), 'idx_dtype', False)
            keyword_359178 = idx_dtype_359177
            kwargs_359179 = {'dtype': keyword_359178, 'copy': keyword_359176}
            # Getting the type of 'np' (line 172)
            np_359172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 31), 'np', False)
            # Obtaining the member 'array' of a type (line 172)
            array_359173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 31), np_359172, 'array')
            # Calling array(args, kwargs) (line 172)
            array_call_result_359180 = invoke(stypy.reporting.localization.Localization(__file__, 172, 31), array_359173, *[indices_359174], **kwargs_359179)
            
            # Getting the type of 'self' (line 172)
            self_359181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'self')
            # Setting the type of the member 'indices' of a type (line 172)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), self_359181, 'indices', array_call_result_359180)
            
            # Assigning a Call to a Attribute (line 173):
            
            # Assigning a Call to a Attribute (line 173):
            
            # Call to array(...): (line 173)
            # Processing the call arguments (line 173)
            # Getting the type of 'indptr' (line 173)
            indptr_359184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 39), 'indptr', False)
            # Processing the call keyword arguments (line 173)
            # Getting the type of 'copy' (line 173)
            copy_359185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 52), 'copy', False)
            keyword_359186 = copy_359185
            # Getting the type of 'idx_dtype' (line 173)
            idx_dtype_359187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 64), 'idx_dtype', False)
            keyword_359188 = idx_dtype_359187
            kwargs_359189 = {'dtype': keyword_359188, 'copy': keyword_359186}
            # Getting the type of 'np' (line 173)
            np_359182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 30), 'np', False)
            # Obtaining the member 'array' of a type (line 173)
            array_359183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 30), np_359182, 'array')
            # Calling array(args, kwargs) (line 173)
            array_call_result_359190 = invoke(stypy.reporting.localization.Localization(__file__, 173, 30), array_359183, *[indptr_359184], **kwargs_359189)
            
            # Getting the type of 'self' (line 173)
            self_359191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'self')
            # Setting the type of the member 'indptr' of a type (line 173)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 16), self_359191, 'indptr', array_call_result_359190)
            
            # Assigning a Call to a Attribute (line 174):
            
            # Assigning a Call to a Attribute (line 174):
            
            # Call to array(...): (line 174)
            # Processing the call arguments (line 174)
            # Getting the type of 'data' (line 174)
            data_359194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 37), 'data', False)
            # Processing the call keyword arguments (line 174)
            # Getting the type of 'copy' (line 174)
            copy_359195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 48), 'copy', False)
            keyword_359196 = copy_359195
            
            # Call to getdtype(...): (line 174)
            # Processing the call arguments (line 174)
            # Getting the type of 'dtype' (line 174)
            dtype_359198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 69), 'dtype', False)
            # Getting the type of 'data' (line 174)
            data_359199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 76), 'data', False)
            # Processing the call keyword arguments (line 174)
            kwargs_359200 = {}
            # Getting the type of 'getdtype' (line 174)
            getdtype_359197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 60), 'getdtype', False)
            # Calling getdtype(args, kwargs) (line 174)
            getdtype_call_result_359201 = invoke(stypy.reporting.localization.Localization(__file__, 174, 60), getdtype_359197, *[dtype_359198, data_359199], **kwargs_359200)
            
            keyword_359202 = getdtype_call_result_359201
            kwargs_359203 = {'dtype': keyword_359202, 'copy': keyword_359196}
            # Getting the type of 'np' (line 174)
            np_359192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 28), 'np', False)
            # Obtaining the member 'array' of a type (line 174)
            array_359193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 28), np_359192, 'array')
            # Calling array(args, kwargs) (line 174)
            array_call_result_359204 = invoke(stypy.reporting.localization.Localization(__file__, 174, 28), array_359193, *[data_359194], **kwargs_359203)
            
            # Getting the type of 'self' (line 174)
            self_359205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'self')
            # Setting the type of the member 'data' of a type (line 174)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), self_359205, 'data', array_call_result_359204)
            # SSA branch for the else part of an if statement (line 159)
            module_type_store.open_ssa_branch('else')
            
            # Call to ValueError(...): (line 176)
            # Processing the call arguments (line 176)
            str_359207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 33), 'str', 'unrecognized bsr_matrix constructor usage')
            # Processing the call keyword arguments (line 176)
            kwargs_359208 = {}
            # Getting the type of 'ValueError' (line 176)
            ValueError_359206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 176)
            ValueError_call_result_359209 = invoke(stypy.reporting.localization.Localization(__file__, 176, 22), ValueError_359206, *[str_359207], **kwargs_359208)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 176, 16), ValueError_call_result_359209, 'raise parameter', BaseException)
            # SSA join for if statement (line 159)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 154)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 131)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_358976:
                # Runtime conditional SSA for else branch (line 130)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_358975) or more_types_in_union_358976):
            # Assigning a type to the variable 'arg1' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'arg1', remove_subtype_from_union(arg1_358974, tuple))
            
            
            # SSA begins for try-except statement (line 179)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 180):
            
            # Assigning a Call to a Name (line 180):
            
            # Call to asarray(...): (line 180)
            # Processing the call arguments (line 180)
            # Getting the type of 'arg1' (line 180)
            arg1_359212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 34), 'arg1', False)
            # Processing the call keyword arguments (line 180)
            kwargs_359213 = {}
            # Getting the type of 'np' (line 180)
            np_359210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'np', False)
            # Obtaining the member 'asarray' of a type (line 180)
            asarray_359211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 23), np_359210, 'asarray')
            # Calling asarray(args, kwargs) (line 180)
            asarray_call_result_359214 = invoke(stypy.reporting.localization.Localization(__file__, 180, 23), asarray_359211, *[arg1_359212], **kwargs_359213)
            
            # Assigning a type to the variable 'arg1' (line 180)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'arg1', asarray_call_result_359214)
            # SSA branch for the except part of a try statement (line 179)
            # SSA branch for the except '<any exception>' branch of a try statement (line 179)
            module_type_store.open_ssa_branch('except')
            
            # Call to ValueError(...): (line 182)
            # Processing the call arguments (line 182)
            str_359216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 33), 'str', 'unrecognized form for %s_matrix constructor')
            # Getting the type of 'self' (line 183)
            self_359217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 51), 'self', False)
            # Obtaining the member 'format' of a type (line 183)
            format_359218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 51), self_359217, 'format')
            # Applying the binary operator '%' (line 182)
            result_mod_359219 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 33), '%', str_359216, format_359218)
            
            # Processing the call keyword arguments (line 182)
            kwargs_359220 = {}
            # Getting the type of 'ValueError' (line 182)
            ValueError_359215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 182)
            ValueError_call_result_359221 = invoke(stypy.reporting.localization.Localization(__file__, 182, 22), ValueError_359215, *[result_mod_359219], **kwargs_359220)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 182, 16), ValueError_call_result_359221, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 179)
            module_type_store = module_type_store.join_ssa_context()
            
            stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 184, 12))
            
            # 'from scipy.sparse.coo import coo_matrix' statement (line 184)
            update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
            import_359222 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 184, 12), 'scipy.sparse.coo')

            if (type(import_359222) is not StypyTypeError):

                if (import_359222 != 'pyd_module'):
                    __import__(import_359222)
                    sys_modules_359223 = sys.modules[import_359222]
                    import_from_module(stypy.reporting.localization.Localization(__file__, 184, 12), 'scipy.sparse.coo', sys_modules_359223.module_type_store, module_type_store, ['coo_matrix'])
                    nest_module(stypy.reporting.localization.Localization(__file__, 184, 12), __file__, sys_modules_359223, sys_modules_359223.module_type_store, module_type_store)
                else:
                    from scipy.sparse.coo import coo_matrix

                    import_from_module(stypy.reporting.localization.Localization(__file__, 184, 12), 'scipy.sparse.coo', None, module_type_store, ['coo_matrix'], [coo_matrix])

            else:
                # Assigning a type to the variable 'scipy.sparse.coo' (line 184)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'scipy.sparse.coo', import_359222)

            remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
            
            
            # Assigning a Call to a Name (line 185):
            
            # Assigning a Call to a Name (line 185):
            
            # Call to tobsr(...): (line 185)
            # Processing the call keyword arguments (line 185)
            # Getting the type of 'blocksize' (line 185)
            blocksize_359231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 65), 'blocksize', False)
            keyword_359232 = blocksize_359231
            kwargs_359233 = {'blocksize': keyword_359232}
            
            # Call to coo_matrix(...): (line 185)
            # Processing the call arguments (line 185)
            # Getting the type of 'arg1' (line 185)
            arg1_359225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 30), 'arg1', False)
            # Processing the call keyword arguments (line 185)
            # Getting the type of 'dtype' (line 185)
            dtype_359226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 42), 'dtype', False)
            keyword_359227 = dtype_359226
            kwargs_359228 = {'dtype': keyword_359227}
            # Getting the type of 'coo_matrix' (line 185)
            coo_matrix_359224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'coo_matrix', False)
            # Calling coo_matrix(args, kwargs) (line 185)
            coo_matrix_call_result_359229 = invoke(stypy.reporting.localization.Localization(__file__, 185, 19), coo_matrix_359224, *[arg1_359225], **kwargs_359228)
            
            # Obtaining the member 'tobsr' of a type (line 185)
            tobsr_359230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 19), coo_matrix_call_result_359229, 'tobsr')
            # Calling tobsr(args, kwargs) (line 185)
            tobsr_call_result_359234 = invoke(stypy.reporting.localization.Localization(__file__, 185, 19), tobsr_359230, *[], **kwargs_359233)
            
            # Assigning a type to the variable 'arg1' (line 185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'arg1', tobsr_call_result_359234)
            
            # Call to _set_self(...): (line 186)
            # Processing the call arguments (line 186)
            # Getting the type of 'arg1' (line 186)
            arg1_359237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'arg1', False)
            # Processing the call keyword arguments (line 186)
            kwargs_359238 = {}
            # Getting the type of 'self' (line 186)
            self_359235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'self', False)
            # Obtaining the member '_set_self' of a type (line 186)
            _set_self_359236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), self_359235, '_set_self')
            # Calling _set_self(args, kwargs) (line 186)
            _set_self_call_result_359239 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), _set_self_359236, *[arg1_359237], **kwargs_359238)
            

            if (may_be_358975 and more_types_in_union_358976):
                # SSA join for if statement (line 130)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 188)
        # Getting the type of 'shape' (line 188)
        shape_359240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'shape')
        # Getting the type of 'None' (line 188)
        None_359241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'None')
        
        (may_be_359242, more_types_in_union_359243) = may_not_be_none(shape_359240, None_359241)

        if may_be_359242:

            if more_types_in_union_359243:
                # Runtime conditional SSA (line 188)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 189):
            
            # Assigning a Name to a Attribute (line 189):
            # Getting the type of 'shape' (line 189)
            shape_359244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'shape')
            # Getting the type of 'self' (line 189)
            self_359245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'self')
            # Setting the type of the member 'shape' of a type (line 189)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), self_359245, 'shape', shape_359244)

            if more_types_in_union_359243:
                # Runtime conditional SSA for else branch (line 188)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_359242) or more_types_in_union_359243):
            
            # Type idiom detected: calculating its left and rigth part (line 191)
            # Getting the type of 'self' (line 191)
            self_359246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'self')
            # Obtaining the member 'shape' of a type (line 191)
            shape_359247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 15), self_359246, 'shape')
            # Getting the type of 'None' (line 191)
            None_359248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 29), 'None')
            
            (may_be_359249, more_types_in_union_359250) = may_be_none(shape_359247, None_359248)

            if may_be_359249:

                if more_types_in_union_359250:
                    # Runtime conditional SSA (line 191)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                
                # SSA begins for try-except statement (line 193)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                
                # Assigning a BinOp to a Name (line 194):
                
                # Assigning a BinOp to a Name (line 194):
                
                # Call to len(...): (line 194)
                # Processing the call arguments (line 194)
                # Getting the type of 'self' (line 194)
                self_359252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 28), 'self', False)
                # Obtaining the member 'indptr' of a type (line 194)
                indptr_359253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 28), self_359252, 'indptr')
                # Processing the call keyword arguments (line 194)
                kwargs_359254 = {}
                # Getting the type of 'len' (line 194)
                len_359251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 24), 'len', False)
                # Calling len(args, kwargs) (line 194)
                len_call_result_359255 = invoke(stypy.reporting.localization.Localization(__file__, 194, 24), len_359251, *[indptr_359253], **kwargs_359254)
                
                int_359256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 43), 'int')
                # Applying the binary operator '-' (line 194)
                result_sub_359257 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 24), '-', len_call_result_359255, int_359256)
                
                # Assigning a type to the variable 'M' (line 194)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 20), 'M', result_sub_359257)
                
                # Assigning a BinOp to a Name (line 195):
                
                # Assigning a BinOp to a Name (line 195):
                
                # Call to max(...): (line 195)
                # Processing the call keyword arguments (line 195)
                kwargs_359261 = {}
                # Getting the type of 'self' (line 195)
                self_359258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 24), 'self', False)
                # Obtaining the member 'indices' of a type (line 195)
                indices_359259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 24), self_359258, 'indices')
                # Obtaining the member 'max' of a type (line 195)
                max_359260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 24), indices_359259, 'max')
                # Calling max(args, kwargs) (line 195)
                max_call_result_359262 = invoke(stypy.reporting.localization.Localization(__file__, 195, 24), max_359260, *[], **kwargs_359261)
                
                int_359263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 45), 'int')
                # Applying the binary operator '+' (line 195)
                result_add_359264 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 24), '+', max_call_result_359262, int_359263)
                
                # Assigning a type to the variable 'N' (line 195)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'N', result_add_359264)
                # SSA branch for the except part of a try statement (line 193)
                # SSA branch for the except '<any exception>' branch of a try statement (line 193)
                module_type_store.open_ssa_branch('except')
                
                # Call to ValueError(...): (line 197)
                # Processing the call arguments (line 197)
                str_359266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 37), 'str', 'unable to infer matrix dimensions')
                # Processing the call keyword arguments (line 197)
                kwargs_359267 = {}
                # Getting the type of 'ValueError' (line 197)
                ValueError_359265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 26), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 197)
                ValueError_call_result_359268 = invoke(stypy.reporting.localization.Localization(__file__, 197, 26), ValueError_359265, *[str_359266], **kwargs_359267)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 197, 20), ValueError_call_result_359268, 'raise parameter', BaseException)
                # SSA branch for the else branch of a try statement (line 193)
                module_type_store.open_ssa_branch('except else')
                
                # Assigning a Attribute to a Tuple (line 199):
                
                # Assigning a Subscript to a Name (line 199):
                
                # Obtaining the type of the subscript
                int_359269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 20), 'int')
                # Getting the type of 'self' (line 199)
                self_359270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 26), 'self')
                # Obtaining the member 'blocksize' of a type (line 199)
                blocksize_359271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 26), self_359270, 'blocksize')
                # Obtaining the member '__getitem__' of a type (line 199)
                getitem___359272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 20), blocksize_359271, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 199)
                subscript_call_result_359273 = invoke(stypy.reporting.localization.Localization(__file__, 199, 20), getitem___359272, int_359269)
                
                # Assigning a type to the variable 'tuple_var_assignment_358863' (line 199)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'tuple_var_assignment_358863', subscript_call_result_359273)
                
                # Assigning a Subscript to a Name (line 199):
                
                # Obtaining the type of the subscript
                int_359274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 20), 'int')
                # Getting the type of 'self' (line 199)
                self_359275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 26), 'self')
                # Obtaining the member 'blocksize' of a type (line 199)
                blocksize_359276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 26), self_359275, 'blocksize')
                # Obtaining the member '__getitem__' of a type (line 199)
                getitem___359277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 20), blocksize_359276, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 199)
                subscript_call_result_359278 = invoke(stypy.reporting.localization.Localization(__file__, 199, 20), getitem___359277, int_359274)
                
                # Assigning a type to the variable 'tuple_var_assignment_358864' (line 199)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'tuple_var_assignment_358864', subscript_call_result_359278)
                
                # Assigning a Name to a Name (line 199):
                # Getting the type of 'tuple_var_assignment_358863' (line 199)
                tuple_var_assignment_358863_359279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'tuple_var_assignment_358863')
                # Assigning a type to the variable 'R' (line 199)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'R', tuple_var_assignment_358863_359279)
                
                # Assigning a Name to a Name (line 199):
                # Getting the type of 'tuple_var_assignment_358864' (line 199)
                tuple_var_assignment_358864_359280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'tuple_var_assignment_358864')
                # Assigning a type to the variable 'C' (line 199)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 22), 'C', tuple_var_assignment_358864_359280)
                
                # Assigning a Tuple to a Attribute (line 200):
                
                # Assigning a Tuple to a Attribute (line 200):
                
                # Obtaining an instance of the builtin type 'tuple' (line 200)
                tuple_359281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 34), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 200)
                # Adding element type (line 200)
                # Getting the type of 'M' (line 200)
                M_359282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 34), 'M')
                # Getting the type of 'R' (line 200)
                R_359283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'R')
                # Applying the binary operator '*' (line 200)
                result_mul_359284 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 34), '*', M_359282, R_359283)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_359281, result_mul_359284)
                # Adding element type (line 200)
                # Getting the type of 'N' (line 200)
                N_359285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 38), 'N')
                # Getting the type of 'C' (line 200)
                C_359286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 40), 'C')
                # Applying the binary operator '*' (line 200)
                result_mul_359287 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 38), '*', N_359285, C_359286)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_359281, result_mul_359287)
                
                # Getting the type of 'self' (line 200)
                self_359288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'self')
                # Setting the type of the member 'shape' of a type (line 200)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 20), self_359288, 'shape', tuple_359281)
                # SSA join for try-except statement (line 193)
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_359250:
                    # SSA join for if statement (line 191)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_359242 and more_types_in_union_359243):
                # SSA join for if statement (line 188)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 202)
        # Getting the type of 'self' (line 202)
        self_359289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'self')
        # Obtaining the member 'shape' of a type (line 202)
        shape_359290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 11), self_359289, 'shape')
        # Getting the type of 'None' (line 202)
        None_359291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 25), 'None')
        
        (may_be_359292, more_types_in_union_359293) = may_be_none(shape_359290, None_359291)

        if may_be_359292:

            if more_types_in_union_359293:
                # Runtime conditional SSA (line 202)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 203)
            # Getting the type of 'shape' (line 203)
            shape_359294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'shape')
            # Getting the type of 'None' (line 203)
            None_359295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 24), 'None')
            
            (may_be_359296, more_types_in_union_359297) = may_be_none(shape_359294, None_359295)

            if may_be_359296:

                if more_types_in_union_359297:
                    # Runtime conditional SSA (line 203)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to ValueError(...): (line 205)
                # Processing the call arguments (line 205)
                str_359299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 33), 'str', 'need to infer shape')
                # Processing the call keyword arguments (line 205)
                kwargs_359300 = {}
                # Getting the type of 'ValueError' (line 205)
                ValueError_359298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 22), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 205)
                ValueError_call_result_359301 = invoke(stypy.reporting.localization.Localization(__file__, 205, 22), ValueError_359298, *[str_359299], **kwargs_359300)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 205, 16), ValueError_call_result_359301, 'raise parameter', BaseException)

                if more_types_in_union_359297:
                    # Runtime conditional SSA for else branch (line 203)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_359296) or more_types_in_union_359297):
                
                # Assigning a Name to a Attribute (line 207):
                
                # Assigning a Name to a Attribute (line 207):
                # Getting the type of 'shape' (line 207)
                shape_359302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 29), 'shape')
                # Getting the type of 'self' (line 207)
                self_359303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'self')
                # Setting the type of the member 'shape' of a type (line 207)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), self_359303, 'shape', shape_359302)

                if (may_be_359296 and more_types_in_union_359297):
                    # SSA join for if statement (line 203)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_359293:
                # SSA join for if statement (line 202)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 209)
        # Getting the type of 'dtype' (line 209)
        dtype_359304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'dtype')
        # Getting the type of 'None' (line 209)
        None_359305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 24), 'None')
        
        (may_be_359306, more_types_in_union_359307) = may_not_be_none(dtype_359304, None_359305)

        if may_be_359306:

            if more_types_in_union_359307:
                # Runtime conditional SSA (line 209)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 210):
            
            # Assigning a Call to a Attribute (line 210):
            
            # Call to astype(...): (line 210)
            # Processing the call arguments (line 210)
            # Getting the type of 'dtype' (line 210)
            dtype_359311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 41), 'dtype', False)
            # Processing the call keyword arguments (line 210)
            kwargs_359312 = {}
            # Getting the type of 'self' (line 210)
            self_359308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 24), 'self', False)
            # Obtaining the member 'data' of a type (line 210)
            data_359309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 24), self_359308, 'data')
            # Obtaining the member 'astype' of a type (line 210)
            astype_359310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 24), data_359309, 'astype')
            # Calling astype(args, kwargs) (line 210)
            astype_call_result_359313 = invoke(stypy.reporting.localization.Localization(__file__, 210, 24), astype_359310, *[dtype_359311], **kwargs_359312)
            
            # Getting the type of 'self' (line 210)
            self_359314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'self')
            # Setting the type of the member 'data' of a type (line 210)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), self_359314, 'data', astype_call_result_359313)

            if more_types_in_union_359307:
                # SSA join for if statement (line 209)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to check_format(...): (line 212)
        # Processing the call keyword arguments (line 212)
        # Getting the type of 'False' (line 212)
        False_359317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 37), 'False', False)
        keyword_359318 = False_359317
        kwargs_359319 = {'full_check': keyword_359318}
        # Getting the type of 'self' (line 212)
        self_359315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'self', False)
        # Obtaining the member 'check_format' of a type (line 212)
        check_format_359316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), self_359315, 'check_format')
        # Calling check_format(args, kwargs) (line 212)
        check_format_call_result_359320 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), check_format_359316, *[], **kwargs_359319)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def check_format(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 214)
        True_359321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 38), 'True')
        defaults = [True_359321]
        # Create a new context for function 'check_format'
        module_type_store = module_type_store.open_function_context('check_format', 214, 4, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.check_format.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.check_format.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.check_format.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.check_format.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.check_format')
        bsr_matrix.check_format.__dict__.__setitem__('stypy_param_names_list', ['full_check'])
        bsr_matrix.check_format.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.check_format.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.check_format.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.check_format.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.check_format.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.check_format.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.check_format', ['full_check'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_format', localization, ['full_check'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_format(...)' code ##################

        str_359322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, (-1)), 'str', 'check whether the matrix format is valid\n\n            *Parameters*:\n                full_check:\n                    True  - rigorous check, O(N) operations : default\n                    False - basic check, O(1) operations\n\n        ')
        
        # Assigning a Attribute to a Tuple (line 223):
        
        # Assigning a Subscript to a Name (line 223):
        
        # Obtaining the type of the subscript
        int_359323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 8), 'int')
        # Getting the type of 'self' (line 223)
        self_359324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 14), 'self')
        # Obtaining the member 'shape' of a type (line 223)
        shape_359325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 14), self_359324, 'shape')
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___359326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), shape_359325, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_359327 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), getitem___359326, int_359323)
        
        # Assigning a type to the variable 'tuple_var_assignment_358865' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_358865', subscript_call_result_359327)
        
        # Assigning a Subscript to a Name (line 223):
        
        # Obtaining the type of the subscript
        int_359328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 8), 'int')
        # Getting the type of 'self' (line 223)
        self_359329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 14), 'self')
        # Obtaining the member 'shape' of a type (line 223)
        shape_359330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 14), self_359329, 'shape')
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___359331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), shape_359330, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_359332 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), getitem___359331, int_359328)
        
        # Assigning a type to the variable 'tuple_var_assignment_358866' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_358866', subscript_call_result_359332)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'tuple_var_assignment_358865' (line 223)
        tuple_var_assignment_358865_359333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_358865')
        # Assigning a type to the variable 'M' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'M', tuple_var_assignment_358865_359333)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'tuple_var_assignment_358866' (line 223)
        tuple_var_assignment_358866_359334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_358866')
        # Assigning a type to the variable 'N' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 10), 'N', tuple_var_assignment_358866_359334)
        
        # Assigning a Attribute to a Tuple (line 224):
        
        # Assigning a Subscript to a Name (line 224):
        
        # Obtaining the type of the subscript
        int_359335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 8), 'int')
        # Getting the type of 'self' (line 224)
        self_359336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 224)
        blocksize_359337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 14), self_359336, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 224)
        getitem___359338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), blocksize_359337, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 224)
        subscript_call_result_359339 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), getitem___359338, int_359335)
        
        # Assigning a type to the variable 'tuple_var_assignment_358867' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_358867', subscript_call_result_359339)
        
        # Assigning a Subscript to a Name (line 224):
        
        # Obtaining the type of the subscript
        int_359340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 8), 'int')
        # Getting the type of 'self' (line 224)
        self_359341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 224)
        blocksize_359342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 14), self_359341, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 224)
        getitem___359343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), blocksize_359342, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 224)
        subscript_call_result_359344 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), getitem___359343, int_359340)
        
        # Assigning a type to the variable 'tuple_var_assignment_358868' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_358868', subscript_call_result_359344)
        
        # Assigning a Name to a Name (line 224):
        # Getting the type of 'tuple_var_assignment_358867' (line 224)
        tuple_var_assignment_358867_359345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_358867')
        # Assigning a type to the variable 'R' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'R', tuple_var_assignment_358867_359345)
        
        # Assigning a Name to a Name (line 224):
        # Getting the type of 'tuple_var_assignment_358868' (line 224)
        tuple_var_assignment_358868_359346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_358868')
        # Assigning a type to the variable 'C' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 10), 'C', tuple_var_assignment_358868_359346)
        
        
        # Getting the type of 'self' (line 227)
        self_359347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'self')
        # Obtaining the member 'indptr' of a type (line 227)
        indptr_359348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 11), self_359347, 'indptr')
        # Obtaining the member 'dtype' of a type (line 227)
        dtype_359349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 11), indptr_359348, 'dtype')
        # Obtaining the member 'kind' of a type (line 227)
        kind_359350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 11), dtype_359349, 'kind')
        str_359351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 37), 'str', 'i')
        # Applying the binary operator '!=' (line 227)
        result_ne_359352 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 11), '!=', kind_359350, str_359351)
        
        # Testing the type of an if condition (line 227)
        if_condition_359353 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 8), result_ne_359352)
        # Assigning a type to the variable 'if_condition_359353' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'if_condition_359353', if_condition_359353)
        # SSA begins for if statement (line 227)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 228)
        # Processing the call arguments (line 228)
        str_359355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 17), 'str', 'indptr array has non-integer dtype (%s)')
        # Getting the type of 'self' (line 229)
        self_359356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 22), 'self', False)
        # Obtaining the member 'indptr' of a type (line 229)
        indptr_359357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 22), self_359356, 'indptr')
        # Obtaining the member 'dtype' of a type (line 229)
        dtype_359358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 22), indptr_359357, 'dtype')
        # Obtaining the member 'name' of a type (line 229)
        name_359359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 22), dtype_359358, 'name')
        # Applying the binary operator '%' (line 228)
        result_mod_359360 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 17), '%', str_359355, name_359359)
        
        # Processing the call keyword arguments (line 228)
        kwargs_359361 = {}
        # Getting the type of 'warn' (line 228)
        warn_359354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'warn', False)
        # Calling warn(args, kwargs) (line 228)
        warn_call_result_359362 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), warn_359354, *[result_mod_359360], **kwargs_359361)
        
        # SSA join for if statement (line 227)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 230)
        self_359363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), 'self')
        # Obtaining the member 'indices' of a type (line 230)
        indices_359364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 11), self_359363, 'indices')
        # Obtaining the member 'dtype' of a type (line 230)
        dtype_359365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 11), indices_359364, 'dtype')
        # Obtaining the member 'kind' of a type (line 230)
        kind_359366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 11), dtype_359365, 'kind')
        str_359367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 38), 'str', 'i')
        # Applying the binary operator '!=' (line 230)
        result_ne_359368 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 11), '!=', kind_359366, str_359367)
        
        # Testing the type of an if condition (line 230)
        if_condition_359369 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 8), result_ne_359368)
        # Assigning a type to the variable 'if_condition_359369' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'if_condition_359369', if_condition_359369)
        # SSA begins for if statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 231)
        # Processing the call arguments (line 231)
        str_359371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 17), 'str', 'indices array has non-integer dtype (%s)')
        # Getting the type of 'self' (line 232)
        self_359372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 22), 'self', False)
        # Obtaining the member 'indices' of a type (line 232)
        indices_359373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 22), self_359372, 'indices')
        # Obtaining the member 'dtype' of a type (line 232)
        dtype_359374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 22), indices_359373, 'dtype')
        # Obtaining the member 'name' of a type (line 232)
        name_359375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 22), dtype_359374, 'name')
        # Applying the binary operator '%' (line 231)
        result_mod_359376 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 17), '%', str_359371, name_359375)
        
        # Processing the call keyword arguments (line 231)
        kwargs_359377 = {}
        # Getting the type of 'warn' (line 231)
        warn_359370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'warn', False)
        # Calling warn(args, kwargs) (line 231)
        warn_call_result_359378 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), warn_359370, *[result_mod_359376], **kwargs_359377)
        
        # SSA join for if statement (line 230)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 234):
        
        # Assigning a Call to a Name (line 234):
        
        # Call to get_index_dtype(...): (line 234)
        # Processing the call arguments (line 234)
        
        # Obtaining an instance of the builtin type 'tuple' (line 234)
        tuple_359380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 234)
        # Adding element type (line 234)
        # Getting the type of 'self' (line 234)
        self_359381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 37), 'self', False)
        # Obtaining the member 'indices' of a type (line 234)
        indices_359382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 37), self_359381, 'indices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 37), tuple_359380, indices_359382)
        # Adding element type (line 234)
        # Getting the type of 'self' (line 234)
        self_359383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 51), 'self', False)
        # Obtaining the member 'indptr' of a type (line 234)
        indptr_359384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 51), self_359383, 'indptr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 37), tuple_359380, indptr_359384)
        
        # Processing the call keyword arguments (line 234)
        kwargs_359385 = {}
        # Getting the type of 'get_index_dtype' (line 234)
        get_index_dtype_359379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 20), 'get_index_dtype', False)
        # Calling get_index_dtype(args, kwargs) (line 234)
        get_index_dtype_call_result_359386 = invoke(stypy.reporting.localization.Localization(__file__, 234, 20), get_index_dtype_359379, *[tuple_359380], **kwargs_359385)
        
        # Assigning a type to the variable 'idx_dtype' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'idx_dtype', get_index_dtype_call_result_359386)
        
        # Assigning a Call to a Attribute (line 235):
        
        # Assigning a Call to a Attribute (line 235):
        
        # Call to asarray(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'self' (line 235)
        self_359389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 33), 'self', False)
        # Obtaining the member 'indptr' of a type (line 235)
        indptr_359390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 33), self_359389, 'indptr')
        # Processing the call keyword arguments (line 235)
        # Getting the type of 'idx_dtype' (line 235)
        idx_dtype_359391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 52), 'idx_dtype', False)
        keyword_359392 = idx_dtype_359391
        kwargs_359393 = {'dtype': keyword_359392}
        # Getting the type of 'np' (line 235)
        np_359387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 22), 'np', False)
        # Obtaining the member 'asarray' of a type (line 235)
        asarray_359388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 22), np_359387, 'asarray')
        # Calling asarray(args, kwargs) (line 235)
        asarray_call_result_359394 = invoke(stypy.reporting.localization.Localization(__file__, 235, 22), asarray_359388, *[indptr_359390], **kwargs_359393)
        
        # Getting the type of 'self' (line 235)
        self_359395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self')
        # Setting the type of the member 'indptr' of a type (line 235)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_359395, 'indptr', asarray_call_result_359394)
        
        # Assigning a Call to a Attribute (line 236):
        
        # Assigning a Call to a Attribute (line 236):
        
        # Call to asarray(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'self' (line 236)
        self_359398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 34), 'self', False)
        # Obtaining the member 'indices' of a type (line 236)
        indices_359399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 34), self_359398, 'indices')
        # Processing the call keyword arguments (line 236)
        # Getting the type of 'idx_dtype' (line 236)
        idx_dtype_359400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 54), 'idx_dtype', False)
        keyword_359401 = idx_dtype_359400
        kwargs_359402 = {'dtype': keyword_359401}
        # Getting the type of 'np' (line 236)
        np_359396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 23), 'np', False)
        # Obtaining the member 'asarray' of a type (line 236)
        asarray_359397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 23), np_359396, 'asarray')
        # Calling asarray(args, kwargs) (line 236)
        asarray_call_result_359403 = invoke(stypy.reporting.localization.Localization(__file__, 236, 23), asarray_359397, *[indices_359399], **kwargs_359402)
        
        # Getting the type of 'self' (line 236)
        self_359404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self')
        # Setting the type of the member 'indices' of a type (line 236)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), self_359404, 'indices', asarray_call_result_359403)
        
        # Assigning a Call to a Attribute (line 237):
        
        # Assigning a Call to a Attribute (line 237):
        
        # Call to to_native(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'self' (line 237)
        self_359406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 30), 'self', False)
        # Obtaining the member 'data' of a type (line 237)
        data_359407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 30), self_359406, 'data')
        # Processing the call keyword arguments (line 237)
        kwargs_359408 = {}
        # Getting the type of 'to_native' (line 237)
        to_native_359405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'to_native', False)
        # Calling to_native(args, kwargs) (line 237)
        to_native_call_result_359409 = invoke(stypy.reporting.localization.Localization(__file__, 237, 20), to_native_359405, *[data_359407], **kwargs_359408)
        
        # Getting the type of 'self' (line 237)
        self_359410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'self')
        # Setting the type of the member 'data' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), self_359410, 'data', to_native_call_result_359409)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 240)
        self_359411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'self')
        # Obtaining the member 'indices' of a type (line 240)
        indices_359412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 11), self_359411, 'indices')
        # Obtaining the member 'ndim' of a type (line 240)
        ndim_359413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 11), indices_359412, 'ndim')
        int_359414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 32), 'int')
        # Applying the binary operator '!=' (line 240)
        result_ne_359415 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 11), '!=', ndim_359413, int_359414)
        
        
        # Getting the type of 'self' (line 240)
        self_359416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 37), 'self')
        # Obtaining the member 'indptr' of a type (line 240)
        indptr_359417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 37), self_359416, 'indptr')
        # Obtaining the member 'ndim' of a type (line 240)
        ndim_359418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 37), indptr_359417, 'ndim')
        int_359419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 57), 'int')
        # Applying the binary operator '!=' (line 240)
        result_ne_359420 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 37), '!=', ndim_359418, int_359419)
        
        # Applying the binary operator 'or' (line 240)
        result_or_keyword_359421 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 11), 'or', result_ne_359415, result_ne_359420)
        
        # Testing the type of an if condition (line 240)
        if_condition_359422 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), result_or_keyword_359421)
        # Assigning a type to the variable 'if_condition_359422' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_359422', if_condition_359422)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 241)
        # Processing the call arguments (line 241)
        str_359424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 29), 'str', 'indices, and indptr should be 1-D')
        # Processing the call keyword arguments (line 241)
        kwargs_359425 = {}
        # Getting the type of 'ValueError' (line 241)
        ValueError_359423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 241)
        ValueError_call_result_359426 = invoke(stypy.reporting.localization.Localization(__file__, 241, 18), ValueError_359423, *[str_359424], **kwargs_359425)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 241, 12), ValueError_call_result_359426, 'raise parameter', BaseException)
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 242)
        self_359427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'self')
        # Obtaining the member 'data' of a type (line 242)
        data_359428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 11), self_359427, 'data')
        # Obtaining the member 'ndim' of a type (line 242)
        ndim_359429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 11), data_359428, 'ndim')
        int_359430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 29), 'int')
        # Applying the binary operator '!=' (line 242)
        result_ne_359431 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 11), '!=', ndim_359429, int_359430)
        
        # Testing the type of an if condition (line 242)
        if_condition_359432 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 8), result_ne_359431)
        # Assigning a type to the variable 'if_condition_359432' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'if_condition_359432', if_condition_359432)
        # SSA begins for if statement (line 242)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 243)
        # Processing the call arguments (line 243)
        str_359434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 29), 'str', 'data should be 3-D')
        # Processing the call keyword arguments (line 243)
        kwargs_359435 = {}
        # Getting the type of 'ValueError' (line 243)
        ValueError_359433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 243)
        ValueError_call_result_359436 = invoke(stypy.reporting.localization.Localization(__file__, 243, 18), ValueError_359433, *[str_359434], **kwargs_359435)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 243, 12), ValueError_call_result_359436, 'raise parameter', BaseException)
        # SSA join for if statement (line 242)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'self' (line 246)
        self_359438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'self', False)
        # Obtaining the member 'indptr' of a type (line 246)
        indptr_359439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), self_359438, 'indptr')
        # Processing the call keyword arguments (line 246)
        kwargs_359440 = {}
        # Getting the type of 'len' (line 246)
        len_359437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'len', False)
        # Calling len(args, kwargs) (line 246)
        len_call_result_359441 = invoke(stypy.reporting.localization.Localization(__file__, 246, 12), len_359437, *[indptr_359439], **kwargs_359440)
        
        # Getting the type of 'M' (line 246)
        M_359442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 32), 'M')
        # Getting the type of 'R' (line 246)
        R_359443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 35), 'R')
        # Applying the binary operator '//' (line 246)
        result_floordiv_359444 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 32), '//', M_359442, R_359443)
        
        int_359445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 39), 'int')
        # Applying the binary operator '+' (line 246)
        result_add_359446 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 32), '+', result_floordiv_359444, int_359445)
        
        # Applying the binary operator '!=' (line 246)
        result_ne_359447 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 12), '!=', len_call_result_359441, result_add_359446)
        
        # Testing the type of an if condition (line 246)
        if_condition_359448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 8), result_ne_359447)
        # Assigning a type to the variable 'if_condition_359448' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'if_condition_359448', if_condition_359448)
        # SSA begins for if statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 247)
        # Processing the call arguments (line 247)
        str_359450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 29), 'str', 'index pointer size (%d) should be (%d)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 248)
        tuple_359451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 248)
        # Adding element type (line 248)
        
        # Call to len(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'self' (line 248)
        self_359453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 37), 'self', False)
        # Obtaining the member 'indptr' of a type (line 248)
        indptr_359454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 37), self_359453, 'indptr')
        # Processing the call keyword arguments (line 248)
        kwargs_359455 = {}
        # Getting the type of 'len' (line 248)
        len_359452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 33), 'len', False)
        # Calling len(args, kwargs) (line 248)
        len_call_result_359456 = invoke(stypy.reporting.localization.Localization(__file__, 248, 33), len_359452, *[indptr_359454], **kwargs_359455)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 33), tuple_359451, len_call_result_359456)
        # Adding element type (line 248)
        # Getting the type of 'M' (line 248)
        M_359457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 51), 'M', False)
        # Getting the type of 'R' (line 248)
        R_359458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 54), 'R', False)
        # Applying the binary operator '//' (line 248)
        result_floordiv_359459 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 51), '//', M_359457, R_359458)
        
        int_359460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 58), 'int')
        # Applying the binary operator '+' (line 248)
        result_add_359461 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 51), '+', result_floordiv_359459, int_359460)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 33), tuple_359451, result_add_359461)
        
        # Applying the binary operator '%' (line 247)
        result_mod_359462 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 29), '%', str_359450, tuple_359451)
        
        # Processing the call keyword arguments (line 247)
        kwargs_359463 = {}
        # Getting the type of 'ValueError' (line 247)
        ValueError_359449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 247)
        ValueError_call_result_359464 = invoke(stypy.reporting.localization.Localization(__file__, 247, 18), ValueError_359449, *[result_mod_359462], **kwargs_359463)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 247, 12), ValueError_call_result_359464, 'raise parameter', BaseException)
        # SSA join for if statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_359465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 24), 'int')
        # Getting the type of 'self' (line 249)
        self_359466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'self')
        # Obtaining the member 'indptr' of a type (line 249)
        indptr_359467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 12), self_359466, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 249)
        getitem___359468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 12), indptr_359467, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 249)
        subscript_call_result_359469 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), getitem___359468, int_359465)
        
        int_359470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 30), 'int')
        # Applying the binary operator '!=' (line 249)
        result_ne_359471 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 12), '!=', subscript_call_result_359469, int_359470)
        
        # Testing the type of an if condition (line 249)
        if_condition_359472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 8), result_ne_359471)
        # Assigning a type to the variable 'if_condition_359472' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'if_condition_359472', if_condition_359472)
        # SSA begins for if statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 250)
        # Processing the call arguments (line 250)
        str_359474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 29), 'str', 'index pointer should start with 0')
        # Processing the call keyword arguments (line 250)
        kwargs_359475 = {}
        # Getting the type of 'ValueError' (line 250)
        ValueError_359473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 250)
        ValueError_call_result_359476 = invoke(stypy.reporting.localization.Localization(__file__, 250, 18), ValueError_359473, *[str_359474], **kwargs_359475)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 250, 12), ValueError_call_result_359476, 'raise parameter', BaseException)
        # SSA join for if statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'self' (line 253)
        self_359478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'self', False)
        # Obtaining the member 'indices' of a type (line 253)
        indices_359479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 16), self_359478, 'indices')
        # Processing the call keyword arguments (line 253)
        kwargs_359480 = {}
        # Getting the type of 'len' (line 253)
        len_359477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'len', False)
        # Calling len(args, kwargs) (line 253)
        len_call_result_359481 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), len_359477, *[indices_359479], **kwargs_359480)
        
        
        # Call to len(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'self' (line 253)
        self_359483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 37), 'self', False)
        # Obtaining the member 'data' of a type (line 253)
        data_359484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 37), self_359483, 'data')
        # Processing the call keyword arguments (line 253)
        kwargs_359485 = {}
        # Getting the type of 'len' (line 253)
        len_359482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 33), 'len', False)
        # Calling len(args, kwargs) (line 253)
        len_call_result_359486 = invoke(stypy.reporting.localization.Localization(__file__, 253, 33), len_359482, *[data_359484], **kwargs_359485)
        
        # Applying the binary operator '!=' (line 253)
        result_ne_359487 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 12), '!=', len_call_result_359481, len_call_result_359486)
        
        # Testing the type of an if condition (line 253)
        if_condition_359488 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 8), result_ne_359487)
        # Assigning a type to the variable 'if_condition_359488' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'if_condition_359488', if_condition_359488)
        # SSA begins for if statement (line 253)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 254)
        # Processing the call arguments (line 254)
        str_359490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 29), 'str', 'indices and data should have the same size')
        # Processing the call keyword arguments (line 254)
        kwargs_359491 = {}
        # Getting the type of 'ValueError' (line 254)
        ValueError_359489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 254)
        ValueError_call_result_359492 = invoke(stypy.reporting.localization.Localization(__file__, 254, 18), ValueError_359489, *[str_359490], **kwargs_359491)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 254, 12), ValueError_call_result_359492, 'raise parameter', BaseException)
        # SSA join for if statement (line 253)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_359493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 24), 'int')
        # Getting the type of 'self' (line 255)
        self_359494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'self')
        # Obtaining the member 'indptr' of a type (line 255)
        indptr_359495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), self_359494, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 255)
        getitem___359496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), indptr_359495, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 255)
        subscript_call_result_359497 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), getitem___359496, int_359493)
        
        
        # Call to len(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'self' (line 255)
        self_359499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 34), 'self', False)
        # Obtaining the member 'indices' of a type (line 255)
        indices_359500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 34), self_359499, 'indices')
        # Processing the call keyword arguments (line 255)
        kwargs_359501 = {}
        # Getting the type of 'len' (line 255)
        len_359498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 30), 'len', False)
        # Calling len(args, kwargs) (line 255)
        len_call_result_359502 = invoke(stypy.reporting.localization.Localization(__file__, 255, 30), len_359498, *[indices_359500], **kwargs_359501)
        
        # Applying the binary operator '>' (line 255)
        result_gt_359503 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 12), '>', subscript_call_result_359497, len_call_result_359502)
        
        # Testing the type of an if condition (line 255)
        if_condition_359504 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 8), result_gt_359503)
        # Assigning a type to the variable 'if_condition_359504' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'if_condition_359504', if_condition_359504)
        # SSA begins for if statement (line 255)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 256)
        # Processing the call arguments (line 256)
        str_359506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 29), 'str', 'Last value of index pointer should be less than the size of index and data arrays')
        # Processing the call keyword arguments (line 256)
        kwargs_359507 = {}
        # Getting the type of 'ValueError' (line 256)
        ValueError_359505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 256)
        ValueError_call_result_359508 = invoke(stypy.reporting.localization.Localization(__file__, 256, 18), ValueError_359505, *[str_359506], **kwargs_359507)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 256, 12), ValueError_call_result_359508, 'raise parameter', BaseException)
        # SSA join for if statement (line 255)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to prune(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_359511 = {}
        # Getting the type of 'self' (line 259)
        self_359509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'self', False)
        # Obtaining the member 'prune' of a type (line 259)
        prune_359510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), self_359509, 'prune')
        # Calling prune(args, kwargs) (line 259)
        prune_call_result_359512 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), prune_359510, *[], **kwargs_359511)
        
        
        # Getting the type of 'full_check' (line 261)
        full_check_359513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'full_check')
        # Testing the type of an if condition (line 261)
        if_condition_359514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 8), full_check_359513)
        # Assigning a type to the variable 'if_condition_359514' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'if_condition_359514', if_condition_359514)
        # SSA begins for if statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 263)
        self_359515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'self')
        # Obtaining the member 'nnz' of a type (line 263)
        nnz_359516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 15), self_359515, 'nnz')
        int_359517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 26), 'int')
        # Applying the binary operator '>' (line 263)
        result_gt_359518 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 15), '>', nnz_359516, int_359517)
        
        # Testing the type of an if condition (line 263)
        if_condition_359519 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 12), result_gt_359518)
        # Assigning a type to the variable 'if_condition_359519' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'if_condition_359519', if_condition_359519)
        # SSA begins for if statement (line 263)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to max(...): (line 264)
        # Processing the call keyword arguments (line 264)
        kwargs_359523 = {}
        # Getting the type of 'self' (line 264)
        self_359520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 19), 'self', False)
        # Obtaining the member 'indices' of a type (line 264)
        indices_359521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 19), self_359520, 'indices')
        # Obtaining the member 'max' of a type (line 264)
        max_359522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 19), indices_359521, 'max')
        # Calling max(args, kwargs) (line 264)
        max_call_result_359524 = invoke(stypy.reporting.localization.Localization(__file__, 264, 19), max_359522, *[], **kwargs_359523)
        
        # Getting the type of 'N' (line 264)
        N_359525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 41), 'N')
        # Getting the type of 'C' (line 264)
        C_359526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 44), 'C')
        # Applying the binary operator '//' (line 264)
        result_floordiv_359527 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 41), '//', N_359525, C_359526)
        
        # Applying the binary operator '>=' (line 264)
        result_ge_359528 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 19), '>=', max_call_result_359524, result_floordiv_359527)
        
        # Testing the type of an if condition (line 264)
        if_condition_359529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 264, 16), result_ge_359528)
        # Assigning a type to the variable 'if_condition_359529' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'if_condition_359529', if_condition_359529)
        # SSA begins for if statement (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 265)
        # Processing the call arguments (line 265)
        str_359531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 37), 'str', 'column index values must be < %d (now max %d)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 265)
        tuple_359532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 88), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 265)
        # Adding element type (line 265)
        # Getting the type of 'N' (line 265)
        N_359533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 88), 'N', False)
        # Getting the type of 'C' (line 265)
        C_359534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 91), 'C', False)
        # Applying the binary operator '//' (line 265)
        result_floordiv_359535 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 88), '//', N_359533, C_359534)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 88), tuple_359532, result_floordiv_359535)
        # Adding element type (line 265)
        
        # Call to max(...): (line 265)
        # Processing the call keyword arguments (line 265)
        kwargs_359539 = {}
        # Getting the type of 'self' (line 265)
        self_359536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 94), 'self', False)
        # Obtaining the member 'indices' of a type (line 265)
        indices_359537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 94), self_359536, 'indices')
        # Obtaining the member 'max' of a type (line 265)
        max_359538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 94), indices_359537, 'max')
        # Calling max(args, kwargs) (line 265)
        max_call_result_359540 = invoke(stypy.reporting.localization.Localization(__file__, 265, 94), max_359538, *[], **kwargs_359539)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 88), tuple_359532, max_call_result_359540)
        
        # Applying the binary operator '%' (line 265)
        result_mod_359541 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 37), '%', str_359531, tuple_359532)
        
        # Processing the call keyword arguments (line 265)
        kwargs_359542 = {}
        # Getting the type of 'ValueError' (line 265)
        ValueError_359530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 265)
        ValueError_call_result_359543 = invoke(stypy.reporting.localization.Localization(__file__, 265, 26), ValueError_359530, *[result_mod_359541], **kwargs_359542)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 265, 20), ValueError_call_result_359543, 'raise parameter', BaseException)
        # SSA join for if statement (line 264)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to min(...): (line 266)
        # Processing the call keyword arguments (line 266)
        kwargs_359547 = {}
        # Getting the type of 'self' (line 266)
        self_359544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 19), 'self', False)
        # Obtaining the member 'indices' of a type (line 266)
        indices_359545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 19), self_359544, 'indices')
        # Obtaining the member 'min' of a type (line 266)
        min_359546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 19), indices_359545, 'min')
        # Calling min(args, kwargs) (line 266)
        min_call_result_359548 = invoke(stypy.reporting.localization.Localization(__file__, 266, 19), min_359546, *[], **kwargs_359547)
        
        int_359549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 40), 'int')
        # Applying the binary operator '<' (line 266)
        result_lt_359550 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 19), '<', min_call_result_359548, int_359549)
        
        # Testing the type of an if condition (line 266)
        if_condition_359551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 16), result_lt_359550)
        # Assigning a type to the variable 'if_condition_359551' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'if_condition_359551', if_condition_359551)
        # SSA begins for if statement (line 266)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 267)
        # Processing the call arguments (line 267)
        str_359553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 37), 'str', 'column index values must be >= 0')
        # Processing the call keyword arguments (line 267)
        kwargs_359554 = {}
        # Getting the type of 'ValueError' (line 267)
        ValueError_359552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 267)
        ValueError_call_result_359555 = invoke(stypy.reporting.localization.Localization(__file__, 267, 26), ValueError_359552, *[str_359553], **kwargs_359554)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 267, 20), ValueError_call_result_359555, 'raise parameter', BaseException)
        # SSA join for if statement (line 266)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to min(...): (line 268)
        # Processing the call keyword arguments (line 268)
        kwargs_359563 = {}
        
        # Call to diff(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'self' (line 268)
        self_359558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 27), 'self', False)
        # Obtaining the member 'indptr' of a type (line 268)
        indptr_359559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 27), self_359558, 'indptr')
        # Processing the call keyword arguments (line 268)
        kwargs_359560 = {}
        # Getting the type of 'np' (line 268)
        np_359556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'np', False)
        # Obtaining the member 'diff' of a type (line 268)
        diff_359557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 19), np_359556, 'diff')
        # Calling diff(args, kwargs) (line 268)
        diff_call_result_359561 = invoke(stypy.reporting.localization.Localization(__file__, 268, 19), diff_359557, *[indptr_359559], **kwargs_359560)
        
        # Obtaining the member 'min' of a type (line 268)
        min_359562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 19), diff_call_result_359561, 'min')
        # Calling min(args, kwargs) (line 268)
        min_call_result_359564 = invoke(stypy.reporting.localization.Localization(__file__, 268, 19), min_359562, *[], **kwargs_359563)
        
        int_359565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 48), 'int')
        # Applying the binary operator '<' (line 268)
        result_lt_359566 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 19), '<', min_call_result_359564, int_359565)
        
        # Testing the type of an if condition (line 268)
        if_condition_359567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 16), result_lt_359566)
        # Assigning a type to the variable 'if_condition_359567' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'if_condition_359567', if_condition_359567)
        # SSA begins for if statement (line 268)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 269)
        # Processing the call arguments (line 269)
        str_359569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 37), 'str', 'index pointer values must form a non-decreasing sequence')
        # Processing the call keyword arguments (line 269)
        kwargs_359570 = {}
        # Getting the type of 'ValueError' (line 269)
        ValueError_359568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 269)
        ValueError_call_result_359571 = invoke(stypy.reporting.localization.Localization(__file__, 269, 26), ValueError_359568, *[str_359569], **kwargs_359570)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 269, 20), ValueError_call_result_359571, 'raise parameter', BaseException)
        # SSA join for if statement (line 268)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 263)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 261)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_format(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_format' in the type store
        # Getting the type of 'stypy_return_type' (line 214)
        stypy_return_type_359572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_359572)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_format'
        return stypy_return_type_359572


    @norecursion
    def _get_blocksize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_blocksize'
        module_type_store = module_type_store.open_function_context('_get_blocksize', 276, 4, False)
        # Assigning a type to the variable 'self' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix._get_blocksize.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix._get_blocksize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix._get_blocksize.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix._get_blocksize.__dict__.__setitem__('stypy_function_name', 'bsr_matrix._get_blocksize')
        bsr_matrix._get_blocksize.__dict__.__setitem__('stypy_param_names_list', [])
        bsr_matrix._get_blocksize.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix._get_blocksize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix._get_blocksize.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix._get_blocksize.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix._get_blocksize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix._get_blocksize.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix._get_blocksize', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_blocksize', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_blocksize(...)' code ##################

        
        # Obtaining the type of the subscript
        int_359573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 31), 'int')
        slice_359574 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 277, 15), int_359573, None, None)
        # Getting the type of 'self' (line 277)
        self_359575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 15), 'self')
        # Obtaining the member 'data' of a type (line 277)
        data_359576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 15), self_359575, 'data')
        # Obtaining the member 'shape' of a type (line 277)
        shape_359577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 15), data_359576, 'shape')
        # Obtaining the member '__getitem__' of a type (line 277)
        getitem___359578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 15), shape_359577, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 277)
        subscript_call_result_359579 = invoke(stypy.reporting.localization.Localization(__file__, 277, 15), getitem___359578, slice_359574)
        
        # Assigning a type to the variable 'stypy_return_type' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'stypy_return_type', subscript_call_result_359579)
        
        # ################# End of '_get_blocksize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_blocksize' in the type store
        # Getting the type of 'stypy_return_type' (line 276)
        stypy_return_type_359580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_359580)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_blocksize'
        return stypy_return_type_359580

    
    # Assigning a Call to a Name (line 278):

    @norecursion
    def getnnz(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 280)
        None_359581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 26), 'None')
        defaults = [None_359581]
        # Create a new context for function 'getnnz'
        module_type_store = module_type_store.open_function_context('getnnz', 280, 4, False)
        # Assigning a type to the variable 'self' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.getnnz.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.getnnz.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.getnnz.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.getnnz.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.getnnz')
        bsr_matrix.getnnz.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        bsr_matrix.getnnz.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.getnnz.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.getnnz.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.getnnz.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.getnnz.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.getnnz.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.getnnz', ['axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getnnz', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getnnz(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 281)
        # Getting the type of 'axis' (line 281)
        axis_359582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'axis')
        # Getting the type of 'None' (line 281)
        None_359583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 23), 'None')
        
        (may_be_359584, more_types_in_union_359585) = may_not_be_none(axis_359582, None_359583)

        if may_be_359584:

            if more_types_in_union_359585:
                # Runtime conditional SSA (line 281)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to NotImplementedError(...): (line 282)
            # Processing the call arguments (line 282)
            str_359587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 38), 'str', 'getnnz over an axis is not implemented for BSR format')
            # Processing the call keyword arguments (line 282)
            kwargs_359588 = {}
            # Getting the type of 'NotImplementedError' (line 282)
            NotImplementedError_359586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 18), 'NotImplementedError', False)
            # Calling NotImplementedError(args, kwargs) (line 282)
            NotImplementedError_call_result_359589 = invoke(stypy.reporting.localization.Localization(__file__, 282, 18), NotImplementedError_359586, *[str_359587], **kwargs_359588)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 282, 12), NotImplementedError_call_result_359589, 'raise parameter', BaseException)

            if more_types_in_union_359585:
                # SSA join for if statement (line 281)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Tuple (line 284):
        
        # Assigning a Subscript to a Name (line 284):
        
        # Obtaining the type of the subscript
        int_359590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 8), 'int')
        # Getting the type of 'self' (line 284)
        self_359591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 284)
        blocksize_359592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 14), self_359591, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 284)
        getitem___359593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), blocksize_359592, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 284)
        subscript_call_result_359594 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), getitem___359593, int_359590)
        
        # Assigning a type to the variable 'tuple_var_assignment_358869' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'tuple_var_assignment_358869', subscript_call_result_359594)
        
        # Assigning a Subscript to a Name (line 284):
        
        # Obtaining the type of the subscript
        int_359595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 8), 'int')
        # Getting the type of 'self' (line 284)
        self_359596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 284)
        blocksize_359597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 14), self_359596, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 284)
        getitem___359598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), blocksize_359597, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 284)
        subscript_call_result_359599 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), getitem___359598, int_359595)
        
        # Assigning a type to the variable 'tuple_var_assignment_358870' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'tuple_var_assignment_358870', subscript_call_result_359599)
        
        # Assigning a Name to a Name (line 284):
        # Getting the type of 'tuple_var_assignment_358869' (line 284)
        tuple_var_assignment_358869_359600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'tuple_var_assignment_358869')
        # Assigning a type to the variable 'R' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'R', tuple_var_assignment_358869_359600)
        
        # Assigning a Name to a Name (line 284):
        # Getting the type of 'tuple_var_assignment_358870' (line 284)
        tuple_var_assignment_358870_359601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'tuple_var_assignment_358870')
        # Assigning a type to the variable 'C' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 10), 'C', tuple_var_assignment_358870_359601)
        
        # Call to int(...): (line 285)
        # Processing the call arguments (line 285)
        
        # Obtaining the type of the subscript
        int_359603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 31), 'int')
        # Getting the type of 'self' (line 285)
        self_359604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 19), 'self', False)
        # Obtaining the member 'indptr' of a type (line 285)
        indptr_359605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 19), self_359604, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 285)
        getitem___359606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 19), indptr_359605, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 285)
        subscript_call_result_359607 = invoke(stypy.reporting.localization.Localization(__file__, 285, 19), getitem___359606, int_359603)
        
        # Getting the type of 'R' (line 285)
        R_359608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 37), 'R', False)
        # Applying the binary operator '*' (line 285)
        result_mul_359609 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 19), '*', subscript_call_result_359607, R_359608)
        
        # Getting the type of 'C' (line 285)
        C_359610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 41), 'C', False)
        # Applying the binary operator '*' (line 285)
        result_mul_359611 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 39), '*', result_mul_359609, C_359610)
        
        # Processing the call keyword arguments (line 285)
        kwargs_359612 = {}
        # Getting the type of 'int' (line 285)
        int_359602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'int', False)
        # Calling int(args, kwargs) (line 285)
        int_call_result_359613 = invoke(stypy.reporting.localization.Localization(__file__, 285, 15), int_359602, *[result_mul_359611], **kwargs_359612)
        
        # Assigning a type to the variable 'stypy_return_type' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'stypy_return_type', int_call_result_359613)
        
        # ################# End of 'getnnz(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getnnz' in the type store
        # Getting the type of 'stypy_return_type' (line 280)
        stypy_return_type_359614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_359614)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getnnz'
        return stypy_return_type_359614

    
    # Assigning a Attribute to a Attribute (line 287):

    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 289, 4, False)
        # Assigning a type to the variable 'self' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.stypy__repr__')
        bsr_matrix.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        bsr_matrix.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        
        # Assigning a Subscript to a Name (line 290):
        
        # Assigning a Subscript to a Name (line 290):
        
        # Obtaining the type of the subscript
        int_359615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 44), 'int')
        
        # Obtaining the type of the subscript
        
        # Call to getformat(...): (line 290)
        # Processing the call keyword arguments (line 290)
        kwargs_359618 = {}
        # Getting the type of 'self' (line 290)
        self_359616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 26), 'self', False)
        # Obtaining the member 'getformat' of a type (line 290)
        getformat_359617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 26), self_359616, 'getformat')
        # Calling getformat(args, kwargs) (line 290)
        getformat_call_result_359619 = invoke(stypy.reporting.localization.Localization(__file__, 290, 26), getformat_359617, *[], **kwargs_359618)
        
        # Getting the type of '_formats' (line 290)
        _formats_359620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 17), '_formats')
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___359621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 17), _formats_359620, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_359622 = invoke(stypy.reporting.localization.Localization(__file__, 290, 17), getitem___359621, getformat_call_result_359619)
        
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___359623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 17), subscript_call_result_359622, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_359624 = invoke(stypy.reporting.localization.Localization(__file__, 290, 17), getitem___359623, int_359615)
        
        # Assigning a type to the variable 'format' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'format', subscript_call_result_359624)
        str_359625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 16), 'str', "<%dx%d sparse matrix of type '%s'\n\twith %d stored elements (blocksize = %dx%d) in %s format>")
        # Getting the type of 'self' (line 293)
        self_359626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 17), 'self')
        # Obtaining the member 'shape' of a type (line 293)
        shape_359627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 17), self_359626, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 293)
        tuple_359628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 293)
        # Adding element type (line 293)
        # Getting the type of 'self' (line 293)
        self_359629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 31), 'self')
        # Obtaining the member 'dtype' of a type (line 293)
        dtype_359630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 31), self_359629, 'dtype')
        # Obtaining the member 'type' of a type (line 293)
        type_359631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 31), dtype_359630, 'type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 31), tuple_359628, type_359631)
        # Adding element type (line 293)
        # Getting the type of 'self' (line 293)
        self_359632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 48), 'self')
        # Obtaining the member 'nnz' of a type (line 293)
        nnz_359633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 48), self_359632, 'nnz')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 31), tuple_359628, nnz_359633)
        
        # Applying the binary operator '+' (line 293)
        result_add_359634 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 17), '+', shape_359627, tuple_359628)
        
        # Getting the type of 'self' (line 293)
        self_359635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 60), 'self')
        # Obtaining the member 'blocksize' of a type (line 293)
        blocksize_359636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 60), self_359635, 'blocksize')
        # Applying the binary operator '+' (line 293)
        result_add_359637 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 58), '+', result_add_359634, blocksize_359636)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 294)
        tuple_359638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 294)
        # Adding element type (line 294)
        # Getting the type of 'format' (line 294)
        format_359639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 18), 'format')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 18), tuple_359638, format_359639)
        
        # Applying the binary operator '+' (line 293)
        result_add_359640 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 75), '+', result_add_359637, tuple_359638)
        
        # Applying the binary operator '%' (line 291)
        result_mod_359641 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 16), '%', str_359625, result_add_359640)
        
        # Assigning a type to the variable 'stypy_return_type' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'stypy_return_type', result_mod_359641)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 289)
        stypy_return_type_359642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_359642)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_359642


    @norecursion
    def diagonal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_359643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 25), 'int')
        defaults = [int_359643]
        # Create a new context for function 'diagonal'
        module_type_store = module_type_store.open_function_context('diagonal', 296, 4, False)
        # Assigning a type to the variable 'self' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.diagonal.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.diagonal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.diagonal.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.diagonal.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.diagonal')
        bsr_matrix.diagonal.__dict__.__setitem__('stypy_param_names_list', ['k'])
        bsr_matrix.diagonal.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.diagonal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.diagonal.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.diagonal.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.diagonal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.diagonal.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.diagonal', ['k'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'diagonal', localization, ['k'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'diagonal(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 297):
        
        # Assigning a Subscript to a Name (line 297):
        
        # Obtaining the type of the subscript
        int_359644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 8), 'int')
        # Getting the type of 'self' (line 297)
        self_359645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 21), 'self')
        # Obtaining the member 'shape' of a type (line 297)
        shape_359646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 21), self_359645, 'shape')
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___359647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), shape_359646, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_359648 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), getitem___359647, int_359644)
        
        # Assigning a type to the variable 'tuple_var_assignment_358871' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'tuple_var_assignment_358871', subscript_call_result_359648)
        
        # Assigning a Subscript to a Name (line 297):
        
        # Obtaining the type of the subscript
        int_359649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 8), 'int')
        # Getting the type of 'self' (line 297)
        self_359650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 21), 'self')
        # Obtaining the member 'shape' of a type (line 297)
        shape_359651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 21), self_359650, 'shape')
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___359652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), shape_359651, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_359653 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), getitem___359652, int_359649)
        
        # Assigning a type to the variable 'tuple_var_assignment_358872' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'tuple_var_assignment_358872', subscript_call_result_359653)
        
        # Assigning a Name to a Name (line 297):
        # Getting the type of 'tuple_var_assignment_358871' (line 297)
        tuple_var_assignment_358871_359654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'tuple_var_assignment_358871')
        # Assigning a type to the variable 'rows' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'rows', tuple_var_assignment_358871_359654)
        
        # Assigning a Name to a Name (line 297):
        # Getting the type of 'tuple_var_assignment_358872' (line 297)
        tuple_var_assignment_358872_359655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'tuple_var_assignment_358872')
        # Assigning a type to the variable 'cols' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 14), 'cols', tuple_var_assignment_358872_359655)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'k' (line 298)
        k_359656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 11), 'k')
        
        # Getting the type of 'rows' (line 298)
        rows_359657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 17), 'rows')
        # Applying the 'usub' unary operator (line 298)
        result___neg___359658 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 16), 'usub', rows_359657)
        
        # Applying the binary operator '<=' (line 298)
        result_le_359659 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 11), '<=', k_359656, result___neg___359658)
        
        
        # Getting the type of 'k' (line 298)
        k_359660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 25), 'k')
        # Getting the type of 'cols' (line 298)
        cols_359661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 30), 'cols')
        # Applying the binary operator '>=' (line 298)
        result_ge_359662 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 25), '>=', k_359660, cols_359661)
        
        # Applying the binary operator 'or' (line 298)
        result_or_keyword_359663 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 11), 'or', result_le_359659, result_ge_359662)
        
        # Testing the type of an if condition (line 298)
        if_condition_359664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 8), result_or_keyword_359663)
        # Assigning a type to the variable 'if_condition_359664' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'if_condition_359664', if_condition_359664)
        # SSA begins for if statement (line 298)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 299)
        # Processing the call arguments (line 299)
        str_359666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 29), 'str', 'k exceeds matrix dimensions')
        # Processing the call keyword arguments (line 299)
        kwargs_359667 = {}
        # Getting the type of 'ValueError' (line 299)
        ValueError_359665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 299)
        ValueError_call_result_359668 = invoke(stypy.reporting.localization.Localization(__file__, 299, 18), ValueError_359665, *[str_359666], **kwargs_359667)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 299, 12), ValueError_call_result_359668, 'raise parameter', BaseException)
        # SSA join for if statement (line 298)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Tuple (line 300):
        
        # Assigning a Subscript to a Name (line 300):
        
        # Obtaining the type of the subscript
        int_359669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 8), 'int')
        # Getting the type of 'self' (line 300)
        self_359670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'self')
        # Obtaining the member 'blocksize' of a type (line 300)
        blocksize_359671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 15), self_359670, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 300)
        getitem___359672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), blocksize_359671, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
        subscript_call_result_359673 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), getitem___359672, int_359669)
        
        # Assigning a type to the variable 'tuple_var_assignment_358873' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'tuple_var_assignment_358873', subscript_call_result_359673)
        
        # Assigning a Subscript to a Name (line 300):
        
        # Obtaining the type of the subscript
        int_359674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 8), 'int')
        # Getting the type of 'self' (line 300)
        self_359675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'self')
        # Obtaining the member 'blocksize' of a type (line 300)
        blocksize_359676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 15), self_359675, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 300)
        getitem___359677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), blocksize_359676, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
        subscript_call_result_359678 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), getitem___359677, int_359674)
        
        # Assigning a type to the variable 'tuple_var_assignment_358874' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'tuple_var_assignment_358874', subscript_call_result_359678)
        
        # Assigning a Name to a Name (line 300):
        # Getting the type of 'tuple_var_assignment_358873' (line 300)
        tuple_var_assignment_358873_359679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'tuple_var_assignment_358873')
        # Assigning a type to the variable 'R' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'R', tuple_var_assignment_358873_359679)
        
        # Assigning a Name to a Name (line 300):
        # Getting the type of 'tuple_var_assignment_358874' (line 300)
        tuple_var_assignment_358874_359680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'tuple_var_assignment_358874')
        # Assigning a type to the variable 'C' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 11), 'C', tuple_var_assignment_358874_359680)
        
        # Assigning a Call to a Name (line 301):
        
        # Assigning a Call to a Name (line 301):
        
        # Call to zeros(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Call to min(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'rows' (line 301)
        rows_359684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'rows', False)
        
        # Call to min(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'k' (line 301)
        k_359686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 36), 'k', False)
        int_359687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 39), 'int')
        # Processing the call keyword arguments (line 301)
        kwargs_359688 = {}
        # Getting the type of 'min' (line 301)
        min_359685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 32), 'min', False)
        # Calling min(args, kwargs) (line 301)
        min_call_result_359689 = invoke(stypy.reporting.localization.Localization(__file__, 301, 32), min_359685, *[k_359686, int_359687], **kwargs_359688)
        
        # Applying the binary operator '+' (line 301)
        result_add_359690 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 25), '+', rows_359684, min_call_result_359689)
        
        # Getting the type of 'cols' (line 301)
        cols_359691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 43), 'cols', False)
        
        # Call to max(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'k' (line 301)
        k_359693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 54), 'k', False)
        int_359694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 57), 'int')
        # Processing the call keyword arguments (line 301)
        kwargs_359695 = {}
        # Getting the type of 'max' (line 301)
        max_359692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 50), 'max', False)
        # Calling max(args, kwargs) (line 301)
        max_call_result_359696 = invoke(stypy.reporting.localization.Localization(__file__, 301, 50), max_359692, *[k_359693, int_359694], **kwargs_359695)
        
        # Applying the binary operator '-' (line 301)
        result_sub_359697 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 43), '-', cols_359691, max_call_result_359696)
        
        # Processing the call keyword arguments (line 301)
        kwargs_359698 = {}
        # Getting the type of 'min' (line 301)
        min_359683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 21), 'min', False)
        # Calling min(args, kwargs) (line 301)
        min_call_result_359699 = invoke(stypy.reporting.localization.Localization(__file__, 301, 21), min_359683, *[result_add_359690, result_sub_359697], **kwargs_359698)
        
        # Processing the call keyword arguments (line 301)
        
        # Call to upcast(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'self' (line 302)
        self_359701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 34), 'self', False)
        # Obtaining the member 'dtype' of a type (line 302)
        dtype_359702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 34), self_359701, 'dtype')
        # Processing the call keyword arguments (line 302)
        kwargs_359703 = {}
        # Getting the type of 'upcast' (line 302)
        upcast_359700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 27), 'upcast', False)
        # Calling upcast(args, kwargs) (line 302)
        upcast_call_result_359704 = invoke(stypy.reporting.localization.Localization(__file__, 302, 27), upcast_359700, *[dtype_359702], **kwargs_359703)
        
        keyword_359705 = upcast_call_result_359704
        kwargs_359706 = {'dtype': keyword_359705}
        # Getting the type of 'np' (line 301)
        np_359681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 301)
        zeros_359682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), np_359681, 'zeros')
        # Calling zeros(args, kwargs) (line 301)
        zeros_call_result_359707 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), zeros_359682, *[min_call_result_359699], **kwargs_359706)
        
        # Assigning a type to the variable 'y' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'y', zeros_call_result_359707)
        
        # Call to bsr_diagonal(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'k' (line 303)
        k_359710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 34), 'k', False)
        # Getting the type of 'rows' (line 303)
        rows_359711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 37), 'rows', False)
        # Getting the type of 'R' (line 303)
        R_359712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 45), 'R', False)
        # Applying the binary operator '//' (line 303)
        result_floordiv_359713 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 37), '//', rows_359711, R_359712)
        
        # Getting the type of 'cols' (line 303)
        cols_359714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 48), 'cols', False)
        # Getting the type of 'C' (line 303)
        C_359715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 56), 'C', False)
        # Applying the binary operator '//' (line 303)
        result_floordiv_359716 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 48), '//', cols_359714, C_359715)
        
        # Getting the type of 'R' (line 303)
        R_359717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 59), 'R', False)
        # Getting the type of 'C' (line 303)
        C_359718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 62), 'C', False)
        # Getting the type of 'self' (line 304)
        self_359719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 34), 'self', False)
        # Obtaining the member 'indptr' of a type (line 304)
        indptr_359720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 34), self_359719, 'indptr')
        # Getting the type of 'self' (line 304)
        self_359721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 47), 'self', False)
        # Obtaining the member 'indices' of a type (line 304)
        indices_359722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 47), self_359721, 'indices')
        
        # Call to ravel(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'self' (line 305)
        self_359725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 43), 'self', False)
        # Obtaining the member 'data' of a type (line 305)
        data_359726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 43), self_359725, 'data')
        # Processing the call keyword arguments (line 305)
        kwargs_359727 = {}
        # Getting the type of 'np' (line 305)
        np_359723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 34), 'np', False)
        # Obtaining the member 'ravel' of a type (line 305)
        ravel_359724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 34), np_359723, 'ravel')
        # Calling ravel(args, kwargs) (line 305)
        ravel_call_result_359728 = invoke(stypy.reporting.localization.Localization(__file__, 305, 34), ravel_359724, *[data_359726], **kwargs_359727)
        
        # Getting the type of 'y' (line 305)
        y_359729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 55), 'y', False)
        # Processing the call keyword arguments (line 303)
        kwargs_359730 = {}
        # Getting the type of '_sparsetools' (line 303)
        _sparsetools_359708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), '_sparsetools', False)
        # Obtaining the member 'bsr_diagonal' of a type (line 303)
        bsr_diagonal_359709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 8), _sparsetools_359708, 'bsr_diagonal')
        # Calling bsr_diagonal(args, kwargs) (line 303)
        bsr_diagonal_call_result_359731 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), bsr_diagonal_359709, *[k_359710, result_floordiv_359713, result_floordiv_359716, R_359717, C_359718, indptr_359720, indices_359722, ravel_call_result_359728, y_359729], **kwargs_359730)
        
        # Getting the type of 'y' (line 306)
        y_359732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), 'y')
        # Assigning a type to the variable 'stypy_return_type' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'stypy_return_type', y_359732)
        
        # ################# End of 'diagonal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'diagonal' in the type store
        # Getting the type of 'stypy_return_type' (line 296)
        stypy_return_type_359733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_359733)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'diagonal'
        return stypy_return_type_359733

    
    # Assigning a Attribute to a Attribute (line 308):

    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 314, 4, False)
        # Assigning a type to the variable 'self' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.__getitem__.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.__getitem__')
        bsr_matrix.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['key'])
        bsr_matrix.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.__getitem__', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        # Getting the type of 'NotImplementedError' (line 315)
        NotImplementedError_359734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 315, 8), NotImplementedError_359734, 'raise parameter', BaseException)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 314)
        stypy_return_type_359735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_359735)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_359735


    @norecursion
    def __setitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setitem__'
        module_type_store = module_type_store.open_function_context('__setitem__', 317, 4, False)
        # Assigning a type to the variable 'self' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.__setitem__.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.__setitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.__setitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.__setitem__.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.__setitem__')
        bsr_matrix.__setitem__.__dict__.__setitem__('stypy_param_names_list', ['key', 'val'])
        bsr_matrix.__setitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.__setitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.__setitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.__setitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.__setitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.__setitem__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.__setitem__', ['key', 'val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setitem__', localization, ['key', 'val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setitem__(...)' code ##################

        # Getting the type of 'NotImplementedError' (line 318)
        NotImplementedError_359736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 318, 8), NotImplementedError_359736, 'raise parameter', BaseException)
        
        # ################# End of '__setitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 317)
        stypy_return_type_359737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_359737)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setitem__'
        return stypy_return_type_359737


    @norecursion
    def matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matvec'
        module_type_store = module_type_store.open_function_context('matvec', 324, 4, False)
        # Assigning a type to the variable 'self' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.matvec.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.matvec.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.matvec')
        bsr_matrix.matvec.__dict__.__setitem__('stypy_param_names_list', ['other'])
        bsr_matrix.matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.matvec', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'matvec', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'matvec(...)' code ##################

        str_359738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 8), 'str', 'Multiply matrix by vector.')
        # Getting the type of 'self' (line 328)
        self_359739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 15), 'self')
        # Getting the type of 'other' (line 328)
        other_359740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 22), 'other')
        # Applying the binary operator '*' (line 328)
        result_mul_359741 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 15), '*', self_359739, other_359740)
        
        # Assigning a type to the variable 'stypy_return_type' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'stypy_return_type', result_mul_359741)
        
        # ################# End of 'matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 324)
        stypy_return_type_359742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_359742)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matvec'
        return stypy_return_type_359742


    @norecursion
    def matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matmat'
        module_type_store = module_type_store.open_function_context('matmat', 330, 4, False)
        # Assigning a type to the variable 'self' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.matmat.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.matmat.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.matmat')
        bsr_matrix.matmat.__dict__.__setitem__('stypy_param_names_list', ['other'])
        bsr_matrix.matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.matmat', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'matmat', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'matmat(...)' code ##################

        str_359743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 8), 'str', 'Multiply this sparse matrix by other matrix.')
        # Getting the type of 'self' (line 334)
        self_359744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 15), 'self')
        # Getting the type of 'other' (line 334)
        other_359745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 22), 'other')
        # Applying the binary operator '*' (line 334)
        result_mul_359746 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 15), '*', self_359744, other_359745)
        
        # Assigning a type to the variable 'stypy_return_type' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'stypy_return_type', result_mul_359746)
        
        # ################# End of 'matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 330)
        stypy_return_type_359747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_359747)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matmat'
        return stypy_return_type_359747


    @norecursion
    def _add_dense(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add_dense'
        module_type_store = module_type_store.open_function_context('_add_dense', 336, 4, False)
        # Assigning a type to the variable 'self' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix._add_dense.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix._add_dense.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix._add_dense.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix._add_dense.__dict__.__setitem__('stypy_function_name', 'bsr_matrix._add_dense')
        bsr_matrix._add_dense.__dict__.__setitem__('stypy_param_names_list', ['other'])
        bsr_matrix._add_dense.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix._add_dense.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix._add_dense.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix._add_dense.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix._add_dense.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix._add_dense.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix._add_dense', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add_dense', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add_dense(...)' code ##################

        
        # Call to _add_dense(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'other' (line 337)
        other_359755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 49), 'other', False)
        # Processing the call keyword arguments (line 337)
        kwargs_359756 = {}
        
        # Call to tocoo(...): (line 337)
        # Processing the call keyword arguments (line 337)
        # Getting the type of 'False' (line 337)
        False_359750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 31), 'False', False)
        keyword_359751 = False_359750
        kwargs_359752 = {'copy': keyword_359751}
        # Getting the type of 'self' (line 337)
        self_359748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 15), 'self', False)
        # Obtaining the member 'tocoo' of a type (line 337)
        tocoo_359749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 15), self_359748, 'tocoo')
        # Calling tocoo(args, kwargs) (line 337)
        tocoo_call_result_359753 = invoke(stypy.reporting.localization.Localization(__file__, 337, 15), tocoo_359749, *[], **kwargs_359752)
        
        # Obtaining the member '_add_dense' of a type (line 337)
        _add_dense_359754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 15), tocoo_call_result_359753, '_add_dense')
        # Calling _add_dense(args, kwargs) (line 337)
        _add_dense_call_result_359757 = invoke(stypy.reporting.localization.Localization(__file__, 337, 15), _add_dense_359754, *[other_359755], **kwargs_359756)
        
        # Assigning a type to the variable 'stypy_return_type' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'stypy_return_type', _add_dense_call_result_359757)
        
        # ################# End of '_add_dense(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add_dense' in the type store
        # Getting the type of 'stypy_return_type' (line 336)
        stypy_return_type_359758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_359758)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add_dense'
        return stypy_return_type_359758


    @norecursion
    def _mul_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_vector'
        module_type_store = module_type_store.open_function_context('_mul_vector', 339, 4, False)
        # Assigning a type to the variable 'self' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix._mul_vector.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix._mul_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix._mul_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix._mul_vector.__dict__.__setitem__('stypy_function_name', 'bsr_matrix._mul_vector')
        bsr_matrix._mul_vector.__dict__.__setitem__('stypy_param_names_list', ['other'])
        bsr_matrix._mul_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix._mul_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix._mul_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix._mul_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix._mul_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix._mul_vector.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix._mul_vector', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mul_vector', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mul_vector(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 340):
        
        # Assigning a Subscript to a Name (line 340):
        
        # Obtaining the type of the subscript
        int_359759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 8), 'int')
        # Getting the type of 'self' (line 340)
        self_359760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 14), 'self')
        # Obtaining the member 'shape' of a type (line 340)
        shape_359761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 14), self_359760, 'shape')
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___359762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), shape_359761, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_359763 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), getitem___359762, int_359759)
        
        # Assigning a type to the variable 'tuple_var_assignment_358875' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_358875', subscript_call_result_359763)
        
        # Assigning a Subscript to a Name (line 340):
        
        # Obtaining the type of the subscript
        int_359764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 8), 'int')
        # Getting the type of 'self' (line 340)
        self_359765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 14), 'self')
        # Obtaining the member 'shape' of a type (line 340)
        shape_359766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 14), self_359765, 'shape')
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___359767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), shape_359766, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_359768 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), getitem___359767, int_359764)
        
        # Assigning a type to the variable 'tuple_var_assignment_358876' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_358876', subscript_call_result_359768)
        
        # Assigning a Name to a Name (line 340):
        # Getting the type of 'tuple_var_assignment_358875' (line 340)
        tuple_var_assignment_358875_359769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_358875')
        # Assigning a type to the variable 'M' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'M', tuple_var_assignment_358875_359769)
        
        # Assigning a Name to a Name (line 340):
        # Getting the type of 'tuple_var_assignment_358876' (line 340)
        tuple_var_assignment_358876_359770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_358876')
        # Assigning a type to the variable 'N' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 10), 'N', tuple_var_assignment_358876_359770)
        
        # Assigning a Attribute to a Tuple (line 341):
        
        # Assigning a Subscript to a Name (line 341):
        
        # Obtaining the type of the subscript
        int_359771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 8), 'int')
        # Getting the type of 'self' (line 341)
        self_359772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 341)
        blocksize_359773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 14), self_359772, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___359774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), blocksize_359773, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_359775 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), getitem___359774, int_359771)
        
        # Assigning a type to the variable 'tuple_var_assignment_358877' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'tuple_var_assignment_358877', subscript_call_result_359775)
        
        # Assigning a Subscript to a Name (line 341):
        
        # Obtaining the type of the subscript
        int_359776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 8), 'int')
        # Getting the type of 'self' (line 341)
        self_359777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 341)
        blocksize_359778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 14), self_359777, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___359779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), blocksize_359778, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_359780 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), getitem___359779, int_359776)
        
        # Assigning a type to the variable 'tuple_var_assignment_358878' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'tuple_var_assignment_358878', subscript_call_result_359780)
        
        # Assigning a Name to a Name (line 341):
        # Getting the type of 'tuple_var_assignment_358877' (line 341)
        tuple_var_assignment_358877_359781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'tuple_var_assignment_358877')
        # Assigning a type to the variable 'R' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'R', tuple_var_assignment_358877_359781)
        
        # Assigning a Name to a Name (line 341):
        # Getting the type of 'tuple_var_assignment_358878' (line 341)
        tuple_var_assignment_358878_359782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'tuple_var_assignment_358878')
        # Assigning a type to the variable 'C' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 10), 'C', tuple_var_assignment_358878_359782)
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to zeros(...): (line 343)
        # Processing the call arguments (line 343)
        
        # Obtaining the type of the subscript
        int_359785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 37), 'int')
        # Getting the type of 'self' (line 343)
        self_359786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 26), 'self', False)
        # Obtaining the member 'shape' of a type (line 343)
        shape_359787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 26), self_359786, 'shape')
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___359788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 26), shape_359787, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_359789 = invoke(stypy.reporting.localization.Localization(__file__, 343, 26), getitem___359788, int_359785)
        
        # Processing the call keyword arguments (line 343)
        
        # Call to upcast(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'self' (line 343)
        self_359791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 54), 'self', False)
        # Obtaining the member 'dtype' of a type (line 343)
        dtype_359792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 54), self_359791, 'dtype')
        # Getting the type of 'other' (line 343)
        other_359793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 66), 'other', False)
        # Obtaining the member 'dtype' of a type (line 343)
        dtype_359794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 66), other_359793, 'dtype')
        # Processing the call keyword arguments (line 343)
        kwargs_359795 = {}
        # Getting the type of 'upcast' (line 343)
        upcast_359790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 47), 'upcast', False)
        # Calling upcast(args, kwargs) (line 343)
        upcast_call_result_359796 = invoke(stypy.reporting.localization.Localization(__file__, 343, 47), upcast_359790, *[dtype_359792, dtype_359794], **kwargs_359795)
        
        keyword_359797 = upcast_call_result_359796
        kwargs_359798 = {'dtype': keyword_359797}
        # Getting the type of 'np' (line 343)
        np_359783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 17), 'np', False)
        # Obtaining the member 'zeros' of a type (line 343)
        zeros_359784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 17), np_359783, 'zeros')
        # Calling zeros(args, kwargs) (line 343)
        zeros_call_result_359799 = invoke(stypy.reporting.localization.Localization(__file__, 343, 17), zeros_359784, *[subscript_call_result_359789], **kwargs_359798)
        
        # Assigning a type to the variable 'result' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'result', zeros_call_result_359799)
        
        # Call to bsr_matvec(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'M' (line 345)
        M_359801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'M', False)
        # Getting the type of 'R' (line 345)
        R_359802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 22), 'R', False)
        # Applying the binary operator '//' (line 345)
        result_floordiv_359803 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 19), '//', M_359801, R_359802)
        
        # Getting the type of 'N' (line 345)
        N_359804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 25), 'N', False)
        # Getting the type of 'C' (line 345)
        C_359805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 28), 'C', False)
        # Applying the binary operator '//' (line 345)
        result_floordiv_359806 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 25), '//', N_359804, C_359805)
        
        # Getting the type of 'R' (line 345)
        R_359807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 31), 'R', False)
        # Getting the type of 'C' (line 345)
        C_359808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 34), 'C', False)
        # Getting the type of 'self' (line 346)
        self_359809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'self', False)
        # Obtaining the member 'indptr' of a type (line 346)
        indptr_359810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 12), self_359809, 'indptr')
        # Getting the type of 'self' (line 346)
        self_359811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 25), 'self', False)
        # Obtaining the member 'indices' of a type (line 346)
        indices_359812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 25), self_359811, 'indices')
        
        # Call to ravel(...): (line 346)
        # Processing the call keyword arguments (line 346)
        kwargs_359816 = {}
        # Getting the type of 'self' (line 346)
        self_359813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 39), 'self', False)
        # Obtaining the member 'data' of a type (line 346)
        data_359814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 39), self_359813, 'data')
        # Obtaining the member 'ravel' of a type (line 346)
        ravel_359815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 39), data_359814, 'ravel')
        # Calling ravel(args, kwargs) (line 346)
        ravel_call_result_359817 = invoke(stypy.reporting.localization.Localization(__file__, 346, 39), ravel_359815, *[], **kwargs_359816)
        
        # Getting the type of 'other' (line 347)
        other_359818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'other', False)
        # Getting the type of 'result' (line 347)
        result_359819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 19), 'result', False)
        # Processing the call keyword arguments (line 345)
        kwargs_359820 = {}
        # Getting the type of 'bsr_matvec' (line 345)
        bsr_matvec_359800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'bsr_matvec', False)
        # Calling bsr_matvec(args, kwargs) (line 345)
        bsr_matvec_call_result_359821 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), bsr_matvec_359800, *[result_floordiv_359803, result_floordiv_359806, R_359807, C_359808, indptr_359810, indices_359812, ravel_call_result_359817, other_359818, result_359819], **kwargs_359820)
        
        # Getting the type of 'result' (line 349)
        result_359822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'stypy_return_type', result_359822)
        
        # ################# End of '_mul_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 339)
        stypy_return_type_359823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_359823)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_vector'
        return stypy_return_type_359823


    @norecursion
    def _mul_multivector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_multivector'
        module_type_store = module_type_store.open_function_context('_mul_multivector', 351, 4, False)
        # Assigning a type to the variable 'self' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix._mul_multivector.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix._mul_multivector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix._mul_multivector.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix._mul_multivector.__dict__.__setitem__('stypy_function_name', 'bsr_matrix._mul_multivector')
        bsr_matrix._mul_multivector.__dict__.__setitem__('stypy_param_names_list', ['other'])
        bsr_matrix._mul_multivector.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix._mul_multivector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix._mul_multivector.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix._mul_multivector.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix._mul_multivector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix._mul_multivector.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix._mul_multivector', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mul_multivector', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mul_multivector(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 352):
        
        # Assigning a Subscript to a Name (line 352):
        
        # Obtaining the type of the subscript
        int_359824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 8), 'int')
        # Getting the type of 'self' (line 352)
        self_359825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 352)
        blocksize_359826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 14), self_359825, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 352)
        getitem___359827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), blocksize_359826, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 352)
        subscript_call_result_359828 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), getitem___359827, int_359824)
        
        # Assigning a type to the variable 'tuple_var_assignment_358879' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_var_assignment_358879', subscript_call_result_359828)
        
        # Assigning a Subscript to a Name (line 352):
        
        # Obtaining the type of the subscript
        int_359829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 8), 'int')
        # Getting the type of 'self' (line 352)
        self_359830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 352)
        blocksize_359831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 14), self_359830, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 352)
        getitem___359832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), blocksize_359831, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 352)
        subscript_call_result_359833 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), getitem___359832, int_359829)
        
        # Assigning a type to the variable 'tuple_var_assignment_358880' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_var_assignment_358880', subscript_call_result_359833)
        
        # Assigning a Name to a Name (line 352):
        # Getting the type of 'tuple_var_assignment_358879' (line 352)
        tuple_var_assignment_358879_359834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_var_assignment_358879')
        # Assigning a type to the variable 'R' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'R', tuple_var_assignment_358879_359834)
        
        # Assigning a Name to a Name (line 352):
        # Getting the type of 'tuple_var_assignment_358880' (line 352)
        tuple_var_assignment_358880_359835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_var_assignment_358880')
        # Assigning a type to the variable 'C' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 10), 'C', tuple_var_assignment_358880_359835)
        
        # Assigning a Attribute to a Tuple (line 353):
        
        # Assigning a Subscript to a Name (line 353):
        
        # Obtaining the type of the subscript
        int_359836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 8), 'int')
        # Getting the type of 'self' (line 353)
        self_359837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 14), 'self')
        # Obtaining the member 'shape' of a type (line 353)
        shape_359838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 14), self_359837, 'shape')
        # Obtaining the member '__getitem__' of a type (line 353)
        getitem___359839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 8), shape_359838, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 353)
        subscript_call_result_359840 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), getitem___359839, int_359836)
        
        # Assigning a type to the variable 'tuple_var_assignment_358881' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'tuple_var_assignment_358881', subscript_call_result_359840)
        
        # Assigning a Subscript to a Name (line 353):
        
        # Obtaining the type of the subscript
        int_359841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 8), 'int')
        # Getting the type of 'self' (line 353)
        self_359842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 14), 'self')
        # Obtaining the member 'shape' of a type (line 353)
        shape_359843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 14), self_359842, 'shape')
        # Obtaining the member '__getitem__' of a type (line 353)
        getitem___359844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 8), shape_359843, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 353)
        subscript_call_result_359845 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), getitem___359844, int_359841)
        
        # Assigning a type to the variable 'tuple_var_assignment_358882' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'tuple_var_assignment_358882', subscript_call_result_359845)
        
        # Assigning a Name to a Name (line 353):
        # Getting the type of 'tuple_var_assignment_358881' (line 353)
        tuple_var_assignment_358881_359846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'tuple_var_assignment_358881')
        # Assigning a type to the variable 'M' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'M', tuple_var_assignment_358881_359846)
        
        # Assigning a Name to a Name (line 353):
        # Getting the type of 'tuple_var_assignment_358882' (line 353)
        tuple_var_assignment_358882_359847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'tuple_var_assignment_358882')
        # Assigning a type to the variable 'N' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 10), 'N', tuple_var_assignment_358882_359847)
        
        # Assigning a Subscript to a Name (line 354):
        
        # Assigning a Subscript to a Name (line 354):
        
        # Obtaining the type of the subscript
        int_359848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 29), 'int')
        # Getting the type of 'other' (line 354)
        other_359849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 17), 'other')
        # Obtaining the member 'shape' of a type (line 354)
        shape_359850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 17), other_359849, 'shape')
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___359851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 17), shape_359850, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_359852 = invoke(stypy.reporting.localization.Localization(__file__, 354, 17), getitem___359851, int_359848)
        
        # Assigning a type to the variable 'n_vecs' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'n_vecs', subscript_call_result_359852)
        
        # Assigning a Call to a Name (line 356):
        
        # Assigning a Call to a Name (line 356):
        
        # Call to zeros(...): (line 356)
        # Processing the call arguments (line 356)
        
        # Obtaining an instance of the builtin type 'tuple' (line 356)
        tuple_359855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 356)
        # Adding element type (line 356)
        # Getting the type of 'M' (line 356)
        M_359856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 27), 'M', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 27), tuple_359855, M_359856)
        # Adding element type (line 356)
        # Getting the type of 'n_vecs' (line 356)
        n_vecs_359857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 29), 'n_vecs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 27), tuple_359855, n_vecs_359857)
        
        # Processing the call keyword arguments (line 356)
        
        # Call to upcast(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'self' (line 356)
        self_359859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 51), 'self', False)
        # Obtaining the member 'dtype' of a type (line 356)
        dtype_359860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 51), self_359859, 'dtype')
        # Getting the type of 'other' (line 356)
        other_359861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 62), 'other', False)
        # Obtaining the member 'dtype' of a type (line 356)
        dtype_359862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 62), other_359861, 'dtype')
        # Processing the call keyword arguments (line 356)
        kwargs_359863 = {}
        # Getting the type of 'upcast' (line 356)
        upcast_359858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 44), 'upcast', False)
        # Calling upcast(args, kwargs) (line 356)
        upcast_call_result_359864 = invoke(stypy.reporting.localization.Localization(__file__, 356, 44), upcast_359858, *[dtype_359860, dtype_359862], **kwargs_359863)
        
        keyword_359865 = upcast_call_result_359864
        kwargs_359866 = {'dtype': keyword_359865}
        # Getting the type of 'np' (line 356)
        np_359853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 17), 'np', False)
        # Obtaining the member 'zeros' of a type (line 356)
        zeros_359854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 17), np_359853, 'zeros')
        # Calling zeros(args, kwargs) (line 356)
        zeros_call_result_359867 = invoke(stypy.reporting.localization.Localization(__file__, 356, 17), zeros_359854, *[tuple_359855], **kwargs_359866)
        
        # Assigning a type to the variable 'result' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'result', zeros_call_result_359867)
        
        # Call to bsr_matvecs(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'M' (line 358)
        M_359869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 20), 'M', False)
        # Getting the type of 'R' (line 358)
        R_359870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 23), 'R', False)
        # Applying the binary operator '//' (line 358)
        result_floordiv_359871 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 20), '//', M_359869, R_359870)
        
        # Getting the type of 'N' (line 358)
        N_359872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 26), 'N', False)
        # Getting the type of 'C' (line 358)
        C_359873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 29), 'C', False)
        # Applying the binary operator '//' (line 358)
        result_floordiv_359874 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 26), '//', N_359872, C_359873)
        
        # Getting the type of 'n_vecs' (line 358)
        n_vecs_359875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 32), 'n_vecs', False)
        # Getting the type of 'R' (line 358)
        R_359876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 40), 'R', False)
        # Getting the type of 'C' (line 358)
        C_359877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 43), 'C', False)
        # Getting the type of 'self' (line 359)
        self_359878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'self', False)
        # Obtaining the member 'indptr' of a type (line 359)
        indptr_359879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 16), self_359878, 'indptr')
        # Getting the type of 'self' (line 359)
        self_359880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 29), 'self', False)
        # Obtaining the member 'indices' of a type (line 359)
        indices_359881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 29), self_359880, 'indices')
        
        # Call to ravel(...): (line 359)
        # Processing the call keyword arguments (line 359)
        kwargs_359885 = {}
        # Getting the type of 'self' (line 359)
        self_359882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 43), 'self', False)
        # Obtaining the member 'data' of a type (line 359)
        data_359883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 43), self_359882, 'data')
        # Obtaining the member 'ravel' of a type (line 359)
        ravel_359884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 43), data_359883, 'ravel')
        # Calling ravel(args, kwargs) (line 359)
        ravel_call_result_359886 = invoke(stypy.reporting.localization.Localization(__file__, 359, 43), ravel_359884, *[], **kwargs_359885)
        
        
        # Call to ravel(...): (line 360)
        # Processing the call keyword arguments (line 360)
        kwargs_359889 = {}
        # Getting the type of 'other' (line 360)
        other_359887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 16), 'other', False)
        # Obtaining the member 'ravel' of a type (line 360)
        ravel_359888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 16), other_359887, 'ravel')
        # Calling ravel(args, kwargs) (line 360)
        ravel_call_result_359890 = invoke(stypy.reporting.localization.Localization(__file__, 360, 16), ravel_359888, *[], **kwargs_359889)
        
        
        # Call to ravel(...): (line 360)
        # Processing the call keyword arguments (line 360)
        kwargs_359893 = {}
        # Getting the type of 'result' (line 360)
        result_359891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 31), 'result', False)
        # Obtaining the member 'ravel' of a type (line 360)
        ravel_359892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 31), result_359891, 'ravel')
        # Calling ravel(args, kwargs) (line 360)
        ravel_call_result_359894 = invoke(stypy.reporting.localization.Localization(__file__, 360, 31), ravel_359892, *[], **kwargs_359893)
        
        # Processing the call keyword arguments (line 358)
        kwargs_359895 = {}
        # Getting the type of 'bsr_matvecs' (line 358)
        bsr_matvecs_359868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'bsr_matvecs', False)
        # Calling bsr_matvecs(args, kwargs) (line 358)
        bsr_matvecs_call_result_359896 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), bsr_matvecs_359868, *[result_floordiv_359871, result_floordiv_359874, n_vecs_359875, R_359876, C_359877, indptr_359879, indices_359881, ravel_call_result_359886, ravel_call_result_359890, ravel_call_result_359894], **kwargs_359895)
        
        # Getting the type of 'result' (line 362)
        result_359897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'stypy_return_type', result_359897)
        
        # ################# End of '_mul_multivector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_multivector' in the type store
        # Getting the type of 'stypy_return_type' (line 351)
        stypy_return_type_359898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_359898)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_multivector'
        return stypy_return_type_359898


    @norecursion
    def _mul_sparse_matrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_sparse_matrix'
        module_type_store = module_type_store.open_function_context('_mul_sparse_matrix', 364, 4, False)
        # Assigning a type to the variable 'self' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix._mul_sparse_matrix.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix._mul_sparse_matrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix._mul_sparse_matrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix._mul_sparse_matrix.__dict__.__setitem__('stypy_function_name', 'bsr_matrix._mul_sparse_matrix')
        bsr_matrix._mul_sparse_matrix.__dict__.__setitem__('stypy_param_names_list', ['other'])
        bsr_matrix._mul_sparse_matrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix._mul_sparse_matrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix._mul_sparse_matrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix._mul_sparse_matrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix._mul_sparse_matrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix._mul_sparse_matrix.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix._mul_sparse_matrix', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mul_sparse_matrix', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mul_sparse_matrix(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 365):
        
        # Assigning a Subscript to a Name (line 365):
        
        # Obtaining the type of the subscript
        int_359899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 8), 'int')
        # Getting the type of 'self' (line 365)
        self_359900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 16), 'self')
        # Obtaining the member 'shape' of a type (line 365)
        shape_359901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 16), self_359900, 'shape')
        # Obtaining the member '__getitem__' of a type (line 365)
        getitem___359902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 8), shape_359901, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 365)
        subscript_call_result_359903 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), getitem___359902, int_359899)
        
        # Assigning a type to the variable 'tuple_var_assignment_358883' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'tuple_var_assignment_358883', subscript_call_result_359903)
        
        # Assigning a Subscript to a Name (line 365):
        
        # Obtaining the type of the subscript
        int_359904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 8), 'int')
        # Getting the type of 'self' (line 365)
        self_359905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 16), 'self')
        # Obtaining the member 'shape' of a type (line 365)
        shape_359906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 16), self_359905, 'shape')
        # Obtaining the member '__getitem__' of a type (line 365)
        getitem___359907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 8), shape_359906, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 365)
        subscript_call_result_359908 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), getitem___359907, int_359904)
        
        # Assigning a type to the variable 'tuple_var_assignment_358884' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'tuple_var_assignment_358884', subscript_call_result_359908)
        
        # Assigning a Name to a Name (line 365):
        # Getting the type of 'tuple_var_assignment_358883' (line 365)
        tuple_var_assignment_358883_359909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'tuple_var_assignment_358883')
        # Assigning a type to the variable 'M' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'M', tuple_var_assignment_358883_359909)
        
        # Assigning a Name to a Name (line 365):
        # Getting the type of 'tuple_var_assignment_358884' (line 365)
        tuple_var_assignment_358884_359910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'tuple_var_assignment_358884')
        # Assigning a type to the variable 'K1' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 11), 'K1', tuple_var_assignment_358884_359910)
        
        # Assigning a Attribute to a Tuple (line 366):
        
        # Assigning a Subscript to a Name (line 366):
        
        # Obtaining the type of the subscript
        int_359911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 8), 'int')
        # Getting the type of 'other' (line 366)
        other_359912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 'other')
        # Obtaining the member 'shape' of a type (line 366)
        shape_359913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 16), other_359912, 'shape')
        # Obtaining the member '__getitem__' of a type (line 366)
        getitem___359914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 8), shape_359913, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 366)
        subscript_call_result_359915 = invoke(stypy.reporting.localization.Localization(__file__, 366, 8), getitem___359914, int_359911)
        
        # Assigning a type to the variable 'tuple_var_assignment_358885' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'tuple_var_assignment_358885', subscript_call_result_359915)
        
        # Assigning a Subscript to a Name (line 366):
        
        # Obtaining the type of the subscript
        int_359916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 8), 'int')
        # Getting the type of 'other' (line 366)
        other_359917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 'other')
        # Obtaining the member 'shape' of a type (line 366)
        shape_359918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 16), other_359917, 'shape')
        # Obtaining the member '__getitem__' of a type (line 366)
        getitem___359919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 8), shape_359918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 366)
        subscript_call_result_359920 = invoke(stypy.reporting.localization.Localization(__file__, 366, 8), getitem___359919, int_359916)
        
        # Assigning a type to the variable 'tuple_var_assignment_358886' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'tuple_var_assignment_358886', subscript_call_result_359920)
        
        # Assigning a Name to a Name (line 366):
        # Getting the type of 'tuple_var_assignment_358885' (line 366)
        tuple_var_assignment_358885_359921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'tuple_var_assignment_358885')
        # Assigning a type to the variable 'K2' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'K2', tuple_var_assignment_358885_359921)
        
        # Assigning a Name to a Name (line 366):
        # Getting the type of 'tuple_var_assignment_358886' (line 366)
        tuple_var_assignment_358886_359922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'tuple_var_assignment_358886')
        # Assigning a type to the variable 'N' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'N', tuple_var_assignment_358886_359922)
        
        # Assigning a Attribute to a Tuple (line 368):
        
        # Assigning a Subscript to a Name (line 368):
        
        # Obtaining the type of the subscript
        int_359923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 8), 'int')
        # Getting the type of 'self' (line 368)
        self_359924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 368)
        blocksize_359925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 14), self_359924, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 368)
        getitem___359926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), blocksize_359925, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 368)
        subscript_call_result_359927 = invoke(stypy.reporting.localization.Localization(__file__, 368, 8), getitem___359926, int_359923)
        
        # Assigning a type to the variable 'tuple_var_assignment_358887' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'tuple_var_assignment_358887', subscript_call_result_359927)
        
        # Assigning a Subscript to a Name (line 368):
        
        # Obtaining the type of the subscript
        int_359928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 8), 'int')
        # Getting the type of 'self' (line 368)
        self_359929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 368)
        blocksize_359930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 14), self_359929, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 368)
        getitem___359931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), blocksize_359930, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 368)
        subscript_call_result_359932 = invoke(stypy.reporting.localization.Localization(__file__, 368, 8), getitem___359931, int_359928)
        
        # Assigning a type to the variable 'tuple_var_assignment_358888' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'tuple_var_assignment_358888', subscript_call_result_359932)
        
        # Assigning a Name to a Name (line 368):
        # Getting the type of 'tuple_var_assignment_358887' (line 368)
        tuple_var_assignment_358887_359933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'tuple_var_assignment_358887')
        # Assigning a type to the variable 'R' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'R', tuple_var_assignment_358887_359933)
        
        # Assigning a Name to a Name (line 368):
        # Getting the type of 'tuple_var_assignment_358888' (line 368)
        tuple_var_assignment_358888_359934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'tuple_var_assignment_358888')
        # Assigning a type to the variable 'n' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 10), 'n', tuple_var_assignment_358888_359934)
        
        
        # Call to isspmatrix_bsr(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'other' (line 371)
        other_359936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 26), 'other', False)
        # Processing the call keyword arguments (line 371)
        kwargs_359937 = {}
        # Getting the type of 'isspmatrix_bsr' (line 371)
        isspmatrix_bsr_359935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 11), 'isspmatrix_bsr', False)
        # Calling isspmatrix_bsr(args, kwargs) (line 371)
        isspmatrix_bsr_call_result_359938 = invoke(stypy.reporting.localization.Localization(__file__, 371, 11), isspmatrix_bsr_359935, *[other_359936], **kwargs_359937)
        
        # Testing the type of an if condition (line 371)
        if_condition_359939 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 8), isspmatrix_bsr_call_result_359938)
        # Assigning a type to the variable 'if_condition_359939' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'if_condition_359939', if_condition_359939)
        # SSA begins for if statement (line 371)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 372):
        
        # Assigning a Subscript to a Name (line 372):
        
        # Obtaining the type of the subscript
        int_359940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 32), 'int')
        # Getting the type of 'other' (line 372)
        other_359941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'other')
        # Obtaining the member 'blocksize' of a type (line 372)
        blocksize_359942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 16), other_359941, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 372)
        getitem___359943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 16), blocksize_359942, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 372)
        subscript_call_result_359944 = invoke(stypy.reporting.localization.Localization(__file__, 372, 16), getitem___359943, int_359940)
        
        # Assigning a type to the variable 'C' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'C', subscript_call_result_359944)
        # SSA branch for the else part of an if statement (line 371)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 374):
        
        # Assigning a Num to a Name (line 374):
        int_359945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 16), 'int')
        # Assigning a type to the variable 'C' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'C', int_359945)
        # SSA join for if statement (line 371)
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 376, 8))
        
        # 'from scipy.sparse.csr import isspmatrix_csr' statement (line 376)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_359946 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 376, 8), 'scipy.sparse.csr')

        if (type(import_359946) is not StypyTypeError):

            if (import_359946 != 'pyd_module'):
                __import__(import_359946)
                sys_modules_359947 = sys.modules[import_359946]
                import_from_module(stypy.reporting.localization.Localization(__file__, 376, 8), 'scipy.sparse.csr', sys_modules_359947.module_type_store, module_type_store, ['isspmatrix_csr'])
                nest_module(stypy.reporting.localization.Localization(__file__, 376, 8), __file__, sys_modules_359947, sys_modules_359947.module_type_store, module_type_store)
            else:
                from scipy.sparse.csr import isspmatrix_csr

                import_from_module(stypy.reporting.localization.Localization(__file__, 376, 8), 'scipy.sparse.csr', None, module_type_store, ['isspmatrix_csr'], [isspmatrix_csr])

        else:
            # Assigning a type to the variable 'scipy.sparse.csr' (line 376)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'scipy.sparse.csr', import_359946)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        
        # Evaluating a boolean operation
        
        # Call to isspmatrix_csr(...): (line 378)
        # Processing the call arguments (line 378)
        # Getting the type of 'other' (line 378)
        other_359949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 26), 'other', False)
        # Processing the call keyword arguments (line 378)
        kwargs_359950 = {}
        # Getting the type of 'isspmatrix_csr' (line 378)
        isspmatrix_csr_359948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 11), 'isspmatrix_csr', False)
        # Calling isspmatrix_csr(args, kwargs) (line 378)
        isspmatrix_csr_call_result_359951 = invoke(stypy.reporting.localization.Localization(__file__, 378, 11), isspmatrix_csr_359948, *[other_359949], **kwargs_359950)
        
        
        # Getting the type of 'n' (line 378)
        n_359952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 37), 'n')
        int_359953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 42), 'int')
        # Applying the binary operator '==' (line 378)
        result_eq_359954 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 37), '==', n_359952, int_359953)
        
        # Applying the binary operator 'and' (line 378)
        result_and_keyword_359955 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 11), 'and', isspmatrix_csr_call_result_359951, result_eq_359954)
        
        # Testing the type of an if condition (line 378)
        if_condition_359956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 8), result_and_keyword_359955)
        # Assigning a type to the variable 'if_condition_359956' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'if_condition_359956', if_condition_359956)
        # SSA begins for if statement (line 378)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 379):
        
        # Assigning a Call to a Name (line 379):
        
        # Call to tobsr(...): (line 379)
        # Processing the call keyword arguments (line 379)
        
        # Obtaining an instance of the builtin type 'tuple' (line 379)
        tuple_359959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 379)
        # Adding element type (line 379)
        # Getting the type of 'n' (line 379)
        n_359960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 43), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 43), tuple_359959, n_359960)
        # Adding element type (line 379)
        # Getting the type of 'C' (line 379)
        C_359961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 45), 'C', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 43), tuple_359959, C_359961)
        
        keyword_359962 = tuple_359959
        # Getting the type of 'False' (line 379)
        False_359963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 54), 'False', False)
        keyword_359964 = False_359963
        kwargs_359965 = {'blocksize': keyword_359962, 'copy': keyword_359964}
        # Getting the type of 'other' (line 379)
        other_359957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 20), 'other', False)
        # Obtaining the member 'tobsr' of a type (line 379)
        tobsr_359958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 20), other_359957, 'tobsr')
        # Calling tobsr(args, kwargs) (line 379)
        tobsr_call_result_359966 = invoke(stypy.reporting.localization.Localization(__file__, 379, 20), tobsr_359958, *[], **kwargs_359965)
        
        # Assigning a type to the variable 'other' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'other', tobsr_call_result_359966)
        # SSA branch for the else part of an if statement (line 378)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 381):
        
        # Assigning a Call to a Name (line 381):
        
        # Call to tobsr(...): (line 381)
        # Processing the call keyword arguments (line 381)
        
        # Obtaining an instance of the builtin type 'tuple' (line 381)
        tuple_359969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 381)
        # Adding element type (line 381)
        # Getting the type of 'n' (line 381)
        n_359970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 43), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 43), tuple_359969, n_359970)
        # Adding element type (line 381)
        # Getting the type of 'C' (line 381)
        C_359971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 45), 'C', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 43), tuple_359969, C_359971)
        
        keyword_359972 = tuple_359969
        kwargs_359973 = {'blocksize': keyword_359972}
        # Getting the type of 'other' (line 381)
        other_359967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 20), 'other', False)
        # Obtaining the member 'tobsr' of a type (line 381)
        tobsr_359968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 20), other_359967, 'tobsr')
        # Calling tobsr(args, kwargs) (line 381)
        tobsr_call_result_359974 = invoke(stypy.reporting.localization.Localization(__file__, 381, 20), tobsr_359968, *[], **kwargs_359973)
        
        # Assigning a type to the variable 'other' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'other', tobsr_call_result_359974)
        # SSA join for if statement (line 378)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 383):
        
        # Assigning a Call to a Name (line 383):
        
        # Call to get_index_dtype(...): (line 383)
        # Processing the call arguments (line 383)
        
        # Obtaining an instance of the builtin type 'tuple' (line 383)
        tuple_359976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 383)
        # Adding element type (line 383)
        # Getting the type of 'self' (line 383)
        self_359977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 37), 'self', False)
        # Obtaining the member 'indptr' of a type (line 383)
        indptr_359978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 37), self_359977, 'indptr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 37), tuple_359976, indptr_359978)
        # Adding element type (line 383)
        # Getting the type of 'self' (line 383)
        self_359979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 50), 'self', False)
        # Obtaining the member 'indices' of a type (line 383)
        indices_359980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 50), self_359979, 'indices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 37), tuple_359976, indices_359980)
        # Adding element type (line 383)
        # Getting the type of 'other' (line 384)
        other_359981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 37), 'other', False)
        # Obtaining the member 'indptr' of a type (line 384)
        indptr_359982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 37), other_359981, 'indptr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 37), tuple_359976, indptr_359982)
        # Adding element type (line 383)
        # Getting the type of 'other' (line 384)
        other_359983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 51), 'other', False)
        # Obtaining the member 'indices' of a type (line 384)
        indices_359984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 51), other_359983, 'indices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 37), tuple_359976, indices_359984)
        
        # Processing the call keyword arguments (line 383)
        # Getting the type of 'M' (line 385)
        M_359985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 44), 'M', False)
        # Getting the type of 'R' (line 385)
        R_359986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 47), 'R', False)
        # Applying the binary operator '//' (line 385)
        result_floordiv_359987 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 44), '//', M_359985, R_359986)
        
        # Getting the type of 'N' (line 385)
        N_359988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 51), 'N', False)
        # Getting the type of 'C' (line 385)
        C_359989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 54), 'C', False)
        # Applying the binary operator '//' (line 385)
        result_floordiv_359990 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 51), '//', N_359988, C_359989)
        
        # Applying the binary operator '*' (line 385)
        result_mul_359991 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 43), '*', result_floordiv_359987, result_floordiv_359990)
        
        keyword_359992 = result_mul_359991
        kwargs_359993 = {'maxval': keyword_359992}
        # Getting the type of 'get_index_dtype' (line 383)
        get_index_dtype_359975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 20), 'get_index_dtype', False)
        # Calling get_index_dtype(args, kwargs) (line 383)
        get_index_dtype_call_result_359994 = invoke(stypy.reporting.localization.Localization(__file__, 383, 20), get_index_dtype_359975, *[tuple_359976], **kwargs_359993)
        
        # Assigning a type to the variable 'idx_dtype' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'idx_dtype', get_index_dtype_call_result_359994)
        
        # Assigning a Call to a Name (line 386):
        
        # Assigning a Call to a Name (line 386):
        
        # Call to empty(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'self' (line 386)
        self_359997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 26), 'self', False)
        # Obtaining the member 'indptr' of a type (line 386)
        indptr_359998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 26), self_359997, 'indptr')
        # Obtaining the member 'shape' of a type (line 386)
        shape_359999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 26), indptr_359998, 'shape')
        # Processing the call keyword arguments (line 386)
        # Getting the type of 'idx_dtype' (line 386)
        idx_dtype_360000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 51), 'idx_dtype', False)
        keyword_360001 = idx_dtype_360000
        kwargs_360002 = {'dtype': keyword_360001}
        # Getting the type of 'np' (line 386)
        np_359995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 17), 'np', False)
        # Obtaining the member 'empty' of a type (line 386)
        empty_359996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 17), np_359995, 'empty')
        # Calling empty(args, kwargs) (line 386)
        empty_call_result_360003 = invoke(stypy.reporting.localization.Localization(__file__, 386, 17), empty_359996, *[shape_359999], **kwargs_360002)
        
        # Assigning a type to the variable 'indptr' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'indptr', empty_call_result_360003)
        
        # Call to csr_matmat_pass1(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'M' (line 388)
        M_360005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 25), 'M', False)
        # Getting the type of 'R' (line 388)
        R_360006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 28), 'R', False)
        # Applying the binary operator '//' (line 388)
        result_floordiv_360007 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 25), '//', M_360005, R_360006)
        
        # Getting the type of 'N' (line 388)
        N_360008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 31), 'N', False)
        # Getting the type of 'C' (line 388)
        C_360009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 34), 'C', False)
        # Applying the binary operator '//' (line 388)
        result_floordiv_360010 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 31), '//', N_360008, C_360009)
        
        
        # Call to astype(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'idx_dtype' (line 389)
        idx_dtype_360014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 44), 'idx_dtype', False)
        # Processing the call keyword arguments (line 389)
        kwargs_360015 = {}
        # Getting the type of 'self' (line 389)
        self_360011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 25), 'self', False)
        # Obtaining the member 'indptr' of a type (line 389)
        indptr_360012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 25), self_360011, 'indptr')
        # Obtaining the member 'astype' of a type (line 389)
        astype_360013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 25), indptr_360012, 'astype')
        # Calling astype(args, kwargs) (line 389)
        astype_call_result_360016 = invoke(stypy.reporting.localization.Localization(__file__, 389, 25), astype_360013, *[idx_dtype_360014], **kwargs_360015)
        
        
        # Call to astype(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'idx_dtype' (line 390)
        idx_dtype_360020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 45), 'idx_dtype', False)
        # Processing the call keyword arguments (line 390)
        kwargs_360021 = {}
        # Getting the type of 'self' (line 390)
        self_360017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 25), 'self', False)
        # Obtaining the member 'indices' of a type (line 390)
        indices_360018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 25), self_360017, 'indices')
        # Obtaining the member 'astype' of a type (line 390)
        astype_360019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 25), indices_360018, 'astype')
        # Calling astype(args, kwargs) (line 390)
        astype_call_result_360022 = invoke(stypy.reporting.localization.Localization(__file__, 390, 25), astype_360019, *[idx_dtype_360020], **kwargs_360021)
        
        
        # Call to astype(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'idx_dtype' (line 391)
        idx_dtype_360026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 45), 'idx_dtype', False)
        # Processing the call keyword arguments (line 391)
        kwargs_360027 = {}
        # Getting the type of 'other' (line 391)
        other_360023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 25), 'other', False)
        # Obtaining the member 'indptr' of a type (line 391)
        indptr_360024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 25), other_360023, 'indptr')
        # Obtaining the member 'astype' of a type (line 391)
        astype_360025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 25), indptr_360024, 'astype')
        # Calling astype(args, kwargs) (line 391)
        astype_call_result_360028 = invoke(stypy.reporting.localization.Localization(__file__, 391, 25), astype_360025, *[idx_dtype_360026], **kwargs_360027)
        
        
        # Call to astype(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'idx_dtype' (line 392)
        idx_dtype_360032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 46), 'idx_dtype', False)
        # Processing the call keyword arguments (line 392)
        kwargs_360033 = {}
        # Getting the type of 'other' (line 392)
        other_360029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 25), 'other', False)
        # Obtaining the member 'indices' of a type (line 392)
        indices_360030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 25), other_360029, 'indices')
        # Obtaining the member 'astype' of a type (line 392)
        astype_360031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 25), indices_360030, 'astype')
        # Calling astype(args, kwargs) (line 392)
        astype_call_result_360034 = invoke(stypy.reporting.localization.Localization(__file__, 392, 25), astype_360031, *[idx_dtype_360032], **kwargs_360033)
        
        # Getting the type of 'indptr' (line 393)
        indptr_360035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 25), 'indptr', False)
        # Processing the call keyword arguments (line 388)
        kwargs_360036 = {}
        # Getting the type of 'csr_matmat_pass1' (line 388)
        csr_matmat_pass1_360004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'csr_matmat_pass1', False)
        # Calling csr_matmat_pass1(args, kwargs) (line 388)
        csr_matmat_pass1_call_result_360037 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), csr_matmat_pass1_360004, *[result_floordiv_360007, result_floordiv_360010, astype_call_result_360016, astype_call_result_360022, astype_call_result_360028, astype_call_result_360034, indptr_360035], **kwargs_360036)
        
        
        # Assigning a Subscript to a Name (line 395):
        
        # Assigning a Subscript to a Name (line 395):
        
        # Obtaining the type of the subscript
        int_360038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 22), 'int')
        # Getting the type of 'indptr' (line 395)
        indptr_360039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 15), 'indptr')
        # Obtaining the member '__getitem__' of a type (line 395)
        getitem___360040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 15), indptr_360039, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 395)
        subscript_call_result_360041 = invoke(stypy.reporting.localization.Localization(__file__, 395, 15), getitem___360040, int_360038)
        
        # Assigning a type to the variable 'bnnz' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'bnnz', subscript_call_result_360041)
        
        # Assigning a Call to a Name (line 397):
        
        # Assigning a Call to a Name (line 397):
        
        # Call to get_index_dtype(...): (line 397)
        # Processing the call arguments (line 397)
        
        # Obtaining an instance of the builtin type 'tuple' (line 397)
        tuple_360043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 397)
        # Adding element type (line 397)
        # Getting the type of 'self' (line 397)
        self_360044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 37), 'self', False)
        # Obtaining the member 'indptr' of a type (line 397)
        indptr_360045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 37), self_360044, 'indptr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 37), tuple_360043, indptr_360045)
        # Adding element type (line 397)
        # Getting the type of 'self' (line 397)
        self_360046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 50), 'self', False)
        # Obtaining the member 'indices' of a type (line 397)
        indices_360047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 50), self_360046, 'indices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 37), tuple_360043, indices_360047)
        # Adding element type (line 397)
        # Getting the type of 'other' (line 398)
        other_360048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 37), 'other', False)
        # Obtaining the member 'indptr' of a type (line 398)
        indptr_360049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 37), other_360048, 'indptr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 37), tuple_360043, indptr_360049)
        # Adding element type (line 397)
        # Getting the type of 'other' (line 398)
        other_360050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 51), 'other', False)
        # Obtaining the member 'indices' of a type (line 398)
        indices_360051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 51), other_360050, 'indices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 37), tuple_360043, indices_360051)
        
        # Processing the call keyword arguments (line 397)
        # Getting the type of 'bnnz' (line 399)
        bnnz_360052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 43), 'bnnz', False)
        keyword_360053 = bnnz_360052
        kwargs_360054 = {'maxval': keyword_360053}
        # Getting the type of 'get_index_dtype' (line 397)
        get_index_dtype_360042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 20), 'get_index_dtype', False)
        # Calling get_index_dtype(args, kwargs) (line 397)
        get_index_dtype_call_result_360055 = invoke(stypy.reporting.localization.Localization(__file__, 397, 20), get_index_dtype_360042, *[tuple_360043], **kwargs_360054)
        
        # Assigning a type to the variable 'idx_dtype' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'idx_dtype', get_index_dtype_call_result_360055)
        
        # Assigning a Call to a Name (line 400):
        
        # Assigning a Call to a Name (line 400):
        
        # Call to astype(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'idx_dtype' (line 400)
        idx_dtype_360058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 31), 'idx_dtype', False)
        # Processing the call keyword arguments (line 400)
        kwargs_360059 = {}
        # Getting the type of 'indptr' (line 400)
        indptr_360056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 17), 'indptr', False)
        # Obtaining the member 'astype' of a type (line 400)
        astype_360057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 17), indptr_360056, 'astype')
        # Calling astype(args, kwargs) (line 400)
        astype_call_result_360060 = invoke(stypy.reporting.localization.Localization(__file__, 400, 17), astype_360057, *[idx_dtype_360058], **kwargs_360059)
        
        # Assigning a type to the variable 'indptr' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'indptr', astype_call_result_360060)
        
        # Assigning a Call to a Name (line 401):
        
        # Assigning a Call to a Name (line 401):
        
        # Call to empty(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'bnnz' (line 401)
        bnnz_360063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 27), 'bnnz', False)
        # Processing the call keyword arguments (line 401)
        # Getting the type of 'idx_dtype' (line 401)
        idx_dtype_360064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 39), 'idx_dtype', False)
        keyword_360065 = idx_dtype_360064
        kwargs_360066 = {'dtype': keyword_360065}
        # Getting the type of 'np' (line 401)
        np_360061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 18), 'np', False)
        # Obtaining the member 'empty' of a type (line 401)
        empty_360062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 18), np_360061, 'empty')
        # Calling empty(args, kwargs) (line 401)
        empty_call_result_360067 = invoke(stypy.reporting.localization.Localization(__file__, 401, 18), empty_360062, *[bnnz_360063], **kwargs_360066)
        
        # Assigning a type to the variable 'indices' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'indices', empty_call_result_360067)
        
        # Assigning a Call to a Name (line 402):
        
        # Assigning a Call to a Name (line 402):
        
        # Call to empty(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'R' (line 402)
        R_360070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 24), 'R', False)
        # Getting the type of 'C' (line 402)
        C_360071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 26), 'C', False)
        # Applying the binary operator '*' (line 402)
        result_mul_360072 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 24), '*', R_360070, C_360071)
        
        # Getting the type of 'bnnz' (line 402)
        bnnz_360073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 28), 'bnnz', False)
        # Applying the binary operator '*' (line 402)
        result_mul_360074 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 27), '*', result_mul_360072, bnnz_360073)
        
        # Processing the call keyword arguments (line 402)
        
        # Call to upcast(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'self' (line 402)
        self_360076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 47), 'self', False)
        # Obtaining the member 'dtype' of a type (line 402)
        dtype_360077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 47), self_360076, 'dtype')
        # Getting the type of 'other' (line 402)
        other_360078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 58), 'other', False)
        # Obtaining the member 'dtype' of a type (line 402)
        dtype_360079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 58), other_360078, 'dtype')
        # Processing the call keyword arguments (line 402)
        kwargs_360080 = {}
        # Getting the type of 'upcast' (line 402)
        upcast_360075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 40), 'upcast', False)
        # Calling upcast(args, kwargs) (line 402)
        upcast_call_result_360081 = invoke(stypy.reporting.localization.Localization(__file__, 402, 40), upcast_360075, *[dtype_360077, dtype_360079], **kwargs_360080)
        
        keyword_360082 = upcast_call_result_360081
        kwargs_360083 = {'dtype': keyword_360082}
        # Getting the type of 'np' (line 402)
        np_360068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), 'np', False)
        # Obtaining the member 'empty' of a type (line 402)
        empty_360069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 15), np_360068, 'empty')
        # Calling empty(args, kwargs) (line 402)
        empty_call_result_360084 = invoke(stypy.reporting.localization.Localization(__file__, 402, 15), empty_360069, *[result_mul_360074], **kwargs_360083)
        
        # Assigning a type to the variable 'data' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'data', empty_call_result_360084)
        
        # Call to bsr_matmat_pass2(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'M' (line 404)
        M_360086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 25), 'M', False)
        # Getting the type of 'R' (line 404)
        R_360087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 28), 'R', False)
        # Applying the binary operator '//' (line 404)
        result_floordiv_360088 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 25), '//', M_360086, R_360087)
        
        # Getting the type of 'N' (line 404)
        N_360089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 31), 'N', False)
        # Getting the type of 'C' (line 404)
        C_360090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 34), 'C', False)
        # Applying the binary operator '//' (line 404)
        result_floordiv_360091 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 31), '//', N_360089, C_360090)
        
        # Getting the type of 'R' (line 404)
        R_360092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 37), 'R', False)
        # Getting the type of 'C' (line 404)
        C_360093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 40), 'C', False)
        # Getting the type of 'n' (line 404)
        n_360094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 43), 'n', False)
        
        # Call to astype(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'idx_dtype' (line 405)
        idx_dtype_360098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 44), 'idx_dtype', False)
        # Processing the call keyword arguments (line 405)
        kwargs_360099 = {}
        # Getting the type of 'self' (line 405)
        self_360095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 25), 'self', False)
        # Obtaining the member 'indptr' of a type (line 405)
        indptr_360096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 25), self_360095, 'indptr')
        # Obtaining the member 'astype' of a type (line 405)
        astype_360097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 25), indptr_360096, 'astype')
        # Calling astype(args, kwargs) (line 405)
        astype_call_result_360100 = invoke(stypy.reporting.localization.Localization(__file__, 405, 25), astype_360097, *[idx_dtype_360098], **kwargs_360099)
        
        
        # Call to astype(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'idx_dtype' (line 406)
        idx_dtype_360104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 45), 'idx_dtype', False)
        # Processing the call keyword arguments (line 406)
        kwargs_360105 = {}
        # Getting the type of 'self' (line 406)
        self_360101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 25), 'self', False)
        # Obtaining the member 'indices' of a type (line 406)
        indices_360102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 25), self_360101, 'indices')
        # Obtaining the member 'astype' of a type (line 406)
        astype_360103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 25), indices_360102, 'astype')
        # Calling astype(args, kwargs) (line 406)
        astype_call_result_360106 = invoke(stypy.reporting.localization.Localization(__file__, 406, 25), astype_360103, *[idx_dtype_360104], **kwargs_360105)
        
        
        # Call to ravel(...): (line 407)
        # Processing the call arguments (line 407)
        # Getting the type of 'self' (line 407)
        self_360109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 34), 'self', False)
        # Obtaining the member 'data' of a type (line 407)
        data_360110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 34), self_360109, 'data')
        # Processing the call keyword arguments (line 407)
        kwargs_360111 = {}
        # Getting the type of 'np' (line 407)
        np_360107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 25), 'np', False)
        # Obtaining the member 'ravel' of a type (line 407)
        ravel_360108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 25), np_360107, 'ravel')
        # Calling ravel(args, kwargs) (line 407)
        ravel_call_result_360112 = invoke(stypy.reporting.localization.Localization(__file__, 407, 25), ravel_360108, *[data_360110], **kwargs_360111)
        
        
        # Call to astype(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'idx_dtype' (line 408)
        idx_dtype_360116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 45), 'idx_dtype', False)
        # Processing the call keyword arguments (line 408)
        kwargs_360117 = {}
        # Getting the type of 'other' (line 408)
        other_360113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 25), 'other', False)
        # Obtaining the member 'indptr' of a type (line 408)
        indptr_360114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 25), other_360113, 'indptr')
        # Obtaining the member 'astype' of a type (line 408)
        astype_360115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 25), indptr_360114, 'astype')
        # Calling astype(args, kwargs) (line 408)
        astype_call_result_360118 = invoke(stypy.reporting.localization.Localization(__file__, 408, 25), astype_360115, *[idx_dtype_360116], **kwargs_360117)
        
        
        # Call to astype(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'idx_dtype' (line 409)
        idx_dtype_360122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 46), 'idx_dtype', False)
        # Processing the call keyword arguments (line 409)
        kwargs_360123 = {}
        # Getting the type of 'other' (line 409)
        other_360119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 25), 'other', False)
        # Obtaining the member 'indices' of a type (line 409)
        indices_360120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 25), other_360119, 'indices')
        # Obtaining the member 'astype' of a type (line 409)
        astype_360121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 25), indices_360120, 'astype')
        # Calling astype(args, kwargs) (line 409)
        astype_call_result_360124 = invoke(stypy.reporting.localization.Localization(__file__, 409, 25), astype_360121, *[idx_dtype_360122], **kwargs_360123)
        
        
        # Call to ravel(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'other' (line 410)
        other_360127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 34), 'other', False)
        # Obtaining the member 'data' of a type (line 410)
        data_360128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 34), other_360127, 'data')
        # Processing the call keyword arguments (line 410)
        kwargs_360129 = {}
        # Getting the type of 'np' (line 410)
        np_360125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'np', False)
        # Obtaining the member 'ravel' of a type (line 410)
        ravel_360126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 25), np_360125, 'ravel')
        # Calling ravel(args, kwargs) (line 410)
        ravel_call_result_360130 = invoke(stypy.reporting.localization.Localization(__file__, 410, 25), ravel_360126, *[data_360128], **kwargs_360129)
        
        # Getting the type of 'indptr' (line 411)
        indptr_360131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 25), 'indptr', False)
        # Getting the type of 'indices' (line 412)
        indices_360132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 25), 'indices', False)
        # Getting the type of 'data' (line 413)
        data_360133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 25), 'data', False)
        # Processing the call keyword arguments (line 404)
        kwargs_360134 = {}
        # Getting the type of 'bsr_matmat_pass2' (line 404)
        bsr_matmat_pass2_360085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'bsr_matmat_pass2', False)
        # Calling bsr_matmat_pass2(args, kwargs) (line 404)
        bsr_matmat_pass2_call_result_360135 = invoke(stypy.reporting.localization.Localization(__file__, 404, 8), bsr_matmat_pass2_360085, *[result_floordiv_360088, result_floordiv_360091, R_360092, C_360093, n_360094, astype_call_result_360100, astype_call_result_360106, ravel_call_result_360112, astype_call_result_360118, astype_call_result_360124, ravel_call_result_360130, indptr_360131, indices_360132, data_360133], **kwargs_360134)
        
        
        # Assigning a Call to a Name (line 415):
        
        # Assigning a Call to a Name (line 415):
        
        # Call to reshape(...): (line 415)
        # Processing the call arguments (line 415)
        int_360138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 28), 'int')
        # Getting the type of 'R' (line 415)
        R_360139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 31), 'R', False)
        # Getting the type of 'C' (line 415)
        C_360140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 33), 'C', False)
        # Processing the call keyword arguments (line 415)
        kwargs_360141 = {}
        # Getting the type of 'data' (line 415)
        data_360136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 15), 'data', False)
        # Obtaining the member 'reshape' of a type (line 415)
        reshape_360137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 15), data_360136, 'reshape')
        # Calling reshape(args, kwargs) (line 415)
        reshape_call_result_360142 = invoke(stypy.reporting.localization.Localization(__file__, 415, 15), reshape_360137, *[int_360138, R_360139, C_360140], **kwargs_360141)
        
        # Assigning a type to the variable 'data' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'data', reshape_call_result_360142)
        
        # Call to bsr_matrix(...): (line 419)
        # Processing the call arguments (line 419)
        
        # Obtaining an instance of the builtin type 'tuple' (line 419)
        tuple_360144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 419)
        # Adding element type (line 419)
        # Getting the type of 'data' (line 419)
        data_360145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 27), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 27), tuple_360144, data_360145)
        # Adding element type (line 419)
        # Getting the type of 'indices' (line 419)
        indices_360146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 32), 'indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 27), tuple_360144, indices_360146)
        # Adding element type (line 419)
        # Getting the type of 'indptr' (line 419)
        indptr_360147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 40), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 27), tuple_360144, indptr_360147)
        
        # Processing the call keyword arguments (line 419)
        
        # Obtaining an instance of the builtin type 'tuple' (line 419)
        tuple_360148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 419)
        # Adding element type (line 419)
        # Getting the type of 'M' (line 419)
        M_360149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 55), 'M', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 55), tuple_360148, M_360149)
        # Adding element type (line 419)
        # Getting the type of 'N' (line 419)
        N_360150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 57), 'N', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 55), tuple_360148, N_360150)
        
        keyword_360151 = tuple_360148
        
        # Obtaining an instance of the builtin type 'tuple' (line 419)
        tuple_360152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 71), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 419)
        # Adding element type (line 419)
        # Getting the type of 'R' (line 419)
        R_360153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 71), 'R', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 71), tuple_360152, R_360153)
        # Adding element type (line 419)
        # Getting the type of 'C' (line 419)
        C_360154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 73), 'C', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 71), tuple_360152, C_360154)
        
        keyword_360155 = tuple_360152
        kwargs_360156 = {'blocksize': keyword_360155, 'shape': keyword_360151}
        # Getting the type of 'bsr_matrix' (line 419)
        bsr_matrix_360143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 15), 'bsr_matrix', False)
        # Calling bsr_matrix(args, kwargs) (line 419)
        bsr_matrix_call_result_360157 = invoke(stypy.reporting.localization.Localization(__file__, 419, 15), bsr_matrix_360143, *[tuple_360144], **kwargs_360156)
        
        # Assigning a type to the variable 'stypy_return_type' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'stypy_return_type', bsr_matrix_call_result_360157)
        
        # ################# End of '_mul_sparse_matrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_sparse_matrix' in the type store
        # Getting the type of 'stypy_return_type' (line 364)
        stypy_return_type_360158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_360158)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_sparse_matrix'
        return stypy_return_type_360158


    @norecursion
    def tobsr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 425)
        None_360159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 30), 'None')
        # Getting the type of 'False' (line 425)
        False_360160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 41), 'False')
        defaults = [None_360159, False_360160]
        # Create a new context for function 'tobsr'
        module_type_store = module_type_store.open_function_context('tobsr', 425, 4, False)
        # Assigning a type to the variable 'self' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.tobsr.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.tobsr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.tobsr.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.tobsr.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.tobsr')
        bsr_matrix.tobsr.__dict__.__setitem__('stypy_param_names_list', ['blocksize', 'copy'])
        bsr_matrix.tobsr.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.tobsr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.tobsr.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.tobsr.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.tobsr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.tobsr.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.tobsr', ['blocksize', 'copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tobsr', localization, ['blocksize', 'copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tobsr(...)' code ##################

        str_360161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, (-1)), 'str', 'Convert this matrix into Block Sparse Row Format.\n\n        With copy=False, the data/indices may be shared between this\n        matrix and the resultant bsr_matrix.\n\n        If blocksize=(R, C) is provided, it will be used for determining\n        block size of the bsr_matrix.\n        ')
        
        
        # Getting the type of 'blocksize' (line 434)
        blocksize_360162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 11), 'blocksize')
        
        # Obtaining an instance of the builtin type 'list' (line 434)
        list_360163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 434)
        # Adding element type (line 434)
        # Getting the type of 'None' (line 434)
        None_360164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 29), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 28), list_360163, None_360164)
        # Adding element type (line 434)
        # Getting the type of 'self' (line 434)
        self_360165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 35), 'self')
        # Obtaining the member 'blocksize' of a type (line 434)
        blocksize_360166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 35), self_360165, 'blocksize')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 28), list_360163, blocksize_360166)
        
        # Applying the binary operator 'notin' (line 434)
        result_contains_360167 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 11), 'notin', blocksize_360162, list_360163)
        
        # Testing the type of an if condition (line 434)
        if_condition_360168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 434, 8), result_contains_360167)
        # Assigning a type to the variable 'if_condition_360168' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'if_condition_360168', if_condition_360168)
        # SSA begins for if statement (line 434)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to tobsr(...): (line 435)
        # Processing the call keyword arguments (line 435)
        # Getting the type of 'blocksize' (line 435)
        blocksize_360174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 48), 'blocksize', False)
        keyword_360175 = blocksize_360174
        kwargs_360176 = {'blocksize': keyword_360175}
        
        # Call to tocsr(...): (line 435)
        # Processing the call keyword arguments (line 435)
        kwargs_360171 = {}
        # Getting the type of 'self' (line 435)
        self_360169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 19), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 435)
        tocsr_360170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 19), self_360169, 'tocsr')
        # Calling tocsr(args, kwargs) (line 435)
        tocsr_call_result_360172 = invoke(stypy.reporting.localization.Localization(__file__, 435, 19), tocsr_360170, *[], **kwargs_360171)
        
        # Obtaining the member 'tobsr' of a type (line 435)
        tobsr_360173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 19), tocsr_call_result_360172, 'tobsr')
        # Calling tobsr(args, kwargs) (line 435)
        tobsr_call_result_360177 = invoke(stypy.reporting.localization.Localization(__file__, 435, 19), tobsr_360173, *[], **kwargs_360176)
        
        # Assigning a type to the variable 'stypy_return_type' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'stypy_return_type', tobsr_call_result_360177)
        # SSA join for if statement (line 434)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'copy' (line 436)
        copy_360178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 11), 'copy')
        # Testing the type of an if condition (line 436)
        if_condition_360179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 436, 8), copy_360178)
        # Assigning a type to the variable 'if_condition_360179' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'if_condition_360179', if_condition_360179)
        # SSA begins for if statement (line 436)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 437)
        # Processing the call keyword arguments (line 437)
        kwargs_360182 = {}
        # Getting the type of 'self' (line 437)
        self_360180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 19), 'self', False)
        # Obtaining the member 'copy' of a type (line 437)
        copy_360181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 19), self_360180, 'copy')
        # Calling copy(args, kwargs) (line 437)
        copy_call_result_360183 = invoke(stypy.reporting.localization.Localization(__file__, 437, 19), copy_360181, *[], **kwargs_360182)
        
        # Assigning a type to the variable 'stypy_return_type' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'stypy_return_type', copy_call_result_360183)
        # SSA branch for the else part of an if statement (line 436)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'self' (line 439)
        self_360184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'stypy_return_type', self_360184)
        # SSA join for if statement (line 436)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'tobsr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tobsr' in the type store
        # Getting the type of 'stypy_return_type' (line 425)
        stypy_return_type_360185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_360185)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tobsr'
        return stypy_return_type_360185


    @norecursion
    def tocsr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 441)
        False_360186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 25), 'False')
        defaults = [False_360186]
        # Create a new context for function 'tocsr'
        module_type_store = module_type_store.open_function_context('tocsr', 441, 4, False)
        # Assigning a type to the variable 'self' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.tocsr.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.tocsr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.tocsr.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.tocsr.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.tocsr')
        bsr_matrix.tocsr.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        bsr_matrix.tocsr.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.tocsr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.tocsr.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.tocsr.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.tocsr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.tocsr.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.tocsr', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tocsr', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tocsr(...)' code ##################

        
        # Call to tocsr(...): (line 442)
        # Processing the call keyword arguments (line 442)
        # Getting the type of 'copy' (line 442)
        copy_360194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 49), 'copy', False)
        keyword_360195 = copy_360194
        kwargs_360196 = {'copy': keyword_360195}
        
        # Call to tocoo(...): (line 442)
        # Processing the call keyword arguments (line 442)
        # Getting the type of 'False' (line 442)
        False_360189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 31), 'False', False)
        keyword_360190 = False_360189
        kwargs_360191 = {'copy': keyword_360190}
        # Getting the type of 'self' (line 442)
        self_360187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 15), 'self', False)
        # Obtaining the member 'tocoo' of a type (line 442)
        tocoo_360188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 15), self_360187, 'tocoo')
        # Calling tocoo(args, kwargs) (line 442)
        tocoo_call_result_360192 = invoke(stypy.reporting.localization.Localization(__file__, 442, 15), tocoo_360188, *[], **kwargs_360191)
        
        # Obtaining the member 'tocsr' of a type (line 442)
        tocsr_360193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 15), tocoo_call_result_360192, 'tocsr')
        # Calling tocsr(args, kwargs) (line 442)
        tocsr_call_result_360197 = invoke(stypy.reporting.localization.Localization(__file__, 442, 15), tocsr_360193, *[], **kwargs_360196)
        
        # Assigning a type to the variable 'stypy_return_type' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'stypy_return_type', tocsr_call_result_360197)
        
        # ################# End of 'tocsr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocsr' in the type store
        # Getting the type of 'stypy_return_type' (line 441)
        stypy_return_type_360198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_360198)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocsr'
        return stypy_return_type_360198

    
    # Assigning a Attribute to a Attribute (line 445):

    @norecursion
    def tocsc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 447)
        False_360199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 25), 'False')
        defaults = [False_360199]
        # Create a new context for function 'tocsc'
        module_type_store = module_type_store.open_function_context('tocsc', 447, 4, False)
        # Assigning a type to the variable 'self' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.tocsc.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.tocsc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.tocsc.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.tocsc.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.tocsc')
        bsr_matrix.tocsc.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        bsr_matrix.tocsc.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.tocsc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.tocsc.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.tocsc.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.tocsc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.tocsc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.tocsc', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tocsc', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tocsc(...)' code ##################

        
        # Call to tocsc(...): (line 448)
        # Processing the call keyword arguments (line 448)
        # Getting the type of 'copy' (line 448)
        copy_360207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 49), 'copy', False)
        keyword_360208 = copy_360207
        kwargs_360209 = {'copy': keyword_360208}
        
        # Call to tocoo(...): (line 448)
        # Processing the call keyword arguments (line 448)
        # Getting the type of 'False' (line 448)
        False_360202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 31), 'False', False)
        keyword_360203 = False_360202
        kwargs_360204 = {'copy': keyword_360203}
        # Getting the type of 'self' (line 448)
        self_360200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 15), 'self', False)
        # Obtaining the member 'tocoo' of a type (line 448)
        tocoo_360201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 15), self_360200, 'tocoo')
        # Calling tocoo(args, kwargs) (line 448)
        tocoo_call_result_360205 = invoke(stypy.reporting.localization.Localization(__file__, 448, 15), tocoo_360201, *[], **kwargs_360204)
        
        # Obtaining the member 'tocsc' of a type (line 448)
        tocsc_360206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 15), tocoo_call_result_360205, 'tocsc')
        # Calling tocsc(args, kwargs) (line 448)
        tocsc_call_result_360210 = invoke(stypy.reporting.localization.Localization(__file__, 448, 15), tocsc_360206, *[], **kwargs_360209)
        
        # Assigning a type to the variable 'stypy_return_type' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'stypy_return_type', tocsc_call_result_360210)
        
        # ################# End of 'tocsc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocsc' in the type store
        # Getting the type of 'stypy_return_type' (line 447)
        stypy_return_type_360211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_360211)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocsc'
        return stypy_return_type_360211

    
    # Assigning a Attribute to a Attribute (line 450):

    @norecursion
    def tocoo(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 452)
        True_360212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 25), 'True')
        defaults = [True_360212]
        # Create a new context for function 'tocoo'
        module_type_store = module_type_store.open_function_context('tocoo', 452, 4, False)
        # Assigning a type to the variable 'self' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.tocoo.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.tocoo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.tocoo.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.tocoo.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.tocoo')
        bsr_matrix.tocoo.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        bsr_matrix.tocoo.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.tocoo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.tocoo.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.tocoo.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.tocoo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.tocoo.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.tocoo', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tocoo', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tocoo(...)' code ##################

        str_360213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, (-1)), 'str', 'Convert this matrix to COOrdinate format.\n\n        When copy=False the data array will be shared between\n        this matrix and the resultant coo_matrix.\n        ')
        
        # Assigning a Attribute to a Tuple (line 459):
        
        # Assigning a Subscript to a Name (line 459):
        
        # Obtaining the type of the subscript
        int_360214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 8), 'int')
        # Getting the type of 'self' (line 459)
        self_360215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 14), 'self')
        # Obtaining the member 'shape' of a type (line 459)
        shape_360216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 14), self_360215, 'shape')
        # Obtaining the member '__getitem__' of a type (line 459)
        getitem___360217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), shape_360216, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 459)
        subscript_call_result_360218 = invoke(stypy.reporting.localization.Localization(__file__, 459, 8), getitem___360217, int_360214)
        
        # Assigning a type to the variable 'tuple_var_assignment_358889' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'tuple_var_assignment_358889', subscript_call_result_360218)
        
        # Assigning a Subscript to a Name (line 459):
        
        # Obtaining the type of the subscript
        int_360219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 8), 'int')
        # Getting the type of 'self' (line 459)
        self_360220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 14), 'self')
        # Obtaining the member 'shape' of a type (line 459)
        shape_360221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 14), self_360220, 'shape')
        # Obtaining the member '__getitem__' of a type (line 459)
        getitem___360222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), shape_360221, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 459)
        subscript_call_result_360223 = invoke(stypy.reporting.localization.Localization(__file__, 459, 8), getitem___360222, int_360219)
        
        # Assigning a type to the variable 'tuple_var_assignment_358890' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'tuple_var_assignment_358890', subscript_call_result_360223)
        
        # Assigning a Name to a Name (line 459):
        # Getting the type of 'tuple_var_assignment_358889' (line 459)
        tuple_var_assignment_358889_360224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'tuple_var_assignment_358889')
        # Assigning a type to the variable 'M' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'M', tuple_var_assignment_358889_360224)
        
        # Assigning a Name to a Name (line 459):
        # Getting the type of 'tuple_var_assignment_358890' (line 459)
        tuple_var_assignment_358890_360225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'tuple_var_assignment_358890')
        # Assigning a type to the variable 'N' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 10), 'N', tuple_var_assignment_358890_360225)
        
        # Assigning a Attribute to a Tuple (line 460):
        
        # Assigning a Subscript to a Name (line 460):
        
        # Obtaining the type of the subscript
        int_360226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 8), 'int')
        # Getting the type of 'self' (line 460)
        self_360227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 460)
        blocksize_360228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 14), self_360227, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 460)
        getitem___360229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 8), blocksize_360228, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 460)
        subscript_call_result_360230 = invoke(stypy.reporting.localization.Localization(__file__, 460, 8), getitem___360229, int_360226)
        
        # Assigning a type to the variable 'tuple_var_assignment_358891' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'tuple_var_assignment_358891', subscript_call_result_360230)
        
        # Assigning a Subscript to a Name (line 460):
        
        # Obtaining the type of the subscript
        int_360231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 8), 'int')
        # Getting the type of 'self' (line 460)
        self_360232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 460)
        blocksize_360233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 14), self_360232, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 460)
        getitem___360234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 8), blocksize_360233, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 460)
        subscript_call_result_360235 = invoke(stypy.reporting.localization.Localization(__file__, 460, 8), getitem___360234, int_360231)
        
        # Assigning a type to the variable 'tuple_var_assignment_358892' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'tuple_var_assignment_358892', subscript_call_result_360235)
        
        # Assigning a Name to a Name (line 460):
        # Getting the type of 'tuple_var_assignment_358891' (line 460)
        tuple_var_assignment_358891_360236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'tuple_var_assignment_358891')
        # Assigning a type to the variable 'R' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'R', tuple_var_assignment_358891_360236)
        
        # Assigning a Name to a Name (line 460):
        # Getting the type of 'tuple_var_assignment_358892' (line 460)
        tuple_var_assignment_358892_360237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'tuple_var_assignment_358892')
        # Assigning a type to the variable 'C' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 10), 'C', tuple_var_assignment_358892_360237)
        
        # Assigning a Call to a Name (line 462):
        
        # Assigning a Call to a Name (line 462):
        
        # Call to diff(...): (line 462)
        # Processing the call arguments (line 462)
        # Getting the type of 'self' (line 462)
        self_360240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 30), 'self', False)
        # Obtaining the member 'indptr' of a type (line 462)
        indptr_360241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 30), self_360240, 'indptr')
        # Processing the call keyword arguments (line 462)
        kwargs_360242 = {}
        # Getting the type of 'np' (line 462)
        np_360238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 22), 'np', False)
        # Obtaining the member 'diff' of a type (line 462)
        diff_360239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 22), np_360238, 'diff')
        # Calling diff(args, kwargs) (line 462)
        diff_call_result_360243 = invoke(stypy.reporting.localization.Localization(__file__, 462, 22), diff_360239, *[indptr_360241], **kwargs_360242)
        
        # Assigning a type to the variable 'indptr_diff' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'indptr_diff', diff_call_result_360243)
        
        
        # Getting the type of 'indptr_diff' (line 463)
        indptr_diff_360244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'indptr_diff')
        # Obtaining the member 'dtype' of a type (line 463)
        dtype_360245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 11), indptr_diff_360244, 'dtype')
        # Obtaining the member 'itemsize' of a type (line 463)
        itemsize_360246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 11), dtype_360245, 'itemsize')
        
        # Call to dtype(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'np' (line 463)
        np_360249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 49), 'np', False)
        # Obtaining the member 'intp' of a type (line 463)
        intp_360250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 49), np_360249, 'intp')
        # Processing the call keyword arguments (line 463)
        kwargs_360251 = {}
        # Getting the type of 'np' (line 463)
        np_360247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 40), 'np', False)
        # Obtaining the member 'dtype' of a type (line 463)
        dtype_360248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 40), np_360247, 'dtype')
        # Calling dtype(args, kwargs) (line 463)
        dtype_call_result_360252 = invoke(stypy.reporting.localization.Localization(__file__, 463, 40), dtype_360248, *[intp_360250], **kwargs_360251)
        
        # Obtaining the member 'itemsize' of a type (line 463)
        itemsize_360253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 40), dtype_call_result_360252, 'itemsize')
        # Applying the binary operator '>' (line 463)
        result_gt_360254 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 11), '>', itemsize_360246, itemsize_360253)
        
        # Testing the type of an if condition (line 463)
        if_condition_360255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 8), result_gt_360254)
        # Assigning a type to the variable 'if_condition_360255' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'if_condition_360255', if_condition_360255)
        # SSA begins for if statement (line 463)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 465):
        
        # Assigning a Call to a Name (line 465):
        
        # Call to astype(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'np' (line 465)
        np_360258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 53), 'np', False)
        # Obtaining the member 'intp' of a type (line 465)
        intp_360259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 53), np_360258, 'intp')
        # Processing the call keyword arguments (line 465)
        kwargs_360260 = {}
        # Getting the type of 'indptr_diff' (line 465)
        indptr_diff_360256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 34), 'indptr_diff', False)
        # Obtaining the member 'astype' of a type (line 465)
        astype_360257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 34), indptr_diff_360256, 'astype')
        # Calling astype(args, kwargs) (line 465)
        astype_call_result_360261 = invoke(stypy.reporting.localization.Localization(__file__, 465, 34), astype_360257, *[intp_360259], **kwargs_360260)
        
        # Assigning a type to the variable 'indptr_diff_limited' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'indptr_diff_limited', astype_call_result_360261)
        
        
        # Call to any(...): (line 466)
        # Processing the call arguments (line 466)
        
        # Getting the type of 'indptr_diff_limited' (line 466)
        indptr_diff_limited_360264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 22), 'indptr_diff_limited', False)
        # Getting the type of 'indptr_diff' (line 466)
        indptr_diff_360265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 45), 'indptr_diff', False)
        # Applying the binary operator '!=' (line 466)
        result_ne_360266 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 22), '!=', indptr_diff_limited_360264, indptr_diff_360265)
        
        # Processing the call keyword arguments (line 466)
        kwargs_360267 = {}
        # Getting the type of 'np' (line 466)
        np_360262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 15), 'np', False)
        # Obtaining the member 'any' of a type (line 466)
        any_360263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 15), np_360262, 'any')
        # Calling any(args, kwargs) (line 466)
        any_call_result_360268 = invoke(stypy.reporting.localization.Localization(__file__, 466, 15), any_360263, *[result_ne_360266], **kwargs_360267)
        
        # Testing the type of an if condition (line 466)
        if_condition_360269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 466, 12), any_call_result_360268)
        # Assigning a type to the variable 'if_condition_360269' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'if_condition_360269', if_condition_360269)
        # SSA begins for if statement (line 466)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 467)
        # Processing the call arguments (line 467)
        str_360271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 33), 'str', 'Matrix too big to convert')
        # Processing the call keyword arguments (line 467)
        kwargs_360272 = {}
        # Getting the type of 'ValueError' (line 467)
        ValueError_360270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 467)
        ValueError_call_result_360273 = invoke(stypy.reporting.localization.Localization(__file__, 467, 22), ValueError_360270, *[str_360271], **kwargs_360272)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 467, 16), ValueError_call_result_360273, 'raise parameter', BaseException)
        # SSA join for if statement (line 466)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 468):
        
        # Assigning a Name to a Name (line 468):
        # Getting the type of 'indptr_diff_limited' (line 468)
        indptr_diff_limited_360274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 26), 'indptr_diff_limited')
        # Assigning a type to the variable 'indptr_diff' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'indptr_diff', indptr_diff_limited_360274)
        # SSA join for if statement (line 463)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 470):
        
        # Assigning a Call to a Name (line 470):
        
        # Call to repeat(...): (line 470)
        # Processing the call arguments (line 470)
        # Getting the type of 'indptr_diff' (line 470)
        indptr_diff_360285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 43), 'indptr_diff', False)
        # Processing the call keyword arguments (line 470)
        kwargs_360286 = {}
        # Getting the type of 'R' (line 470)
        R_360275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 15), 'R', False)
        
        # Call to arange(...): (line 470)
        # Processing the call arguments (line 470)
        # Getting the type of 'M' (line 470)
        M_360278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 29), 'M', False)
        # Getting the type of 'R' (line 470)
        R_360279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 32), 'R', False)
        # Applying the binary operator '//' (line 470)
        result_floordiv_360280 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 29), '//', M_360278, R_360279)
        
        # Processing the call keyword arguments (line 470)
        kwargs_360281 = {}
        # Getting the type of 'np' (line 470)
        np_360276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 19), 'np', False)
        # Obtaining the member 'arange' of a type (line 470)
        arange_360277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 19), np_360276, 'arange')
        # Calling arange(args, kwargs) (line 470)
        arange_call_result_360282 = invoke(stypy.reporting.localization.Localization(__file__, 470, 19), arange_360277, *[result_floordiv_360280], **kwargs_360281)
        
        # Applying the binary operator '*' (line 470)
        result_mul_360283 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 15), '*', R_360275, arange_call_result_360282)
        
        # Obtaining the member 'repeat' of a type (line 470)
        repeat_360284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 15), result_mul_360283, 'repeat')
        # Calling repeat(args, kwargs) (line 470)
        repeat_call_result_360287 = invoke(stypy.reporting.localization.Localization(__file__, 470, 15), repeat_360284, *[indptr_diff_360285], **kwargs_360286)
        
        # Assigning a type to the variable 'row' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'row', repeat_call_result_360287)
        
        # Assigning a Call to a Name (line 471):
        
        # Assigning a Call to a Name (line 471):
        
        # Call to reshape(...): (line 471)
        # Processing the call arguments (line 471)
        int_360296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 38), 'int')
        # Getting the type of 'R' (line 471)
        R_360297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 41), 'R', False)
        # Getting the type of 'C' (line 471)
        C_360298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 43), 'C', False)
        # Processing the call keyword arguments (line 471)
        kwargs_360299 = {}
        
        # Call to repeat(...): (line 471)
        # Processing the call arguments (line 471)
        # Getting the type of 'R' (line 471)
        R_360290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 25), 'R', False)
        # Getting the type of 'C' (line 471)
        C_360291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 27), 'C', False)
        # Applying the binary operator '*' (line 471)
        result_mul_360292 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 25), '*', R_360290, C_360291)
        
        # Processing the call keyword arguments (line 471)
        kwargs_360293 = {}
        # Getting the type of 'row' (line 471)
        row_360288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 14), 'row', False)
        # Obtaining the member 'repeat' of a type (line 471)
        repeat_360289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 14), row_360288, 'repeat')
        # Calling repeat(args, kwargs) (line 471)
        repeat_call_result_360294 = invoke(stypy.reporting.localization.Localization(__file__, 471, 14), repeat_360289, *[result_mul_360292], **kwargs_360293)
        
        # Obtaining the member 'reshape' of a type (line 471)
        reshape_360295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 14), repeat_call_result_360294, 'reshape')
        # Calling reshape(args, kwargs) (line 471)
        reshape_call_result_360300 = invoke(stypy.reporting.localization.Localization(__file__, 471, 14), reshape_360295, *[int_360296, R_360297, C_360298], **kwargs_360299)
        
        # Assigning a type to the variable 'row' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'row', reshape_call_result_360300)
        
        # Getting the type of 'row' (line 472)
        row_360301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'row')
        
        # Call to tile(...): (line 472)
        # Processing the call arguments (line 472)
        
        # Call to reshape(...): (line 472)
        # Processing the call arguments (line 472)
        int_360310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 44), 'int')
        int_360311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 47), 'int')
        # Processing the call keyword arguments (line 472)
        kwargs_360312 = {}
        
        # Call to arange(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'R' (line 472)
        R_360306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 33), 'R', False)
        # Processing the call keyword arguments (line 472)
        kwargs_360307 = {}
        # Getting the type of 'np' (line 472)
        np_360304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 23), 'np', False)
        # Obtaining the member 'arange' of a type (line 472)
        arange_360305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 23), np_360304, 'arange')
        # Calling arange(args, kwargs) (line 472)
        arange_call_result_360308 = invoke(stypy.reporting.localization.Localization(__file__, 472, 23), arange_360305, *[R_360306], **kwargs_360307)
        
        # Obtaining the member 'reshape' of a type (line 472)
        reshape_360309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 23), arange_call_result_360308, 'reshape')
        # Calling reshape(args, kwargs) (line 472)
        reshape_call_result_360313 = invoke(stypy.reporting.localization.Localization(__file__, 472, 23), reshape_360309, *[int_360310, int_360311], **kwargs_360312)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 472)
        tuple_360314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 472)
        # Adding element type (line 472)
        int_360315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 52), tuple_360314, int_360315)
        # Adding element type (line 472)
        # Getting the type of 'C' (line 472)
        C_360316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 54), 'C', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 52), tuple_360314, C_360316)
        
        # Processing the call keyword arguments (line 472)
        kwargs_360317 = {}
        # Getting the type of 'np' (line 472)
        np_360302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'np', False)
        # Obtaining the member 'tile' of a type (line 472)
        tile_360303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 15), np_360302, 'tile')
        # Calling tile(args, kwargs) (line 472)
        tile_call_result_360318 = invoke(stypy.reporting.localization.Localization(__file__, 472, 15), tile_360303, *[reshape_call_result_360313, tuple_360314], **kwargs_360317)
        
        # Applying the binary operator '+=' (line 472)
        result_iadd_360319 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 8), '+=', row_360301, tile_call_result_360318)
        # Assigning a type to the variable 'row' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'row', result_iadd_360319)
        
        
        # Assigning a Call to a Name (line 473):
        
        # Assigning a Call to a Name (line 473):
        
        # Call to reshape(...): (line 473)
        # Processing the call arguments (line 473)
        int_360322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 26), 'int')
        # Processing the call keyword arguments (line 473)
        kwargs_360323 = {}
        # Getting the type of 'row' (line 473)
        row_360320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 14), 'row', False)
        # Obtaining the member 'reshape' of a type (line 473)
        reshape_360321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 14), row_360320, 'reshape')
        # Calling reshape(args, kwargs) (line 473)
        reshape_call_result_360324 = invoke(stypy.reporting.localization.Localization(__file__, 473, 14), reshape_360321, *[int_360322], **kwargs_360323)
        
        # Assigning a type to the variable 'row' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'row', reshape_call_result_360324)
        
        # Assigning a Call to a Name (line 475):
        
        # Assigning a Call to a Name (line 475):
        
        # Call to reshape(...): (line 475)
        # Processing the call arguments (line 475)
        int_360336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 53), 'int')
        # Getting the type of 'R' (line 475)
        R_360337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 56), 'R', False)
        # Getting the type of 'C' (line 475)
        C_360338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 58), 'C', False)
        # Processing the call keyword arguments (line 475)
        kwargs_360339 = {}
        
        # Call to repeat(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'R' (line 475)
        R_360330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 40), 'R', False)
        # Getting the type of 'C' (line 475)
        C_360331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 42), 'C', False)
        # Applying the binary operator '*' (line 475)
        result_mul_360332 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 40), '*', R_360330, C_360331)
        
        # Processing the call keyword arguments (line 475)
        kwargs_360333 = {}
        # Getting the type of 'C' (line 475)
        C_360325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 15), 'C', False)
        # Getting the type of 'self' (line 475)
        self_360326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 19), 'self', False)
        # Obtaining the member 'indices' of a type (line 475)
        indices_360327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 19), self_360326, 'indices')
        # Applying the binary operator '*' (line 475)
        result_mul_360328 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 15), '*', C_360325, indices_360327)
        
        # Obtaining the member 'repeat' of a type (line 475)
        repeat_360329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 15), result_mul_360328, 'repeat')
        # Calling repeat(args, kwargs) (line 475)
        repeat_call_result_360334 = invoke(stypy.reporting.localization.Localization(__file__, 475, 15), repeat_360329, *[result_mul_360332], **kwargs_360333)
        
        # Obtaining the member 'reshape' of a type (line 475)
        reshape_360335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 15), repeat_call_result_360334, 'reshape')
        # Calling reshape(args, kwargs) (line 475)
        reshape_call_result_360340 = invoke(stypy.reporting.localization.Localization(__file__, 475, 15), reshape_360335, *[int_360336, R_360337, C_360338], **kwargs_360339)
        
        # Assigning a type to the variable 'col' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'col', reshape_call_result_360340)
        
        # Getting the type of 'col' (line 476)
        col_360341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'col')
        
        # Call to tile(...): (line 476)
        # Processing the call arguments (line 476)
        
        # Call to arange(...): (line 476)
        # Processing the call arguments (line 476)
        # Getting the type of 'C' (line 476)
        C_360346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 33), 'C', False)
        # Processing the call keyword arguments (line 476)
        kwargs_360347 = {}
        # Getting the type of 'np' (line 476)
        np_360344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 23), 'np', False)
        # Obtaining the member 'arange' of a type (line 476)
        arange_360345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 23), np_360344, 'arange')
        # Calling arange(args, kwargs) (line 476)
        arange_call_result_360348 = invoke(stypy.reporting.localization.Localization(__file__, 476, 23), arange_360345, *[C_360346], **kwargs_360347)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 476)
        tuple_360349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 476)
        # Adding element type (line 476)
        # Getting the type of 'R' (line 476)
        R_360350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 38), 'R', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 38), tuple_360349, R_360350)
        # Adding element type (line 476)
        int_360351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 38), tuple_360349, int_360351)
        
        # Processing the call keyword arguments (line 476)
        kwargs_360352 = {}
        # Getting the type of 'np' (line 476)
        np_360342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 15), 'np', False)
        # Obtaining the member 'tile' of a type (line 476)
        tile_360343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 15), np_360342, 'tile')
        # Calling tile(args, kwargs) (line 476)
        tile_call_result_360353 = invoke(stypy.reporting.localization.Localization(__file__, 476, 15), tile_360343, *[arange_call_result_360348, tuple_360349], **kwargs_360352)
        
        # Applying the binary operator '+=' (line 476)
        result_iadd_360354 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 8), '+=', col_360341, tile_call_result_360353)
        # Assigning a type to the variable 'col' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'col', result_iadd_360354)
        
        
        # Assigning a Call to a Name (line 477):
        
        # Assigning a Call to a Name (line 477):
        
        # Call to reshape(...): (line 477)
        # Processing the call arguments (line 477)
        int_360357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 26), 'int')
        # Processing the call keyword arguments (line 477)
        kwargs_360358 = {}
        # Getting the type of 'col' (line 477)
        col_360355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 14), 'col', False)
        # Obtaining the member 'reshape' of a type (line 477)
        reshape_360356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 14), col_360355, 'reshape')
        # Calling reshape(args, kwargs) (line 477)
        reshape_call_result_360359 = invoke(stypy.reporting.localization.Localization(__file__, 477, 14), reshape_360356, *[int_360357], **kwargs_360358)
        
        # Assigning a type to the variable 'col' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'col', reshape_call_result_360359)
        
        # Assigning a Call to a Name (line 479):
        
        # Assigning a Call to a Name (line 479):
        
        # Call to reshape(...): (line 479)
        # Processing the call arguments (line 479)
        int_360363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 33), 'int')
        # Processing the call keyword arguments (line 479)
        kwargs_360364 = {}
        # Getting the type of 'self' (line 479)
        self_360360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 15), 'self', False)
        # Obtaining the member 'data' of a type (line 479)
        data_360361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 15), self_360360, 'data')
        # Obtaining the member 'reshape' of a type (line 479)
        reshape_360362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 15), data_360361, 'reshape')
        # Calling reshape(args, kwargs) (line 479)
        reshape_call_result_360365 = invoke(stypy.reporting.localization.Localization(__file__, 479, 15), reshape_360362, *[int_360363], **kwargs_360364)
        
        # Assigning a type to the variable 'data' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'data', reshape_call_result_360365)
        
        # Getting the type of 'copy' (line 481)
        copy_360366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 11), 'copy')
        # Testing the type of an if condition (line 481)
        if_condition_360367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 481, 8), copy_360366)
        # Assigning a type to the variable 'if_condition_360367' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'if_condition_360367', if_condition_360367)
        # SSA begins for if statement (line 481)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 482):
        
        # Assigning a Call to a Name (line 482):
        
        # Call to copy(...): (line 482)
        # Processing the call keyword arguments (line 482)
        kwargs_360370 = {}
        # Getting the type of 'data' (line 482)
        data_360368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 19), 'data', False)
        # Obtaining the member 'copy' of a type (line 482)
        copy_360369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 19), data_360368, 'copy')
        # Calling copy(args, kwargs) (line 482)
        copy_call_result_360371 = invoke(stypy.reporting.localization.Localization(__file__, 482, 19), copy_360369, *[], **kwargs_360370)
        
        # Assigning a type to the variable 'data' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'data', copy_call_result_360371)
        # SSA join for if statement (line 481)
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 484, 8))
        
        # 'from scipy.sparse.coo import coo_matrix' statement (line 484)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_360372 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 484, 8), 'scipy.sparse.coo')

        if (type(import_360372) is not StypyTypeError):

            if (import_360372 != 'pyd_module'):
                __import__(import_360372)
                sys_modules_360373 = sys.modules[import_360372]
                import_from_module(stypy.reporting.localization.Localization(__file__, 484, 8), 'scipy.sparse.coo', sys_modules_360373.module_type_store, module_type_store, ['coo_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 484, 8), __file__, sys_modules_360373, sys_modules_360373.module_type_store, module_type_store)
            else:
                from scipy.sparse.coo import coo_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 484, 8), 'scipy.sparse.coo', None, module_type_store, ['coo_matrix'], [coo_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.coo' (line 484)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'scipy.sparse.coo', import_360372)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Call to coo_matrix(...): (line 485)
        # Processing the call arguments (line 485)
        
        # Obtaining an instance of the builtin type 'tuple' (line 485)
        tuple_360375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 485)
        # Adding element type (line 485)
        # Getting the type of 'data' (line 485)
        data_360376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 27), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 27), tuple_360375, data_360376)
        # Adding element type (line 485)
        
        # Obtaining an instance of the builtin type 'tuple' (line 485)
        tuple_360377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 485)
        # Adding element type (line 485)
        # Getting the type of 'row' (line 485)
        row_360378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 33), 'row', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 33), tuple_360377, row_360378)
        # Adding element type (line 485)
        # Getting the type of 'col' (line 485)
        col_360379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 37), 'col', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 33), tuple_360377, col_360379)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 27), tuple_360375, tuple_360377)
        
        # Processing the call keyword arguments (line 485)
        # Getting the type of 'self' (line 485)
        self_360380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 50), 'self', False)
        # Obtaining the member 'shape' of a type (line 485)
        shape_360381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 50), self_360380, 'shape')
        keyword_360382 = shape_360381
        kwargs_360383 = {'shape': keyword_360382}
        # Getting the type of 'coo_matrix' (line 485)
        coo_matrix_360374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 15), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 485)
        coo_matrix_call_result_360384 = invoke(stypy.reporting.localization.Localization(__file__, 485, 15), coo_matrix_360374, *[tuple_360375], **kwargs_360383)
        
        # Assigning a type to the variable 'stypy_return_type' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'stypy_return_type', coo_matrix_call_result_360384)
        
        # ################# End of 'tocoo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocoo' in the type store
        # Getting the type of 'stypy_return_type' (line 452)
        stypy_return_type_360385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_360385)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocoo'
        return stypy_return_type_360385


    @norecursion
    def toarray(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 487)
        None_360386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 28), 'None')
        # Getting the type of 'None' (line 487)
        None_360387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 38), 'None')
        defaults = [None_360386, None_360387]
        # Create a new context for function 'toarray'
        module_type_store = module_type_store.open_function_context('toarray', 487, 4, False)
        # Assigning a type to the variable 'self' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.toarray.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.toarray.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.toarray.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.toarray.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.toarray')
        bsr_matrix.toarray.__dict__.__setitem__('stypy_param_names_list', ['order', 'out'])
        bsr_matrix.toarray.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.toarray.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.toarray.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.toarray.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.toarray.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.toarray.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.toarray', ['order', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'toarray', localization, ['order', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'toarray(...)' code ##################

        
        # Call to toarray(...): (line 488)
        # Processing the call keyword arguments (line 488)
        # Getting the type of 'order' (line 488)
        order_360395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 52), 'order', False)
        keyword_360396 = order_360395
        # Getting the type of 'out' (line 488)
        out_360397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 63), 'out', False)
        keyword_360398 = out_360397
        kwargs_360399 = {'order': keyword_360396, 'out': keyword_360398}
        
        # Call to tocoo(...): (line 488)
        # Processing the call keyword arguments (line 488)
        # Getting the type of 'False' (line 488)
        False_360390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 31), 'False', False)
        keyword_360391 = False_360390
        kwargs_360392 = {'copy': keyword_360391}
        # Getting the type of 'self' (line 488)
        self_360388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 15), 'self', False)
        # Obtaining the member 'tocoo' of a type (line 488)
        tocoo_360389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 15), self_360388, 'tocoo')
        # Calling tocoo(args, kwargs) (line 488)
        tocoo_call_result_360393 = invoke(stypy.reporting.localization.Localization(__file__, 488, 15), tocoo_360389, *[], **kwargs_360392)
        
        # Obtaining the member 'toarray' of a type (line 488)
        toarray_360394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 15), tocoo_call_result_360393, 'toarray')
        # Calling toarray(args, kwargs) (line 488)
        toarray_call_result_360400 = invoke(stypy.reporting.localization.Localization(__file__, 488, 15), toarray_360394, *[], **kwargs_360399)
        
        # Assigning a type to the variable 'stypy_return_type' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'stypy_return_type', toarray_call_result_360400)
        
        # ################# End of 'toarray(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'toarray' in the type store
        # Getting the type of 'stypy_return_type' (line 487)
        stypy_return_type_360401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_360401)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'toarray'
        return stypy_return_type_360401

    
    # Assigning a Attribute to a Attribute (line 490):

    @norecursion
    def transpose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 492)
        None_360402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 29), 'None')
        # Getting the type of 'False' (line 492)
        False_360403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 40), 'False')
        defaults = [None_360402, False_360403]
        # Create a new context for function 'transpose'
        module_type_store = module_type_store.open_function_context('transpose', 492, 4, False)
        # Assigning a type to the variable 'self' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.transpose.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.transpose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.transpose.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.transpose.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.transpose')
        bsr_matrix.transpose.__dict__.__setitem__('stypy_param_names_list', ['axes', 'copy'])
        bsr_matrix.transpose.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.transpose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.transpose.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.transpose.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.transpose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.transpose.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.transpose', ['axes', 'copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transpose', localization, ['axes', 'copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transpose(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 493)
        # Getting the type of 'axes' (line 493)
        axes_360404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'axes')
        # Getting the type of 'None' (line 493)
        None_360405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 23), 'None')
        
        (may_be_360406, more_types_in_union_360407) = may_not_be_none(axes_360404, None_360405)

        if may_be_360406:

            if more_types_in_union_360407:
                # Runtime conditional SSA (line 493)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 494)
            # Processing the call arguments (line 494)
            str_360409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 30), 'str', "Sparse matrices do not support an 'axes' parameter because swapping dimensions is the only logical permutation.")
            # Processing the call keyword arguments (line 494)
            kwargs_360410 = {}
            # Getting the type of 'ValueError' (line 494)
            ValueError_360408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 494)
            ValueError_call_result_360411 = invoke(stypy.reporting.localization.Localization(__file__, 494, 18), ValueError_360408, *[str_360409], **kwargs_360410)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 494, 12), ValueError_call_result_360411, 'raise parameter', BaseException)

            if more_types_in_union_360407:
                # SSA join for if statement (line 493)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Tuple (line 498):
        
        # Assigning a Subscript to a Name (line 498):
        
        # Obtaining the type of the subscript
        int_360412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 8), 'int')
        # Getting the type of 'self' (line 498)
        self_360413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 15), 'self')
        # Obtaining the member 'blocksize' of a type (line 498)
        blocksize_360414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 15), self_360413, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 498)
        getitem___360415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), blocksize_360414, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 498)
        subscript_call_result_360416 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), getitem___360415, int_360412)
        
        # Assigning a type to the variable 'tuple_var_assignment_358893' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'tuple_var_assignment_358893', subscript_call_result_360416)
        
        # Assigning a Subscript to a Name (line 498):
        
        # Obtaining the type of the subscript
        int_360417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 8), 'int')
        # Getting the type of 'self' (line 498)
        self_360418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 15), 'self')
        # Obtaining the member 'blocksize' of a type (line 498)
        blocksize_360419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 15), self_360418, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 498)
        getitem___360420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), blocksize_360419, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 498)
        subscript_call_result_360421 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), getitem___360420, int_360417)
        
        # Assigning a type to the variable 'tuple_var_assignment_358894' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'tuple_var_assignment_358894', subscript_call_result_360421)
        
        # Assigning a Name to a Name (line 498):
        # Getting the type of 'tuple_var_assignment_358893' (line 498)
        tuple_var_assignment_358893_360422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'tuple_var_assignment_358893')
        # Assigning a type to the variable 'R' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'R', tuple_var_assignment_358893_360422)
        
        # Assigning a Name to a Name (line 498):
        # Getting the type of 'tuple_var_assignment_358894' (line 498)
        tuple_var_assignment_358894_360423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'tuple_var_assignment_358894')
        # Assigning a type to the variable 'C' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 11), 'C', tuple_var_assignment_358894_360423)
        
        # Assigning a Attribute to a Tuple (line 499):
        
        # Assigning a Subscript to a Name (line 499):
        
        # Obtaining the type of the subscript
        int_360424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 8), 'int')
        # Getting the type of 'self' (line 499)
        self_360425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 15), 'self')
        # Obtaining the member 'shape' of a type (line 499)
        shape_360426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 15), self_360425, 'shape')
        # Obtaining the member '__getitem__' of a type (line 499)
        getitem___360427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 8), shape_360426, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 499)
        subscript_call_result_360428 = invoke(stypy.reporting.localization.Localization(__file__, 499, 8), getitem___360427, int_360424)
        
        # Assigning a type to the variable 'tuple_var_assignment_358895' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'tuple_var_assignment_358895', subscript_call_result_360428)
        
        # Assigning a Subscript to a Name (line 499):
        
        # Obtaining the type of the subscript
        int_360429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 8), 'int')
        # Getting the type of 'self' (line 499)
        self_360430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 15), 'self')
        # Obtaining the member 'shape' of a type (line 499)
        shape_360431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 15), self_360430, 'shape')
        # Obtaining the member '__getitem__' of a type (line 499)
        getitem___360432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 8), shape_360431, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 499)
        subscript_call_result_360433 = invoke(stypy.reporting.localization.Localization(__file__, 499, 8), getitem___360432, int_360429)
        
        # Assigning a type to the variable 'tuple_var_assignment_358896' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'tuple_var_assignment_358896', subscript_call_result_360433)
        
        # Assigning a Name to a Name (line 499):
        # Getting the type of 'tuple_var_assignment_358895' (line 499)
        tuple_var_assignment_358895_360434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'tuple_var_assignment_358895')
        # Assigning a type to the variable 'M' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'M', tuple_var_assignment_358895_360434)
        
        # Assigning a Name to a Name (line 499):
        # Getting the type of 'tuple_var_assignment_358896' (line 499)
        tuple_var_assignment_358896_360435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'tuple_var_assignment_358896')
        # Assigning a type to the variable 'N' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 11), 'N', tuple_var_assignment_358896_360435)
        
        # Assigning a BinOp to a Name (line 500):
        
        # Assigning a BinOp to a Name (line 500):
        # Getting the type of 'self' (line 500)
        self_360436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), 'self')
        # Obtaining the member 'nnz' of a type (line 500)
        nnz_360437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 15), self_360436, 'nnz')
        # Getting the type of 'R' (line 500)
        R_360438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 26), 'R')
        # Getting the type of 'C' (line 500)
        C_360439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 28), 'C')
        # Applying the binary operator '*' (line 500)
        result_mul_360440 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 26), '*', R_360438, C_360439)
        
        # Applying the binary operator '//' (line 500)
        result_floordiv_360441 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 15), '//', nnz_360437, result_mul_360440)
        
        # Assigning a type to the variable 'NBLK' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'NBLK', result_floordiv_360441)
        
        
        # Getting the type of 'self' (line 502)
        self_360442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 11), 'self')
        # Obtaining the member 'nnz' of a type (line 502)
        nnz_360443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 11), self_360442, 'nnz')
        int_360444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 23), 'int')
        # Applying the binary operator '==' (line 502)
        result_eq_360445 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 11), '==', nnz_360443, int_360444)
        
        # Testing the type of an if condition (line 502)
        if_condition_360446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 8), result_eq_360445)
        # Assigning a type to the variable 'if_condition_360446' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'if_condition_360446', if_condition_360446)
        # SSA begins for if statement (line 502)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to bsr_matrix(...): (line 503)
        # Processing the call arguments (line 503)
        
        # Obtaining an instance of the builtin type 'tuple' (line 503)
        tuple_360448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 503)
        # Adding element type (line 503)
        # Getting the type of 'N' (line 503)
        N_360449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 31), 'N', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 31), tuple_360448, N_360449)
        # Adding element type (line 503)
        # Getting the type of 'M' (line 503)
        M_360450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 34), 'M', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 31), tuple_360448, M_360450)
        
        # Processing the call keyword arguments (line 503)
        
        # Obtaining an instance of the builtin type 'tuple' (line 503)
        tuple_360451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 503)
        # Adding element type (line 503)
        # Getting the type of 'C' (line 503)
        C_360452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 49), 'C', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 49), tuple_360451, C_360452)
        # Adding element type (line 503)
        # Getting the type of 'R' (line 503)
        R_360453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 52), 'R', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 49), tuple_360451, R_360453)
        
        keyword_360454 = tuple_360451
        # Getting the type of 'self' (line 504)
        self_360455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 36), 'self', False)
        # Obtaining the member 'dtype' of a type (line 504)
        dtype_360456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 36), self_360455, 'dtype')
        keyword_360457 = dtype_360456
        # Getting the type of 'copy' (line 504)
        copy_360458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 53), 'copy', False)
        keyword_360459 = copy_360458
        kwargs_360460 = {'blocksize': keyword_360454, 'copy': keyword_360459, 'dtype': keyword_360457}
        # Getting the type of 'bsr_matrix' (line 503)
        bsr_matrix_360447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 19), 'bsr_matrix', False)
        # Calling bsr_matrix(args, kwargs) (line 503)
        bsr_matrix_call_result_360461 = invoke(stypy.reporting.localization.Localization(__file__, 503, 19), bsr_matrix_360447, *[tuple_360448], **kwargs_360460)
        
        # Assigning a type to the variable 'stypy_return_type' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'stypy_return_type', bsr_matrix_call_result_360461)
        # SSA join for if statement (line 502)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 506):
        
        # Assigning a Call to a Name (line 506):
        
        # Call to empty(...): (line 506)
        # Processing the call arguments (line 506)
        # Getting the type of 'N' (line 506)
        N_360464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 26), 'N', False)
        # Getting the type of 'C' (line 506)
        C_360465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 29), 'C', False)
        # Applying the binary operator '//' (line 506)
        result_floordiv_360466 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 26), '//', N_360464, C_360465)
        
        int_360467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 33), 'int')
        # Applying the binary operator '+' (line 506)
        result_add_360468 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 26), '+', result_floordiv_360466, int_360467)
        
        # Processing the call keyword arguments (line 506)
        # Getting the type of 'self' (line 506)
        self_360469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 42), 'self', False)
        # Obtaining the member 'indptr' of a type (line 506)
        indptr_360470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 42), self_360469, 'indptr')
        # Obtaining the member 'dtype' of a type (line 506)
        dtype_360471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 42), indptr_360470, 'dtype')
        keyword_360472 = dtype_360471
        kwargs_360473 = {'dtype': keyword_360472}
        # Getting the type of 'np' (line 506)
        np_360462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 17), 'np', False)
        # Obtaining the member 'empty' of a type (line 506)
        empty_360463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 17), np_360462, 'empty')
        # Calling empty(args, kwargs) (line 506)
        empty_call_result_360474 = invoke(stypy.reporting.localization.Localization(__file__, 506, 17), empty_360463, *[result_add_360468], **kwargs_360473)
        
        # Assigning a type to the variable 'indptr' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'indptr', empty_call_result_360474)
        
        # Assigning a Call to a Name (line 507):
        
        # Assigning a Call to a Name (line 507):
        
        # Call to empty(...): (line 507)
        # Processing the call arguments (line 507)
        # Getting the type of 'NBLK' (line 507)
        NBLK_360477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 27), 'NBLK', False)
        # Processing the call keyword arguments (line 507)
        # Getting the type of 'self' (line 507)
        self_360478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 39), 'self', False)
        # Obtaining the member 'indices' of a type (line 507)
        indices_360479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 39), self_360478, 'indices')
        # Obtaining the member 'dtype' of a type (line 507)
        dtype_360480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 39), indices_360479, 'dtype')
        keyword_360481 = dtype_360480
        kwargs_360482 = {'dtype': keyword_360481}
        # Getting the type of 'np' (line 507)
        np_360475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 18), 'np', False)
        # Obtaining the member 'empty' of a type (line 507)
        empty_360476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 18), np_360475, 'empty')
        # Calling empty(args, kwargs) (line 507)
        empty_call_result_360483 = invoke(stypy.reporting.localization.Localization(__file__, 507, 18), empty_360476, *[NBLK_360477], **kwargs_360482)
        
        # Assigning a type to the variable 'indices' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'indices', empty_call_result_360483)
        
        # Assigning a Call to a Name (line 508):
        
        # Assigning a Call to a Name (line 508):
        
        # Call to empty(...): (line 508)
        # Processing the call arguments (line 508)
        
        # Obtaining an instance of the builtin type 'tuple' (line 508)
        tuple_360486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 508)
        # Adding element type (line 508)
        # Getting the type of 'NBLK' (line 508)
        NBLK_360487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 25), 'NBLK', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 25), tuple_360486, NBLK_360487)
        # Adding element type (line 508)
        # Getting the type of 'C' (line 508)
        C_360488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 31), 'C', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 25), tuple_360486, C_360488)
        # Adding element type (line 508)
        # Getting the type of 'R' (line 508)
        R_360489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 34), 'R', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 25), tuple_360486, R_360489)
        
        # Processing the call keyword arguments (line 508)
        # Getting the type of 'self' (line 508)
        self_360490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 44), 'self', False)
        # Obtaining the member 'data' of a type (line 508)
        data_360491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 44), self_360490, 'data')
        # Obtaining the member 'dtype' of a type (line 508)
        dtype_360492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 44), data_360491, 'dtype')
        keyword_360493 = dtype_360492
        kwargs_360494 = {'dtype': keyword_360493}
        # Getting the type of 'np' (line 508)
        np_360484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 15), 'np', False)
        # Obtaining the member 'empty' of a type (line 508)
        empty_360485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 15), np_360484, 'empty')
        # Calling empty(args, kwargs) (line 508)
        empty_call_result_360495 = invoke(stypy.reporting.localization.Localization(__file__, 508, 15), empty_360485, *[tuple_360486], **kwargs_360494)
        
        # Assigning a type to the variable 'data' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'data', empty_call_result_360495)
        
        # Call to bsr_transpose(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'M' (line 510)
        M_360497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 22), 'M', False)
        # Getting the type of 'R' (line 510)
        R_360498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 25), 'R', False)
        # Applying the binary operator '//' (line 510)
        result_floordiv_360499 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 22), '//', M_360497, R_360498)
        
        # Getting the type of 'N' (line 510)
        N_360500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 28), 'N', False)
        # Getting the type of 'C' (line 510)
        C_360501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 31), 'C', False)
        # Applying the binary operator '//' (line 510)
        result_floordiv_360502 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 28), '//', N_360500, C_360501)
        
        # Getting the type of 'R' (line 510)
        R_360503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 34), 'R', False)
        # Getting the type of 'C' (line 510)
        C_360504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 37), 'C', False)
        # Getting the type of 'self' (line 511)
        self_360505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 22), 'self', False)
        # Obtaining the member 'indptr' of a type (line 511)
        indptr_360506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 22), self_360505, 'indptr')
        # Getting the type of 'self' (line 511)
        self_360507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 35), 'self', False)
        # Obtaining the member 'indices' of a type (line 511)
        indices_360508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 35), self_360507, 'indices')
        
        # Call to ravel(...): (line 511)
        # Processing the call keyword arguments (line 511)
        kwargs_360512 = {}
        # Getting the type of 'self' (line 511)
        self_360509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 49), 'self', False)
        # Obtaining the member 'data' of a type (line 511)
        data_360510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 49), self_360509, 'data')
        # Obtaining the member 'ravel' of a type (line 511)
        ravel_360511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 49), data_360510, 'ravel')
        # Calling ravel(args, kwargs) (line 511)
        ravel_call_result_360513 = invoke(stypy.reporting.localization.Localization(__file__, 511, 49), ravel_360511, *[], **kwargs_360512)
        
        # Getting the type of 'indptr' (line 512)
        indptr_360514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 22), 'indptr', False)
        # Getting the type of 'indices' (line 512)
        indices_360515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 30), 'indices', False)
        
        # Call to ravel(...): (line 512)
        # Processing the call keyword arguments (line 512)
        kwargs_360518 = {}
        # Getting the type of 'data' (line 512)
        data_360516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 39), 'data', False)
        # Obtaining the member 'ravel' of a type (line 512)
        ravel_360517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 39), data_360516, 'ravel')
        # Calling ravel(args, kwargs) (line 512)
        ravel_call_result_360519 = invoke(stypy.reporting.localization.Localization(__file__, 512, 39), ravel_360517, *[], **kwargs_360518)
        
        # Processing the call keyword arguments (line 510)
        kwargs_360520 = {}
        # Getting the type of 'bsr_transpose' (line 510)
        bsr_transpose_360496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'bsr_transpose', False)
        # Calling bsr_transpose(args, kwargs) (line 510)
        bsr_transpose_call_result_360521 = invoke(stypy.reporting.localization.Localization(__file__, 510, 8), bsr_transpose_360496, *[result_floordiv_360499, result_floordiv_360502, R_360503, C_360504, indptr_360506, indices_360508, ravel_call_result_360513, indptr_360514, indices_360515, ravel_call_result_360519], **kwargs_360520)
        
        
        # Call to bsr_matrix(...): (line 514)
        # Processing the call arguments (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 514)
        tuple_360523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 514)
        # Adding element type (line 514)
        # Getting the type of 'data' (line 514)
        data_360524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 27), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 27), tuple_360523, data_360524)
        # Adding element type (line 514)
        # Getting the type of 'indices' (line 514)
        indices_360525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 33), 'indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 27), tuple_360523, indices_360525)
        # Adding element type (line 514)
        # Getting the type of 'indptr' (line 514)
        indptr_360526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 42), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 27), tuple_360523, indptr_360526)
        
        # Processing the call keyword arguments (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 515)
        tuple_360527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 515)
        # Adding element type (line 515)
        # Getting the type of 'N' (line 515)
        N_360528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 33), 'N', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 33), tuple_360527, N_360528)
        # Adding element type (line 515)
        # Getting the type of 'M' (line 515)
        M_360529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 36), 'M', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 33), tuple_360527, M_360529)
        
        keyword_360530 = tuple_360527
        # Getting the type of 'copy' (line 515)
        copy_360531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 45), 'copy', False)
        keyword_360532 = copy_360531
        kwargs_360533 = {'shape': keyword_360530, 'copy': keyword_360532}
        # Getting the type of 'bsr_matrix' (line 514)
        bsr_matrix_360522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 15), 'bsr_matrix', False)
        # Calling bsr_matrix(args, kwargs) (line 514)
        bsr_matrix_call_result_360534 = invoke(stypy.reporting.localization.Localization(__file__, 514, 15), bsr_matrix_360522, *[tuple_360523], **kwargs_360533)
        
        # Assigning a type to the variable 'stypy_return_type' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'stypy_return_type', bsr_matrix_call_result_360534)
        
        # ################# End of 'transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 492)
        stypy_return_type_360535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_360535)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transpose'
        return stypy_return_type_360535

    
    # Assigning a Attribute to a Attribute (line 517):

    @norecursion
    def eliminate_zeros(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'eliminate_zeros'
        module_type_store = module_type_store.open_function_context('eliminate_zeros', 523, 4, False)
        # Assigning a type to the variable 'self' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.eliminate_zeros.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.eliminate_zeros.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.eliminate_zeros.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.eliminate_zeros.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.eliminate_zeros')
        bsr_matrix.eliminate_zeros.__dict__.__setitem__('stypy_param_names_list', [])
        bsr_matrix.eliminate_zeros.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.eliminate_zeros.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.eliminate_zeros.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.eliminate_zeros.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.eliminate_zeros.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.eliminate_zeros.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.eliminate_zeros', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'eliminate_zeros', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'eliminate_zeros(...)' code ##################

        str_360536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 8), 'str', 'Remove zero elements in-place.')
        
        # Assigning a Attribute to a Tuple (line 525):
        
        # Assigning a Subscript to a Name (line 525):
        
        # Obtaining the type of the subscript
        int_360537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 8), 'int')
        # Getting the type of 'self' (line 525)
        self_360538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 525)
        blocksize_360539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 14), self_360538, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 525)
        getitem___360540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 8), blocksize_360539, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 525)
        subscript_call_result_360541 = invoke(stypy.reporting.localization.Localization(__file__, 525, 8), getitem___360540, int_360537)
        
        # Assigning a type to the variable 'tuple_var_assignment_358897' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'tuple_var_assignment_358897', subscript_call_result_360541)
        
        # Assigning a Subscript to a Name (line 525):
        
        # Obtaining the type of the subscript
        int_360542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 8), 'int')
        # Getting the type of 'self' (line 525)
        self_360543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 525)
        blocksize_360544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 14), self_360543, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 525)
        getitem___360545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 8), blocksize_360544, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 525)
        subscript_call_result_360546 = invoke(stypy.reporting.localization.Localization(__file__, 525, 8), getitem___360545, int_360542)
        
        # Assigning a type to the variable 'tuple_var_assignment_358898' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'tuple_var_assignment_358898', subscript_call_result_360546)
        
        # Assigning a Name to a Name (line 525):
        # Getting the type of 'tuple_var_assignment_358897' (line 525)
        tuple_var_assignment_358897_360547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'tuple_var_assignment_358897')
        # Assigning a type to the variable 'R' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'R', tuple_var_assignment_358897_360547)
        
        # Assigning a Name to a Name (line 525):
        # Getting the type of 'tuple_var_assignment_358898' (line 525)
        tuple_var_assignment_358898_360548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'tuple_var_assignment_358898')
        # Assigning a type to the variable 'C' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 10), 'C', tuple_var_assignment_358898_360548)
        
        # Assigning a Attribute to a Tuple (line 526):
        
        # Assigning a Subscript to a Name (line 526):
        
        # Obtaining the type of the subscript
        int_360549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 8), 'int')
        # Getting the type of 'self' (line 526)
        self_360550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 14), 'self')
        # Obtaining the member 'shape' of a type (line 526)
        shape_360551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 14), self_360550, 'shape')
        # Obtaining the member '__getitem__' of a type (line 526)
        getitem___360552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 8), shape_360551, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 526)
        subscript_call_result_360553 = invoke(stypy.reporting.localization.Localization(__file__, 526, 8), getitem___360552, int_360549)
        
        # Assigning a type to the variable 'tuple_var_assignment_358899' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'tuple_var_assignment_358899', subscript_call_result_360553)
        
        # Assigning a Subscript to a Name (line 526):
        
        # Obtaining the type of the subscript
        int_360554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 8), 'int')
        # Getting the type of 'self' (line 526)
        self_360555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 14), 'self')
        # Obtaining the member 'shape' of a type (line 526)
        shape_360556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 14), self_360555, 'shape')
        # Obtaining the member '__getitem__' of a type (line 526)
        getitem___360557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 8), shape_360556, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 526)
        subscript_call_result_360558 = invoke(stypy.reporting.localization.Localization(__file__, 526, 8), getitem___360557, int_360554)
        
        # Assigning a type to the variable 'tuple_var_assignment_358900' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'tuple_var_assignment_358900', subscript_call_result_360558)
        
        # Assigning a Name to a Name (line 526):
        # Getting the type of 'tuple_var_assignment_358899' (line 526)
        tuple_var_assignment_358899_360559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'tuple_var_assignment_358899')
        # Assigning a type to the variable 'M' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'M', tuple_var_assignment_358899_360559)
        
        # Assigning a Name to a Name (line 526):
        # Getting the type of 'tuple_var_assignment_358900' (line 526)
        tuple_var_assignment_358900_360560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'tuple_var_assignment_358900')
        # Assigning a type to the variable 'N' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 10), 'N', tuple_var_assignment_358900_360560)
        
        # Assigning a Call to a Name (line 528):
        
        # Assigning a Call to a Name (line 528):
        
        # Call to sum(...): (line 528)
        # Processing the call keyword arguments (line 528)
        int_360573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 57), 'int')
        keyword_360574 = int_360573
        kwargs_360575 = {'axis': keyword_360574}
        
        # Call to reshape(...): (line 528)
        # Processing the call arguments (line 528)
        int_360566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 40), 'int')
        # Getting the type of 'R' (line 528)
        R_360567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 43), 'R', False)
        # Getting the type of 'C' (line 528)
        C_360568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 45), 'C', False)
        # Applying the binary operator '*' (line 528)
        result_mul_360569 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 43), '*', R_360567, C_360568)
        
        # Processing the call keyword arguments (line 528)
        kwargs_360570 = {}
        
        # Getting the type of 'self' (line 528)
        self_360561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 16), 'self', False)
        # Obtaining the member 'data' of a type (line 528)
        data_360562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 16), self_360561, 'data')
        int_360563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 29), 'int')
        # Applying the binary operator '!=' (line 528)
        result_ne_360564 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 16), '!=', data_360562, int_360563)
        
        # Obtaining the member 'reshape' of a type (line 528)
        reshape_360565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 16), result_ne_360564, 'reshape')
        # Calling reshape(args, kwargs) (line 528)
        reshape_call_result_360571 = invoke(stypy.reporting.localization.Localization(__file__, 528, 16), reshape_360565, *[int_360566, result_mul_360569], **kwargs_360570)
        
        # Obtaining the member 'sum' of a type (line 528)
        sum_360572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 16), reshape_call_result_360571, 'sum')
        # Calling sum(args, kwargs) (line 528)
        sum_call_result_360576 = invoke(stypy.reporting.localization.Localization(__file__, 528, 16), sum_360572, *[], **kwargs_360575)
        
        # Assigning a type to the variable 'mask' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'mask', sum_call_result_360576)
        
        # Assigning a Subscript to a Name (line 530):
        
        # Assigning a Subscript to a Name (line 530):
        
        # Obtaining the type of the subscript
        int_360577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 40), 'int')
        
        # Call to nonzero(...): (line 530)
        # Processing the call keyword arguments (line 530)
        kwargs_360580 = {}
        # Getting the type of 'mask' (line 530)
        mask_360578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 25), 'mask', False)
        # Obtaining the member 'nonzero' of a type (line 530)
        nonzero_360579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 25), mask_360578, 'nonzero')
        # Calling nonzero(args, kwargs) (line 530)
        nonzero_call_result_360581 = invoke(stypy.reporting.localization.Localization(__file__, 530, 25), nonzero_360579, *[], **kwargs_360580)
        
        # Obtaining the member '__getitem__' of a type (line 530)
        getitem___360582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 25), nonzero_call_result_360581, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 530)
        subscript_call_result_360583 = invoke(stypy.reporting.localization.Localization(__file__, 530, 25), getitem___360582, int_360577)
        
        # Assigning a type to the variable 'nonzero_blocks' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'nonzero_blocks', subscript_call_result_360583)
        
        
        
        # Call to len(...): (line 532)
        # Processing the call arguments (line 532)
        # Getting the type of 'nonzero_blocks' (line 532)
        nonzero_blocks_360585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 15), 'nonzero_blocks', False)
        # Processing the call keyword arguments (line 532)
        kwargs_360586 = {}
        # Getting the type of 'len' (line 532)
        len_360584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 11), 'len', False)
        # Calling len(args, kwargs) (line 532)
        len_call_result_360587 = invoke(stypy.reporting.localization.Localization(__file__, 532, 11), len_360584, *[nonzero_blocks_360585], **kwargs_360586)
        
        int_360588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 34), 'int')
        # Applying the binary operator '==' (line 532)
        result_eq_360589 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 11), '==', len_call_result_360587, int_360588)
        
        # Testing the type of an if condition (line 532)
        if_condition_360590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 532, 8), result_eq_360589)
        # Assigning a type to the variable 'if_condition_360590' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'if_condition_360590', if_condition_360590)
        # SSA begins for if statement (line 532)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 532)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Subscript (line 535):
        
        # Assigning a Subscript to a Subscript (line 535):
        
        # Obtaining the type of the subscript
        # Getting the type of 'nonzero_blocks' (line 535)
        nonzero_blocks_360591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 52), 'nonzero_blocks')
        # Getting the type of 'self' (line 535)
        self_360592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 42), 'self')
        # Obtaining the member 'data' of a type (line 535)
        data_360593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 42), self_360592, 'data')
        # Obtaining the member '__getitem__' of a type (line 535)
        getitem___360594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 42), data_360593, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 535)
        subscript_call_result_360595 = invoke(stypy.reporting.localization.Localization(__file__, 535, 42), getitem___360594, nonzero_blocks_360591)
        
        # Getting the type of 'self' (line 535)
        self_360596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'self')
        # Obtaining the member 'data' of a type (line 535)
        data_360597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 8), self_360596, 'data')
        
        # Call to len(...): (line 535)
        # Processing the call arguments (line 535)
        # Getting the type of 'nonzero_blocks' (line 535)
        nonzero_blocks_360599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 23), 'nonzero_blocks', False)
        # Processing the call keyword arguments (line 535)
        kwargs_360600 = {}
        # Getting the type of 'len' (line 535)
        len_360598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 19), 'len', False)
        # Calling len(args, kwargs) (line 535)
        len_call_result_360601 = invoke(stypy.reporting.localization.Localization(__file__, 535, 19), len_360598, *[nonzero_blocks_360599], **kwargs_360600)
        
        slice_360602 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 535, 8), None, len_call_result_360601, None)
        # Storing an element on a container (line 535)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 8), data_360597, (slice_360602, subscript_call_result_360595))
        
        # Call to csr_eliminate_zeros(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'M' (line 538)
        M_360605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 41), 'M', False)
        # Getting the type of 'R' (line 538)
        R_360606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 44), 'R', False)
        # Applying the binary operator '//' (line 538)
        result_floordiv_360607 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 41), '//', M_360605, R_360606)
        
        # Getting the type of 'N' (line 538)
        N_360608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 47), 'N', False)
        # Getting the type of 'C' (line 538)
        C_360609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 50), 'C', False)
        # Applying the binary operator '//' (line 538)
        result_floordiv_360610 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 47), '//', N_360608, C_360609)
        
        # Getting the type of 'self' (line 538)
        self_360611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 53), 'self', False)
        # Obtaining the member 'indptr' of a type (line 538)
        indptr_360612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 53), self_360611, 'indptr')
        # Getting the type of 'self' (line 539)
        self_360613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 41), 'self', False)
        # Obtaining the member 'indices' of a type (line 539)
        indices_360614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 41), self_360613, 'indices')
        # Getting the type of 'mask' (line 539)
        mask_360615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 55), 'mask', False)
        # Processing the call keyword arguments (line 538)
        kwargs_360616 = {}
        # Getting the type of '_sparsetools' (line 538)
        _sparsetools_360603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), '_sparsetools', False)
        # Obtaining the member 'csr_eliminate_zeros' of a type (line 538)
        csr_eliminate_zeros_360604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 8), _sparsetools_360603, 'csr_eliminate_zeros')
        # Calling csr_eliminate_zeros(args, kwargs) (line 538)
        csr_eliminate_zeros_call_result_360617 = invoke(stypy.reporting.localization.Localization(__file__, 538, 8), csr_eliminate_zeros_360604, *[result_floordiv_360607, result_floordiv_360610, indptr_360612, indices_360614, mask_360615], **kwargs_360616)
        
        
        # Call to prune(...): (line 540)
        # Processing the call keyword arguments (line 540)
        kwargs_360620 = {}
        # Getting the type of 'self' (line 540)
        self_360618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'self', False)
        # Obtaining the member 'prune' of a type (line 540)
        prune_360619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 8), self_360618, 'prune')
        # Calling prune(args, kwargs) (line 540)
        prune_call_result_360621 = invoke(stypy.reporting.localization.Localization(__file__, 540, 8), prune_360619, *[], **kwargs_360620)
        
        
        # ################# End of 'eliminate_zeros(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'eliminate_zeros' in the type store
        # Getting the type of 'stypy_return_type' (line 523)
        stypy_return_type_360622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_360622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'eliminate_zeros'
        return stypy_return_type_360622


    @norecursion
    def sum_duplicates(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sum_duplicates'
        module_type_store = module_type_store.open_function_context('sum_duplicates', 542, 4, False)
        # Assigning a type to the variable 'self' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.sum_duplicates.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.sum_duplicates.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.sum_duplicates.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.sum_duplicates.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.sum_duplicates')
        bsr_matrix.sum_duplicates.__dict__.__setitem__('stypy_param_names_list', [])
        bsr_matrix.sum_duplicates.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.sum_duplicates.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.sum_duplicates.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.sum_duplicates.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.sum_duplicates.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.sum_duplicates.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.sum_duplicates', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sum_duplicates', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sum_duplicates(...)' code ##################

        str_360623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, (-1)), 'str', 'Eliminate duplicate matrix entries by adding them together\n\n        The is an *in place* operation\n        ')
        
        # Getting the type of 'self' (line 547)
        self_360624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 11), 'self')
        # Obtaining the member 'has_canonical_format' of a type (line 547)
        has_canonical_format_360625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 11), self_360624, 'has_canonical_format')
        # Testing the type of an if condition (line 547)
        if_condition_360626 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 547, 8), has_canonical_format_360625)
        # Assigning a type to the variable 'if_condition_360626' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'if_condition_360626', if_condition_360626)
        # SSA begins for if statement (line 547)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 547)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to sort_indices(...): (line 549)
        # Processing the call keyword arguments (line 549)
        kwargs_360629 = {}
        # Getting the type of 'self' (line 549)
        self_360627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'self', False)
        # Obtaining the member 'sort_indices' of a type (line 549)
        sort_indices_360628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 8), self_360627, 'sort_indices')
        # Calling sort_indices(args, kwargs) (line 549)
        sort_indices_call_result_360630 = invoke(stypy.reporting.localization.Localization(__file__, 549, 8), sort_indices_360628, *[], **kwargs_360629)
        
        
        # Assigning a Attribute to a Tuple (line 550):
        
        # Assigning a Subscript to a Name (line 550):
        
        # Obtaining the type of the subscript
        int_360631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 8), 'int')
        # Getting the type of 'self' (line 550)
        self_360632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 15), 'self')
        # Obtaining the member 'blocksize' of a type (line 550)
        blocksize_360633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 15), self_360632, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 550)
        getitem___360634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 8), blocksize_360633, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 550)
        subscript_call_result_360635 = invoke(stypy.reporting.localization.Localization(__file__, 550, 8), getitem___360634, int_360631)
        
        # Assigning a type to the variable 'tuple_var_assignment_358901' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'tuple_var_assignment_358901', subscript_call_result_360635)
        
        # Assigning a Subscript to a Name (line 550):
        
        # Obtaining the type of the subscript
        int_360636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 8), 'int')
        # Getting the type of 'self' (line 550)
        self_360637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 15), 'self')
        # Obtaining the member 'blocksize' of a type (line 550)
        blocksize_360638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 15), self_360637, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 550)
        getitem___360639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 8), blocksize_360638, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 550)
        subscript_call_result_360640 = invoke(stypy.reporting.localization.Localization(__file__, 550, 8), getitem___360639, int_360636)
        
        # Assigning a type to the variable 'tuple_var_assignment_358902' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'tuple_var_assignment_358902', subscript_call_result_360640)
        
        # Assigning a Name to a Name (line 550):
        # Getting the type of 'tuple_var_assignment_358901' (line 550)
        tuple_var_assignment_358901_360641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'tuple_var_assignment_358901')
        # Assigning a type to the variable 'R' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'R', tuple_var_assignment_358901_360641)
        
        # Assigning a Name to a Name (line 550):
        # Getting the type of 'tuple_var_assignment_358902' (line 550)
        tuple_var_assignment_358902_360642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'tuple_var_assignment_358902')
        # Assigning a type to the variable 'C' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 11), 'C', tuple_var_assignment_358902_360642)
        
        # Assigning a Attribute to a Tuple (line 551):
        
        # Assigning a Subscript to a Name (line 551):
        
        # Obtaining the type of the subscript
        int_360643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 8), 'int')
        # Getting the type of 'self' (line 551)
        self_360644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 15), 'self')
        # Obtaining the member 'shape' of a type (line 551)
        shape_360645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 15), self_360644, 'shape')
        # Obtaining the member '__getitem__' of a type (line 551)
        getitem___360646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 8), shape_360645, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 551)
        subscript_call_result_360647 = invoke(stypy.reporting.localization.Localization(__file__, 551, 8), getitem___360646, int_360643)
        
        # Assigning a type to the variable 'tuple_var_assignment_358903' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'tuple_var_assignment_358903', subscript_call_result_360647)
        
        # Assigning a Subscript to a Name (line 551):
        
        # Obtaining the type of the subscript
        int_360648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 8), 'int')
        # Getting the type of 'self' (line 551)
        self_360649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 15), 'self')
        # Obtaining the member 'shape' of a type (line 551)
        shape_360650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 15), self_360649, 'shape')
        # Obtaining the member '__getitem__' of a type (line 551)
        getitem___360651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 8), shape_360650, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 551)
        subscript_call_result_360652 = invoke(stypy.reporting.localization.Localization(__file__, 551, 8), getitem___360651, int_360648)
        
        # Assigning a type to the variable 'tuple_var_assignment_358904' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'tuple_var_assignment_358904', subscript_call_result_360652)
        
        # Assigning a Name to a Name (line 551):
        # Getting the type of 'tuple_var_assignment_358903' (line 551)
        tuple_var_assignment_358903_360653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'tuple_var_assignment_358903')
        # Assigning a type to the variable 'M' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'M', tuple_var_assignment_358903_360653)
        
        # Assigning a Name to a Name (line 551):
        # Getting the type of 'tuple_var_assignment_358904' (line 551)
        tuple_var_assignment_358904_360654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'tuple_var_assignment_358904')
        # Assigning a type to the variable 'N' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 11), 'N', tuple_var_assignment_358904_360654)
        
        # Assigning a BinOp to a Name (line 554):
        
        # Assigning a BinOp to a Name (line 554):
        # Getting the type of 'M' (line 554)
        M_360655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'M')
        # Getting the type of 'R' (line 554)
        R_360656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 21), 'R')
        # Applying the binary operator '//' (line 554)
        result_floordiv_360657 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 16), '//', M_360655, R_360656)
        
        # Assigning a type to the variable 'n_row' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'n_row', result_floordiv_360657)
        
        # Assigning a Num to a Name (line 555):
        
        # Assigning a Num to a Name (line 555):
        int_360658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 14), 'int')
        # Assigning a type to the variable 'nnz' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'nnz', int_360658)
        
        # Assigning a Num to a Name (line 556):
        
        # Assigning a Num to a Name (line 556):
        int_360659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 18), 'int')
        # Assigning a type to the variable 'row_end' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'row_end', int_360659)
        
        
        # Call to range(...): (line 557)
        # Processing the call arguments (line 557)
        # Getting the type of 'n_row' (line 557)
        n_row_360661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 23), 'n_row', False)
        # Processing the call keyword arguments (line 557)
        kwargs_360662 = {}
        # Getting the type of 'range' (line 557)
        range_360660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 17), 'range', False)
        # Calling range(args, kwargs) (line 557)
        range_call_result_360663 = invoke(stypy.reporting.localization.Localization(__file__, 557, 17), range_360660, *[n_row_360661], **kwargs_360662)
        
        # Testing the type of a for loop iterable (line 557)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 557, 8), range_call_result_360663)
        # Getting the type of the for loop variable (line 557)
        for_loop_var_360664 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 557, 8), range_call_result_360663)
        # Assigning a type to the variable 'i' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'i', for_loop_var_360664)
        # SSA begins for a for statement (line 557)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Name (line 558):
        
        # Assigning a Name to a Name (line 558):
        # Getting the type of 'row_end' (line 558)
        row_end_360665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 17), 'row_end')
        # Assigning a type to the variable 'jj' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'jj', row_end_360665)
        
        # Assigning a Subscript to a Name (line 559):
        
        # Assigning a Subscript to a Name (line 559):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 559)
        i_360666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 34), 'i')
        int_360667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 36), 'int')
        # Applying the binary operator '+' (line 559)
        result_add_360668 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 34), '+', i_360666, int_360667)
        
        # Getting the type of 'self' (line 559)
        self_360669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 22), 'self')
        # Obtaining the member 'indptr' of a type (line 559)
        indptr_360670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 22), self_360669, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 559)
        getitem___360671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 22), indptr_360670, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 559)
        subscript_call_result_360672 = invoke(stypy.reporting.localization.Localization(__file__, 559, 22), getitem___360671, result_add_360668)
        
        # Assigning a type to the variable 'row_end' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 12), 'row_end', subscript_call_result_360672)
        
        
        # Getting the type of 'jj' (line 560)
        jj_360673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 18), 'jj')
        # Getting the type of 'row_end' (line 560)
        row_end_360674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 23), 'row_end')
        # Applying the binary operator '<' (line 560)
        result_lt_360675 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 18), '<', jj_360673, row_end_360674)
        
        # Testing the type of an if condition (line 560)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 560, 12), result_lt_360675)
        # SSA begins for while statement (line 560)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Subscript to a Name (line 561):
        
        # Assigning a Subscript to a Name (line 561):
        
        # Obtaining the type of the subscript
        # Getting the type of 'jj' (line 561)
        jj_360676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 33), 'jj')
        # Getting the type of 'self' (line 561)
        self_360677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 20), 'self')
        # Obtaining the member 'indices' of a type (line 561)
        indices_360678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 20), self_360677, 'indices')
        # Obtaining the member '__getitem__' of a type (line 561)
        getitem___360679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 20), indices_360678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 561)
        subscript_call_result_360680 = invoke(stypy.reporting.localization.Localization(__file__, 561, 20), getitem___360679, jj_360676)
        
        # Assigning a type to the variable 'j' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), 'j', subscript_call_result_360680)
        
        # Assigning a Subscript to a Name (line 562):
        
        # Assigning a Subscript to a Name (line 562):
        
        # Obtaining the type of the subscript
        # Getting the type of 'jj' (line 562)
        jj_360681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 30), 'jj')
        # Getting the type of 'self' (line 562)
        self_360682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'self')
        # Obtaining the member 'data' of a type (line 562)
        data_360683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 20), self_360682, 'data')
        # Obtaining the member '__getitem__' of a type (line 562)
        getitem___360684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 20), data_360683, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 562)
        subscript_call_result_360685 = invoke(stypy.reporting.localization.Localization(__file__, 562, 20), getitem___360684, jj_360681)
        
        # Assigning a type to the variable 'x' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 16), 'x', subscript_call_result_360685)
        
        # Getting the type of 'jj' (line 563)
        jj_360686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 16), 'jj')
        int_360687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 22), 'int')
        # Applying the binary operator '+=' (line 563)
        result_iadd_360688 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 16), '+=', jj_360686, int_360687)
        # Assigning a type to the variable 'jj' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 16), 'jj', result_iadd_360688)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'jj' (line 564)
        jj_360689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 22), 'jj')
        # Getting the type of 'row_end' (line 564)
        row_end_360690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 27), 'row_end')
        # Applying the binary operator '<' (line 564)
        result_lt_360691 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 22), '<', jj_360689, row_end_360690)
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'jj' (line 564)
        jj_360692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 52), 'jj')
        # Getting the type of 'self' (line 564)
        self_360693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 39), 'self')
        # Obtaining the member 'indices' of a type (line 564)
        indices_360694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 39), self_360693, 'indices')
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___360695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 39), indices_360694, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 564)
        subscript_call_result_360696 = invoke(stypy.reporting.localization.Localization(__file__, 564, 39), getitem___360695, jj_360692)
        
        # Getting the type of 'j' (line 564)
        j_360697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 59), 'j')
        # Applying the binary operator '==' (line 564)
        result_eq_360698 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 39), '==', subscript_call_result_360696, j_360697)
        
        # Applying the binary operator 'and' (line 564)
        result_and_keyword_360699 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 22), 'and', result_lt_360691, result_eq_360698)
        
        # Testing the type of an if condition (line 564)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 564, 16), result_and_keyword_360699)
        # SSA begins for while statement (line 564)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'x' (line 565)
        x_360700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 20), 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'jj' (line 565)
        jj_360701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 35), 'jj')
        # Getting the type of 'self' (line 565)
        self_360702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 25), 'self')
        # Obtaining the member 'data' of a type (line 565)
        data_360703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 25), self_360702, 'data')
        # Obtaining the member '__getitem__' of a type (line 565)
        getitem___360704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 25), data_360703, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 565)
        subscript_call_result_360705 = invoke(stypy.reporting.localization.Localization(__file__, 565, 25), getitem___360704, jj_360701)
        
        # Applying the binary operator '+=' (line 565)
        result_iadd_360706 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 20), '+=', x_360700, subscript_call_result_360705)
        # Assigning a type to the variable 'x' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 20), 'x', result_iadd_360706)
        
        
        # Getting the type of 'jj' (line 566)
        jj_360707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 20), 'jj')
        int_360708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 26), 'int')
        # Applying the binary operator '+=' (line 566)
        result_iadd_360709 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 20), '+=', jj_360707, int_360708)
        # Assigning a type to the variable 'jj' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 20), 'jj', result_iadd_360709)
        
        # SSA join for while statement (line 564)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 567):
        
        # Assigning a Name to a Subscript (line 567):
        # Getting the type of 'j' (line 567)
        j_360710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 36), 'j')
        # Getting the type of 'self' (line 567)
        self_360711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 16), 'self')
        # Obtaining the member 'indices' of a type (line 567)
        indices_360712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 16), self_360711, 'indices')
        # Getting the type of 'nnz' (line 567)
        nnz_360713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 29), 'nnz')
        # Storing an element on a container (line 567)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 16), indices_360712, (nnz_360713, j_360710))
        
        # Assigning a Name to a Subscript (line 568):
        
        # Assigning a Name to a Subscript (line 568):
        # Getting the type of 'x' (line 568)
        x_360714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 33), 'x')
        # Getting the type of 'self' (line 568)
        self_360715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 16), 'self')
        # Obtaining the member 'data' of a type (line 568)
        data_360716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 16), self_360715, 'data')
        # Getting the type of 'nnz' (line 568)
        nnz_360717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 26), 'nnz')
        # Storing an element on a container (line 568)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 16), data_360716, (nnz_360717, x_360714))
        
        # Getting the type of 'nnz' (line 569)
        nnz_360718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 16), 'nnz')
        int_360719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 23), 'int')
        # Applying the binary operator '+=' (line 569)
        result_iadd_360720 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 16), '+=', nnz_360718, int_360719)
        # Assigning a type to the variable 'nnz' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 16), 'nnz', result_iadd_360720)
        
        # SSA join for while statement (line 560)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 570):
        
        # Assigning a Name to a Subscript (line 570):
        # Getting the type of 'nnz' (line 570)
        nnz_360721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 31), 'nnz')
        # Getting the type of 'self' (line 570)
        self_360722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'self')
        # Obtaining the member 'indptr' of a type (line 570)
        indptr_360723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 12), self_360722, 'indptr')
        # Getting the type of 'i' (line 570)
        i_360724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 24), 'i')
        int_360725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 26), 'int')
        # Applying the binary operator '+' (line 570)
        result_add_360726 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 24), '+', i_360724, int_360725)
        
        # Storing an element on a container (line 570)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 12), indptr_360723, (result_add_360726, nnz_360721))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to prune(...): (line 572)
        # Processing the call keyword arguments (line 572)
        kwargs_360729 = {}
        # Getting the type of 'self' (line 572)
        self_360727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'self', False)
        # Obtaining the member 'prune' of a type (line 572)
        prune_360728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 8), self_360727, 'prune')
        # Calling prune(args, kwargs) (line 572)
        prune_call_result_360730 = invoke(stypy.reporting.localization.Localization(__file__, 572, 8), prune_360728, *[], **kwargs_360729)
        
        
        # Assigning a Name to a Attribute (line 573):
        
        # Assigning a Name to a Attribute (line 573):
        # Getting the type of 'True' (line 573)
        True_360731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 36), 'True')
        # Getting the type of 'self' (line 573)
        self_360732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'self')
        # Setting the type of the member 'has_canonical_format' of a type (line 573)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 8), self_360732, 'has_canonical_format', True_360731)
        
        # ################# End of 'sum_duplicates(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sum_duplicates' in the type store
        # Getting the type of 'stypy_return_type' (line 542)
        stypy_return_type_360733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_360733)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sum_duplicates'
        return stypy_return_type_360733


    @norecursion
    def sort_indices(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sort_indices'
        module_type_store = module_type_store.open_function_context('sort_indices', 575, 4, False)
        # Assigning a type to the variable 'self' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.sort_indices.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.sort_indices.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.sort_indices.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.sort_indices.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.sort_indices')
        bsr_matrix.sort_indices.__dict__.__setitem__('stypy_param_names_list', [])
        bsr_matrix.sort_indices.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.sort_indices.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.sort_indices.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.sort_indices.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.sort_indices.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.sort_indices.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.sort_indices', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sort_indices', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sort_indices(...)' code ##################

        str_360734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, (-1)), 'str', 'Sort the indices of this matrix *in place*\n        ')
        
        # Getting the type of 'self' (line 578)
        self_360735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 11), 'self')
        # Obtaining the member 'has_sorted_indices' of a type (line 578)
        has_sorted_indices_360736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 11), self_360735, 'has_sorted_indices')
        # Testing the type of an if condition (line 578)
        if_condition_360737 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 578, 8), has_sorted_indices_360736)
        # Assigning a type to the variable 'if_condition_360737' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'if_condition_360737', if_condition_360737)
        # SSA begins for if statement (line 578)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 578)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Tuple (line 581):
        
        # Assigning a Subscript to a Name (line 581):
        
        # Obtaining the type of the subscript
        int_360738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 8), 'int')
        # Getting the type of 'self' (line 581)
        self_360739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 581)
        blocksize_360740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 14), self_360739, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 581)
        getitem___360741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 8), blocksize_360740, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 581)
        subscript_call_result_360742 = invoke(stypy.reporting.localization.Localization(__file__, 581, 8), getitem___360741, int_360738)
        
        # Assigning a type to the variable 'tuple_var_assignment_358905' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_358905', subscript_call_result_360742)
        
        # Assigning a Subscript to a Name (line 581):
        
        # Obtaining the type of the subscript
        int_360743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 8), 'int')
        # Getting the type of 'self' (line 581)
        self_360744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 581)
        blocksize_360745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 14), self_360744, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 581)
        getitem___360746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 8), blocksize_360745, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 581)
        subscript_call_result_360747 = invoke(stypy.reporting.localization.Localization(__file__, 581, 8), getitem___360746, int_360743)
        
        # Assigning a type to the variable 'tuple_var_assignment_358906' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_358906', subscript_call_result_360747)
        
        # Assigning a Name to a Name (line 581):
        # Getting the type of 'tuple_var_assignment_358905' (line 581)
        tuple_var_assignment_358905_360748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_358905')
        # Assigning a type to the variable 'R' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'R', tuple_var_assignment_358905_360748)
        
        # Assigning a Name to a Name (line 581):
        # Getting the type of 'tuple_var_assignment_358906' (line 581)
        tuple_var_assignment_358906_360749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_358906')
        # Assigning a type to the variable 'C' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 10), 'C', tuple_var_assignment_358906_360749)
        
        # Assigning a Attribute to a Tuple (line 582):
        
        # Assigning a Subscript to a Name (line 582):
        
        # Obtaining the type of the subscript
        int_360750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 8), 'int')
        # Getting the type of 'self' (line 582)
        self_360751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 14), 'self')
        # Obtaining the member 'shape' of a type (line 582)
        shape_360752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 14), self_360751, 'shape')
        # Obtaining the member '__getitem__' of a type (line 582)
        getitem___360753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 8), shape_360752, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 582)
        subscript_call_result_360754 = invoke(stypy.reporting.localization.Localization(__file__, 582, 8), getitem___360753, int_360750)
        
        # Assigning a type to the variable 'tuple_var_assignment_358907' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'tuple_var_assignment_358907', subscript_call_result_360754)
        
        # Assigning a Subscript to a Name (line 582):
        
        # Obtaining the type of the subscript
        int_360755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 8), 'int')
        # Getting the type of 'self' (line 582)
        self_360756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 14), 'self')
        # Obtaining the member 'shape' of a type (line 582)
        shape_360757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 14), self_360756, 'shape')
        # Obtaining the member '__getitem__' of a type (line 582)
        getitem___360758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 8), shape_360757, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 582)
        subscript_call_result_360759 = invoke(stypy.reporting.localization.Localization(__file__, 582, 8), getitem___360758, int_360755)
        
        # Assigning a type to the variable 'tuple_var_assignment_358908' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'tuple_var_assignment_358908', subscript_call_result_360759)
        
        # Assigning a Name to a Name (line 582):
        # Getting the type of 'tuple_var_assignment_358907' (line 582)
        tuple_var_assignment_358907_360760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'tuple_var_assignment_358907')
        # Assigning a type to the variable 'M' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'M', tuple_var_assignment_358907_360760)
        
        # Assigning a Name to a Name (line 582):
        # Getting the type of 'tuple_var_assignment_358908' (line 582)
        tuple_var_assignment_358908_360761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'tuple_var_assignment_358908')
        # Assigning a type to the variable 'N' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 10), 'N', tuple_var_assignment_358908_360761)
        
        # Call to bsr_sort_indices(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 'M' (line 584)
        M_360763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 25), 'M', False)
        # Getting the type of 'R' (line 584)
        R_360764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 28), 'R', False)
        # Applying the binary operator '//' (line 584)
        result_floordiv_360765 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 25), '//', M_360763, R_360764)
        
        # Getting the type of 'N' (line 584)
        N_360766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 31), 'N', False)
        # Getting the type of 'C' (line 584)
        C_360767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 34), 'C', False)
        # Applying the binary operator '//' (line 584)
        result_floordiv_360768 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 31), '//', N_360766, C_360767)
        
        # Getting the type of 'R' (line 584)
        R_360769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 37), 'R', False)
        # Getting the type of 'C' (line 584)
        C_360770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 40), 'C', False)
        # Getting the type of 'self' (line 584)
        self_360771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 43), 'self', False)
        # Obtaining the member 'indptr' of a type (line 584)
        indptr_360772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 43), self_360771, 'indptr')
        # Getting the type of 'self' (line 584)
        self_360773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 56), 'self', False)
        # Obtaining the member 'indices' of a type (line 584)
        indices_360774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 56), self_360773, 'indices')
        
        # Call to ravel(...): (line 584)
        # Processing the call keyword arguments (line 584)
        kwargs_360778 = {}
        # Getting the type of 'self' (line 584)
        self_360775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 70), 'self', False)
        # Obtaining the member 'data' of a type (line 584)
        data_360776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 70), self_360775, 'data')
        # Obtaining the member 'ravel' of a type (line 584)
        ravel_360777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 70), data_360776, 'ravel')
        # Calling ravel(args, kwargs) (line 584)
        ravel_call_result_360779 = invoke(stypy.reporting.localization.Localization(__file__, 584, 70), ravel_360777, *[], **kwargs_360778)
        
        # Processing the call keyword arguments (line 584)
        kwargs_360780 = {}
        # Getting the type of 'bsr_sort_indices' (line 584)
        bsr_sort_indices_360762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'bsr_sort_indices', False)
        # Calling bsr_sort_indices(args, kwargs) (line 584)
        bsr_sort_indices_call_result_360781 = invoke(stypy.reporting.localization.Localization(__file__, 584, 8), bsr_sort_indices_360762, *[result_floordiv_360765, result_floordiv_360768, R_360769, C_360770, indptr_360772, indices_360774, ravel_call_result_360779], **kwargs_360780)
        
        
        # Assigning a Name to a Attribute (line 586):
        
        # Assigning a Name to a Attribute (line 586):
        # Getting the type of 'True' (line 586)
        True_360782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 34), 'True')
        # Getting the type of 'self' (line 586)
        self_360783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'self')
        # Setting the type of the member 'has_sorted_indices' of a type (line 586)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 8), self_360783, 'has_sorted_indices', True_360782)
        
        # ################# End of 'sort_indices(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sort_indices' in the type store
        # Getting the type of 'stypy_return_type' (line 575)
        stypy_return_type_360784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_360784)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sort_indices'
        return stypy_return_type_360784


    @norecursion
    def prune(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'prune'
        module_type_store = module_type_store.open_function_context('prune', 588, 4, False)
        # Assigning a type to the variable 'self' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix.prune.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix.prune.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix.prune.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix.prune.__dict__.__setitem__('stypy_function_name', 'bsr_matrix.prune')
        bsr_matrix.prune.__dict__.__setitem__('stypy_param_names_list', [])
        bsr_matrix.prune.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix.prune.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix.prune.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix.prune.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix.prune.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix.prune.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix.prune', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'prune', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'prune(...)' code ##################

        str_360785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, (-1)), 'str', ' Remove empty space after all non-zero elements.\n        ')
        
        # Assigning a Attribute to a Tuple (line 592):
        
        # Assigning a Subscript to a Name (line 592):
        
        # Obtaining the type of the subscript
        int_360786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 8), 'int')
        # Getting the type of 'self' (line 592)
        self_360787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 592)
        blocksize_360788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 14), self_360787, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 592)
        getitem___360789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 8), blocksize_360788, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 592)
        subscript_call_result_360790 = invoke(stypy.reporting.localization.Localization(__file__, 592, 8), getitem___360789, int_360786)
        
        # Assigning a type to the variable 'tuple_var_assignment_358909' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'tuple_var_assignment_358909', subscript_call_result_360790)
        
        # Assigning a Subscript to a Name (line 592):
        
        # Obtaining the type of the subscript
        int_360791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 8), 'int')
        # Getting the type of 'self' (line 592)
        self_360792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 592)
        blocksize_360793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 14), self_360792, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 592)
        getitem___360794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 8), blocksize_360793, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 592)
        subscript_call_result_360795 = invoke(stypy.reporting.localization.Localization(__file__, 592, 8), getitem___360794, int_360791)
        
        # Assigning a type to the variable 'tuple_var_assignment_358910' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'tuple_var_assignment_358910', subscript_call_result_360795)
        
        # Assigning a Name to a Name (line 592):
        # Getting the type of 'tuple_var_assignment_358909' (line 592)
        tuple_var_assignment_358909_360796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'tuple_var_assignment_358909')
        # Assigning a type to the variable 'R' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'R', tuple_var_assignment_358909_360796)
        
        # Assigning a Name to a Name (line 592):
        # Getting the type of 'tuple_var_assignment_358910' (line 592)
        tuple_var_assignment_358910_360797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'tuple_var_assignment_358910')
        # Assigning a type to the variable 'C' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 10), 'C', tuple_var_assignment_358910_360797)
        
        # Assigning a Attribute to a Tuple (line 593):
        
        # Assigning a Subscript to a Name (line 593):
        
        # Obtaining the type of the subscript
        int_360798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 8), 'int')
        # Getting the type of 'self' (line 593)
        self_360799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 14), 'self')
        # Obtaining the member 'shape' of a type (line 593)
        shape_360800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 14), self_360799, 'shape')
        # Obtaining the member '__getitem__' of a type (line 593)
        getitem___360801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 8), shape_360800, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 593)
        subscript_call_result_360802 = invoke(stypy.reporting.localization.Localization(__file__, 593, 8), getitem___360801, int_360798)
        
        # Assigning a type to the variable 'tuple_var_assignment_358911' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_358911', subscript_call_result_360802)
        
        # Assigning a Subscript to a Name (line 593):
        
        # Obtaining the type of the subscript
        int_360803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 8), 'int')
        # Getting the type of 'self' (line 593)
        self_360804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 14), 'self')
        # Obtaining the member 'shape' of a type (line 593)
        shape_360805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 14), self_360804, 'shape')
        # Obtaining the member '__getitem__' of a type (line 593)
        getitem___360806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 8), shape_360805, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 593)
        subscript_call_result_360807 = invoke(stypy.reporting.localization.Localization(__file__, 593, 8), getitem___360806, int_360803)
        
        # Assigning a type to the variable 'tuple_var_assignment_358912' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_358912', subscript_call_result_360807)
        
        # Assigning a Name to a Name (line 593):
        # Getting the type of 'tuple_var_assignment_358911' (line 593)
        tuple_var_assignment_358911_360808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_358911')
        # Assigning a type to the variable 'M' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'M', tuple_var_assignment_358911_360808)
        
        # Assigning a Name to a Name (line 593):
        # Getting the type of 'tuple_var_assignment_358912' (line 593)
        tuple_var_assignment_358912_360809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_358912')
        # Assigning a type to the variable 'N' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 10), 'N', tuple_var_assignment_358912_360809)
        
        
        
        # Call to len(...): (line 595)
        # Processing the call arguments (line 595)
        # Getting the type of 'self' (line 595)
        self_360811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 15), 'self', False)
        # Obtaining the member 'indptr' of a type (line 595)
        indptr_360812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 15), self_360811, 'indptr')
        # Processing the call keyword arguments (line 595)
        kwargs_360813 = {}
        # Getting the type of 'len' (line 595)
        len_360810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 11), 'len', False)
        # Calling len(args, kwargs) (line 595)
        len_call_result_360814 = invoke(stypy.reporting.localization.Localization(__file__, 595, 11), len_360810, *[indptr_360812], **kwargs_360813)
        
        # Getting the type of 'M' (line 595)
        M_360815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 31), 'M')
        # Getting the type of 'R' (line 595)
        R_360816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 34), 'R')
        # Applying the binary operator '//' (line 595)
        result_floordiv_360817 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 31), '//', M_360815, R_360816)
        
        int_360818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 38), 'int')
        # Applying the binary operator '+' (line 595)
        result_add_360819 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 31), '+', result_floordiv_360817, int_360818)
        
        # Applying the binary operator '!=' (line 595)
        result_ne_360820 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 11), '!=', len_call_result_360814, result_add_360819)
        
        # Testing the type of an if condition (line 595)
        if_condition_360821 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 595, 8), result_ne_360820)
        # Assigning a type to the variable 'if_condition_360821' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'if_condition_360821', if_condition_360821)
        # SSA begins for if statement (line 595)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 596)
        # Processing the call arguments (line 596)
        str_360823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 29), 'str', 'index pointer has invalid length')
        # Processing the call keyword arguments (line 596)
        kwargs_360824 = {}
        # Getting the type of 'ValueError' (line 596)
        ValueError_360822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 596)
        ValueError_call_result_360825 = invoke(stypy.reporting.localization.Localization(__file__, 596, 18), ValueError_360822, *[str_360823], **kwargs_360824)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 596, 12), ValueError_call_result_360825, 'raise parameter', BaseException)
        # SSA join for if statement (line 595)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 598):
        
        # Assigning a Subscript to a Name (line 598):
        
        # Obtaining the type of the subscript
        int_360826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 27), 'int')
        # Getting the type of 'self' (line 598)
        self_360827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 15), 'self')
        # Obtaining the member 'indptr' of a type (line 598)
        indptr_360828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 15), self_360827, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 598)
        getitem___360829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 15), indptr_360828, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 598)
        subscript_call_result_360830 = invoke(stypy.reporting.localization.Localization(__file__, 598, 15), getitem___360829, int_360826)
        
        # Assigning a type to the variable 'bnnz' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'bnnz', subscript_call_result_360830)
        
        
        
        # Call to len(...): (line 600)
        # Processing the call arguments (line 600)
        # Getting the type of 'self' (line 600)
        self_360832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 15), 'self', False)
        # Obtaining the member 'indices' of a type (line 600)
        indices_360833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 15), self_360832, 'indices')
        # Processing the call keyword arguments (line 600)
        kwargs_360834 = {}
        # Getting the type of 'len' (line 600)
        len_360831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 11), 'len', False)
        # Calling len(args, kwargs) (line 600)
        len_call_result_360835 = invoke(stypy.reporting.localization.Localization(__file__, 600, 11), len_360831, *[indices_360833], **kwargs_360834)
        
        # Getting the type of 'bnnz' (line 600)
        bnnz_360836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 31), 'bnnz')
        # Applying the binary operator '<' (line 600)
        result_lt_360837 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 11), '<', len_call_result_360835, bnnz_360836)
        
        # Testing the type of an if condition (line 600)
        if_condition_360838 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 600, 8), result_lt_360837)
        # Assigning a type to the variable 'if_condition_360838' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'if_condition_360838', if_condition_360838)
        # SSA begins for if statement (line 600)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 601)
        # Processing the call arguments (line 601)
        str_360840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 29), 'str', 'indices array has too few elements')
        # Processing the call keyword arguments (line 601)
        kwargs_360841 = {}
        # Getting the type of 'ValueError' (line 601)
        ValueError_360839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 601)
        ValueError_call_result_360842 = invoke(stypy.reporting.localization.Localization(__file__, 601, 18), ValueError_360839, *[str_360840], **kwargs_360841)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 601, 12), ValueError_call_result_360842, 'raise parameter', BaseException)
        # SSA join for if statement (line 600)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 602)
        # Processing the call arguments (line 602)
        # Getting the type of 'self' (line 602)
        self_360844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 15), 'self', False)
        # Obtaining the member 'data' of a type (line 602)
        data_360845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 15), self_360844, 'data')
        # Processing the call keyword arguments (line 602)
        kwargs_360846 = {}
        # Getting the type of 'len' (line 602)
        len_360843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 11), 'len', False)
        # Calling len(args, kwargs) (line 602)
        len_call_result_360847 = invoke(stypy.reporting.localization.Localization(__file__, 602, 11), len_360843, *[data_360845], **kwargs_360846)
        
        # Getting the type of 'bnnz' (line 602)
        bnnz_360848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 28), 'bnnz')
        # Applying the binary operator '<' (line 602)
        result_lt_360849 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 11), '<', len_call_result_360847, bnnz_360848)
        
        # Testing the type of an if condition (line 602)
        if_condition_360850 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 602, 8), result_lt_360849)
        # Assigning a type to the variable 'if_condition_360850' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'if_condition_360850', if_condition_360850)
        # SSA begins for if statement (line 602)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 603)
        # Processing the call arguments (line 603)
        str_360852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 29), 'str', 'data array has too few elements')
        # Processing the call keyword arguments (line 603)
        kwargs_360853 = {}
        # Getting the type of 'ValueError' (line 603)
        ValueError_360851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 603)
        ValueError_call_result_360854 = invoke(stypy.reporting.localization.Localization(__file__, 603, 18), ValueError_360851, *[str_360852], **kwargs_360853)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 603, 12), ValueError_call_result_360854, 'raise parameter', BaseException)
        # SSA join for if statement (line 602)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Attribute (line 605):
        
        # Assigning a Subscript to a Attribute (line 605):
        
        # Obtaining the type of the subscript
        # Getting the type of 'bnnz' (line 605)
        bnnz_360855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 31), 'bnnz')
        slice_360856 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 605, 20), None, bnnz_360855, None)
        # Getting the type of 'self' (line 605)
        self_360857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 20), 'self')
        # Obtaining the member 'data' of a type (line 605)
        data_360858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 20), self_360857, 'data')
        # Obtaining the member '__getitem__' of a type (line 605)
        getitem___360859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 20), data_360858, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 605)
        subscript_call_result_360860 = invoke(stypy.reporting.localization.Localization(__file__, 605, 20), getitem___360859, slice_360856)
        
        # Getting the type of 'self' (line 605)
        self_360861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'self')
        # Setting the type of the member 'data' of a type (line 605)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 8), self_360861, 'data', subscript_call_result_360860)
        
        # Assigning a Subscript to a Attribute (line 606):
        
        # Assigning a Subscript to a Attribute (line 606):
        
        # Obtaining the type of the subscript
        # Getting the type of 'bnnz' (line 606)
        bnnz_360862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 37), 'bnnz')
        slice_360863 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 606, 23), None, bnnz_360862, None)
        # Getting the type of 'self' (line 606)
        self_360864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 23), 'self')
        # Obtaining the member 'indices' of a type (line 606)
        indices_360865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 23), self_360864, 'indices')
        # Obtaining the member '__getitem__' of a type (line 606)
        getitem___360866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 23), indices_360865, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 606)
        subscript_call_result_360867 = invoke(stypy.reporting.localization.Localization(__file__, 606, 23), getitem___360866, slice_360863)
        
        # Getting the type of 'self' (line 606)
        self_360868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'self')
        # Setting the type of the member 'indices' of a type (line 606)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 8), self_360868, 'indices', subscript_call_result_360867)
        
        # ################# End of 'prune(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'prune' in the type store
        # Getting the type of 'stypy_return_type' (line 588)
        stypy_return_type_360869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_360869)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'prune'
        return stypy_return_type_360869


    @norecursion
    def _binopt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 609)
        None_360870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 42), 'None')
        # Getting the type of 'None' (line 609)
        None_360871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 58), 'None')
        defaults = [None_360870, None_360871]
        # Create a new context for function '_binopt'
        module_type_store = module_type_store.open_function_context('_binopt', 609, 4, False)
        # Assigning a type to the variable 'self' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix._binopt.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix._binopt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix._binopt.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix._binopt.__dict__.__setitem__('stypy_function_name', 'bsr_matrix._binopt')
        bsr_matrix._binopt.__dict__.__setitem__('stypy_param_names_list', ['other', 'op', 'in_shape', 'out_shape'])
        bsr_matrix._binopt.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix._binopt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix._binopt.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix._binopt.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix._binopt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix._binopt.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix._binopt', ['other', 'op', 'in_shape', 'out_shape'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_binopt', localization, ['other', 'op', 'in_shape', 'out_shape'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_binopt(...)' code ##################

        str_360872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 8), 'str', 'Apply the binary operation fn to two sparse matrices.')
        
        # Assigning a Call to a Name (line 614):
        
        # Assigning a Call to a Name (line 614):
        
        # Call to __class__(...): (line 614)
        # Processing the call arguments (line 614)
        # Getting the type of 'other' (line 614)
        other_360875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 31), 'other', False)
        # Processing the call keyword arguments (line 614)
        # Getting the type of 'self' (line 614)
        self_360876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 48), 'self', False)
        # Obtaining the member 'blocksize' of a type (line 614)
        blocksize_360877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 48), self_360876, 'blocksize')
        keyword_360878 = blocksize_360877
        kwargs_360879 = {'blocksize': keyword_360878}
        # Getting the type of 'self' (line 614)
        self_360873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 16), 'self', False)
        # Obtaining the member '__class__' of a type (line 614)
        class___360874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 16), self_360873, '__class__')
        # Calling __class__(args, kwargs) (line 614)
        class___call_result_360880 = invoke(stypy.reporting.localization.Localization(__file__, 614, 16), class___360874, *[other_360875], **kwargs_360879)
        
        # Assigning a type to the variable 'other' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'other', class___call_result_360880)
        
        # Assigning a Call to a Name (line 617):
        
        # Assigning a Call to a Name (line 617):
        
        # Call to getattr(...): (line 617)
        # Processing the call arguments (line 617)
        # Getting the type of '_sparsetools' (line 617)
        _sparsetools_360882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 21), '_sparsetools', False)
        # Getting the type of 'self' (line 617)
        self_360883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 35), 'self', False)
        # Obtaining the member 'format' of a type (line 617)
        format_360884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 35), self_360883, 'format')
        # Getting the type of 'op' (line 617)
        op_360885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 49), 'op', False)
        # Applying the binary operator '+' (line 617)
        result_add_360886 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 35), '+', format_360884, op_360885)
        
        # Getting the type of 'self' (line 617)
        self_360887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 54), 'self', False)
        # Obtaining the member 'format' of a type (line 617)
        format_360888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 54), self_360887, 'format')
        # Applying the binary operator '+' (line 617)
        result_add_360889 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 52), '+', result_add_360886, format_360888)
        
        # Processing the call keyword arguments (line 617)
        kwargs_360890 = {}
        # Getting the type of 'getattr' (line 617)
        getattr_360881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 13), 'getattr', False)
        # Calling getattr(args, kwargs) (line 617)
        getattr_call_result_360891 = invoke(stypy.reporting.localization.Localization(__file__, 617, 13), getattr_360881, *[_sparsetools_360882, result_add_360889], **kwargs_360890)
        
        # Assigning a type to the variable 'fn' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'fn', getattr_call_result_360891)
        
        # Assigning a Attribute to a Tuple (line 619):
        
        # Assigning a Subscript to a Name (line 619):
        
        # Obtaining the type of the subscript
        int_360892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 8), 'int')
        # Getting the type of 'self' (line 619)
        self_360893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 619)
        blocksize_360894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 14), self_360893, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 619)
        getitem___360895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 8), blocksize_360894, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 619)
        subscript_call_result_360896 = invoke(stypy.reporting.localization.Localization(__file__, 619, 8), getitem___360895, int_360892)
        
        # Assigning a type to the variable 'tuple_var_assignment_358913' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'tuple_var_assignment_358913', subscript_call_result_360896)
        
        # Assigning a Subscript to a Name (line 619):
        
        # Obtaining the type of the subscript
        int_360897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 8), 'int')
        # Getting the type of 'self' (line 619)
        self_360898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 14), 'self')
        # Obtaining the member 'blocksize' of a type (line 619)
        blocksize_360899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 14), self_360898, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 619)
        getitem___360900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 8), blocksize_360899, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 619)
        subscript_call_result_360901 = invoke(stypy.reporting.localization.Localization(__file__, 619, 8), getitem___360900, int_360897)
        
        # Assigning a type to the variable 'tuple_var_assignment_358914' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'tuple_var_assignment_358914', subscript_call_result_360901)
        
        # Assigning a Name to a Name (line 619):
        # Getting the type of 'tuple_var_assignment_358913' (line 619)
        tuple_var_assignment_358913_360902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'tuple_var_assignment_358913')
        # Assigning a type to the variable 'R' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'R', tuple_var_assignment_358913_360902)
        
        # Assigning a Name to a Name (line 619):
        # Getting the type of 'tuple_var_assignment_358914' (line 619)
        tuple_var_assignment_358914_360903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'tuple_var_assignment_358914')
        # Assigning a type to the variable 'C' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 10), 'C', tuple_var_assignment_358914_360903)
        
        # Assigning a BinOp to a Name (line 621):
        
        # Assigning a BinOp to a Name (line 621):
        
        # Call to len(...): (line 621)
        # Processing the call arguments (line 621)
        # Getting the type of 'self' (line 621)
        self_360905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 23), 'self', False)
        # Obtaining the member 'data' of a type (line 621)
        data_360906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 23), self_360905, 'data')
        # Processing the call keyword arguments (line 621)
        kwargs_360907 = {}
        # Getting the type of 'len' (line 621)
        len_360904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 19), 'len', False)
        # Calling len(args, kwargs) (line 621)
        len_call_result_360908 = invoke(stypy.reporting.localization.Localization(__file__, 621, 19), len_360904, *[data_360906], **kwargs_360907)
        
        
        # Call to len(...): (line 621)
        # Processing the call arguments (line 621)
        # Getting the type of 'other' (line 621)
        other_360910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 40), 'other', False)
        # Obtaining the member 'data' of a type (line 621)
        data_360911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 40), other_360910, 'data')
        # Processing the call keyword arguments (line 621)
        kwargs_360912 = {}
        # Getting the type of 'len' (line 621)
        len_360909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 36), 'len', False)
        # Calling len(args, kwargs) (line 621)
        len_call_result_360913 = invoke(stypy.reporting.localization.Localization(__file__, 621, 36), len_360909, *[data_360911], **kwargs_360912)
        
        # Applying the binary operator '+' (line 621)
        result_add_360914 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 19), '+', len_call_result_360908, len_call_result_360913)
        
        # Assigning a type to the variable 'max_bnnz' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'max_bnnz', result_add_360914)
        
        # Assigning a Call to a Name (line 622):
        
        # Assigning a Call to a Name (line 622):
        
        # Call to get_index_dtype(...): (line 622)
        # Processing the call arguments (line 622)
        
        # Obtaining an instance of the builtin type 'tuple' (line 622)
        tuple_360916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 622)
        # Adding element type (line 622)
        # Getting the type of 'self' (line 622)
        self_360917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 37), 'self', False)
        # Obtaining the member 'indptr' of a type (line 622)
        indptr_360918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 37), self_360917, 'indptr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 37), tuple_360916, indptr_360918)
        # Adding element type (line 622)
        # Getting the type of 'self' (line 622)
        self_360919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 50), 'self', False)
        # Obtaining the member 'indices' of a type (line 622)
        indices_360920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 50), self_360919, 'indices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 37), tuple_360916, indices_360920)
        # Adding element type (line 622)
        # Getting the type of 'other' (line 623)
        other_360921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 37), 'other', False)
        # Obtaining the member 'indptr' of a type (line 623)
        indptr_360922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 37), other_360921, 'indptr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 37), tuple_360916, indptr_360922)
        # Adding element type (line 622)
        # Getting the type of 'other' (line 623)
        other_360923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 51), 'other', False)
        # Obtaining the member 'indices' of a type (line 623)
        indices_360924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 51), other_360923, 'indices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 37), tuple_360916, indices_360924)
        
        # Processing the call keyword arguments (line 622)
        # Getting the type of 'max_bnnz' (line 624)
        max_bnnz_360925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 43), 'max_bnnz', False)
        keyword_360926 = max_bnnz_360925
        kwargs_360927 = {'maxval': keyword_360926}
        # Getting the type of 'get_index_dtype' (line 622)
        get_index_dtype_360915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 20), 'get_index_dtype', False)
        # Calling get_index_dtype(args, kwargs) (line 622)
        get_index_dtype_call_result_360928 = invoke(stypy.reporting.localization.Localization(__file__, 622, 20), get_index_dtype_360915, *[tuple_360916], **kwargs_360927)
        
        # Assigning a type to the variable 'idx_dtype' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'idx_dtype', get_index_dtype_call_result_360928)
        
        # Assigning a Call to a Name (line 625):
        
        # Assigning a Call to a Name (line 625):
        
        # Call to empty(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'self' (line 625)
        self_360931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 26), 'self', False)
        # Obtaining the member 'indptr' of a type (line 625)
        indptr_360932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 26), self_360931, 'indptr')
        # Obtaining the member 'shape' of a type (line 625)
        shape_360933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 26), indptr_360932, 'shape')
        # Processing the call keyword arguments (line 625)
        # Getting the type of 'idx_dtype' (line 625)
        idx_dtype_360934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 51), 'idx_dtype', False)
        keyword_360935 = idx_dtype_360934
        kwargs_360936 = {'dtype': keyword_360935}
        # Getting the type of 'np' (line 625)
        np_360929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 17), 'np', False)
        # Obtaining the member 'empty' of a type (line 625)
        empty_360930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 17), np_360929, 'empty')
        # Calling empty(args, kwargs) (line 625)
        empty_call_result_360937 = invoke(stypy.reporting.localization.Localization(__file__, 625, 17), empty_360930, *[shape_360933], **kwargs_360936)
        
        # Assigning a type to the variable 'indptr' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'indptr', empty_call_result_360937)
        
        # Assigning a Call to a Name (line 626):
        
        # Assigning a Call to a Name (line 626):
        
        # Call to empty(...): (line 626)
        # Processing the call arguments (line 626)
        # Getting the type of 'max_bnnz' (line 626)
        max_bnnz_360940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 27), 'max_bnnz', False)
        # Processing the call keyword arguments (line 626)
        # Getting the type of 'idx_dtype' (line 626)
        idx_dtype_360941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 43), 'idx_dtype', False)
        keyword_360942 = idx_dtype_360941
        kwargs_360943 = {'dtype': keyword_360942}
        # Getting the type of 'np' (line 626)
        np_360938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 18), 'np', False)
        # Obtaining the member 'empty' of a type (line 626)
        empty_360939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 18), np_360938, 'empty')
        # Calling empty(args, kwargs) (line 626)
        empty_call_result_360944 = invoke(stypy.reporting.localization.Localization(__file__, 626, 18), empty_360939, *[max_bnnz_360940], **kwargs_360943)
        
        # Assigning a type to the variable 'indices' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'indices', empty_call_result_360944)
        
        # Assigning a List to a Name (line 628):
        
        # Assigning a List to a Name (line 628):
        
        # Obtaining an instance of the builtin type 'list' (line 628)
        list_360945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 628)
        # Adding element type (line 628)
        str_360946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 20), 'str', '_ne_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 628, 19), list_360945, str_360946)
        # Adding element type (line 628)
        str_360947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 28), 'str', '_lt_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 628, 19), list_360945, str_360947)
        # Adding element type (line 628)
        str_360948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 36), 'str', '_gt_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 628, 19), list_360945, str_360948)
        # Adding element type (line 628)
        str_360949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 44), 'str', '_le_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 628, 19), list_360945, str_360949)
        # Adding element type (line 628)
        str_360950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 52), 'str', '_ge_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 628, 19), list_360945, str_360950)
        
        # Assigning a type to the variable 'bool_ops' (line 628)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 8), 'bool_ops', list_360945)
        
        
        # Getting the type of 'op' (line 629)
        op_360951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 11), 'op')
        # Getting the type of 'bool_ops' (line 629)
        bool_ops_360952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 17), 'bool_ops')
        # Applying the binary operator 'in' (line 629)
        result_contains_360953 = python_operator(stypy.reporting.localization.Localization(__file__, 629, 11), 'in', op_360951, bool_ops_360952)
        
        # Testing the type of an if condition (line 629)
        if_condition_360954 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 629, 8), result_contains_360953)
        # Assigning a type to the variable 'if_condition_360954' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'if_condition_360954', if_condition_360954)
        # SSA begins for if statement (line 629)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 630):
        
        # Assigning a Call to a Name (line 630):
        
        # Call to empty(...): (line 630)
        # Processing the call arguments (line 630)
        # Getting the type of 'R' (line 630)
        R_360957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 28), 'R', False)
        # Getting the type of 'C' (line 630)
        C_360958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 30), 'C', False)
        # Applying the binary operator '*' (line 630)
        result_mul_360959 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 28), '*', R_360957, C_360958)
        
        # Getting the type of 'max_bnnz' (line 630)
        max_bnnz_360960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 32), 'max_bnnz', False)
        # Applying the binary operator '*' (line 630)
        result_mul_360961 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 31), '*', result_mul_360959, max_bnnz_360960)
        
        # Processing the call keyword arguments (line 630)
        # Getting the type of 'np' (line 630)
        np_360962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 48), 'np', False)
        # Obtaining the member 'bool_' of a type (line 630)
        bool__360963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 48), np_360962, 'bool_')
        keyword_360964 = bool__360963
        kwargs_360965 = {'dtype': keyword_360964}
        # Getting the type of 'np' (line 630)
        np_360955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 19), 'np', False)
        # Obtaining the member 'empty' of a type (line 630)
        empty_360956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 19), np_360955, 'empty')
        # Calling empty(args, kwargs) (line 630)
        empty_call_result_360966 = invoke(stypy.reporting.localization.Localization(__file__, 630, 19), empty_360956, *[result_mul_360961], **kwargs_360965)
        
        # Assigning a type to the variable 'data' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'data', empty_call_result_360966)
        # SSA branch for the else part of an if statement (line 629)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 632):
        
        # Assigning a Call to a Name (line 632):
        
        # Call to empty(...): (line 632)
        # Processing the call arguments (line 632)
        # Getting the type of 'R' (line 632)
        R_360969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 28), 'R', False)
        # Getting the type of 'C' (line 632)
        C_360970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 30), 'C', False)
        # Applying the binary operator '*' (line 632)
        result_mul_360971 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 28), '*', R_360969, C_360970)
        
        # Getting the type of 'max_bnnz' (line 632)
        max_bnnz_360972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 32), 'max_bnnz', False)
        # Applying the binary operator '*' (line 632)
        result_mul_360973 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 31), '*', result_mul_360971, max_bnnz_360972)
        
        # Processing the call keyword arguments (line 632)
        
        # Call to upcast(...): (line 632)
        # Processing the call arguments (line 632)
        # Getting the type of 'self' (line 632)
        self_360975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 55), 'self', False)
        # Obtaining the member 'dtype' of a type (line 632)
        dtype_360976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 55), self_360975, 'dtype')
        # Getting the type of 'other' (line 632)
        other_360977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 66), 'other', False)
        # Obtaining the member 'dtype' of a type (line 632)
        dtype_360978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 66), other_360977, 'dtype')
        # Processing the call keyword arguments (line 632)
        kwargs_360979 = {}
        # Getting the type of 'upcast' (line 632)
        upcast_360974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 48), 'upcast', False)
        # Calling upcast(args, kwargs) (line 632)
        upcast_call_result_360980 = invoke(stypy.reporting.localization.Localization(__file__, 632, 48), upcast_360974, *[dtype_360976, dtype_360978], **kwargs_360979)
        
        keyword_360981 = upcast_call_result_360980
        kwargs_360982 = {'dtype': keyword_360981}
        # Getting the type of 'np' (line 632)
        np_360967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 19), 'np', False)
        # Obtaining the member 'empty' of a type (line 632)
        empty_360968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 19), np_360967, 'empty')
        # Calling empty(args, kwargs) (line 632)
        empty_call_result_360983 = invoke(stypy.reporting.localization.Localization(__file__, 632, 19), empty_360968, *[result_mul_360973], **kwargs_360982)
        
        # Assigning a type to the variable 'data' (line 632)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 12), 'data', empty_call_result_360983)
        # SSA join for if statement (line 629)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to fn(...): (line 634)
        # Processing the call arguments (line 634)
        
        # Obtaining the type of the subscript
        int_360985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 22), 'int')
        # Getting the type of 'self' (line 634)
        self_360986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 11), 'self', False)
        # Obtaining the member 'shape' of a type (line 634)
        shape_360987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 11), self_360986, 'shape')
        # Obtaining the member '__getitem__' of a type (line 634)
        getitem___360988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 11), shape_360987, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 634)
        subscript_call_result_360989 = invoke(stypy.reporting.localization.Localization(__file__, 634, 11), getitem___360988, int_360985)
        
        # Getting the type of 'R' (line 634)
        R_360990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 26), 'R', False)
        # Applying the binary operator '//' (line 634)
        result_floordiv_360991 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 11), '//', subscript_call_result_360989, R_360990)
        
        
        # Obtaining the type of the subscript
        int_360992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 40), 'int')
        # Getting the type of 'self' (line 634)
        self_360993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 29), 'self', False)
        # Obtaining the member 'shape' of a type (line 634)
        shape_360994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 29), self_360993, 'shape')
        # Obtaining the member '__getitem__' of a type (line 634)
        getitem___360995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 29), shape_360994, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 634)
        subscript_call_result_360996 = invoke(stypy.reporting.localization.Localization(__file__, 634, 29), getitem___360995, int_360992)
        
        # Getting the type of 'C' (line 634)
        C_360997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 44), 'C', False)
        # Applying the binary operator '//' (line 634)
        result_floordiv_360998 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 29), '//', subscript_call_result_360996, C_360997)
        
        # Getting the type of 'R' (line 634)
        R_360999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 47), 'R', False)
        # Getting the type of 'C' (line 634)
        C_361000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 50), 'C', False)
        
        # Call to astype(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'idx_dtype' (line 635)
        idx_dtype_361004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 30), 'idx_dtype', False)
        # Processing the call keyword arguments (line 635)
        kwargs_361005 = {}
        # Getting the type of 'self' (line 635)
        self_361001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 11), 'self', False)
        # Obtaining the member 'indptr' of a type (line 635)
        indptr_361002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 11), self_361001, 'indptr')
        # Obtaining the member 'astype' of a type (line 635)
        astype_361003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 11), indptr_361002, 'astype')
        # Calling astype(args, kwargs) (line 635)
        astype_call_result_361006 = invoke(stypy.reporting.localization.Localization(__file__, 635, 11), astype_361003, *[idx_dtype_361004], **kwargs_361005)
        
        
        # Call to astype(...): (line 636)
        # Processing the call arguments (line 636)
        # Getting the type of 'idx_dtype' (line 636)
        idx_dtype_361010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 31), 'idx_dtype', False)
        # Processing the call keyword arguments (line 636)
        kwargs_361011 = {}
        # Getting the type of 'self' (line 636)
        self_361007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 11), 'self', False)
        # Obtaining the member 'indices' of a type (line 636)
        indices_361008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 11), self_361007, 'indices')
        # Obtaining the member 'astype' of a type (line 636)
        astype_361009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 11), indices_361008, 'astype')
        # Calling astype(args, kwargs) (line 636)
        astype_call_result_361012 = invoke(stypy.reporting.localization.Localization(__file__, 636, 11), astype_361009, *[idx_dtype_361010], **kwargs_361011)
        
        # Getting the type of 'self' (line 637)
        self_361013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 11), 'self', False)
        # Obtaining the member 'data' of a type (line 637)
        data_361014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 11), self_361013, 'data')
        
        # Call to astype(...): (line 638)
        # Processing the call arguments (line 638)
        # Getting the type of 'idx_dtype' (line 638)
        idx_dtype_361018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 31), 'idx_dtype', False)
        # Processing the call keyword arguments (line 638)
        kwargs_361019 = {}
        # Getting the type of 'other' (line 638)
        other_361015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 11), 'other', False)
        # Obtaining the member 'indptr' of a type (line 638)
        indptr_361016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 11), other_361015, 'indptr')
        # Obtaining the member 'astype' of a type (line 638)
        astype_361017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 11), indptr_361016, 'astype')
        # Calling astype(args, kwargs) (line 638)
        astype_call_result_361020 = invoke(stypy.reporting.localization.Localization(__file__, 638, 11), astype_361017, *[idx_dtype_361018], **kwargs_361019)
        
        
        # Call to astype(...): (line 639)
        # Processing the call arguments (line 639)
        # Getting the type of 'idx_dtype' (line 639)
        idx_dtype_361024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 32), 'idx_dtype', False)
        # Processing the call keyword arguments (line 639)
        kwargs_361025 = {}
        # Getting the type of 'other' (line 639)
        other_361021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 11), 'other', False)
        # Obtaining the member 'indices' of a type (line 639)
        indices_361022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 11), other_361021, 'indices')
        # Obtaining the member 'astype' of a type (line 639)
        astype_361023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 11), indices_361022, 'astype')
        # Calling astype(args, kwargs) (line 639)
        astype_call_result_361026 = invoke(stypy.reporting.localization.Localization(__file__, 639, 11), astype_361023, *[idx_dtype_361024], **kwargs_361025)
        
        
        # Call to ravel(...): (line 640)
        # Processing the call arguments (line 640)
        # Getting the type of 'other' (line 640)
        other_361029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 20), 'other', False)
        # Obtaining the member 'data' of a type (line 640)
        data_361030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 20), other_361029, 'data')
        # Processing the call keyword arguments (line 640)
        kwargs_361031 = {}
        # Getting the type of 'np' (line 640)
        np_361027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 11), 'np', False)
        # Obtaining the member 'ravel' of a type (line 640)
        ravel_361028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 11), np_361027, 'ravel')
        # Calling ravel(args, kwargs) (line 640)
        ravel_call_result_361032 = invoke(stypy.reporting.localization.Localization(__file__, 640, 11), ravel_361028, *[data_361030], **kwargs_361031)
        
        # Getting the type of 'indptr' (line 641)
        indptr_361033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 11), 'indptr', False)
        # Getting the type of 'indices' (line 642)
        indices_361034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 11), 'indices', False)
        # Getting the type of 'data' (line 643)
        data_361035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 11), 'data', False)
        # Processing the call keyword arguments (line 634)
        kwargs_361036 = {}
        # Getting the type of 'fn' (line 634)
        fn_360984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'fn', False)
        # Calling fn(args, kwargs) (line 634)
        fn_call_result_361037 = invoke(stypy.reporting.localization.Localization(__file__, 634, 8), fn_360984, *[result_floordiv_360991, result_floordiv_360998, R_360999, C_361000, astype_call_result_361006, astype_call_result_361012, data_361014, astype_call_result_361020, astype_call_result_361026, ravel_call_result_361032, indptr_361033, indices_361034, data_361035], **kwargs_361036)
        
        
        # Assigning a Subscript to a Name (line 645):
        
        # Assigning a Subscript to a Name (line 645):
        
        # Obtaining the type of the subscript
        int_361038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 29), 'int')
        # Getting the type of 'indptr' (line 645)
        indptr_361039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 22), 'indptr')
        # Obtaining the member '__getitem__' of a type (line 645)
        getitem___361040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 22), indptr_361039, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 645)
        subscript_call_result_361041 = invoke(stypy.reporting.localization.Localization(__file__, 645, 22), getitem___361040, int_361038)
        
        # Assigning a type to the variable 'actual_bnnz' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'actual_bnnz', subscript_call_result_361041)
        
        # Assigning a Subscript to a Name (line 646):
        
        # Assigning a Subscript to a Name (line 646):
        
        # Obtaining the type of the subscript
        # Getting the type of 'actual_bnnz' (line 646)
        actual_bnnz_361042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 27), 'actual_bnnz')
        slice_361043 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 646, 18), None, actual_bnnz_361042, None)
        # Getting the type of 'indices' (line 646)
        indices_361044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 18), 'indices')
        # Obtaining the member '__getitem__' of a type (line 646)
        getitem___361045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 18), indices_361044, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 646)
        subscript_call_result_361046 = invoke(stypy.reporting.localization.Localization(__file__, 646, 18), getitem___361045, slice_361043)
        
        # Assigning a type to the variable 'indices' (line 646)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'indices', subscript_call_result_361046)
        
        # Assigning a Subscript to a Name (line 647):
        
        # Assigning a Subscript to a Name (line 647):
        
        # Obtaining the type of the subscript
        # Getting the type of 'R' (line 647)
        R_361047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 21), 'R')
        # Getting the type of 'C' (line 647)
        C_361048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 23), 'C')
        # Applying the binary operator '*' (line 647)
        result_mul_361049 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 21), '*', R_361047, C_361048)
        
        # Getting the type of 'actual_bnnz' (line 647)
        actual_bnnz_361050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 25), 'actual_bnnz')
        # Applying the binary operator '*' (line 647)
        result_mul_361051 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 24), '*', result_mul_361049, actual_bnnz_361050)
        
        slice_361052 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 647, 15), None, result_mul_361051, None)
        # Getting the type of 'data' (line 647)
        data_361053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 15), 'data')
        # Obtaining the member '__getitem__' of a type (line 647)
        getitem___361054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 15), data_361053, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 647)
        subscript_call_result_361055 = invoke(stypy.reporting.localization.Localization(__file__, 647, 15), getitem___361054, slice_361052)
        
        # Assigning a type to the variable 'data' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'data', subscript_call_result_361055)
        
        
        # Getting the type of 'actual_bnnz' (line 649)
        actual_bnnz_361056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 11), 'actual_bnnz')
        # Getting the type of 'max_bnnz' (line 649)
        max_bnnz_361057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 25), 'max_bnnz')
        int_361058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 34), 'int')
        # Applying the binary operator 'div' (line 649)
        result_div_361059 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 25), 'div', max_bnnz_361057, int_361058)
        
        # Applying the binary operator '<' (line 649)
        result_lt_361060 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 11), '<', actual_bnnz_361056, result_div_361059)
        
        # Testing the type of an if condition (line 649)
        if_condition_361061 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 649, 8), result_lt_361060)
        # Assigning a type to the variable 'if_condition_361061' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'if_condition_361061', if_condition_361061)
        # SSA begins for if statement (line 649)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 650):
        
        # Assigning a Call to a Name (line 650):
        
        # Call to copy(...): (line 650)
        # Processing the call keyword arguments (line 650)
        kwargs_361064 = {}
        # Getting the type of 'indices' (line 650)
        indices_361062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 22), 'indices', False)
        # Obtaining the member 'copy' of a type (line 650)
        copy_361063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 22), indices_361062, 'copy')
        # Calling copy(args, kwargs) (line 650)
        copy_call_result_361065 = invoke(stypy.reporting.localization.Localization(__file__, 650, 22), copy_361063, *[], **kwargs_361064)
        
        # Assigning a type to the variable 'indices' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 12), 'indices', copy_call_result_361065)
        
        # Assigning a Call to a Name (line 651):
        
        # Assigning a Call to a Name (line 651):
        
        # Call to copy(...): (line 651)
        # Processing the call keyword arguments (line 651)
        kwargs_361068 = {}
        # Getting the type of 'data' (line 651)
        data_361066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 19), 'data', False)
        # Obtaining the member 'copy' of a type (line 651)
        copy_361067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 19), data_361066, 'copy')
        # Calling copy(args, kwargs) (line 651)
        copy_call_result_361069 = invoke(stypy.reporting.localization.Localization(__file__, 651, 19), copy_361067, *[], **kwargs_361068)
        
        # Assigning a type to the variable 'data' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 12), 'data', copy_call_result_361069)
        # SSA join for if statement (line 649)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 653):
        
        # Assigning a Call to a Name (line 653):
        
        # Call to reshape(...): (line 653)
        # Processing the call arguments (line 653)
        int_361072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 28), 'int')
        # Getting the type of 'R' (line 653)
        R_361073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 31), 'R', False)
        # Getting the type of 'C' (line 653)
        C_361074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 33), 'C', False)
        # Processing the call keyword arguments (line 653)
        kwargs_361075 = {}
        # Getting the type of 'data' (line 653)
        data_361070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 15), 'data', False)
        # Obtaining the member 'reshape' of a type (line 653)
        reshape_361071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 15), data_361070, 'reshape')
        # Calling reshape(args, kwargs) (line 653)
        reshape_call_result_361076 = invoke(stypy.reporting.localization.Localization(__file__, 653, 15), reshape_361071, *[int_361072, R_361073, C_361074], **kwargs_361075)
        
        # Assigning a type to the variable 'data' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'data', reshape_call_result_361076)
        
        # Call to __class__(...): (line 655)
        # Processing the call arguments (line 655)
        
        # Obtaining an instance of the builtin type 'tuple' (line 655)
        tuple_361079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 655)
        # Adding element type (line 655)
        # Getting the type of 'data' (line 655)
        data_361080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 31), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 655, 31), tuple_361079, data_361080)
        # Adding element type (line 655)
        # Getting the type of 'indices' (line 655)
        indices_361081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 37), 'indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 655, 31), tuple_361079, indices_361081)
        # Adding element type (line 655)
        # Getting the type of 'indptr' (line 655)
        indptr_361082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 46), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 655, 31), tuple_361079, indptr_361082)
        
        # Processing the call keyword arguments (line 655)
        # Getting the type of 'self' (line 655)
        self_361083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 61), 'self', False)
        # Obtaining the member 'shape' of a type (line 655)
        shape_361084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 61), self_361083, 'shape')
        keyword_361085 = shape_361084
        kwargs_361086 = {'shape': keyword_361085}
        # Getting the type of 'self' (line 655)
        self_361077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 655)
        class___361078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 15), self_361077, '__class__')
        # Calling __class__(args, kwargs) (line 655)
        class___call_result_361087 = invoke(stypy.reporting.localization.Localization(__file__, 655, 15), class___361078, *[tuple_361079], **kwargs_361086)
        
        # Assigning a type to the variable 'stypy_return_type' (line 655)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'stypy_return_type', class___call_result_361087)
        
        # ################# End of '_binopt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_binopt' in the type store
        # Getting the type of 'stypy_return_type' (line 609)
        stypy_return_type_361088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_361088)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_binopt'
        return stypy_return_type_361088


    @norecursion
    def _with_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 658)
        True_361089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 34), 'True')
        defaults = [True_361089]
        # Create a new context for function '_with_data'
        module_type_store = module_type_store.open_function_context('_with_data', 658, 4, False)
        # Assigning a type to the variable 'self' (line 659)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bsr_matrix._with_data.__dict__.__setitem__('stypy_localization', localization)
        bsr_matrix._with_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bsr_matrix._with_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        bsr_matrix._with_data.__dict__.__setitem__('stypy_function_name', 'bsr_matrix._with_data')
        bsr_matrix._with_data.__dict__.__setitem__('stypy_param_names_list', ['data', 'copy'])
        bsr_matrix._with_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        bsr_matrix._with_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bsr_matrix._with_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        bsr_matrix._with_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        bsr_matrix._with_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bsr_matrix._with_data.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bsr_matrix._with_data', ['data', 'copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_with_data', localization, ['data', 'copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_with_data(...)' code ##################

        str_361090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, (-1)), 'str', 'Returns a matrix with the same sparsity structure as self,\n        but with different data.  By default the structure arrays\n        (i.e. .indptr and .indices) are copied.\n        ')
        
        # Getting the type of 'copy' (line 663)
        copy_361091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 11), 'copy')
        # Testing the type of an if condition (line 663)
        if_condition_361092 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 663, 8), copy_361091)
        # Assigning a type to the variable 'if_condition_361092' (line 663)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'if_condition_361092', if_condition_361092)
        # SSA begins for if statement (line 663)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __class__(...): (line 664)
        # Processing the call arguments (line 664)
        
        # Obtaining an instance of the builtin type 'tuple' (line 664)
        tuple_361095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 664)
        # Adding element type (line 664)
        # Getting the type of 'data' (line 664)
        data_361096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 35), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 35), tuple_361095, data_361096)
        # Adding element type (line 664)
        
        # Call to copy(...): (line 664)
        # Processing the call keyword arguments (line 664)
        kwargs_361100 = {}
        # Getting the type of 'self' (line 664)
        self_361097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 40), 'self', False)
        # Obtaining the member 'indices' of a type (line 664)
        indices_361098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 40), self_361097, 'indices')
        # Obtaining the member 'copy' of a type (line 664)
        copy_361099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 40), indices_361098, 'copy')
        # Calling copy(args, kwargs) (line 664)
        copy_call_result_361101 = invoke(stypy.reporting.localization.Localization(__file__, 664, 40), copy_361099, *[], **kwargs_361100)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 35), tuple_361095, copy_call_result_361101)
        # Adding element type (line 664)
        
        # Call to copy(...): (line 664)
        # Processing the call keyword arguments (line 664)
        kwargs_361105 = {}
        # Getting the type of 'self' (line 664)
        self_361102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 60), 'self', False)
        # Obtaining the member 'indptr' of a type (line 664)
        indptr_361103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 60), self_361102, 'indptr')
        # Obtaining the member 'copy' of a type (line 664)
        copy_361104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 60), indptr_361103, 'copy')
        # Calling copy(args, kwargs) (line 664)
        copy_call_result_361106 = invoke(stypy.reporting.localization.Localization(__file__, 664, 60), copy_361104, *[], **kwargs_361105)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 35), tuple_361095, copy_call_result_361106)
        
        # Processing the call keyword arguments (line 664)
        # Getting the type of 'self' (line 665)
        self_361107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 41), 'self', False)
        # Obtaining the member 'shape' of a type (line 665)
        shape_361108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 41), self_361107, 'shape')
        keyword_361109 = shape_361108
        # Getting the type of 'data' (line 665)
        data_361110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 58), 'data', False)
        # Obtaining the member 'dtype' of a type (line 665)
        dtype_361111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 58), data_361110, 'dtype')
        keyword_361112 = dtype_361111
        kwargs_361113 = {'dtype': keyword_361112, 'shape': keyword_361109}
        # Getting the type of 'self' (line 664)
        self_361093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 19), 'self', False)
        # Obtaining the member '__class__' of a type (line 664)
        class___361094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 19), self_361093, '__class__')
        # Calling __class__(args, kwargs) (line 664)
        class___call_result_361114 = invoke(stypy.reporting.localization.Localization(__file__, 664, 19), class___361094, *[tuple_361095], **kwargs_361113)
        
        # Assigning a type to the variable 'stypy_return_type' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'stypy_return_type', class___call_result_361114)
        # SSA branch for the else part of an if statement (line 663)
        module_type_store.open_ssa_branch('else')
        
        # Call to __class__(...): (line 667)
        # Processing the call arguments (line 667)
        
        # Obtaining an instance of the builtin type 'tuple' (line 667)
        tuple_361117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 667)
        # Adding element type (line 667)
        # Getting the type of 'data' (line 667)
        data_361118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 35), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 35), tuple_361117, data_361118)
        # Adding element type (line 667)
        # Getting the type of 'self' (line 667)
        self_361119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 40), 'self', False)
        # Obtaining the member 'indices' of a type (line 667)
        indices_361120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 40), self_361119, 'indices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 35), tuple_361117, indices_361120)
        # Adding element type (line 667)
        # Getting the type of 'self' (line 667)
        self_361121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 53), 'self', False)
        # Obtaining the member 'indptr' of a type (line 667)
        indptr_361122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 53), self_361121, 'indptr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 35), tuple_361117, indptr_361122)
        
        # Processing the call keyword arguments (line 667)
        # Getting the type of 'self' (line 668)
        self_361123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 41), 'self', False)
        # Obtaining the member 'shape' of a type (line 668)
        shape_361124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 41), self_361123, 'shape')
        keyword_361125 = shape_361124
        # Getting the type of 'data' (line 668)
        data_361126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 58), 'data', False)
        # Obtaining the member 'dtype' of a type (line 668)
        dtype_361127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 58), data_361126, 'dtype')
        keyword_361128 = dtype_361127
        kwargs_361129 = {'dtype': keyword_361128, 'shape': keyword_361125}
        # Getting the type of 'self' (line 667)
        self_361115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 19), 'self', False)
        # Obtaining the member '__class__' of a type (line 667)
        class___361116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 19), self_361115, '__class__')
        # Calling __class__(args, kwargs) (line 667)
        class___call_result_361130 = invoke(stypy.reporting.localization.Localization(__file__, 667, 19), class___361116, *[tuple_361117], **kwargs_361129)
        
        # Assigning a type to the variable 'stypy_return_type' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'stypy_return_type', class___call_result_361130)
        # SSA join for if statement (line 663)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_with_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_with_data' in the type store
        # Getting the type of 'stypy_return_type' (line 658)
        stypy_return_type_361131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_361131)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_with_data'
        return stypy_return_type_361131


# Assigning a type to the variable 'bsr_matrix' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'bsr_matrix', bsr_matrix)

# Assigning a Str to a Name (line 118):
str_361132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 13), 'str', 'bsr')
# Getting the type of 'bsr_matrix'
bsr_matrix_361133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bsr_matrix')
# Setting the type of the member 'format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bsr_matrix_361133, 'format', str_361132)

# Assigning a Call to a Name (line 278):

# Call to property(...): (line 278)
# Processing the call keyword arguments (line 278)
# Getting the type of '_get_blocksize' (line 278)
_get_blocksize_361135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 30), '_get_blocksize', False)
keyword_361136 = _get_blocksize_361135
kwargs_361137 = {'fget': keyword_361136}
# Getting the type of 'property' (line 278)
property_361134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'property', False)
# Calling property(args, kwargs) (line 278)
property_call_result_361138 = invoke(stypy.reporting.localization.Localization(__file__, 278, 16), property_361134, *[], **kwargs_361137)

# Getting the type of 'bsr_matrix'
bsr_matrix_361139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bsr_matrix')
# Setting the type of the member 'blocksize' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bsr_matrix_361139, 'blocksize', property_call_result_361138)

# Assigning a Attribute to a Attribute (line 287):
# Getting the type of 'spmatrix' (line 287)
spmatrix_361140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 21), 'spmatrix')
# Obtaining the member 'getnnz' of a type (line 287)
getnnz_361141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 21), spmatrix_361140, 'getnnz')
# Obtaining the member '__doc__' of a type (line 287)
doc___361142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 21), getnnz_361141, '__doc__')
# Getting the type of 'bsr_matrix'
bsr_matrix_361143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bsr_matrix')
# Obtaining the member 'getnnz' of a type
getnnz_361144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bsr_matrix_361143, 'getnnz')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), getnnz_361144, '__doc__', doc___361142)

# Assigning a Attribute to a Attribute (line 308):
# Getting the type of 'spmatrix' (line 308)
spmatrix_361145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 23), 'spmatrix')
# Obtaining the member 'diagonal' of a type (line 308)
diagonal_361146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 23), spmatrix_361145, 'diagonal')
# Obtaining the member '__doc__' of a type (line 308)
doc___361147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 23), diagonal_361146, '__doc__')
# Getting the type of 'bsr_matrix'
bsr_matrix_361148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bsr_matrix')
# Obtaining the member 'diagonal' of a type
diagonal_361149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bsr_matrix_361148, 'diagonal')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), diagonal_361149, '__doc__', doc___361147)

# Assigning a Attribute to a Attribute (line 445):
# Getting the type of 'spmatrix' (line 445)
spmatrix_361150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'spmatrix')
# Obtaining the member 'tocsr' of a type (line 445)
tocsr_361151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 20), spmatrix_361150, 'tocsr')
# Obtaining the member '__doc__' of a type (line 445)
doc___361152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 20), tocsr_361151, '__doc__')
# Getting the type of 'bsr_matrix'
bsr_matrix_361153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bsr_matrix')
# Obtaining the member 'tocsr' of a type
tocsr_361154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bsr_matrix_361153, 'tocsr')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tocsr_361154, '__doc__', doc___361152)

# Assigning a Attribute to a Attribute (line 450):
# Getting the type of 'spmatrix' (line 450)
spmatrix_361155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 20), 'spmatrix')
# Obtaining the member 'tocsc' of a type (line 450)
tocsc_361156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 20), spmatrix_361155, 'tocsc')
# Obtaining the member '__doc__' of a type (line 450)
doc___361157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 20), tocsc_361156, '__doc__')
# Getting the type of 'bsr_matrix'
bsr_matrix_361158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bsr_matrix')
# Obtaining the member 'tocsc' of a type
tocsc_361159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bsr_matrix_361158, 'tocsc')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tocsc_361159, '__doc__', doc___361157)

# Assigning a Attribute to a Attribute (line 490):
# Getting the type of 'spmatrix' (line 490)
spmatrix_361160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 22), 'spmatrix')
# Obtaining the member 'toarray' of a type (line 490)
toarray_361161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 22), spmatrix_361160, 'toarray')
# Obtaining the member '__doc__' of a type (line 490)
doc___361162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 22), toarray_361161, '__doc__')
# Getting the type of 'bsr_matrix'
bsr_matrix_361163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bsr_matrix')
# Obtaining the member 'toarray' of a type
toarray_361164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bsr_matrix_361163, 'toarray')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), toarray_361164, '__doc__', doc___361162)

# Assigning a Attribute to a Attribute (line 517):
# Getting the type of 'spmatrix' (line 517)
spmatrix_361165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 24), 'spmatrix')
# Obtaining the member 'transpose' of a type (line 517)
transpose_361166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 24), spmatrix_361165, 'transpose')
# Obtaining the member '__doc__' of a type (line 517)
doc___361167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 24), transpose_361166, '__doc__')
# Getting the type of 'bsr_matrix'
bsr_matrix_361168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bsr_matrix')
# Obtaining the member 'transpose' of a type
transpose_361169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bsr_matrix_361168, 'transpose')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), transpose_361169, '__doc__', doc___361167)

@norecursion
def isspmatrix_bsr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isspmatrix_bsr'
    module_type_store = module_type_store.open_function_context('isspmatrix_bsr', 678, 0, False)
    
    # Passed parameters checking function
    isspmatrix_bsr.stypy_localization = localization
    isspmatrix_bsr.stypy_type_of_self = None
    isspmatrix_bsr.stypy_type_store = module_type_store
    isspmatrix_bsr.stypy_function_name = 'isspmatrix_bsr'
    isspmatrix_bsr.stypy_param_names_list = ['x']
    isspmatrix_bsr.stypy_varargs_param_name = None
    isspmatrix_bsr.stypy_kwargs_param_name = None
    isspmatrix_bsr.stypy_call_defaults = defaults
    isspmatrix_bsr.stypy_call_varargs = varargs
    isspmatrix_bsr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isspmatrix_bsr', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isspmatrix_bsr', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isspmatrix_bsr(...)' code ##################

    str_361170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, (-1)), 'str', 'Is x of a bsr_matrix type?\n\n    Parameters\n    ----------\n    x\n        object to check for being a bsr matrix\n\n    Returns\n    -------\n    bool\n        True if x is a bsr matrix, False otherwise\n\n    Examples\n    --------\n    >>> from scipy.sparse import bsr_matrix, isspmatrix_bsr\n    >>> isspmatrix_bsr(bsr_matrix([[5]]))\n    True\n\n    >>> from scipy.sparse import bsr_matrix, csr_matrix, isspmatrix_bsr\n    >>> isspmatrix_bsr(csr_matrix([[5]]))\n    False\n    ')
    
    # Call to isinstance(...): (line 701)
    # Processing the call arguments (line 701)
    # Getting the type of 'x' (line 701)
    x_361172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 22), 'x', False)
    # Getting the type of 'bsr_matrix' (line 701)
    bsr_matrix_361173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 25), 'bsr_matrix', False)
    # Processing the call keyword arguments (line 701)
    kwargs_361174 = {}
    # Getting the type of 'isinstance' (line 701)
    isinstance_361171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 701)
    isinstance_call_result_361175 = invoke(stypy.reporting.localization.Localization(__file__, 701, 11), isinstance_361171, *[x_361172, bsr_matrix_361173], **kwargs_361174)
    
    # Assigning a type to the variable 'stypy_return_type' (line 701)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'stypy_return_type', isinstance_call_result_361175)
    
    # ################# End of 'isspmatrix_bsr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isspmatrix_bsr' in the type store
    # Getting the type of 'stypy_return_type' (line 678)
    stypy_return_type_361176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_361176)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isspmatrix_bsr'
    return stypy_return_type_361176

# Assigning a type to the variable 'isspmatrix_bsr' (line 678)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 0), 'isspmatrix_bsr', isspmatrix_bsr)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
